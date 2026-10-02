"""Neural sequence/MLP forecaster (v2 era, 2026-09-02) — a non-tree family.

Why: trees predict a level and cannot leave their training leaves, which is
the tail failure in the matrix. A neural net with a linear output head
extrapolates, and predicting all 30 leads JOINTLY gives one coherent
trajectory per origin instead of 30 independent conditional means.

Inputs: the standard 60-day windows (12 daily series + 14 far-side
aggregates as optional-NaN), flux block expressed relative to the origin
envelope, everything log1p'd and standardized on train, NaN -> 0 plus a
missing-indicator channel per family (ar_*, fs_*); plus static
log(env81) and the persistence ratio. Target: flux(t+h)/env81(t) for
h=1..30. Loss: (env81 x error)^2 = flux-space MSE, i.e. exactly the metric.
Seed ensemble, early stopping on val, train span 1947-2021 by default.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

FLUX = "f107_adj_rob"
H, T = common.HORIZON, common.HIST
COLS = common.FEATURE_COLS + common.FS_COLS  # 26 series
N_AR = len(common.FEATURE_COLS) - 2


def prep(df, split, required=None):
    X, y, orig, _ = common.build_samples(df, split, required=required, extra=common.FS_COLS, flux=FLUX)
    env = common.envelope(df, FLUX).loc[orig].to_numpy()
    ok = np.isfinite(env) & (env > 0)
    X, y, orig, env = X[ok], y[ok], orig[ok], env[ok]
    S = X.reshape(len(X), len(COLS), T)  # (n, series, lag) oldest->newest
    F = np.empty_like(S)
    F[:, 0] = np.log(S[:, 0] / env[:, None])              # flux relative to envelope
    F[:, 1:] = np.log1p(np.maximum(S[:, 1:], 0))           # ssn, ar_*, fs_*
    F = np.nan_to_num(F, nan=0.0)
    miss_ar = np.isnan(S[:, 2:2 + N_AR]).any(axis=1, keepdims=True).astype(np.float32)
    miss_fs = np.isnan(S[:, 2 + N_AR:]).any(axis=1, keepdims=True).astype(np.float32)
    F = np.concatenate([F, miss_ar, miss_fs], axis=1)     # (n, 28, 60)
    static = np.column_stack([np.log(env), np.log(S[:, 0, -1] / env)])
    Y = y / env[:, None]
    return F.astype(np.float32), static.astype(np.float32), Y.astype(np.float32), env.astype(np.float32), orig


class Net(nn.Module):
    def __init__(self, n_ch, n_static, kind="gru", hidden=96, drop=0.2):
        super().__init__()
        self.kind = kind
        if kind == "gru":
            self.enc = nn.GRU(n_ch, hidden, batch_first=True)
            d = hidden
        else:  # mlp on the flattened window
            self.enc = nn.Sequential(nn.Flatten(), nn.Linear(n_ch * T, 256), nn.GELU(), nn.Dropout(drop),
                                     nn.Linear(256, hidden), nn.GELU())
            d = hidden
        self.head = nn.Sequential(nn.Linear(d + n_static, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, H))

    def forward(self, x, s):  # x (n, ch, T)
        if self.kind == "gru":
            _, h = self.enc(x.transpose(1, 2))
            z = h[-1]
        else:
            z = self.enc(x)
        return 1.0 + self.head(torch.cat([z, s], dim=1))  # ratio ~ 1


def fit_one(seed, tr, va, kind, hidden, drop, lr, epochs, patience, log):
    torch.manual_seed(seed)
    Ftr, Str, Ytr, Etr = (torch.tensor(a) for a in tr)
    Fva, Sva, Yva, Eva = (torch.tensor(a) for a in va)
    net = Net(Ftr.shape[1], Str.shape[1], kind, hidden, drop)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(Ftr)
    best, best_state, bad = np.inf, None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, 256):
            b = perm[i:i + 256]
            loss = ((net(Ftr[b], Str[b]) - Ytr[b]) * Etr[b, None]).pow(2).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            v = ((net(Fva, Sva) - Yva) * Eva[:, None]).pow(2).mean().sqrt().item()
        if v < best - 1e-3:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    net.load_state_dict(best_state)
    log(f"  seed {seed}: best val RMSE(flux, adj) {best:.2f} at epoch {ep + 1 - bad}")
    return net


def run(cell_dir, kind="gru", span="train47", seeds=(1, 2, 3, 4, 5), hidden=96, drop=0.2,
        lr=1e-3, epochs=120, patience=15, threads=None, oof=True, n_folds=5):
    """Test predictions (seed ensemble) -> out/predictions.csv + out/lead_XX.csv;
    with oof=True also 5 chronological folds over the 1996-2021 origins
    (single seed, as the LightGBM combos do) -> out/oof_XX.csv, the
    per-lead OOF record the stacking/ensemble layers consume."""
    threads = threads or int(os.environ.get("TORCH_THREADS", 0)) or None
    if threads:
        torch.set_num_threads(threads)
    t0 = time.time()
    log = lambda m: print(m, flush=True)
    df = common.load_daily()
    required = ["f107_adj", "ssn"] if span == "train47" else None
    tr = prep(df, span, required)
    va = prep(df, "val")
    te = prep(df, "test")
    log(f"{os.path.basename(cell_dir)}: kind={kind} span={span} hidden={hidden} drop={drop} "
        f"| train {tr[0].shape} val {va[0].shape} test {te[0].shape}")
    out = os.path.join(cell_dir, "out")
    os.makedirs(out, exist_ok=True)
    preds = []
    for s in seeds:
        net = fit_one(s, tr[:4], va[:4], kind, hidden, drop, lr, epochs, patience, log)
        net.eval()
        with torch.no_grad():
            preds.append(net(torch.tensor(te[0]), torch.tensor(te[1])).numpy())
    pred_adj = np.mean(preds, axis=0) * te[3][:, None]
    for h in range(H):
        pd.DataFrame({"t_date": te[4].date, "pred_adj": np.round(pred_adj[:, h], 4)}).to_csv(
            os.path.join(out, f"lead_{h + 1:02d}.csv"), index=False)
    common.write_predictions(cell_dir, te[4], pred_adj)
    if oof:
        o96 = prep(df, "train")[4]
        pos = tr[4].get_indexer(o96)
        assert (pos >= 0).all()
        folds = np.array_split(np.arange(len(o96)), n_folds)
        oof_adj = np.empty((len(o96), H), np.float32)
        for k in folds:
            mask = np.ones(len(tr[4]), bool)
            mask[pos[k]] = False
            sub = tuple(a[mask] for a in tr[:4])
            net = fit_one(seeds[0], sub, va[:4], kind, hidden, drop, lr, epochs, patience, log)
            net.eval()
            with torch.no_grad():
                r = net(torch.tensor(tr[0][pos[k]]), torch.tensor(tr[1][pos[k]])).numpy()
            oof_adj[k] = r * tr[3][pos[k], None]
        y_adj = tr[2][pos] * tr[3][pos, None]
        for h in range(H):
            pd.DataFrame({"t_date": o96.date, "oof_adj": np.round(oof_adj[:, h], 4),
                          "y_adj": np.round(y_adj[:, h], 4)}).to_csv(
                os.path.join(out, f"oof_{h + 1:02d}.csv"), index=False)
        log(f"OOF written for {len(o96)} train origins")
    log(f"done in {time.time() - t0:.0f}s")
