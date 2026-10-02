#!/usr/bin/env python3
"""E31: trajectory ensemble from the champion's out-of-fold residual record.

The E30 combo saves, for every 1996-2021 train origin and every lead, its
OOF blend prediction and the truth. In log space the residual trajectories
  L_i(h) = log y_i(t+h) - log yhat_i(t+h),  h = 1..30
are a library of ~9,000 physically-shaped 30-day "what the model missed"
paths (rotational echoes, region emergence, decay). A test forecast's
members are
  member_i(h) = yhat_test(h) * exp(L_i(h) * s_t)
for every library trajectory i in the same activity bin as the test origin
(terciles of the 81-day envelope, edges from train). Quantiles per lead come
from the full member set (~3,000 paths); 100 member paths per origin are
kept for plotting. s_t = 1 here (E32 adds the adaptive scale).

Outputs (observed space, canonical test origins):
  out/ensemble.csv     t_date, lead, target_date, mean, q05..q95
  out/predictions.csv  pred_obs = ensemble mean  (deterministic scorer)
  out/members.parquet  100 member paths per origin/lead (plots)
Scored by shared/score_prob.py (CRPS, coverage, spike capture).
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import common

SRC = os.environ.get("ENS_SRC", os.path.join(HERE, "..", "F107_E30_combo_v2", "out"))
FLUX = "f107_adj_rob"
QS = np.arange(5, 100, 5)  # q05..q95
N_KEEP = 100
ADAPT = False  # E32 flips this
SEED = 7


def load_src(df):
    _, _, o96, _ = common.build_samples(df, "train", extra=common.FS_COLS, flux=FLUX)
    _, _, ote, _ = common.build_samples(df, "test", extra=common.FS_COLS, flux=FLUX)
    L = np.empty((len(o96), common.HORIZON))
    P = np.empty((len(ote), common.HORIZON))
    for h in range(1, common.HORIZON + 1):
        o = pd.read_csv(os.path.join(SRC, f"oof_{h:02d}.csv"), parse_dates=["t_date"])
        assert (pd.DatetimeIndex(o.t_date) == o96).all()
        L[:, h - 1] = np.log(o.y_adj / o.oof_adj)
        p = pd.read_csv(os.path.join(SRC, f"lead_{h:02d}.csv"), parse_dates=["t_date"])
        assert (pd.DatetimeIndex(p.t_date) == ote).all()
        P[:, h - 1] = p.pred_adj
    return o96, L, ote, P


def adaptive_scale(df, ote, P, L_train_rms):
    """s_t per test origin: RMS of realized log-residuals of forecasts whose
    targets fell in the 90 days before t (known at t), over the train OOF
    RMS at the same leads; 1.0 while fewer than 200 realized pairs exist."""
    flux = df[FLUX]
    H = common.HORIZON
    tgt = (ote.to_numpy()[:, None] + np.arange(1, H + 1) * np.timedelta64(1, "D"))  # (n, H)
    y = flux.reindex(pd.DatetimeIndex(tgt.ravel())).to_numpy().reshape(tgt.shape)
    R2 = np.log(y / P) ** 2                                   # realized sq log-residual
    norm = (L_train_rms**2)[None, :] * np.ones_like(R2)
    s = np.ones(len(ote))
    for j, t in enumerate(ote):
        lo = max(0, j - 120)
        m = (tgt[lo:j] <= np.datetime64(t)) & (tgt[lo:j] >= np.datetime64(t - pd.Timedelta(days=90))) \
            & np.isfinite(R2[lo:j])
        if m.sum() >= 200:
            s[j] = np.clip(np.sqrt(R2[lo:j][m].sum() / norm[lo:j][m].sum()), 0.7, 2.0)
    return s


def main():
    df = common.load_daily()
    o96, L, ote, P = load_src(df)
    env = common.envelope(df, FLUX)
    edges = np.quantile(env.loc[o96].to_numpy(), [1 / 3, 2 / 3])
    bin_tr = np.digitize(env.loc[o96].to_numpy(), edges)
    bin_te = np.digitize(env.loc[ote].to_numpy(), edges)
    rms = np.sqrt(np.mean(L**2, axis=0))
    print("train OOF log-residual RMS by lead:", np.round(rms[[0, 6, 13, 20, 29]], 3),
          "| by bin:", [float(np.sqrt(np.mean(L[bin_tr == b] ** 2)).round(3)) for b in range(3)])
    s = adaptive_scale(df, ote, P, rms) if ADAPT else np.ones(len(ote))
    if ADAPT:
        print("adaptive scale s_t: min/median/max", np.round([s.min(), np.median(s), s.max()], 2))

    rng = np.random.default_rng(SEED)
    rows, mrows, pred_mean = [], [], np.empty_like(P)
    for j, t in enumerate(ote):
        lib = L[bin_tr == bin_te[j]] * s[j]                 # (n_lib, 30)
        M = P[j][None, :] * np.exp(lib)                       # members, adjusted
        tdates = pd.date_range(t + pd.Timedelta(days=1), periods=common.HORIZON)
        M = common.adj_to_obs(M, tdates)                      # broadcast over rows
        q = np.percentile(M, QS, axis=0)                      # (19, 30)
        mean = M.mean(axis=0)
        pred_mean[j] = mean
        keep = rng.choice(len(M), size=min(N_KEEP, len(M)), replace=False)
        for h in range(common.HORIZON):
            rows.append((t.date(), h + 1, tdates[h].date(), round(float(mean[h]), 2),
                         *np.round(q[:, h], 2)))
            mrows.append((t.date(), h + 1, *np.round(M[keep, h], 1)))
    cols = ["t_date", "lead", "target_date", "mean"] + [f"q{int(v):02d}" for v in QS]
    ens = pd.DataFrame(rows, columns=cols)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    ens.to_csv(os.path.join(HERE, "out", "ensemble.csv"), index=False)
    pd.DataFrame(mrows, columns=["t_date", "lead"] + [f"m{i:03d}" for i in range(N_KEEP)]).to_parquet(
        os.path.join(HERE, "out", "members.parquet"), index=False)
    # deterministic contract: the ensemble mean (already observed space)
    out = ens[["t_date", "lead", "target_date"]].copy()
    out["pred_obs"] = ens["mean"]
    out.to_csv(os.path.join(HERE, "out", "predictions.csv"), index=False)
    print(f"wrote ensemble.csv / members.parquet / predictions.csv for {len(ote)} origins")


if __name__ == "__main__":
    main()
