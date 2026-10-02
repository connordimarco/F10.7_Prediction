#!/usr/bin/env python3
"""One lead of E34 — regime-gated multi-base combo (v2 data).

  e34_lead.py <lead 1..30>  ->  out/lead_<h>.csv (test pred_adj),
                                out/oof_<h>.csv  (train-origin OOF blend pred)

Bases per lead (all LightGBM, E10 feature set, cfg-44 params, robust flux):
  lgbm96_raw, lgbm47_raw       level target       (E20/E21 family)
  lgbm96_ratio, lgbm47_ratio   envelope-ratio target, multiplied back (E22)
plus recur27 / mean81 / persist. The E22 screen showed the ratio target
wins the high tail and loses the mid range, so the blend weights are
fitted SEPARATELY per activity regime: terciles of the origin's 81-day
envelope (edges from train), nonneg linear blend + intercept on 5-fold OOF
predictions within each regime; the test origin's regime picks its
weights. Final bases are 3-seed ensembles. Idempotent per lead.
"""

import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import common
from single_model import CFG44

FLUX = "f107_adj_rob"
BASES = [("train", "raw"), ("train47", "raw"), ("train", "ratio_env"), ("train47", "ratio_env")]
SEEDS = [11, 23, 37]
N_FOLDS = 5
PARAMS = dict(CFG44)
PARAMS.pop("random_state")


def lead(h):
    out_path = os.path.join(HERE, "out", f"lead_{h:02d}.csv")
    oof_path = os.path.join(HERE, "out", f"oof_{h:02d}.csv")
    if os.path.exists(out_path) and os.path.exists(oof_path):
        print(f"lead {h}: exists, skipping")
        return
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    df = common.load_daily()
    flux, env = df[FLUX], common.envelope(df, FLUX)
    hc = h - 1
    data = {}
    for span in ("train", "train47"):
        req = ["f107_adj", "ssn"] if span == "train47" else None
        X, y, o, _ = common.build_samples(df, span, required=req, extra=common.FS_COLS, flux=FLUX)
        e = env.loc[o].to_numpy()
        ok = np.isfinite(e) & (e > 0)
        data[span] = (X[ok], y[ok], o[ok], e[ok])
    Xva, yva, ova, _ = common.build_samples(df, "val", extra=common.FS_COLS, flux=FLUX)
    Xte, _, ote, _ = common.build_samples(df, "test", extra=common.FS_COLS, flux=FLUX)
    eva, ete = env.loc[ova].to_numpy(), env.loc[ote].to_numpy()
    o96 = data["train"][2]
    folds = np.array_split(np.arange(len(o96)), N_FOLDS)
    in47 = data["train47"][2].get_indexer(o96)
    assert (in47 >= 0).all()

    def den(e, target):
        return e if target == "ratio_env" else np.ones_like(e)

    def fit(X, y, d, seed, target):
        m = lgb.LGBMRegressor(random_state=seed, **PARAMS)
        m.fit(X, y[:, hc] / d, eval_set=[(Xva, yva[:, hc] / den(eva, target))], eval_metric="rmse",
              callbacks=[lgb.early_stopping(150, verbose=False)])
        return m

    def baselines(orig):
        lag = 27 if h <= 27 else 54
        rec = np.array([flux.loc[t + np.timedelta64(h - lag, "D")] for t in orig])
        return np.stack([rec, env.loc[orig].to_numpy(), flux.loc[orig].to_numpy()], axis=1)

    oof_cols, te_cols = [], []
    for span, target in BASES:
        X, y, o, e = data[span]
        d = den(e, target)
        idx = np.arange(len(o96)) if span == "train" else in47
        oof = np.empty(len(o96))
        for k in folds:
            mask = np.ones(len(o), bool)
            mask[idx[k]] = False
            m = fit(X[mask], y[mask], d[mask], SEEDS[0], target)
            oof[k] = m.predict(X[idx[k]], num_iteration=m.best_iteration_) * d[idx[k]]
        oof_cols.append(oof)
        ps = []
        for s in SEEDS:
            m = fit(X, y, d, s, target)
            ps.append(m.predict(Xte, num_iteration=m.best_iteration_) * den(ete, target))
        te_cols.append(np.mean(ps, axis=0))
    Ztr = np.column_stack(oof_cols + [baselines(o96)])
    Zte = np.column_stack(te_cols + [baselines(ote)])
    y96 = data["train"][1][:, hc]
    e96 = data["train"][3]
    edges = np.quantile(e96, [1 / 3, 2 / 3])
    btr, bte = np.digitize(e96, edges), np.digitize(ete, edges)
    oof_pred, pred = np.empty(len(o96)), np.empty(len(ote))
    ws = []
    for b in range(3):
        blend = LinearRegression(positive=True).fit(Ztr[btr == b], y96[btr == b])
        oof_pred[btr == b] = blend.predict(Ztr[btr == b])
        if (bte == b).any():
            pred[bte == b] = blend.predict(Zte[bte == b])
        ws.append("[" + " ".join(f"{x:.2f}" for x in blend.coef_) + f" | b={blend.intercept_:.0f}]")
    pd.DataFrame({"t_date": o96.date, "oof_adj": np.round(oof_pred, 4), "y_adj": np.round(y96, 4)}).to_csv(oof_path, index=False)
    pd.DataFrame({"t_date": ote.date, "pred_adj": np.round(pred, 4)}).to_csv(out_path, index=False)
    print(f"lead {h:2d}: w[96raw 47raw 96ratio 47ratio recur mean81 persist] quiet/mid/busy = "
          + " ".join(ws), flush=True)


if __name__ == "__main__":
    lead(int(sys.argv[1]))
