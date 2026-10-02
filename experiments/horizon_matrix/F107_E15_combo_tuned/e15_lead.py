#!/usr/bin/env python3
"""One lead of the E15 pipeline (E09 recipe, sweep-winning cfg-44 params).

  e15_lead.py <lead 1..30>   -> out/lead_<h>.csv  (pred_adj per test origin)

Per lead: 5-fold OOF for both LGBM bases (single seed) -> nonneg blend fit
on train -> 3-seed final bases on full spans -> blended test predictions in
ADJUSTED space. merge_leads.py assembles the standard predictions.csv.
Idempotent: skips if the lead file exists. n_jobs pinned (Athena lesson).
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

h = int(sys.argv[1])
out_path = os.path.join(HERE, "out", f"lead_{h:02d}.csv")
if os.path.exists(out_path):
    print(f"lead {h}: exists, skipping")
    sys.exit(0)
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)

SEEDS = [11, 23, 37]
N_FOLDS = 5
PARAMS = dict(  # sweep cfg 44 (2026-08-18) + thread pin
    objective="regression",
    n_estimators=3000,
    learning_rate=0.0161,
    num_leaves=31,
    min_child_samples=160,
    feature_fraction=0.725,
    bagging_fraction=0.694,
    lambda_l1=1.4619,
    bagging_freq=1,
    verbose=-1,
    n_jobs=32,
)

df = common.load_daily()
adj = df["f107_adj"]
f107a = adj.rolling(81, min_periods=60).mean()
X96, y96, orig96, _ = common.build_samples(df, "train", extra=common.FS_COLS)
X47, y47, orig47, _ = common.build_samples(
    df, "train47", required=["f107_adj", "ssn"], extra=common.FS_COLS)
Xva, yva, _, _ = common.build_samples(df, "val", extra=common.FS_COLS)
Xte, _, orig_te, _ = common.build_samples(df, "test", extra=common.FS_COLS)
folds = np.array_split(np.arange(len(orig96)), N_FOLDS)
in47 = orig47.get_indexer(orig96)
assert (in47 >= 0).all()
hc = h - 1


def fit(X, y, seed):
    m = lgb.LGBMRegressor(random_state=seed, **PARAMS)
    m.fit(X, y[:, hc], eval_set=[(Xva, yva[:, hc])], eval_metric="rmse",
          callbacks=[lgb.early_stopping(150, verbose=False)])
    return m


def baselines(orig):
    lag = 27 if h <= 27 else 54
    rec = np.array([adj.loc[t + np.timedelta64(h - lag, "D")] for t in orig])
    return np.stack([rec, f107a.loc[orig].to_numpy(), adj.loc[orig].to_numpy()], axis=1)


oof96, oof47 = np.empty(len(orig96)), np.empty(len(orig96))
for k in folds:
    m96 = np.ones(len(orig96), bool); m96[k] = False
    m = fit(X96[m96], y96[m96], SEEDS[0])
    oof96[k] = m.predict(X96[k], num_iteration=m.best_iteration_)
    m47 = np.ones(len(orig47), bool); m47[in47[k]] = False
    m = fit(X47[m47], y47[m47], SEEDS[0])
    oof47[k] = m.predict(X47[in47[k]], num_iteration=m.best_iteration_)
blend = LinearRegression(positive=True).fit(
    np.column_stack([oof96, oof47, baselines(orig96)]), y96[:, hc])


def seed_mean(X, y):
    ps = []
    for s in SEEDS:
        m = fit(X, y, s)
        ps.append(m.predict(Xte, num_iteration=m.best_iteration_))
    return np.mean(ps, axis=0)


Zte = np.column_stack([seed_mean(X96, y96), seed_mean(X47, y47), baselines(orig_te)])
pred = blend.predict(Zte)
pd.DataFrame({"t_date": orig_te.date, "pred_adj": np.round(pred, 4)}).to_csv(
    out_path, index=False)
w = ", ".join(f"{x:.2f}" for x in blend.coef_)
print(f"lead {h:2d}: w=[{w}] b={blend.intercept_:.1f} -> {out_path}")
