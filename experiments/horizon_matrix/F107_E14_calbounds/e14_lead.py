#!/usr/bin/env python3
"""One lead of E14: conformally calibrated p10-p90 bounds (cfg-44 params).

  e14_lead.py <lead 1..30>  ->  out/lead_<h>.csv

Recipe (the E08/E13 lessons made law):
  1. p10/p90 quantile-objective LGBMs, tuned cfg-44 params, trained on
     train96 (fs_* optional-NaN), early stopping on val (its only role).
  2. FIVE-FOLD OOF predictions of both quantiles on the train origins —
     conformity score s_i = max(p10_i - y_i, y_i - p90_i) (>0 means the
     raw band missed y_i by that much).
  3. Widening q = 80th percentile of s within each ACTIVITY BIN (terciles
     of f107a at the origin, bin edges from train) — busy Sun gets wider
     bands. Fitted on train OOF only; val and test never touched.
  4. Test band = [p10 - q(bin), p90 + q(bin)], bin from the test origin's
     f107a. Written in adjusted space; merge_leads.py converts.
Idempotent per lead; n_jobs pinned (Athena lesson).
"""

import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import common

h = int(sys.argv[1])
out_path = os.path.join(HERE, "out", f"lead_{h:02d}.csv")
if os.path.exists(out_path):
    print(f"lead {h}: exists, skipping")
    sys.exit(0)
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)

N_FOLDS = 5
TARGET = 0.80
PARAMS = dict(  # sweep cfg 44 + thread pin
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
f107a = df["f107_adj"].rolling(81, min_periods=60).mean()
X96, y96, orig96, _ = common.build_samples(df, "train", extra=common.FS_COLS)
Xva, yva, _, _ = common.build_samples(df, "val", extra=common.FS_COLS)
Xte, _, orig_te, _ = common.build_samples(df, "test", extra=common.FS_COLS)
folds = np.array_split(np.arange(len(orig96)), N_FOLDS)
hc = h - 1


def fit(X, y, alpha):
    m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, random_state=11,
                          **PARAMS)
    m.fit(X, y[:, hc], eval_set=[(Xva, yva[:, hc])], eval_metric="quantile",
          callbacks=[lgb.early_stopping(150, verbose=False)])
    return m


# OOF raw quantiles on train
oof = {0.1: np.empty(len(orig96)), 0.9: np.empty(len(orig96))}
for k in folds:
    mask = np.ones(len(orig96), bool)
    mask[k] = False
    for a in (0.1, 0.9):
        m = fit(X96[mask], y96[mask], a)
        oof[a][k] = m.predict(X96[k], num_iteration=m.best_iteration_)
lo = np.minimum(oof[0.1], oof[0.9])
hi = np.maximum(oof[0.1], oof[0.9])
score = np.maximum(lo - y96[:, hc], y96[:, hc] - hi)

# activity-binned conformal widening (bin edges from train f107a terciles)
act_tr = f107a.loc[orig96].to_numpy()
edges = np.quantile(act_tr, [1 / 3, 2 / 3])
bin_tr = np.digitize(act_tr, edges)
q = np.array([np.quantile(score[bin_tr == b], TARGET) for b in range(3)])

# final quantile models on full train -> raw test band -> widen by bin
p10 = fit(X96, y96, 0.1)
p90 = fit(X96, y96, 0.9)
te10 = p10.predict(Xte, num_iteration=p10.best_iteration_)
te90 = p90.predict(Xte, num_iteration=p90.best_iteration_)
te_lo, te_hi = np.minimum(te10, te90), np.maximum(te10, te90)
bin_te = np.digitize(f107a.loc[orig_te].to_numpy(), edges)
te_lo -= q[bin_te]
te_hi += q[bin_te]

pd.DataFrame({
    "t_date": orig_te.date,
    "p10_adj": np.round(te_lo, 4),
    "p90_adj": np.round(te_hi, 4),
}).to_csv(out_path, index=False)
print(f"lead {h:2d}: widen q(quiet/mid/busy) = {q.round(1)} -> {out_path}")
