#!/usr/bin/env python3
"""E08: E00 + post-hoc quantile-map calibration (LAUREN E08 pattern).

E00-style models are retrained; their VAL predictions (pooled over leads,
adjusted space) define a monotone quantile map pred-dist -> truth-dist,
linearly extrapolated beyond the 1st/99th percentile using the q90-q99
slope. The map is applied to the test predictions. Zero new information —
pure un-shrinkage, aimed at the high-activity bin.
"""

import os
import sys

import lightgbm as lgb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import common

PARAMS = dict(
    objective="regression",
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=40,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    verbose=-1,
)

df = common.load_daily()
Xtr, ytr, _, names = common.build_samples(df, "train")
Xva, yva, _, _ = common.build_samples(df, "val")
Xte, _, orig_te, _ = common.build_samples(df, "test")
print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")

pv, pt = [], np.empty((len(orig_te), common.HORIZON))
for h in range(common.HORIZON):
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(
        Xtr,
        ytr[:, h],
        eval_set=[(Xva, yva[:, h])],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(150, verbose=False)],
        feature_name=names,
    )
    pv.append(m.predict(Xva, num_iteration=m.best_iteration_))
    pt[:, h] = m.predict(Xte, num_iteration=m.best_iteration_)
    print(f"lead {h + 1:2d}: best_iter {m.best_iteration_:4d}")

pv = np.concatenate(pv)
tv = yva.T.ravel()
qs = np.linspace(1, 99, 197)
pq, tq = np.percentile(pv, qs), np.percentile(tv, qs)
slope_hi = (tq[-1] - tq[-19]) / max(pq[-1] - pq[-19], 1e-9)   # q90->q99
slope_lo = (tq[18] - tq[0]) / max(pq[18] - pq[0], 1e-9)       # q1->q10


def qmap(x):
    y = np.interp(x, pq, tq)
    y = np.where(x > pq[-1], tq[-1] + slope_hi * (x - pq[-1]), y)
    y = np.where(x < pq[0], tq[0] + slope_lo * (x - pq[0]), y)
    return y


print(f"map: pred q99 {pq[-1]:.0f} -> {tq[-1]:.0f}, hi-slope {slope_hi:.2f}, lo-slope {slope_lo:.2f}")
common.write_predictions(HERE, orig_te, qmap(pt))
