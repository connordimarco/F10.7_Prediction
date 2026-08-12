#!/usr/bin/env python3
"""E07: E00 with a 5-seed ensemble per lead (predictions averaged).

Single change vs E00: each lead trains 5 LightGBMs differing only in
bagging/feature seed; test predictions are the mean. Pure variance
reduction — no new information.
"""

import os
import sys

import lightgbm as lgb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import common

SEEDS = [11, 23, 37, 51, 73]
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
print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}  seeds {SEEDS}")

pred = np.empty((len(orig_te), common.HORIZON))
for h in range(common.HORIZON):
    ps = []
    for s in SEEDS:
        m = lgb.LGBMRegressor(random_state=s, **PARAMS)
        m.fit(
            Xtr,
            ytr[:, h],
            eval_set=[(Xva, yva[:, h])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(150, verbose=False)],
            feature_name=names,
        )
        ps.append(m.predict(Xte, num_iteration=m.best_iteration_))
    pred[:, h] = np.mean(ps, axis=0)
    print(f"lead {h + 1:2d}: 5 seeds done")

common.write_predictions(HERE, orig_te, pred)
