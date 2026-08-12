#!/usr/bin/env python3
"""E02: E01 + solar wind (MIDL @ 14 Re, daily aggregates).

Single change vs E01: sw=True in build_samples — adds 8 daily series
(median Bx/By/Bz/Ux/rho/T + daily min Bz + daily max Ux) x 60 lags = 480
extra features (1200 total). Same train05 span, params, early stopping.
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
Xtr, ytr, _, names = common.build_samples(df, "train05", sw=True)
Xva, yva, _, _ = common.build_samples(df, "val", sw=True)
Xte, _, orig_te, _ = common.build_samples(df, "test", sw=True)
print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")

pred = np.empty((len(orig_te), common.HORIZON))
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
    pred[:, h] = m.predict(Xte, num_iteration=m.best_iteration_)
    print(f"lead {h + 1:2d}: best_iter {m.best_iteration_:4d}  val_rmse {m.best_score_['valid_0']['rmse']:.2f}")

common.write_predictions(HERE, orig_te, pred)
