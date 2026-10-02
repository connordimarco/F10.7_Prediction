#!/usr/bin/env python3
"""E10: E00 + far-side AR catalog features (NaN pre-2010 + map outages).

Single change vs E00: the 14 fs_*/sard_* daily aggregates (GONG f6x
helioseismic far-side detections + JSOC SARD lists, see
../FARSIDE_SCOPING.md) join the 60-day windows as optional-NaN features —
the E03/E05 pattern, LightGBM native missing handling. Train span, splits,
and the canonical origin sets are unchanged; hypothesis is skill in the
L8-30 band, where the far side is the information Earth-view features lack.
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
Xtr, ytr, _, names = common.build_samples(df, "train", extra=common.FS_COLS)
Xva, yva, _, _ = common.build_samples(df, "val", extra=common.FS_COLS)
Xte, _, orig_te, _ = common.build_samples(df, "test", extra=common.FS_COLS)
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
