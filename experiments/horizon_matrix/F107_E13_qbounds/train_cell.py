#!/usr/bin/env python3
"""E13: quantile bounds — p10/p50/p90 LightGBM per lead (E10 feature set).

Motivation (owner, 2026-08-17): the MSE point forecast is a conditional
mean and structurally cannot draw the spikes; operations want calibrated
bounds ("probably ~160; if the returning complex survived, 200+").

Three quantile-objective models per lead, trained on train96 with the
fs_* optional-NaN features, early stopping on val (its usual role only —
NO distributional fitting on val; that was E08's fatal mistake).
Quantiles are sorted per forecast to fix occasional crossing.
out/predictions.csv carries pred_obs = p50 (standard scorer contract)
plus p10_obs / p90_obs columns; coverage is scored by eval_bounds.py.
"""

import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import common

ALPHAS = [0.1, 0.5, 0.9]
PARAMS = dict(
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

pred = np.empty((len(ALPHAS), len(orig_te), common.HORIZON))
for h in range(common.HORIZON):
    for a_i, alpha in enumerate(ALPHAS):
        m = lgb.LGBMRegressor(objective="quantile", alpha=alpha,
                              random_state=11, **PARAMS)
        m.fit(
            Xtr, ytr[:, h],
            eval_set=[(Xva, yva[:, h])],
            eval_metric="quantile",
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )
        pred[a_i, :, h] = m.predict(Xte, num_iteration=m.best_iteration_)
    print(f"lead {h + 1:2d}: p10/p50/p90 done", flush=True)

pred = np.sort(pred, axis=0)  # enforce p10 <= p50 <= p90 per forecast

# p50 through the standard contract, bounds as extra columns
common.write_predictions(HERE, orig_te, pred[1])
path = os.path.join(HERE, "out", "predictions.csv")
out = pd.read_csv(path, parse_dates=["t_date", "target_date"])
for a_i, name in ((0, "p10_obs"), (2, "p90_obs")):
    rows = []
    for i, t in enumerate(orig_te):
        tdates = pd.date_range(t + pd.Timedelta(days=1), periods=common.HORIZON)
        rows.extend(np.round(common.adj_to_obs(pred[a_i, i], tdates), 2))
    out[name] = rows
out.to_csv(path, index=False)
print(f"rewrote {path} with p10/p90 columns")
