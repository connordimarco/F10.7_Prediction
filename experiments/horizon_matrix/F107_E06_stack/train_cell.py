#!/usr/bin/env python3
"""E06: per-lead stacking blend of E00 + the three baselines.

Base predictors, all in adjusted space: E00-style LightGBM (retrained here),
27-day recurrence, 81-day trailing mean, persistence adj(t). A per-lead
nonnegative linear blend is fitted on the VAL years and applied to test.

Caveat (documented, accepted for screening): the LightGBM bases early-stop
on val, so the blend's val fit is mildly optimistic. A full-scale version
would use out-of-fold train predictions instead.
"""

import os
import sys

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LinearRegression

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
adj = df["f107_adj"]
f107a = adj.rolling(81, min_periods=60).mean()


def baselines(orig, h):
    """recur27, mean81, persist for one lead, adjusted space."""
    lag = 27 if h <= 27 else 54
    rec = np.array([adj.loc[t + np.timedelta64(h - lag, "D")] for t in orig])
    return np.stack([rec, f107a.loc[orig].to_numpy(), adj.loc[orig].to_numpy()], axis=1)


Xtr, ytr, _, names = common.build_samples(df, "train")
Xva, yva, orig_va, _ = common.build_samples(df, "val")
Xte, _, orig_te, _ = common.build_samples(df, "test")
print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")

pred = np.empty((len(orig_te), common.HORIZON))
for h in range(1, common.HORIZON + 1):
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(
        Xtr,
        ytr[:, h - 1],
        eval_set=[(Xva, yva[:, h - 1])],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(150, verbose=False)],
        feature_name=names,
    )
    Zva = np.column_stack([m.predict(Xva, num_iteration=m.best_iteration_), baselines(orig_va, h)])
    Zte = np.column_stack([m.predict(Xte, num_iteration=m.best_iteration_), baselines(orig_te, h)])
    blend = LinearRegression(positive=True).fit(Zva, yva[:, h - 1])
    pred[:, h - 1] = blend.predict(Zte)
    w = ", ".join(f"{x:.2f}" for x in blend.coef_)
    print(f"lead {h:2d}: weights [lgbm, recur, mean81, persist] = [{w}]  b={blend.intercept_:.1f}")

common.write_predictions(HERE, orig_te, pred)
