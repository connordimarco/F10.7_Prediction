#!/usr/bin/env python3
"""E04: E00 + explicit rotation-aligned recurrence features, per lead.

Single change vs E00: each lead-h model gets 2 extra columns — f107_adj one
and two Carrington rotations (27 d / 54 d) before ITS target date t+h, i.e.
lags 27-h and 54-h (shifted one more rotation back when h > 27). The
information is already inside the 60-day window, but the alignment shifts
with lead; persistence exploits it for free at h≈27, the trees have to find
it in 720 columns. This hands it to them.
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
tgt = df["f107_adj"].to_numpy()


def aligned(split, h):
    """f107_adj at t+h-27 and t+h-54 (one more rotation back if h > 27)."""
    orig = common.origins(split, df)
    pos = df.index.get_indexer(orig)
    off = 27 if h <= 27 else 54
    return np.stack([tgt[pos + h - off], tgt[pos + h - off - 27]], axis=1)


Xtr, ytr, _, names = common.build_samples(df, "train")
Xva, yva, _, _ = common.build_samples(df, "val")
Xte, _, orig_te, _ = common.build_samples(df, "test")
print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}  (+2 aligned cols per lead)")

pred = np.empty((len(orig_te), common.HORIZON))
for h in range(1, common.HORIZON + 1):
    fn = names + ["recur_1rot", "recur_2rot"]
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(
        np.hstack([Xtr, aligned("train", h)]),
        ytr[:, h - 1],
        eval_set=[(np.hstack([Xva, aligned("val", h)]), yva[:, h - 1])],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(150, verbose=False)],
        feature_name=fn,
    )
    pred[:, h - 1] = m.predict(np.hstack([Xte, aligned("test", h)]), num_iteration=m.best_iteration_)
    print(f"lead {h:2d}: best_iter {m.best_iteration_:4d}  val_rmse {m.best_score_['valid_0']['rmse']:.2f}")

common.write_predictions(HERE, orig_te, pred)
