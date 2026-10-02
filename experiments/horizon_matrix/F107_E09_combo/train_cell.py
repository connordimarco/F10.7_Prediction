#!/usr/bin/env python3
"""E09: combination cell — everything that won, blended with OOF weights.

Bases per lead, all in adjusted space:
  lgbm96  E10 model (train 1996-2021, fs_*/sard_* optional-NaN)
  lgbm47  E05xE10 deep-span (train 1947-2021; ar_* NaN pre-1996, fs_* pre-2010)
  recur27 / mean81 / persist (as in E06)

vs E06 (three changes, all previously screened):
  - the LGBM base carries the E10 far-side features, and there are two of
    them (the deep-span variant won L1-14 in E05);
  - final test predictions of each LGBM base are 3-seed ensembles (E07);
  - blend weights (nonneg linreg + intercept, per lead) are fitted on
    OUT-OF-FOLD train predictions — 5 chronological folds over the
    1996-2021 origins — instead of on val, removing E06's documented val
    double-dip (val keeps exactly its usual role: early stopping only).
    OOF fits are single-seed; the final bases are seed-averaged, so the
    weights are fitted on slightly noisier bases than they act on
    (documented, conservative direction).
"""

import os
import sys

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LinearRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import common

SEEDS = [11, 23, 37]
N_FOLDS = 5
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

X96, y96, orig96, names = common.build_samples(df, "train", extra=common.FS_COLS)
X47, y47, orig47, _ = common.build_samples(
    df, "train47", required=["f107_adj", "ssn"], extra=common.FS_COLS
)
Xva, yva, _, _ = common.build_samples(df, "val", extra=common.FS_COLS)
Xte, _, orig_te, _ = common.build_samples(df, "test", extra=common.FS_COLS)
print(f"train96 {X96.shape}  train47 {X47.shape}  val {Xva.shape}  test {Xte.shape}")

# 5 chronological folds over the 1996-2021 origins; the 47-span's extra
# pre-1996 rows are never scored, so they stay in every fold's training set.
folds = np.array_split(np.arange(len(orig96)), N_FOLDS)
in47 = orig47.get_indexer(orig96)  # position of each 96-origin in the 47 set
assert (in47 >= 0).all()


def baselines(orig, h):
    lag = 27 if h <= 27 else 54
    rec = np.array([adj.loc[t + np.timedelta64(h - lag, "D")] for t in orig])
    return np.stack([rec, f107a.loc[orig].to_numpy(), adj.loc[orig].to_numpy()], axis=1)


def fit(X, y, seed, hcol, eval_y):
    m = lgb.LGBMRegressor(random_state=seed, **PARAMS)
    m.fit(
        X, y[:, hcol],
        eval_set=[(Xva, eval_y)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    return m


pred = np.empty((len(orig_te), common.HORIZON))
for h in range(1, common.HORIZON + 1):
    hc = h - 1
    # --- OOF predictions on the train origins, both LGBM bases ---
    oof96 = np.empty(len(orig96))
    oof47 = np.empty(len(orig96))
    for k in folds:
        mask96 = np.ones(len(orig96), bool)
        mask96[k] = False
        m = fit(X96[mask96], y96[mask96], SEEDS[0], hc, yva[:, hc])
        oof96[k] = m.predict(X96[k], num_iteration=m.best_iteration_)
        mask47 = np.ones(len(orig47), bool)
        mask47[in47[k]] = False
        m = fit(X47[mask47], y47[mask47], SEEDS[0], hc, yva[:, hc])
        oof47[k] = m.predict(X47[in47[k]], num_iteration=m.best_iteration_)
    Ztr = np.column_stack([oof96, oof47, baselines(orig96, h)])
    blend = LinearRegression(positive=True).fit(Ztr, y96[:, hc])
    # --- final full-train bases, seed-ensembled, on test ---
    def seed_mean(X, y):
        ps = []
        for s in SEEDS:
            m = fit(X, y, s, hc, yva[:, hc])
            ps.append(m.predict(Xte, num_iteration=m.best_iteration_))
        return np.mean(ps, axis=0)

    p96 = seed_mean(X96, y96)
    p47 = seed_mean(X47, y47)
    Zte = np.column_stack([p96, p47, baselines(orig_te, h)])
    pred[:, hc] = blend.predict(Zte)
    w = ", ".join(f"{x:.2f}" for x in blend.coef_)
    print(
        f"lead {h:2d}: w[lgbm96, lgbm47, recur, mean81, persist] = [{w}]"
        f"  b={blend.intercept_:.1f}",
        flush=True,
    )

common.write_predictions(HERE, orig_te, pred)
