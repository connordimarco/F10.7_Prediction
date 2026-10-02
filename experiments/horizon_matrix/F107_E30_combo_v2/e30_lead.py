#!/usr/bin/env python3
"""One lead of E30 — the E15 champion recipe rebuilt on the v2 data.

  e30_lead.py <lead 1..30>  ->  out/lead_<h>.csv (test pred_adj) and
                                out/oof_<h>.csv  (train-origin OOF blend pred)

Recipe (E09/E15): per lead, 5 chronological folds give OOF predictions of
two LightGBM bases (train96 and deep-span train47, E10 feature set, cfg-44
params) on the 1996-2021 origins; a nonneg linear blend of
[lgbm96, lgbm47, recur27, mean81, persist] (+ intercept) is fitted on those
OOF predictions; final 3-seed bases predict test; the blend is applied.

v2 changes (2026-09-02), each screened as a single-model cell first:
  - data: F10.7 dates corrected (were one day late), flux FLUX (robust
    series if the E21 screen won);
  - target: TARGET ("ratio_env" = flux(t+h)/env81(t), multiplied back
    before blending, if the E22 screen won);
  - the OOF blend prediction on every train origin is saved: it is the
    residual record the probabilistic layer (E31) is built from.
Idempotent per lead. n_jobs pinned.
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
from single_model import CFG44, denominators

FLUX = "f107_adj_rob"
TARGET = "ratio_env"
SEEDS = [11, 23, 37]
N_FOLDS = 5
PARAMS = dict(CFG44)
PARAMS.pop("random_state")


def lead(h):
    out_path = os.path.join(HERE, "out", f"lead_{h:02d}.csv")
    oof_path = os.path.join(HERE, "out", f"oof_{h:02d}.csv")
    if os.path.exists(out_path) and os.path.exists(oof_path):
        print(f"lead {h}: exists, skipping")
        return
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    df = common.load_daily()
    flux = df[FLUX]
    env = common.envelope(df, FLUX)
    X96, y96, o96, _ = common.build_samples(df, "train", extra=common.FS_COLS, flux=FLUX)
    X47, y47, o47, _ = common.build_samples(df, "train47", required=["f107_adj", "ssn"],
                                            extra=common.FS_COLS, flux=FLUX)
    Xva, yva, ova, _ = common.build_samples(df, "val", extra=common.FS_COLS, flux=FLUX)
    Xte, _, ote, _ = common.build_samples(df, "test", extra=common.FS_COLS, flux=FLUX)
    d96, d47, dva, dte = (denominators(df, o, FLUX, TARGET) for o in (o96, o47, ova, ote))
    ok47 = np.isfinite(d47) & (d47 > 0)
    X47, y47, o47, d47 = X47[ok47], y47[ok47], o47[ok47], d47[ok47]
    folds = np.array_split(np.arange(len(o96)), N_FOLDS)
    in47 = o47.get_indexer(o96)
    assert (in47 >= 0).all()
    hc = h - 1

    def fit(X, y, d, seed):
        m = lgb.LGBMRegressor(random_state=seed, **PARAMS)
        m.fit(X, y[:, hc] / d, eval_set=[(Xva, yva[:, hc] / dva)], eval_metric="rmse",
              callbacks=[lgb.early_stopping(150, verbose=False)])
        return m

    def baselines(orig):
        lag = 27 if h <= 27 else 54
        rec = np.array([flux.loc[t + np.timedelta64(h - lag, "D")] for t in orig])
        return np.stack([rec, env.loc[orig].to_numpy(), flux.loc[orig].to_numpy()], axis=1)

    oof96, oof47 = np.empty(len(o96)), np.empty(len(o96))
    for k in folds:
        m96 = np.ones(len(o96), bool); m96[k] = False
        m = fit(X96[m96], y96[m96], d96[m96], SEEDS[0])
        oof96[k] = m.predict(X96[k], num_iteration=m.best_iteration_) * d96[k]
        m47 = np.ones(len(o47), bool); m47[in47[k]] = False
        m = fit(X47[m47], y47[m47], d47[m47], SEEDS[0])
        oof47[k] = m.predict(X47[in47[k]], num_iteration=m.best_iteration_) * d47[in47[k]]
    Ztr = np.column_stack([oof96, oof47, baselines(o96)])
    blend = LinearRegression(positive=True).fit(Ztr, y96[:, hc])
    pd.DataFrame({"t_date": o96.date, "oof_adj": np.round(blend.predict(Ztr), 4),
                  "y_adj": np.round(y96[:, hc], 4)}).to_csv(oof_path, index=False)

    def seed_mean(X, y, d):
        ps = []
        for s in SEEDS:
            m = fit(X, y, d, s)
            ps.append(m.predict(Xte, num_iteration=m.best_iteration_) * dte)
        return np.mean(ps, axis=0)

    Zte = np.column_stack([seed_mean(X96, y96, d96), seed_mean(X47, y47, d47), baselines(ote)])
    pred = blend.predict(Zte)
    pd.DataFrame({"t_date": ote.date, "pred_adj": np.round(pred, 4)}).to_csv(out_path, index=False)
    w = ", ".join(f"{x:.2f}" for x in blend.coef_)
    print(f"lead {h:2d}: w[lgbm96, lgbm47, recur, mean81, persist]=[{w}] b={blend.intercept_:.1f}", flush=True)


if __name__ == "__main__":
    lead(int(sys.argv[1]))
