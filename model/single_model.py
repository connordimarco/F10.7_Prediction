"""Single-model screening runner (v2 data era, 2026-09-02).

One LightGBM per lead on the E10 feature set (60-day windows of the 12
daily series + 14 far-side aggregates as optional-NaN), cfg-44 params,
early stopping on val. Cells call run(**options) with ONE change each:

  flux     "f107_adj" (canonical noon value) | "f107_adj_rob" (flare-robust)
  target   "raw"        y = flux(t+h)
           "ratio_env"  y = flux(t+h) / env81(t)   (pred x env81 at predict)
           "ratio_pers" y = flux(t+h) / flux(t)
  span     "train" (1996-2021) | "train47" (1947-2021, ar_*/fs_* optional)
  weights  None | "env"  sample weight = env81(t) / mean env81 (clipped 0.5-3)
  params   dict of LightGBM overrides (e.g. min_child_samples)
"""

import os
import sys
import time

import lightgbm as lgb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

CFG44 = dict(
    objective="regression",
    n_estimators=3000,
    learning_rate=0.0161,
    num_leaves=31,
    min_child_samples=160,
    feature_fraction=0.725,
    bagging_fraction=0.694,
    lambda_l1=1.4619,
    bagging_freq=1,
    verbose=-1,
    n_jobs=int(os.environ.get("LGBM_THREADS", 10)),
    random_state=11,
)


def denominators(df, orig, flux, target):
    if target == "raw":
        return np.ones(len(orig))
    if target == "ratio_env":
        return common.envelope(df, flux).loc[orig].to_numpy()
    if target == "ratio_pers":
        return df[flux].loc[orig].to_numpy()
    raise ValueError(target)


def run(cell_dir, flux="f107_adj", target="raw", span="train", weights=None, params=None, extra=None):
    P = dict(CFG44, **(params or {}))
    extra = list(common.FS_COLS) + list(extra or [])
    df = common.load_daily()
    required = ["f107_adj", "ssn"] if span == "train47" else None
    Xtr, ytr, otr, names = common.build_samples(df, span, required=required, extra=extra, flux=flux)
    Xva, yva, ova, _ = common.build_samples(df, "val", extra=extra, flux=flux)
    Xte, _, ote, _ = common.build_samples(df, "test", extra=extra, flux=flux)
    dtr, dva, dte = (denominators(df, o, flux, target) for o in (otr, ova, ote))
    ok = np.isfinite(dtr) & (dtr > 0)  # env81 needs 60 prior days (1947 span head)
    Xtr, ytr, otr, dtr = Xtr[ok], ytr[ok], otr[ok], dtr[ok]
    w = None
    if weights == "env":
        e = common.envelope(df, flux).loc[otr].to_numpy()
        w = np.clip(e / np.nanmean(e), 0.5, 3.0)
    print(f"{os.path.basename(cell_dir)}: flux={flux} target={target} span={span} "
          f"weights={weights} params={params or {}} extra={extra[len(common.FS_COLS):]} | train {Xtr.shape} val {Xva.shape} test {Xte.shape}",
          flush=True)
    pred = np.empty((len(ote), common.HORIZON))
    t0 = time.time()
    for h in range(common.HORIZON):
        m = lgb.LGBMRegressor(**P)
        m.fit(Xtr, ytr[:, h] / dtr, sample_weight=w,
              eval_set=[(Xva, yva[:, h] / dva)], eval_metric="rmse",
              callbacks=[lgb.early_stopping(150, verbose=False)])
        pred[:, h] = m.predict(Xte, num_iteration=m.best_iteration_) * dte
        print(f"lead {h + 1:2d}: best_iter {m.best_iteration_:4d}  "
              f"val_rmse(target space) {m.best_score_['valid_0']['rmse']:.4f}  "
              f"[{time.time() - t0:.0f}s]", flush=True)
    common.write_predictions(cell_dir, ote, pred)
