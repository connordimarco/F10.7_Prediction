#!/usr/bin/env python3
"""Train the 30 E24 boosters once and save them (realtime/models/).

Identical to the E24 cell (single_model.run with flux=f107_adj_rob,
target=ratio_env, span=train47, extra=FS_COLS) except that every fitted model
is kept, and the canonical test-set predictions are written to
out/predictions_test.csv for a bit-for-bit check against the cell's output
(verify.py). Takes a few minutes on 10 cores.
"""

import datetime as dt
import json
import os
import time

import joblib
import lightgbm as lgb
import numpy as np

import e24
from e24 import common

P = dict(e24.CFG44)
P["n_jobs"] = int(os.environ.get("LGBM_THREADS", 10))


def main():
    df = e24.load_table()
    Xtr, ytr, otr, names = common.build_samples(df, "train47", required=e24.REQUIRED, extra=common.FS_COLS, flux=e24.FLUX)
    Xva, yva, ova, _ = common.build_samples(df, "val", extra=common.FS_COLS, flux=e24.FLUX)
    Xte, _, ote, _ = common.build_samples(df, "test", extra=common.FS_COLS, flux=e24.FLUX)
    assert names == e24.NAMES
    env = e24.envelope(df)
    dtr, dva, dte = (env.loc[o].to_numpy() for o in (otr, ova, ote))
    ok = np.isfinite(dtr) & (dtr > 0)
    Xtr, ytr, otr, dtr = Xtr[ok], ytr[ok], otr[ok], dtr[ok]
    print(f"train {Xtr.shape} ({otr[0].date()}..{otr[-1].date()})  val {Xva.shape}  test {Xte.shape}", flush=True)

    os.makedirs(e24.MODELS, exist_ok=True)
    os.makedirs(e24.OUT, exist_ok=True)
    pred = np.empty((len(ote), e24.HORIZON))
    best = []
    t0 = time.time()
    for h in range(e24.HORIZON):
        m = lgb.LGBMRegressor(**P)
        m.fit(Xtr, ytr[:, h] / dtr, eval_set=[(Xva, yva[:, h] / dva)], eval_metric="rmse",
              callbacks=[lgb.early_stopping(150, verbose=False)])
        pred[:, h] = m.predict(Xte, num_iteration=m.best_iteration_) * dte
        joblib.dump(m, os.path.join(e24.MODELS, f"lead_{h + 1:02d}.joblib"))
        best.append(int(m.best_iteration_))
        print(f"lead {h + 1:2d}: best_iter {m.best_iteration_:4d}  val_rmse {m.best_score_['valid_0']['rmse']:.4f}  [{time.time() - t0:.0f}s]", flush=True)

    meta = dict(
        model="F107_E24_ratio_deep", flux=e24.FLUX, target="ratio_env", span="train47",
        params={k: v for k, v in P.items() if k != "n_jobs"},
        feature_names=e24.NAMES, n_features=len(e24.NAMES), hist_days=e24.HIST, horizon_days=e24.HORIZON,
        train_origins=[str(otr[0].date()), str(otr[-1].date()), int(len(otr))],
        val_origins=[str(ova[0].date()), str(ova[-1].date()), int(len(ova))],
        best_iteration=best, data_version=common.DATA_VERSION,
        data_last_day=str(df.index[-1].date()), trained_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        lightgbm=lgb.__version__,
    )
    json.dump(meta, open(os.path.join(e24.MODELS, "meta.json"), "w"), indent=1)
    e24.long_frame(ote, e24.to_obs(pred, ote)).to_csv(os.path.join(e24.OUT, "predictions_test.csv"), index=False)
    print(f"saved 30 models + meta.json to {e24.MODELS}; test predictions for {len(ote)} origins to out/")


if __name__ == "__main__":
    main()
