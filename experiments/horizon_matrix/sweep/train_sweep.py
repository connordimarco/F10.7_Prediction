#!/usr/bin/env python3
"""Train one sweep config (E10 feature set) and score it on VAL only.

  train_sweep.py <cfg_idx>

Trains the standard 30 per-lead LightGBM models (train 1996-2021, fs_*
optional-NaN, early stopping on val — identical contract to the E-cells),
predicts the VAL origins, scores in observed space, and writes
out/cfg_<idx>.json. NO test predictions: the sweep tunes on val only.
Idempotent: exits if the result file already exists.
"""

import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import common

idx = int(sys.argv[1])
out_path = os.path.join(HERE, "out", f"cfg_{idx:03d}.json")
if os.path.exists(out_path):
    print(f"cfg {idx}: result exists, skipping")
    sys.exit(0)
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)

with open(os.path.join(HERE, "configs.json")) as f:
    cfg = json.load(f)[idx]

PARAMS = dict(
    objective="regression",
    n_estimators=3000,
    bagging_freq=1,
    verbose=-1,
    # LightGBM's sklearn wrapper defaults to ALL cores and ignores
    # OMP_NUM_THREADS — on tur_ath that meant 8 workers x 512 threads and
    # a ~100x oversubscription collapse (0 configs in 4h, twice). Pin it.
    n_jobs=32,
    **cfg,
)

df = common.load_daily()
Xtr, ytr, _, _ = common.build_samples(df, "train", extra=common.FS_COLS)
Xva, yva, orig_va, _ = common.build_samples(df, "val", extra=common.FS_COLS)

obs = pd.read_csv(
    os.path.join(common.DATA, "daily.csv"), index_col="date", parse_dates=True
)["f107_obs"]

pred = np.empty((len(orig_va), common.HORIZON))
iters = []
for h in range(common.HORIZON):
    m = lgb.LGBMRegressor(random_state=11, **PARAMS)
    m.fit(
        Xtr, ytr[:, h],
        eval_set=[(Xva, yva[:, h])],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    pred[:, h] = m.predict(Xva, num_iteration=m.best_iteration_)
    iters.append(int(m.best_iteration_))

# score on observed F10.7 over val origins (mirrors score_cell's bands)
rows = []
for i, t in enumerate(orig_va):
    tdates = pd.date_range(t + pd.Timedelta(days=1), periods=common.HORIZON)
    p_obs = common.adj_to_obs(pred[i], tdates)
    y_obs = obs.reindex(tdates).to_numpy()
    for h in range(common.HORIZON):
        rows.append((h + 1, p_obs[h], y_obs[h]))
sc = pd.DataFrame(rows, columns=["lead", "p", "y"]).dropna()


def rmse(g):
    return float(np.sqrt(np.mean((g.p - g.y) ** 2)))


result = dict(
    idx=idx,
    config=cfg,
    n=len(orig_va),
    val_rmse=rmse(sc),
    val_rmse_L1_7=rmse(sc[sc.lead <= 7]),
    val_rmse_L8_14=rmse(sc[(sc.lead >= 8) & (sc.lead <= 14)]),
    val_rmse_L15_30=rmse(sc[sc.lead >= 15]),
    val_rmse_hi=rmse(sc[sc.y >= 150]) if (sc.y >= 150).any() else None,
    median_best_iter=int(np.median(iters)),
)
with open(out_path, "w") as f:
    json.dump(result, f, indent=1)
print(f"cfg {idx}: val_rmse {result['val_rmse']:.3f}  (L8-14 {result['val_rmse_L8_14']:.3f})")
