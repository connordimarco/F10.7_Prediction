#!/usr/bin/env python3
"""Checks that the realtime path reproduces the E24 experiment cell.

 1. out/predictions_test.csv (written by train.py) vs the cell's
    out/predictions.csv on the canonical test origins: must match to 0.01 sfu.
 2. The standalone feature builder (what predict.py uses) vs the matrix's
    build_samples on 50 random test origins: identical rows, identical
    predictions.
 3. Band sanity: coverage of the q05-q95 / q10-q90 bands on the calibration
    era (in-sample, so ~0.90 / ~0.80 by construction).
"""

import json
import os
import sys

import numpy as np
import pandas as pd

import e24
from e24 import common

CELL = os.path.join(e24.ROOT, "experiments", "horizon_matrix", "F107_E24_ratio_deep", "out", "predictions.csv")


def main():
    ok = True
    a = pd.read_csv(os.path.join(e24.OUT, "predictions_test.csv"), parse_dates=["t_date", "target_date"])
    b = pd.read_csv(CELL, parse_dates=["t_date", "target_date"])
    m = a.merge(b, on=["t_date", "lead", "target_date"], suffixes=("_rt", "_cell"))
    d = (m.pred_obs_rt - m.pred_obs_cell).abs()
    print(f"1. retrained vs cell: {len(m)} pairs (cell {len(b)}, retrained {len(a)}); max |diff| {d.max():.3f} sfu; "
          f"{(d > 0.011).sum()} pairs differ by > 0.01")
    ok &= len(m) == len(b) and d.max() <= 0.011

    df = e24.load_table()
    models, _ = e24.load_models()
    Xte, _, ote, _ = common.build_samples(df, "test", extra=common.FS_COLS, flux=e24.FLUX)
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(ote), 50, replace=False))
    X1 = e24.feature_rows(df, ote[pick])
    same = np.array_equal(np.nan_to_num(X1, nan=-9e9), np.nan_to_num(Xte[pick], nan=-9e9))
    env = e24.envelope(df).loc[ote[pick]].to_numpy()
    P1 = e24.to_obs(e24.predict_adj(models, X1, env), ote[pick])
    ref = a.set_index(["t_date", "lead"]).pred_obs
    P0 = np.array([[ref.loc[(t, h + 1)] for h in range(e24.HORIZON)] for t in ote[pick]])
    dd = np.abs(P1 - P0).max()
    print(f"2. standalone feature rows == build_samples rows: {same}; predict path vs batch: max |diff| {dd:.3f} sfu")
    ok &= same and dd <= 0.011

    bp = os.path.join(e24.MODELS, "band.json")
    if os.path.exists(bp):
        band = json.load(open(bp))
        orig = e24.valid_origins(df, band["origins"][0], band["origins"][1], need_truth=True)
        X = e24.feature_rows(df, orig)
        P = e24.to_obs(e24.predict_adj(models, X, e24.envelope(df).loc[orig].to_numpy()), orig)
        pos = df.index.get_indexer(orig)
        Y = np.stack([df.f107_obs.to_numpy()[p + 1 : p + 1 + e24.HORIZON] for p in pos])
        lo90 = np.array([np.exp(band["band"][str(h)]["q05"]) for h in range(1, 31)])
        hi90 = np.array([np.exp(band["band"][str(h)]["q95"]) for h in range(1, 31)])
        lo80 = np.array([np.exp(band["band"][str(h)]["q10"]) for h in range(1, 31)])
        hi80 = np.array([np.exp(band["band"][str(h)]["q90"]) for h in range(1, 31)])
        c90 = np.mean((Y >= P * lo90) & (Y <= P * hi90))
        c80 = np.mean((Y >= P * lo80) & (Y <= P * hi80))
        print(f"3. band on {len(orig)} calibration origins: 90% band covers {c90:.3f}, 80% band covers {c80:.3f}")
    else:
        print("3. no band.json yet (run calibrate.py)")
    print("ALL OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
