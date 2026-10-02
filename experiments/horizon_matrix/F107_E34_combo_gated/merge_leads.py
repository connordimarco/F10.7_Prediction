#!/usr/bin/env python3
"""Assemble out/lead_*.csv (30 files from e34_lead.py) into the standard
out/predictions.csv via common.write_predictions."""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import common

df = common.load_daily()
_, _, orig_te, _ = common.build_samples(df, "test", extra=common.FS_COLS)

pred = np.empty((len(orig_te), common.HORIZON))
for h in range(1, common.HORIZON + 1):
    p = pd.read_csv(os.path.join(HERE, "out", f"lead_{h:02d}.csv"),
                    parse_dates=["t_date"])
    assert len(p) == len(orig_te), f"lead {h}: {len(p)} rows != {len(orig_te)}"
    assert (pd.DatetimeIndex(p.t_date) == orig_te).all(), f"lead {h}: origin mismatch"
    pred[:, h - 1] = p.pred_adj.to_numpy()
common.write_predictions(HERE, orig_te, pred)
