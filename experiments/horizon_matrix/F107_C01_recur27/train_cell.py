#!/usr/bin/env python3
"""C01: 27-day solar-rotation recurrence, in adjusted space.

pred_adj(t+h) = f107_adj(t+h-27) for h<=27, else f107_adj(t+h-54) — always
the most recent same-longitude observation. Converted to observed at the
target date (common.write_predictions).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import common

df = common.load_daily()
orig = common.origins("test", df)
adj = df["f107_adj"]
pred = np.empty((len(orig), common.HORIZON))
for i, t in enumerate(orig):
    for h in range(1, common.HORIZON + 1):
        lag = 27 if h <= 27 else 54
        pred[i, h - 1] = adj.loc[t + np.timedelta64(h - lag, "D")]
common.write_predictions(HERE, orig, pred)
