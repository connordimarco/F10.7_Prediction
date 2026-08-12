#!/usr/bin/env python3
"""C02: 81-day trailing mean of adjusted F10.7 (F10.7a persistence) at every
lead — the "ignore rotation, hold the envelope" baseline."""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import common

df = common.load_daily()
orig = common.origins("test", df)
f107a = df["f107_adj"].rolling(81, min_periods=60).mean()
pred = np.repeat(f107a.loc[orig].to_numpy()[:, None], common.HORIZON, axis=1)
common.write_predictions(HERE, orig, pred)
