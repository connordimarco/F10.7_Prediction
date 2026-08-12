#!/usr/bin/env python3
"""C00: raw persistence in OBSERVED space — pred_obs(t+h) = f107_obs(t).

Identical to the scorer's skill reference by construction: its skill must be
exactly 0.000 at every lead, which doubles as a harness integrity check.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import common

df = common.load_daily()
orig = common.origins("test", df)
rows = []
for t in orig:
    v = df.at[t, "f107_obs"]
    for h in range(1, common.HORIZON + 1):
        rows.append((t.date(), h, (t + pd.Timedelta(days=h)).date(), v))
out = pd.DataFrame(rows, columns=["t_date", "lead", "target_date", "pred_obs"])
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
out.to_csv(os.path.join(HERE, "out", "predictions.csv"), index=False)
print(f"C00 persistence: {len(orig)} origins")
