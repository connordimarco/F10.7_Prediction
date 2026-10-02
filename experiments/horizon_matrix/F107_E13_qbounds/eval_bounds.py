#!/usr/bin/env python3
"""Coverage / width diagnostics for the E13 p10-p90 band on the test set.

Target: 80% of observed values inside the band — overall, by lead band,
by year (guards the val->test shift that broke E08), and in the
high-activity bin (true F10.7 >= 150).
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import common

obs = pd.read_csv(os.path.join(common.DATA, "daily.csv"),
                  index_col="date", parse_dates=True)["f107_obs"]
p = pd.read_csv(os.path.join(HERE, "out", "predictions.csv"),
                parse_dates=["t_date", "target_date"])
p["y"] = obs.reindex(p.target_date).to_numpy()
p = p.dropna(subset=["y"])
p["in_band"] = (p.y >= p.p10_obs) & (p.y <= p.p90_obs)
p["width"] = p.p90_obs - p.p10_obs


def line(label, g):
    print(f"  {label:<14} coverage {100 * g.in_band.mean():5.1f}%   "
          f"median width {g.width.median():5.1f} sfu   (n={len(g)})")


print(f"E13 p10-p90 band on test (target 80%):")
line("overall", p)
for lo, hi in ((1, 7), (8, 14), (15, 30)):
    line(f"L{lo}-{hi}", p[(p.lead >= lo) & (p.lead <= hi)])
for yr, g in p.groupby(p.t_date.dt.year):
    line(str(yr), g)
line("hi-act y>=150", p[p.y >= 150])
