#!/usr/bin/env python3
"""Assemble E14 out/lead_*.csv -> out/bounds.csv in OBSERVED space
(t_date, lead, target_date, p10_obs, p90_obs)."""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import common

df = common.load_daily()
_, _, orig_te, _ = common.build_samples(df, "test", extra=common.FS_COLS)

rows = []
for h in range(1, common.HORIZON + 1):
    p = pd.read_csv(os.path.join(HERE, "out", f"lead_{h:02d}.csv"),
                    parse_dates=["t_date"])
    assert (pd.DatetimeIndex(p.t_date) == orig_te).all(), f"lead {h} mismatch"
    tdates = orig_te + pd.Timedelta(days=h)
    lo = common.adj_to_obs(p.p10_adj.to_numpy(), tdates)
    hi = common.adj_to_obs(p.p90_adj.to_numpy(), tdates)
    for i, t in enumerate(orig_te):
        rows.append((t.date(), h, tdates[i].date(),
                     round(float(lo[i]), 2), round(float(hi[i]), 2)))
out = pd.DataFrame(rows, columns=["t_date", "lead", "target_date",
                                  "p10_obs", "p90_obs"])
path = os.path.join(HERE, "out", "bounds.csv")
out.to_csv(path, index=False)
print(f"wrote {path}: {len(orig_te)} origins x {common.HORIZON} leads")
