#!/usr/bin/env python3
"""Tail diagnostic across cells (read-only; complements the fixed scorer).

Usage: tail_report.py <cell_dir> [<cell_dir> ...]

For each cell's out/predictions.csv on the canonical test origins: pooled
RMSE, RMSE + bias by truth bin, max prediction vs max truth, RMSE by
target year. The matrix's failure mode is a capped high tail (bias -56 sfu
above 240 in the v1 champion), so this is the view the scorecard lacks.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

BINS = [0, 150, 180, 210, 240, 400]


def main(cells):
    df = common.load_daily()
    orig = common.origins("test", df)
    truth = df["f107_obs"]
    rows = []
    for c in cells:
        p = pd.read_csv(os.path.join(c, "out", "predictions.csv"), parse_dates=["t_date", "target_date"])
        p = p[p.t_date.isin(orig)]
        p["y"] = truth.reindex(p.target_date).to_numpy()
        p = p.dropna(subset=["y"])
        e = p.pred_obs - p.y
        r = {"cell": os.path.basename(c.rstrip("/")), "RMSE": np.sqrt((e**2).mean()),
             "maxpred": p.pred_obs.max()}
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = (p.y > lo) & (p.y <= hi)
            r[f"rmse{lo}-{hi}"] = np.sqrt((e[m] ** 2).mean())
            r[f"bias{lo}-{hi}"] = e[m].mean()
        for yr, g in p.groupby(p.target_date.dt.year):
            r[f"rmse{yr}"] = np.sqrt(((g.pred_obs - g.y) ** 2).mean())
        rows.append(r)
    out = pd.DataFrame(rows).set_index("cell")
    pd.set_option("display.width", 250)
    print(f"max truth {p.y.max():.0f}; truth-bin columns are (lo, hi] sfu")
    print(out.round(1).to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
