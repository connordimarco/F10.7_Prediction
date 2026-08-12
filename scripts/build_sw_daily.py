#!/usr/bin/env python3
"""Aggregate MIDL 1-min solar wind -> data/sw_daily.csv (daily features).

GeoDGP's 93-driver recipe adapted to daily cadence: the same six covariates
(Bx, By, Bz, Ux, rho, T), daily-aggregated. The sub-hour lag-median "memory"
of the original recipe is supplied here by the harness's 60-day lag window
instead. Median = robust daily level; for Bz and Ux a daily extreme is kept
too (southward-field and fast-stream episodes matter beyond their median).

A day needs >=30% minute coverage, else NaN. Gaps <=3 days interpolated
(flagged sw_filled); longer gaps stay NaN and those windows drop out.
"""

import glob
import os

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
COVS = ["Bx", "By", "Bz", "Ux", "rho", "T"]


def main():
    parts = [pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(DATA, "midl", "midl_*.parquet")))]
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="first")].sort_index()[COVS]

    day = df.resample("1D")
    cov = day["Bz"].count() / 1440.0
    out = day.median().add_prefix("sw_").add_suffix("_med")
    out["sw_Bz_min"] = day["Bz"].min()      # most-southward IMF of the day
    out["sw_Ux_min"] = day["Ux"].min()      # fastest wind of the day (Ux is negative antisunward)
    out[cov < 0.3] = np.nan

    filled = out.interpolate(limit=3, limit_area="inside")
    out_flag = (out["sw_Bz_med"].isna() & filled["sw_Bz_med"].notna()).astype(int)
    out = filled
    out["sw_filled"] = out_flag
    out.index.name = "date"

    path = os.path.join(DATA, "sw_daily.csv")
    out.to_csv(path, float_format="%.3f")
    n_nan = out["sw_Bz_med"].isna().sum()
    print(f"{path}: {len(out)} days {out.index[0].date()} -> {out.index[-1].date()}")
    print(f"  NaN days after interp: {n_nan}  interpolated: {out.sw_filled.sum()}")
    print(f"  low-coverage days (<30%): {(cov < 0.3).sum()}")
    print(out.describe().round(2).to_string())


if __name__ == "__main__":
    main()
