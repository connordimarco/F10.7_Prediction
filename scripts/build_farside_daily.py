#!/usr/bin/env python3
"""data/farside/ catalogs -> data/farside_daily.csv (one row per day, 2010->).

Per day we use the LATEST available map (t1200, else t0000) from each
catalog — the freshest information an end-of-day forecast origin would have.
A day whose map exists but lists no regions is a legitimate all-zeros row;
a day with no map at all (GONG duty-cycle discard / outage) is NaN for that
family. Seismic strength is a noisy magnitude estimator (Liewer+2017), so
sums are also detection-probability-weighted where the catalog provides one.

f6x has rare pathological rows (per-region flux up to 8e37 on 2025-05-05 vs
a ~500 median; also 2011-07-03, 2026-xx) — GONG pipeline glitches, present
in the source files. Per-region values are therefore winsorized at caps ~10x
the all-years p95 before aggregation, so glitch days rank "very high", not
"astronomically wrong".

fs_*   from GONG f6x   (count, NOAA-crossmatched count, effective area,
                        est. magnetic flux, |phase strength|, flux returning
                        to the front side within 7/14 days)
sard_* from JSOC SARD  (count, strength, min days-to-east-limb; 20.0 =
                        sentinel for "no region in sight", distinct from NaN)
"""

import datetime as dt
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
FS_DIR = os.path.join(DATA, "farside")
ETA_SENTINEL = 20.0

FS_COLS = [
    "fs_n", "fs_nnoaa", "fs_area_sum", "fs_area_max", "fs_flux_sum",
    "fs_flux_max", "fs_flux_wsum", "fs_str_sum", "fs_ret7_flux",
    "fs_ret14_flux",
]
SARD_COLS = ["sard_n", "sard_str_sum", "sard_str_max", "sard_eta_min"]


def parse_f6x(path, day):
    """-> dict of fs_* or None if the file is unusable."""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines or not lines[0].startswith('"C. Long"'):
        return None
    rows = []
    for ln in lines[1:]:
        p = ln.split(None, 9)
        if len(p) < 9:
            continue
        try:
            rows.append(dict(
                strength=min(abs(float(p[3])), 5.0), prob=float(p[4]) / 100.0,
                area=min(float(p[5]), 5e3), flux=min(float(p[6]), 1e5),
                ret=dt.datetime.strptime(p[8], "%Y%m%d").date(),
                noaa=len(p) > 9 and "NOAA" in p[9],
            ))
        except ValueError:
            continue
    flux = np.array([r["flux"] for r in rows])
    area = np.array([r["area"] for r in rows])
    prob = np.array([r["prob"] for r in rows])
    ret_in = lambda n: sum(r["flux"] for r in rows if r["ret"] <= day + dt.timedelta(days=n))
    return dict(
        fs_n=len(rows),
        fs_nnoaa=sum(r["noaa"] for r in rows),
        fs_area_sum=area.sum() if rows else 0.0,
        fs_area_max=area.max() if rows else 0.0,
        fs_flux_sum=flux.sum() if rows else 0.0,
        fs_flux_max=flux.max() if rows else 0.0,
        fs_flux_wsum=(flux * prob).sum() if rows else 0.0,
        fs_str_sum=sum(r["strength"] for r in rows),
        fs_ret7_flux=ret_in(7),
        fs_ret14_flux=ret_in(14),
    )


def parse_sard(path):
    """-> dict of sard_* or None if the file is unusable."""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines or not any("Far-Side Vantage" in ln for ln in lines[:3]):
        return None
    strength, eta = [], []
    for ln in lines:
        if not ln.startswith("FS-"):
            continue
        p = ln.split()
        if len(p) < 6:
            continue
        try:
            strength.append(float(p[3]))
            eta.append(float(p[5]))
        except ValueError:
            continue
    return dict(
        sard_n=len(strength),
        sard_str_sum=sum(strength),
        sard_str_max=max(strength) if strength else 0.0,
        sard_eta_min=min(eta) if eta else ETA_SENTINEL,
    )


def main():
    start, today = dt.date(2010, 1, 1), dt.date.today()
    idx = pd.date_range(start, today, freq="D")
    out = pd.DataFrame(index=idx, columns=FS_COLS + SARD_COLS, dtype=float)
    out.index.name = "date"
    n_maps = {"f6x": 0, "sard": 0}
    for day in idx:
        d = day.date()
        for hh in ("1200", "0000"):  # latest map of the day wins
            p = os.path.join(FS_DIR, "f6x", f"mrf6x{d:%y%m%d}t{hh}.txt")
            if os.path.exists(p) and (row := parse_f6x(p, d)) is not None:
                out.loc[day, FS_COLS] = pd.Series(row)
                n_maps["f6x"] += 1
                break
        for hh in ("12", "00"):
            p = os.path.join(FS_DIR, "sard", f"AR_LIST_{d:%Y.%m.%d}_{hh}0000.txt")
            if os.path.exists(p) and (row := parse_sard(p)) is not None:
                out.loc[day, SARD_COLS] = pd.Series(row)
                n_maps["sard"] += 1
                break
    path = os.path.join(DATA, "farside_daily.csv")
    out.to_csv(path, float_format="%.4g")
    print(f"wrote {path}: {len(out)} days")
    for fam, col in (("f6x", "fs_n"), ("sard", "sard_n")):
        have = out[col].notna()
        print(
            f"  {fam}: {n_maps[fam]} days with maps ({100 * have.mean():.1f}%), "
            f"longest gap {int((~have).astype(int).groupby(have.cumsum()).sum().max())} d, "
            f"mean regions/day {out[col].mean():.2f}"
        )


if __name__ == "__main__":
    main()
