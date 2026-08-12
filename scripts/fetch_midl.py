#!/usr/bin/env python3
"""Download MIDL solar wind @ 14 Re, monthly chunks -> data/midl/*.parquet.

Adapted from LAUREN's fetch_midl.py. Span 2005-01 -> last complete month
(MIDL is only to be used 2005-on; earlier files exist from 1999 but are not
trusted for training). Cached: existing parquets are skipped, so rerun to
extend as new months appear.
"""

import os
import sys
import time
from pathlib import Path

import midl

OUT = Path(__file__).resolve().parent.parent / "data" / "midl"
OUT.mkdir(parents=True, exist_ok=True)

months = []
for y in range(2005, 2027):
    for m in range(1, 13):
        if (y, m) >= (2026, 7):  # last posted file at fetch time: 2026-06
            break
        months.append((y, m))

fail = 0
for y, m in months:
    dest = OUT / f"midl_{y}{m:02d}.parquet"
    if dest.exists():
        continue
    start = f"{y}-{m:02d}-01 00:00"
    end = f"{y + 1}-01-01 00:00" if m == 12 else f"{y}-{m + 1:02d}-01 00:00"
    try:
        df = midl.load(start, end, 14).to_dataframe()
    except Exception as e:
        print(f"FAIL {y}-{m:02d}: {str(e)[:100]}", flush=True)
        fail += 1
        time.sleep(5)
        continue
    tmp = dest.with_suffix(".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, dest)
    cov = df[["Bz", "Ux", "rho"]].notna().mean().round(3).to_dict()
    print(f"{dest.name}: {len(df)} rows coverage {cov}", flush=True)
    time.sleep(1)

print(f"done, {fail} failures")
sys.exit(1 if fail else 0)
