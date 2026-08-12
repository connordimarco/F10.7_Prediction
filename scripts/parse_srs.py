#!/usr/bin/env python3
"""Parse SWPC Solar Region Summary text files into one tidy CSV.

Reads every data/srs/**/YYYYMMDDSRS.txt file and emits data/srs_parsed.csv
with one row per region per day, from all three report sections:
  I   regions with sunspots (location, area, class, spot count)
  IA  H-alpha plages without spots (location only)
  II  regions due to return (lat + Carrington longitude only)

Conventions: lat north-positive, lon east-positive (matches SWPC
solar_regions.json). valid_date is the file date minus one day — SRS issued
on day D reports locations valid at (D-1)/2400Z. Region numbers are as
printed, i.e. mod 10000.
"""

import csv
import glob
import os
import re
import sys
from datetime import date, timedelta

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(ROOT, "srs_parsed.csv")

LOC_RE = re.compile(r"^([NS])(\d{1,2})([EW])(\d{1,3})$")
SEC_RE = re.compile(r"^(I|IA|II)\.", re.IGNORECASE)
NUM_RE = re.compile(r"^\d{1,4}$")


def parse_location(tok):
    m = LOC_RE.match(tok)
    if not m:
        return None
    lat = int(m.group(2)) * (1 if m.group(1) == "N" else -1)
    lon = int(m.group(4)) * (1 if m.group(3) == "E" else -1)
    return lat, lon


def parse_file(path, warn):
    fname = os.path.basename(path)
    d = date(int(fname[0:4]), int(fname[4:6]), int(fname[6:8]))
    valid = (d - timedelta(days=1)).isoformat()
    section = None
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith((":", "#")):
                continue
            u = line.upper()
            if u.startswith("IA."):
                section = "IA"
                continue
            if u.startswith("II."):
                section = "II"
                continue
            if u.startswith("I."):
                section = "I"
                continue
            if section is None or u == "NONE" or u.startswith(("NMBR", "COMMENT")):
                continue
            parts = line.split()
            if not NUM_RE.match(parts[0]):
                continue  # prose, e.g. forecaster comments
            row = {
                "valid_date": valid,
                "section": section,
                "region": int(parts[0]),
                "lat": "",
                "lon": "",
                "carrington_lo": "",
                "area": "",
                "mcintosh": "",
                "extent": "",
                "num_spots": "",
                "mag_type": "",
            }
            try:
                if section in ("I", "IA"):
                    loc = parse_location(parts[1])
                    if loc is None:
                        warn.append(f"{fname}: bad location {line!r}")
                        continue
                    row["lat"], row["lon"] = loc
                    row["carrington_lo"] = int(parts[2])
                    if section == "I":
                        row["area"] = int(parts[3])
                        row["mcintosh"] = parts[4]
                        row["extent"] = int(parts[5])
                        row["num_spots"] = int(parts[6])
                        row["mag_type"] = parts[7].upper()
                else:  # II: Nmbr Lat Lo
                    m = re.match(r"^([NS])(\d{1,2})$", parts[1])
                    if not m:
                        warn.append(f"{fname}: bad return lat {line!r}")
                        continue
                    row["lat"] = int(m.group(2)) * (1 if m.group(1) == "N" else -1)
                    row["carrington_lo"] = int(parts[2])
            except (IndexError, ValueError):
                warn.append(f"{fname}: unparseable {line!r}")
                continue
            rows.append(row)
    return rows


def main():
    files = sorted(
        glob.glob(os.path.join(ROOT, "srs", "**", "[12]*SRS.txt"), recursive=True),
        key=os.path.basename,
    )
    warn, rows, empty = [], [], 0
    for p in files:
        if os.path.getsize(p) == 0:
            empty += 1
            continue
        rows.extend(parse_file(p, warn))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    days = {r["valid_date"] for r in rows}
    print(f"files: {len(files)} ({empty} empty)  rows: {len(rows)}  days with regions: {len(days)}")
    for s in ("I", "IA", "II"):
        print(f"  section {s}: {sum(1 for r in rows if r['section'] == s)} rows")
    print(f"warnings: {len(warn)}")
    for m in warn[:20]:
        print("  " + m)
    if len(warn) > 20:
        print(f"  ... {len(warn) - 20} more")


if __name__ == "__main__":
    sys.exit(main())
