#!/usr/bin/env python3
"""SWPC 27-day outlook issues -> deploy/web/swpc.json (for "Show SWPC").

Mirrors the weekly PRF PDFs for this year and last (past years were seeded
from the yearly WeeklyPDF tarballs into data/swpc_prf/<year>/), parses new
ones with benchmarks/swpc_27day.parse_pdf (cached by file name), and adds
the current-issue text product in case this week's PDF is late. Each issue
covers its issue date + 0..26 days; issued Mondays ~02 UT, so it pairs with
a model forecast from the day before (the benchmark's matching rule).
"""
import datetime as dt
import glob
import json
import os
import re
import sys
import urllib.request

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
from swpc_27day import parse_pdf  # noqa: E402

PRF = os.path.join(ROOT, "data", "swpc_prf")
CACHE = os.path.join(PRF, "parsed_cache.json")
OUT = os.path.join(ROOT, "deploy", "web", "swpc.json")
SINCE = "2023-10-01"
FTP = "ftp://ftp.swpc.noaa.gov/pub/warehouse/{y}/WeeklyPDF/"
TEXT = "https://services.swpc.noaa.gov/text/27-day-outlook.txt"
MON = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}


def mirror():
    y = dt.date.today().year
    for year in (y - 1, y):
        try:
            listing = urllib.request.urlopen(FTP.format(y=year), timeout=60).read().decode()
        except Exception:  # last year's loose dir is gone once tarballed
            continue
        os.makedirs(os.path.join(PRF, str(year)), exist_ok=True)
        for name in re.findall(r"(prf\d+\.pdf)", listing):
            dest = os.path.join(PRF, str(year), name)
            if not os.path.exists(dest):
                data = urllib.request.urlopen(FTP.format(y=year) + name, timeout=60).read()
                with open(dest + ".tmp", "wb") as f:
                    f.write(data)
                os.replace(dest + ".tmp", dest)


def text_issue():
    """Current issue from the plain-text product -> (issue_date, [27 fluxes])."""
    txt = urllib.request.urlopen(TEXT, timeout=60).read().decode()
    rows = re.findall(r"^(\d{4}) (\w{3}) (\d{2})\s+(\d+)\s+\d+\s+\d+\s*$", txt, re.M)
    days = [(dt.date(int(y), MON[m], int(d)), float(f)) for y, m, d, f in rows]
    if len(days) != 27:
        raise ValueError(f"27-day outlook text: {len(days)} rows")
    return days[0][0].isoformat(), [f for _, f in days]


def main():
    try:
        mirror()
    except Exception as e:  # noqa: BLE001 — keep going with what is on disk
        print(f"WARNING: SWPC PDF mirror failed: {e}")
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    issues = {}
    for p in sorted(glob.glob(os.path.join(PRF, "**", "prf*.pdf"), recursive=True)):
        name = os.path.basename(p)
        if name not in cache:
            try:
                issue, vals = parse_pdf(p)
                cache[name] = [issue.strftime("%Y-%m-%d"), [vals[k] for k in sorted(vals)]]
            except Exception:  # noqa: BLE001 — a few PDFs lack the table
                cache[name] = None
        if cache[name]:
            issues[cache[name][0]] = cache[name][1]
    with open(CACHE + ".tmp", "w") as f:
        json.dump(cache, f)
    os.replace(CACHE + ".tmp", CACHE)
    try:
        d, vals = text_issue()
        issues.setdefault(d, vals)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: SWPC 27-day outlook text failed: {e}")
    keep = sorted(d for d in issues if d >= SINCE)
    with open(OUT + ".tmp", "w") as f:
        json.dump({"generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "source": "NOAA SWPC 27-day outlook (weekly PRF); values for issue date + 0..26 days",
                   "issues": keep, "f107": [issues[d] for d in keep]}, f, separators=(",", ":"))
    os.replace(OUT + ".tmp", OUT)
    print(f"swpc view: {len(keep)} issues {keep[0]}..{keep[-1]}")


if __name__ == "__main__":
    main()
