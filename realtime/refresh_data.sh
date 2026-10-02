#!/bin/sh
# Daily data refresh for the realtime E24 forecast. Re-pulls the archives that
# update in place (LISIRD F10.7, SILSO SSN), the current year's SRS files, and
# the far-side catalogs; rebuilds data/daily.csv + data/farside_daily.csv; then
# bridges the last days from EISN / SWPC. ~3-5 min (GONG is rate-limited).
set -eu
cd "$(dirname "$0")/.."
PY=env/bin/python
echo "== $(date -u '+%F %T UTC') refresh start"
rm -f data/f107_penticton_lisird.csv data/ssn_daily_silso.csv
sh scripts/fetch_raw.sh | grep -v cached || true
$PY scripts/fetch_farside.py all
$PY scripts/parse_srs.py
$PY scripts/build_dataset.py
$PY scripts/build_farside_daily.py
cd realtime && ../$PY bridge_tail.py
echo "== $(date -u '+%F %T UTC') refresh done"
