#!/bin/sh
# Fetch every raw archive into data/ (idempotent: existing files are kept).
# Full rebuild after this: parse_srs.py -> build_dataset.py, fetch_midl.py ->
# build_sw_daily.py. Endpoints documented in data/SOURCES.md.
set -eu
cd "$(dirname "$0")/../data"

get() {  # get <dest> <url>
    [ -s "$1" ] && { echo "$1 cached"; return; }
    echo "fetching $1"
    curl -s --fail --max-time 300 -o "$1.tmp" "$2" && mv "$1.tmp" "$1"
}

get f107_penticton_lisird.csv "https://lasp.colorado.edu/lisird/latis/dap/penticton_radio_flux.csv"
get ssn_daily_silso.csv "https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv"
get ssn_monthly_silso.csv "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv"
get sw_all_celestrak.csv "https://celestrak.org/SpaceData/SW-All.csv"

# SRS: yearly tarballs 1996 -> last full year, then current-year daily files.
# Plain FTP only (HTTPS to the warehouse times out).
mkdir -p srs/raw
YEAR=$(date +%Y)
y=1996
while [ "$y" -lt "$YEAR" ]; do
    get "srs/raw/${y}_SRS.tar.gz" "ftp://ftp.swpc.noaa.gov/pub/warehouse/${y}/${y}_SRS.tar.gz"
    [ -d "srs/$y" ] || { mkdir -p "srs/$y" && tar -xzf "srs/raw/${y}_SRS.tar.gz" -C "srs/$y"; }
    y=$((y + 1))
done
mkdir -p "srs/$YEAR"
curl -s --max-time 60 "ftp://ftp.swpc.noaa.gov/pub/warehouse/${YEAR}/SRS/" | awk '{print $NF}' |
while read -r f; do
    get "srs/$YEAR/$f" "ftp://ftp.swpc.noaa.gov/pub/warehouse/${YEAR}/SRS/$f"
done
echo "raw archives complete"
