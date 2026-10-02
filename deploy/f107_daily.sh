#!/bin/bash
# Daily E24 F10.7 forecast: refresh data -> issue the forecast -> website
# views (deploy/web/) -> optional rsync to a web host. Meant for cron after
# ~01:30 UT, once the SRS for the UT day just ended is out, e.g.
#   30 21 * * * bash <repo>/deploy/f107_daily.sh >> <repo>/deploy/daily.log 2>&1
# One summary line per run; details go to the same log.
#
# Machine-specific settings live in deploy/local.env (gitignored), e.g.
#   F107_PUBLISH_DEST=user@host:/path/to/site/data/f107/
#   TMPDIR=/some/scratch/dir
# Without F107_PUBLISH_DEST the views are built but not published.
set -uo pipefail
cd "$(dirname "$0")/.."
exec 9>deploy/.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) SKIP previous run still going"; exit 0; }
if [ -f deploy/local.env ]; then set -a; . deploy/local.env; set +a; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=env/bin/python
echo "== $(date -u +%FT%TZ) start"

nice sh realtime/refresh_data.sh; refresh=$?
# predict.py exits 1 when the origin was already issued (it never overwrites
# the archive); that is not a failure here.
nice $PY realtime/predict.py; predict=$?
nice $PY deploy/web_json.py; web=$?
# SWPC 27-day outlook issues for the page's "Show SWPC"; on failure the
# previous swpc.json stays in deploy/web/ and is re-published as is.
nice $PY deploy/swpc_web.py; swpc=$?
pub=skip
if [ $web -eq 0 ] && [ -n "${F107_PUBLISH_DEST:-}" ]; then
    timeout -k 10 120 rsync -a --timeout=45 --chmod=D755,F644 --exclude="*.tmp" \
        -e "ssh -o BatchMode=yes -o ConnectTimeout=10" \
        --delete deploy/web/ "$F107_PUBLISH_DEST" \
        && pub=ok || pub=FAIL
fi
echo "$(date -u +%FT%TZ) DONE refresh=$refresh predict=$predict web=$web swpc=$swpc pub=$pub latest=$(sed -n 2p realtime/forecasts/latest.csv | cut -d, -f1)"
