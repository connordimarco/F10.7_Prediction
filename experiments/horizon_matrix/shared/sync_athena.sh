#!/bin/sh
# Mirror the project to Athena over the owner's live `ssh sfe` session
# (ControlMaster socket; open `ssh sfe` in another terminal first).
#   shared/sync_athena.sh          push code + data (no env/, no .git, no *.parquet)
#   shared/sync_athena.sh pull     pull experiment outputs back (out/ dirs, logs/)
set -eu
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
REMOTE=athfe:/nobackupp28/cdimarco/F10.7_Prediction
ssh -O check sfe >/dev/null 2>&1 || { echo "no live sfe session — run 'ssh sfe' in another terminal"; exit 1; }
if [ "${1:-push}" = pull ]; then
    rsync -az \
        --include='*/' --include='out/**' --include='logs/**' --exclude='*' \
        "$REMOTE/experiments/horizon_matrix/" "$ROOT/experiments/horizon_matrix/"
else
    rsync -az \
        --exclude='env/' --exclude='.git/' --exclude='*.parquet' --exclude='data/midl/' \
        --exclude='__pycache__/' --exclude='.DS_Store' --exclude='data/daily_v1_shifted.csv' \
        "$ROOT/" "$REMOTE/"
fi
echo "sync ($1) done $(date '+%F %T')"
