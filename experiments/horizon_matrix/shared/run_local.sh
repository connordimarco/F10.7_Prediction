#!/bin/sh
# Run one cell locally:  shared/run_local.sh F107_E00_control
# Skip-if-exists: delete <cell>/out/scorecard.json to force a rerun.
set -eu
cd "$(dirname "$0")/.."
CELL=$1
PY=../../env/bin/python
if [ -f "$CELL/out/scorecard.json" ]; then
    echo "scorecard exists, skipping $CELL"
    exit 0
fi
mkdir -p "$CELL/out"
$PY "$CELL/train_cell.py" 2>&1 | tee "$CELL/out/train.log"
$PY shared/score_cell.py "$CELL"
