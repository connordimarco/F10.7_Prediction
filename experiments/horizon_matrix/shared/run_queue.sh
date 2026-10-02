#!/bin/sh
# Run cells sequentially (local, one at a time so LightGBM gets the cores):
#   shared/run_queue.sh CELL [CELL ...]     (logs to logs/queue.log)
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
for c in "$@"; do
    echo "=== $c start $(date '+%F %T')" >> logs/queue.log
    shared/run_local.sh "$c" >> logs/queue.log 2>&1 || echo "=== $c FAILED" >> logs/queue.log
    echo "=== $c done $(date '+%F %T')" >> logs/queue.log
done
../../env/bin/python shared/leaderboard.py >> logs/queue.log 2>&1
echo "=== QUEUE COMPLETE $(date '+%F %T')" >> logs/queue.log
