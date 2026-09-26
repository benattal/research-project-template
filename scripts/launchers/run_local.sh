#!/usr/bin/env bash
# Run one command detached on one explicit local GPU, with a log and a PID.
#
# Usage:
#   bash scripts/launchers/run_local.sh <gpu-index> -- <command> [args...]
# Example:
#   bash scripts/launchers/run_local.sh 1 -- python -m src.apps.example example-default
#
# Refuses a GPU that already runs compute processes (set ALLOW_SHARED_GPU=1
# only for a small bounded diagnostic under standing permission). Prints the
# user's thread count and the PID to stop the job with: kill <pid>.
set -euo pipefail

GPU="${1:?usage: run_local.sh <gpu-index> -- <command> [args...]}"
shift
if [ "${1:-}" = "--" ]; then
    shift
fi
if [ "$#" -eq 0 ]; then
    echo "run_local.sh: missing command" >&2
    exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

busy="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader)"
if [ -n "$busy" ] && [ "${ALLOW_SHARED_GPU:-0}" != "1" ]; then
    echo "GPU $GPU already runs compute processes: $busy" >&2
    echo "Choose a free GPU; see docs/environments/index.md." >&2
    exit 3
fi
echo "user threads before launch: $(ps -L -u "$USER" --no-headers | wc -l)"

mkdir -p logs
LOG="logs/local_$(date +%Y%m%d_%H%M%S)_gpu${GPU}.log"
# The inner shell owns the per-job caches and removes them when the command
# exits, so it runs the command rather than exec-ing it.
CUDA_VISIBLE_DEVICES="$GPU" nohup bash -c \
    'source scripts/lib/job_cache.sh; job_cache_setup; "$@"' _ "$@" \
    > "$LOG" 2>&1 &
echo "pid $! on GPU $GPU, log $LOG"
