#!/bin/bash
# Run one application preset as a SLURM job.
#
# Usage:
#   sbatch scripts/launchers/train_slurm.sh <application> <preset> [overrides...]
# Example:
#   sbatch scripts/launchers/train_slurm.sh example example-default --seed 1
#
# Change resources with ordinary sbatch flags at submit time, for example
#   sbatch --time=4:00:00 scripts/launchers/train_slurm.sh ...
#
# TODO(template): set the partition and GPU request for the project's cluster,
# and the environment activation below (docs/environments/index.md).
#SBATCH --job-name=train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

APPLICATION="${1:?usage: train_slurm.sh <application> <preset> [overrides...]}"
PRESET="${2:?missing preset}"
shift 2

cd "${SLURM_SUBMIT_DIR:?submit this script with sbatch}"
if [ ! -d "src/apps/${APPLICATION}" ]; then
    echo "unknown application '${APPLICATION}'" >&2
    exit 2
fi

# TODO(template): load environment modules here if the cluster needs them.
source venv/bin/activate

source scripts/lib/job_cache.sh
job_cache_setup
# Thread pools follow the allocation, not the node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -Iseconds)"
echo "commit $(git rev-parse HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
printf 'command: python -m src.apps.%s %s' "$APPLICATION" "$PRESET"
printf ' %q' "$@"
printf '\n'

python -m "src.apps.${APPLICATION}" "$PRESET" "$@"
