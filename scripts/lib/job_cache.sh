# shellcheck shell=bash
# Per-job JIT and driver caches that delete themselves when the job ends.
#
# Why: compilers and GPU drivers keep on-disk caches (CUDA's PTX cache,
# Triton, TorchInductor, OptiX, DrJit, ...). Two jobs that write one shared
# cache at the same time can corrupt it, and a default cache under an NFS home
# directory is slow and small. Each job therefore gets its own node-local
# directories. They are pure build products: deleting them costs one
# recompilation.
#
# Usage, the common case:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/../lib/job_cache.sh"
#     job_cache_setup
#
# When the script arms its own EXIT trap, defer the cleanup so the traps
# compose in the right order:
#
#     job_cache_setup --defer-trap
#     trap_append sync_outputs EXIT TERM INT
#     job_cache_arm_cleanup
#
# Both entry points are idempotent and safe to call outside SLURM.
#
# TODO(template): export the cache variables of the project's own frameworks
# in job_cache_setup, using one of the prefixes job_cache_cleanup accepts.

# Append a command to a trap instead of replacing whatever is already armed.
# `trap -p SIG` prints  trap -- 'body' SIG  (nothing when unarmed). Bodies
# containing a literal single quote would not survive this; keep trap bodies
# to function names.
trap_append() {
    local cmd=$1; shift
    local sig prev
    for sig in "$@"; do
        prev=$(trap -p "$sig")
        prev=${prev#trap -- \'}
        prev=${prev%\' $sig}
        if [ -n "$prev" ]; then
            trap -- "${prev}; ${cmd}" "$sig"
        else
            trap -- "${cmd}" "$sig"
        fi
    done
}

# Remove this job's cache directories. Idempotent: the trap fires on EXIT and
# again on TERM/INT.
job_cache_cleanup() {
    [ -n "${_JOB_CACHE_DIRS:-}" ] || return 0
    local d
    for d in $_JOB_CACHE_DIRS; do
        # Refuse anything that is not one of our own cache directories: a bad
        # expansion here would be an rm -rf on something that matters.
        case "$(basename "$d")" in
            cuda_cache_*|triton_cache_*|inductor_cache_*) rm -rf -- "$d" ;;
            *) echo "job_cache_cleanup: refusing to remove '$d'" >&2 ;;
        esac
    done
    _JOB_CACHE_DIRS=""
}

job_cache_arm_cleanup() {
    trap_append job_cache_cleanup EXIT TERM INT
}

# Export the cache variables, create the directories, and (unless
# --defer-trap) arm the cleanup.
#
# Base directory, in order of preference:
#   $JOB_CACHE_ROOT  explicit override
#   $SLURM_TMPDIR    a scheduler's per-job node-local directory, if provided
#   /tmp/$USER       otherwise
# The suffix defaults to the job ID, which is what makes the caches per job.
# $JOB_CACHE_TAG overrides it for a sweep driver that deliberately shares one
# cache across the sequential runs of a sweep on one GPU.
job_cache_setup() {
    local defer=0
    [ "${1:-}" = "--defer-trap" ] && defer=1
    local root="${JOB_CACHE_ROOT:-${SLURM_TMPDIR:-/tmp/${USER}}}"
    local tag="${JOB_CACHE_TAG:-${SLURM_JOB_ID:-$$}}"
    export CUDA_CACHE_PATH="${root}/cuda_cache_${tag}"
    export TRITON_CACHE_DIR="${root}/triton_cache_${tag}"
    export TORCHINDUCTOR_CACHE_DIR="${root}/inductor_cache_${tag}"
    mkdir -p "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
    _JOB_CACHE_DIRS="$CUDA_CACHE_PATH $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR"
    [ "$defer" -eq 1 ] || job_cache_arm_cleanup
}
