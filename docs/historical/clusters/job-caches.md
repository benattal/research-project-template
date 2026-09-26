# Compilation caches in jobs

GPU jobs write several on-disk caches: the OptiX pipeline cache, a JIT
compiler's kernel cache (DrJit's here), the CUDA driver's PTX cache, and
Python bytecode. Shared between concurrent jobs, or placed on a network
filesystem, each of them has corrupted or slowed runs. The rule that held up
is one private, node-local directory per job for each cache, removed when the
job ends.

## Give every job its own OptiX and JIT caches

Point each job's OptiX and JIT kernel caches at directories named by the job
ID, or by the process ID outside the scheduler.

**Why.** Concurrent writers corrupt a shared OptiX cache. The failure is an
abort in native code, not a Python exception, so the log can look like a
clean run with its last lines missing. A fresh OptiX compile costs under a
minute.

**How to apply.**

```bash
root=${SLURM_TMPDIR:-/tmp/$USER}
tag=${SLURM_JOB_ID:-$$}
export OPTIX_CACHE_PATH=$root/optix_cache_$tag
export CUDA_CACHE_PATH=$root/cuda_cache_$tag      # see the next entry
mkdir -p "$OPTIX_CACHE_PATH" "$CUDA_CACHE_PATH"
```

Check that each library reads the variable before relying on it. DrJit
1.3.1 reads no environment variable for its kernel cache and always writes
`$HOME/.drjit` (the library contains no other path), so an exported
`DRJIT_CACHE_DIR` has no effect; with a home directory on NFS, a symlink
from `~/.drjit` to a local or scratch filesystem is the available
relocation. See the [Mitsuba notes](../mitsuba/jit-and-kernel-caches.md).

Processes started by hand on a shared host need the same treatment: one
cache per process, or per GPU when only one process runs on each GPU at a
time (`OPTIX_CACHE_PATH=/tmp/optix-$USER-gpu$GPU`). So does a test suite run
on a GPU.

**Evidence.** A job died with `OPTIX_ERROR_PIPELINE_LINK_ERROR` and exit
status 134 (`SIGABRT`) while five other jobs used the same cache directory.
A test suite on the shared default OptiX cache aborted the interpreter
mid-run with `OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR` when another process
wrote the cache, instead of failing one test.

## Move the CUDA driver's PTX cache off the network filesystem

Set `CUDA_CACHE_PATH` to a per-job node-local directory. This cache is
separate from OptiX's and defaults to `~/.nv/ComputeCache`.

**Why.** When home directories are on NFS, every driver JIT compile reads
and writes that cache over the network. The cache has a size cap
(`CUDA_CACHE_MAXSIZE`), and near the cap, concurrent jobs evict each other's
entries and recompile.

**Evidence.** The home-directory cache sat near its 256 MB cap. A driver
recompile cost 0.9 to 2.0 s against it and about 0.11 s against a node-local
cache. Moving the cache alone took one rendering iteration from 1,231 ms to
177 ms.

## Remove per-job caches when the job ends

Delete the job's cache directories in an exit trap, and refuse to delete any
path whose name does not match a cache prefix.

**Why.** Some clusters never purge node-local `/tmp`, so per-job caches leak
a few directories per job. The caches are build products: DrJit's kernels
are content-addressed and OptiX recompiles in under a minute, so deleting
them loses nothing. The name check keeps a bad variable expansion from
turning the cleanup into an `rm -rf` of something that matters.

**How to apply.** The trap fires on `EXIT` and again on `TERM` or `INT`, so
the cleanup must be idempotent. Arm it after any output-copy trap, so outputs
are copied before build products go; [shell safety](shell-safety.md) shows
how to compose traps.

```bash
cleanup_caches() {
    local d
    for d in ${_CACHE_DIRS:-}; do
        case $(basename "$d") in
            optix_cache_*|cuda_cache_*) rm -rf -- "$d" ;;
            *) echo "cleanup_caches: refusing to remove '$d'" >&2 ;;
        esac
    done
    _CACHE_DIRS=
}
_CACHE_DIRS="$OPTIX_CACHE_PATH $CUDA_CACHE_PATH"
trap_append cleanup_caches EXIT TERM INT
```

**Evidence.** Without the cleanup, one user's `/tmp` on a head node
accumulated 143 orphaned cache directories (646 MB) from four projects.

## Guard the scheduler's per-job directory variable

Where the site provides `$SLURM_TMPDIR`, a per-job node-local directory that
the scheduler purges, use it with a fallback: `${SLURM_TMPDIR:-/tmp/$USER}`.
Unguarded, `"${SLURM_TMPDIR}/optix_cache"` expands to `/optix_cache` on a
cluster that does not set the variable, and the `mkdir` fails for lack of
permission.

## Share a cache on purpose, within one job

A sweep that runs many short runs one after another inside one job, on one
GPU, can share one cache so it pays the OptiX compile once. Set the tag
explicitly (for example `CACHE_TAG=sweep_gpu0`) and never share it with a
concurrent job.

## Clear the OptiX module cache after upgrading DrJit

Delete DrJit's OptiX cache database (`~/.drjit/optix7cache.db`) when
installing a new DrJit version. Modules cached by an older version fail at
the first kernel launch with `CUDA_ERROR_ILLEGAL_ADDRESS`, which looks like a
bug in the new code.

## Keep Python bytecode off the network filesystem

Set `PYTHONPYCACHEPREFIX` to a node-local directory in every job.

**Why.** Python trusts a `.pyc` whose recorded source modification time (in
whole seconds) and size match the source file. An import racing an NFS
update can see the new file attributes with the old contents, compile the
old source, and stamp it with the new attributes. That `.pyc` then passes
validation for every later job. `PYTHONDONTWRITEBYTECODE` stops writes but
still reads existing `__pycache__` entries; the prefix makes Python ignore
them.

**How to apply.**

```bash
export PYTHONPYCACHEPREFIX=/tmp/$USER/pycache
```

The same validation rule applies locally. An edit that leaves a file's size
and whole-second modification time unchanged, such as two quick `sed -i`
edits that swap one fixed-length hash for another with a test run between
them, leaves pytest's cached, assertion-rewritten bytecode in use. `touch`
the edited file, or clear the cache, before running the tests again.

**Evidence.** A probe job printed the new source file's MD5 and, from the
loaded module, the old function signature. Three jobs ran old code this way.

## Keep model download caches on project storage, shared across jobs

Set `HF_HOME` (and `TORCH_HOME` for `torch.hub` weights) to a directory on
project or scratch storage in every login shell and every job script,
download pretrained weights once from a login node, and let jobs read that
cache.

**Why.** Hugging Face libraries otherwise download into
`~/.cache/huggingface`, and pretrained artifacts are large: one frozen
feature encoder was 10.7 GB. A job without the variable downloads again
into the home directory, which on most clusters has the smallest quota.
Unlike the compilation caches above, the hub cache stores files by content
hash and model revision and locks concurrent downloads, so jobs can share
it.

**How to apply.**

```bash
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/hf_cache}"
```
