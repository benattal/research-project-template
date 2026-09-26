# Knowing which code a job runs

A job imports whatever source its node shows when each of its processes
starts. Between an edit and that import sit a deployment step, the network
filesystem's attribute cache, and possibly further edits to the same tree.
These entries make the imported code knowable and keep edits from leaking
into runs that are already going. Stale bytecode, a related cause, is in
[job caches](job-caches.md).

## Treat the cluster deployment as a separate checkout, verified by hash

Editing a local checkout does not change what cluster jobs import. Deploy
with `rsync`, then compare hashes of the changed files on both sides. A zero
exit status is not proof.

**Why.** Interactive `cp -i` aliases, partial transfers and exclusion rules
can leave the old file in place while the command succeeds.

**How to apply.** Exclude generated state (outputs, logs, local
environments, caches, build-only third-party trees), but not source that
jobs import, such as vendored drivers under a third-party directory.

```bash
rsync -a --exclude outputs/ --exclude logs/ --exclude venv/ \
      --exclude '__pycache__/' ./ "$DEPLOY/"
for f in $CHANGED; do
    [ "$(md5sum < "$f")" = "$(md5sum < "$DEPLOY/$f")" ] || echo "STALE $f"
done
```

## Print the hashes of the modules a job imports

At job start, print a hash of each module whose version matters, so a stale
import is visible in the log.

**Why.** Compute nodes cache NFS file attributes and contents for a short
time, so a job submitted shortly after a deployment can import the old file.
The symptom is either an argument parser rejecting a new option, such as
`Unrecognized options: --training.seed`, or a plausible result from the old
code.

**How to apply.**

```bash
for f in src/pkg/config.py src/pkg/model.py; do
    [ -f "$f" ] && echo "deployed $(md5sum "$f")"
done
```

## Gate a job on expected hashes when the version matters

Pass the expected hashes with the submission. The job waits for its node to
show those contents and aborts if it still does not after a bounded wait.

```bash
# submitted as: EXPECT="src/pkg/config.py:<md5>,src/pkg/model.py:<md5>" \
#                   sbatch job.sh   (the default --export=ALL carries it)
for spec in ${EXPECT//,/ }; do
    file=${spec%%:*}; want=${spec##*:}
    for _ in $(seq 1 60); do
        [ "$(md5sum "$file" | cut -d' ' -f1)" = "$want" ] && break
        echo "waiting for $file"; sleep 10
    done
    [ "$(md5sum "$file" | cut -d' ' -f1)" = "$want" ] ||
        { echo "ABORT: stale $file" >&2; exit 1; }
done
```

## Every new process imports the working tree as it is at that moment

Each link of a chained run, each post-hoc evaluation step and each requeued
job is a fresh process that imports the working tree at its start,
committed or not. An edit made while a chain runs enters it at the next link
boundary.

**Why.** Python reads source at import time; nothing ties a process to the
commit its run started from.

**How to apply.** Make multi-file edits in a separate `git worktree` and
merge them in one step once tests pass. When a change alters numerics (a
gradient, a fit, a loss), restart the affected run rather than letting its
chain continue, or launch long chains from a worktree pinned to a commit.
Record the commit and the source hashes with every link.

**Evidence.** A half-edited tree kills the next link at import. Two
uncommitted numerical fixes, made in the shared tree by parallel sessions,
were found about to enter running chains at their next link boundary, which
would have mixed two estimators in one plotted trajectory.

## Under `sbatch`, locate the repository from `SLURM_SUBMIT_DIR`

SLURM copies a batch script to a spool directory on the compute node and
runs the copy, so `$0` and `BASH_SOURCE` name the spool directory, not the
repository. Resolve the repository from `SLURM_SUBMIT_DIR` under the
scheduler and from the script's own location otherwise, change into it, and
check for a file that marks the repository root:

```bash
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
[ -f pyproject.toml ] ||
    { echo "submit from the repository root ($REPO)" >&2; exit 2; }
```

`SLURM_SUBMIT_DIR` is the directory `sbatch` ran in, so submit from the
repository root. The marker check turns a submission from another directory
into an immediate error instead of a job that reads and writes relative
paths somewhere else.

## An editable install imports the checkout it was installed from

`pip install -e` records one checkout's path. A script in a `git worktree`
or a second clone that does not put its own root first on `sys.path`
imports the package from the installed checkout, without any message.
`python path/to/script.py` puts the script's directory on `sys.path`, not
the current directory, so a script under `scripts/` does not see the package
beside it.

**How to apply.** Run tools as modules from the tree's root
(`python -m pkg.tool`), or have scripts insert the root, found by a marker
file rather than by counting parent directories, at the front of
`sys.path`. Print `pkg.__file__` at the start of each job.

**Evidence.** A sweep script inserted a path one level short of the
repository root. Its imports resolved through the editable install, so
running it from a worktree imported the main checkout's code.
