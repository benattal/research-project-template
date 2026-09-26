# Output syncing, retention and cleanup

Run outputs move from node-local disk to shared storage and then to a
durable remote copy, and each move can lose data without an error. Cleanup
later deletes files that are sometimes the only record of a run. These
entries cover getting outputs off the node, surviving a full quota, deciding
what to keep, and deleting only through explicit, verified manifests.

## Create the scheduler's log directory before submitting

`#SBATCH --output=logs/%x_%j.out` is resolved relative to the submission
directory, and SLURM does not create missing directories. If `logs/` is
absent, the job fails at start and writes no log at all. A `mkdir -p logs`
inside the batch script runs too late for the script's own log. Track the
directory with a placeholder file and ignore its contents:

```gitignore
logs/*
!logs/README.md
```

## Write outputs to node-local disk and copy them back at exit

Have the program write into a per-job node-local directory, and copy the tree
to shared storage from an exit trap that also runs on `TERM`.

**Why.** Many small writes to a network filesystem stall a training loop. At
the time limit SLURM sends `SIGTERM` and then `SIGKILL` after the `KillWait`
grace period (`scontrol show config | grep KillWait`), so the copy made at a
timeout has to fit in that period. The next entries keep it small.

**How to apply.**

```bash
OUT_LOCAL=/tmp/$USER/runs/$SLURM_JOB_ID
mkdir -p "$OUT_LOCAL"
trap_append sync_outputs EXIT TERM INT    # arm before the cache cleanup
srun python -m app ... --output-dir "$OUT_LOCAL"
```

**Evidence.** The shared filesystem sustained about 120 KB/s on workloads of
many small files and about 350 MB/s on large files.

## Copy the artifacts needed to evaluate a run first

Order the exit copy by importance: first the small set of files that
evaluation reads, then the bulk of the tree. If the quota fills or the grace
period ends partway through, the run can still be scored.

```bash
for d in "$OUT_LOCAL"/*/; do
    run=$(basename "$d")
    rsync -aR "$OUT_LOCAL/./$run/final/" "$OUT_LOCAL/./$run/config.yaml" \
          "$DEST/" || echo "critical-artifact copy failed for $run"
done
```

The `/./` marks where `rsync -R` starts the relative path it recreates under
`$DEST`.

**Evidence.** A bulk copy that died with `Disk quota exceeded` partway
through the tree delivered training snapshots and an empty render
directory: a run that trained correctly but could not be scored.

## Pack numbered snapshot series into one archive per run

Exclude per-iteration files from the file-by-file copy and ship them as one
compressed archive per run. Extract an archive onto local disk when it is
needed.

```bash
set -o pipefail
rsync -a --exclude='*_[0-9][0-9][0-9][0-9][0-9].*' "$OUT_LOCAL"/ "$DEST"/
for d in "$OUT_LOCAL"/*/; do
    run=$(basename "$d")
    tar -C "$OUT_LOCAL" -cf - "$run" | zstd -q -1 -T4 > "$DEST/$run.tar.zst" ||
        archive_ok=0
done
# later: tar -I zstd -xf "$DEST/$run.tar.zst" -C "/tmp/$USER/extract/"
```

**Evidence.** A run with more than 11,000 files (8.2 GB) took over
20 minutes to copy file by file.

## Delete the local copy only after the copy succeeded

Retry the copy, require every step to succeed, and otherwise keep the
node-local tree and report where it is on both output streams.

**Why.** A failed copy followed by an unconditional `rm -rf` destroys the
only complete copy. Under quota exhaustion either log file can lose appends,
so the message goes to both.

**How to apply.**

```bash
sync_outputs() {
    if [ ! -d "$OUT_LOCAL" ]; then
        echo "sync_outputs: already synced and removed"; return 0
    fi
    local ok=0 attempt
    for attempt in 1 2 3; do
        rsync -a "$OUT_LOCAL"/ "$DEST"/ && { ok=1; break; }
        sleep 3
    done
    if [ "$ok" = 1 ]; then
        rm -rf -- "${OUT_LOCAL:?}"
    else
        echo "sync_outputs: FAILED; KEEPING $OUT_LOCAL on $(hostname)"
        echo "sync_outputs: FAILED; KEEPING $OUT_LOCAL on $(hostname)" >&2
    fi
}
```

Recover a kept tree with a job pinned to that node before its `/tmp` is
cleaned: `sbatch -w <node> --wrap="rsync -a $OUT_LOCAL/ $DEST/"`.

**Evidence.** Two runs lost their final outputs to a copy that failed
without notice, followed by an unconditional delete. One left five zero-byte
`.npy` files, which a scorer counted as a finished run.

## Make the exit handler idempotent

An exit handler armed on `EXIT` and `TERM` runs twice when the job is
signalled. Check at its top whether the work is already done, as the
`[ ! -d "$OUT_LOCAL" ]` test above does.

**Why.** Without the check, the second pass finds the local tree gone, fails
every copy, and prints the failure message after a successful copy. That
false message is identical to the one real data loss produces; it costs a
diagnosis and teaches readers to ignore the message.

## A full quota truncates files without stopping the writer

Monitor the quota on shared storage and prune before it fills. A write past
the quota can leave a zero-byte or partial file while the program continues,
because many writers do not check the errors that `write` and `close`
report.

**Evidence.** Per-iteration snapshots of about 2.7 GB per run reached 311 GB
over about 250 runs and filled the quota, while evaluation read only final
artifacts totalling about 13 GB. Runs' snapshots stopped partway while
training continued, a launcher script was blanked, and a documentation file
was emptied and committed before anyone noticed.

## Prune intermediates only from finished, idle runs

Delete per-iteration snapshots only from runs that have final artifacts and
have not been written recently. Abort if a final artifact reaches the delete
list, and default to a dry run that prints counts and size.

**Why.** A run without a final artifact failed or is still running, and its
intermediates are its only record. A run written in the last hour may still
be writing.

**How to apply.**

```bash
LIST=$(mktemp)
for d in outputs/*/; do
    recent=$(find "$d" -maxdepth 2 -mmin -45 -print -quit) || exit 1
    final=$(find "$d" -maxdepth 2 -name 'final_*' -print -quit) || exit 1
    [ -n "$recent" ] || [ -z "$final" ] && continue
    find "$d" -maxdepth 2 -type f \
         -name '*_[0-9][0-9][0-9][0-9][0-9]_*' >> "$LIST"
done
if grep -q '/final_' "$LIST"; then
    echo "ABORT: a final artifact reached the delete list" >&2; exit 2
fi
wc -l < "$LIST"
xargs -a "$LIST" -r stat -c %s | awk '{s += $1} END {print s / 2^30 " GB"}'
[ "${1:-}" = --apply ] && xargs -a "$LIST" -r -n 200 rm -f --
```

Use `-mmin` or `-mtime`, not a relative `-newermt` string (see
[shell safety](shell-safety.md)). For a series that has no final artifact by
name, keep at least the newest iterate in each directory.

**Evidence.** Dropping intermediates whenever a final artifact existed
removed about 79 GB across about 276,500 files, 90% of all files, from a
durable upload.

## Separate the inventory from the deletion

Build a manifest with a non-destructive inventory script, review it, and let
a separate executor delete only the entries classified `delete`.

**How to apply.** The inventory records the repository revision, each run's
configuration hash and size, and protects a run when:

- a tracked file (documentation, configuration, code) references its path;
- its job ID is on a maintained keep list or is cited in a tracked file;
- the scheduler still lists its job (`squeue -h -u "$USER" -o %A`);
- it is a parent of a protected run: its path appears in a protected run's
  saved configuration as a checkpoint, cache, reference or warm-start
  source. Repeat this step until no new run is protected.

The executor checks again when it runs. The manifest's outputs root must be
this checkout's; each target must be a direct child of the outputs directory
and a directory or symlink; no target may belong to a job that is active
now. It unlinks symlinks rather than deleting their targets, and deletes
nothing without `--execute`.

## Verify the remote copy before removing a local one

Upload with a command that never deletes, verify the uploaded set against the
upload manifest, and quarantine local runs before deleting them.

**Why.** `rclone sync` deletes remote objects that are absent locally, which
removes other machines' results. A zero exit status from a copy does not
show that every object arrived intact.

**How to apply.**

```bash
rclone copy ./outputs/ remote:bucket/outputs/ --files-from keep.txt \
    --transfers=16
rclone check ./outputs/ remote:bucket/outputs/ --files-from keep.txt --one-way
rclone size remote:bucket/outputs/<run>/
rclone lsf --recursive remote:bucket/outputs/<run>/ | sort | sha256sum
```

Then confirm that no active job uses the run and that nothing still needs
local random access to it. Move it into a quarantine directory on the same
filesystem, which is an instant and reversible rename, record the paths and
byte counts, and delete the quarantine only after a later review. Keep the
upload manifest until verification finishes.

## Keep finals, drop superseded intermediates, keep failed runs whole

Store final results, trained weights, predictions, reference images,
configurations, timings and metrics, and the inputs needed to rerun. Drop a
per-iteration snapshot only when a final artifact supersedes it; a run with
no final artifact keeps its full iteration history, because its iterates are
its result. Do not store scheduler logs, code (version control has it),
virtual environments or JIT caches.

The upload list below drops files whose path under the run directory carries
a four-or-more-digit iteration index, and only for runs that have a final
artifact; the run directory's own timestamp is not tested.

```bash
cd outputs
if [ -n "$(find "$run" -name 'final*' -print -quit)" ]; then
    find "$run" -type f -printf '%P\n' |
        awk -v r="$run" '$0 !~ /[0-9]{4,}/ { print r "/" $0 }'
else
    find "$run" -type f -printf '%P\n' | awk -v r="$run" '{ print r "/" $0 }'
fi >> keep.txt
```

## Treat a missing final artifact as a failed run

A scorer must not substitute an earlier snapshot when a run's final output
is missing, and should treat zero-byte files as missing. A fallback turns a
crash into a plausible-looking number.
