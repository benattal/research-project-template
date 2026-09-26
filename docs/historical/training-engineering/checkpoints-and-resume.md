# Checkpoints and resume

Long runs here were chains of short scheduler jobs, each resuming the last
checkpoint. A resumed run is correct only when it reproduces the run that
would have continued without interruption: the same weights and optimizer
state, and also the same data order and noise draws from that step on. The
first section covers what a checkpoint holds and how it is written and found;
the second covers the random streams. The drivers that launch the links are
described in
[long runs and resumption](../clusters/long-runs-and-resumption.md).

## Writing and finding checkpoints

### Save every piece of state that evolves

A checkpoint holds the weights, every EMA shadow, the state of every
optimizer (auxiliary ones included), the global step and epoch, the run's
configuration, and any state fitted during training: coefficients of a
periodically refit auxiliary model and the buffer it refits from, the
position of a streaming data source, and the record of which samples a
data-driven initialization used.

**Why.** Whatever the checkpoint omits is rebuilt from its initial value at
every resume. The run then regresses at each link boundary in a way that
looks like noise in the loss curve.

**Evidence.** Fitted kernel coefficients were not checkpointed, so the first
epoch of every resumed link ran with the fit from the start of training. A
data-driven initialization that resampled its subset on resume would have
changed the initialization mid-run; the fix restores the saved record and
never resamples.

### Make approximate resume an explicit choice

When some state is too large to checkpoint, such as banks of reference
images, refill it on resume only when an explicit configuration flag
allows an approximate resume, and keep the rest of the state exact.

**Why.** A silent refill makes the resumed run differ from the continuous
one while every log line looks the same. With a flag, the configuration
records that the trajectory is approximate.

### Write checkpoints atomically in the destination directory

Write to a temporary file in the same directory, then rename it over the
final name. A job killed mid-write then leaves the previous checkpoint
intact rather than a truncated file that automatic resume would pick up.

```python
def atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)            # atomic within one filesystem
    finally:
        if tmp.exists():
            tmp.unlink()
```

The temporary file must be in the destination directory: `os.replace` across
filesystems is a copy, not an atomic rename.

### Resume from the newest readable checkpoint; be strict with explicit paths

With `resume_from="auto"`, try checkpoints from newest to oldest, skip any
that fail to load or lack required keys, and print why each was skipped.
With an explicit path, fail on any error. Start from scratch only when no
checkpoint exists.

```python
for path in sorted(ckpt_dir.glob("model_epoch*.pt"), reverse=True):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        missing = [k for k in REQUIRED_KEYS if k not in ckpt]
        if missing:
            raise ValueError(f"missing keys {missing}")
        return path, ckpt
    except Exception as exc:
        print(f"resume=auto: skipping {path}: {type(exc).__name__}: {exc}")
return None, None
```

Zero-pad the epoch in file names (`model_epoch%04d.pt`) so that lexical order
is chronological.

### Prune old checkpoints, and copy milestones out first

Keep the newest few checkpoints and delete older ones after each save, on
the main process only. Checkpoints needed later, for example at the epochs
where a metric is computed, must be copied elsewhere before pruning can
reach them.

**Evidence.** Image-model checkpoints were about 2 GB each, too large to keep
for every epoch of a long chained run. Pruning kept the newest three, and a
separate watcher copied target-epoch checkpoints into a milestones directory
for evaluation before pruning could delete them.

### Keep the run's configuration file from its first start

Write the configuration file when a run starts fresh, and leave it untouched
on resume: it records what the run was launched with. Evaluation tools should
rebuild the model from the configuration embedded in the checkpoint rather
than from current defaults.

### Give each fresh run its own output directory

Automatic resume looks for checkpoints in the output directory. A new
configuration launched into a directory that holds an older run's
checkpoints restores that run's weights, optimizer, EMA and epoch, and the
"fresh" test continues the old trajectory under new hyperparameters. Put a
job ID or a configuration name in every default output directory, and check
that no two registered configurations share one.

**Evidence.** Two model variants were registered with the same output
directory, so resuming one would have loaded the other architecture's
checkpoint.

### Checkpoint at least once per link

The checkpoint interval must not exceed the number of epochs a link
completes. Otherwise a link ends without writing a checkpoint, and the next
link starts from the beginning.

**Evidence.** Links that ran 8 epochs with a checkpoint every 10 epochs
restarted from epoch 1 every time.

### Stop at an epoch boundary before the time limit

Before each epoch, estimate its end from the previous epoch's duration and
exit cleanly if it would pass the time budget. The previous epoch is already
checkpointed, and no epoch is lost to a kill signal.

```python
max_runtime = float(os.environ["MAX_RUNTIME_SEC"])
start, last = time.monotonic(), None
for epoch in range(start_epoch, num_epochs + 1):
    if last is not None:
        stop = time.monotonic() - start + 1.15 * last > max_runtime
        if any_process(stop):            # collective decision
            accelerator.end_training()
            return
    t0 = time.monotonic()
    train_one_epoch(epoch)
    save_checkpoint_if_due(epoch)
    last = time.monotonic() - t0
```

The 15% margin covers the checkpoint and sample writes after the epoch. How
to derive the budget from the job's allocation is in
[long runs and resumption](../clusters/long-runs-and-resumption.md).

## Reproducible random streams

### Reseed every generator from the seed and the epoch at each epoch start

At the start of every epoch, before the data loader is iterated and before
any noise is drawn, seed Python, NumPy and torch from a function of the base
seed and the epoch. The data order, generation noise, dropout masks and any
sampled training parameters are then a function of the epoch alone, and a
resume at epoch $k$ reproduces epoch $k$ of the continuous run.

**Why.** A checkpoint holds parameters, not generator states. If a process
seeds once at start, every link begins its first epoch from the same
generator state, so the data order depends on the position within the link
rather than on the epoch. The error has the same sign at every link and
accumulates.

**How to apply.** Seed as in
[multi-GPU](multi-gpu.md#seed-the-host-stream-identically-and-the-device-stream-per-process).
Saving and restoring every generator state (`torch.get_rng_state`,
`torch.cuda.get_rng_state_all`, `np.random.get_state`, `random.getstate`,
and the sampler's state) also works, and allows resuming mid-epoch;
per-epoch reseeding requires resuming at epoch boundaries.

**Evidence.** A run chained through short jobs showed a training loss that
rose steadily from link to link while every link's weights, optimizer state
and EMA restored correctly.

### Key per-sample randomness by epoch and index

Inside a dataset, draw each sample's noise from a generator seeded by the
epoch and the index, so noise changes between epochs, is reproducible after
resume, and does not depend on which worker loads the sample. Seed anything
meant to stay fixed across epochs, such as a training subset or a frozen
noise realization, from a fixed seed, never from the epoch.

```python
seq = np.random.SeedSequence([STREAM_NOISE, epoch, index])
rng = np.random.default_rng(seq)
```

Give each stream its own identifier. Seeds built as arithmetic expressions
collide: a partition stream seeded with `epoch * size + index` equals a fixed
stream seeded with `index` at epoch 0, which made the two statistically
dependent during the first epoch.

**Evidence.** The fix for that collision changed the fixed noise
realization, so a live run using it could not be resumed across the fix
without its training data changing mid-run.

### Seed before constructing the model

Set the seeds before building the model, so that runs with the same seed
share their initial weights.

**Evidence.** A sweep that seeded only at epoch boundaries gave same-seed
arms different initial weights, and the paired comparisons built on them had
to be rerun.
