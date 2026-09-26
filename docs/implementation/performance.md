# Randomness and performance

Performance is part of the method's contract. A faster implementation is not
acceptable when it correlates samples that should be independent, changes a
fit or a gradient, recompiles per iteration, transfers hot-loop data through
the host, or leaks memory.

## Random domains

One base seed deterministically derives named domains
(`src/core/utils/randomness.py`) for, for example:

- training data and minibatches;
- model initialization;
- evaluation and references;
- diagnostics and snapshots;
- finite-difference or replay replicates.

Repeating a complete run with the same base seed reproduces every domain. A
domain is identified by its name, not by the order in which domains were
created, so adding a diagnostic stream never shifts a training stream.
Consecutive calls that promise fresh samples advance their stream; they do not
restart at the same seed.

Reseeding updates sampler or device state without branching Python control
flow or changing batch shape. An explicit replay seed is used only when the
caller is intentionally reproducing a recording.

Identical random draws do not guarantee identical results: a parallel
floating-point reduction can change its summation order between launches.
Check reductions and categorical sampling tables before blaming a model for a
repeated-run mismatch.

## Stable compiled structure

JIT compilers (`torch.compile`, Triton, JAX, DrJit and similar) specialize
kernels on expression structure, literal values, and array shapes. Keep all
three stable in the hot loop:

- pass changing scalars (learning rates, iteration counters, seeds) as device
  values rather than Python literals baked into the kernel;
- keep batch widths in a small fixed set and record where the width is
  chosen;
- keep per-job compilation caches separate (`scripts/lib/job_cache.sh`);
  concurrent writers can corrupt a shared cache.

TODO(template): add the project framework's specific rules.

## Persistent state and graph growth

Arrays that persist across iterations must not retain their autograd graph.
Evaluate and detach them at the end of each iteration. The symptom of a
retained graph is iterations that get progressively slower and eventually run
out of memory; see [troubleshooting](troubleshooting.md).

## Device residency

Keep training data on device across framework boundaries. Avoid NumPy
conversions, `.item()`-style scalar reads, and printing device values inside
the hot loop: each forces a synchronization. Accumulate logged quantities on
device and read them at the logging interval.

## Host threads

A process inherits thread-pool sizes from the node, not from its allocation.
Bind OpenMP, BLAS, and framework intra-op pools to the allocated CPU count
(the launchers set `OMP_NUM_THREADS` from `SLURM_CPUS_PER_TASK`), and keep the
user's total thread count under the limit in
[environments](../environments/index.md).

## Verification checklist

A performance change is accepted only with:

1. matched-seed numerical parity of the fits, estimates, and trajectories it
   touches;
2. a stable compiled-kernel count across iterations;
3. flat memory over a long run;
4. a matched-hardware wall-clock comparison that states warmup and whether
   compilation and artifact writing are included.
