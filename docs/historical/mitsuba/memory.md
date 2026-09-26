# Memory and graph growth

Memory problems in a DrJit training loop come in three kinds: an AD graph
that grows with every iteration, device arrays that stay referenced after
they are needed, and allocator caches that look like leaks but are not. This
file covers each, then out-of-memory handling and what to measure before
calling memory flat.

## Graph growth

### `dr.eval` materializes a value but keeps its graph

Any DrJit array that survives from one iteration to the next must be
detached after it is evaluated. Evaluation computes the value; it does not
cut the AD edges back to the arrays that produced it.

**Why.** A recurrence such as `state = f(state, batch)` without a detach
links each iteration's graph to all earlier ones, so graph size, traversal
time and memory grow with the iteration count.

**How to apply.**

```python
dr.eval(state)
state = type(state)(dr.detach(state))
```

This applies to accumulators, histograms, running counts, progressive radii,
exponential moving averages, scale state and any other value carried across
iterations. For moving averages of network weights updated by DrJit
optimizers, detach the optimizer's values before the average reads them.

### Symptoms point to different mechanisms

Read the symptom before choosing a fix:

| Symptom | Likely mechanism |
|---|---|
| compile count grows with iteration | a changing literal or launch width |
| compile count flat, steps slow down | retained graph depth |
| memory jumps at snapshot or averaging cadence | a persistent shadow or artifact graph |
| low GPU utilization with stable kernels | host synchronization or tiny launches |

Measure compile count, kernel hashes, launch widths, memory (all four kinds
below) and per-phase time before assigning a cause.

## Retained references

### Inference through `dr.wrap` retains PyTorch activations until a backward pass

A `dr.wrap` call with AD-enabled inputs records a custom AD operation that
keeps the PyTorch inputs and activations alive so a later `dr.backward` can
reach them. Training consumes these records every iteration; a pass that
never runs a backward, such as a render, keeps every one of them.

**How to apply.** Give inference its own bridge that runs under
`torch.no_grad()` and never marks DrJit inputs as AD leaves. To detect an
inference call without threading a flag through every caller, call
`dr.enable_grad` on the inputs and then test `dr.grad_enabled`: inside
`dr.suspend_grad()` the enable has no effect, so a false result identifies
inference. See [the interop notes](torch-interop.md).

**Evidence.** A render through the training bridge retained about 0.6 GiB per
chunk of one million rays. A 770-chunk render ran out of memory at chunk 78
with 8.71 of 47.52 GiB free, after `gc.collect()`,
`dr.flush_malloc_cache()` and `torch.cuda.empty_cache()`. Training runs never
showed it.

### Device arrays held by Python cannot be flushed

Allocator flushes and garbage collection release only memory that nothing
references. Diagnostic histories that append device tensors every iteration
(gradient statistics, per-step records) hold their memory until the list is
cleared.

**How to apply.** Record diagnostics as host scalars at a logging cadence, or
disable the history in long loops. A flush that "does not help" is evidence
of a live reference, not of an allocator problem.

**Evidence.** A gradient-statistics history kept 24 GPU tensors per
iteration alive in a loop that never read it.

## Allocator caches

### Flush both allocators at a fixed cadence and compare like with like

DrJit and PyTorch each keep a cache of freed device blocks. Returning them
takes three calls:

```python
gc.collect()
dr.flush_malloc_cache()
torch.cuda.empty_cache()
```

Run this between phases (after a snapshot, before a large render), not every
iteration. A periodic flush makes process-level GPU memory a sawtooth: judge
growth by comparing samples taken at the same point of the cleanup cycle. A
cleanup call is not a substitute for showing that persistent state is flat.

**Evidence.** A cleanup every ten steps cost about 0.17 seconds. Process GPU
memory alternated between bands about 1 GiB apart before and after cleanup,
and each band was flat to within 0.03 MiB per step over 1,000 steps.

### A one-time workspace plateau is not a leak

Libraries grow their workspaces when a larger batch or a new code path first
appears. A single step up that then stays level is a plateau; a leak keeps
rising.

**Evidence.** Midway through a 1,024-step run, PyTorch's peak allocation rose
once from 1,022 to 1,187 MiB and process memory by about 0.5 GiB, then stayed
level for the remaining 600 steps while live allocations and DrJit's peak did
not move.

## Out of memory

### Retry a genuine device OOM at a smaller quantized width

A full-film render or large batch that runs out of device memory can retry at
half the chunk width. Retry only a genuine allocation failure, flush the
caches first, never grow the width again within the pass, and stop at a
minimum width.

**How to apply.** Different layers raise different messages, so test the
whole exception chain:

```python
OOM_MARKERS = ("CUDA_ERROR_OUT_OF_MEMORY", "CUDA out of memory",
               "cudaErrorMemoryAllocation", "cuMemAlloc", "cuMemCreate",
               "OutOfMemoryError")

def is_device_oom(error):
    seen, current = set(), error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if any(m in str(current) for m in OOM_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False
```

Halving keeps the width inside a small quantized set (see
[JIT specialization](jit-and-kernel-caches.md)).

### Common causes of out-of-memory errors

In rough order of frequency:

1. persistent arrays that are not detached (above);
2. inference through an AD-recording bridge (above);
3. no cleanup after snapshots or evaluation passes;
4. a batch or chunk width too large for the network workspace;
5. a hash-grid table too large (`log2_hashmap_size`).

## Measuring memory

### Measure four memories, not one

Device memory has several owners, and each needs its own reading:

- DrJit's device peak, `dr.detail.malloc_watermark(dr.detail.AllocType.Device)`;
- PyTorch's live, reserved and peak allocation,
  `torch.cuda.memory_allocated()`, `memory_reserved()`,
  `max_memory_allocated()`;
- the process's GPU memory from
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`;
- host resident memory from the second field of `/proc/self/statm` (times
  the page size), not `ru_maxrss`, which reports the peak.

Query them at a cadence outside the timed region: `nvidia-smi` and allocator
queries cost milliseconds and the `nvidia-smi` call spawns a process.

**Evidence.** Over 1,024 steps of an inverse-rendering loop, DrJit's device
peak stayed at one value (about 1.94 GiB) and PyTorch's live allocation moved
within 1 MiB after warm-up. Host resident memory rose by about 0.04 MiB per
step, consistent with the profiler and optimizer keeping per-step Python
records; the flat device readings separate that host history from a retained
device graph.
