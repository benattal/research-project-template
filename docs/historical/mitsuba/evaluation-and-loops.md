# Evaluation, loops and replay

DrJit builds a graph of pending operations and compiles it into a kernel when
a result is needed. Where that evaluation happens decides how many kernels
run and how much work is repeated, and it also changes rounding (see
[precision](precision.md)). This file covers evaluation placement, scatter
semantics inside traced walks, symbolic and evaluated loops, compaction of
live lanes, frozen functions (`dr.freeze`), differentiated walks, and host
synchronization.

## Evaluation placement

### Evaluate a shared record once

An unevaluated record that several consumers read is recomputed inside every
kernel that consumes it.

**Why.** Evaluating an output compiles its whole dependency cone. Work shared
by two outputs is not reused unless it was evaluated before either, or both
outputs are evaluated in the same call.

**How to apply.** Evaluate a record that has more than one consumer as soon
as it is complete, with all its fields in one call, or evaluate the consumers
together:

```python
walk = trace_camera_walk(scene, sampler, rays)
dr.eval(walk.si, walk.throughput, walk.valid)   # one kernel, reused below
radiance = query_field(walk.si)
residual = target - radiance * walk.throughput
```

Moving an evaluation point changes the bits of the result (fused
multiply-add contraction), so treat it as a numerical change in parity tests.

**Evidence.** A camera walk of 8.4 million lanes feeding a residual was never
evaluated, and each of its consumers (two network queries, the BSDF
evaluation, the emission term, three channel sums) re-traced it: nine OptiX
kernels of 8.4 million lanes per update. Evaluating the walk once made it
one kernel and cut the phase from 0.82 to 0.57 seconds. In another loop,
materializing complete projection records before their consumers reduced a
step from 5.4 to 4.4 seconds.

### Scattering into the same buffer at every depth relaunches the walk

In a walk unrolled in Python, scattering a per-depth record into the same
target at each depth makes every depth re-execute the walk from its first
segment.

**Why.** DrJit evaluates the pending scatters into a target before it
accepts another scatter into that target. The pending scatter at depth $d$
depends on the whole walk up to $d$, so each depth launches a kernel that
traces the walk again: quadratic work in the depth and one OptiX kernel per
depth.

**How to apply.** When each lane records at most once, carry the record as
lazy state with `dr.select` and write it after the loop:

```python
rec_p = dr.zeros(mi.Point3f, n)
rec_w = dr.zeros(mi.Spectrum, n)
recorded = dr.zeros(mi.Bool, n)
for depth in range(max_depth):
    si = scene.ray_intersect(ray, active)
    take = active & is_deposit(si) & ~recorded
    rec_p = dr.select(take, si.p, rec_p)
    rec_w = dr.select(take, beta, rec_w)
    recorded |= take
    ...
```

**Evidence.** Replacing the per-depth scatter with a select chain produced
identical output and cut a walk from 73 to 43 milliseconds per call, from 95
to 74 launches, and from 11 OptiX walks to 4.

### Scatter every field of a slot under the same mask

When several loop iterations may write the same output slot, write every
field of the slot under the same mask. A field written unconditionally is
overwritten by a later iteration while the masked fields keep the earlier
value, and the record becomes inconsistent.

**Evidence.** A validity flag written at every depth overwrote the slot
selected at an earlier depth, so a recorded deposit kept its position and
weight but was marked invalid. Scattering the flag only for an actual deposit
fixed it; slots written once per depth were unaffected.

### Broadcast width-one fields before concatenating records

Concatenating records whose parts have different widths requires every field
to have its part's full width. A scalar field left at width one produces an
array of the wrong length, and later gathers read the wrong lanes.

**How to apply.** Expand before joining (`dr.full(mi.Bool, flag, width)`),
and assert the concatenated widths match.

**Evidence.** Concatenating per-part scalar flags produced 131,073 entries for
a record of 1,179,648 lanes.

### Remove inactive lanes before expanding them

When a query expands each lane into candidate pairs, pass the live mask into
the expansion; do not mask after it. Masking after the expansion sizes the
expansion by all lanes, live or not.

**Evidence.** A sparse neighbour query masked after the expansion rounded up
to a $2^{32}$-lane array, which DrJit rejects. Passing the mask to the
lookup also stops a sampled source from tracing anything for failed
connections.

### Use 64-bit logical indices past $2^{32}$ rays

A `UInt32` ray index wraps at $2^{32}$ rays, which a modest film reaches: a
$512^2$ film at 65,536 samples per pixel has $2^{34}$ rays.

**How to apply.** Choose the index type from the size of the whole request,
not the current chunk offset, so small renders keep their arithmetic and
their random draws. Keep film indices and individual launch widths in
`UInt32`, and pass the chunk offset as an opaque value:

```python
index_t = mi.UInt64 if total_rays > 0xFFFFFFFF else mi.UInt32
ray = dr.arange(index_t, count) + dr.opaque(index_t, start)
pixel = mi.UInt32(ray // index_t(spp))
```

## Loops

### Symbolic loops for rendering; evaluated loops for walks differentiated in reverse

A symbolic loop compiles the whole walk into one kernel; an evaluated loop
launches one kernel per iteration. Rendering and sample generation should use
symbolic loops. In DrJit 1.3.1, walks that were differentiated in reverse
mode through their vertices ran in evaluated mode: reverse-mode AD through
the bounded symbolic loop was not available to them, and a forward-mode pass
through a symbolic loop could not be repeated for a second tangent.

```python
@dr.syntax
def walk(ray, active, ...):
    while dr.hint(active, mode="evaluated"):   # differentiated walk
        ...
```

DrJit documents `max_iterations` as the hint that lets reverse-mode AD
allocate per-iteration state for a loop; test it on the installed version
before converting a differentiated walk.

### Inside a symbolic loop, scatter into preallocated buffers

Values inside a symbolic loop are symbolic and cannot be used as concrete
indices for a gather ("variable is symbolic" errors). Allocate the output
buffers before the loop as concrete arrays, `dr.scatter` into them inside the
loop (scatters accept symbolic indices), and gather after the loop ends. A
flat layout `depth * n + lane` keeps the buffers one-dimensional.

### A fixed-depth symbolic loop can match an unrolled loop bit for bit

Converting a Python-unrolled walk to a symbolic loop preserves its results
exactly when the symbolic loop runs every iteration, including iterations in
which all lanes are inactive, so that every sampler advance happens as in the
unrolled version.

**Evidence.** A continuation walk converted this way matched the unrolled
version on every interaction field, weight, mask and the next sampler variate
across eight production batches. Its cost fell from 31.5 to 4.3 milliseconds,
a whole update from 1.92 to 1.81 seconds, and the kernel count by 14. A
symbolic version of a bounded visibility walk cut a gradient phase from 0.80
to 0.58 seconds.

## Lane compaction

### Compact live lanes to a quantized width before expensive work

When most lanes of a wavefront are dead (camera rays that miss the scene,
terminated walk slots), gather the live lanes into a power-of-two batch, run
the expensive work on it, and gather the results back.

**Why.** Dead lanes still cost a full network evaluation or trace. The
compaction's live count is the one host read; padding to a power of two keeps
the set of launch widths small.

**How to apply.** Pad by repeating the last live lane, keep the count opaque,
and gather results back so AD edges survive:

```python
index = dr.compress(active)                  # live lanes
count = dr.width(index)                      # host read
width = 1 << (count - 1).bit_length()
last = dr.opaque(mi.UInt32, count - 1)
source = dr.gather(mi.UInt32, index,
                   dr.minimum(dr.arange(mi.UInt32, width), last))
out = expensive(dr.gather(mi.Float, x, source))
slot = dr.zeros(mi.UInt32, dr.width(active))
dr.scatter(slot, dr.arange(mi.UInt32, count), index)
result = dr.select(active, dr.gather(mi.Float, out, slot), 0.0)
```

The padded copies are never read back. Below some width the host read costs
as much as the work it saves; for a hash-grid network that threshold was
about $2^{19}$ lanes, where the network runs in about a millisecond.

**Evidence.** With 93% of lanes dead, a network update fell from 147 to 78
milliseconds. Merging two 262,144-lane queries into one crossed the
compaction threshold and made them slower (106 against 100 milliseconds per
batch), because the merged query paid the host read.

### A fixed-capacity compaction needs a proven bound

Compacting into a fixed capacity avoids the host read only if the number of
live lanes can never exceed the capacity. That bound has to come from the
structure of the computation (for example, at most one deposit per path),
not from observed occupancy. Lanes past the capacity are dropped, and a
device overflow flag raised afterwards cannot recover them.

**Evidence.** Live lanes in a 1.18-million-lane record numbered 8,750 to
9,177 in every observed batch. A 131,072-lane capacity would have cut the
network backward pass from 68.7 to 17.4 milliseconds with matching gradients,
but no structural bound existed, so it was not adopted.

## Frozen functions

### Frozen functions remove Python and tracing overhead

`dr.freeze` records the kernels a function launches on its first call and
replays them on later calls with compatible inputs, without re-running the
Python body.

**Evidence.** Freezing the geometry part of a gradient estimator cut a step
from 8.1 to 3.4 seconds (the gradient phase from 7.0 to 2.0 seconds) with
identical hits, texture coordinates and Jacobian rows. The first call, which
records, took 23 seconds.

### Pass everything that changes as an explicit DrJit input

A frozen function replays against its declared inputs. Anything it reads from
elsewhere is baked into the recording or triggers an error.

**How to apply.**

- Pass the seed as an opaque `UInt32` argument, call `sampler.seed()` inside
  the body, and advance the Python counter outside it.
- Pass scene-dependent values (bounding spheres, emitter positions) as
  arguments, not closure variables.
- Python objects are compared by `hash()` or `id()`. A mutable object whose
  identity stays fixed does not trigger re-recording when its contents
  change, so its state must reach the function as DrJit arrays.
- DrJit reports "variable was created before recording was started, but it
  was not specified as an input" when the body reads a DrJit variable that is
  not a traversable member of an input (for example, a member missing from
  `DRJIT_STRUCT`).
- `dr.width()` inside the body is a literal in the recording.
- Set `warn_after` low (1 or 2) and `limit` small, so an accidental
  re-recording is reported instead of silently multiplying recordings.

```python
@dr.freeze(warn_after=1, limit=2)
def trace(seed, center, radius):
    sampler.seed(seed, n)
    record = walk(scene, sampler, center, radius)
    dr.eval(record)
    return record

record = trace(dr.opaque(mi.UInt32, next_seed()), center, radius)
```

### Evaluate a frozen function's outputs inside it

End the body with `dr.eval` on its outputs, so the recording has an explicit
boundary and cannot absorb later lazy work (a backward pass, an optimizer
step) into the replayed kernels. Evaluate lazily computed inputs before the
call for the same reason.

### Synchronize before reusing a recording

A replay reuses the intermediate buffers captured at recording time. Call
`dr.sync_thread()` before replaying again while a previous replay's outputs
may still be read by pending work, or the second replay overwrites buffers
the first one's consumers are reading.

### Keep the network outside the recording

Record the traced geometry and export what the network needs (hit points,
Jacobian rows, weights), then query the network after the replay and contract
its outputs and input derivatives with the exported rows. Network parameters
then change freely between replays, and the PyTorch and tiny-cuda-nn kernels,
which a DrJit recording does not capture, stay outside it.

### Feed the raw and replayed paths the same evaluated inputs

When one code path fuses a sampled input into its trace kernel and another
reads it evaluated, the two round differently in the last bit, and through
refractive geometry a last-bit difference changes which surface a ray hits.

**Evidence.** An area-light sample fused into the trace kernel moved about
one light hit in seven, through a glass box, relative to the replayed walk
that read the evaluated sample. Evaluating the source samples once and giving
both paths the same arrays made every hit identical.

### A replay that retains a graph can be slower than no replay

**Evidence.** A frozen replay of a camera-side differential matched the raw
computation exactly in isolation but made the whole loop 16% slower: its
explicit per-depth record and retained AD graph slowed the network bridge
that followed. Time replay in the full loop, not in isolation.

## Differentiated walks

### Make AD leaves full width for per-lane derivatives

A width-one AD leaf broadcast to $n$ lanes accumulates the sum of all lanes'
derivatives into one value. For a per-lane derivative, make the leaf full
width before enabling gradients:

```python
lanes = dr.zeros(mi.Float, n)
origin = mi.Point3f(*(lanes + dr.opaque(mi.Float, float(v)) for v in p))
dr.enable_grad(origin)
```

### Differentiate the walk that was sampled

Trace the walk once with its source attached (`dr.enable_grad` on the
per-lane origin) and run the derivative through that walk's own vertices.
Do not trace a second time to obtain an attached copy, and do not replay a
walk from a reused seed: the second trace can choose different lobes at
stochastic interfaces and pair derivatives with the wrong samples. Reseeding
to reproduce a draw belongs only in controls such as common-random-number
finite differences. A geometric surrogate for a path's motion (for example,
the chord of a refracted prefix) is not a substitute.

**Evidence.** A chord surrogate for a refracted prefix's motion was 11% off
on the tangential slide it stood for.

## Host synchronization

### Keep host reads off the steady path

Every host read waits for all queued GPU work and splits the kernel stream.
In a steady loop, avoid `.numpy()`, `np.asarray`, `.item()`, `int()` or
`float()` of a device value, printing device values, image conversion, file
writes, GPU memory queries, and data-dependent widths (`dr.width` of a
`dr.compress` result). Reduce diagnostics on the device and read one scalar
at the logging interval. Treat every remaining read as a named, justified
exception.

**Evidence.** Porting per-step NumPy reductions, source proposals and
host-staged network queries to the device cut a step from 6.9 to 5.4 seconds
with an identical trajectory.

### Exchange arrays with PyTorch on the device

Move arrays between DrJit and PyTorch through DLPack (`dr.wrap`, `.torch()`,
or the `__dlpack__` protocol), which shares device memory. Converting through
NumPy moves the data to the host and back.
