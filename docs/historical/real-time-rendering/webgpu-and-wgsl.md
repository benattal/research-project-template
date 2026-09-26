# WebGPU and WGSL

The browser version of the demo trains a hash-grid encoding and MLP and path
traces the scene entirely in WGSL compute shaders. WebGPU hides a GPU behind
a validation layer and a shader compiler that differ from CUDA in ways that
cost time: errors surface far from their cause, float atomics do not exist,
and the compiler rewrites arithmetic. These notes collect those lessons and
the kernel designs that brought a full training update from about 23 ms to
about 3.3 ms on an H100, roughly twice the native CUDA time.

## Request the device limits the workload needs

**Insight.** WebGPU's default limits are far below what a desktop GPU offers,
and a device only gets more if it asks.

**Mechanism.** `requestDevice()` without `requiredLimits` gives the spec
defaults: 8 storage buffers per shader stage and a 128 MiB storage binding.
A transport kernel that binds scene, material, light and record buffers needs
more buffers, and activation buffers for half a million training lanes need
larger bindings.

**How to apply.** Read `adapter.limits`, fail early if the adapter cannot meet
the workload, and request what is needed:

```ts
const device = await adapter.requestDevice({ requiredLimits: {
  maxStorageBuffersPerShaderStage: 14,
  maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  maxBufferSize: adapter.limits.maxBufferSize,
} });
```

## `writeBuffer` lands at submit, so every dispatch sees the last write

**Insight.** Writing a uniform buffer between dispatches that share one
submit does not give each dispatch its own value.

**Mechanism.** `writeBuffer` calls are queued and all of them land before any
command in the following submit runs. Every dispatch reads the final value.
The symptoms were misleading: an MLP whose layers all read the output layer's
offsets trained to near-zero predictions, and a tiled cache inference filled
only the last tile, so the rest of the image fell back to a path-traced
estimate whose noise looked like a cache that reset on every camera move.

**How to apply.** Give each dispatch that needs a different value its own
uniform buffer and bind group, written once at construction, or put all the
per-dispatch values in one buffer and select each dispatch's entry with a
dynamic bind-group offset or an index derived from the workgroup ID.

## An invalid command buffer drops the whole submit, pending writes included

**Insight.** When readbacks return all zeros, including buffers that were only
uploaded, suspect a command buffer that failed validation.

**Mechanism.** `dispatchWorkgroups` is limited to 65,535 workgroups per
dimension. A one-dimensional Adam dispatch over 17M parameters asked for
about 260,000. The encoder's `finish()` produced an invalid command buffer,
and per the spec the entire `submit()` was dropped, including the queued
`writeBuffer` that initialized the parameters.

**How to apply.** Spill large one-dimensional dispatches into the Y
dimension and recover the flat index in the shader
(`gid.y * 65535 * WG + gid.x`). Wrap pipeline and bind-group creation in
`device.pushErrorScope('validation')` and listen for `uncapturederror` and
`device.lost`, so the first error is reported where it happens.

## Validate WGSL on the CPU before any GPU job

**Insight.** A shader compile error in the browser shows up only as an opaque
"Invalid CommandBuffer" at submit; validate every assembled shader module on
the CPU first.

**Mechanism.** The browser compiles WGSL lazily and reports module errors
through `getCompilationInfo()`, which most code never reads. A reserved word
used as an identifier (`meta`) or a type mismatch (a stale `u32` random state
passed to a function whose signature had changed) therefore costs a full GPU
job to find.

**How to apply.** Install the `naga` command-line validator (`cargo install
naga-cli`; version 30 was used) and run it on each module as the app
assembles it. WGSL has no `#include`, so an app that concatenates a common
file with each entry shader must be validated on the concatenation:

```bash
cat common.wgsl lighting.wgsl transport.wgsl > /tmp/transport.wgsl
naga /tmp/transport.wgsl
```

Script one such line per module and run it before every browser job. In the
page, log `getCompilationInfo()` messages for every module. Shaders that tests
splice together from strings are not covered by a file-based check.

## Rebuild every bind group that references a recreated buffer

**Insight.** Resizing a buffer invalidates every bind group that points at the
old one, not only the ones the resize code remembers.

**Mechanism.** A bind group holds the buffer object, not a name. After a
window resize recreated the per-pixel record buffer, one forgotten bind group
still referenced the destroyed buffer and the next submit failed with "used in
submit while destroyed".

**How to apply.** Build every bind group that uses a resizable buffer in one
function and call it from the resize path.

## WGSL has no float atomic add; sum in fixed point

**Insight.** Accumulating float gradients with a compare-and-swap loop is slow
and fragile; 64-bit fixed-point integer sums are faster and deterministic.

**Mechanism.** The float emulation reads the old bits, adds, and retries
`atomicCompareExchangeWeak` until no other thread intervened. Each attempt
waits for the round trip to memory, so throughput collapses when the
gradient table does not fit in L2 and when many lanes hit the same entries
(the dense coarse hash levels). In one browser/driver combination the loop
also silently produced zeros.

**How to apply.** For each hash level, find the batch's largest
$|\partial L/\partial y|$ from its float bits, scale so the worst-case sum
fits below $2^{62}$ (trilinear weights sum to one, so the bound holds per
parameter), and add each scaled value as two 32-bit `atomicAdd` calls: the low
word, then the carry into the high word. Round each sum to float once, in the
optimizer kernel, which also clears it. A level whose bound is non-finite
writes NaN to its parameters so the failure stays visible. For loss sums, give
each thread its own slot and reduce afterwards instead of using atomics.

**Evidence.** Measured compare-and-swap throughput was about 16 G adds/s when
the gradient table fit in L2, 4.3 G/s for a 31 MB table and 2.6 G/s for
49 MB, against about 80 G/s for plain loads of the same entries. Fixed point
took one update from 16.2 to 5.8 ms (grid kernels 13.6 to 1.6 ms) on the
largest field, with fit quality unchanged across three seeds and run-to-run
spread exactly zero. Applying it only to the contended dense levels was as
fast but left a 1e-11 spread; applying it to every level made updates
bitwise reproducible.

## Register-tile the MLP without changing any output's arithmetic

**Insight.** Tiling the MLP kernels so each thread computes several queries
and rows can be bitwise identical to the untiled kernel, which makes it a
pure speedup.

**Mechanism.** Each output is still one fused multiply-add chain over the same
inputs in the same order; the tile only changes which thread loads what. The
weights are broadcast from workgroup memory and reused across queries.

**How to apply.** For the forward and backward passes, one 256-thread
workgroup covered 128 queries × 64 rows, each thread holding 4 queries × 8
rows in `vec4` accumulators and reading one broadcast weight vector per step
of the reduction. For weight gradients, compute split-K partials in 32×32
blocks with 2×2 outputs per thread, run all matrices in one dispatch, and
prefetch the next round of rows into registers while computing the current
one. Keep the reduction index ascending from zero and the split-K slice
boundaries fixed.

**Evidence.** Per-update times on three scenes went 5.68 / 11.46 / 4.96 ms
(untiled) to 4.49 / 9.47 / 3.62 ms, with byte-identical films. Larger 64×64
partial blocks were slower (6.24 ms against 5.47 on one scene) because one
block per split-K slice left too few workgroups to fill the GPU. 4×4 outputs
per thread in 64-thread workgroups were exact but no faster: lost occupancy
offset the saved shared-memory loads.

## Register tiles scatter stores; route them through the staging tile

**Insight.** After register tiling, write intermediate rows through workgroup
memory so that consecutive threads store consecutive addresses.

**Mechanism.** In the register tile a warp's 32 lanes hold 32 different
queries, so storing one hidden row touches 32 memory sectors per
instruction. One forward pass issued 67M such scattered stores and one
backward pass 75M.

**How to apply.** Store each hidden row when the next layer stages it in
workgroup memory, with consecutive threads taking consecutive rows of one
query. Apply the activation derivative and store the deltas the same way.

**Evidence.** On the largest-MLP scene the forward went 2.27 to .91 ms and
the backward 2.21 to 1.15 ms; the whole update went 7.28 to 6.14 ms, bitwise
unchanged.

## Fuse the MLP passes and specialize them with pipeline constants

**Insight.** One dispatch for the whole forward and one for the whole backward
removes per-layer launches and keeps activations in registers.

**Mechanism.** A layer's outputs stay in the registers that formed them and
reach the next matrix through the staging tile. The training forward still
stores hidden rows for the backward; inference does not need them.

**How to apply.** Specialize the inference variant with a pipeline-overridable
constant (`override KEEP_HIDDEN_ROWS: bool`) rather than a runtime branch.
Run the backward from the output matrix down with each matrix's input deltas
carried in registers. Put all matrices in one table read by every pass instead
of per-layer uniforms.

**Evidence.** The fused forward took one scene's forward from 3.42 to
1.99 ms; the fused backward saved another .1 to .3 ms per update. Both were
bitwise unchanged.

## Compact the lanes that need a value and dispatch indirectly

**Insight.** Run inference only on the lanes whose result is read, and let the
GPU size the dispatch.

**Mechanism.** A self-training read is needed on only part of the batch. A
full-batch forward wastes the rest. The compaction order is
nondeterministic, but a query's value does not depend on its slot, so the
result is bitwise identical to a full-batch pass.

**How to apply.** Each workgroup reserves output slots with one atomic, writes
its active lanes densely and records each lane's slot. A small kernel writes
the dispatch size into an indirect buffer and the forward runs with
`dispatchWorkgroupsIndirect`. The consumer reads each value back by slot.

## Fold clears and resolves into the kernels that consume the data

**Insight.** A pass that only clears or converts a buffer costs a full sweep
of memory and is usually free to merge into its consumer.

**How to apply.** Let the optimizer kernel round and clear each fixed-point
gradient sum, and let the kernel that consumes per-level bounds clear them
before the next producer. This removed a separate sweep over 12M grid
parameters per update.

## Parallelize serial reductions inside one thread

**Insight.** A kernel in which each thread reduces over the whole batch for one
weight is latency-bound; a workgroup reduction per weight fixes it.

**Evidence.** On a small early field, 64-lane workgroup reductions for each
weight and bias gradient took those kernels from 3.46 to .25 ms per update
and a 10,000-update fit from 44.5 to 12.7 s. The summation order changed, so
it was gated statistically rather than bitwise (see
[exactness gates](exactness-gates.md)).

## WGSL compilers contract multiply-adds and approximate division

**Insight.** To reproduce another backend's float results, write the fused
multiply-adds explicitly and use IEEE division.

**Mechanism.** The shader compiler may contract `a*b + c` in `dot` and `cross`
into fused multiply-adds, and may lower `1/x` to an approximate reciprocal.
Either changes results in the last bit, which matters when two surfaces are a
single ulp apart (see [web and native parity](web-and-native-parity.md)).

**How to apply.** Write `fma()` calls in the pattern the reference compiler
produces and compute quotients with true division. Where a reciprocal per ray
is needed for speed, refine it and correct each bound once. Vectorizing the
ray/box test (compare all axes at once, test the final bounds once) gave back
part of the cost with byte-identical results.

## Detect NaN from the bits

**Insight.** In WGSL, `x != x` is not a reliable NaN test.

**Mechanism.** The compiler may assume floats are finite and fold the
comparison away.

**How to apply.** Test the exponent bits:
`(bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u`. Record the first update
whose gradient was non-finite and raise it at every CPU fence (idle wait,
readback, save), so a NaN is reported with the update that caused it instead
of surfacing later as a black film. Test the detector by forcing an infinite
weight on one lane.

## Evaluate numerically fragile activations with care in float32

**Insight.** `log(1 + exp(x))` loses precision in float32 for moderately
negative inputs.

**Evidence.** At a scaled input of −8 the gradient was off by 2e-4. One
shared activation function with a small-argument `log1p` series and a stable
derivative for large negative inputs fixed it everywhere it was used.

## `select()` evaluates both operands

**Insight.** Do not draw random numbers inside the arguments of `select()`.

**Mechanism.** WGSL's `select(f, t, cond)` is a function, so both arguments
are evaluated. A random draw in either one advances the generator on every
lane, and the stream no longer matches a backend that branches.

## Readback alignment and ordering

**Insight.** Map one whole staging buffer rather than each tensor, and read
counters before starting an asynchronous readback.

**Mechanism.** `mapAsync` offsets must be multiples of 8, so per-tensor maps
of odd-sized tensors failed. Updates queued after a readback starts can change
counters that the saved state must match.

## Pack integer and float data through one `ArrayBuffer`

**Insight.** Build mixed integer and float GPU structures (BVH nodes) with
overlapping `Float32Array` and `Uint32Array` views of one `ArrayBuffer`, not by
bit-casting integers through a `Float32Array`.

**Mechanism.** Writing integer bit patterns as floats can flush denormal
patterns to zero on some platforms, corrupting child indices.

**How to apply.** Pack each node as two `vec4`s through two views of the same
buffer. A breadth-first layout with siblings adjacent needs only one child
index per node.

## Measure kernels in the pipeline, not in isolation

**Insight.** Dispatching one idempotent kernel twice and measuring the added
time gives its in-pipeline cost without instrumentation overhead.

**Mechanism.** Timestamp queries need each dispatch in its own compute pass,
which changes scheduling: an instrumented update took 27.1 ms against 22.7 ms
uninstrumented. The doubling method runs the real pipeline and agrees across
repeats to .05 ms.

**How to apply.** Use per-pass timestamps to find where time goes (per-level
timing was what showed one dense hash level taking 10.9 of 12.5 ms), and
kernel doubling or uninstrumented totals to decide whether a change helped.
Time throughput with a queue barrier that does no readback, after warm-up
updates, and keep it separate from single-frame latency and from presentation.
Check that instrumented runs produce the same numbers as ordinary ones.

## Keep owned buffers flat and account for them

**Insight.** Track the bytes of every buffer the renderer allocates and gate on
them staying constant.

**How to apply.** Allocate through one helper that records each buffer, and
check the total across 1,000 updates and 20 resize cycles. This measures the
application's buffers, not the driver's allocation peaks.
