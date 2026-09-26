# CUDA kernel performance for online cache training

The native backend of the demo traces training paths with its own CUDA
kernels, runs the hash-grid encoding and MLP through tiny-cuda-nn, and applies
its own Adam and weight-average (EMA) step. One update traces 65,536 light
paths and 65,536 gather paths (four times that on one scene), fits the field,
and steps the optimizer. In the final binaries an update cost about .9 to
2.2 ms on an H100, depending on the scene, batch and precision. These notes
record what the profiles showed, which changes paid, and which did not.

Every speedup below was exact (bitwise equal records, films or MLP values)
unless it says otherwise; [exactness gates](exactness-gates.md) describes how
that was checked.

## Profile before optimizing; the bottleneck moves

**Insight.** Each round of optimization moved the dominant cost to a
different kernel, and none of them was where intuition put it.

**Evidence.** With the fp32 CUTLASS MLP, split-K weight-gradient GEMMs took
56% of kernel time and transport only 18%. Switching to the fp16 fully fused
MLP made a 10,000-update fit 1.94× faster (12.6 to 6.5 s), after which
transport took 53% and the dense Adam/EMA pass 20%. After the transport fixes
below, the hash-grid backward and the MLP weight gradients led again.

**How to apply.** Profile with `nsys` for the kernel mix and Nsight Compute for
the dominant kernel, on the real workload, after every adopted change. Report
profiled timings separately from clean throughput, since instrumentation
changes both.

## Classify each workload as fit-bound or trace-bound

**Insight.** With sample generation and fitting on separate streams, measure
each stream's busy time before choosing what to optimize.

**Mechanism.** When the GPU is never idle, the update time is set by the
busier stream. Speeding up the other one gains nothing.

**Evidence.** Per update (ms): a textured room scene 2.86 wall, 2.02 fit busy,
1.40 trace busy; a glass-box scene 2.35 / 1.91 / 1.03; a glass-caustic scene
1.18 / .84 / 1.16; a water-caustic scene 1.55 / 1.14 / 1.50. The first two
are fit-bound, the last two trace-bound. The sum of busy time was within 2% of
wall time, so there were no host gaps to remove.

## The path-tracing kernel is bound by divergence, not arithmetic

**Insight.** A wavefront that traces random paths through glass keeps only a
few lanes of each warp active, so reducing arithmetic barely helps.

**Evidence.** Nsight Compute showed 3 to 6 active threads per 32-lane warp,
achieved occupancy of 19 to 25% against a register-limited 37.5%, and SMs
idle for 22 to 37% of the kernel in its tail (1,024 blocks against 792
resident slots, 1.29 waves). Divisions were about a third of the
instructions, yet hoisting them gained 1 to 2%.

**How to apply.** Look for work to overlap or delete rather than
instructions to save. The large exact wins in this kernel came from
concurrency (below), from removing contention and from early exits.

## Spread invalid lanes instead of parking them at one coordinate

**Insight.** Lanes with no sample must not all query the hash grid at the same
point, even though their gradient is zero.

**Mechanism.** Missed photons were packed at the grid centre with zero loss
gradient. The hash-grid backward still issues an atomic add per corner, and
tens of thousands of lanes adding zero to the same few entries serialize on
those addresses. Hashing each invalid lane's position by its lane index
spreads them across the table; adding zero elsewhere changes no bit.

**Evidence.** About 60,000 of 65,536 photons per update missed on one scene.
Grid backward went from 1.18 to .16 ms per update there (from 48% of update
time), and training throughput rose 1.5 to 1.8× on three scenes, with fits of
the same quality (a 30,000-update fit from 78 to 50 s).

## End occlusion rays at the first hit

**Insight.** A shadow ray only needs to know whether anything is hit, so its
traversal can stop at the first accepted triangle.

**Mechanism.** The any-hit walk makes the same tests as the closest-hit walk
up to the first hit, so the boolean it returns is identical.

**Evidence.** 5 to 6% more updates per second on two scenes, with a
byte-identical 1,024-sample film.

## Launch bounds help the trace kernel a little and must be gated per scene

**Insight.** Capping registers on a divergent kernel can raise occupancy, but
the gain is small and depends on the scene.

**Evidence.** `__launch_bounds__(128, 6)` took the trace kernel from 112
registers and a 256-byte stack to 80 registers and 384 bytes, and raised
updates per second 5 to 11% on every scene. Tightening to `(128, 7)` (72
registers) saved 6% of kernel time on one scene and 0 to 1% on the others;
`(128, 8)` was 12% faster on one scene and 9% slower on another. Bounding the
camera kernel made it slower.

**How to apply.** Try bounds per kernel, adopt one only if it is exact and
nowhere slower, and record it in the build provenance (an interrupted edit
once left a bound in a delivered binary).

## Overlap the next update's trace with the current fit

**Insight.** When the training samples of update `i` depend only on the scene
and the update index, trace update `i + 1` on a second stream while update `i`
fits.

**Mechanism.** The trace never reads the field, so it does not have to wait
for the previous optimizer step. It fills the SMs that the fit leaves idle and
vice versa. Double-buffer the records, order each trace after the last reader
of its buffer with events, make each fit wait for its own trace, and make the
side stream wait for any earlier work on the main stream (such as a geometry
deformation or a render that uses the record buffer as scratch).

**Evidence.** Records were byte-identical for any split of N updates into
calls. Updates were 14 to 15% faster on three scenes and 7% faster on the
fit-bound one; a water scene's live display went from 29.5 to 34 FPS.

**How to apply.** Expose a batched host call (`train_steps(n)`) so the
runtime can pipeline across updates, and test that `train_steps(3)` followed
by `train_steps(5)` gives the same records as eight single steps.

## A second trace in flight helps only trace-bound workloads

**Insight.** Two training traces in flight (three record buffers, two trace
streams) fill each other's tails.

**Evidence.** The two trace-bound scenes gained 10 to 12% per update
(1.04 to .92 ms and 1.44 to 1.30 ms). The fit-bound scenes gained 3% and 0%.

## Stream priorities did not pay

**Insight.** Giving the fit streams higher priority than the trace helped
fit-bound scenes by 2 to 3% and cost the others 1 to 7%; it was dropped.

## Keep host waits off the critical path

**Insight.** A host synchronization inside an update inflates the kernels
around it, even when the waited-on work is short.

**Mechanism.** The host waited for a lane-selection count before issuing each
fit. In situ, that selection took 186 µs against 64 µs standalone, and it sat
between the trace and the fit.

**How to apply.** Have the device write counts to pinned host memory (CUB's
select writes its count directly there), run the selection on its own stream,
and size later launches on the device (indirect launches or kernels that read
the count).

## Run inference only where the result is read, and skip zero atomics

**Insight.** Compact the lanes whose self-training value is actually read,
and skip gradient atomics that would add ±0.

**Mechanism.** In a padded batch, rows without a record still ran the
inference and still added zeros into the grid. A grid entry starts at +0 and
never becomes −0, so skipping zero addends leaves every entry bitwise equal.
The compaction keeps lane order, so the values are bitwise equal too.

**Evidence.** Only 143,683 of 262,144 lanes needed the value on one scene and
322 of 65,536 on another. Together the two changes cut update time 5 to 18%
(one scene's grid backward 642 to 406 µs). Putting the selection on the trace
stream first regressed a trace-bound scene by 2.6%; moving it to a third
stream made every scene faster.

## Use a float2 atomic per corner for an fp32 grid with two features

**Insight.** On sm_90, one `atomicAdd(float2*)` per hash-grid corner replaces
two scalar atomics with the same addends.

**Mechanism.** tiny-cuda-nn's fp32 backward adds each of the two features per
level with its own `atomicAdd`. The order of atomics was already unspecified,
so only the atomic order changes. fp16 grids already use `half2` atomics.

**Evidence.** Grid backward .69 to .38 ms per update, about 9% of an fp32
update, with fits inside the run-to-run spread.

## A dense optimizer makes hash-table size a per-update cost

**Insight.** When Adam and the EMA stream every parameter each update, sweep
the table size down before paying for it.

**Mechanism.** A dense fp32 Adam/EMA pass reads and writes the master weights,
both moments and the EMA for every table entry, touched or not.

**Evidence.** On one fp32 field (16 levels, 15,000 updates), 2^21 entries per
level cost 3.30 ms per update, 2^19 cost 2.74 and 2^18 cost 2.62. The
low-band error stayed inside the spread of five 2^21 fits down to 2^18; 2^17
was the first to leave it. 2^19 was adopted (−17% per update, render time
unchanged).

## Remove optimizer passes the gradient contract does not need

**Insight.** Check what the optimizer actually needs before keeping copy and
clear passes.

**Evidence.** In fp32, Adam could read the master and EMA planes directly,
with no cast copies, and every fit overwrote its gradients, so the zeroing
pass was dead. Removing both took Adam/EMA from .76 to .64 ms (+4% updates
per second).

## The library's fp32 MLP path is slow; a fused kernel can match it bit for bit

**Insight.** tiny-cuda-nn's fp32 `CutlassMLP` runs each layer as a separate
SIMT GEMM through global memory at about 5% of peak; a fused kernel that keeps
each output's accumulation order is bitwise equal and much faster.

**Mechanism.** A small MLP fits in shared memory. One query per thread, all
layer weights in shared memory and activations in registers, each output
accumulated from zero in input order with `fmaf`, reproduces the SIMT GEMM's
arithmetic.

**Evidence.** The fused inference kernel rendered 32-sample frames 1.7×
faster (93 to 55 ms) with a maximum difference of 0 and byte-identical films.
A fused training MLP reproduced the split-K weight gradients bitwise by
following CUTLASS's slicing (4,096-query slices of whole 8-query tiles, the
last slice taking the remainder, slices summed in order, then the epilogue).

**How to apply.** Keep the library path as a permanent parity reference. Test
batches whose size is not a multiple of the split-K slice, since an equal
slice split hides a wrong boundary (for example 12,544 and 131,328 queries).
The tile's k-depth sets the slice boundaries, so it cannot change.

## A weight-gradient kernel must hide its load latency

**Insight.** An exact fused weight-gradient kernel was only 12% faster than
the library until it overlapped its global loads.

**Mechanism.** The first version stalled on serialized global loads, one round
trip per staged element (1.16 ms). The second staged 32-query chunks with the
next chunk's loads in flight, gave each warp a 32×16 tile at 4×4 per lane in
one launch across all layers, and read `float4` weight rows in the backward.

**Evidence.** Weight partials 1.16 to .29 ms; the fused training path reached
+37% over the library.

## Tile the MLP so shared-memory broadcasts do not bound it

**Insight.** A kernel that issues one shared-memory load per FMA and uses 200
registers is bound by the loads; a register tile fixes both.

**Evidence.** 64 queries × 64 outputs per 128-thread block, each thread owning
4 queries × 8 outputs, with the per-output `fmaf` chain unchanged: training
forward 231 to 123 µs, inference 145 to 85 µs, backward 347 to 100 µs at width
48 and 131,072 queries, bitwise equal. One scene's update went 2.04 to
1.79 ms and its 16-sample render 26.7 to 23.8 ms.

## When a reduction chain cannot split, shrink the per-lane tile

**Insight.** Occupancy for a reduction that must keep its order comes from a
smaller per-lane register tile, not from splitting the chain.

**Evidence.** Per-lane tiles of 4×4, 4×2, 2×4 and 2×2 took 285, 260, 196 and
235 µs (2×2 was bound by redundant global loads). 2×4 with `float2`/`float4`
staging reads was adopted: 1.72 to 1.65 ms per update.

## Claim tiles dynamically when another kernel holds SMs

**Insight.** Blocks that take work from an atomic counter finish together even
when a concurrent kernel delays some of them.

**Mechanism.** With a fixed share of tiles per block, a block whose SM is
held by the concurrent trace starts late and becomes the tail. With claiming,
early blocks take its tiles. The last block out resets the counter. Results
stay bitwise equal because each tile's arithmetic is unchanged.

**Evidence.** In situ, the MLP forwards went 318 to 282 µs and the backward
135 to 110 µs; the update 1.79 to 1.72 ms.

## Benchmark kernels in situ, not standalone

**Insight.** Kernels running beside a concurrent trace slow down by very
different factors, so standalone wins can vanish.

**Evidence.** A split-K GEMM variant with smaller tiles and deeper pipelines
was bitwise equal and 35 to 40% faster standalone, but gained only 1 to 3% in
the real update: 1-warp 32×32 blocks slowed about 5× beside the trace, against
about 2.5× for tiny-cuda-nn's 4-warp 64×64 blocks. Running both GEMMs serially
on one side stream also put them on the critical path; one side stream per
GEMM fixed that. The training forwards took 318 µs in situ against 150 µs
alone.

## Look for kernels far slower than their bytes justify

**Insight.** A kernel that takes many times its bandwidth cost often has
scattered or scalar stores, and fixing them is exact.

**Evidence.** A library identity encoding wrote label rows with the feature
index fastest, as scattered 2-byte stores (139 µs); the loss kernel stored 16
scalars per query (91 µs). Query-fastest writes and two 16-byte vector stores
in the same arithmetic order gained 1.4 to 5.8% per update.

## Hoist a reciprocal exactly by reproducing the division's fast path

**Insight.** A per-ray reciprocal can replace per-box divisions bitwise if each
bound repeats the compiler's own division fast path.

**Mechanism.** CUDA's IEEE division is a refined reciprocal followed by a
short FMA correction sequence. Refining the reciprocal once per ray and
repeating the correction per bound gives the same quotient; outside a safe
exponent range (here |x| in [2^-40, 2^40]) fall back to a real division.
Reading the SASS confirms the instruction sequence matches.

**Evidence.** 53.8M of 67M quotients hoisted with zero mismatches on 1M rays
per scene; 1 to 2% more updates per second, as the divergence profile
predicted.

## Rejected ideas and their measured reasons

- **Near-child-first BVH traversal.** Faster on isolated random rays (4.1 to
  3.1 ms) but no training gain, and near-ties between box entry and triangle
  distance changed 32 of 1M hits. With a canonical tie-break it became slower
  everywhere: loading both children and a second stack cost more than the
  pruning saved.
- **Persistent warps** (resident warps drawing lanes from a counter). 2%
  faster, but wrapping the walk in a loop changed FMA contraction and 3 to 36%
  of records.
- **Path regeneration** (restart a finished lane on the next index). Exact,
  but mixed (−12% to +7%) and it hung after about 1,000 updates on one scene.
- **Separate photon and gather kernels.** 16 to 31% slower on every scene:
  each kernel alone still needed 97 to 109 registers, against 112 for the
  combined kernel.
- **Cheaper fields to buy more updates.** Fewer levels or a narrower head
  bought 1.1 to 1.25× more updates and no quality.
- **Overlapping host work in the live loop.** The frame was 37.7 ms, 37.2 ms
  of it GPU compute, with encoding already on another thread.

## OptiX on a GPU without RT cores is software traversal

**Insight.** An H100 has no RT cores, so OptiX there is a compute
implementation and its timings say nothing about RTX hardware.

**Evidence.** OptiX traversal fitted one scene's 10,000 updates in 5.25 s, but
a strict replay against the software traversal still found differing path
flags, and it ran serially rather than overlapped with the fit, so it was not
adopted. Its
uncached startup took 4.8 s. Disable its disk cache on shared machines, where
concurrent writers collide.

## Deform geometry and refit the BVH on the GPU

**Insight.** For a surface that deforms at fixed topology, a GPU refit with a
precomputed bottom-up order costs almost nothing.

**How to apply.** Export the refit levels with the scene, keep triangle
identities, evaluate the deformation on the device, refit bounds level by
level, and clear film accumulation (not the optimizer, learning rate or EMA)
on each pose change.

**Evidence.** .09 ms per update over 120 poses, against about 60 ms per frame
for a CPU deform, refit and upload path, with positions and bounds within
2.4e-7 of a fresh build.

## Half precision: loss scale, overflow and quality

**Insight.** fp16 training of a hash grid needs a gradient scale, a guard
against silent overflow, and a comparison against fp32 before trusting it.

**Mechanism.** tiny-cuda-nn accumulates fp16 grid gradients with `half2`
atomics. With per-sample weights near 1/65,536, gradients fall into the
subnormal range and vanish at scale 1. A fixed scale large enough to prevent
that can overflow fp16's 65,504 limit in the backward pass, after the output
gradients (which look finite) and before the optimizer. The overflow
followed parameter growth late in the fit (the grid maximum rose from 11 to 29
between 10,000 and 30,000 updates): a logistic loss keeps pushing regions with
no positive samples toward −∞, and Adam's normalization turns their vanishing
gradients into full-size steps. Capping the input light power did not help;
flooring the logit did.

**How to apply.**

- Start from tiny-cuda-nn's default loss scale of 128 rather than 1, and treat
  the scale as a per-scene stamped setting.
- Record on the device the first update whose unscaled parameter gradient is
  non-finite (one `atomicCAS` flag, no throughput cost) and report it with the
  scale.
- Floor the logit of a logistic loss (−15 worked) so the loss is flat below
  it; this let fp16 finish 30,000 updates and changed nothing in fp32.
- Do not rely on skip-on-overflow: once the forward pass overflows, nearly
  every update is skipped (9,237 of the last 10,934 at scale 128) and the
  cache silently freezes. Count skips; skipping should still advance the
  schedule clock.
- Search for quality in fp32 and recover fp16 speed afterwards.

**Evidence.** On a glass-box scene, scale 1 gave .094 error with colour bands;
scales 16 to 1,024 gave .048. The same scale 128 turned 64% of another
scene's parameters into NaN. A larger field (2^21, 4,096 finest resolution,
30,000 updates) scored .0587 in fp16 and .0465 in fp32 with no non-finite
flag. On easier scenes fp16 tied fp32 at 2.3 to 3.2× the update rate.

## Build hygiene for a tiny-cuda-nn application

**Insight.** Several failures came from the build, not the kernels.

- tiny-cuda-nn's default Adam skips zero-gradient parameters, adds an L2
  penalty on matrices and places epsilon before bias correction. Write an
  explicit dense Adam with a global step and the EMA in one fused kernel, and
  check it against a float64 oracle.
- The library's public backward wrapper references an RTC symbol, so build it
  with RTC support even with runtime JIT disabled.
- Include the library's vendored JSON header everywhere; a newer system
  header changed a factory signature and broke linking.
- Pass pointer-bearing scene structs to kernels and device helpers by const
  reference. A large aggregate passed by value caused a local-address fault
  that `__grid_constant__` alone did not fix; `compute-sanitizer` with source
  lines and PTX inspection found it.
- Set the CUDA architecture explicitly (`-DTCNN_CUDA_ARCHITECTURES=90`) so
  configuring does not probe whatever GPU the build host has, and set every
  experimental CMake option explicitly, since a cached option leaked into
  later builds.
- Hash the sources before and after compiling and reject the build if they
  changed; run experiments from staged, hashed binaries with a patch of the
  source over the last commit beside each one.
- Use 64-bit sample indices: a 512² film at 65,536 samples per pixel needs
  2^34 indices.

## Keep a regression bench across binaries

**Insight.** Time every staged binary on identical scene packages whenever the
page gets slower, before suspecting the latest change.

**How to apply.** Run each binary for 2,000 fresh updates, time updates 500 to
2,000, and render one frozen 16-sample frame. Interleave two passes; they
should agree within .5%. Leave unexplained slips above that on record rather
than attributing them.

**Evidence.** When recordings slowed, the bench showed every release as fast
or faster than its predecessor; the slowdowns came from recipe budgets (a
64-update warm-up per frame, 64 camera samples, an fp32 field with a 2^21
table).
