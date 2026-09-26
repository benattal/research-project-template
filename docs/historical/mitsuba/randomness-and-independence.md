# Randomness and independence

A Monte Carlo program is correct only when the samples it treats as
independent are independent, and it can be debugged only when a run repeats
itself from its seed. Both properties are easy to lose without any visible
error: a correlated "fresh" batch, a diagnostic that shifts a training stream
or a table that rounds differently on each construction all produce
plausible numbers. This file covers how to derive every random stream from
one base seed, where an estimator needs independent draws and what sharing
them costs, how to reseed Mitsuba samplers without recompiling, when replay
with a fixed seed is legitimate, and what breaks bitwise reproduction even
when every stream repeats.

## Streams and domains

### Derive named domains from one base seed

Give every stochastic role its own named random domain, and derive each
domain's seeds from the base seed and the domain name alone. Typical domains
are training batches, model initialization, reference samples, the
independent paths of a two-sample estimator, diagnostics and snapshots, and
finite-difference controls.

**Why.** When roles share one counter, or when a domain's seeds depend on
the order in which domains were created, adding a stream or a diagnostic
shifts every stream after it, so an unrelated change alters the trained
result. Offsetting one shared counter by the base seed has a second defect:
base seeds 0 and 1 then produce the same sequence of per-call seeds, shifted
by one call, so two "replicates" draw almost the same samples.

**How to apply.** Hash the base seed and the name to a starting point, and
advance a per-domain counter from there:

```python
import hashlib

class SeedStreams:
    """Named random domains derived from one base seed.

    A domain's seeds depend only on the base seed, the domain name and an
    index, never on which other domains exist or when they were created.
    """

    def __init__(self, base_seed):
        self.base_seed = int(base_seed)
        self.counters = {}                    # checkpoint these

    def start(self, name):
        key = f"{self.base_seed}/{name}".encode()
        digest = hashlib.blake2b(key, digest_size=4).digest()
        return int.from_bytes(digest, "little")

    def at(self, name, index):
        """Seed for an explicit index, such as the global iteration."""
        return (self.start(name) + int(index)) & 0xFFFFFFFF

    def next(self, name):
        """Seed for the next fresh batch of a domain; advances it."""
        k = self.counters.get(name, 0)
        self.counters[name] = k + 1
        return self.at(name, k)
```

Use a domain either through `next` (a counter) or through `at` with an
explicit index, not both. Within a domain, seeds do not repeat for $2^{32}$
calls. Two domains overlap only if their hashed ranges intersect, which for
ranges of a million seeds each has a probability near $5 \times 10^{-4}$.

Do not derive domains with Python's built-in `hash()` of a string: string
hashing is salted per process (`PYTHONHASHSEED`), so the same name maps to a
different seed in every run. For the same reason, do not assign seeds or
order work by iterating over a `set` of strings. `hashlib` is stable across
processes and machines.

Mitsuba's own integrators derive per-role seeds from one seed. `mi.render`
derives the differentiation pass's seed from the primal seed with
`mi.sample_tea_32`, and the projective integrators combine the seed with
fixed constants (`0xaa00aa00 ^ seed`) for their separate sample sets.

### Scope reseeds by domain so diagnostics cannot move training streams

Diagnostics, snapshots, evaluation renders and profiling passes draw from
their own domain. Changing how often they run, or removing them, must leave
every training batch unchanged.

**Why.** A snapshot that reseeds a sampler from the training counter shifts
every later training batch by one call. The trained result then depends on
the logging cadence, and a run with diagnostics cannot be compared with one
without them.

**How to apply.** Route every reseed through the active domain and switch
domains with a scope that restores the caller's domain even on failure:

```python
from contextlib import contextmanager

class Streams(SeedStreams):
    domain = "train"

    @contextmanager
    def sampling_domain(self, name):
        previous, self.domain = self.domain, name
        try:
            yield
        finally:
            self.domain = previous

    def reseed(self, sampler, width):
        seed = self.next(self.domain)
        sampler.seed(dr.opaque(mi.UInt32, seed), width)

with streams.sampling_domain("diagnostic"):
    render_snapshot(sampler)          # advances "diagnostic" only
```

A diagnostic must also leave the training computation's structure alone: an
added `dr.eval` or host read inside the training step changes kernel fusion
and rounding (see [precision](precision.md)). A control that deliberately
replays training draws saves and restores the training counter around
itself, and is labelled as a control.

Store every domain counter in the checkpoint, and refuse to resume from a
checkpoint that lacks them. A resumed run then continues each stream where
the interrupted run stopped.

### Give each role its own sampler

Two roles that draw from one sampler in sequence are coupled through their
draw counts: a lane's variates are positional, so the second role reads
whichever dimensions the first left unused.

**Why.** A change in how many dimensions the first role consumes (a Russian
roulette decision, an extra read, a longer prefix) shifts every later draw of
the second role on that lane. Two implementations of the first role then
change the second role's samples, and a comparison of the second role's
output can no longer be bitwise.

**How to apply.** Build one sampler per role, each reseeded from its own
domain. When roles must share a sampler, fix the number of dimensions each
consumes per lane, drawing and discarding unused variates on inactive lanes.

**Evidence.** Two implementations of a walk's continuation consumed
different numbers of variates from a sampler shared with a later supervision
walk that neither changed. The supervision's fitted weights could then be
compared only statistically; they differed by a relative RMS of 3e-5.

### Advance the stream on every call that promises fresh samples

A function that returns "a new batch" advances its domain on every call; it
never reseeds with the value it used last time.

**Why.** A call that restarts at the same seed returns the same batch.
Two identical batches counted as independent halve the estimated variance
of their mean while its actual error stays that of one batch, a fitted field
sees one noise pattern over and over, and an optimizer steps repeatedly in
the direction of that pattern's error. Nothing fails, so the bias is silent.

The common forms of this bug:

- a helper that builds a sampler and seeds it with a default of 0 on each
  call;
- `mi.render(scene, params)` in a loop: its `seed` defaults to 0, so every
  iteration renders the same noise. Its documentation states that different
  seeds are required for statistically independent images;
- two roles seeded with the same value at different wavefront widths.
  Mitsuba's PCG32-based samplers seed lane `i` from a hash of the seed and
  `i` alone, independent of the width, so the two roles draw identical
  variates on every lane they share.

**How to apply.** Take every seed from `streams.next(domain)` or
`streams.at(domain, index)`. A unit test that calls a sampling function twice
and asserts that the first variates differ catches the first two forms.

**Evidence.** On Mitsuba 3.8.0, two `independent` samplers seeded with 7 at
widths 4 and 8 returned the same first four variates, while seed 8 at width
4 returned unrelated ones.

### Change the Monte Carlo seed at every optimization step

An optimizer driven by gradients estimated with a fixed seed converges to the
minimizer of one noise realization.

**Why.** A fixed seed repeats the same estimation error at every step, so an
outlier in that realization pushes the parameters in the same direction each
time instead of averaging out.

**How to apply.** Key the gradient seed by the global iteration,
`streams.at("gradient", global_iteration)`, not by the index within a resumed
segment, so a run resumed from a checkpoint draws the same samples as an
uninterrupted run. A step that needs several independent sample sets takes
them from separate domains (`"gradient/residual"`, `"gradient/deposit"`), not
from arithmetic offsets such as `base + 100003 * it + 977 * b` and
`base + 100003 * it + 55511 * b`, which coincide for some combinations of
iteration and batch indices.

## Independence an estimator needs

### Estimate the factors of a product from independent draws

When two Monte Carlo estimates are multiplied, draw them independently. For
estimates $\hat A$ and $\hat B$ of $A$ and $B$,

$$
\mathbb{E}[\hat A \hat B] = \mathbb{E}[\hat A]\,\mathbb{E}[\hat B]
  + \mathrm{Cov}(\hat A, \hat B),
$$

so shared draws add the covariance to the product's expectation. The bias
does not shrink when either estimate is averaged over more samples of the
same shared draws.

Products of this kind appear throughout differentiable rendering:

- **A loss gradient.** The gradient of $\tfrac{1}{2}(I - T)^2$ is the
  residual $I - T$ times the image derivative $\partial I$. The two come
  from two launches with independent seeds. `mi.render` does this: its
  differentiation pass uses a seed derived from the primal seed, and it
  raises an error when both are given the same value.
- **A residual and the path that carries it.** When a per-pixel residual is
  transported into the scene along a sampled path (to supervise a field, for
  example), the residual and the path's throughput come from independent
  draws. Sharing them correlates the residual with the throughput, so the
  expected update is nonzero even where the expected residual is zero. Share
  only the discrete index that must match, such as the pixel; resample the
  sub-pixel jitter and every scattering decision for each role.
- **A numerator and its normalizer.** A ratio of two estimates, or an
  estimate divided by a normalizing sum, takes the two from independent
  draws. This removes their covariance; the next entry covers the bias that
  remains.
- **A fit and the reference that scores it.** A reference rendered from the
  same stream as the training samples shares their noise, so the fit looks
  closer to it than to the truth. Render references from their own domain,
  freeze one reference per metric, and never score a run against a
  reference it generated itself.
- **A reference and its response.** A finite difference whose target
  estimate appears both in the residual and in the perturbed response
  contains the target's variance divided by the step.

**Evidence.** A finite-difference reference that reused one noisy target
estimate in its residual and its response contained a term
$-\mathrm{Var}(\text{target noise}) / \Delta$. Estimating that term from two
independent draws and removing it changed the sign of one gradient component
(from -2.1e-6 to +1.07e-6) and another component by a factor of five. A
reference field built with the residual evaluated on the same path whose
endpoint received it included the path-residual covariance as an extra
field. Training draws the two independently, so that reference did not
represent the target training converges to.

### Independence removes covariance, not nonlinear bias

Drawing a denominator independently of its numerator removes their
covariance but leaves the bias of the nonlinearity:
$\mathbb{E}[1 / \hat X] > 1 / \mathbb{E}[\hat X]$ for a noisy positive
$\hat X$, by Jensen's inequality.

**Consequence.** A loss whose normalizer contains a noisy estimate, such as
a relative error divided by $\max(\hat I, T)^2$, has a biased gradient even
when the normalizer is sampled separately. Detaching the normalizer changes
which quantities the derivative flows through, not the statistics. More
samples reduce the bias; they do not remove it at any finite count.

**How to apply.** Build normalizers from deterministic quantities (the
target image, a fixed exposure) or keep the estimator linear in each noisy
estimate.

### Terms that enter additively may share draws

Sharing draws between two estimates is harmless when they are added:
$\mathbb{E}[f(X) + g(X)] = \mathbb{E}[f(X)] + \mathbb{E}[g(X)]$ whether or
not the two terms see the same $X$. Sharing then changes the variance, not
the expectation.

**Examples.**

- The two terms of a pointwise objective evaluated at one sampled location
  (a data term and a reference term at the same point) reach the same
  stationary point with shared or independent locations.
- In an objective that compares expectations under two distributions, each
  expectation is normalized by its own sample count and its own proposal
  density; the two batches can have different sizes.

Sharing never improves an estimate that is supposed to be independent per
query. When two queries fall in the same pixel, give each its own draw of
that pixel's value: a shared draw correlates the two estimates without
making either more accurate.

### Check index coverage with the caller's actual indices

Samples chosen by a deterministic stride (pixel `(offset + k * stride) % n`)
can alias with the batch schedule and visit only a subset of indices. Test
coverage with the index sequence the caller produces, not with the selector
in isolation, and randomize the permutation or shift per batch from a
domain.

## Reseeding Mitsuba samplers

### Pass seeds to samplers as opaque device values

A seed passed to `sampler.seed()` as a Python integer becomes a literal in
every kernel that reads the sampler, so each new seed compiles a new set of
kernels.

**Why.** DrJit inlines Python scalars into the kernel source as constants and
identifies kernels by a hash of that source. In Mitsuba 3.8, `Sampler.seed`
takes a `UInt32` array, so a Python integer becomes a literal. A seed that
changes every iteration therefore changes the hash of every downstream
kernel.

**How to apply.** Advance the seed in Python, convert it once, and pass it as
an opaque `UInt32`:

```python
seed = streams.next("train")                  # Python int below 2**32
sampler.seed(dr.opaque(mi.UInt32, seed), wavefront_size)
```

`mi.render` passes its `seed` argument to the same call, so pass an opaque
`mi.UInt32` there too, and leave `seed_grad` at its default, which Mitsuba
derives from `seed` on the device.

**Evidence.** Reseeding with a literal seed compiled 43 fresh kernels per
iteration with the GPU near idle; the opaque seed compiled none after
warm-up, and the same seed value produced the same samples.

### Build each sampler once and only reseed it

Construct samplers at setup and reseed them each iteration; do not call
`mi.load_dict({"type": "independent"})` inside the loop. Together with the
next two entries, the per-iteration pattern is:

```python
sampler = mi.load_dict({"type": "independent"})    # once, at setup
sampler.set_sample_count(spp)                       # once
width = pixel_count * spp                           # fixed for this role

for it in range(first_iteration, last_iteration):
    seed = streams.at("gradient", it)               # global iteration
    sampler.seed(dr.opaque(mi.UInt32, seed), width)
    loss = step(scene, sampler)
```

**Evidence.** A renderer that built a new sampler object every iteration
showed a per-iteration OptiX recompilation stall; building each sampler once
and reseeding it removed the stall with the same samples per seed.

### Keep the width and the sample count fixed across reseeds

A reseed changes device memory and nothing else. It must not change the
wavefront width, the sample count or any Python control flow.

**Why.** `sampler.seed(seed, wavefront_size)` must be called with the width
of the wavefront that draws from it, and that width is a literal in the
kernels. Mitsuba's integrators compute the width as film pixels times samples
per pixel, so a changed `spp` (or `sampler.set_sample_count()`) compiles a
new set of kernels. A branch on the seed's value in Python does the same.

**How to apply.** Set the sample count once at construction, keep each role
at one width from a small fixed set, and record widths at the sites that
choose them (see [JIT specialization](jit-and-kernel-caches.md)). A pass
that needs another sample count uses its own sampler at its own fixed width.

### Wrap seed arithmetic to 32 bits at one boundary

Mitsuba samplers take `UInt32` seeds, and seed arithmetic done on the host
exceeds that range deep into a long run.

**Why.** Python integers are unbounded, so the overflow appears only when
the value reaches the device, often hours in.

**How to apply.** Keep the Python counters unbounded and reduce modulo
$2^{32}$ in the single function that produces seeds (`SeedStreams.at`
above). The device stream then has a declared $2^{32}$ seed period, and
nothing else in the program needs to know about it.

**Evidence.** A base seed multiplied by 1,000,003 exceeded `UInt32` as soon
as a second base seed was used, and `base + 104_729 * k` failed at sample
40,832 of a 65,536-sample reference render.

### Seed parallel generators through a hash, not a linear sequence

When a program builds its own per-lane generators, scramble the seed and the
lane (or key) with a hash before seeding.

**Why.** DrJit's documentation for `PCG32` states that streams obtained by
varying only the sequence increment are not independent enough across
streams, and recommends seeding both the state and the increment through a
hash such as the Tiny Encryption Algorithm. Mitsuba's PCG32-based samplers
do this: each lane's state and increment come from
`sample_tea_32(seed, lane)`, where `seed` includes the sampler's own `seed`
property as an offset.

**How to apply.** For draws that must depend on a key rather than on call
order (a pixel and a realization index, so a query can be re-evaluated and
reproduce its draw), hash the key into the generator's state:

```python
# Per-lane draws determined by (key, role); key is a UInt32 array.
v0, v1 = mi.sample_tea_32(key, mi.UInt32(role))
rng = mi.PCG32(size=dr.width(key), initstate=mi.UInt64(v0),
               initseq=mi.UInt64(v1))
u = rng.next_float32()
```

The `PCG32` constructor adds the lane index to both arguments, so two lanes
with the same key draw different variates. For unkeyed draws,
`dr.rng(seed)` returns a counter-based Philox4x32 generator that makes its
seed and counter opaque and advances the counter on every call.

### Evaluate or reseed generator state carried across iterations

A `PCG32` whose state is advanced every iteration and never evaluated builds
an ever larger expression for that state, and kernel compilation grows with
it. Call `dr.eval(rng)` after drawing, or reseed the generator each iteration
as the samplers above are reseeded. `dr.rng()` evaluates its state eagerly.

## Replay and common random numbers

### Fixed seeds and replay are controls, never production

Reusing a seed to reproduce draws, common random numbers, is a measurement
device. It belongs in finite-difference references and parity probes, not in
the estimator that drives optimization.

**Why.** A central difference taken with the same seed at $\theta + h$ and
$\theta - h$ cancels the sampling noise shared by the two evaluations and
measures the derivative of one realization. The difference includes changes
of discrete choices (a lobe or branch that switches between the two
evaluations), which makes it a reference for a derivative estimator that
must account for them. In
production, a fixed seed biases the optimizer (above), and replaying a walk
to obtain its derivative pairs derivatives with the wrong samples wherever
the second trace makes a different choice (see
[differentiated walks](evaluation-and-loops.md)).

**How to apply.** Take control seeds from their own domain so a control
never advances a training stream:

```python
seed = dr.opaque(mi.UInt32, streams.at("fd-control", replicate))
plus = render(theta + h, seed)
minus = render(theta - h, seed)
derivative = (plus - minus) / (2 * h)
```

Average over several control replicates, each with a new seed, and report
their spread. For keyed draws, a realization index selects the epoch: the
same realization reproduces the draws for a secant, and a new realization
draws everything afresh.

**Evidence.** Against a common-random-number central difference (8 batches
of 2 million lanes), a derivative estimator that dropped the rate of a
sampled weight had ratios of 1.019, 1.094 and 1.302 at facet tilts of 22, 35
and 54 degrees; with the rate kept, the ratios were 1.001, 0.995 and 0.991.

### Inspect draws with a second, identically seeded sampler

To examine the variates a production sampler produced, seed a separate
sampler with the same seed and width, or take `sampler.clone()` before the
production draws, and read from that copy.

**Why.** Reading the production sampler's values adds an evaluation boundary
to the production graph, which changes kernel fusion and can change bits
(see [precision](precision.md)). A copy exposes the same draws without
touching the production state.

## Reproducing a run

### A complete run with the same base seed repeats its streams

Two runs with the same base seed, configuration and code draw the same
samples in every domain. That requires:

- every seed derived from the base seed through named domains;
- domain counters saved in checkpoints and restored on resume, with
  optimization seeds keyed by the global iteration;
- no seed or work order that depends on process-salted hashing or on the
  order of a `set`;
- no diagnostic that draws from a training domain.

Repeating the streams does not make the arithmetic repeat. Three device
behaviors still change bits between runs or between code versions: parallel
prefix scans and atomic accumulations (below), and fused multiply-add
contraction, which changes when an evaluation boundary moves (see
[precision](precision.md)). The first two differ between two runs of the same
code; the third differs between two versions of the code that compute the
same mathematics.

### `mi.DiscreteDistribution` tables can differ between constructions

Two `mi.DiscreteDistribution` objects built from identical weights can hold
CDFs that differ in the last bits, so a fixed seed stops reproducing its
samples.

**Why.** The CDF is built on the device by a single-precision parallel prefix
scan. Its partial sums are associated according to block scheduling, and
floating-point addition is not associative, so tables larger than one scan
block can round differently each time. Two samplers are affected:

- `sample()` picks a different index when a variate falls within rounding
  distance of a bin edge;
- `sample_reuse()` returns `(value - cdf) / pmf`, so a one-ulp change in the
  CDF moves the reused coordinate of every later bin by `ulp / pmf`, about
  half a percent of a bin at $10^5$ bins, and rounding can push it slightly
  outside $[0, 1]$.

A mesh's own area distribution uses the same scan. `Mesh.surface_area()`
and `Mesh.sample_position()` read it, and Mitsuba rebuilds it whenever
`vertex_positions` changes, so every pose update of a moving mesh
reconstructs the table.

**How to apply.** Build the running sum once per table on the host in double
precision from the device's single-precision weights, upload it, and sample
with a binary search over half-open intervals:

```python
weight = np.asarray(weight_f32, dtype=np.float64)   # one host read
cdf = np.cumsum(weight)
cdf /= cdf[-1]
cdf_dev = mi.Float64(cdf)                           # same in every process

# First bin whose CDF exceeds u. A zero-weight bin has an empty interval
# and is never selected; the reused variate stays inside [0, 1].
index = dr.binary_search(
    0, len(cdf) - 1,
    lambda i: dr.gather(mi.Float64, cdf_dev, i) <= mi.Float64(u))
```

A device double-precision scan normalized by its own endpoint also removed
the variation on the tables tested, but double precision does not make a
summation associative in general. Validate it on the weight ranges you use.

**Evidence.** A 1,720-entry table always repeated; 30,000- and 38,284-entry
tables never did, with up to 9,700 differing entries. On an edge-length CDF,
1,971 entries changed (largest difference 4.9e-4) and 252 of every million
draws selected a different edge. Terms that depend on area-sampled points
varied by up to a fifth per batch at a fixed model state, and by 1 to 3%
between runs. With host-built tables, three repeats were identical on every
lane, and totals in separate processes agreed to 1e-7.

### Atomic scatter-adds accumulate in launch order

A floating-point `dr.scatter_add` or `dr.scatter_reduce(dr.ReduceOp.Add, ...)`
adds its contributions in whatever order the GPU schedules them, so two runs
from the same seed differ in the last bits of every accumulated entry.

**Why.** Each atomic add rounds; the order of the adds depends on launch
scheduling. `dr.block_sum` and `dr.block_reduce` also fall back to atomic
scatters in their symbolic mode, which the default mode selects for
unevaluated inputs or inputs above about 1 GiB.

**How to apply.** Where a sum must repeat, lay the replicas out in fixed
positions and add them in a fixed order:

```python
def fixed_order_sum(values, n, copies):
    """Sum values[k * n + i] over replicas k, in order k = 0, 1, ..."""
    dr.eval(values)
    lane = dr.arange(mi.UInt32, n)
    total = dr.gather(type(values), values, lane)
    for k in range(1, copies):
        total += dr.gather(type(values), values, lane + k * n)
    return total
```

When each output's replicas are contiguous instead, `dr.block_sum(values,
copies, mode="evaluated")` reduces them with a dedicated kernel rather than
atomic scatters; confirm that it repeats on your device before relying on
it. Anything that reads an atomically accumulated image, such as a later
pass that samples it, inherits its variation.

### tiny-cuda-nn training does not repeat bit for bit

Two identical training runs diverge after the first update, because the
hash-grid encoding's backward pass accumulates parameter gradients with
atomic adds.

**Why.** Same mechanism as above, inside the network library, where it cannot
be changed without replacing the kernel.

**How to apply.** Accept a run as numerically unchanged by comparing it with
an unchanged repeat of itself, not with zero. Measure the repeat envelope
first: the spread between two runs of the same code over the same horizon.
A candidate whose difference from the reference stays inside that envelope
has not changed the numerics; a candidate compared against exact equality
fails for reasons unrelated to it.

**Evidence.** Two runs of the same code from the same checkpoint and seed
held weights 1e-8 apart after one update. A derivative computed from them
differed by 2.2e-6 after two updates and 6.1e-2 after eighteen; a changed
implementation compared with the old code drifted by the same amounts. Two
walk implementations that agreed to rounding on every batch produced pose
trajectories identical for 140 steps, which then separated, by 0.007 scene
units at step 255.

### Separate sampling differences from transport differences

When two implementations of a traced walk disagree, first run both on the
same materialized single-precision rays.

**Why.** A full run mixes two sources of difference: which paths were
sampled, and what the walk computes for a given path. Fixed rays remove the
first.

**Evidence.** A symbolic and an unrolled implementation of the same walk
matched bit for bit on fixed rays, while full runs differed by a few paths in
a grazing-angle count from the first step. The unchanged code repeated in a
fresh process differed in the same count, and the cause was the table
construction above, not the loop.

## Replicates

### Independent replicates vary the base seed

Replicates that measure Monte Carlo variability, or the reliability of a
training recipe, each use a different base seed. Reruns with the same base
seed repeat the same streams; their spread measures only the nondeterministic
reductions above, and they are not independent evidence.

**Why.** Same-seed reruns share every sample, so a failure that depends on an
unlucky draw either appears in all of them or in none. Their spread is the
right tolerance for a parity gate (the repeat envelope) and the wrong one
for a confidence interval.

**How to apply.** Make the base seed a required parameter of every entry
point, record it with the results, and report the replicate count and seed
policy beside every aggregate. Derive every domain from the base seed, so
changing it changes all streams at once.

**Evidence.** A program without a seed parameter produced "replicates" that
repeated the same streams. Once the base seed could vary, one of three
training objectives collapsed to an identically zero output in 3 of 6 seeded
runs, a failure the reruns had never shown; the other two objectives, run on
the same seeds, never collapsed.
