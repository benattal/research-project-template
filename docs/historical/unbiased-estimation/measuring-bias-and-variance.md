# Measuring bias and variance

A claim that an estimator is unbiased is a claim about an expectation, and it
is checked by estimating that expectation: repeat the estimator on
independent draws and compare the mean with an exact reference. Training
curves cannot make that check, because the loss mixes bias, variance and
optimizer state. This file records how to separate bias from variance on a
frozen model, how to attribute each to one ingredient, which ceilings, floors
and baselines a result needs, and the protocol checks (paired seeds,
numerical identity tests, configuration validation, diagnostic parity) that
keep the measurements honest.

## Separating bias from variance

### Probe a frozen model with repeated independent draws and compare the mean with the exact clean quantity

Freeze a checkpoint, fix a set of query points, and recompute the estimator
many times with fresh noisy captures. The mean minus the exact clean value is
the bias; the spread across repeats is the variance. Repeating at several
batch sizes separates the two: the spread falls as $1 / \sqrt{B}$ and the bias
does not move.

**Why.** For an estimate $\hat V$ of a reference $V$ at a fixed query, with
$R$ independent repeats and single-draw standard deviation $s$,

$$
\hat b = \frac{1}{R} \sum_{r=1}^{R} \hat V_r - V ,
\qquad
\mathrm{sd}(\hat b) \approx \frac{s}{\sqrt{R}} .
$$

A measured bias means something only against this repeat floor. Freezing the
model removes the optimizer from the measurement, and fixing the queries
makes repeats and batch sizes comparable.

**How to apply.** Rebuild the model from its recorded configuration, load
the checkpoint, fix a few hundred queries, and repeat over 24 or more
independent draws at two or more batch sizes. Measure the error in the
quantity that sets the equilibrium (for a normalized drift, the numerator
before division) and bin it by distance to the support: a global root mean
square is dominated by the few far queries whose denominators are near zero.
Report the bias relative to the reference magnitude and to the repeat floor,
and the cosine between the mean estimate and the reference. Confirm a
batch-independent bias in training by scaling the batch at matched optimizer
steps (see [bias-variance-trade-offs.md](bias-variance-trade-offs.md)).

**Evidence.** On a converged high-noise checkpoint, the velocity noise fell
from 0.017 to 0.012 to 0.009 as the batch went from 512 to 1024 to 2048,
while the on-support bias did not shrink: 0.57 to 0.61 of the reference
magnitude at batch 512 and 0.79 at 2048. Training
with twice the batch at matched optimizer steps agreed: total variation 0.450
against 0.445. An earlier global velocity root mean square had been dominated
by about nine far-off-support queries with near-zero denominators.

### Swap one ingredient at a time for its clean or exact counterpart

To find where a bias comes from, replace one ingredient (the fit locations,
the routing captures, the evaluation views, the denominator) with its clean or
exact version and leave the rest noisy. The ingredient whose replacement
removes the bias owns it. Clean substitutes are diagnostics, not deployable
methods.

**Why.** An ingredient that enters the estimator linearly and independently
cannot bias it; one that enters a nonlinear fit, a routing decision or a
division can. Swapping one ingredient at a time turns a list of suspects into
a measurement, and swapping all of them gives the probe's own floor.

**How to apply.** List the ingredients by role. Run the frozen probe with each
one clean in turn, and with all of them clean. Compare each bias with its own
repeat floor, because a clean substitute can change the variance: an unblurred
kernel has larger values and larger noise. In training, apply the same swap
to the denominator by training with exact clean masses and the noisy
numerator.

**Evidence.** In the probe above, with 24 repeats on on-support queries, the
training configuration had relative bias 0.58. Clean fit locations with noisy
evaluation gave 0.55 against a repeat floor of 0.62, so no measurable bias;
noisy fit locations with clean evaluation gave 0.59; everything clean gave
0.23 against a floor of 0.25. The bias belonged to the fit locations (see
[sources-of-bias.md](sources-of-bias.md)). In training at noise $2h/3$ on a
2-D spiral, exact clean masses gave total variation 0.313 against 0.317 with
the method's batch-level denominator, so the denominator was not the limit
there. At $h/3$, exact masses gave 0.175 against a three-seed mean of 0.223,
so there it was part of the gap.

### Audit trained checkpoints, with the audit's geometry scaled to the kernel

An estimator's tails depend on where the generated samples are, and that
changes during training: an audit at initialization misses tails that appear
late. The audit's own geometry must also scale with the kernel bandwidth, or
narrow kernels underflow and the audit measures zeros.

**Why.** Early in training the samples are spread out and every query carries
substantial mass. Late in training they concentrate, and some fall where the
denominator is orders of magnitude below its batch mean, which is where
reciprocals and batch-mean normalizations fail. A Gaussian kernel with
bandwidth 0.03 evaluated at distance one is
$\exp(-1 / (2 \cdot 0.03^2)) \approx e^{-556}$, below single-precision range.

**How to apply.** Run the probe on checkpoints along the trajectory,
including the best and the last, at the queries training actually produces.
Place audit points at distances proportional to the bandwidth. Report the
smallest mass relative to the batch mean and the fraction of nonpositive
masses alongside the error.

**Evidence.** At a trained checkpoint, the smallest exact mass was
$8.6 \times 10^{-7}$ of the batch mean, a dead zone that an audit at
initialization did not show. An allocation audit with fixed unit-scale
geometry made narrow kernel products underflow to zero; with geometry scaled
to the bandwidth it reported single-draw errors of 0.0054 for disjoint
captures and 0.0035 for a leave-one-out cross-fit. A 512-repeat audit of a
best checkpoint passed and the run still escaped later (see
[normalization-and-self-normalization.md](normalization-and-self-normalization.md)).

## Ceilings, floors and baselines

### The estimator on clean captures must reproduce the clean method, and its clean twin is the ceiling

Feeding clean captures into every role must reduce the estimator to the clean
kernel method it approximates, with zero error. The same estimator trained on
clean captures gives the ceiling that separates the cost of the noise from the
cost of the approximation.

**Why.** With clean views, unbiased products become exact products,
cross-fitting averages identical terms, and a routed local approximation is
exact at its expansion point. Any error left at zero noise is an
implementation or approximation error, and it has to be known before noise is
added. An unbiased estimator converges toward its clean twin, so the twin's
quality bounds every noisy arm.

**How to apply.** Choose the kernel and bandwidth by training candidates on
clean data and inspecting the samples. Before any noisy run, set all views to
the same clean capture and check the single-draw error against the exact
clean kernel. Train the clean-capture twin of every promoted noisy
configuration with the same approximation, batch and schedule. The gap from
the noisy run to the twin is the noise cost; the gap from the twin to the
exact clean kernel is the approximation cost.

**Evidence.** On clean spiral data, a Gaussian kernel of bandwidth 0.03
reached total variation 0.162 and a thin spiral, a Laplace kernel 0.195 and a
slightly thicker one, and an inverse multiquadric kernel 0.540 with the
spiral's interior filled. With clean coincident views, a three-view
cross-fitted local estimator had zero single-draw error and trained to 0.160.
On Poisson histograms, the clean-capture twin reached sliced Wasserstein
distance 0.102 and the noisy run 0.101, so the remaining gap to the 0.055
evaluation floor was all approximation (see
[noise-models.md](noise-models.md)).

### Compare with a single-view plug-in, a capture-matched average and the clean method, each at its own best settings

A debiased method is judged against three baselines: the clean method (the
ceiling), the plug-in on a single noisy view, and the plug-in on the average
of as many captures as the method uses. Each baseline gets its own tuned
settings; a baseline that inherits the new method's choices flatters it.

**Why.** The capture-matched average is the natural competitor: it uses the
same data and is biased only by the noise left after averaging (see
[bias-variance-trade-offs.md](bias-variance-trade-offs.md)). A baseline that
shares a weak component with the new method, such as its denominator or
schedule, degrades with it, and the comparison then measures nothing.

**How to apply.** Train all baselines to the same convergence rule, with
method-specific learning rates and schedules. In every results table, state
what each arm's numerator and denominator are unbiased for: the exact clean
kernel, a routed or fitted approximation of it, a biased plug-in, or an
undeployable oracle. Label a method unbiased for an approximation as such in
its name.

**Evidence.** At noise $h/3$, a routed local estimator beat five- and
eight-view averages that used its batch-level denominator, and lost to the
standard five-view average with its own per-query normalization: total
variation 0.223 against 0.170.

### Report metrics next to their finite-sample floor, use several, and look at the samples

Distribution metrics computed from finite samples have a floor: two
independent clean samples of the same size do not score zero. Different
metrics can rank the same arms in opposite orders, and a better score can
come with visibly failed samples.

**Why.** Histogram, sliced Wasserstein and energy distances between finite
samples have positive expectation even when the distributions are identical.
Metrics weight failures differently: total variation on a grid penalizes mass
off the support, while a transport distance penalizes mass moved far.

**How to apply.** Report every metric beside its clean-versus-clean floor at
the evaluation sample size. Use at least one support-sensitive metric and one
transport metric, and inspect samples from fixed seeds. Validate structural
metrics, such as mode or peak counts, on clean samples first; smooth before
counting and set prominence and separation thresholds so they count modes,
not ripples. Show unconditional samples on their own rather than beside
unrelated clean examples, which suggests a correspondence that does not
exist.

**Evidence.** On Poisson histograms, the clean-versus-clean floor was 0.055 in
sliced Wasserstein distance and 0.016 in lobe-count total variation. At noise
$6.7h$ on the spiral, the independent-view estimator won on total variation
(0.538 against 0.791), Jensen-Shannon divergence and off-support mass, while
the capture-matched average won on sliced Wasserstein distance; both sample
sets were filled disks. A first peak-count metric counted pixel-scale ripples
as peaks and was replaced.

## Protocol

### Pair seeds across arms and compare the difference with the seed spread

Two arms differ reliably only when their paired difference exceeds the spread
across seeds. The seeds must control everything random, including model
initialization, and arms compared on one seed share it.

**Why.** Seed-to-seed variation in a converged metric includes rare late
instabilities that make a single seed an outlier. A difference smaller than
the spread is not a result.

**How to apply.** Seed the model before constructing it, and draw data and
noise from named seeds. Run three paired seeds for any claimed win or
crossover, report per-seed values beside the mean, and inspect the trajectory
of any outlier seed.

**Evidence.** Doubling the batch at matched steps gave per-seed total
variations of 0.466, 0.460 and 0.425 against 0.450, 0.440 and 0.446: a mean
difference of 0.005 against a spread of 0.04. A variant of the data-side
construction gave 0.417, 0.292 and 0.274 (mean 0.328) against a mean of
0.278; its first seed degraded late in training and the other two matched.
Tables produced before model initialization was seeded had to be rerun.

### Test every identity numerically before training on it

Unbiasedness rests on algebraic facts: a map is affine, a product has the
right expectation, two code paths compute the same thing. Each is cheap to
test in isolation against a known answer. A test that passes because a branch
is silent tests nothing.

**Why.** A training run cannot distinguish an implementation error from a
statistical effect; a test with a known answer can. Zero-initialized output
layers, common in residual and attention branches, make linearity and
equality tests pass trivially at initialization.

**How to apply.** For every map that must be affine, fix the conditioning,
draw random inputs, and check

$$
A(\alpha u_1 + \beta u_2) - b = \alpha \big(A(u_1) - b\big)
+ \beta \big(A(u_2) - b\big)
$$

to a tolerance after perturbing zero-initialized parameters. Test
unbiasedness by Monte Carlo against a closed form, in two dimensions and in
higher ones. Check that a new implementation reproduces a reference
implementation step by step from the same seed. Pin the numerical precision
in these tests: a mixed-precision wrapper can produce differences that look
like a failed identity.

**Evidence.** The linearity test of an affine map with an attention path
perturbed the zero-initialized output projection, which would otherwise have
silenced that path. A per-image ridge fit recovered a synthetic chart to
$6 \times 10^{-7}$. Unit tests covered unbiasedness in three and four
dimensions, windowed charts, and the equality of two denominator code paths.
A generic reimplementation matched the reference recipe exactly at step 640
(total variation 0.97303) and then tracked it up to random-number ordering
(0.815 against 0.847 at step 2560). Differences in an earlier bfloat16 run of
the tests came from an autocast wrapper, not from the identity.

### Reject configurations that break the identity before they run

A validator that runs at startup and rejects every option that breaks the
estimator's independence or linearity requirements keeps a biased
configuration from being reported as unbiased. It also makes each
configuration state what it is unbiased for.

**Why.** Options accumulate, and clamps, averaging proxies, nonlinear feature
maps, reused views and data-dependent parameters each break the identity
without an error. A unit mistake does the same. Checking them in one place is
cheaper than auditing every run.

**How to apply.** In a strict mode, require distinct view indices for every
role, all of them independent noisy captures; direction views disjoint from
kernel views; no more roles than available realizations; no kernel clamps or
averaged proxies; only certified kernel families, rejecting nonlinear feature
kernels; affine parameters flagged as fitted on independent data; a
deconvolution bandwidth above the noise level; and positive floors. Require a
declared scope (unbiased for the clean kernel, or for a stated approximation)
and reject approximations declared as clean. Put units in parameter names,
and version results produced before a unit correction.

**Evidence.** A guard that required every view index to address a real noisy
realization stopped several clean-data configurations at their first batch;
without it, a sentinel index would have gathered misaligned views without an
error. A fit radius given in distance units where the code expected squared
distance enlarged every fit region by a factor $1/h$ and produced signed
masses and exploding velocities, while the fit error near the support looked
good.

### Diagnostics must use the trainer's conventions and fixed inputs

A diagnostic that normalizes, masks or averages differently from the trainer
measures a different estimator. A diagnostic whose inputs change from epoch to
epoch cannot be compared across epochs.

**Why.** Scale factors, masks and averaging order all change the value of a
nonlinear quantity (see
[normalization-and-self-normalization.md](normalization-and-self-normalization.md)).
Random evaluation subsets add variance that looks like training progress or
regression.

**How to apply.** Compute diagnostics through the trainer's own functions
with the trainer's normalization and averaging order. Fix evaluation subsets,
queries and seeds once and reuse them at every checkpoint. When a diagnostic
disagrees with the samples, check its inputs against the trainer's before
believing either.

**Evidence.** A round-trip reconstruction diagnostic applied a sparse
training view's normalization mask to a full clean image and reported a
meaningless score; it had to be ignored. Seeding the evaluation split made the
metrics of a partitioned-measurement run comparable across epochs.
