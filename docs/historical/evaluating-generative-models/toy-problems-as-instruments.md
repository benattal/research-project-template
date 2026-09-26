# Toy problems as the primary instrument

A generator's failures are easiest to see and to attribute when the target
distribution is known exactly. Low-dimensional targets with closed-form
densities, synthetic signals drawn from a known family, and exact posteriors
over a finite catalogue let every metric be compared with its true value,
and let an oracle version of each pipeline stage be substituted for the
learned one. This file covers how to build such a suite, how to gate on it,
and how to use probes and oracle arms without mistaking them for results.

## Use targets whose answer is exact

Make the first test bed a problem where the true distribution, its support
and its density are known: 2D shapes with analytic support (spirals, rings,
checkerboards, moons, lattices of discs), or 1D signals drawn from a known
mixture family with a known number of components. Evaluate against fresh
draws from the true distribution, not against a held-out slice of a finite
training set.

**Why.** With an exact target, a metric value can be compared with the
clean-versus-clean floor and with an oracle, and a failure can be localized
on the support: which arm of the spiral is missing, which square is empty,
which component count is overproduced. Image datasets allow none of this.
Their only reference is a finite sample, their feature-space metrics mix
many failures into one number, and a visual check depends on the viewer.

**How to apply.** Rasterize the support once on a fixed grid and compute
occupancy, distance-to-support and on-support mass against it. Draw the
reference set from the analytic distribution with a fixed seed. For
synthetic signals, keep the generating parameters of every clean sample so
that fitted-parameter distances have a true population to compare with.
Only move to images once the method recovers the toy targets.

**Evidence.** On toy 2D targets, an analytic support raster made it possible
to report that a failed run reached only three of five lattice columns and
joined them with bridges, and that another had collapsed onto a single
checkerboard square (TV .95, coverage .05). On images, the corresponding
failures appeared only as an FID that stayed high without saying why.

## Build a suite in which each scene stresses one failure

Choose toy scenes so that each one exposes a different failure, and run the
same recipe on all of them. A method is promising when it recovers the whole
suite without per-scene settings.

**Why.** Different geometries break different parts of a method. Thin curves
expose blur and mass imbalance along the support. Many separated modes expose
mode dropping and bridging. Sharp-edged regions expose rounding and leakage.
Targets whose support reaches the domain bound expose saturation. A method
tuned on one scene can pass it by accident.

**How to apply.** Include at least: a thin curve (spiral or ring), a
multi-modal lattice, a region with sharp edges (checkerboard), two
interleaved non-convex shapes (moons), and one target whose support touches
the output bound. Fix one recipe, run all scenes, and report the suite mean
and every per-scene value. Record how the grid cell size compares with the
noise scale and any kernel bandwidth, since a noise scale near one cell
makes grid metrics insensitive to blur.

**Evidence.** In one suite, the same method collapsed onto one checkerboard
square by an early epoch, reached only part of a 25-disc lattice with
bridges between the columns, and on a sine-shaped target drove the whole
generator to the output bound within a few epochs. A change that fixed the
checkerboard (a disconnected multi-component source) failed on two moons at
the full horizon. The grid cell in that suite was about .032 wide, close to
both the noise scale and the kernel bandwidth.

## Pass a clean-data gate before adding noise

Before testing recovery from corrupted measurements, run the same method on
clean data and require it to match the clean reference. A method that cannot
learn the target from clean samples has no business being tuned for noise.

**Why.** Recovering a distribution from corrupted data compounds two
problems: representing and learning the target, and undoing the corruption.
A failure on noisy data is ambiguous until the first has been shown to work.
Many candidate estimators fail the first problem outright.

**How to apply.** Define the clean gate numerically, for example "clean TV
within a stated tolerance of the best clean reference on every scene of the
suite". Run it before the noisy campaign for every new estimator family.
Close the family if it fails, and state that it was closed on clean data.

**Evidence.** A family of global polynomial estimators reached clean TV .90,
.79 and .79 at degrees one to three on one scene, against .16 for the
reference kernel method, and was closed before any noisy run. A full-matrix
variant stayed flat near TV .8 on clean data. A local refit passed the clean
gate at .147 against the oracle's .152 and was the only one of these moved
on to noise.

## Bracket every arm with oracle and reference arms

Run arms that replace one learned stage with its exact version: training on
clean samples, deconvolving with the known noise, using the exact
normalizer, or using the exact target field. Label each as a reference that
could not be deployed, and report where the method sits between the plain
baseline and these ceilings.

**Why.** An oracle arm answers a specific question: how much of the gap is
due to this stage? If the oracle for a stage does no better than the method,
that stage is not the bottleneck. If the oracle reproduces a failure, the
failure is not caused by the learned approximation of that stage.

**How to apply.** Add one oracle per learned stage, not one global oracle.
Keep oracle arms in tables with a clear label and never count them as
baselines to beat. When a method crosses an oracle's value, check for
finite-sample noise before claiming it (see the
[sample-size note](sample-size-and-seeds.md)).

**Evidence.** A deconvolved kernel method at low noise reached TV .149,
matching the best clean oracle (.152). An exact-denominator oracle reached
.175 against .211 for the deployable estimator, which bounded what a better
denominator could give. An exact-field oracle reproduced an early
oscillation (good by epoch 50, collapsed around epochs 100 to 120,
recovered by 130), which assigned that oscillation to the learning-rate
schedule. A larger late collapse appeared only with the noisy field.

## Separate bias from variance with frozen-generator probes

To decide whether a training signal is biased or noisy, freeze the generator
at a checkpoint and evaluate the training signal many times at increasing
batch size. The noise part falls as one over the square root of the batch
size; the bias part does not.

**Why.** A larger batch, more samples or a longer run can only fix variance.
Knowing which part dominates chooses between spending compute and changing
the estimator.

**How to apply.** Hold the generator, the query points and the reference
fixed. Repeat the estimate over independent seeds at two or three batch
sizes and compare its spread with its mean error against the exact target.
Report the error on the support separately from the global average, which
far-off-support queries with near-zero denominators can dominate. Then
confirm the conclusion with a step-matched training run.

**Evidence.** At batch sizes 512, 1024 and 2048, the noise of a field
estimate fell from .017 to .012 to .009 while its bias on the support stayed
near .58. A step-matched training run with twice the batch reached TV .450
against .445 at the original batch: no gain, as the probe predicted. In the
same probes the global RMS error was dominated by about nine far-off-support
query points.

## Probes localize; trained trajectories decide

Use probes, oracles and single-step diagnostics to find where an error
arises. Accept or reject a change only on the trajectory of a trained
generator at a matched budget.

**Why.** A probe measures one quantity at one state. Training feeds errors
back through the generator, so a change that improves a probe can still
lead to collapse, and a probe-level error can be harmless in training.

**How to apply.** State what each probe predicts about training before
running it, then run the training arm that tests the prediction. Stop adding
probes when none of the remaining ones can change the next training
decision.

**Evidence.** A routing change cut a probe error from 5.9% to 2.5%, and the
trained generator still escaped the support within a few thousand steps
(TV 1.0). An initialization whose velocity field matched the target with
cosine similarity .999 still produced a clean run stuck at TV .8.

## Diagnostics must repeat the training computation

A diagnostic that does not reproduce the training computation exactly
measures something else. Check that it uses the same query points, the same
basis, the same order of normalization and averaging, and the same
multipliers as the step it claims to describe.

**Why.** Small differences in a diagnostic change its conclusion: resampled
query points turn a stable field into apparent noise, and averaging before
normalizing gives a different field from normalizing before averaging.

**How to apply.** Build diagnostics by calling the training code path, not a
reimplementation. Freeze query points across epochs. Compare oracle and
learned fits on the same basis. Report relative errors only where the
reference is far from zero.

**Evidence.** Audits of diagnostic tooling found five mismatches of this
kind: query points re-selected every epoch; oracle and recovered fits
computed on shifted anchor bases; kernels averaged before normalization
while the trainer averaged normalized velocities; epoch-zero frames drawn
with a multiplier of one while the first training step used four; and a
tiny relative error at epoch zero computed against a near-zero matrix.
