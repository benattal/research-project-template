# Normalization and self-normalization

Kernel distribution-matching objectives divide their attraction and repulsion
terms by kernel masses (sums of kernel weights), so that the field keeps a
usable scale as the generator moves through regions of high and low data
density. Softmax row weights, which divide each weight by the row's own sum,
do the same thing and are self-normalized. A mass estimated from the batch is
a random quantity, and division by it is biased even when the numerator and
the mass are each unbiased. This file separates what must be unbiased (the
numerator, whose zero is the equilibrium) from what may be approximate (a
positive scale), and records how estimated denominators, floors, caches and
scale constants fail.

Notation: for a query $x$, the attraction is
$A(x) = \mathbb{E}[k(x, y^{+}) (y^{+} - x)]$ over data $y^{+}$, the repulsion
is $R(x) = \mathbb{E}[k(x, y^{-}) (y^{-} - x)]$ over generated samples
$y^{-}$, and the masses are $K_p(x) = \mathbb{E}[k(x, y^{+})]$ and
$K_q(x) = \mathbb{E}[k(x, y^{-})]$.

## Ratios and equilibria

### Make the numerator unbiased and treat the denominator as a separate, labelled choice

A ratio of unbiased estimates is biased. For a field that is used through its
zero set, a positive scalar denominator that is independent of the
numerator's noise changes only the speed of the flow, not where it stops. A
denominator that shares captures with the numerator, or different
denominators on the two terms, can move the equilibrium.

**Why.** The normalized field can be written over a common denominator:

$$
V(x) = \frac{A(x)}{K_p(x)} - \frac{R(x)}{K_q(x)}
= \frac{K_q(x) A(x) - K_p(x) R(x)}{K_p(x) K_q(x)} .
$$

The cross-weighted numerator $K_q A - K_p R$ is a product of expectations over
independent data and generated samples, so it has an unbiased estimator
built from independent factors. If the denominator $D$ is positive and
independent of the numerator estimate $N$ given $x$, then
$\mathbb{E}[N / D \mid x] = \mathbb{E}[N \mid x] \, \mathbb{E}[1 / D \mid x]$,
which is zero exactly where the clean numerator is zero. Estimating
$A / K_p$ and $R / K_q$ separately with batch masses gives each term its own
ratio bias, and the two biases need not cancel at the clean equilibrium.
Dividing each term by the square root of its mass changes the equilibrium
even with exact masses, since $A / \sqrt{K_p} = R / \sqrt{K_q}$ is a different
condition from $A / K_p = R / K_q$. Self-normalized softmax rows use the
batch's own sums, so their bias is of order one over the number of samples in
the row.

**How to apply.** Implement the cross-weighted numerator and certify only it.
Choose the denominator from a short list and name the choice in every result:
a fixed positive constant (no bias), empirical per-query masses (ratio bias),
or learned masses (approximation error). Keep it positive. Report the
numerator's bias and the denominator's error as separate numbers.

**Evidence.** On a 2-D spiral with noisy data, the mean of 200 noisy
velocities computed with empirical per-query masses had cosine 0.997 to the
same estimator's clean-data velocity. The same numerator divided by masses
fitted on the noisy captures themselves had mean relative error of 10 to 300.
The separate-normalization form filled thin supports that the cross-weighted
form recovered. On an unconditional image task, switching softmax (linear)
division for square-root division changed convergence speed but not the
final quality.

### Never put a signed unbiased mass in a reciprocal

An unbiased estimate of a small positive mass is negative with nonzero
probability. Dividing by it produces rare, huge, sign-flipped velocities, and
one such step can move samples into a region the field cannot recover from.
A biased but positive estimate of the mass is the better denominator.

**Why.** A bilinear or polynomial mass estimate has variance comparable to its
mean where the kernel is narrow. The reciprocal of a quantity whose
distribution reaches zero has no finite mean. The numerator is unaffected by
this choice, so a biased positive denominator costs only a smooth rescaling.

**How to apply.** Compute the denominator from a positive estimator, such as
the exact kernel on the average of all available captures, while keeping the
numerator unbiased. Log the fraction of nonpositive masses and the upper tail
of the velocity every epoch; a nonzero fraction of nonpositive masses in a
reciprocal is a defect regardless of how the mean behaves.

**Evidence.** With per-query signed masses, rare velocities reached the
hundreds or thousands. A three-view run normalized per query by its signed
masses escaped between epochs 5 and 10 and saturated at the output boundary,
although a 512-repeat audit of the epoch-5 checkpoint showed positive masses
and bounded velocities: the failure was a rare event on the training path.
The same numerator with a positive batch-level denominator stayed stable. With
the numerator unchanged, an exact-kernel
mass on the average of eight captures (a biased plug-in, with RMSE 0.0029 and
mean bias $-0.0002$ against the clean mass) gave total variation 0.154,
against 0.181 for a batch-mean positive mass and 0.164 for the three-view
average baseline.

## Floors, caps and global scales

### Floors and caps are part of the estimator; set floors below the natural mass scale

A floor on a mass protects the reciprocal, and a cap on the velocity bounds a
step, but both change the field. A floor above the masses that occur on the
support zeroes the field exactly where it is needed; a floor far below them
lets an inaccurate numerator divided by a tiny off-support mass produce huge
velocities.

**Why.** Masses of narrow kernels are small: a Gaussian kernel whose bandwidth
is about one data-grid cell has masses near $5 \times 10^{-5}$ early in
training, so their product $K_p K_q$ is near $10^{-9}$. A floor of $10^{-3}$
on that product replaces every such value, and the field becomes the
numerator divided by the floor, orders of magnitude smaller than the
normalized field. Floors hidden in defaults are the dangerous case, because
nothing in the experiment description mentions them.

**How to apply.** Print the distribution of masses at initialization and at a
late checkpoint, and set the floor below the on-support values. Route every
floor through one named parameter and forbid literals elsewhere. Report the
floor, and any velocity cap, as part of the denominator policy, and do not
treat a cap that bounds a failure's severity as a fix for the failure.

**Evidence.** A floor of $10^{-3}$ zeroed the exact narrow-kernel field on the
support of a 2-D spiral. A sanity arm, whose fitted kernel reduced exactly to
the exact kernel, reached total variation 0.575 after 30 epochs against 0.385
for the oracle; the difference was a default floor of $10^{-5}$ in that code
path, and with $10^{-8}$ the arm reached 0.398. With a $10^{-8}$ floor, a
fitted numerator with small off-support errors, divided by off-support masses,
gave velocities 100 times too large. A matched velocity cap bounded the
severity of a later instability but did not remove it.

### A batch-mean denominator creates a dead zone at low density

Replacing each query's mass by the batch mean is positive and stable on
average, but it attenuates the restoring field for queries in low-density
regions by the ratio of their mass to the mean. A sample that leaves the
support stops being pulled back.

**Why.** With a per-query denominator, the normalized attraction toward the
data is order one wherever the kernel reaches the data. With the batch mean,
it is $K_p(x) / \overline{K_p}$ times smaller, which for a narrow kernel and a
query off the support is many orders of magnitude.

**How to apply.** Use a per-query positive denominator. If a global scalar is
used for stability, test it with the exact clean kernel first: a failure
there is a denominator failure, not an estimator failure.

**Evidence.** With the exact kernel and an average-of-eight target, one seed
using the batch-mean mass escaped the support near step 3,200 and saturated
at the corners of the output box. At a trained checkpoint, the lowest query
mass was $8.6 \times 10^{-7}$ of the batch mean.

### Match the denominator's kernel scale to the numerator's

A broad kernel gives positive, smooth masses and is tempting as a denominator
for a narrow numerator, but its masses are much larger, so the normalized
velocity becomes tiny. Rescaling by the ratio of batch means fixes the
magnitude but not the shape mismatch.

**Why.** The denominator sets the step size per query. A mass from a kernel ten
times wider is orders of magnitude larger than the narrow numerator's natural
normalizer, and its dependence on the query's position is smoother, so the
relative scaling between dense and sparse regions is also wrong.

**How to apply.** Normalize with masses of the same kernel as the numerator,
estimated positively. If a different kernel must be used, check the median
and the 99th percentile of the normalized velocity against an exact-mass
reference before training.

**Evidence.** Dividing a narrow numerator by broad-kernel masses reduced the
median normalized velocity at a trained checkpoint from $2.1 \times 10^{-2}$
to $4.3 \times 10^{-5}$ and produced a filled disk (total variation 0.54).
Rescaling each side by its batch-mean ratio restored the 99th-percentile
velocity to 0.27 against 0.35 with exact masses, but the run still reached
only 0.36, against 0.17 for the five-view average baseline.

### Fix kernel scale constants once, share them across matched arms, and remember that noise inflates norms

A distance scale computed from noisy measurements is larger than one computed
from clean signals, because noise adds its energy to every norm. When a clean
control and a noisy arm calibrate their scales separately, they train with
different kernels, and the difference is read as a difference between
methods.

**Why.** $\mathbb{E}\lVert \tilde y\rVert^2 = \lVert y\rVert^2 + \mathrm{tr}\,\Sigma$, so a
percentile of noisy norms overstates the clean scale, severely at low photon
counts. A scale recomputed from each batch is a random function of that
batch's noise, which also takes the kernel out of the setting where the
product identities apply. A rescaling that depends only on a known quantity,
such as the fraction of pixels a partition keeps, is deterministic and keeps
the estimate unbiased.

**How to apply.** Compute the scale once at startup from a seeded calibration
draw, store it with the run, and pass the same value to every arm of a
comparison, including the clean oracle. When a noise model changes the
coordinate scale (for example, scaled counts), convert bandwidths through a
fixed reference scale rather than recalibrating per arm. Normalize subset
distances by the known kept fraction.

**Evidence.** On face images at a low photon count, the 99th-percentile norm
was about 1,567 for noisy measurements and about 367 for clean images, a
factor of about 18 in the squared scale that enters the kernel. A clean
control calibrated on clean images drove an unbounded kernel negative and
diverged to a loss near $10^{30}$, while its noisy counterpart trained. On
Poisson histograms, a coordinate scale of 121.9 under the noisy normalization
against 102.7 under clean calibration made a nominal bandwidth too local
until bandwidths were converted through a shared reference scale.

## Learned normalizers and diagnostics

### Judge a learned normalizer by the field it produces, not by its regression loss

A network that predicts masses can fit its regression target well and still
worsen the recovered distribution, because small relative errors in the
denominator change the balance of attraction and repulsion near the
equilibrium, where the net field is small.

**Why.** At equilibrium, attraction and repulsion nearly cancel, so the net
field is a small difference of large terms. A one-percent error in one mass
is a one-percent error in one term, which can be a large fraction of the
difference. A learned predictor also lags the moving generator.

**How to apply.** Regress on linear, unbiased mass targets; the logarithm of a
clamped single draw is biased. Learn only the noisy data-side mass and keep
the exact generated-side mass, which needs no approximation. Exclude each
query's own datum from its target. Promote a learned normalizer only after an
end-to-end comparison against the empirical denominator on matched runs.

**Evidence.** On a 2-D checkerboard, a learned positive normalizer reached
mass RMSE of 0.0047 and 0.0015 with grid correlation 0.98, yet the sliced
Wasserstein distance rose from 0.037 to 0.052, the energy distance from
0.0055 to 0.0095, and bridges appeared between squares. On histograms, a
learned normalizer reached a regression loss of 0.0006 and worsened every
population distance. On a spiral, fitting the learned normalizer more closely
to its routed target moved it further from the exact clean mass.

### Averaging and normalization do not commute; diagnostics must follow the trainer's order

The mean of normalized velocities is not the normalized mean of kernels. A
diagnostic that averages in a different order from the trainer measures a
different estimator.

**Why.** Unless the denominators are equal,

$$
\frac{1}{m} \sum_i \frac{N_i}{D_i}
\ne \frac{\frac{1}{m} \sum_i N_i}{\frac{1}{m} \sum_i D_i} .
$$

When a trainer averages the normalized velocities of two independent anchor
halves, a probe that averages the two fitted kernels first and then normalizes
shows a field the trainer never used.

**How to apply.** Build diagnostics by calling the trainer's own estimator
function on stored inputs, and reproduce its averaging order, its halves and
its role permutations. Test parity on one batch before reading any
diagnostic plot.

**Evidence.** A geometric probe first rendered only one of the trainer's two
anchor halves; the corrected version averaged the fitted kernels before
normalizing, which a final check showed differed from the trainer's average
of two separately normalized velocities. Only the third version, matching the
trainer's order, was used for conclusions.

### Evaluate distance expansions and small linear solves in full precision

Pairwise squared distances computed as
$\lVert a\rVert^2 + \lVert b\rVert^2 - 2 \langle a, b \rangle$ cancel large terms, and cross-view
products do so more severely because each factor is a small difference of noisy
quantities. Reduced precision turns the cancellation error into bias-like noise
in the masses.

**Why.** bfloat16 keeps about three significant decimal digits, so its
rounding error on the large terms can exceed the small difference that
carries the signal. Ridge systems solved from Gram matrices have the same
problem when the ridge parameter is small.

**How to apply.** Disable mixed-precision autocast inside the drift or loss
estimator even when the model runs in reduced precision. Accumulate Gram
matrices and solve small ridge systems in double precision.

**Evidence.** Rounding noise in single-precision Gram matrices was at the
scale of a $10^{-6}$ ridge parameter, which prompted double-precision
accumulation for per-sample ridge fits. The estimator is kept outside
autocast in all trainers for the cancellation reason above.
