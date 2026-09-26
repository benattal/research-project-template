# Bias and variance trade-offs

Removing bias usually costs variance, and at a finite number of captures, a
finite batch and a finite training budget, a biased low-variance estimator
can beat an unbiased one. This file records where the crossover fell, when
the clean target is the wrong target, how to spend a fixed capture budget,
and a family of unbiased losses that are unstable because the term they
remove was also the one keeping the fit bounded. The last entries cover how
batch size separates bias from variance and why support lost under a narrow
kernel is not regained.

## When bias is worth removing

### A plug-in average wins at low noise; the unbiased estimator wins above a crossover set by noise relative to kernel bandwidth

Averaging the available views and treating the average as clean is biased,
but its variance is low. Its bias grows with the noise variance relative to
the kernel's scale, so it wins when the noise is small against the bandwidth
and loses beyond a crossover.

**Why.** The plug-in's bias is the diagonal noise energy over the number of
views plus a curvature term of order $\sigma^2 / h^2$ for a kernel of
bandwidth $h$ (see [independent-views.md](independent-views.md) and
[sources-of-bias.md](sources-of-bias.md)). The unbiased estimator spends its
views on independence instead of on averaging, so its variance is higher at
every noise level. At small $\sigma / h$ the plug-in's bias is smaller than the
unbiased estimator's excess variance.

**How to apply.** Always run a capture-matched plug-in average as a baseline,
and locate the crossover on a problem with a known answer before choosing an
estimator for a new noise level. A lower distance to the target at high noise
does not mean the support was recovered; check the samples.

**Evidence.** On a 2-D spiral with a clean-optimal Gaussian kernel of
bandwidth $h$, every arm trained from scratch to a plateau at each Gaussian
noise level. The independent-view estimator used three views, cross-fitted
over all role permutations, with a numerator unbiased for a routed local
approximation of the kernel and a positive denominator taken from the view
average. At noise $h/3$ the three-view average reached total variation 0.164
against 0.169. From $2h/3$ upward the independent-view estimator won: 0.175
against 0.182 at $2h/3$, 0.411 against 0.545 at $3.3h$, and 0.538 against
0.791 at $6.7h$, with the clean oracle at 0.152. The independent-view samples
stayed a recognizable, complete spiral up to about $1.7h$ with growing
interior mass; at $3.3h$ they lost the inner turn, and at $6.7h$ both
methods produced filled disks.

### Biased estimators converge faster; compare at convergence

A plug-in target has low variance from the first step, so it makes fast
early progress and then stops at its biased optimum. An unbiased target
improves more slowly and keeps improving. Comparisons at a fixed early epoch
favour the biased estimator.

**Why.** Early in training the error is dominated by distance from the
optimum, which low-variance gradients reduce quickly. Late in training the
error is dominated by the optimum's own bias, which only the unbiased
estimator removes.

**How to apply.** Train every arm to a stated convergence rule (for example,
no improvement above a fixed threshold over the last 50 evaluations, and
stable samples) and compare the converged results. Plot whole trajectories
rather than single epochs.

**Evidence.** On Poisson histograms, the three-view average led at epoch 60
(sliced Wasserstein distance 0.098 against 0.124). By epoch 120 the
independent-view estimator was ahead (0.097 against 0.105), and it kept
improving to epoch 200 while the average plateaued.

### Unbiased for the clean target is not optimal for a noisy deployment input

A model that maps one noisy input to an estimate of the clean signal is best,
in mean squared error, when it shrinks the noisy input toward its mean (the
Wiener or MMSE solution). The ordinary noisy-input loss produces that
shrinkage. A cross-view loss that is unbiased for the clean-input loss removes
the shrinkage, which helps on clean or averaged inputs and amplifies noise on
a single noisy one.

**Why.** For an affine map $A(v) = W v + b$ followed by a decoder $D$ with
Jacobian $J$, the ordinary loss on a noisy input decomposes as

$$
\mathbb{E}\lVert D(A(v)) - \tilde y_t\rVert^2
\approx \lVert D(A(x)) - x\rVert^2 + \sigma^2 \lVert J W\rVert_F^2 + \mathrm{const} ,
$$

and the second term is a noise-scaled ridge penalty on the map's gain. The
cross-view loss cancels it. The penalized map attenuates the low
signal-to-noise bands, and the same map applied to a clean image attenuates
that image's genuine high frequencies. Without the penalty the map keeps
those frequencies and also passes the noise in them.

**How to apply.** Decide which input the model will see at deployment. For a
single noisy measurement, the shrunk estimator is the right answer and its
lost detail is the posterior mean's own smoothness, not a defect. For
generation, evaluate samples with per-sample sharpness, diversity across
seeds, and the gap between a single sample and the average of samples:
posterior diversity keeps sharpness near the clean data's, while amplified
noise raises it and makes the average much better than any sample.

**Evidence.** For Poisson images with photon scale $\alpha$ (mean count
$\alpha x$ per pixel), the map trained with the ordinary loss kept its total
gain energy constant across $\alpha$ while its high-frequency gain halved
(0.00249 to 0.00129), and the decode of a clean image kept 92 percent of the
reference high-frequency energy at $\alpha = 100$ and 75 percent at
$\alpha = 5$. At $\alpha = 5$ the shrunk map scored 31.6 dB on both clean and
single noisy inputs; the cross-view-trained map scored 17.2 dB on a single
noisy input. Its outputs carried 3.9 times the high-frequency energy of the
clean images, against 0.77 for the shrunk map, and averaging its outputs over
noise draws gained 9.7 dB against 0.6 dB: its spread was amplified
measurement noise, not posterior diversity.

## Spending captures and pixels

### Spend a fixed capture budget where the error is

When the method may use a fixed number of captures, each role (routing a
local approximation, evaluating it, supplying the direction, fitting
coefficients) competes for them. The allocation is an experimental axis, and
the best one puts captures on the dominant error term.

**Why.** Routing captures reduce the noise in the choice of local expansion
point; evaluation captures reduce the variance of the evaluated polynomial;
the direction capture's variance averages out over steps. A higher polynomial
degree reduces approximation error but consumes evaluation captures that
could have gone to routing.

**How to apply.** Measure the total error of the evaluated numerator against
the exact clean numerator for each allocation on a frozen checkpoint before
training. Move captures toward the term with the largest error, one at a
time.

**Evidence.** Under an eight-capture budget at noise $2h/3$, moving captures
from routing to direction worsened total variation monotonically: 0.317 with
five routing captures, then 0.345, 0.366, 0.745 and 0.798 as routing
captures fell to four, three, two and one. For the local polynomial, degree
two with three routing and four evaluation captures had total bias 0.057 and
5.7 percent nonpositive masses; degree three with one routing and six
evaluation captures had total bias 0.46.

### An unbiased input to a nonlinear stage can be worse than a biased, lower-variance one

Feeding an unbiased but noisy estimate into a nonlinear operation (a routing
decision, a clamp, a per-sample least-squares solve) produces a biased output
anyway, and often a noisier one than feeding a biased, averaged estimate.

**Why.** The nonlinearity converts variance into bias (see
[sources-of-bias.md](sources-of-bias.md)). An unbiased input has more
variance to convert.

**How to apply.** Use unbiased constructions where the downstream use is
linear, and prefer the lowest-variance input where it is not. Compare both on
the total error of the final quantity, not on the bias of the intermediate
one.

**Evidence.** An unbiased U-statistic estimate of the squared distance used
for routing gave total numerator bias 0.080, against 0.077 for routing on the
distance to the averaged routing captures. A per-image chart fitted by ridge
regression from noisy data each step trailed a trained network that predicted
the chart by 3.6 dB at matched epochs, with three times the view-to-view
output distance: the per-sample fit injected estimation variance into every
prediction.

### Every extra role cut from one measurement costs information

A single measurement split into more disjoint parts gives more independent
roles, but each part sees fewer pixels. A linear map of a very sparse view
cannot carry fine detail, and a consistency term between two sparse views
constrains the model to the content they share, which is the low frequencies.

**Why.** The information available to each role is proportional to its pixel
count. Splitting the input side into two disjoint halves for a cross-view
product halves each view again, so a partition designed for exact
independence of four roles can leave each view with a few percent of the
pixels.

**How to apply.** Use the smallest number of disjoint roles the estimator
needs. When the loss is a mean squared error against a held-out target, one
held-out subset suffices: give the input and the conditioning the whole
remaining measurement. At inference, several predictions from different
disjoint splits of the same measurement can be averaged when the prediction
is linear in its input.

**Evidence.** A four-way partition that left each input view with 6 to 16
percent of the pixels produced blurred reconstructions: the learned decoder
applied to a reference latent reached about 23 dB, against about 31 dB when
views were full images. Giving the input 75 to 87.5 percent of the pixels
and holding out the rest as the target matched the full-view reference (see
[noise-models.md](noise-models.md)). For a chart computed from a 50 percent
split, the latent error was 1.08 against 0.62 for the full measurement;
averaging the charts of four splits reduced it to 0.81.

## Losses that lose their minimum

### Unbiased cross-view losses are not bounded below

A cross-view loss $\langle r_0, r_2 \rangle$, with $r_i$ the residual of view
$i$ against an independent target, is unbiased for the clean loss but can be
driven to minus infinity by growing the part of the residuals that differs
between the views. Its only restoring force is the shared-signal term, which
weakens as the signal-to-noise ratio falls.

**Why.** The polarization identity gives

$$
\langle r_0, r_2 \rangle
= \Big\lVert \frac{r_0 + r_2}{2}\Big\rVert^2 - \Big\lVert \frac{r_0 - r_2}{2}\Big\rVert^2 .
$$

The second term is the view-to-view variation, which a model can increase
without limit by amplifying input noise. In expectation the cross terms
cancel, but a high-capacity model finds per-sample directions where the
negative term wins. Adding $\lambda \lVert r_0 - r_2\rVert^2$ bounds the loss and
reintroduces a bias of $\lambda$ times the noise variance; at
$\lambda = 1/2$ the sum is the ordinary single-view loss. A mean squared
error against a held-out target is both unbiased (up to a constant) and
bounded.

**How to apply.** Monitor the distance between the model's outputs on two
independent views; a steady rise is the failure. Choose between a small
consistency weight (small bias, bounded), a held-out-target mean squared
error, or pacing with a lower learning rate while keeping checkpoints of the
best exponential-moving-average weights. Expect the failure sooner with
sparser views and lower photon counts.

**Evidence.** With full-image Poisson views at photon scale 5, a
cross-view autoencoder loss peaked at 34.2 dB after six epochs, then fell to
23.1 dB while the view-to-view distance rose from 0.56 to 2.34 and
single-view reconstruction collapsed from 19 to 2.8 dB. Adding a consistency
term with weight 0.1 held the distance at 0.60 and the reconstruction at
33.6 dB. With partitioned single-measurement views of 5 to 16 percent of the
pixels at photon scale 0.25, the same loss went from 13 to $-1.4 \times 10^{7}$
within about one epoch; a held-out-target mean squared error trained stably.

### Detaching one branch of a product loss keeps the gradient unbiased but removes the minimum

For a loss that multiplies two model outputs, stopping the gradient through
one of them halves the expected gradient, which a learning rate absorbs, but
it also makes each batch's objective linear in the other output, with no
minimum.

**Why.** By the product rule,

$$
\nabla \langle f_0 - t, f_2 - t \rangle
= \langle \nabla f_0, f_2 - t \rangle + \langle f_0 - t, \nabla f_2 \rangle ,
$$

and by symmetry each term has half the expected full gradient. With the
second output detached, the batch objective is linear in $f_0$, and the
restoring force that came from the second term (the shared signal growing
with the model's gain) is gone.

**How to apply.** Keep gradients through both branches and pay the second
backward pass; use gradient checkpointing if memory is the reason for
detaching. Check coercivity directly: scale the model's gain by several
factors on held-out data and confirm that the loss value has a minimum.

**Evidence.** The detached version diverged: reconstruction fell from 24.7 to
21.6 dB between two evaluations, and the view-to-view distance rose from 0.77
to 1.15, against about 0.30 for the ordinary loss. The symmetric loss, scaled
by factors of 0.5, 1, 2, 4 and 8 in the map's gain, took the values 0.36,
0.29, 0.35, 0.39 and 0.41: a minimum at the trained gain. (That symmetric run
later drifted toward the curvature-biased optimum described in
[sources-of-bias.md](sources-of-bias.md), a separate effect.)

## Finite samples and training

### More draws, larger batches and smaller learning rates reduce variance only

Averaging more samples per step, or more steps per unit of progress, shrinks
the variance of an unbiased or biased estimator. It leaves the bias where it
is. A gap that does not move with the batch is bias.

**Why.** The mean of $B$ independent draws has standard error proportional to
$1 / \sqrt{B}$ and the same expectation as one draw. A smaller learning rate
averages more steps into each unit of parameter change, with the same effect.

**How to apply.** When a method is short of a target, first scale the batch
at a matched number of optimizer steps. Compare batch sizes at matched steps
and at matched data passes; matching epochs alone halves the step count of the
larger batch and confounds the two. Lower the learning rate only together
with a longer horizon.

**Evidence.** At high noise on a 2-D spiral, doubling the batch at matched
optimizer steps gave total variation 0.450 against 0.445, no change, which
located the remaining gap in bias. A larger batch at matched epochs had
reached 0.425 on half the optimizer steps, a comparison that could not
separate batch size from schedule. A three-times smaller learning rate at the
same horizon under-trained (0.591, degrading after an early best of 0.463).

### Coverage lost under a narrow kernel is not regained

A narrow kernel exerts no force on samples a few bandwidths from the data.
Samples pushed there, by one outsized step or by gradual drift, stay there,
and the support they abandon stays empty. An estimator can be accurate on
average and pass an audit of a checkpoint, and still fail this way in
training.

**Why.** The kernel's field decays like $\exp(-d^2 / 2 h^2)$ with the distance
$d$ to the nearest data, so off-support samples receive essentially no
restoring pull. Two routes lead there. A reciprocal of a small or signed
mass produces a rare outsized velocity that throws samples off the support.
Without any tail event, a high learning rate lets the generator drift off
some segments of the support; the samples that remain receive no pull back
toward the empty segments.

**How to apply.** Log the tails and the coverage every epoch: the upper
percentiles of the velocity, the fraction of nonpositive masses, the smallest
mass relative to the batch mean, and the fraction of the target's occupied
histogram cells (or modes) that hold samples. Use positive denominators, broad
initial coverage of the output space, a locality schedule that starts broad,
and a learning-rate decay that arrives before coverage is lost. Judge an
estimator by its training trajectories, not only by audits of a fixed
checkpoint.

**Evidence.** A three-view run normalized per query by its estimated masses
passed a 512-repeat audit at its best checkpoint (positive masses, velocity
maxima of 0.084), then escaped between epochs 5 and 10 and saturated at the
output boundary. In other runs at high noise, with the velocity's 99th
percentile at only 0.03 to 0.09 and no negative masses, the fraction of the
target's occupied histogram cells that held samples fell from 0.71 to 0.36
and from 0.58 to 0.05.
A baseline run with the same excursion (0.69 to 0.38) recovered only as its
learning rate decayed.
