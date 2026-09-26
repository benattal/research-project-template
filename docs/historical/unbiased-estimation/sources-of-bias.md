# Sources of bias

Unbiasedness is a property of the whole path from the captures to the
parameter gradient, and one step anywhere on that path can remove it. The
same few failures recur: a nonlinearity applied to an unbiased estimate, a
fitted or preprocessed object that depends on the noise it is later
evaluated against, a leak of target noise through a side channel, and an
approximation that is estimated without bias but is itself the wrong object.
This file lists them with the mechanism, the fix, and the measurement that
exposed each. Reuse of one capture inside a product is covered in
[independent-views.md](independent-views.md); division by estimated
quantities is covered in
[normalization-and-self-normalization.md](normalization-and-self-normalization.md).

## Nonlinearity

### A nonlinear function of an unbiased estimate is biased by the curvature times the noise covariance

Applying a nonlinear map to a noisy view, or to an unbiased estimate, shifts
its mean. To second order the shift is half the trace of the map's Hessian
against the noise covariance, so it grows with both the curvature and the
noise.

**Why.** For a smooth map $g$ and a zero-mean perturbation $\delta$,

$$
\mathbb{E}\big[g(z + \delta)\big]
\approx g(z) + \tfrac{1}{2}\,\mathrm{tr}\big(H_g \, \mathrm{Cov}(\delta)\big) .
$$

The forms met in practice are: a clamp at zero on a bilinear kernel estimate
(the clamp is nonlinear exactly where the estimate is noisy); an exact
nonlinear kernel evaluated on the average of views; the logarithm of a signed
single-draw mass target clamped to be positive; a pretrained nonlinear
encoder applied to a noisy view to produce a regression target; and a
pretrained nonlinear decoder placed between an affine map and a cross-view
loss. In the last case, with $A(v) = W v + b$ and view noise covariance
$\Sigma$, the latent perturbation has covariance $W \Sigma W^{\top}$ and the
cross-view loss is minimized at $D(A(x)) = x - \beta(x)$ with
$\beta_i = \tfrac{1}{2}\,\mathrm{tr}(H_i W \Sigma W^{\top})$, where $H_i$ is
the Hessian of output $i$ of the decoder $D$. The loss pins the
noise-averaged decode to the clean image, not the clean decode.

**How to apply.** Keep every nonlinear stage off the path between the views
and the product, or move it to an independent capture that only chooses
parameters (see the linear-map entry in
[independent-views.md](independent-views.md)). Replace a nonlinear
decoder inside the loss with a learned affine one, so the composition stays
affine. Label any clamped or plug-in variant as biased and exclude it from
unbiasedness claims; keep it as a baseline. Regress caches on linear
targets rather than on logarithms of clamped single draws.

**Evidence.** In the decoder case, at about five photons per pixel, the
noise-averaged decode $\mathbb{E}[D(A(v))]$ scored 27.8 dB against the clean
image while the clean decode $D(A(x))$ scored 23.7 dB, the ordering the
formula predicts. The bias energy was 376 times that of the ordinary
single-view run, and 69 percent of it lay in the constant and low-frequency
bands, so a per-channel colour correction recovered 3.7 dB. In the encoder
case, the mean encoding of noisy views differed from the encoding of the
clean image by 0.43, 0.68 and 0.79 of the clean encoding's norm at about 100,
5 and 1 photons per pixel. Decoding the best achievable latent under that
bias sat 6.9, 16.3 and 21.4 dB below the autoencoder's ceiling.

### A loss that keeps falling while the clean metric degrades has converged to a biased optimum

When removing a variance term also removes the only force that kept a model
in the regime where a curvature bias is small, the model moves toward larger
bias while its loss improves. The pattern looks like divergence but is
convergence to the wrong point.

**Why.** The ordinary noisy-input loss contains a term proportional to the
noise variance times the squared gain of the map, which shrinks the gain. A
cross-view loss cancels that term. For an affine encoder feeding a nonlinear
decoder, the curvature bias grows with $W \Sigma W^{\top}$, quadratically in
the gain, so once the shrinking term is gone the gain grows and the bias
grows faster.

**How to apply.** Track a clean-reference metric and a noise-gain statistic
(for example the distance between the model's outputs on two independent
views) alongside the training loss. If the loss falls toward its noise floor
while the clean metric falls, compare the clean output with the output
averaged over many noisy inputs; a large gap in favour of the average
identifies a curvature bias. Lowering the learning rate does not move the
optimum.

**Evidence.** The cross-view run's training loss fell monotonically (0.465 to
0.359, approaching the target-variance floor) while its reconstruction from a
single noisy input fell from 24.5 to 17.2 dB and the noise-gain statistic rose
from 0.79 to 1.41. The ordinary loss at the same noise level held about
31.6 dB on both clean and noisy inputs, with clean and noise-averaged decodes
agreeing to within 0.02 dB.

### Low-order bias corrections need the noise to be perturbative

Extrapolation in the noise level (Richardson extrapolation) and linearization
around an anchor remove the leading curvature term. They fail when the noise
is as large as the signal, because the higher-order terms they ignore are no
longer small.

**Why.** If $\mathbb{E}[g(z + s \delta)] = g(z) + s^2 q + O(s^4)$, then

$$
2\, \mathbb{E}\big[g(z + \delta)\big]
- \mathbb{E}\big[g(z + \sqrt{2}\, \delta)\big]
= g(z) + O(\lVert \delta\rVert^4) ,
$$

and higher orders use more noise scales. Each extrapolation step amplifies
both the neglected terms and the Monte Carlo noise. Linearizing the map
around a noisy anchor carries the same bias through the anchor itself;
averaging $k$ independent anchors divides it by $k$ but does not remove it.

**How to apply.** Before building a correction, measure the ratio of noise
norm to signal norm and test the correction on a single example with many
noise draws, where the clean answer is known. Prefer designs that avoid the
nonlinearity (affine compositions, distribution-level objectives) when the
ratio is near one.

**Evidence.** On one image with many noisy realizations at about five
photons per pixel, extrapolation over one to four noise scales available from
the measurement gave 19.2, 22.1, 24.0 and 23.3 dB against a 36.2 dB ceiling:
it helped, plateaued and then regressed. Over lower noise scales, used as a
diagnostic, the same orders gave 19.2, 24.3, 27.6 and 30.2 dB, improving
monotonically. First-order extrapolation of the encoder
bias recovered 2.3 dB at about 100 photons per pixel but left gaps of 13 and
19 dB at 5 and 1.

## Fitted and preprocessed objects

### An estimator is unbiased for the object it evaluates; fitting that object on noisy data changes the object

A local polynomial fitted to a target kernel and then evaluated with
independent U-statistic views gives an unbiased estimate of the fitted
polynomial. If the fit is done at noisy locations, the fitted polynomial is a
blurred version of the target, and the estimator is unbiased for the wrong
kernel.

**Why.** A weighted least-squares fit is a projection of the target function
under the measure of the fit locations. If those locations lie in a band of
width comparable to the noise around the support, and the target kernel is
narrower than that band, a low-degree polynomial cannot follow the kernel's
peak there and fits a broad bump instead. Evaluating the bump without bias
does not recover the peak. More captures or a larger batch reduce the
evaluation variance and leave this bias unchanged.

**How to apply.** Separate the two errors in every claim: estimator bias
relative to the object actually evaluated, and approximation error of that
object relative to the desired clean target. Declare which one the method
controls; a conditionally unbiased local method is "unbiased for the routed
or fitted kernel", not "unbiased for the clean kernel". Locate the bias by
swapping clean for noisy data in one role at a time (see
[measuring-bias-and-variance.md](measuring-bias-and-variance.md)).

**Evidence.** On a 2-D spiral with Gaussian noise about three times the
target kernel's bandwidth, local cells fitted on the average of three noisy
captures gave an on-support bias of 0.57 to 0.61 of the reference velocity
at batch 512, and 0.79 at batch 2048: a larger batch did not reduce it. The
fitted kernel's mass at clean points was
64 percent of the exact value. Fitting the same cells on clean points, with
noisy evaluation unchanged, removed the bias to within the noise floor. At
half that noise the band was about one bandwidth wide and the bias was four to
five times smaller.

### Fit charts, bandwidths and projections on data independent of the noise they will meet

A coordinate chart, a principal-component projection, a kernel scale or a
feature projection computed from the captures it is later applied to becomes
a random function of their noise. The product identities hold only
conditional on these objects being fixed.

**Why.** $\mathbb{E}[\langle P(a - \tilde y_1), P(a - \tilde y_2) \rangle]$
equals the clean value when the projection $P$ is independent of $n_1$ and
$n_2$. If $P$ is chosen to explain the variance of the same views, it aligns
with their noise, and the cross terms no longer vanish.

**How to apply.** Compute projections from noise-free generated samples, from
a disjoint pool of captures that is never evaluated, or once at startup from a
seeded calibration draw that is stored with the run. Treat a projection that
maximizes generated variance as a computational shortcut: it can discard
low-variance directions that matter to the target. A chart that is linear
with orthonormal rows keeps both the U-statistic features and a Gaussian
window deconvolution exact.

**Evidence.** In the fitted-kernel work, charts computed from the generated
batch alone kept the evaluation certified, while a chart selected by held-out
residual on the noisy routing pool was excluded from the unbiasedness claim.
In a feature-space experiment, projections for three feature scales were fit
on a disjoint pool of four noisy images per class and frozen before training.
A target-aware low-rank chart recovered an informative low-variance direction
that a principal-component chart of generated samples missed.

### Use a fitted object only on the pairs it was fitted on; keep exact quantities exact

A kernel approximation fitted on generated-to-data pairs is accurate only on
that product measure. Using it for generated-to-generated pairs extrapolates.
Noise-free quantities, such as kernels between generated samples, need no
debiasing and should use the exact kernel.

**Why.** A least-squares fit controls error only where the fit data lie.
Generated samples early in training sit far from the data, so the
generated-to-generated pairs occupy a region the fit never saw. An error
there enters the repulsion term directly, where it is not cancelled by
anything.

**How to apply.** Apply fitted or debiased kernels only to the noisy data side
of the field. Compute every noise-free term exactly. Record which pairs each
approximation was fitted on and assert it at the call site.

**Evidence.** A per-batch fitted polynomial kernel that was also used for
repulsion escaped the data support by epoch 25 on clean data (total variation
0.82). Adding generated-to-generated targets to the fit while still using it
for repulsion was less stable (loss about 1,600 against about 100).
Restricting the fit to the data side and using the exact kernel for generated
pairs removed this failure; the corrected run's remaining instability came
from its denominator (see
[normalization-and-self-normalization.md](normalization-and-self-normalization.md)).

## Leaks and side channels

### Look for target noise leaking through queries, seeds and conditioning

The prediction path can see the target's noise without any explicit reuse:
through a regression query centred on the datum it is scored against, through
two random streams that share a seed, or through a conditioning input that
includes target pixels.

**Why.** Each channel creates a dependence between the prediction and the
target noise, so the cross terms that should vanish keep a nonzero mean. The
dependence is invisible in the loss and in most metrics because the model
fits it.

**How to apply.** For a cache or regressor trained on noisy targets, exclude
the query's own datum from the target sum. Give the partition of a
measurement and the noise of that measurement separate random streams with
seeds that cannot coincide (offset one by a large constant). When a
conditioning network sees a masked input, give it only input-side pixels and
masks during training; at deployment feed the full measurement with all-ones
masks. Treat a sudden improvement after adding an input as a prompt to check
that input's provenance.

**Evidence.** A learned normalizer queried at data-centred points included the
query's own source datum in its target; the corrected version excluded it, and
in the same change the best total variation improved from 0.224 to 0.204 (a
stronger target was introduced at the same time, so the leak's share is not
isolated). In the single-measurement datasets, the partition stream and the
frozen noise stream shared a seed in the first epoch, which made the
partition statistically dependent on the noise it had to be independent of.

## Right estimate, wrong objective

### An unbiased estimate of a non-identifying objective does not recover the distribution

Unbiasedness makes the estimate match the clean objective in expectation. It
does not make the clean objective identify the distribution, and it does not
make the field stable.

**Why.** A kernel that is a finite-degree polynomial in the squared distance
constrains only finitely many moments of the distribution; a constant kernel
matches only the mean. An unbounded kernel that turns negative far from a
point turns attraction into repulsion there.

**How to apply.** Before any noisy experiment, train the clean-data version of
the estimator's kernel and require it to recover the target. A kernel that
fails on clean data is rejected regardless of how it behaves under noise.

**Evidence.** On a 2-D spiral with clean data, global polynomial kernels of
degree one to three reached best total variation 0.90, 0.79 and 0.79 against
0.16 for a Gaussian RBF; the samples formed a wedge, a radial cloud and a
disk, consistent with matching successively richer but finite moments. The
unclamped degree-one kernel diverged at every temperature and normalizer
tried, because its negative far-field weights made attraction anti-restoring.

### A tuned correction that cancels bias on average is not an unbiased estimator

A correction chosen because it reduces the measured error, such as scaling a
first-order term by one half, can cancel the leading approximation error on
the data where it was tuned. It is still a biased estimator and must be
labelled as one.

**Why.** Its expectation equals the target only where the cancellation holds.
There is no identity behind it, so its error can return on another scene,
noise level or training stage.

**How to apply.** Keep empirically corrected variants out of the unbiased
category, compare them against the mandatory baselines rather than against
the unbiased method alone, and report the uncorrected estimator next to them.

**Evidence.** Halving the first-order slope of a symmetrized local-affine
kernel estimate reduced its total error against the exact clean kernel from
6.35 to 0.67 percent on the calibration problem. At the lowest noise level
the corrected estimator won two of three paired seeds against the three-view
average but lost the paired mean, and it was kept out of the unbiased
category.

### Clipping, quantization and floors change the mean

Any operation that truncates a noisy value, a mass or a velocity changes its
expectation. Storing measurements clipped to a display range is the most
common and the least visible case.

**Why.** A scaled photon count at intensity near the top of the range exceeds
it about half the time; clipping at the top removes that half of the
distribution and lowers the mean. For $N \sim \mathrm{Poisson}(\alpha)$ scaled
by $1/\alpha$ at intensity one, the lost mean is about
$\mathbb{E}[\max(N/\alpha - 1, 0)] \approx 1 / \sqrt{2 \pi \alpha}$, 0.04 at
100 photons. Floors on masses and caps on velocities are biased in the same
way.

**How to apply.** Store raw integer counts (an unsigned 16-bit array holds
them without clipping) and convert at use. Treat every floor or cap as part of
the method's stated bias. Do not keep a floor that is known to bias the
result merely to run a controlled ablation on another parameter.

**Evidence.** A multi-capture measurement bank stored raw counts for this
reason. In the 2-D recovery work, a persistent floor on the output spread of
the generator's source left off-support thickness in the final samples and
was removed from the recipe.

### A biased plug-in can erase a conditional signal

When the attraction target is dominated by a noisy proxy, the part of the
field that depends on the conditioning label can vanish, and the generator
ignores the label.

**Why.** The class-specific part of the attraction is a difference between
kernel-weighted averages of targets from different classes. If the plug-in's
bias is larger than that difference, the field no longer distinguishes the
classes.

**How to apply.** Measure label adherence directly (for example, a feature of
the samples that the label controls) for every arm, including the biased
baselines.

**Evidence.** On Poisson histograms conditioned on the number of mixture
components (one to four), generators trained on the three-view average
produced a mean peak count of about 1.4 regardless of the requested number at
mild noise. The independent-view estimator's peak count rose monotonically
with the label, offset from the clean values by 0.25 to 0.3.
