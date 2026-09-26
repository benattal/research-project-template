# Noise models

The product identities in [independent-views.md](independent-views.md) need
two things from each view: its conditional mean is the clean signal, and its
noise is independent of every other view's noise given the clean signal.
This file records how common sensor corruptions meet those conditions,
which corruption-specific corrections are available and what they cost in
variance, how to manufacture independent views from a single measurement, and
how masks and partitions interact with normalization and conditioning.

## General

### Zero mean and conditional independence are all the polynomial identities need

An estimator built from products over distinct views is unbiased for any
corruption whose views have the clean signal as their conditional mean and
independent noise. It needs no noise model, no noise variance and no
knowledge of whether the noise depends on the signal.

**Why.** The cross terms in a product of views vanish because each factor's
noise has zero mean and the factors are independent (see the first entry of
[independent-views.md](independent-views.md)). Higher moments of the noise
never enter a product of distinct views.

**How to apply.** For each corruption, write down the per-view transform that
makes the conditional mean equal the clean signal (scaling counts by the
exposure, inverse-probability weighting for random deletion), and check
independence across views. Keep corruption-specific corrections, such as
Gaussian deconvolution, as labelled references rather than as the general
method. When a corruption is a known linear operator that does not have the
identity as its mean, see the last entry of this file.

**Evidence.** The same estimator was configured without changes for additive
Gaussian, Laplace and Rademacher noise, scaled Poisson counts and
inverse-probability Bernoulli deletion. On Poisson histograms at mild noise,
the same fitted-kernel estimator reached a sliced Wasserstein distance of
0.102 when trained on clean captures and 0.101 (three-seed mean) when trained
on independent noisy captures: the noise cost nothing, and the remaining gap
to the clean evaluation floor of 0.055 came from the kernel approximation.

## Photon counts

### Scale photon counts by the exposure, keep them as counts, and expect non-perturbative noise at low exposure

A photon count divided by its exposure is an unbiased estimate of the
intensity, and any affine map applied afterwards (for example to a symmetric
display range) preserves that. The noise variance equals the intensity
divided by the exposure, so it depends on the signal, which the identities
tolerate. At low exposure the noise is as large as the signal, and every
nonlinear correction degrades.

**Why.** For $N \sim \mathrm{Poisson}(\alpha x)$, the scaled count $N/\alpha$
has mean $x$ and variance $x / \alpha$. Independent exposures of the same
scene give independent counts, and distinct pixels of one exposure have
independent counts. The ratio of noise norm to signal norm scales as
$1 / \sqrt{\alpha \bar x}$, so a few photons per pixel puts the noise on the
same scale as the signal, where second-order expansions of nonlinear maps no
longer describe their bias (see [sources-of-bias.md](sources-of-bias.md)).

**How to apply.** Store integer counts and convert at use, so no value is
clipped at the top of a display range. Use the scaled count, not a
variance-stabilizing transform, as the view: such transforms are nonlinear and
break the conditional-mean condition. If a transform is needed for a
corruption-specific reference, state the transform and the noise model and
apply it identically to every control. When raw counts are available, binomial
thinning (keeping each photon independently with probability $p$) splits one
count into two independent Poisson counts at the same pixel, with means
$p \alpha x$ and $\alpha x (1 - p)$, each scaled by its own exposure to
estimate $x$; this is a standard identity that produces two views without the
coordinate mismatch of a spatial partition, though the work behind these notes
used spatial partitions instead.

**Evidence.** On face images at about 0.25 expected photons per unit
intensity, the 99th-percentile norm of noisy measurements was about 4.3
times that of clean images, so the noise dominated the norm. At about five
photons per pixel, a pretrained nonlinear encoder's mean output on noisy
views missed its clean output by 0.68 of the clean norm, and extrapolation in
the noise level plateaued 12 dB below the ceiling. On 2-D points, Poisson
counts at 2,400 photons per unit intensity gave a coordinate RMSE of 0.030,
matching additive Gaussian noise of standard deviation 0.03.

## Additive Gaussian noise with known variance

### Known Gaussian noise admits exact deconvolution of Gaussian kernels, at a variance cost that grows as the bandwidth approaches the noise

When the noise is Gaussian with known covariance, a Gaussian kernel of the
clean signal has an unbiased single-view estimator: evaluate a narrower
Gaussian on the noisy view and rescale. The estimator requires the target
bandwidth to exceed the noise standard deviation, and its variance grows
without bound as the two approach each other.

**Why.** For $n \sim \mathcal{N}(0, \sigma^2 I_d)$, window width $w > \sigma$
and $v^2 = w^2 - \sigma^2$, with $\omega_w(z) = \exp(-\lVert z\rVert^2 / 2 w^2)$,

$$
\mathbb{E}\big[\omega_v(z + n)\big]
= \Big(\frac{v^2}{w^2}\Big)^{d/2} \omega_w(z),
\qquad
\mathbb{E}\big[\omega_v(z + n)\,(z + n)\big]
= \Big(\frac{v^2}{w^2}\Big)^{d/2 + 1} \omega_w(z)\, z .
$$

Dividing by the prefactors gives unbiased estimates of the clean kernel mass
and of the kernel-weighted displacement from one capture. The prefactor
$\big(w^2 / v^2\big)^{d/2}$ multiplies the estimator's fluctuations, and it
diverges as $w \to \sigma$.

**How to apply.** Use this only for additive Gaussian noise of known
variance, never for photon counts, masking or partitioned views. Choose the
window at least twice the noise standard deviation, and check the variance
factor before training. Keep the corruption-agnostic estimator as the method
and this one as the known-noise reference that shows how much of the
noise-free result is recoverable.

**Evidence.** On a 2-D spiral with noise of two thirds of the kernel
bandwidth, the deconvolved Gaussian kernel reached total variation 0.149,
the clean oracle's 0.152, and beat both the three-view average (0.182) and a
single noisy view (0.216). The variance factor on the squared window
prefactor was 1.8 at a window of twice the noise and 1.4 at 2.5 times the
noise. Above noise equal to the kernel bandwidth, the clean-optimal kernel
could not be targeted at all.

## Random deletion (Bernoulli masking)

### Inverse-probability weighting is unbiased but expensive; observed-only residuals are exact when observed values are clean

Random deletion with keep probability $p$ can be made unbiased by dividing
kept values by $p$, at a variance cost proportional to the squared signal. When
the kept values carry no further noise, a cheaper route is to compare only on
observed coordinates and leave the rest to the model.

**Why.** For a keep indicator $m \sim \mathrm{Bernoulli}(p)$, the weighted view
$m x / p$ has mean $x$ and variance $x^2 (1 - p) / p$. The variance is largest
where the signal is largest and grows quickly as $p$ drops. Restricting a
residual to the observed set uses each observed value as it is, with no
rescaling, and when the observed values are noise-free there is nothing to
debias on those coordinates.

**How to apply.** For clean observed values, feed the model the raw masked
measurement together with its mask, restrict every loss and kernel residual
to observed coordinates, and use independent masks (or a partition of the
observed set) to build independent views. Reserve inverse-probability
weighting for cases where a full-coordinate unbiased view is required, and
check its variance against the other noise sources first.

**Evidence.** On 2-D points, inverse-probability deletion with keep
probability 0.995 already produced a coordinate RMSE of 0.031, the same as
additive Gaussian noise of standard deviation 0.03: deleting half a percent of
values cost as much as noise of about one data-grid cell.

### Normalize a cross-view product by the support it lives on, and match the normalizer to the estimator

A product of two residuals restricted to two independent masks is nonzero
only on their intersection. Normalizing it by one mask's size underestimates
the per-coordinate distance by the keep fraction. The correct normalizer
depends on the estimator: a plug-in on the average of several masked views
lives on their union, not their intersection.

**Why.** With independent keep masks of probability $1 - \rho$, the
intersection of two masks has expected fraction $\big(1 - \rho\big)^2$, one
mask has $1 - \rho$, and the union of three has $1 - \rho^3$. A distance
divided by the wrong fraction is scaled by the ratio of fractions; for a
bounded kernel that changes the kernel weights exponentially, and for
attraction alone it changes the balance against repulsion. A normalizer that
depends only on the masks, which are independent of the values, keeps the
estimate unbiased.

**How to apply.** Compute the realized support of each product (the
elementwise product of the masks it uses) and divide by its size. Keep the
normalizer path-specific: when an estimator changes (cross-view product,
single view, plug-in average), recompute which support it uses. Add a test
that feeds constant residuals through each path and checks that the
normalized distance equals the constant.

**Evidence.** Normalizing the cross-view product by one view's kept fraction
underestimated the attraction distance by a factor of 2, 4 and 8 at deletion
rates of 0.5, 0.75 and 0.875, while repulsion was normalized correctly, so
the two terms ran at different effective bandwidths. Switching to the
intersection fixed the cross-view path but was also applied to a plug-in path
that used the three-view average. There it inflated distances by about
$\sqrt{2}$ (the attraction kernel value at the first step fell from 0.65 to
0.54 while the repulsion kernel matched to seven decimals), bounded kernels
underflowed, repulsion dominated, and the field died within about 150 steps.

## Single measurements

### Disjoint subsets of one measurement are independent views when the noise is independent per pixel

When a sensor delivers one measurement, independent views can be made by
splitting its pixels into disjoint subsets. Photon counts, additive per-pixel
noise and per-pixel deletion all have independent noise across pixels, so the
subsets are independent given the scene.

**Why.** Independence across views follows from independence across pixels.
Two things break it: spatially correlated corruption (binning several pixels
into one reading, blur), which couples neighbouring pixels, and pooling, which
averages pixels across subsets. A spatial partition also means the views
observe different coordinates, so a product of residuals at paired pixels
estimates the cross-residual between neighbouring pixels; it equals the
squared distance where the clean signal and the model's residual vary slowly
at the pixel scale.

**How to apply.** Partition by selection: each view gathers its own pixels at
full resolution. Partition in units at least as large as the corruption's
correlation length (whole superpixels for binned sensors). Under deletion,
partition only the observed set and invalidate assigned-but-unobserved units.
Keep a held-out target subset disjoint from every input and conditioning
subset. Seed the partition and the noise from separate random streams.

**Evidence.** A single-measurement partition with a large input subset (75 to
87.5 percent of pixels) and a small held-out target (12.5 to 25 percent)
tracked the multi-capture reference epoch for epoch at about five photons per
pixel: reconstruction from one noisy input reached 27.1 dB against 26.9 for
the reference at a matched early epoch, and 29.3 against 28.8 at saturation.
A partition that shared a random seed with the frozen noise in its first epoch
was statistically dependent on that noise; offsetting one stream's seed
removes the dependence.

### Conditioning on a masked or partitioned input needs the mask and a consistent scale

A model conditioned on a sparse subset at training time and on the full
measurement at deployment sees two different input distributions. Missing
mask information and scale changes between the two produce plateaus that
look like limits of the objective.

**Why.** Deleted pixels filled with a constant are indistinguishable from real
pixels of that value unless the mask is an input. A map trained on inputs
with half the pixels removed learns to amplify, so a full-energy input lands
outside its training range. Dividing each patch by its kept fraction removes
the density dependence; the factor depends only on the mask, so any linearity
the method relies on is preserved.

**How to apply.** Feed the observation mask (and, for partitions, the
partition masks) as input channels. At deployment, set the masks to the true
observation mask, or to all ones for a full measurement, never to all ones
over an image whose deleted pixels are still filled with a constant. Normalize
input energy by the kept fraction, and sample the kept fraction over a range
during training so that dense inputs are in distribution.

**Evidence.** A single-measurement deletion run plateaued near 15 dB with gray
patches. The causes were an inference path that supplied an all-ones mask
while deleted pixels stayed gray, and a count-dependent input scale that
differed between training and inference. With a two-channel mask and the
scale removed it climbed past the plateau toward the multi-capture reference.
In a separate model trained on split inputs, full-energy inputs looked out of
distribution until per-patch energy normalization was added; with it,
conditioning on the full measurement scored above conditioning on the
training split (30.96 against 29.84 dB at an early epoch), as intended.

## Linear operators

### A known linear operator is recovered through its range; invert it at evaluation or restrict the claim

When a measurement is $A x$ plus zero-mean noise for a known operator $A$ that
is not the identity (a blur, a projection), the views have conditional mean
$A x$, not $x$. The identities then give unbiased estimates of quantities in
$A x$, and nothing more.

**Why.** $\mathbb{E}[\tilde y \mid x] = A x$, so products of views estimate
products of $A x$. Directions in the null space of $A$ are not observed at
all, and directions with small gain are observed with amplified noise after
any inversion.

**How to apply.** Either learn the distribution of $A x$ with the unbiased
objective and apply a regularized inverse of $A$ at evaluation, or build an
unbiased preconditioned view for the identifiable subspace and state that the
claim covers only that subspace. Do not describe results as recovering $x$
where $A$ has a nontrivial null space.

**Evidence.** For Gaussian blur, the generator learned the blurred clean
images with the same training objective and data path as the denoising tasks,
and a Tikhonov-regularized conjugate-gradient inverse of the known blur was
applied only at evaluation, with no change to training.
