# Independent views and unbiased products

Many training signals are nonlinear in a clean quantity that is never
observed: a squared distance to the clean signal, a kernel weight, a
polynomial feature, a weight multiplied by a direction. When several noisy
observations ("views" or "captures") of the same clean signal are available
and their noise is zero-mean and independent, each of these quantities has an
unbiased estimator built from products over distinct views. This file covers
the identity, how to assign views to factors, how to cut variance without new
views, how the identity passes through linear maps, and where the estimate
must sit so that its unbiasedness reaches the parameter gradient.

Notation: $y$ is the clean signal, $\tilde y_i = y + n_i$ is view $i$, and
the noises $n_i$ satisfy $\mathbb{E}[n_i \mid y] = 0$ and are independent of
each other given $y$. $a$ is a query point, such as a model output, that does
not depend on the noise of any view it is compared with.

## The identity

### A product of independent unbiased factors is unbiased for the product

The inner product of two residuals against two independent views is an
unbiased estimate of the squared distance to the clean signal:

$$
\mathbb{E}\big[\langle a - \tilde y_1, a - \tilde y_2 \rangle \mid y\big]
= \lVert a - y\rVert^2 .
$$

**Why.** Expanding the product gives

$$
\langle a - \tilde y_1, a - \tilde y_2 \rangle
= \lVert a - y\rVert^2 - \langle n_1 + n_2, a - y \rangle + \langle n_1, n_2 \rangle .
$$

The middle term has zero mean because each noise has zero mean, and the last
term has zero mean because the two noises are independent. Nothing else about
the noise enters: its distribution, its variance, and whether it depends on
the signal are all irrelevant. The same argument covers any multilinear form
evaluated on distinct views, so a degree-$k$ polynomial in $y$ has an
unbiased estimator whenever $k$ independent views are available. A kernel
that is a polynomial in the squared distance, such as
$k(a, y) = 1 - \lVert a - y\rVert^2 / c$, inherits the property directly.

**How to apply.** Check two conditions for every view: its conditional mean
is the clean signal, and its noise is independent of every other view's noise
given the clean signal. Signal-dependent noise such as scaled photon counts
satisfies both. The query $a$ may depend on a third capture (for example the
model's conditioning input), but not on either of the two views in the
product.

**Evidence.** For a fixed trained encoder whose output passed through a
pretrained decoder, the cross-view loss minus the target's noise variance
reproduced the clean reconstruction loss (0.00234) to within 0.00003 at three
photon levels spanning a factor of 20 (0.00234, 0.00233, 0.00231). The
single-view loss minus the same variance grew with the noise (0.00236,
0.00243, 0.00272). The decoder is nonlinear, so the match also shows that its
curvature bias was negligible at that encoder's small noise gain (see
[sources-of-bias.md](sources-of-bias.md) for the case where it is not).

### Averaging views first is a V-statistic; averaging over distinct pairs is a U-statistic

Plugging the mean of $n$ views into a squared distance keeps the diagonal
terms, each of which carries the full noise energy. The plug-in is biased by
the noise energy divided by $n$, and more views shrink that bias without
removing it.

**Why.** For $\bar y = \frac{1}{n} \sum_i \tilde y_i$ and noise covariance
$\Sigma$ per view,

$$
\lVert a - \bar y\rVert^2
= \frac{1}{n^2} \sum_{i,j} \langle a - \tilde y_i, a - \tilde y_j \rangle ,
\qquad
\mathbb{E}\lVert a - \bar y\rVert^2 = \lVert a - y\rVert^2 + \frac{\mathrm{tr}\,\Sigma}{n} .
$$

The $i \ne j$ terms are unbiased by the identity above; the $n$ diagonal terms
each contribute $\mathrm{tr}\,\Sigma$. Averaging only over distinct pairs
(the U-statistic) drops them. A nonlinear kernel evaluated on the averaged
point, such as a Gaussian RBF, adds a further curvature bias on top of the
diagonal term.

**How to apply.** Average over all $n(n-1)/2$ distinct pairs of views. For a
degree-$k$ monomial, average over all $k$-subsets of views and symmetrize over
the assignment of views to factors. Keep the average-then-plug-in estimator
as a baseline; it is often strong at low noise (see
[bias-variance-trade-offs.md](bias-variance-trade-offs.md)).

**Evidence.** On 256-bin histograms corrupted by Poisson counts, generators
trained with an exact kernel on one view or on the three-view average
imprinted the noise into their samples: the samples had a ripply floor, with
a mean sample total variation of 7.9 to 9.5. An independent-view estimator (a
locally fitted kernel evaluated with U-statistic features) produced samples
with a flat floor and a mean total variation of 5.7, the same as clean
histograms.

### Exclude self pairs from batch sums, and measure what the diagonal costs

Kernel sums over a batch have diagonal terms too. A generated sample paired
with itself adds a kernel value of one and a zero direction; a data anchor
fitted from the same capture it is later evaluated against shares that
capture's noise. Both are $O(1/n)$ biases that vanish when the diagonal is
dropped.

**Why.** The repulsion mass of sample $i$ is an average of $k(x_i, x_j)$ over
other samples $j$. Including $j = i$ adds $k(x_i, x_i) = 1$ to a sum of $n - 1$
terms, which inflates the mass by about $\frac{1 - \bar k}{n}$ for mean kernel
value $\bar k$. For fitted local kernels, a data sample used both as a fit
anchor and as an evaluated point makes the fitted coefficients depend on the
noise of the point they are evaluated at.

**How to apply.** Mask the diagonal of generated-generated kernel matrices
(sum minus trace, divided by $n - 1$). For fitted data-side objects, fit on
one random half of the batch and evaluate on the other, then swap the halves,
so every sample meets anchors whose noise is independent of its own. When a
leave-one-out form is implemented through contracted sums, unit-test it
against brute-force masking.

**Evidence.** On a 2-D spiral at high noise, three variants of a fitted local
kernel over three seeds gave total variation 0.445 with two swapped halves,
0.460 with all 512 data points as anchors and the self term removed, and
0.466 with all anchors and the self term kept. At 512 anchors the self term
cost about 0.006, smaller than the seed spread of about 0.03; the swapped
halves remained the best of the three.

## Assigning views to factors

### Give every factor of a term its own capture

A term that multiplies $k$ noisy factors needs $k$ distinct captures. An
attraction term, a kernel weight multiplied by a direction toward the data,
therefore needs the kernel's views plus one more for the direction.

**Why.** The expectation of a product factorizes only when the factors are
independent given the clean signal:

$$
\mathbb{E}\big[\hat k \, (\tilde y_3 - a) \mid y\big]
= \mathbb{E}[\hat k \mid y] \, (y - a)
\quad \text{when } \tilde y_3 \text{ is independent of the views in } \hat k .
$$

Reusing view 1 as the direction for the kernel estimate
$\hat k = 1 - \langle a - \tilde y_1, a - \tilde y_2 \rangle / c$ adds
$\mathbb{E}[\hat k \, n_1] = \Sigma (a - y) / c$, so for isotropic noise of
variance $\sigma^2$ the attraction becomes

$$
\mathbb{E}\big[\hat k \, (\tilde y_1 - a)\big]
= \Big(k(a, y) - \frac{\sigma^2}{c}\Big) (y - a) .
$$

Every attraction weight shrinks by the noise variance over the kernel scale,
a bias that grows with the noise and with the kernel's locality.

**How to apply.** Write the role table before implementing: conditioning,
routing or fitting views, kernel-evaluation views, and direction views. Count
the captures each term consumes. A degree-$p$ polynomial kernel in squared
distance needs $2p$ evaluation views and one direction view; a degree-$d$
polynomial feature map needs $d$ views, plus one for the direction. Reject
configurations where a direction view is also a kernel view or where two
kernel views coincide (see
[measuring-bias-and-variance.md](measuring-bias-and-variance.md) on encoding
this as validation).

**Evidence.** The formula above is exact under the stated assumptions and gives
the size of the bias directly. Two implementation audits found the defect in
forms that are easy to miss: a guidance term that reused one view's
observation mask for both residuals of a cross-view product, and a fitted
per-image base that consumed the same spare view that a cross-view loss needed
as its second input. Both were found by listing capture roles, not by
metrics, which is why the role table comes first.

### Reuse views across averaged terms, never within a term

Averaging several unbiased terms is unbiased, so a view may appear in
different terms of an average as long as no single term uses it twice. This
allows averaging over all role assignments and all perfect matchings of the
available views at no cost in captures.

**Why.** Each term in the average is a product over distinct views, so each
has the right expectation. The terms are correlated through shared views,
which limits the variance reduction, but the reduction is still substantial
because each term's noise enters through different factors.

**How to apply.** With three views and three roles (routing, evaluation,
direction), average all six permutations. For a degree-$p$ product of pairwise
factors, average several distinct perfect matchings of the $2p$ views; one
matching is unbiased, several have lower variance. When the evaluation budget
exceeds the degree, average over the distinct matchings of the extra views
too.

**Evidence.** Averaging all six role permutations of three views roughly
halved the root-mean-square error of an evaluated kernel numerator against its
target, from $6.3 \times 10^{-5}$ to $3.0 \times 10^{-5}$ at the lowest noise
level and from $1.4 \times 10^{-4}$ to $8.0 \times 10^{-5}$ at twice that
noise, with no additional captures and no nonpositive averaged masses at the
lower level.

## Beyond raw coordinates

### Linear maps carry the identity; the map may depend on an independent capture

For a map $A$ that is affine in its input,
$\mathbb{E}[A(\tilde y) \mid y] = A(y)$, so
$\langle A(\tilde y_1) - z, A(\tilde y_2) - z \rangle$ is unbiased for
$\lVert A(y) - z\rVert^2$. The parameters of $A$ may be an arbitrary nonlinear function
of a separate capture, provided the same map instance is applied to both views.

**Why.** Conditional on its parameters, $A(\tilde y_i) = A(y) + W n_i$, and
the identity applies to the transformed noises $W n_1$ and $W n_2$, which are
still zero-mean and independent when $W$ is independent of both. Two different
maps $A_1$ and $A_2$ for the two views give
$\langle A_1(y) - z, A_2(y) - z \rangle$ in expectation, which is not a
squared norm and does not bound either residual. A composition of affine maps
is affine, so an encoder and a decoder that are each affine in their input
give a round trip whose cross-view loss is exact. Any nonlinear stage between
the view and the product breaks this (see
[sources-of-bias.md](sources-of-bias.md)).

**How to apply.** Keep every nonlinearity on the parameter side: a network may
read an independent conditioning capture and emit the weights and bias of a
per-sample affine map, and only that affine map touches the views. The
generator's random seed must not enter the map, because a seed-dependent map
differs between the two views. A rescaling that depends only on a known mask
(such as dividing a patch by its kept fraction) keeps the map linear. Test
linearity numerically at startup for every parameterization, and perturb any
zero-initialized branch first, since a branch that outputs zero passes a
linearity test trivially.

**Evidence.** A pair of learned per-sample affine maps (image to latent and
latent to image), trained only from noisy views at about five photons per
pixel, reconstructed the clean image at 33.9 dB, 1.8 dB below the ceiling of
the pretrained nonlinear autoencoder it approximated. Mapping into that
autoencoder's fixed latent space through a single affine map reached only
about 24 dB, because the fixed target carried the nonlinear encoder's noise
bias. A fixed affine feature chart applied identically to every view was
certified by the same argument on 2-D point clouds.

### A noisy target independent of the prediction gives the clean loss plus a constant

Training against an independent noisy target minimizes the clean-target loss
(the Noise2Noise argument). The requirement is that nothing on the prediction
path carries the target's noise.

**Why.** For a prediction $f$ that is independent of the target noise $n_t$,

$$
\mathbb{E}\lVert f - \tilde y_t\rVert^2 = \mathbb{E}\lVert f - y\rVert^2 + \mathbb{E}\lVert n_t\rVert^2 ,
$$

and the second term does not depend on the parameters. The same holds for a
held-out subset of pixels of one measurement when the noise is independent per
pixel (the Noise2Self and Neighbor2Neighbor constructions). If the input, the
conditioning, a fitted base, or a normalization constant sees the target
pixels, the prediction can fit their noise and the constant is no longer
constant.

**How to apply.** Trace every input to the prediction back to its capture. In
a model where a network reads a conditioning capture to produce the weights of
a map applied to a second input, keep the conditioning, the input and the
target mutually independent. The conditioning must also be independent of the
input: a weight network that sees the same input it transforms can route all
information through the bias and send the weights to zero, so the map stops
responding to its input. Monitor the distance between the map's outputs on
two independent views; a value collapsing toward zero is the signature.

**Evidence.** A per-image least-squares base chart, fitted each step toward an
encoding of the full measurement, gave the prediction access to the held-out
target pixels. Fitting one half of the input side toward the other half, both
disjoint from the target, removed the leak. With the three roles independent,
the chart's weight term carried more energy than its bias term (ratio 1.8 to
2.1) and swapping the input image under a fixed chart changed the output by
about a quarter of the signal range, confirming that the weights were in use.

### Keep the noisy estimate where the gradient is linear in it

An unbiased estimate gives an unbiased parameter gradient only if the
gradient is linear in it. Losses of the form "move the output toward a
stop-gradient target" meet this condition by construction.

**Why.** A kernel-based distribution-matching loss that moves each output
along an estimated velocity field is $\lVert f - \mathrm{sg}(f + V)\rVert^2$, where $\mathrm{sg}$
stops gradients and $V$ is the estimated velocity. Its gradient is
$-2 V^{\top} \nabla f$, linear in $V$, so an unbiased $V$ gives an unbiased
stochastic gradient and its noise averages out over steps. The Noise2Noise
loss has gradient $2 (f - \tilde y_t)^{\top} \nabla f$, linear in the target.
If the estimate is squared, divided by another estimate, clipped or logged
before it reaches the gradient, its unbiasedness does not carry through.

**How to apply.** Form the unbiased estimate, detach it, and use it only as a
target or as a linear coefficient on the model output. Put any normalization
that must be nonlinear into a separate, explicitly labelled denominator (see
[normalization-and-self-normalization.md](normalization-and-self-normalization.md)).
A per-step relative error well above one is acceptable for an unbiased target
because training averages it; the same error inside a nonlinearity is not.

**Evidence.** On a 2-D spiral with Gaussian noise of two thirds of the kernel
bandwidth, a single draw of a fitted-kernel velocity had a relative error of
1.6 to 2.0, while the mean of 200 draws had cosine 0.997 to the same
estimator's velocity on clean data. Trained with this target, the noisy run
matched its clean twin after 30 epochs (total variation 0.418 against 0.414).
