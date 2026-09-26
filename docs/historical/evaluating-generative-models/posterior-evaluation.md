# Evaluating conditional and posterior samplers

A conditional sampler must produce the right distribution for each
condition, not only good samples on average. For a measurement-conditioned
(posterior) sampler this means the right mean, the right spread and the
right mode probabilities for each measurement. For a sampler conditioned on
discrete labels it means the right distribution per label. Pooled metrics
and reconstruction scores such as PSNR miss most of these failures. This
file covers how to build an exact posterior to test against, which
per-condition quantities separate the failure modes, and why reconstruction
accuracy rewards the wrong behavior.

## Build an exact posterior with a finite catalogue

For a synthetic problem, draw a large finite catalogue of clean signals from
the prior and compute the exact posterior over that catalogue for each
measurement by Bayes' rule. Score the sampler against this posterior.

**Why.** With a uniform prior over $M$ catalogue atoms $x_m$ and a known
likelihood, the posterior weights are

$$w_m(y) = \frac{p(y \mid x_m)}{\sum_{j=1}^{M} p(y \mid x_j)},$$

computed stably as a softmax of log-likelihoods. Every posterior moment and
every credible set is then exact up to the catalogue approximation, which
can be checked through the effective sample size
$\mathrm{ESS} = 1 / \sum_m w_m^2$. With a posterior of this kind, a failure
can be attributed to mean, spread or mode weights for each measurement,
which is impossible when the only reference is one clean signal per
measurement.

**How to apply.** Use the same forward model and noise law as in training.
Draw the catalogue once with a fixed seed and keep it. Report the mean and
minimum ESS over conditions: a minimum near one means some posteriors sit on
a single atom, where spread-based metrics are undefined. When the prior has
discrete structure, such as a component count, restrict the catalogue to
atoms compatible with the condition. Score an untrained checkpoint with the
same evaluator as a reference for "nothing learned".

**Evidence.** A finite-catalogue evaluator of this kind, built for 1D
Poisson-count mixture signals, reports per-condition sliced Wasserstein
distance to catalogue draws, mean and variance errors, a diversity ratio,
credible-set coverage and mode-probability error. Three early training
attempts produced posterior metrics identical to those of the untrained
model, which showed at once that the training signal had been zero from the
first epoch.

## Separate mean shift from dispersion error per condition

Score the posterior mean and the posterior variance separately for each
condition, and compare the sampler's spread with the exact spread through a
ratio. Report where the ratio is undefined.

**Why.** A sampler can match the posterior mean while having the wrong
spread (a mean predictor has zero spread), or match the spread around the
wrong center. A single distance mixes both. The ratio of generated to exact
spread is above one for over-dispersion and below one for under-dispersion,
and it is not defined where the exact posterior is a point.

**How to apply.** For each condition, generate a fixed number of samples
and compute the root-mean-square error of their mean against the exact
posterior mean, the same for their variance, and the ratio of mean
distances between consecutive samples, generated over exact. Skip
conditions whose exact spread is below a small floor, and report the
fraction of conditions where the ratio is defined together with its mean,
minimum and maximum.

**Evidence.** In a smoke evaluation whose conditions all had ESS between
1.00 and 1.01, the exact spread was tiny but above a floor of $10^{-6}$, and
the diversity ratio ranged from 7 to about $6 \times 10^{7}$. The floor
must sit on the scale of the signal, or conditions must be excluded by ESS,
for the ratio to mean anything. The two failures this entry separates, a
mean predictor with no spread and a noisy reconstruction with too much,
both occurred on images (see the entries below on PSNR and on diversity).

## Check credible-set coverage and mode probabilities together

Assign each generated sample to its nearest catalogue atom and report the
fraction of samples that fall inside the exact 90% credible set. Report the
total variation between generated and exact probabilities of discrete modes
beside it.

**Why.** If the sampler is calibrated, about 90% of its samples fall in the
90% credible set (slightly more, because the set of atoms holds at least
90% of the mass). Coverage well above .9 means the samples concentrate on
the highest-weight atoms: under-dispersion. Coverage below .9 means the
samples spread onto low-probability atoms or sit away from the posterior:
over-dispersion or a shift. A sampler that always returns the most probable
atom scores coverage 1.0, so coverage alone cannot certify calibration; the
mode-probability error catches that case.

**How to apply.** Assign samples in a fixed low-dimensional random
projection when signals are high-dimensional, with the projection seeded
and shared across arms. Build the credible set by sorting the exact weights
and taking atoms until the cumulative mass reaches .9. Report the mean and
minimum coverage over conditions.

**Evidence.** In a smoke evaluation after one epoch of training, coverage
was 0 on every condition and the mode-probability TV was .87: the samples
sat away from every high-weight atom. The two failures the pair is built to
separate both appeared in image experiments: a conditional sampler that
returned nearly the same image for every noise draw (coverage near one by
the definition above), and a linear reconstruction whose draws differed
only by amplified measurement noise (coverage below .9).

## PSNR to the ground truth rewards the posterior mean

Do not rank posterior samplers by PSNR or mean squared error against the
clean signal. A sampler that returns the posterior mean scores best on both,
and a calibrated sampler scores about 3 dB worse by construction.

**Why.** Let $\mu$ be the posterior mean and $\mathrm{MMSE}$ the posterior
variance summed over pixels. A calibrated sample $s$ and the true signal $x$
are independent draws from the same posterior, so

$$\mathbb{E}\lVert s - x\rVert^2 = 2\,\mathrm{MMSE}.$$

The mean of $N$ calibrated draws has error $\mathrm{MMSE}\,(1 + 1/N)$, so
the PSNR of that mean exceeds the PSNR of one draw by

$$10\log_{10}\frac{2N}{N+1}\ \text{dB},$$

which is 1.25 dB at $N = 2$ and approaches 3 dB as $N$ grows. A sampler that
collapses to the mean gains those 3 dB and loses all diversity.

**How to apply.** Score posterior samplers with per-condition diversity, a
sharpness measure and a distribution metric (FID on images, the exact
posterior on synthetic data). Report PSNR only for the mean of many draws,
as an estimate of the MMSE, and report the gap between single-draw PSNR and
mean-of-draws PSNR as a calibration check.

**Evidence.** A measurement-conditioned image sampler reached about 30 dB
PSNR, and two draws with different noise differed by a mean of 1.56/255 per
pixel, against an image standard deviation of about 70/255: 99% identical.
Its training targets had real spread across views, so the sampler had
collapsed to the conditional mean despite diverse targets.

## Diversity is not calibration: check sharpness and the mean gap

Draw-to-draw variation can come from amplified measurement noise rather
than from the prior. Compare each draw's high-frequency energy with that of
clean images, and compare the single-draw and mean-of-draws PSNR with the
3 dB expected of a calibrated sampler.

**Why.** A posterior sample should look like a clean image: its
high-frequency energy should match clean data. Noise passed through the
reconstruction adds high-frequency energy, and averaging draws removes it,
so the mean of noisy draws is much better than any single draw.

**How to apply.** Report three numbers per sampler: the ratio of per-draw
high-frequency energy to that of clean images (near one when calibrated),
the mean pixel difference between draws, and the PSNR gain from averaging
draws. A gain well below the calibrated value means under-dispersion; a gain
well above 3 dB means the variation is noise.

**Evidence.** Two linear reconstructions of the same measurements were
compared this way. The plain one had a sharpness ratio of .77, draw-to-draw
difference .018 and an averaging gain of +0.6 dB (30.97 to 31.60 dB): a
smoothed, nearly deterministic posterior mean. A debiased one had a
sharpness ratio of 3.93, draw-to-draw difference .219 and an averaging gain
of +9.7 dB (17.14 to 26.85 dB): four times too sharp, with its variation
coming from amplified measurement noise. The first gain is below the
1.25 dB a calibrated sampler gives even for two draws; the second is far
above 3 dB. Neither was sampling the posterior. A reconstruction linear in
the measurement can only vary by linear combinations of it, so it cannot
draw from a multimodal prior.

## Verify the diversity measurement before interpreting it

Before concluding that a sampler has collapsed or is diverse, confirm that
the draws being compared used different noise, and compare their spread with
a reference that has a known answer.

**Why.** A batching or seeding error that reuses the noise makes a healthy
sampler look collapsed. Without a reference value, a spread of any size has
no meaning.

**How to apply.** Check in the generation code that draws for one
condition receive distinct noise. Report the draw spread next to the spread
of two distinct real images and, for synthetic data, next to the exact
posterior spread. Choose a test regime whose posterior is wide enough for
diversity to be measurable: lower measurement signal-to-noise ratio or
heavier masking widens the posterior.

**Evidence.** In the collapse above, the generation path was checked to
draw distinct noise for the two draws per condition before the collapse was
reported. A second sampler, trained on Poisson-count measurements, reached
about 21 dB with a draw-to-draw difference of 2.42/255: also collapsed,
although its lower PSNR left room for a wider posterior.

## For discrete conditions, score each class and check label adherence

With a discrete condition (class, attribute, component count), compute the
distances per class and report their mean and the worst class. Measure
label adherence with a statistic that should change with the label.

**Why.** Pooled distances average over classes, so a method that is good on
most classes and bad on one looks fine. A sampler that ignores the label and
reproduces the pooled distribution can score reasonably on pooled metrics.

**How to apply.** Generate the same number of samples per class. Report
per-class distances, their mean and their maximum. Pick an adherence
statistic tied to the label, such as the number of detected components when
conditioning on a component count, and compare it per label with the clean
data.

**Evidence.** On 1D mixtures conditioned on the number of components, a
wide kernel gave the best pooled SWD (.027) while its one-component class
degraded (.081 against .062 at the width that was best per class). Mean
detected peak counts for clean signals were 1.00, 1.54, 1.90 and 2.18 for
one to four components. An averaging baseline produced 1.33 to 1.42 for
every label at one noise level and 2.06 to 2.10 at another: it ignored the
label, which pooled SWD had not shown. The proposed method tracked the
label, with an offset of +.25 to +.3 from spurious small peaks.
