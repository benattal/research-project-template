# Choosing metrics that expose the failure

A generator can miss its target distribution in several distinct ways: it
can drop modes, shrink or inflate the spread, shift the mean, fill the space
between thin structures, bridge separated components, or put mass off the
support. Each metric is sensitive to some of these and blind to others. This
file covers how to choose a set of metrics whose blind spots do not overlap,
how to read them when they disagree, and which popular quantities do not
measure distribution recovery at all.

## Decide on a family of population distances, not one scalar

Rank arms with several distribution-to-distribution distances computed on
the same generated and reference samples: sliced Wasserstein distance (SWD),
energy distance, RBF maximum mean discrepancy (MMD), and the Fréchet
distance between Gaussian fits. Promote an arm only when the family agrees,
and report disagreements instead of quoting the favorable column.

**Why.** Each distance weighs errors differently. SWD averages
one-dimensional transport costs over random projections, so it responds to
where the bulk of the mass sits. Energy distance,

$$E = 2\,\mathbb{E}\lVert X - Y\rVert- \mathbb{E}\lVert X - X'\rVert- \mathbb{E}\lVert Y - Y'\rVert,$$

responds to all scales but is dominated by large separations. MMD with a
median-heuristic bandwidth is most sensitive near the typical pairwise
distance and weak on much finer structure. The Fréchet distance compares
only means and covariances: it catches a mean shift or a variance change,
and it cannot see mode dropping that preserves the first two moments. A
change can help one of these and hurt another.

**How to apply.** Compute all four with fixed seeds on identical sample
sets. Write the decision rule before the runs, for example "better on at
least three of four and no visual regression". Keep binned divergences,
occupancy and per-sample scores as secondary diagnostics that explain a
result, not as rankings.

**Evidence.** On a ring-shaped target, doubling a kernel width improved
support occupancy from .456 to .690 and grid total variation from .645 to
.491, while all four population distances worsened (SWD .0157 to .0197, MMD
.00025 to .00069); the samples stayed visibly too thin. Across a nine-scene
suite, a longer training schedule improved mean SWD, energy and Fréchet
distance but worsened mean MMD from .00189 to .00229. On 1D mixture data,
one locality schedule improved a parameter-space MMD while worsening all
four signal-space distances; no schedule won on every distance.

## Binned divergences and transport distances rank failures differently

Total variation (TV) or Jensen-Shannon divergence between histograms charges
every misplaced unit of mass the same, however far it moved. Transport
distances charge by the distance moved. A method that concentrates its mass
on a thin support but drops or unbalances whole segments can win on binned
divergence and lose on transport distance. A method that blurs slightly off
the support can do the reverse.

**Why.** For reference and generated histograms $p$ and $q$ over cells
$c$, $\mathrm{TV} = \frac{1}{2}\sum_c \lvert p_c - q_c\rvert$. Mass moved
one cell off a curve and mass moved to the far end of the curve cost the
same. Under a
Wasserstein distance the first costs one cell width and the second the
length of the curve. Neither ranking is wrong: they measure different
failures, and the evaluation has to decide which failure matters.

**How to apply.** Report both kinds. When a transport distance worsens while
a binned divergence improves, look for mass imbalance or a missing segment.
Add support diagnostics that separate the two routes to the same score: the
fraction of mass on the support, the fraction of support cells reaching half
the mean density, and the coefficient of variation of density along the
support.

**Evidence.** At one noise level on a spiral, a proposed method and an
averaging baseline had nearly equal grid TV (.335 to .355 against .336). The
proposed method kept 90% of its mass on the support but left 30 to 37% of
support cells below half the mean density. The baseline kept only 68% of its
mass on the support but reached half density in every support cell. The
first dropped segments; the second blurred.

## Re-score past decisions when the primary metrics change

When a new metric family becomes primary, re-evaluate every converged
checkpoint behind an earlier decision. A ranking made under the old metric
can reverse.

**Why.** Decisions compound. A method that wins becomes the default, takes
the tuning budget, and becomes the reference for later arms. If the win came
from a metric that is blind to that method's failure mode, every later result
inherits the error.

**How to apply.** Keep the converged checkpoints of all arms, not only the
winner, so that re-scoring costs an evaluation rather than a retraining.
State a reversal at the top of the report, where a reader meets it before the
superseded tables.

**Evidence.** A compact-support kernel beat the previous method on grid TV on
a noisy spiral across three seeds (mean .343 against .375) and was adopted as
the method of record. Re-scoring the converged checkpoints with the four
population distances ranked it last of four arms: SWD .022 to .028, against
.015 to .019 for the previous method, .0098 for a simple averaging baseline
and .0057 for a known-noise reference. It dropped or unbalanced mass along
the curve, which grid TV and visual thinness had rewarded. Adding the
population distances and re-scoring old checkpoints is what caught it.

## Coverage cannot redeem a filled or bridged distribution

Occupancy (the fraction of support cells receiving at least one sample) and
the mean distance to the support both reward spreading mass. A distribution
that fills the inside of a spiral or bridges separated modes can score well
on both. Use the tail of the distance-to-support distribution and the
fraction of samples within one cell of the support, and let the image veto.

**Why.** Occupancy saturates once every support cell has one sample, which a
broad blob achieves. Mean distance averages a small far tail with a large
near bulk, and a thin fill sits a short distance from the support, so it
barely moves the mean.

**How to apply.** Record the median, 90th and 99th percentiles of distance to
the support and the within-one-cell fraction. Promote only when these agree
with the density image. Name the visual failures that reject a run whatever
its metrics: filled interiors, bridges between components, missing turns or
components, excess thickness, mass piled at corners or at the output bound,
and collapse on some seeds.

**Evidence.** A numerically stable control reached SWD .0245 on the spiral,
in the range of accepted runs, while its samples filled the whole disk inside
the spiral; it was rejected on the image. A cyclic schedule improved
checkerboard SWD, MMD and Fréchet distance while occupancy fell and visible
paths between the squares remained; it was not promoted.

## Per-sample validity is not population recovery

Scores computed on each sample alone (distance to the nearest training
example, residual of a fit to the known sample family, number of peaks) say
whether individual samples look plausible. They do not say whether the
samples occur with the right frequencies, widths and scales.

**Why.** A generator can produce only plausible samples of the wrong
mixture: too many of one kind, all too wide, or all at the wrong amplitude.
A per-sample score averages over the generated set and never compares it
with the reference population.

**How to apply.** Use per-sample scores to explain a population-distance
result. Do not promote on them.

**Evidence.** On 1D Gaussian-mixture histograms, a structured initialization
improved peak-count TV from .289 to .078 and the per-sample mixture-fit
residual from .155 to .0066, while signal-space SWD worsened from .070 to
.192, energy distance from .080 to .779 and Fréchet distance from 5.3 to
33.3. Its samples carried 44% more total mass than clean signals (140.8
against 97.9). In a separate low-signal screen, the setting with the lowest
per-sample residual (.066) lost on energy distance, MMD and every
parameter-space distance.

## Measure in a representation that exposes the structure

When samples have known generative structure, compare distributions of
fitted structural parameters as well as raw signals. For mixture-shaped
signals, fit each sample, sort its components by location, and compute the
population distances over the vector of locations, widths and weights,
matched within component count.

**Why.** Raw-signal distances in high dimension respond to where most of the
energy sits. They can improve while the number, width and weight of
components drift. Parameter-space distances make those quantities the
coordinates. Sorting removes label switching, and matching within component
count keeps the vectors the same length.

**How to apply.** Report the signal-space and parameter-space families side
by side, per condition. Keep the per-sample fit residual as a separate shape
diagnostic. A fitter that may choose its own component count can spend spare
components on clutter and hide artifacts; fixing the count to the
conditioned value is stricter, and residuals from the two settings are not
comparable.

**Evidence.** A variant beat two proxy baselines on per-class signal SWD
(.126 against .140 and .157) and on sample shape, but lost to both on
parameter-space Fréchet distance (.079 against .054 and .038). It was not
counted as a recovery win. Residuals with a free component count understated
low-amplitude ripple that fixed-count residuals exposed.

## Validate a custom metric on clean data before scoring models

A hand-built metric (a peak counter, a mode detector, a fitted-model
residual) needs its clean-versus-clean value before it scores anything. A
metric that is large between two clean sets measures its own artifacts.

**Why.** Detectors respond to features at scales they were not designed for:
pixel-scale ripple, quantization, degenerate fits.

**How to apply.** Compute the metric between two independent clean draws of
the evaluation size and report that floor beside every model value. Smooth
below the structure scale and require minimum prominence and separation in
peak detectors.

**Evidence.** A first peak counter counted pixel-scale ripple as peaks. Its
replacement smooths below the component width and requires relative
prominence and a minimum separation; its clean-versus-clean peak-count TV is
.016, against .22 for a rejected model.

## Fit accuracy of a pipeline component is not a proxy for recovery

A fitted or learned part of the training pipeline (a normalizer, a local
kernel fit, a cache) can reach a low fitting error while the generated
distribution gets worse. Score the generator.

**Why.** The component fits its target on the inputs it sees. Its errors
matter only where they change the generator's updates, often in rare regions
such as low-density areas that a global error average weights lightly.

**How to apply.** Treat component errors as monitors. Promote a component
change only on the population distances and images of the trained generator
at a matched budget.

**Evidence.** A learned normalizer matched its targets with RMSE .0047 and a
grid correlation of .98, yet checkerboard SWD rose from .037 to .052 and
Fréchet distance from .0019 to .0074, with visibly worse paths. Another
learned normalizer reached a training loss of .0006 and still worsened
recovery. The scalar fit error of a local kernel improved steadily through
two episodes in which the generated population collapsed and recovered.

## Measure diversity directly and calibrate it

When collapse is a risk, measure the spread between samples that share a
condition, for example the mean absolute pixel difference between two draws
with different noise, and compare it with the same quantity for two distinct
real samples. Quality metrics can improve while this spread stays near zero.

**Why.** Distribution distances computed in a feature space, and visual
impressions of a grid with one sample per condition, respond to per-sample
quality and to variety across conditions. Neither isolates variety within a
condition.

**How to apply.** Generate at least two draws per condition with verified
different noise and report their spread next to the real-pair reference.
Pair it with a sharpness measure, since grain and noise also raise pixel
spread. The [image metrics](image-metrics.md) and
[posterior evaluation](posterior-evaluation.md) notes give the details.

**Evidence.** While an image model's FID fell from 178 to 141, two draws with
the same condition differed by 3 to 6 out of 255 per pixel, against about 72
for two distinct real faces. A grainy model scored 55 to 87 on the same
measure, above the real-pair reference, because grain counts as spread.
