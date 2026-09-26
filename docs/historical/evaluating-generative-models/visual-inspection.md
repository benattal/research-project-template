# Visual inspection layouts that expose failures

Images of samples catch failures that scalar metrics average away: a filled
interior, a bridge between modes, a missing segment, a collapse to one
output per condition. They also mislead when the layout implies something
that is not there. This file covers layouts that make failures visible and
comparisons fair across checkpoints and arms, and the display conventions
that keep a picture honest.

## Fix the generation noise across checkpoints

Render every checkpoint's grid from the same generator noise and the same
conditions, with a separate fixed seed for choosing which dataset items to
show.

**Why.** With fixed inputs, a change between two frames is a change in the
model. With fresh noise every frame, sample-to-sample variation looks like
training progress or regression.

**How to apply.** Seed the noise used for display grids from a constant,
independent of the training random state, so that displaying does not
perturb training. Use another constant seed to pick the displayed dataset
items. Keep both constant across arms so that the same column shows the same
condition everywhere.

**Evidence.** Matched visual comparisons across four method families used
the same eight displayed conditions, the same four draws per condition and
fixed seeds for every run, so a column could be compared across methods and
epochs directly.

## Show several draws per condition side by side

For conditional models, show at least two draws per condition next to each
other, and do not read variety across columns as diversity.

**Why.** Each column shows a different condition, so columns vary even for a
generator that returns one output per condition. Only draws that share a
condition show whether the generator samples.

**How to apply.** Lay out draws for one condition in adjacent rows or
cells, with a clean reference in the same column. Pair the grid with a
number for within-condition spread (see [image metrics](image-metrics.md)).

**Evidence.** A face-generation grid was read as "per-column variety,
collapse breaking" while the two draws in each column differed by 3 to
6/255 per pixel. The variety was across conditions.

## Do not imply a correspondence that does not exist

Overlay a generated sample only on references that it is supposed to match.
Show the model's inputs as the model received them.

**Why.** An overlay tells the reader that two curves or images are meant to
agree. An unconditional draw drawn over an arbitrary clean example looks
like a bad reconstruction of it. A display of the conditioning that pools,
upsamples or fills in values shows information the model never had.

**How to apply.** Show unconditional draws alone. Put clean and noisy
observations in their own rows. For masked or sparse conditioning, show the
observed values in place and leave the rest blank.

**Evidence.** Early 1D grids overlaid each unconditional draw with an
unrelated clean example and a curve estimated from a measurement. They were
meant as references but read as targets. They were replaced by draws alone,
with clean and noisy examples in separate rows.

## Use density views with a support overlay and a noise reference

For low-dimensional targets, render generated samples as a density image
from a large sample, with a toggle that overlays the true support, and keep
a panel that shows clean and noisy data on identical axes.

**Why.** Scatter plots saturate and hide density differences; a density
image with a compressive transfer function shows both thin structure and
faint mass. The support overlay makes off-support mass and missing segments
visible at a glance. The noise panel gives the scale against which a
blurred result should be judged.

**How to apply.** Render from converged EMA checkpoints with a large sample
(such as 200,000 points) on a fixed grid, with square-root intensity
compression. Color on-support and off-support mass differently in the
overlay. Draw the noise reference with the same axes, sample count and seed
as the clean panel.

**Evidence.** A tight but incomplete support, with inner spiral turns
missing, was first seen in a two-mode display of this kind (plain density,
and generated mass colored against the true support). On-support mass and
the fraction of support cells at half the mean density were then computed
from the same 200,000-sample densities to quantify it, so the picture and
the numbers describe the same samples.

## Show per-class and mixed grids at every checkpoint

For discrete conditions, save a grid for each class and a mixed grid at
every displayed checkpoint, with a selector to switch between them.

**Why.** A mixed grid shows overall quality; per-class grids show label
adherence and the worst class. A failure on one class, such as extra
components when conditioned on one, is easy to miss in a mixed grid.

**How to apply.** Write both grid types from the same checkpoint and seeds.
Keep the class selector independent of any other selector on the page, so
that filtering one view does not refit another.

**Evidence.** Per-class views and their peak counts showed that one method
produced a spurious second peak in about a third of one-component samples,
and that an averaging baseline produced the same number of peaks whatever
the label. Pooled metrics had shown neither.

## Pair live and EMA grids during training

During training, show grids from the live weights and from the EMA weights
side by side, with the same conditions and seeds, and label which is which.

**Why.** EMA weights lag the live weights by roughly the EMA horizon. Early
EMA grids can look broken while the live model is learning, and the reverse
can happen after a late instability.

**How to apply.** Save both grids at each displayed checkpoint and put them
on separate, labeled timelines. Use EMA for reported results; use live grids
to judge whether training is progressing.

**Evidence.** At step 2,000 of an ImageNet-32 run, live-weight grids showed
recognizable objects while EMA grids with decay .999 were brown and green
texture. Without the live grid the run would have been judged stalled.

## Keep trajectory pages truthful

When displaying diagnostics over training, show each epoch's own artifacts,
mark missing ones as unavailable, label run status explicitly, and never fill
an empty slot with another run.

**Why.** A page that substitutes the nearest available frame or an older run
shows a state that did not occur at the displayed epoch, and a reader
cannot tell. Absent slots are information: they say what has not run.

**How to apply.** Look up artifacts by exact epoch. Label every method slot
as pending, legacy, diagnostic, rejected or promoted. Keep baseline slots
visible, marked not applicable where a panel does not apply. Render the
epoch-zero frame with the same schedule multipliers as the first training
step.

**Evidence.** A diagnostic page showed the nearest stale geometry frame when
the selected epoch had none, and silently displayed legacy runs in slots
whose new runs had not finished. Both were replaced with exact-epoch lookup
and explicit status labels, and completed failures were labeled rejected
instead of showing their good final frames as promoted recipes. Epoch-zero
frames had been drawn with a multiplier of one while the first training
step used four.

## Name the visual rejection criteria

Write down the visual failures that reject a run regardless of its metrics,
and apply them to every arm.

**Why.** Without named criteria, visual judgment drifts toward the favored
method and becomes a tiebreaker used only when it helps.

**How to apply.** For 2D targets: filled interiors, bridges between modes,
missing turns or components, excess thickness, mass piled at corners or at
the output bound, and collapse on some seeds. For images: within-condition
duplicates, grain, off-manifold textures and memorized training examples.
Record which criterion rejected which run.

**Evidence.** A control run with SWD in the accepted range was rejected
because its samples filled the disk inside a spiral. A conditional hybrid
improved per-class SWD on all three seeds but was rejected when one seed's
grids showed wrong peak counts (peak-count TV rising from .084 to .262).

## Encode field direction and magnitude in separate channels

When plotting a vector field such as a velocity field, draw arrows of
fixed length for direction and use color for log magnitude.

**Why.** Arrows scaled by magnitude vanish where the field is small and
dominate where it is large, which hides structure in the weak
regions where problems begin.

**How to apply.** Normalize arrow length to a display constant, map
$\log_{10}$ of the magnitude to a color scale with a legend, and keep the
scale fixed across epochs.

**Evidence.** In toy progression frames, the population field was
invisible wherever its magnitude was small until arrows were drawn at fixed
length with log-magnitude color; small but structured fields then became
readable.

## Keep display-only transforms out of metrics

A transform applied to make images easier to compare (color alignment,
per-image normalization, cropping) must not feed any metric, and metrics that
need a matched target must be skipped where no target exists.

**Why.** Aligning each generated image's channel energy with the clean image
uses the answer, so a PSNR computed after alignment is flattered. For
unconditional samples there is no matching clean image, so a PSNR against
one is meaningless.

**How to apply.** Compute metrics on raw outputs before any display
transform, apply the transform only when the model is conditioned on a
specific target, and skip target-matched metrics for unconditional samples.

**Evidence.** In one image pipeline, per-channel energy alignment to the
clean image is applied only for signal-conditioned runs and only for
display; the printed PSNR is computed before alignment and is skipped for
unconditional samples.
