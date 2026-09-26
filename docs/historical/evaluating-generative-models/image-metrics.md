# Image metrics: FID and its companions

On images, the target distribution is known only through a finite dataset,
and distances are computed in the feature space of a pretrained network. The
Fréchet Inception Distance (FID) is the standard summary, but its value
depends on the sample count, the feature extractor, the preprocessing, the
reference statistics and the conditioning protocol, and it ranks some
failures in an order that does not match what a person sees. This file
covers how to make an FID number mean something and which companion
measurements catch what it misses.

## Sanity-gate the FID pipeline with bracketing references

Before scoring a model, score three reference sets against the same
reference statistics: held-out training images, validation images, and
uniform noise. The first two must be near zero and the third must be large.

**Why.** Every stage of the pipeline can be wrong without raising an error:
the image range, the resize, the channel order, the feature layer, the
reference file. A wrong stage usually moves all three references, so the
bracket catches it. It also gives the scale on which a model's value should
be read.

**How to apply.** Store the thresholds with the evaluator, for example
held-out training at most 5, validation between 0.1 and 15, and uniform
noise at least 250. Rerun the gate whenever the evaluator, the reference
statistics or the image resolution changes. Place every model value on the
line from the real-data value to the noise value, not only against other
models.

**Evidence.** On CelebA at 64 pixels, held-out training images scored 1.5,
validation images 1.7 and uniform noise about 293 against 50,000-image
training statistics. Against that bracket, models whose selected grids
showed recognizable faces scored 191 to 208, much closer to noise than to
real images.

## FID depends on the number of generated samples

FID is biased upward at small sample counts. Compare values only at the same
count, and label small-count values as trend readings.

**Why.** The Fréchet distance is computed from an estimated mean and
covariance in a 2,048-dimensional feature space. With a few thousand
samples the covariance estimate is noisy and the distance absorbs that noise
as a positive bias.

**How to apply.** Fix the count per table and put it in the header. Use a
small count (such as 5,000) for trends during training and the published
count (usually 50,000) for any number compared with the literature. Measure
the real-versus-real curve at both counts to see the size of the bias.

**Evidence.** Real ImageNet images scored FID 26.2 at 2,000 samples and 9.5
at 10,000 samples against the same reference statistics, with no model
involved.

## The feature extractor, preprocessing and protocol define the number

State the Inception weights, the resize filter, the reference statistics and
the conditioning protocol with every FID. Two FIDs computed with different
choices are different metrics.

**Why.** Several Inception weight sets are in use; the one shipped with
common vision libraries is not the one used for published numbers. Resizing
with and without antialiasing gives different features. The reference
statistics must be computed from images cropped and resized as in training.
For conditional models, the generated set is drawn under a distribution of
conditions, and that distribution changes the score.

**How to apply.** For published comparisons, use the released feature
network and reference statistics of the benchmark, with its resize. For
internal work, fix one pipeline and never mix numbers across pipelines. For
conditional generation, sample conditions the way the benchmark does: class
labels from the validation loader, or whole attribute rows from the data,
never independent draws from per-attribute marginals.

**Evidence.** A CelebA evaluator sampled each of the 40 binary attributes
independently from its marginal frequency. Correlated attributes then formed
combinations that never occur in the data, and the absolute FID was
inflated: a checkpoint logged at 203 under independent marginals
corresponded to about 176 under real attribute rows. Relative comparisons
under the same protocol stayed valid, which is why nothing looked wrong.
The reference statistics for the same dataset had to be rebuilt with the
training crop and resize.

## Inception features penalize grain more than blur

FID can rank a diverse but grainy generator worse than a collapsed but
smooth one. Read FID together with a diversity measure and the images.

**Why.** High-frequency noise moves Inception features far from the natural
image manifold; smooth, averaged images stay close to it. A generator that
produces one sharp-ish mean face per condition sits closer in feature
statistics than one that produces varied, noisy faces.

**How to apply.** Report FID next to within-condition diversity and a
sharpness measure. When a diverse run has a worse FID than a collapsed run,
check for grain before concluding that the diverse run is worse.

**Evidence.** Under one protocol, a collapsed generator with within-condition
pixel spread near 1/255 scored FID 150 to 185, while a diverse generator
with spread 26 to 42/255 and visible grain scored 276. A grainy run without
guidance scored 300, above the uniform-noise reference of 293.

## An FID gain can come from sharper mean samples

A falling FID does not show that diversity is improving. Measure the spread
between draws that share a condition and compare it with the spread between
distinct real images.

**Why.** For a collapsed conditional generator, the only way to lower FID
is to make its single output per condition look more realistic, which it
can keep doing without producing any variety. A grid that shows one sample
per condition then shows variety across columns, which is variety across
conditions.

**How to apply.** Generate two or more draws per condition with different
noise, report their mean absolute pixel difference next to the value for
two distinct real images, and look at the draws side by side. Treat
"diversity is improving" as a claim that needs this number.

**Evidence.** A conditional face model's FID fell from 177.6 to 170.0 to
140.7 over training, and the fall was read as collapse breaking, supported
by a grid with varied faces across its columns. The two draws per condition
differed by 3 to 6/255 throughout, against about 72/255 for two distinct
real faces. The FID gain came from sharper mean faces. The within-condition
difference measurement caught it once the reading was challenged.

## Separate a narrow manifold from memorization

Report k-nearest-neighbor precision and recall and an exact-duplicate rate
alongside FID. They distinguish "realistic but narrow", "broad but
unrealistic" and "copies".

**Why.** FID mixes fidelity and coverage into one number. Precision asks
how many generated samples fall inside the real manifold estimate; recall
asks how much of the real manifold the generated samples cover. A duplicate
rate over hashed, quantized images checks for exact repetition, though it
misses near-duplicates.

**How to apply.** Use a fixed k (for example 3) and fixed sample counts (for
example 5,000) for precision and recall. Report KID, the unbiased
polynomial-kernel MMD averaged over subsets, with its standard error when
sample counts are small. Read a high precision with near-zero recall as
collapse onto a small region.

**Evidence.** An early EMA checkpoint of a class-conditional ImageNet-32
model had FID 187, KID .17, precision .96, recall .002 and a duplicate rate
of zero. Its grids showed brown and green texture. The samples fell inside
the k-nearest-neighbor estimate of the real manifold while covering almost
none of it, so high precision alone did not mean realistic images.

## Compare class consistency with chance and with real data

For class-conditional models, classify the generated images with a
pretrained classifier and report how often the prediction matches the
conditioning label. Place the value between chance and the classifier's
accuracy on real images at the same resolution.

**Why.** A generator can produce realistic samples that ignore the label.
FID computed over all classes pooled does not see this.

**How to apply.** Report class consistency with its chance level (one over
the number of classes) and the real-image value. Pair it with intra-class
diversity so that a generator cannot score well by emitting one prototype
per class.

**Evidence.** The same ImageNet-32 checkpoint had class consistency .0011,
at chance for 1,000 classes, while its precision was .96.

## Sweep inference guidance before declaring a plateau

For models trained with classifier-free guidance, sweep the inference
guidance scale on a fixed checkpoint before judging a recipe. Report the
scale with every number and do not extrapolate far beyond the trained range.

**Why.** Guidance trades diversity for fidelity at inference time, and the
best scale changes over training. A recipe judged at one scale can look
stalled when it is not, or improved when only the scale changed. Scales
outside the range seen in training produce inputs the network was not
trained on.

**How to apply.** Sweep a handful of scales on the EMA checkpoint with a
fixed sample count and seed. Report the best scale and the whole curve. When
the curve is flat, guidance is not the lever.

**Evidence.** For a model trained with guidance in the range 1 to 4, FID was
406, 368, 297, 276, 299, 307 and 311 at scales 1, 2, 3, 4, 5, 6 and 8:
extrapolation past the trained maximum hurt. For another model the sweep
was flat between 203 and 208, which ruled out over-guidance as the cause of
its quality ceiling.
