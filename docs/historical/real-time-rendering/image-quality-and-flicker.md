# Image quality and flicker

A cache trained online is judged by the frames it displays, and a displayed
frame mixes three errors: the learned field's bias, the Monte Carlo noise of
the camera paths that read it, and the change of the field from one frame to
the next. They call for different fixes, and a single image metric hides
which one dominates. These notes record how to separate them and what
reduced each.

## Separate camera noise from cache error before changing the cache

**Insight.** At display sample counts, the camera's own sampling noise can be
larger than the learned field's error.

**Mechanism.** The camera path runs through glass and mirrors to the first
diffuse surface before it reads the cache. Each reflect-or-refract choice at a
dielectric is a binary event, so a few samples per pixel give large noise
that no change to the cache can remove.

**How to apply.** Measure the display noise of the camera estimator (the RMS
difference between frames rendered from the same frozen field with fresh
camera samples) and compare it with the converged field's error against the
reference. Accumulate the camera prefix and the cache read into separate
diagnostic films from the same samples, so their sum equals the ordinary film
byte for byte and each part can be scored.

**Evidence.** On a glass-box scene at 32 samples per frame, camera noise was
.025 in display RMS while the converged field's error to the reference was
.009.

## Stratify the camera with padded, scrambled Sobol

**Insight.** Padded Owen-scrambled Sobol sampling over each frame's camera
samples cut display noise by a third to two thirds without bias.

**Mechanism.** Stratifying the pixel jitter and the first dielectric choice
alone cut noise by 24%. Adding more dimensions to one lattice did not help,
because the lattice's two-dimensional projections weaken. Padding instead
treats the jitter as one 2D Sobol pair and each of the first eight
dielectric choices as its own independently scrambled 1D dimension, so every
choice is stratified on its own.

**How to apply.** Draw the jitter and the first few discrete choices from a
padded scrambled sequence, and make sure the choices draw from that sampler
rather than a side stream. Index the sequence over the film's accumulation
window, which restarts after every update and camera change, so each frame's
samples form a complete net; seed the scrambling per pixel and per window.
Check the bias at a high sample count against the unstratified estimator.

**Evidence.** At 32 samples on a glass-box scene, display RMS fell from .0309
to .0162 (48%), and from .0933 to .0305 on a floor seen through four or more
glass events; pixels off by more than .1 fell from 5,518 to 1,219. Two other
glass scenes gained 37% and 29% at 16 samples. The 4,096-sample mean was
unchanged to four digits. A rank-1 lattice over five dimensions tied the
three-dimensional one.

## Do not clamp camera samples when no sample is extreme

**Insight.** A clamp on camera sample values helps only when outliers exist.

**Mechanism.** When no camera sample exceeds a small value, any clamp low
enough to change the image removes real highlights.

**How to apply.** Histogram the camera sample values, and split the camera
estimate into its terms, before adding a clamp.

**Evidence.** No sample exceeded 8, so clamps at 16 and 8 changed nothing and
a clamp at 4 raised the error from .045 to .091. The visible sparkle was
partial coverage by moderate samples: area-light hits through reflect and
refract chains on a glass rim, which held 91 to 98% of the squared error in
pixels beyond 5σ.

## Measure flicker in two bands, per frame and per second

**Insight.** Flicker has a fine part (pixel noise) and a coarse part (the low
band where caustics boil), and what a viewer perceives is the change per
second, not per frame.

**Mechanism.** The same change per frame looks worse at a higher frame rate,
because it is redrawn more often. A faster page can therefore flicker more
than a slower one with identical fields.

**How to apply.** Compare consecutive distinct canvas frames on the canvas
crop, in 8-bit display luma (duplicates held in a 60 Hz file would read as
zero change). Measure the RMS difference (fine) and the mean absolute
difference after a 6-pixel blur (coarse). Report both per frame and per
second, in bins of about 15 s, separately for each phase (moving geometry,
still while training, frozen), and compare with an accepted earlier clip
before recording a new recipe.

**Evidence.** Two recipes for a water-caustic scene had the same coarse change
per frame (1.26 and 1.27) but ran at 6.3 and 23.7 frames per second, and the
faster one was seen to flicker much more. An orbit clip at 64 samples and
6 FPS changed by 7.7 coarse levels per second; at 16 samples and 18 FPS it
changed by 15.4, and halving the orbit speed brought it back to 7.4 to 9.

## Split frame differences into camera noise and field churn

**Insight.** With a fixed camera, re-render with the same camera samples to
separate the noise of new camera samples from the change of the field.

**Evidence.** On a water-caustic scene (fine / coarse, in display levels):

| Frame difference | Fine | Coarse |
| --- | ---: | ---: |
| camera noise alone (next 16 samples, same field) | 3.57 | .122 |
| field after 1 update, same samples | .99 | .848 |
| field after 16 updates (one frame), same samples | 2.11 | 1.074 |

Camera noise is fine-grained; field churn is coarse. An orbiting camera draws
a fresh film every frame, so its frame differences always include camera
noise.

## Display the averaged weights, and choose the window for flicker

**Insight.** The weight average (EMA) that the display reads sets how much the
image churns between frames; the learning rate matters much less.

**Mechanism.** The optimizer's latest iterate jitters around the target. An
EMA with a short window (.5) shows that jitter directly, and caustics appear
to boil. A longer window averages it out at the cost of some lag. The
optimizer updates the live weights; the display and any self-training reads
use the EMA.

**How to apply.** Read displayed frames and training targets from the EMA
weights and update them after each optimizer step. Size the window to cover
at least one frame's worth of updates. When lengthening it for a moving
scene, raise Adam's β2 as well.

**Evidence.** On a moving water caustic at 10× the base learning rate, the
coarse change over a 16-update frame was 1.66 at EMA .5, .96 at .8, .75 at .9
and .54 at .95 with β2 .999. A 3× lower learning rate at EMA .5 still churned
1.18. EMA .95 (a window of about 20 updates) lagged a 16-update frame by about
one frame and cost .008 in tracking error at one pose; with β2 .999 it stayed
within about .003 of the EMA .5 fits, where β2 .99 was .005 to .009 worse. In
the recorded clip at the same 18.8 FPS, coarse change per second fell from
30.6 to 14.6 while the surface moved and from 23.5 to 7.4 while it was still.

## More updates per frame trade frame rate for less churn per second

**Insight.** Putting more updates in each frame gives fewer frames that each
change more, and less change per second.

**Evidence.** On the water scene, 16 updates per frame at 33 FPS churned .33
coarse per frame and 10.7 per second; 32 updates per frame at 17 FPS churned
.54 per frame and 9.1 per second, and tracked slightly better. Halving the
learning rate churned least but tracked 5% worse.

## Forcing a discrete camera branch is a labelled visualization

**Insight.** The random reflect-or-refract choice at a dielectric the camera
sees through is a major source of grain; forcing one branch removes it but
changes what the image shows.

**How to apply.** Keep supervision physical, apply the forced branch only to
camera paths (keeping the transmitted Fresnel energy and falling back to
reflection at total internal reflection), stamp the view policy in every
frame's metadata, and label it on the page.

**Evidence.** At one camera sample, forced refraction through a water surface
gave fine change 6.6 against 26.1 for the physical view, and 66% less pixel
variance below the surface. At four samples the physical view reached 8.9,
and it was preferred to the forced view for the final clip.

## A residual that no field setting changes is not a cache error

**Insight.** If the same error pattern survives every change of field size,
resolution and precision, look for the cause in the transport or the emitter.

**Mechanism.** A capacity-limited field changes its error when capacity
changes. An error that stays put comes from a mismatch between the renderer
and the reference, which the cache learns faithfully.

**How to apply.** Before tuning a cache against a reference, compare the
blurred difference maps of two or three capacity settings. If they agree,
audit emitter profiles, clamps and material conventions.

**Evidence.** On a glass-caustic scene, hash-table sizes from 2^16 to 2^20,
level counts and precision all left the same residual: a red band just inside
a spotlight's edge and a blue band just outside. The renderer's spot falloff
was smoothstep in cosine where the reference is linear in angle. Fixing it cut
the low-band error 2.4 to 6×.

## Match the reference's clamps and stamp them

**Insight.** A clamp in the reference renderer can dominate the error and can
make a biased cache look correct.

**Mechanism.** A per-photon power cap that binds on every photon scales the
caustics by a constant. A cache trained with the same default cap agrees with
the reference while both sit below the physical field, and any change of
emission sampling breaks that agreement.

**How to apply.** Stamp every clamp in both the reference and the fitted
recipe, and have the scorer refuse a comparison when they differ.

**Evidence.** A 10,000 power cap bound on every photon of a glass-box scene
(launch powers 15,080 and 12,566) and scaled its caustics by .66 and .80.
Removing it raised the reference's photon term by 32%.

## Audit the random number generator conditionally

**Insight.** Test a sampler's joint and conditional distributions, not only
its marginal histograms; a defect there can cost more quality than any
training setting.

**Mechanism.** A reference point is drawn by selecting a surface cell and then
drawing a position inside it. With a weak generator, the draws that follow a
given cell selection are correlated with the draw that selected it, so parts
of each cell are never sampled. The marginal histograms look uniform.

**How to apply.** Condition on selecting one cell, enumerate every generator
state that selects it, and bin the next two draws. A good generator fills
every bin with a chi-square per degree of freedom near 1. Replace a failing
generator (a PCG variant with the full lane and sequence identity scrambled
into its state worked), stamp the stream version in checkpoints, and keep the
old stream available for replay.

**Evidence.** A xorshift generator left 64 of 1,024 conditional bins empty for
one cell; PCG left none (chi-square per degree of freedom .95 to 1.06).
Changing only the generator took a water-caustic fit from .353 to .072 error
at the same training time and removed spurious bright points. Changing both
the generator and the CDF (below) lowered three static scenes' error by 2%,
41% and 51%.

## Build sampling CDFs in double precision

**Insight.** A float32 cumulative distribution over hundreds of thousands of
cells silently gives some positive-mass cells zero probability.

**Mechanism.** Once the running sum is large, a small cell's mass is below
half an ulp of it, so its CDF increment rounds to zero and it can never be
selected, although its evaluated density stays positive. The estimator is
then biased on those cells.

**How to apply.** Rebuild the CDF in float64 from the canonical per-cell
masses, draw 53-bit uniforms for the selection, and normalize the selection
and the evaluated density by the same total. Test every cell's probability on
the CPU and the GPU.

**Evidence.** 16,990 of 270,294 cells had positive density and a zero float32
CDF increment; the float64 table had none.

## Decompose the error budget before reading a difference

**Insight.** Estimate reference noise, fit-to-fit spread and camera noise
first; only the remainder is systematic.

**How to apply.** Render the reference with two seeds: their difference gives
its noise. Fit several seeds of the same recipe: their spread is the
threshold below which a one-seed difference means nothing. Cross-products of
two independent reference seeds cancel reference noise and allow band-pass
energy comparisons that show which spatial frequencies the field loses.

**Evidence.** On a water caustic, reference noise was .0288 per seed,
fit-to-fit spread .0246 and camera noise about .012, leaving about .051
systematic. The field reproduced 78.6% of the true amplitude in the finest
band (0 to .7 px) and 98.9% in the 2.8 to 5.6 px band, so the remaining bias
was pixel-scale detail. Repeated fits of one static scene spread by about
.003 in relative L2.

## Look at the images, not only the score

**Insight.** A global metric can rank a worse field higher.

**Mechanism.** A few bright outliers, sub-pixel emitter highlights or
reference noise dominate squared error, while sharp structure barely moves it.

**How to apply.** Judge every candidate on reference, cache and difference
panels at a common exposure and with no fitted scale, and on registered crops
of the structure that matters (caustic filaments, textures, thin surfaces).
Score a low-pass band (a Gaussian blur of σ = 2 px on both images) alongside
the raw error when reference grain dominates. Evaluate at the moved views a
recording will show, not only the canonical camera. Small thumbnails cannot
validate fine detail.

**Evidence.** A field trained with 8× batches looked sharper but scored .192
against .095. A large model scored 1.30 because of a few bright outliers. On
one scene, 84% of the squared error sat in an 11-pixel emitter highlight whose
total was within 4% of the reference; every field and sampling variant scored
within 2% of the others.

## Measure detail separately when the target moves

**Insight.** For a cache that tracks moving geometry, image error cannot tell
a sharp but late field from a washed-out one.

**Mechanism.** A fully settled field shown one frame late already scores
.13 to .14 relative L2 against the current pose, about twice the settled
error, so small lags dominate the metric.

**How to apply.** Add a high-pass energy and correlation relative to the
reference to the error. Measure the tracking target before tuning: the change
of the true image between consecutive frames at the intended motion speed,
and the settled error, bound what tracking can achieve.

**Evidence.** The two high-pass scores were about .82 for settled fits, .48
to .53 at 16 updates per pose, and .54 to .59 with a 10× learning rate after
the initial fit.

## Coordinates tied to one camera spend resolution along its rays

**Insight.** A field parameterized in a camera's contracted NDC shows radial
streaks from other viewpoints; a camera-free contraction around the subject
avoids them.

**Mechanism.** NDC cells are long along the authored camera's depth. Views
from the side see those long cells as streaks.

**How to apply.** For orbiting cameras, map positions linearly inside a tight
box around the subject and contract them outside it. Keep the box tight, and
keep the camera outside it. Draw the training samples' reference positions in
the same coordinates as the field. When streaks appear, change the
coordinate allocation before raising grid resolution: finer grids reproduce
the same streaks.

**Evidence.** On an orbit around a glass box, this matched in 10,000 updates an
NDC field that needed 30,000 updates and a 2× finer grid, and lowered the
orbit's low-band error 10 to 16%. Looser boxes (±25, then ±40 against ±12
around the subject) were progressively worse. Leaving the reference sampling
in the camera's coordinates cost 30% of the low band. In a room scene where
the camera sits inside the box, the contraction was slightly worse than NDC
(.0714 against .0704). In the browser field, raising the finest grid
resolution to 2,048 and 8,192 left the floor streaks unchanged (.113), while
widening the NDC contraction margin from .05 to .25 per side cut them to .025
and a plain world box to .015.
