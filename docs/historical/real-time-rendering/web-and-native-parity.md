# Web and native parity

The demo has two backends: a WebGPU page that trains and renders on the
visitor's GPU, and a native CUDA backend that is faster and was tuned first.
Bringing the web page to the native page's pictures meant porting the native
transport, field and recipes into WGSL, then proving that the two produce the
same training samples, the same field and the same films. These notes record
the method that made that proof possible and the places where two correct
implementations still disagreed. The final web fits scored within about 1% of
native at every acceptance seed on three scenes, inside the seed-to-seed
spread.

## Port behaviour first, at the reference's batch sizes, then optimize

**Insight.** Match the reference's recipes and features before closing the
speed gap, even if the port starts slow.

**Mechanism.** The web page originally trained 32 to 128× fewer samples per
update and lacked several of the native features (directed light emission, a
camera-free coordinate mapping, the native field layout, a stratified camera).
Optimizing that page would have produced a fast demo of different pictures.

**How to apply.** Stage the port and gate each stage against the reference
before starting the next: transport records, one recipe source, field
representation, runtime schedule, quality acceptance, and only then speed.
Request the device limits that native batch sizes need (half a million lanes
per update) rather than shrinking the batch.

## Keep one recipe source for both backends

**Insight.** Put each scene's recipe in one place and export it to the other
backend, so the two can never run different numbers under the same name.

**How to apply.** Let the exporter read the recipe without override files, and
check that re-exported scene packages reproduce the production payload hashes.
When checking compatibility, compare resolved values rather than request
forms: a check failed because one side stored a light-emission request as
`{kind, alpha}` while the other stored the resolved targets.

## Reuse one scene parser and export a hashed binary package

**Insight.** Parse scenes once, in one implementation, and hand the other
backend a versioned binary package, instead of writing a second parser.

**Mechanism.** Two parsers of the same XML, OBJ, PLY and material set can
drift in defaults, transforms and texture handling, and each drift then looks
like a transport bug.

**How to apply.** Run the existing parser, BVH builder, texture packer and
sampling-table builder offline on the CPU. Write a binary payload plus a JSON
manifest that stamps source and section hashes, byte order, strides and sizes.
Make the reader strict: reject unknown versions, partial packages and failed
hashes. When testing a change to one section, confirm every other section is
byte-identical so the A/B comparison is clean.

## Share the random streams and compare training records lane by lane

**Insight.** With the same generator, seed and sequence on both sides, the
training samples can be compared lane by lane, which localizes any transport
difference to one path.

**Mechanism.** Identical integer seeds in two different generators do not give
identical paths. Porting the generator itself (here PCG64, with reference-cell
selection by a 53-bit uniform against a float64 CDF) makes each lane's record
a deterministic function of the scene, seed and index on both backends.

**How to apply.** Dump the native records for a package, seed and sequence,
and compare the web's records for validity, surface side, sampled cell and
value. Validity and discrete choices should agree on all lanes except those
flipped by rounding; values should agree to float rounding.

## Check aggregate sums as well as per-lane agreement

**Insight.** A lane-by-lane gate can pass while a small population of lanes
carries a systematic deficit.

**Evidence.** The record gate had passed, yet one camera emission term summed
to .978 of native's while every multiplicative factor agreed to 1e-4. The
deficit was 3 to 4 lanes of light out of about 160. Ranking pixels by their
difference showed that all of the added squared error lay in the 1,000 most
different pixels (the rails and legs of a glass box), with the rest of the
image equal.

**How to apply.** Sum each record term over the batch on both backends and
compare the sums. Rank film pixels by difference before reading any global
score.

## Space acceptance seeds so their streams do not overlap

**Insight.** Adjacent seeds were not independent trials.

**Mechanism.** The stream index was `seed XOR counter`, so seeds 123, 124 and
125 permuted overlapping windows of the same counters.

**How to apply.** Choose seeds whose hash inputs are disjoint below the
largest counter used (here 123, 65,543 and 131,083 for counters below 65,536),
and add a CPU gate that proves it.

## Render the reference's weights in the port to split rendering from training

**Insight.** Loading the reference backend's trained field into the port shows
whether a quality gap comes from rendering or from training.

**How to apply.** Publish the native checkpoints read-only to the page, render
them at native's camera, film, sample count and camera sequence, and compare
pixelwise with native's film of the same weights. Only then compare fits
trained on each side.

**Evidence.** Rendering native weights on the web matched native's films to
relative L2 1e-5, 4e-4 and 3e-3 on three scenes. On the scene with the gap,
the web-trained field scored .04560 and native's weights rendered by the web
scored .04559, so the entire gap was in rendering.

## Two coplanar surfaces make ray arithmetic part of the scene

**Insight.** Where two surfaces are coplanar, the last bit of the ray/box and
ray/triangle arithmetic decides which one a ray hits, and a correct port can
still render a different image.

**Mechanism.** A glass box's bottom face and a metal base's top face were both
at z = 0. A ray reaching that plane got two hit distances one ulp apart
(4.5437703 and 4.5437708). If the metal won, the ray reflected toward the
light; if the glass won, it transmitted, hit the metal from behind and went
black. The web's box test multiplied by a reciprocal and the WGSL compiler
contracted `dot` and `cross` differently from CUDA, which biased the tie
toward the glass.

**How to apply.** Port the reference's traversal arithmetic, not only its
logic: IEEE slab quotients (a refined per-ray reciprocal plus one correction
per bound), the reference's explicit parallel-ray test, the reference
compiler's FMA pattern written as explicit `fma`, and an IEEE `1/det`. Near a
triangle edge, recompute the test in higher precision (double-float on the
web, double in CUDA).

**Evidence.** Before the fix the web scored 1.5% worse than native with
native's own weights; after it the scores matched (.04484 against .04484).
The fix cost .8 to 1.1 ms per frozen 512² frame. An independent reference
renderer, with its own arithmetic, sat 35% away from both in that pixel
cluster, so only matching the reference backend is well defined there. The
first hypothesis, an fp32 near-edge triangle miss, moved the error only from
2.8e-3 to 2.6e-3 and was dropped.

## Replay device code on the host to rank arithmetic choices

**Insight.** CUDA path-sampling code written as host-callable functions can be
compiled for the CPU and replayed lane by lane, which localizes a GPU
difference without a GPU job.

**How to apply.** Keep the transport headers host-callable
(`__host__ __device__`, no device-only intrinsics in the path logic). Build a
scene view from the package's host vectors and compile with the production
random-generator defines:

```bash
g++ -std=c++17 -O2 -ffp-contract=off -Iinclude replay.cpp -o replay
g++ -std=c++17 -O2 -march=x86-64-v3 -ffp-contract=fast -Iinclude replay.cpp \
    -o replay-fma
```

Replay the camera sampler per lane (printing hits) or as emission-only films
over a pixel region, and compare the two builds to see how much rounding
alone moves the region. To see the GPU's actual arithmetic without a GPU,
compile a small kernel with `nvcc -cubin -arch=sm_90` and read its
instructions with `cuobjdump -sass`.

**Evidence.** On the coplanar-tie region, FMA contraction alone moved the tie
statistics by under 1%, while a reciprocal-multiply slab test moved them 1 to
2.5% (675 of 294,912 lanes changed in one cluster). Independent sample sets
moved the same numbers by about ±.0008, so the ranking was meaningful.

## Adopt the reference's field layout, initialization and constants exactly

**Insight.** To load one backend's weights in the other, reproduce its
parameter order, padding, hash constants, initialization draw and per-level
constants, and compute those constants where the reference does.

**Mechanism.** tiny-cuda-nn's hash grid, identity label rows padded to a
multiple of 16, bias-free MLP with padded output, and PCG32 initialization all
differ from a hand-written web field. A per-level scale computed on the host
differed from the device's `exp2f`, which is one ulp off on several levels.

**How to apply.** Evaluate the per-level scale on the device, as the library
does. Gate with a field comparison: a native checkpoint loaded on the web must
give native's predictions to float rounding, and one and eight updates on a
fixed batch must give native's gradients, moments and EMA to the reference's
own run-to-run spread. Stamp the representation, and refuse to load a
checkpoint of a different precision (an fp16 checkpoint into an fp32 field).

**Evidence.** After moving the scale to the device, the host construction was
bitwise equal to native for four fields of 1.25M to 13.1M parameters.

## Know the reference's own spread before judging the port

**Insight.** Measure how much two identical reference runs differ before
calling a port difference a bug.

**Evidence.** Two identical native one-update runs were bitwise equal in
master weights, EMA and outputs, but their grid Adam moments differed by up
to 9% on single entries, because float atomics add in an unspecified order.
Only 18 to 45% of the web's MLP moment entries were bitwise equal to native's.

**How to apply.** Compare tensors relative to each plane's maximum (an
absolute 2e-6 tolerance failed on second moments near 4,096, where one ulp is
2^-11). Gate speed work in the port against the port's own deterministic
outputs, not against the reference.

## CPU mirrors must use the GPU's arithmetic

**Insight.** A CPU check written in double precision, or with its own copy of
the constants, tests itself.

**Evidence.** A test-only CPU hash mirror carried a wrong multiplier. A double
precision comparison of predictions failed at 2e-4; switched to float32
products it matched 4,070 GPU predictions to 5.7e-6. Finite-difference tests
needed double output accumulation and exactly representable parameter
perturbations.

## Follow the reference renderer's code, not its scene file

**Insight.** When an independent renderer provides the ground truth, match
what its code does, which can differ from what the scene description says.

**Evidence.** The reference renderer's film scheduler integrated the pixel box
although the scene file named a Gaussian filter. A reference silently kept a
camera-path depth of 8; at depths 16 and 32 the cache's mean ratio to it went
from 1.053 to 1.003 and 1.001. A light's cone falloff was smoothstep in cosine
in the port and linear in angle in the reference.

**How to apply.** Stamp depth, clamps, filter and emitter conventions in each
reference, and have the scorer check them against the fitted recipe.
