# Frame budget and online training

The demo renders a scene through a neural radiance cache (a hash-grid encoding
followed by a small MLP) while it trains that cache from light and camera
paths generated on the same GPU. Every displayed frame therefore splits one
time budget between optimizer updates and camera samples. These notes record
how that split was chosen, what the display rate needs, and which training
parameters matter when the cache must follow a moving scene. Numbers are from
an H100 unless stated otherwise.

## Hold a display floor while training, not a target frame rate

**Insight.** A training page needs a minimum display rate, about 15 Hz, and
should spend everything above that floor on updates.

**Mechanism.** Below about 15 Hz a moving camera looks jerky, so the floor is
a presentation requirement. Above it, extra frames only redraw the same field
more often: once the per-frame render and host overhead are small, the display
rate barely changes how long the training schedule takes to finish. At one
camera sample per frame the render and host overhead took about 2 ms of a
50 ms frame, so running at 17.5 FPS instead of 19 shortened a schedule by less
than 1%. Time to a given quality is set by the number of updates and their
cost, so per-update speed is what shortens it.

**How to apply.** Fill each frame with updates up to the floor. Judge a recipe
by its wall time to a reference-verified quality at that display rate, not by
its FPS.

## Predict the frame before running it

**Insight.** Frame time is close to `updates_per_frame × ms_per_update +
render(camera_samples)`, plus about 4% and 1 ms in the live page, so the
display rate can be predicted from two benchmarks.

**Mechanism.** Updates and camera samples run back to back on the GPU, and
host work overlaps them. The average hides variance: the slowest one-second
window runs about 1.5 frames below the mean.

**How to apply.** Benchmark ms per update and the render cost per camera
sample for each scene and binary. Keep the predicted compute at or below about
58 ms for a 15 Hz floor (66 ms frame). After a run, compare the prediction
with the recorded per-frame timings, and gate on the slowest one-second window
of training, warm-up included, rather than on the mean.

Read the live per-update cost as the difference between a training frame's
and a frozen frame's compute time, divided by the updates per frame; this
isolates training cost as it runs in the real loop. When an exact speedup
lands, scale the updates per frame by the ratio of old to new update cost so
the frame time, and the display rate, stay where they were.

**Evidence.** Representative costs on an H100 at a 512-pixel film: 1.3 to
2.5 ms per update in fp16 and 2.7 to 3.3 ms in fp32, 14 to 27 ms for a
16-sample render, 1.1 to 1.7 ms for a 1-sample render. Predictions held
within 3 to 4%: eight 3.30 ms updates and a 26.7 ms render predict 53.1 ms,
about 55 ms with the live margin, and the page measured 55.3 ms. When one
update fell from 2.04 to 1.65 ms,
raising the updates per frame from 25 to 31 kept 19 FPS and brought a
15,000-update schedule in at 25.5 s instead of 32.6 s. Final recordings ran
16 to 51 updates plus 4 camera samples per frame at about 19 FPS while
training.

## Give training frames and frozen frames separate camera budgets

**Insight.** The number of camera samples per frame while the cache trains and
after training stops are two independent settings, and neither should set the
other.

**Mechanism.** While training, every frame shows a different field, so the
film resets each frame and shows only that frame's samples; camera samples
spent there compete with updates. Once the cache is frozen the film
accumulates across frames at a fixed camera, and the full frame budget can go
to samples. Moving the camera resets the film but not the learned weights.

**How to apply.** Store `training_camera_samples_per_frame` and
`frozen_camera_samples_per_frame` as separate required fields and reject any
older single field instead of reinterpreting it. Give the camera its own
random stream so neither budget changes training. Stamp the executed sample
count in each frame's metadata, separately from the accumulated film samples.

**Evidence.** One sample per training frame freed 13 to 25 ms per frame for
updates, but the frames were visibly noisy. Four samples per training frame
and 32 per frozen frame kept the 15 Hz floor and gave clean frames in every
scene.

## Batch updates per redraw, with one frame in flight

**Insight.** Run several updates between redraws and wait for the GPU to finish
a frame before submitting the next.

**Mechanism.** One update per redraw lets the display rate cap training: at
60 Hz, 10,000 updates take at least 167 s however fast the kernels are. Each
update keeps its own samples, optimizer step and weight average; batching
changes only how often the field is presented. Without a completion fence the
CPU can queue many frames ahead of the GPU, which adds latency to camera input
and training toggles and makes measured frame times meaningless.

**How to apply.** Submit `updates_per_frame` updates and the frame's camera
samples, present the last sample, then await a completion fence
(`device.queue.onSubmittedWorkDone()` on WebGPU, a stream synchronization in
CUDA) before scheduling the next frame. Keep input events live while waiting.
Use a separate `warmup_updates_per_frame` before the warm-up count if the
early schedule benefits from it; the warm-up count should only switch the
per-frame budget, not stop training. The warm-up batch must respect the
display floor too: 64 updates per frame during warm-up was the slowest second
of one clip.

**Evidence.** In the browser, 32 updates per redraw fitted 10,000 updates in
about 17 s at a 512² film, including redraws.

## Search for quality first, then for speed

**Insight.** Establish each scene's best reference-verified quality with no
frame-time limit, then find the fastest recipe that keeps it.

**Mechanism.** Self-chosen budgets (a fixed number of updates per frame, a
fixed schedule length, a fixed camera sample count) look like constraints but
are variables. Optimizing throughput under them spent effort on exact
speedups while the image still lagged the reference; a fast, blurry fit is
not an improvement.

**How to apply.** Treat updates per frame, schedule length, field size and
camera samples as free. Search in fp32 first, since half precision can cost
quality silently (see [kernel performance](cuda-kernel-performance.md)), and
recover the fp16 speed once the recipe is known. Report two numbers for each
candidate: quality at equal wall time and wall time to equal quality. Bounded
bias (a clamp) is acceptable when it improves the picture, provided it is
recorded in the recipe and measured against the unclamped reference.

## Choose precision and schedule length together so a fresh fit fits the floor

**Insight.** A clip that trains from zero must reach its quality within the
clip at the display floor, which constrains the schedule length and the
precision jointly.

**Evidence.** An fp32 field with a 2^21 table fitted only about 6 updates plus
16 camera samples in a 66 ms frame, so its 30,000-update schedule would have
taken over five minutes. A 15,000-update fp32 schedule with a 2^19 table kept
most of the 30,000-update gain on the orbit views and finished in under 30 s
at the floor. Reaching 20 FPS at 32 updates per frame needed about 1.0 ms per
update against a measured 1.23.

## Keep a floor on training samples per update

**Insight.** Do not shrink the training batch to meet a frame budget; reduce
updates per frame instead.

**Mechanism.** A small batch makes each update cheap, but at a given schedule
the fit ends far from the reference, and no number of camera samples recovers
that. Camera samples are an inference budget, separate from the training
batch.

**How to apply.** Fix a minimum number of light paths and gather paths per
update (65,536 of each worked across all scenes) and reject smaller requests
rather than clamping them. Report launched light paths, gather paths, updates
and camera samples as four separate numbers.

**Evidence.** On one scene with the same 7.8M-parameter field, 2,048 light
plus 2,048 gather paths per update scored .111 error in 6.4 s; 65,536 light
paths with 8,192 gathers scored .084; 65,536 of each scored .044 in 24 s. The
gather budget mattered as much as the light budget.

## For a static scene, a larger batch at equal samples is scene-dependent

**Insight.** Four times the batch at a quarter of the updates (the same total
samples) can save time on some scenes and cost quality on others.

**Mechanism.** Per-sample throughput rises with batch size, about 1.5× for a
4× batch, because per-update overhead is amortized. Fewer updates also means
fewer optimizer steps, which some fits need.

**How to apply.** Test it per scene at equal samples against the reference,
with the learning rate scaled (×2 worked for a 4× batch).

**Evidence.** A 4× batch (262,144 + 262,144) at twice the learning rate reached
the quality of the 10,000-update standard fit in 2,500 updates and 7.9 s on
one scene, two thirds of the time. On a glass-caustic scene it matched on
orbit views but lost 8 to 10% of the low-band error at the canonical view; on
a glass-box scene it scored .059 against .053.

## A cache that tracks a moving scene needs more updates, not bigger batches

**Insight.** When the target moves, the number of optimizer steps per pose
decides how closely the cache follows it.

**Mechanism.** Adam moves each parameter by about one learning rate per
update, so the update count bounds how far the field can travel between
poses. Larger batches reduce gradient noise but not the distance covered.

**How to apply.** Spend extra compute on more updates per displayed frame at
the same batch size. Tie scene motion to wall-clock time, not to frame count,
so a slow frame cannot hide lag by slowing the motion.

**Evidence.** On a deforming water surface at equal compute, 16 updates of
65,536 + 65,536 samples per frame reached a receiver error of .19, against .22
for 4 updates of 262,144 + 262,144 with a 4× learning rate. At 4× compute, 64
small updates (.155) beat 16 large ones (.160). Four and sixteen times the
updates per pose reached .15 and .13.

## Size the tracking budget with a fixed-geometry refit ladder

**Insight.** Before tuning live tracking, measure how many updates a settled
cache needs to recover after one geometry change, with the geometry then held
fixed.

**How to apply.** Start from a converged fit, move the geometry to a
registered new pose, hold it, and save the field at 0, 128, 512, 1,000, 4,000,
10,000 and 30,000 updates. Render every checkpoint with the same camera
samples so only the field changes. Measure the change of the true image
between consecutive displayed frames at the intended motion speed as well;
together they bound what live tracking can reach.

**Evidence.** On a water-caustic scene, error after 0 to 30,000 updates was
.129, .117, .106, .096, .076, .069 and .065, taking 0 to 53 s; 100,000 reached
.063. Sharp caustic filaments needed on the order of 1,000 updates at a pose.
At the intended motion, one 30 Hz frame changed the true caustic by .138, twice
the settled error, so even a one-frame lag cost more than the fit's own bias.

## Evaluate tracking on a sampled-pose clock, and keep I/O out of live runs

**Insight.** Test tracking offline with a declared pose clock (pose `k` at
time `k / 30`, a fixed number of updates per pose), separately from wall-clock
runs.

**Mechanism.** A wall-clock run mixes tracking with timing jitter, and
periodic checkpoint or image writes stall frames and make the motion jump.

**How to apply.** Label sampled-pose runs as offline controls, never as
real-time evidence. Run wall-clock checks with no diagnostic I/O. Report
updates per second and displayed frame rate together.

**Evidence.** Wall-clock runs at 16, 64 and 128 updates per redraw reached
538, 567 and 572 updates per second at 33.6, 8.9 and 4.5 FPS: larger batches
per redraw barely raised update throughput. A 5 Hz display holding each pose
for 96 updates matched 4× the throughput at 30 Hz.

## Step-size settings meet one floor on a moving target

**Insight.** Once a sensible learning rate is in place, retuning step size does
not make a tracking cache follow faster.

**Mechanism.** Post-warm-up learning rate ×10 or ×30, Adam β2 from .9 to
.9999, separate rates for the MLP and for coarse grid levels, and annealing
the rate within each pose all reached the same error. A noise-free analytic
moving ridge showed the same floor, so it does not come from sample noise.
A larger β2 does let a lower rate reach the floor: with β2 = .999 a ×3 rate
matched ×10.

**How to apply.** Raise the rate after the initial fit (keeping the optimizer
state and declaring the full rising schedule up front), then spend remaining
budget on updates per pose. Do not test a higher rate by training from
scratch at it: a 10× rate from the start changed the scale the field
converged to (encoding RMS 4.26 against .41), which confounds the comparison.

**Evidence.** All step-size variants at 16 updates per pose reached about .164
to .171 receiver error; ×100 diverged.

## A pose input does not give a hash grid memory of earlier poses

**Insight.** Adding the animation phase as a fourth grid coordinate does not
let the cache reuse what it learned at poses it has already visited.

**Mechanism.** Later poses overwrite the fine-level hash entries of earlier
ones, since hash collisions share entries across the phase axis.

**How to apply.** Refit at the current pose; do not budget on revisiting.

**Evidence.** A 4D grid over position and phase, trained at 256 updates per
pose, tracked no better than the plain field. Rendered without updates at a
phase it had fitted 360 poses earlier, it scored worse (.54 error) than the
plain field that had never seen that phase (.48).

## Profile the non-neural frame work on a dynamic scene

**Insight.** When geometry moves, rebuilding its acceleration structure can
cost more than the neural update.

**Evidence.** The browser's deforming-water page refitted its BVH on the CPU
and uploaded it every frame, 60 to 62 ms per frame, more than all of the
frame's training updates. A GPU refit over precomputed bottom-up levels cost
.09 ms per update in the native backend.

## Stream frames from a server GPU without letting the stream set the pace

**Insight.** When the renderer runs on a remote GPU and a browser only
displays its frames, encoding and network transfer must overlap the GPU work
and must drop frames rather than queue them.

**Mechanism.** PNG encoding and socket writes take several milliseconds per
frame. On the render thread they would add directly to frame time; in an
unbounded queue a slow client accumulates delay.

**How to apply.** Encode and send on an I/O thread while the next updates run.
Bound the pending-frame queue and coalesce it so a slow client receives the
newest frame and superseded ones are dropped. Stamp frames with a scene
generation so a frame from the previous scene is never shown after a switch.
Send per-frame metadata that separates compute, readback, CPU queue,
tonemapping and encoding time, and measure the displayed FPS in the client
rather than inferring it from GPU throughput. Keep the server on loopback and
reach it through an SSH tunnel; check the WebSocket `Origin` against the
configured localhost port.

In the client, measure FPS over animation-frame callbacks, since frame
arrivals come in bursts and exaggerate the rate. Acknowledge each control
change from the server and keep the local edit until it is acknowledged;
frames already in flight otherwise overwrote a checkbox the user had just
clicked.

**Evidence.** At 256², system zlib encoded a lossless PNG in 3.5 ms against
11.4 ms for the stb encoder, with byte-identical decoded pixels and 23%
smaller files. With encoding on the I/O thread, the displayed rate rose from
27.6 to 40.8 FPS against 56.7 completed frames per second, with the browser
limited to one CPU core.
