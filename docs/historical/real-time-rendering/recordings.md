# Recording real-time clips

The demo's deliverable was a set of clips under a minute each, captured from
the actual interactive page while the cache trained, so that a viewer sees
the real frame rate, the real training time and the real image. Capturing a
GPU page headlessly on a cluster and encoding it without misrepresenting any
of those took several failed rounds. These notes record the capture pipeline,
the checks that made a clip trustworthy, and the clip design that looked
best.

## Capture the real page with headed Chromium under a virtual display

**Insight.** Headless Chromium recorded a black WebGPU canvas; normal Chromium
under Xvfb recorded the scene.

**Mechanism.** In headless mode the compositor frames delivered to the
screencast did not contain the WebGPU canvas, although the GPU film was
valid. Only the surrounding HTML appeared. A headed browser on a virtual
display composites the canvas like a desktop browser.

**How to apply.** Run `xvfb-run -a -s '-screen 0 1280x960x24 -nolisten tcp'`
around the recorder, on an isolated hardware GPU. When the renderer runs on a
server GPU and the page only displays streamed frames, launch Chromium with
`--disable-gpu` so its 2D compositor stays off the GPU. Keep a permanent
nonblack-canvas gate (for example, more than 5% of canvas pixels with an RGB
sum above 20 in frames sampled across the clip). Never substitute an offline
render or a software adapter for a failed capture.

**Evidence.** A six-second probe failed the gate with a lit fraction of 0: the
sidebar and overlay were captured and the canvas was black. The next probe,
headed under Xvfb, passed.

## Record compositor frames with their swap timestamps

**Insight.** Capture frames through the Chrome DevTools Protocol screencast
and keep each frame's own timestamp, not its arrival time.

**Mechanism.** `Page.startScreencast` delivers frames with
`metadata.timestamp`, the compositor's frame-swap time. Frames can arrive out
of order, so arrival order misstates timing.

**How to apply.** Store each frame with its swap time and arrival index,
acknowledge each with `Page.screencastFrameAck`, and sort by swap time before
encoding. Capture PNG rather than JPEG sources (see below). Write a capture
log before encoding, so timing evidence survives a later failure. When
stopping, remove the listener, await every pending write and acknowledgement,
and only then close the page; late acknowledgements otherwise race the page
closure.

## Produce a 60 Hz file by holding real frames

**Insight.** Encode at a fixed 60 Hz where each output frame is the latest
captured frame at or before its time, with no interpolation and no speed-up.

**Mechanism.** The page renders at 15 to 50 FPS. Holding frames preserves the
real cadence in a file any player can show; interpolation or retiming would
misrepresent the frame rate.

**How to apply.** Walk output times `start + k/60` and advance a source index
while the next source timestamp is at or before that time. Pipe the selected
frames to the encoder in that order (`ffmpeg -f image2pipe -framerate 60 -i
pipe:0`) and store the output-to-source index map in the clip's metadata.
Avoid ffmpeg's concat demuxer with per-file durations: version 4.4 rejects the
per-file `framerate` option, and the image demuxer's default rate silently
quantizes timestamps to 25 fps. Encode after the capture finishes, so the
encoder cannot take CPU from the page.

**Evidence.** A 60-second clip is exactly 3,600 output frames. A six-second
probe captured 168 source frames and produced 360 output frames.

## Encode lossless captures at low CRF in 4:4:4, and ship a 4:2:0 copy

**Insight.** Grainy low-sample frames need a lower CRF than clean ones, and
4:2:0 chroma loses fine coloured noise that 4:4:4 keeps, but many players
cannot play 4:4:4.

**Mechanism.** Monte Carlo grain is expensive to encode, and the encoder
smooths it. With JPEG sources, a similarity gate against the source measures
only the H.264 loss, not the JPEG loss; lossless PNG sources make the gate
measure the total.

**How to apply.** Capture PNG, keep the source frames locally for audit and
re-encoding, and encode a master with `-c:v libx264 -crf 4 -pix_fmt yuv444p
-movflags +faststart`. Produce a separately labelled broadly compatible copy
from the same source frames and frame map (H.264 High, level 4.1,
`yuv420p`, CRF 4), and inspect it as well as the master. Gate each on the SSIM
of the canvas crop between source and decoded frames (above .98 worked).

**Evidence.** A 4:2:0 CRF 8 clip of a glass-box scene failed the .98 gate;
4:4:4 at CRF 8 reached .9988. A one-sample-per-frame water clip failed at CRF
8 4:4:4 (.9788) and passed at CRF 4 (.990). The compatible 4:2:0 CRF 4 copies
scored .980 to .996. The 4:4:4 master played in Chromium and not in a common
desktop player. Grain also inflates files: a 60-second four-sample water clip
at CRF 4 was 138 MB against 43 to 65 MB for static scenes.

## Audit each clip independently of the recorder

**Insight.** A separate script that trusts none of the recorder's flags is
what makes a clip's claims checkable.

**How to apply.** From the saved artifacts alone, check that:

- the MP4's duration, frame count, rate and size match (`ffprobe`), and its
  hash matches the manifest;
- every output frame maps to the latest source frame at its time;
- the displayed FPS in each window, recomputed from per-frame completion
  observations, matches what the overlay showed;
- every frame after the training stop shows the same update count and the
  frozen camera budget, and every training frame shows the training budget;
- the camera followed its tour (fixed target, constant orbit radius, the
  expected number of turns by unwrapped angle, enough displacement);
- the canvas SSIM of decoded against source frames is above the threshold at
  several points in the clip.

Then play and seek each file in a real browser (for example at .25, 30 and
59.5 s) and check playback advances without page errors. Browser playback is
evidence for that browser only.

## Gate the slowest second of training, not the average FPS

**Insight.** Reject a clip if any one-second window of training, warm-up
included, shows fewer than 15 displayed frames.

**Mechanism.** Averages hide the warm-up phase and host hitches, and a slow
second is what a viewer notices. Applied retroactively, this rule failed four
earlier clips whose averages looked acceptable (their slowest seconds had 4,
5, 9 and 14 frames).

**How to apply.** Slide a one-second window in quarter-second steps over the
training phase. Report the average and the slowest second together. Log GPU
compute per frame separately from display cadence, so a drop can be
attributed: one clip's slowest second had 50 to 80 ms frame gaps while compute
stayed at 28 ms, which placed the hitch on the host.

**Evidence.** The final clips averaged 18.9, 19.8, 20.2 and 37.4 FPS while
training, with slowest seconds of 18, 19, 20 and 33 frames.

## Nothing on the page may jump

**Insight.** Reject a clip whose canvas moves or whose camera steps
discontinuously; both are visible and both came from the page, not the
renderer.

**Mechanism.** A status line that wrapped to a second line moved the canvas by
12 px when training stopped. The recorder read the field of view from the HTML
default before the app replaced it, which produced a zoom jump. The first
camera-pose request went out before capture started and nothing waited for it,
so clips opened with a 3 to 12° snap.

**How to apply.** Reserve fixed space for status text (two lines, tabular
numerals) and measure the canvas box every animation frame, rejecting any
shift. Wait for an explicit readiness signal (the app enabling its controls)
before reading initial state. Send the tour's opening pose, wait until the
page reports presenting it, then wait two more animation frames before
starting capture. Reject any presented camera step over 2° or over 5% of the
orbit radius; at 15 FPS an orbit of 20 s per turn moves under 1° per frame.

## Drive the real page only through its own controls

**Insight.** The recorder should use the same controls a user has, so the clip
shows the product.

**How to apply.** Start the page paused (`?train=0&animate=0`), then release
training through the page's own checkbox. Move the camera with a validated
world-space pose event that the page's camera controller applies and reports
back. Stop training through the same checkbox. Stamp the executed render state
in every frame's metadata (camera transport mode, camera samples, updates,
training flag) and assert it: a checkpoint's stored display stamp once
silently overrode the intended view policy, and only the frame metadata
revealed it.

## Label pretrained starts and partial fits

**Insight.** A clip that starts from saved weights must say so on the page, and
the audit must check it.

**How to apply.** Allow a checkpoint start only when the server configuration
declares it, show a "pretrained" label, and require a nonzero initial update
count. Otherwise require zero updates at the start. Label a clip that stops
before its schedule's horizon as a partial fit. Start static scenes from
fresh weights; if the warm-up is slow, change the recipe or the clip length
rather than pretraining.

## Train, then show the frozen cache

**Insight.** The clip design that looked best trains at a modest camera
budget until a fixed update count, then tours the frozen cache at a high
sample count. In static scenes the camera moves from the first frame to the
last; in a dynamic scene it stays fixed while the geometry moves and tours
the cache once training stops.

**Mechanism.** While training, camera samples compete with updates and every
frame resets the film. Once frozen, the full frame budget can go to camera
samples, which shows the converged quality.

**How to apply.** Use 4 camera samples per training frame and 32 per frozen
frame, and fill a training frame of about 50 ms with updates (16 to 51
updates depending on the scene's cost). Size the schedule to the clip: a
schedule that needs 56 s at four samples does not fit a 50-second clip with a
frozen tour, so use a shorter schedule of the same recipe. Preview stills along
a camera tour before committing to it, since a tour can expose unfinished
parts of a scene. For a dynamic scene, stop the geometry before stopping
training so the training-to-frozen comparison is made at fixed geometry.

**Evidence.** One sample per training frame finished schedules sooner but
raised the fine frame-to-frame change 2.3 to 5×; four samples kept the 15 Hz
floor with clean frames. Frozen tours ran at 20 to 46 FPS at 32 samples.

## Measure flicker before a recording round

**Insight.** Compare each candidate recipe's frame-to-frame change with an
accepted clip before recording it; see [image quality and
flicker](image-quality-and-flicker.md) for the metric.

**Evidence.** Re-recording an accepted clip's configuration on a newer, faster
binary reproduced its flicker metrics exactly (8.89 fine / .68 coarse while
moving), which made it a reliable baseline for new recipes.

## Keep each clip reproducible

**Insight.** Store the exact server configuration, binary hash and scene
package hashes with every clip.

**How to apply.** Write the configuration byte for byte into the clip's
manifest along with the executable's SHA-256, the source revision with a patch
of uncommitted changes, and each scene package's payload hash. Run a short
probe (six seconds) before any full-length round; probes caught every
pipeline bug above before a full clip was spent on it.

## Publish only verified artifacts

**Insight.** Treat upload as a checked copy, and delete old versions only after
the new copy verifies.

**How to apply.** Upload each clip to its own folder (video, telemetry JSON,
poster, playback check, audit, capture manifest and a README), copy rather
than sync, then run a one-way check of file count, bytes and hashes. Remove
the superseded folder of the same clip only after that passes, and keep a
never-delete folder for a reference clip. Do not upload raw source frames,
logs or credentials. Read the test report rather than the scheduler's exit
status: jobs reported success while the capture test had failed.
