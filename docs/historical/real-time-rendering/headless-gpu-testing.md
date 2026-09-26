# Headless GPU testing of a WebGPU page

A WebGPU demo that trains and renders in the browser can only be tested on a
real GPU: shader compilation, driver arithmetic, device limits and presentation
all differ from any CPU mirror. These notes cover running such tests
unattended on a shared multi-GPU Linux host with Playwright and Chromium, with
no display server for ordinary tests and a virtual one for capture.

## A non-null `navigator.gpu` does not prove hardware execution

**Insight.** The gate for "WebGPU works here" is a test that compiles and
dispatches the page's real compute shaders on an identified hardware adapter
and reads back finite values, not the presence of the API.

**Mechanism.** Chromium can expose WebGPU through a software adapter
(SwiftShader) when it fails to initialize Vulkan. Pages load, pipelines
compile and small tests pass, but the numbers and timings come from the CPU.
Node unit tests and a non-null `navigator.gpu` cannot tell the two apart.

**How to apply.** Request the adapter with `powerPreference:
'high-performance'` and throw when `adapter.info.isFallbackAdapter` is true.
Record the adapter info, browser and driver versions, and the output of
`chrome://gpu` in every test report. Never let a test fall back to software
rendering for training or timing; fail instead, and surface the page's error
text in the test log. On one node, a capture failed at startup with an
uninformative error until the page's message ("requires a hardware WebGPU
adapter") was logged; the node was excluded and the guard kept.

## Chromium's Vulkan flags for headless WebGPU on Linux

**Insight.** A pinned Chromium in its new headless mode, with Vulkan forced
through ANGLE and no Vulkan surface, runs WebGPU compute on an NVIDIA GPU with
no X server.

**Mechanism.** Playwright's default headless shell is a separate, reduced
build. `channel: 'chromium'` selects the full browser in new headless mode.
On Linux the WebGPU backend needs Vulkan; the surface extension is not needed
for compute and fails without a display.

**How to apply.** In `playwright.config.ts`:

```ts
use: {
  channel: 'chromium',
  headless: true,
  launchOptions: {
    args: ['--enable-logging=stderr', '--use-angle=vulkan',
           '--enable-features=Vulkan', '--disable-vulkan-surface',
           '--enable-unsafe-webgpu'],
  },
},
```

Install the browser into a project-local directory
(`PLAYWRIGHT_BROWSERS_PATH=$PWD/.cache/ms-playwright npx playwright install
chromium`) so the version is pinned with the lockfile rather than taken from a
shared cache.

## On a multi-GPU host, the browser may compute on a GPU you did not choose

**Insight.** `CUDA_VISIBLE_DEVICES` does not isolate a browser, and even a
correct `vulkaninfo` result does not: Chromium's graphics process and its
WebGPU compute device can open different GPUs.

**Mechanism.** CUDA's visibility mask applies only to the CUDA runtime. Vulkan
enumerates devices through its own loader, and Mesa's device-select layer
honours `DRI_PRIME`, but Chromium did not load that layer from its default
library path. Chromium's graphics context and its WebGPU implementation (Dawn)
each pick an adapter, and they need not agree. With `DRI_PRIME` and the Mesa
selector set, `vulkaninfo` listed only the requested GPU, yet per-process
telemetry showed the page's WebGPU training on GPU 0 while its graphics
context opened the requested one. A test that stalled in that state passed on
an isolated GPU.

**How to apply.**

- Prefer an allocation in which exactly one GPU is visible (a scheduler
  device cgroup). Refuse to run when more than one device is visible.
- Select the Vulkan device by identity, not ordinal: read the GPU UUID and PCI
  bus ID from `nvidia-smi`, set `DRI_PRIME=pci-0000_xx_xx_x`,
  `MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1` and
  `VK_ICD_FILENAMES` to the NVIDIA ICD, then check that `vulkaninfo
  --summary` lists exactly one `deviceUUID` and that it matches.
- A device cgroup can renumber the allocated GPU to ordinal 0, so ordinals in
  logs are not evidence of which physical device ran.
- When isolation is not available, confirm placement with per-process GPU
  telemetry (`nvidia-smi` process list) while the test runs, before trusting
  any timing.

## Sandbox and writable directories on compute nodes

**Insight.** Chromium can fail at startup on a cluster compute node for two
reasons unrelated to the GPU: an unwritable home directory and a sandbox that
traps.

**Mechanism.** Chromium's crash database and config follow the XDG
directories, separately from Playwright's user-data directory; a read-only
home mount aborts startup. On some cluster hosts the sandbox-enabled browser
traps before loading a page, for a reason that was not identified. A job whose
launch omitted the sandbox exception never started Chromium.

**How to apply.** Point `XDG_CONFIG_HOME` and `XDG_CACHE_HOME` at job-local
scratch. Keep the sandbox enabled by default and disable it only through an
explicit launcher argument, only for trusted loopback test pages inside an
isolated allocation. When the browser dies before any page loads, run it under
`gdb --batch -ex run -ex 'thread apply all bt'` with `--headless
--disable-gpu --dump-dom about:blank` to see where it stops.

## Run browser GPU tests serially

**Insight.** One worker and no parallel projects keep GPU tests
interpretable.

**Mechanism.** Two pages on one GPU share its queues and memory, so their
timings and even their device-loss behaviour interact. After a device loss the
next tests on the same browser fail for unrelated reasons, which is why
stopping at the first failure is tempting; but a stop-at-first-failure
setting also hides every later result.

**How to apply.** Set `workers: 1` and `fullyParallel: false`, write a JSON
report (`report.json`), and define separate Playwright projects for hardware
gates, performance, acceptance, interactive checks and recordings so each can
be selected alone and run as its own job. If `maxFailures: 1` is kept, split
independent suites into separate jobs so one failure cannot hide another
suite's evidence. Keep browser tests out of the unit-test runner's glob.

## Each job owns its web server and port

**Insight.** A test must start its own server on its own port and refuse to
attach to one that is already listening.

**Mechanism.** GPU jobs on one host share a network namespace. Two jobs on
one host collided on the development server's default port, and with server
reuse enabled a test would exercise another job's server and build.

**How to apply.** Derive the port from the job ID (for example
`20000 + job_id % 40000`), pass it with `--strictPort`, and set Playwright's
`webServer.reuseExistingServer: false`. Serve the built page (`vite preview`)
for tests of a build that will be published, and rebuild before testing: a
preview server serves the last `dist/`, not current source.

## A presented canvas can stall later readbacks in headless Chromium

**Insight.** Readback tests should render without presenting to a canvas.

**Mechanism.** On some jobs, one presented canvas frame in headless Chromium
made every later `mapAsync` readback hang; the adapter was then lost and the
next few tests failed with "requires a hardware WebGPU adapter". The driver,
device and GPU count matched jobs that passed, and the trigger was not found.
The same cause had earlier appeared as slow probe timeouts, as a fit that
stalled at its first rendered frame after all its updates completed, and as an
adapter-loss flake.

**How to apply.** Give the renderer a `present: false` option and use it in
every readback probe. Present only in the interactive and recording projects,
which run under a virtual display. Rerun a failure of this signature once
before suspecting the change under test.

**Evidence.** After one presented frame, 8 of 8 readbacks stalled until a
45 s timeout; without presentation all 8 finished in about 35 ms. Tests that
read an offscreen texture never stalled.

## Headless canvas pixels are not evidence

**Insight.** Validate numerics from the GPU film buffer, not from the canvas.

**Mechanism.** In headless Chromium a `drawImage` readback of the WebGPU
canvas after presentation returned black, and a recording gate captured a
canvas with no lit pixels, while the film buffer held the correct image.

**How to apply.** Read the renderer's own film buffer for numerical checks and
test the display transform (exposure, tonemap, sRGB) separately. Capture
video from a headed browser under Xvfb (see [recordings](recordings.md)).

## A machine-readable test interface on the page

**Insight.** Tests drive the same renderer lifecycle as the UI through a small
page API instead of simulating clicks.

**Mechanism.** A separate test renderer drifts from the product. Driving the
real one through explicit calls keeps one code path while making each step
bounded and observable.

**How to apply.** Expose a ready promise, explicit seed and configuration,
bounded `step(n)` calls, a GPU completion fence, typed-array readback and a
result JSON. Disable automatic animation during tests. Fail the test on any
`uncapturederror`, device loss, shader compilation message, page error,
timeout or missing asset. For camera tours, dispatch a custom event carrying
a world-space `{origin, target, up}` that the page's own camera controller
validates and applies, and publish the accepted pose on a data attribute so
the test can check it.

## Wait for explicit readiness, not for an element to look enabled

**Insight.** Wait for the app's own readiness signal before reading its
state.

**Evidence.** Playwright's "enabled" check on a fieldset passed before the app
had replaced the HTML placeholder values with the scene's camera, and a
recorder read the placeholder field of view. Waiting for the app to remove the
`disabled` attribute it set fixed it. Readouts derived from a buffer must also
reset synchronously when the buffer changes: a sample-count readout still
described the old film after a resize.

## Keep test jobs short and their failures separate

**Insight.** Bound each job and each stage so that a stall is located rather
than only timed out.

**How to apply.** Split long gate suites into jobs with short limits (45
minutes rather than two hours), run independent pages in separate jobs so one
failure cannot hide the other's evidence, give long stages named deadlines
(30 s worked), and log progress markers (every 1,000 updates) inside fits.
Match whole words in text assertions: a check for "EMA" matched inside
"TONEMAP". For CPU unit tests of scene parsing, use JSDOM rather than Happy
DOM, which rejected legal single-quoted XML declarations.

## Trust the report, not the exit code

**Insight.** A scheduler job that "completed" has not passed.

**Mechanism.** Launcher scripts can exit zero after a test runner reports
failures, or after producing no artifacts.

**How to apply.** Read Playwright's `report.json` and the job's stderr for
every run. Typecheck browser test files explicitly when they sit outside the
app's `tsconfig` include set (`npx tsc --noEmit --strict ...
tests/browser/<file>.spec.ts`); otherwise a type error surfaces only on the
GPU.
