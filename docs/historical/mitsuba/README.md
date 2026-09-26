# Mitsuba 3 and DrJit engineering notes

These notes record what was learned building a GPU differentiable path tracer
on Mitsuba 3 and DrJit, in which neural fields (tiny-cuda-nn hash grids run
through PyTorch) are fitted to path samples and queried at path vertices.
Each file covers one subject. Each entry states the lesson first, then the
mechanism behind it, how to apply it, and the measurement that established
it where one exists.

The measurements come from this software and hardware:

- Mitsuba 3.8.0 in the `cuda_ad_rgb` variant;
- DrJit 1.3.1 with its CUDA and OptiX backends;
- PyTorch 2.7 (CUDA 12.8 build) and tiny-cuda-nn built for fp32;
- NVIDIA H100 and RTX A6000 GPUs on 580-series drivers.

Behavior that depends on a version says so in its entry. Timings are
wall-clock medians of steady iterations after warm-up unless an entry states
otherwise; they show the size of an effect, not a benchmark.

## Files

- [Randomness and independence](randomness-and-independence.md): named
  random domains derived from one base seed, where estimators need
  independent draws, reseeding samplers without recompiling, replay as a
  control, and the parallel scans and atomic adds that make repeated runs
  differ.
- [JIT specialization and kernel caches](jit-and-kernel-caches.md): what makes
  DrJit compile a new kernel, how to detect it, and how to keep the DrJit,
  OptiX and CUDA driver caches fast and uncorrupted.
- [Evaluation, loops and replay](evaluation-and-loops.md): where to place
  `dr.eval`, scatter semantics inside walks, symbolic and evaluated loops,
  lane compaction, frozen functions, differentiated walks and host reads.
- [Precision and bitwise parity](precision.md): `FastMath`, fused
  multiply-add contraction at evaluation boundaries, hardware texture
  filtering in finite differences, and how to build parity probes.
- [Memory and graph growth](memory.md): detaching persistent state, the
  retention leak in `dr.wrap` inference, allocator caches, OOM retry and
  what to measure.
- [DrJit, PyTorch and tiny-cuda-nn](torch-interop.md): the `dr.wrap` bridge,
  its three gradient paths, and the host-shape constraints of tiny-cuda-nn.
- [Geometry and BSDF conventions](geometry-conventions.md): geometric and
  shading normals, what Mitsuba's BSDF and mesh APIs return, complete surface
  interactions, and derivatives of fields supervised on surfaces.
- [Testing and profiling](testing-and-profiling.md): running tests on the
  GPU variant, acceptance gates for performance changes, and reading DrJit,
  Torch and host profiles.

## Where to start

For a slow or recompiling loop, read
[JIT specialization](jit-and-kernel-caches.md), then
[evaluation](evaluation-and-loops.md), then the profiling half of
[testing and profiling](testing-and-profiling.md). For a run that does not
repeat itself, or an estimate that looks biased, read
[randomness and independence](randomness-and-independence.md) and
[precision](precision.md). For memory that grows, read [memory](memory.md).
