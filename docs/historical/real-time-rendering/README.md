# Real-time rendering with a cache trained online

Notes from building two interactive demos of a neural radiance cache (a
hash-grid encoding and a small MLP) that trains online from light and camera
paths while it renders: a native CUDA backend using tiny-cuda-nn, streamed to
a browser page, and a WebGPU page that trains and renders in WGSL on the
visitor's GPU. The scenes were static glass and caustic scenes and a
deforming water surface. Measurements are from one NVIDIA H100 unless a note
says otherwise.

| Note | Subject |
|---|---|
| [frame-budget-and-training.md](frame-budget-and-training.md) | Splitting each frame between optimizer updates and camera samples, the display floor, and training parameters for static and moving scenes |
| [image-quality-and-flicker.md](image-quality-and-flicker.md) | Separating camera noise, field error and frame-to-frame churn; flicker metrics, EMA display, stratified camera sampling, sampler defects |
| [cuda-kernel-performance.md](cuda-kernel-performance.md) | Profiling and exact speedups for the CUDA training trace, hash grid, MLP and optimizer; stream overlap; half precision; rejected ideas |
| [exactness-gates.md](exactness-gates.md) | Proving a speed change leaves training numerics unchanged: record dumps, atomic-order floors, deterministic reductions, tiered gates |
| [webgpu-and-wgsl.md](webgpu-and-wgsl.md) | WebGPU limits and pitfalls, CPU validation of WGSL with naga, fixed-point atomics, tiled and fused MLP kernels, compiler arithmetic |
| [web-and-native-parity.md](web-and-native-parity.md) | Bringing a WebGPU port to a CUDA reference: shared random streams, cross-loading weights, ulp ties at coplanar surfaces, host replay |
| [headless-gpu-testing.md](headless-gpu-testing.md) | Running WebGPU tests unattended with Playwright and Chromium on multi-GPU Linux hosts: Vulkan flags, device selection, readback stalls |
| [recordings.md](recordings.md) | Capturing faithful real-time clips of a GPU page: virtual display, compositor timestamps, frame holding, encoding, audits, clip design |

Read [frame-budget-and-training.md](frame-budget-and-training.md) first: it
sets the terms (updates per frame, camera samples, display floor) the other
notes use.
