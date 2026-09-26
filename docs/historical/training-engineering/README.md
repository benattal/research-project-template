# Training engineering notes

These notes record what was learned training PyTorch generative models with
HuggingFace accelerate and diffusers on SLURM GPU nodes, with a separate JAX
environment used to check a PyTorch port against a reference
implementation. The models were one-step image generators trained in pixel
space and in the latent space of a frozen pretrained VQ-VAE, plus small 1D
models. Their losses compare a batch of generated samples with data through
pairwise kernels computed under stop-gradient, so in-batch statistics, random
streams and numerical precision all reach the objective directly.

Each file covers one subject. Each entry states the lesson first, then the
mechanism behind it, how to apply it, and the measurement or error message
that established it.

The measurements come from this software and hardware:

- Python 3.11, PyTorch 2 (up to 2.12) with torchvision, accelerate 1.x,
  diffusers and transformers;
- JAX and Flax on CPU, in their own virtual environment, for reference
  parity tests;
- NVIDIA H100 (80 GB) and L40S (44 GiB) GPUs, launched with `torchrun` on
  one node.

Timings are steady-state means per training step after warm-up unless an
entry says otherwise. They show the size of an effect, not a benchmark.

## Files

| File | Subject |
|---|---|
| [mixed-precision-and-memory.md](mixed-precision-and-memory.md) | bf16 autocast reach, fp32 islands, frozen submodules, two-phase backward, OOM diagnosis, large solves |
| [multi-gpu.md](multi-gpu.md) | accelerate and DDP: global batch, per-rank seeds, broadcast state, collective decisions, `no_sync`, DDP errors |
| [optimizer-steps-and-ema.md](optimizer-steps-and-ema.md) | Non-finite gradient guards and their scope, accumulation scaling, schedule horizons, EMA storage and evaluation |
| [checkpoints-and-resume.md](checkpoints-and-resume.md) | What a checkpoint holds, atomic writes, automatic resume, per-epoch reseeding, keyed per-sample randomness |
| [data-loading.md](data-loading.md) | Persistent workers and epoch state, pinned memory, main-process thread pools, decode caches, index and label mapping |
| [profiling-and-throughput.md](profiling-and-throughput.md) | Section timing with CUDA events, host syncs, GEMM rewrites, batched forwards, `channels_last`, classifying speed changes |
| [codebase-audits.md](codebase-audits.md) | Running an audit, recurring findings, configuration hygiene, refactor parity against the committed original, docs CI |
| [jax-and-reference-parity.md](jax-and-reference-parity.md) | Separate JAX and PyTorch environments, file-based parity tests, weight layouts, framework default differences |

## Where to start

For an out-of-memory error, read the first entries of
[mixed precision and memory](mixed-precision-and-memory.md). For a run whose
loss drifts across resumed jobs, read
[checkpoints and resume](checkpoints-and-resume.md), then
[data loading](data-loading.md). For a multi-GPU run that hangs, errors on
its second iteration, or trains differently from one GPU, read
[multi-GPU](multi-gpu.md). For a run that stops improving with NaN skips in
its log, read [optimizer steps and EMA](optimizer-steps-and-ema.md).

Scheduler, login-node and environment lessons from the same work are in
[clusters](../clusters/README.md).
