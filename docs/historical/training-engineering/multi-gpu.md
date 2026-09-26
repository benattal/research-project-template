# Multi-GPU training with accelerate

These runs used HuggingFace accelerate over PyTorch DistributedDataParallel
(DDP), launched with `torchrun` on one node. DDP keeps replicas equal only if
every process takes the same number of collective steps with the same
decisions, and a multi-GPU run is comparable with a one-GPU run only if the
batch, the random streams and every data-derived constant mean the same
thing at every process count. The first section covers launching and batch
semantics, the second the correctness rules, the third errors seen at
startup.

## Launching and batch semantics

### Define the batch size as global and split it across processes

Treat the configured batch size as the global batch, give each process
`batch_size // num_processes` samples, and reject a batch size the process
count does not divide. The effective batch, learning rate and schedule are
then the same at every GPU count; more GPUs buy wall-clock time. Do not
rescale the batch size when changing the process count.

**How to apply.** A DataLoader prepared by accelerate with default settings
yields its own `batch_size` on every process, so divide before building it:

```python
world = accelerator.num_processes
if cfg.batch_size % world:
    raise ValueError(f"batch_size={cfg.batch_size} not divisible by {world}")
loader = DataLoader(dataset, batch_size=cfg.batch_size // world,
                    shuffle=True, drop_last=True, pin_memory=True,
                    num_workers=cfg.num_workers)
loader = accelerator.prepare(loader)
```

Statistics computed inside one process's batch are the exception. In-batch
pairwise kernels, contrastive negatives and batch normalization see
$B/W$ rows per process at world size $W$, not $B$. When those statistics
matter, either hold the per-process batch fixed and accept a larger gradient
batch, or gather the batch across processes before computing them.

**Evidence.** With the earlier per-process semantics, doubling the GPU count
doubled the effective batch, and runs at different GPU counts were not
comparable at equal step counts.

### Take the process count from the allocation

Launch one process per GPU that the scheduler allocated
(`SLURM_GPUS_ON_NODE`) or that the caller listed in `CUDA_VISIBLE_DEVICES`,
and one process otherwise. Do not count devices with `nvidia-smi`: on a
login node it lists GPUs that a launcher must not fan out to.

```bash
NGPU=${SLURM_GPUS_ON_NODE:-}
if [ -z "$NGPU" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NGPU=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
fi
[[ "$NGPU" =~ ^[1-9][0-9]*$ ]] || NGPU=1
if [ "$NGPU" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$NGPU" train.py "$@"
else
    python train.py "$@"
fi
```

`torchrun --standalone` runs on a single node. Using more GPUs than one node
holds needs a multi-node rendezvous (`--nnodes`, `--rdzv-endpoint`) and a
launcher per node.

### Expect scaling to follow the per-step cost

The gradient all-reduce costs roughly the same per step whatever the step's
compute, so cheap steps lose a larger fraction to it.

**Evidence.** On two L40S GPUs, a model with an 800 ms step reached 1.92×
the one-GPU throughput (4% per-step overhead), while a model with a 170 ms
step reached 1.63× (22% overhead).

## Correctness rules

### Seed the host stream identically and the device stream per process

Seed Python, NumPy and the torch host generator identically on every
process, and give the CUDA generator a per-process offset.

**Why.** If every process seeds the device from the same value, every
`torch.randn(..., device="cuda")` returns the same noise on every process. A
global batch over $W$ GPUs then holds only $B/W$ distinct noise draws, each
repeated $W$ times, and any diversity term that relies on independent draws
is divided by $W$. The host stream must stay identical when the data
sharding depends on every process drawing the same permutation.

**How to apply.** `torch.manual_seed` seeds the host and every device; the
following `torch.cuda.manual_seed_all` replaces the device seeds:

```python
def seed_epoch(seed: int, epoch: int, rank: int) -> None:
    s = (seed + epoch) % (2**31 - 1)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all((s + 1_000_003 * rank) % (2**31 - 1))
```

**Evidence.** A two-GPU smoke test that gathered the noise from both
processes showed identical draws before the change and different draws
after it, with the host-side sampler stream unchanged. One-GPU runs stayed
bit-identical; multi-GPU runs were no longer seed-comparable with earlier
multi-GPU runs.

### Compute data-derived state once and broadcast it

Compute normalization constants calibrated from data, periodically refit
auxiliary models, and data-driven initializations on process 0, then
broadcast the result.

**Why.** Each process sees a different shard and a different random stream.
If each computes its own constant, the processes optimize different
objectives, and DDP averages their gradients without any error. Computing
once also avoids repeating a slow calibration on every process.

**How to apply.**

```python
from accelerate.utils import broadcast

if accelerator.is_main_process:
    values = torch.tensor(calibrate(dataset), dtype=torch.float64,
                          device=accelerator.device)
else:
    values = torch.zeros(3, dtype=torch.float64, device=accelerator.device)
values = broadcast(values, from_process=0)
```

Initialize weights on process 0 before `accelerator.prepare`; DDP broadcasts
process 0's parameters and buffers when it wraps the model. Metadata about
the initialization, such as which samples it used, is not a module buffer and
needs its own broadcast.

**Evidence.** An auxiliary regression refit from each process's own buffer
of recent samples gave each process a different loss; the fix fit it once
from the gathered buffers and broadcast the coefficients.

### Make every control-flow decision collectively

Any decision that changes whether a process runs a collective operation,
such as skipping an optimizer step on a non-finite gradient, stopping for a
time budget, or ending an epoch early, must be gathered across processes and
combined before anyone acts on it.

**Why.** If one process returns while the others continue, the others block
forever in the next all-reduce, with no error. If one process skips an
optimizer step that the others take, the replicas diverge.

**How to apply.** Stop if any process wants to stop; skip if any process
saw a non-finite gradient:

```python
def any_process(flag: bool) -> bool:
    t = torch.tensor([float(flag)], device=accelerator.device)
    return bool((accelerator.gather(t) > 0.5).any().item())
```

The gather is a host synchronization. Read the loss value for logging in the
same device-to-host copy (see [profiling](profiling-and-throughput.md)).

### Skip gradient synchronization on accumulation micro-steps

Wrap the forward and backward of every micro-step except the last in
`accelerator.no_sync(module)` for every DDP-wrapped module that receives
gradients, or use `accelerator.accumulate(model)` with accelerate's
accumulation setting. Without it, each accumulation cycle runs
`accum - 1` redundant all-reduces of the full gradient.

```python
from contextlib import ExitStack

def no_sync_all(modules, enabled):
    stack = ExitStack()
    if enabled and accelerator.num_processes > 1:
        for m in modules:
            stack.enter_context(accelerator.no_sync(m))
    return stack

with no_sync_all([model, feature_extractor], enabled=not at_boundary):
    with accelerator.autocast():
        loss = compute_loss(batch) / cycle_size
    accelerator.backward(loss)
```

**Evidence.** One configuration accumulated over 64 micro-batches, so 63 of
every 64 all-reduces carried gradients that the next micro-step overwrote.

### Call non-forward methods through `unwrap_model`

DDP exposes only `forward`. Calling another method on the wrapped model
fails:

```text
AttributeError: 'DistributedDataParallel' object has no attribute 'decode'
```

Call such methods on `accelerator.unwrap_model(model)`. The same route
suits no-grad forwards, such as auxiliary samples and evaluation, which gain
nothing from DDP's hooks. A forward that produces gradients must go through
the wrapper, or its gradients are never synchronized.

### Rotate the sampler each epoch and close the process group at exit

Call `dataloader.set_epoch(epoch)` at the start of every epoch; without it a
`DistributedSampler` gives each process the same shard in every epoch. Call
`accelerator.end_training()` before exiting to close NCCL cleanly, which
removes the `destroy_process_group()` warning at shutdown. Gate every file
write (configuration dump, checkpoints, pruning, sample grids) on
`accelerator.is_main_process`.

## Startup errors

### Freeze parameters that run only under `no_grad` before `prepare`

A module whose parameters require gradients but that runs only inside
`torch.no_grad()` stalls DDP's reducer, which waits for gradients that never
arrive. The error appears on the second iteration:

```text
RuntimeError: Expected to have finished reduction in the prior iteration
before starting a new one.
```

Set `requires_grad_(False)` on such modules at construction, before
`accelerator.prepare`. A pretrained VAE's encoder, used only to encode
targets under `no_grad`, caused this error until it was frozen.

### Gradient checkpointing inside DDP may need both `find_unused_parameters` and `static_graph`

With a gradient-checkpointed decoder inside the wrapped model, some decoder
parameters did not receive gradients every step. `find_unused_parameters`
alone then failed with a "marked as ready twice" error, and `static_graph`
alone failed on the unused parameters. Both together worked:

```python
from accelerate import Accelerator, DistributedDataParallelKwargs

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True,
                                           static_graph=True)
accelerator = Accelerator(mixed_precision="bf16",
                          kwargs_handlers=[ddp_kwargs])
```

Treat this as a record of one model's behavior, not a default: check whether
a new model needs either flag.

### A multi-process run may import packages a one-process run never touches

With the HuggingFace `datasets` package installed, `accelerator.prepare`
under `torchrun` failed with `ModuleNotFoundError: No module named
'pyarrow'`, because the cluster's package index had no `pyarrow`.
One-process runs did not reach that import, and the project did not use
`datasets`. Uninstall packages the training code does not need, and install
a one-off tool's dependencies into a throwaway prefix
(`pip install --target "$TMPDIR/tool" datasets`) rather than the training
environment. The failure returns whenever the environment is rebuilt from a
list that includes the package.

### Refuse to train on the CPU

A node whose GPU driver has failed can make accelerate select the CPU, and
training then continues at a small fraction of its normal speed until the
allocation ends. Check `accelerator.device.type == "cuda"` at the start of
training and raise otherwise, with an explicit override for intentional CPU
runs; see
[toolchains and GPU architectures](../clusters/toolchains-and-gpu-architectures.md#refuse-to-train-when-the-framework-falls-back-to-the-cpu).
