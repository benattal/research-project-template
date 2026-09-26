# Profiling and throughput

Throughput work here followed one pattern: time each section of the training
step, remove host synchronizations and redundant work, rewrite the few
kernels that materialize large intermediates, and verify that the numbers
did not change. The first section covers measurement, the second the changes
that paid off, the third how to keep a speed change from changing the
results.

## Measuring

### Time sections with CUDA events and synchronize once per step

Record a pair of CUDA events around each section of the step, synchronize
once when the step ends, and read the elapsed times then. Exclude warm-up
steps, time the whole step with a synchronization on each side, and record
peak allocated memory.

```python
class SectionTimer:
    def __init__(self):
        self.open, self.times = [], defaultdict(list)

    @contextmanager
    def section(self, name):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        self.open.append((name, start, end))

    def flush(self):
        torch.cuda.synchronize()
        for name, start, end in self.open:
            self.times[name].append(start.elapsed_time(end))   # ms
        self.open = []
```

Event timing does not stall the host between sections, so the measured step
keeps its normal overlap of host and device work. For kernel-level detail,
wrap the same loop in `torch.profiler.profile` with
`schedule(wait=0, warmup=W, active=A)` and a TensorBoard trace handler.

### Read utilization over the training loop, not over the job

Sample `nvidia-smi --query-gpu=utilization.gpu,memory.used
--format=csv,noheader,nounits -lms 200` in the background and summarize
percentiles over the steady training steps.

**Why.** A mean over the whole job includes dataset construction,
calibration and pretrained-model loading, and mostly measures how much of
the job is spent outside the loop.

**Evidence.** In one profile the job-wide mean GPU utilization was 12.8%,
while the 90th percentile within the training loop was 86%.

### Drive the profiler through the training step's public interface

A profiling script that called eight private methods of the trainer broke
when the trainer was split into separate owners for inputs, loss and
optimizer steps: six of those methods no longer existed. Profile through
the same entry point the training loop calls, and run a one-step profile in
the smoke tests so that a refactor that breaks the profiler fails there.

## Changes that paid off

### Replace [B, B, D] broadcasts with matrix products

Pairwise quantities over a batch of $B$ samples with $D$ features, written
as broadcasts, materialize $B \times B \times D$ intermediates. Rewrite them
as matrix products plus per-row scaling:

- a kernel-weighted mean of differences,
  $\sum_j k_{ij}(y_j - x_i) = (K y)_i - (\sum_j k_{ij}) x_i$;
- a product of two differences,
  $(g - a)\cdot(g - b) = \lVert g \rVert^2 - g \cdot (a + b) + a \cdot b$;
- squared distances,
  $\lVert x - y \rVert^2 = \lVert x \rVert^2 + \lVert y \rVert^2 - 2 x \cdot y$.

These expansions cancel large terms, so run them in fp32 outside autocast
(see [mixed precision](mixed-precision-and-memory.md#escape-autocast-for-cancellation-sensitive-math)).

**Evidence.** At $B = 8$ and $D = 196{,}608$ the section fell from 19.5 to
9.6 ms and two 50 MiB intermediates per step disappeared. At $D = 256$ the
gain was under 1 ms. The rewritten functions agreed with the originals to
within 2e-6 in fp32.

### Remove host synchronizations from the loop

Every `.item()`, `float(tensor)`, Python `if` on a device tensor, and
`torch.equal` on device tensors waits for the GPU to finish all queued work,
which stops the host from queueing the next kernels. Keep one synchronization
per step at most:

- accumulate losses as device tensors, and read them for display every N
  steps and at epoch end;
- decide configuration-dependent branches on the host from the
  configuration, not from `tensor.max() > 0`;
- detect properties that are constant for a run once, not every step;
- when a per-step read is unavoidable, such as the collective
  non-finite-gradient flag, read the loss in the same device-to-host copy.

```python
# The guard needs its flag every step; the loss rides along in one copy.
bad_value, loss_value = torch.stack([bad.float(), loss.detach()]).tolist()
if global_step % 20 == 0:
    pbar.set_postfix(loss=f"{loss_value:.4f}")
```

**Evidence.** An audit estimated the extra per-step reads at 1 to 3% of step
time. With a progress bar refreshed every 20 steps and its output piped
through `tee`, the log repeats the same loss for 20 lines; training is not
stalled.

### Batch independent no-grad forwards

When a step runs K no-grad forward passes that share their conditioning and
differ only in noise, run them as one forward over a K-times larger batch
and split the output.

```python
cond_k = cond.repeat_interleave(K, dim=0)          # [K*B, ...]
noise = torch.randn(K * B, *shape, device=device)
with torch.no_grad():
    out = model(noise, cond_k).view(B, K, *out_shape)
```

**Evidence.** With K = 8 the auxiliary forwards were about half the step;
the estimated gain was 5 to 20% of step time from fewer launches and better
occupancy, and about zero at K = 1. The change alters the shape of the noise
draws (see the last section).

### Drop work whose output is discarded

Audit for computations whose result a mask discards for every row under a
given configuration. An auxiliary no-grad forward whose output `torch.where`
replaced in every row, because the batch composition guaranteed the other
branch, cost an estimated 20 to 25% of the backbone compute of each step.

### Use `channels_last` for convolutional backbones under bf16

Convert convolution-heavy models with
`model.to(memory_format=torch.channels_last)` and make their inputs
`x.contiguous(memory_format=torch.channels_last)`.

**Evidence.** For a U-Net of 3×3 convolutions at batch 64 and 64×64
resolution under bf16 on an H100, the no-grad forward went from 29.2 to
25.5 ms (1.15×) and forward plus backward from 90.5 to 82.7 ms (1.10×).
Channel-wise layer norms and attention reshapes worked unchanged.

### Vectorize Python loops in launch-bound models

A small model is limited by kernel launches and Python overhead rather than
by arithmetic, so the host-side code is what to optimize.

**Evidence.** Replacing a Python double loop that built polynomial features
with vectorized tensor operations cut its call from 3.75 to 0.20 ms, with
identical outputs, and raised training from 21.5 to about 25 iterations per
second. In an earlier profile, a 1D U-Net at batch 256 reached 65% GPU
utilization (90th percentile) after all other changes; larger batches, CUDA
graph capture and `torch.compile` were the remaining options, and none was
tried.

### Smaller wins

- `torch.var_mean` computes both moments in one pass; replacing separate
  `mean` and `var` calls in a channel norm used 36 times per forward saved
  about 2% of the forward.
- Register constant tables, such as sinusoidal embedding frequencies, as
  non-persistent buffers instead of rebuilding them in every forward.
- In an evaluation sweep, build the metric network once and reuse it; a
  25M-parameter Inception network was rebuilt at every sweep point.
- Inside a scheduler allocation, set BLAS threads to the allocated CPUs. A
  metric job that computed covariances and a matrix square root of
  2048×2048 matrices ran with a one-thread cap copied from login-node
  settings.

## Keeping speed changes honest

### Classify each speed change by what it changes numerically

Sort each speed change into one of three classes and land the classes
separately:

1. bit-identical: caching, removing synchronizations, I/O;
2. equal in expectation but not bit-identical: memory-format changes and
   reassociated arithmetic, batched draws that change the shape of the
   random stream, removal of draws whose results were discarded;
3. changes to the data stream, the objective or an evaluation protocol.

**Why.** Class 1 needs a matched-seed parity check; class 2 needs a
trajectory comparison and breaks bit-level comparability with earlier runs,
including resumed links of a running chain; class 3 is a method change and
needs its own decision.

**Evidence.** In one profile, a change described as preserving numerics
included the persistent-worker option that froze the training noise (see
[data loading](data-loading.md#persistent-workers-freeze-per-epoch-dataset-state)).
The step-time gain it reported was real; the samples were worse.

### Headline numbers from one profiling pass

On an L40S, a 256×256 latent-space image model went from 790 to 660 ms per
step (−17%): 102 ms from turning off decoder gradient checkpointing, 13 ms
from loader options, 10 ms from matrix-product rewrites and 5 ms from
reusing targets instead of recomputing them. A small 1D model went from 143
to 129 ms (−10%), mostly from the foreach EMA update (7 ms) and loader
options (4 ms). Part of the loader gain came from the persistent-worker
option later removed for correctness.
