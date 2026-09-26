# Optimizer steps and EMA

An optimizer step is where a bad batch becomes permanent: one non-finite
gradient written into Adam's moment estimates poisons every later step.
These entries cover guarding the step and everything that consumes the same
batch, scaling losses under gradient accumulation, keeping learning-rate
schedules on their intended horizon, and handling the exponential moving
average (EMA) of the weights that the samples and evaluations use.

## Non-finite gradients

### Check the gradient norm after clipping and skip a non-finite step

`clip_grad_norm_` does not remove NaN or infinity: it scales every gradient
by a factor computed from the total norm, and a non-finite norm makes every
scaled gradient non-finite. Read the total norm it returns, and skip the
step (zeroing the gradients) when that norm is not finite on any process.

**Why.** A single Adam step with NaN gradients writes NaN into the weights
and into both moment estimates, and the moments never recover. Skipping a
rare bad step costs one batch; taking it ends the run.

**How to apply.** Count skips and print the first few, so a recurring cause
is visible in the log. Under DDP the decision must be collective (see
[multi-GPU](multi-gpu.md#make-every-control-flow-decision-collectively)):

```python
norm = accelerator.clip_grad_norm_(params, max_norm)
bad = (~torch.isfinite(norm)).float().reshape(1)
if accelerator.num_processes > 1:
    bad = accelerator.gather(bad)
if bool((bad > 0.5).any()):
    optimizer.zero_grad(set_to_none=True)
    skips += 1
    if accelerator.is_main_process and skips <= 10:
        print(f"non-finite grad norm; skipped step (total {skips})")
else:
    optimizer.step()
    ema.update()
```

`torch.nn.utils.clip_grad_norm_(..., error_if_nonfinite=True)` raises
instead, which suits a run that should stop at the first bad batch.

### Guard every consumer of the batch, not only the main optimizer

A skipped main step must also skip everything that consumed the same
non-finite values: auxiliary optimizers stepped in the same cycle, auxiliary
heads trained on targets computed from that batch, and buffers that feed a
later fit. Stage those per accumulation cycle and commit them only when the
guarded main step succeeds.

**Why.** A clamp does not repair NaN: `torch.clamp_min(nan, 1e-4)` is NaN.
An auxiliary head that learns from a NaN target outputs NaN from then on, and
if the main loss reads that head, every later loss is NaN. The guard then
skips every step, and the run is dead while its progress bar advances.

**How to apply.**

```python
staged = []                                  # this cycle's buffer rows
...
staged.append((features.detach(), targets.detach()))
if at_boundary:
    if step_model_guarded():                 # returns False on a skip
        step_auxiliary_heads(cache_targets)
        for row in staged:
            fit_buffer.push(*row)
    else:
        zero_auxiliary_grads()
    staged.clear()
    cache_targets.clear()
```

**Evidence.** An audit found the silent-death cascade through an unguarded
auxiliary head. The first fix guarded the head and its optimizer; a second
audit found the same cascade through the buffer of a periodically refit
regression, which the first fix had missed.

### Clamp before dividing, and log the unclamped value

A quantity that can go negative, such as a kernel mass estimated from an
unbiased but unbounded kernel, must be clamped before it is used as a
divisor or raised to a power. Record the raw value in diagnostics alongside
the clamped one.

**Why.** Dividing by `mass ** 0.5` with a negative mass gives NaN, which the
step guard catches. Dividing by `mass ** 1.0` with a negative mass flips the
sign of the update, which no guard catches. Diagnostics that record only the
clamped value show the floor and hide the negative values that caused the
flip.

**Evidence.** An audit found that a change had removed the clamp from one
normalization path, and that the debug log recorded the clamped value. The
log would have shown masses at the clamp floor while the divisor in use was
negative.

### Keep per-step diagnostics opt-in and free when off

An environment variable that switches on a per-step CSV of value ranges,
maxima, normalizer minima and the name of the first non-finite quantity is
the tool for locating the source of recurring skips. Keep every `.item()`
call of such diagnostics behind the switch, so the default run pays nothing
for them.

## Gradient accumulation

### Scale each micro-step by the size of its own cycle

Divide each micro-step's loss by the number of micro-steps in its cycle, and
treat the last batch of the epoch as a cycle boundary. A final partial cycle
of $k$ micro-steps divided by the nominal accumulation count is under-weighted
by $k$ over that count.

```python
cycle_start = (micro_step // accum) * accum
cycle_size = min(accum, len(loader) - cycle_start)
at_boundary = ((micro_step + 1) % accum == 0
               or micro_step + 1 == len(loader))
loss = compute_loss(batch) / cycle_size
```

**Evidence.** Until an audit fixed it, the final partial cycle of an epoch
was neither stepped at the end of the epoch nor scaled by its own size.

## Learning-rate schedules

### Clamp schedule progress to the schedule's horizon

Clamp the fraction of the schedule completed to at most 1. A cosine decay
computed from an unclamped fraction climbs back toward the base learning
rate once the step count passes the horizon.

```python
def lr_at(step, base, warmup, total):
    if warmup > 0 and step < warmup:
        return base * (step + 1) / warmup
    progress = min((step - warmup) / max(1, total - warmup), 1.0)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))
```

**Why.** The step count can exceed the horizon when a resumed run's epoch
length changes, when the configured epoch count is lowered for a
continuation, or when resumed links overrun.

### Compute schedules from the restored step, in absolute units

Evaluate every schedule from the global step restored from the checkpoint,
never from a per-process counter. When runs of different lengths must share
an early phase, such as a short preview and a long production run, express
the schedule in steps or epochs rather than as a fraction of the total
horizon; a fraction stretches the early phase with the run length.

**Evidence.** Bandwidth schedules defined in epochs gave preview and
converged runs the same early phase, and resumed links reconstructed the
schedule position from the restored step alone.

## EMA

### Keep the EMA shadow in fp32 and update it with foreach operations

Store the shadow copy of each parameter in fp32 whatever the parameter's
dtype, and update all of them with multi-tensor `torch._foreach_*`
operations instead of a Python loop over parameters.

```python
class EMA:
    def __init__(self, params, decay):
        self.decay = decay
        self.params = list(params)
        self.shadow = [p.detach().float().clone() for p in self.params]

    @torch.no_grad()
    def update(self):
        torch._foreach_mul_(self.shadow, self.decay)
        torch._foreach_add_(self.shadow, [p.float() for p in self.params],
                            alpha=1.0 - self.decay)
```

Update the EMA only after an optimizer step that was taken, inside the
branch that the non-finite guard allows.

**Evidence.** On a small 1D model, the foreach update cut the section of
the step holding backward, optimizer step and EMA update from 41.4 to
34.2 ms.

### Swap EMA weights in and out with `try`/`finally`

Sampling and evaluation copy the EMA weights into the live model and restore
the training weights afterward. Put the restore in a `finally` block, and
restore the module's train or eval mode as well.

```python
backup = [p.detach().clone() for p in params]
was_training = model.training
try:
    ema.copy_to(params)
    model.eval()
    samples = sample(model)
finally:
    for p, b in zip(params, backup):
        p.data.copy_(b)
    model.train(was_training)
```

**Why.** Without the `finally`, an exception during sampling, such as a
decoder OOM that the caller catches, leaves the EMA weights in the training
model, and training continues from them without any message.

### Save and apply every EMA shadow

Keep one EMA per trained module (main model, fine-tuned decoder, auxiliary
heads), save each shadow in the checkpoint next to the live weights, and
apply every one of them in evaluation tools.

**Why.** An evaluation that applies the main model's EMA but loads the raw
weights of a fine-tuned decoder produces samples that differ from the
training-time sample grids, which used both shadows.

**How to apply.** Store each shadow keyed by parameter name, or record the
parameter list whose order a positional shadow follows. A shadow saved as a
list of tensors cannot be passed to `load_state_dict`; a loader must zip it
with the same parameter list in the same order.

**Evidence.** A metric tool applied the model EMA and loaded the decoder's
raw weights; an analysis tool passed a positional shadow to
`load_state_dict` and crashed.

### A zero-initialized output head delays the EMA on short schedules

Zero-initializing a network's output layer, standard for adaptive-norm
transformers and many U-Nets, makes the first steps update only that layer.
The EMA then lags the live weights by its horizon of about
$1/(1-\mathrm{decay})$ steps, 1000 steps at decay 0.999. On a short
schedule, use a small nonzero output initialization or a shorter EMA
horizon; on a long schedule, zero initialization is appropriate.

**Evidence.** On a training budget of about 8k steps, a transformer with a
zero-initialized output head produced EMA samples that stayed near zero for
hundreds of steps, and the generated outputs collapsed toward zero. A
full-rank nonzero-initialized head removed the lag.
