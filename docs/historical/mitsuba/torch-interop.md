# DrJit, PyTorch and tiny-cuda-nn

A neural field queried at path vertices runs as a tiny-cuda-nn module under
PyTorch, while the path tracer that produces its inputs and consumes its
outputs runs in DrJit. `dr.wrap` connects the two autodiff systems. This file
covers the bridge itself, the three ways gradients flow through it, and the
constraints that come from tiny-cuda-nn sizing its launches on the host.

## The bridge

### Bridge DrJit and PyTorch with `dr.wrap`

`@dr.wrap(source="drjit", target="torch")` turns a PyTorch function into one
that takes and returns DrJit arrays, exchanging memory through DLPack without
copies, and connects DrJit's AD graph to PyTorch's autograd.

**How to apply.** Pass each input channel as a separate `mi.Float`, stack
them into a `(batch, channels)` tensor inside the wrapped function, and
return `(channels, batch)` so the caller can split the result back into
`mi.Float`s:

```python
@dr.wrap(source="drjit", target="torch")
def forward(*channels):
    parts = [c.unsqueeze(0) if c.dim() == 1 else c for c in channels]
    batch = max(p.shape[-1] for p in parts)
    parts = [p.expand(p.shape[0], batch).contiguous()
             if p.shape[-1] != batch else p for p in parts]
    x = torch.cat(parts, dim=0).t().contiguous().float()
    return module(x).float().t().contiguous()

out = forward(p.x, p.y, p.z)
r, g, b = (mi.Float(out[i]) for i in range(3))
```

The broadcast step matters: DrJit passes width-one arrays unchanged (for
example a constant coordinate in a planar slice render), and concatenating
them with full-width tensors fails or misaligns the batch.

### The bridge's backward runs only when a DrJit input is AD-enabled

A network whose parameters live in PyTorch receives parameter gradients from
a DrJit loss only if at least one DrJit input to the wrapped call has
gradients enabled. Without one, `dr.wrap` records no AD operation and
`dr.backward` never reaches PyTorch.

**How to apply.** Distinguish three calls:

- **Genuine input derivatives.** The inputs are AD-attached (for example,
  hit points that move with a scene parameter). Call the bridge as is.
- **Parameter gradients only.** Mark the inputs as AD leaves with
  `dr.enable_grad` so the custom operation exists, and detach the packed
  PyTorch input inside the wrapped function so tiny-cuda-nn does not compute
  input derivatives that are then discarded.
- **Inference.** No leaves, and the wrapped function runs under
  `torch.no_grad()`. This path is also what prevents the retention leak
  described in [memory](memory.md).

```python
args = tuple(mi.Float(a) for a in args)
genuine = any(dr.grad_enabled(a) for a in args)
for a in args:
    dr.enable_grad(a)                  # no effect under dr.suspend_grad()
if not any(dr.grad_enabled(a) for a in args):
    out = forward_no_grad(*args)       # inference
elif genuine:
    out = forward(*args)               # input and parameter derivatives
else:
    out = forward_parameters(*args)    # packed input detached inside
```

**Evidence.** Detaching the packed input for parameter-only calls cut a
training microstep at 131,072 queries from 4.7–5.8 to 4.1–4.7 milliseconds
(11 to 25% depending on the network), and whole iterations by 2.7 to 4.7%.
Forward values and genuine input derivatives were identical; parameter
gradients differed by at most 8.7e-11 and Adam updates by at most 2.4e-9.

### Keep samples on the device across the boundary

Every tensor that crosses `dr.wrap` stays in device memory. Converting
through NumPy (`np.asarray(x)`, `.numpy()`) or reading scalars (`.item()`)
inside the training loop moves data to the host and synchronizes; see
[evaluation](evaluation-and-loops.md).

### Fewer crossings are not automatically faster

Packing all channels into one tensor before crossing the bridge reduces
DLPack exchanges but adds packing launches, and at the batch sizes of a
path-traced training step the two cancel.

**Evidence.** A one-tensor bridge gave identical outputs and parameter
gradients at widths 17, 1,024 and 262,144 (input gradients within 3.7e-9),
and was no faster over whole steps: 4.54 and 4.51 seconds for two packing
layouts against 4.44 for the per-channel bridge. The fixed host cost of a
network call, about 1.5 milliseconds, mattered more than the crossing count.

## tiny-cuda-nn constraints

### tiny-cuda-nn sizes its launches from the host tensor shape

A tiny-cuda-nn module reads its batch size from `x.shape[0]` on the host, so
any data-dependent batch (a compacted set of live lanes) requires a host read
of the count before the call. Keep the resulting widths in a quantized set
(see [JIT specialization](jit-and-kernel-caches.md)).

PyTorch 2.7 offered no device-only way around this: eager `torch.cond`
evaluates its predicate in Python, and the compiled conditional emits `.item()`
to move a tensor predicate to the host. CUDA graph capture did not help
either: capturing a tiny-cuda-nn forward and backward at a second batch width
failed with an event-record capture error, with or without conditional
graphs and after pre-warming at four times the width. Plain PyTorch layers
captured and replayed correctly under the same device predicate.

### A fused module exposes one parameter buffer

A tiny-cuda-nn `NetworkWithInputEncoding` presents its encoding and MLP
weights as a single parameter tensor, so per-layer learning rates or weight
decay through PyTorch parameter groups are not available. Build separate
encoding and network modules when they need different optimizer settings.

### Hash-grid interpolation changes the function without changing shapes

A hash-grid encoding with `Linear` interpolation and one with `Smoothstep`
interpolation have identical parameter shapes, so a checkpoint loads into
either without error, yet the represented function and its input derivative
differ. Record the interpolation mode (and every other encoding setting that
leaves shapes unchanged) in the checkpoint and check it on load.

### Report the batch shape on tiny-cuda-nn allocation failures

tiny-cuda-nn sizes its workspace from the batch, and an allocation failure
surfaces as a bare `cuMemCreate` error. Catch it at the call and re-raise
with the input shape and the free device memory:

```python
try:
    y = module(x)
except RuntimeError as e:
    free, total = torch.cuda.mem_get_info()
    raise RuntimeError(
        f"network forward failed at batch {tuple(x.shape)}; "
        f"GPU free {free / 2**30:.2f} of {total / 2**30:.2f} GiB") from e
```

### Build tiny-cuda-nn for the precision you train in

A tiny-cuda-nn build in fp32 runs fp32 end to end, with no loss scaling and
no half-precision copy of the weights to keep in sync with the optimizer. The
hash-grid backward accumulates with atomic adds in either precision, so
training is not bit-reproducible (see
[randomness and independence](randomness-and-independence.md)).

When installing, pin the PyTorch build to the CUDA toolkit that will compile
tiny-cuda-nn and install related packages in the same step: a later install
of a package that depends on PyTorch can resolve a newer PyTorch built for a
different CUDA major version, after which the tiny-cuda-nn extension no
longer links.
