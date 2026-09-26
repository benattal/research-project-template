# Mixed precision and memory

bf16 autocast through accelerate halves activation memory and speeds up large
convolutions and matrix products, but it acts per operation, only inside an
autocast context. Most memory failures in these runs came from three sources:
a submodule that ran outside autocast, autograd graphs kept alive longer than
needed, and transient tensors whose size grew with the square of a feature
dimension. The first section covers precision, the second memory budgeting
and out-of-memory (OOM) diagnosis.

## Mixed precision

### Run every forward and loss inside autocast, frozen submodules included

Wrap the whole forward pass and the loss in `accelerator.autocast()`,
including frozen pretrained modules that sit between the trainable model and
the loss, such as a VAE decoder that maps generated latents to pixels.

**Why.** `Accelerator(mixed_precision="bf16")` applies autocast to the
`forward` of each prepared model. A frozen decoder called through
`accelerator.unwrap_model(model).decode(...)`, or held as a separate module,
runs in fp32 unless the caller opens an autocast context. In the loss path
the fp32 decoder saves fp32 activations for the backward pass, about twice
the bf16 footprint.

**How to apply.** Keep every new forward path, such as a custom sampler or an
auxiliary loss, inside the same context:

```python
with accelerator.autocast():
    z = model(noise, cond)                              # prepared, trainable
    x = accelerator.unwrap_model(model).decode(z)       # frozen, same context
    loss = F.mse_loss(x, target.detach())
accelerator.backward(loss)
```

**Evidence.** Without the context, a latent-space image model ran out of
memory at batch size 16 on a 44 GiB L40S; with it, the same batch fit. This
was the most frequent OOM cause in the project.

### Escape autocast for cancellation-sensitive math

Calling `.float()` on inputs does not keep a matrix product in fp32 inside an
autocast region: autocast casts the inputs of the operations it lists
(`matmul`, `linear`, convolutions) to bf16 whatever their dtype. Code that
needs fp32 must disable autocast for its own scope.

**Why.** Pairwise squared distances computed through the expansion
$\lVert x \rVert^2 + \lVert y \rVert^2 - 2 x \cdot y$ subtract a cross term
from two large self terms. With the cross term from a bf16 product (8
significant bits) and the self terms in fp32, the difference loses most of
its digits whenever two points are close relative to their norms. At feature
dimensions from 12k to 200k the rounding error exceeded the small distances
that matter most to a kernel.

**How to apply.** Leave the model forward under autocast and decorate only
the numerically sensitive block:

```python
@torch.autocast("cuda", enabled=False)
def pairwise_sq_dist(x, y):
    x, y = x.float(), y.float()
    return (x * x).sum(-1)[:, None] + (y * y).sum(-1)[None] - 2 * x @ y.T
```

The fp32 block was a small fraction of step time.

**Evidence.** Before the decorator, kernel features were cast with `.float()`
and still reached bf16 GEMMs. After it, a unit test confirmed that the loss
target computed inside a bf16 autocast region was bit-identical to the one
computed outside.

### Under autocast, parameters and the residual stream stay fp32

Autocast changes the dtype of the inputs of each listed operation, not of
stored tensors. Parameters stay fp32, and a residual stream built by adding
fp32 tensors stays fp32; only the outputs of listed operations are bf16. A
line such as `pos = self.pos_embed.to(x.dtype)` does nothing when `x`
arrives in fp32.

**How to apply.** Where precision matters, print the dtypes of the
intermediates instead of assuming that "the model runs in bf16". Reference
implementations sometimes compute attention or adaptive-normalization
modulation in fp32 on purpose. A port that needs parity reproduces those
choices with autocast disabled for the block; otherwise autocast runs them in
bf16.

**Evidence.** An audit found that a cast of positional embeddings "to the
input dtype" had no effect under bf16 autocast, and that the transformer's
residual stream ran in fp32 throughout.

### Measure before enabling bf16 on a small model

Choose mixed precision per model, by measurement. bf16 autocast slowed a
small model.

**Why.** Autocast adds per-operation work: dtype checks, casts, and a cache
of cast weights. A small model is limited by kernel launches rather than by
arithmetic, so the faster bf16 matrix products save less than the casts
cost.

**Evidence.** A 1D U-Net with 8.3M parameters at batch 256 went from 131 to
138 ms per step with bf16 autocast on an L40S. The image models in the same
codebase ran out of memory without bf16 and used it by default.

### Keep frozen metric networks in fp32 and generate as in training

For a metric that passes images through a frozen network, such as Inception
features for FID, run the metric network in fp32, and generate the samples
under the autocast setting the model was trained with.

**Why.** A generator trained under bf16 autocast and sampled in fp32 differs
from the trained function at the rounding level. The metric network is a
fixed measuring instrument, and its own rounding adds error to every score.

**Evidence.** Inception features computed in bf16 differed from the fp32
features by about 1.9%, enough to move FID by more than the differences
being compared.

## Memory

### Measure peak memory on an idle device

Reset the allocator's peak counter before the measured steps and read it
after them; do not read `nvidia-smi`, which includes the caching allocator's
reserve and every other process on the device.

```python
torch.cuda.reset_peak_memory_stats()
for _ in range(active_steps):
    train_step()
torch.cuda.synchronize()
peak_gib = torch.cuda.max_memory_allocated() / 2**30
```

**Evidence.** The first benchmark of a large configuration ran on a shared
GPU where other processes held about 17 GB. Its OOM said nothing about the
configuration, and the benchmark had to be repeated on an idle 80 GB H100.

### Freeze every parameter that no optimizer steps

Call `requires_grad_(False)` on every module without an optimizer. Gradients
still flow through a frozen module to its inputs.

**Why.** Backward accumulates `.grad` for every parameter that requires
gradients. With no optimizer to zero them, the buffers occupy memory the size
of the module and their values grow without bound. Under DDP the same
parameters also trigger reducer errors (see [multi-GPU](multi-gpu.md)).

**Evidence.** A pretrained decoder in the loss path, configured with a
fine-tuning learning rate of zero, kept `requires_grad=True` and accumulated
gradients every step that nothing ever cleared.

### Checkpoint a frozen decoder only when memory requires it

Gradient checkpointing a large frozen decoder in the loss path trades a full
recompute in backward for a modest memory saving. Leave it off by default and
enable it when peak memory is the constraint.

**Evidence.** For 256×256 images at batch 8 on a 44 GiB L40S, turning
decoder checkpointing off cut backward from 404 to 303 ms, about 13% of the
790 ms step, and raised peak allocated memory from 22.0 to 23.4 GiB. It was
the largest single speedup in that profile.

### Backpropagate a many-term loss in two phases

When a loss sums many terms computed from large intermediate features, for
example one loss per feature group of a frozen extractor, backpropagate each
term into a detached copy of its features, free that term's graph, and then
run one backward from the features through the extractor and the generator
with the accumulated gradients. The gradients equal those of one backward
through the summed loss.

**Why.** In a single backward, every term's saved tensors stay alive until
the backward starts. The two-phase form holds one term's graph at a time. The
model sees exactly one backward call, so DDP gradient synchronization and
`no_sync` scopes behave as before.

**How to apply.**

```python
feats = extractor(generator(noise))      # feature groups, graph attached
grads = {}
for key, f in feats.items():
    leaf = f.detach().requires_grad_(True)
    term_loss(leaf, reference[key]).backward()   # frees this term's graph
    grads[key] = leaf.grad
keys = list(feats)
torch.autograd.backward([feats[k] for k in keys], [grads[k] for k in keys])
```

**Evidence.** With about 85 feature groups, each term held 20 to 30 fp32
tensors of shape (rows, generated, reference), over 100 MB each for the
largest groups. A configuration that ran out of memory at about 78 GB on an
80 GB H100 ran with a 45.5 GB peak after the change. The gradients matched
the single backward bit for bit on CPU.

### Whole-module activation checkpointing fails while loss graphs are alive

Wrapping the entire feature extractor in `torch.utils.checkpoint` did not
prevent the OOM above. During the backward recompute the extractor's full
graph materializes at once, while the loss terms' saved tensors are still
held. Checkpointing each block inside the extractor, combined with the
two-phase backward, did.

### Chunk no-grad forwards over large reference sets

Run a large no-grad batch, such as a bank of reference images passed through
a frozen feature extractor, in fixed chunks and concatenate the outputs. In
the OOM sequence above, the first failure was this bank forward; chunks of 64
images removed it.

### Store large replay banks as uint8

Per-class ring buffers of reference images held in fp32 would have needed
about 100 GB per process. Stored as uint8 and converted per batch, they fit;
the quantization error is smaller than JPEG decoding error.

### Load checkpoints to host memory first

Load with `torch.load(path, map_location="cpu")` and copy the state dicts
into the constructed model. Mapping a multi-gigabyte checkpoint directly to
the GPU places a second full copy of the weights on the device while the
model and optimizers are being built.

### Load only the submodule you need from a pretrained pipeline

Load a pipeline component from its subfolder instead of deserializing the
whole pipeline and keeping one attribute:

```python
from diffusers import VQModel
vqvae = VQModel.from_pretrained(repo_id, subfolder="vqvae")
```

The full latent-diffusion pipeline deserialized a 1.1 GB U-Net that was then
discarded, in every process at every job start. A warning such as
`No file named diffusion_pytorch_model.safetensors` on an older `.bin`
checkpoint is harmless.

### Run sampling and evaluation under `torch.no_grad`, encoders included

A trainable conditioning encoder called outside `torch.no_grad()` during a
periodic sample dump kept its autograd graph and activations alive for the
whole dump. Wrap every sampling path in `no_grad`, not only the generator
call.

### Solve a least-squares fit in its smaller dimension

A ridge regression over $p$-dimensional features that forms the $p \times p$
normal equations is quadratic in memory and cubic in time. When the number
of samples $n$ is much smaller than $p$, solve the $n \times n$ dual system,
in fp64, with a ridge scaled to the matrix's magnitude.

**Why.** An absolute ridge such as $10^{-6}$ falls below fp32 resolution once
the Gram matrix entries are large, so the regularized system stays singular
on small sample pools.

**How to apply.**

```python
# X: [n, p] features, y: [n] targets, n << p
K = (X @ X.T).double()
ridge = 1e-6 * K.diagonal().mean()
eye = torch.eye(n, dtype=K.dtype, device=K.device)
alpha = torch.linalg.solve(K + ridge * eye, y.double())
w = (X.double().T @ alpha).float()
```

**Evidence.** The primal system needed a 2.4 GB transient and about 5 TFLOP
per solve at $p$ of 24,577 and would have needed 619 GB at the pixel-space
dimension. The dual solve used about 100 MB regardless of $p$. On a small
test problem the held-out error dropped from 17.9 (fp32 primal) to 0.52
(fp64 dual).
