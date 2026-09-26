# JAX references and cross-framework parity

Two JAX codebases served as references for PyTorch ports here: the official
release of a published one-step image generator, with its frozen feature
encoders and weights, and a conditional U-Net from another paper's code.
Each ported component was checked against the reference by running both on
the same inputs and comparing intermediate outputs. The first section covers
running the two frameworks side by side, the second converting weights, the
third the differences that parity tests found, and the fourth the parts of a
recipe that live outside the model.

## Running both frameworks

### Keep JAX and PyTorch in separate environments and exchange `.npz` files

Install the JAX reference in its own virtual environment (a CPU build of
`jax` and `flax` is enough for fp32 parity on small inputs) and split each
parity test into stages that run in one environment each and communicate
through `.npz` files: generate inputs, run the reference, run the port,
compare.

**Why.** Installing both frameworks in one environment couples their CUDA,
cuDNN and NumPy requirements. With files in between, neither environment
needs the other framework, and each stage's output stays on disk for
inspection.

**How to apply.**

```python
# python parity.py gen      (torch env: inputs, and weights if needed)
# python parity.py jax      (jax env: reference outputs)
# python parity.py torch    (torch env: port outputs)
# python parity.py compare  (either env)
if __name__ == "__main__":
    {"gen": gen, "jax": run_jax, "torch": run_torch,
     "compare": compare}[sys.argv[1]]()
```

Draw inputs from a seeded NumPy generator in the reference's layout (NHWC
for Flax), and permute to NCHW only inside the PyTorch stage.

**Evidence.** A training loss ported this way matched the reference to a
maximum relative error of about 5e-6 over four input cases, including
weighted negatives and an empty negative set; a 2.1B-parameter frozen
encoder matched to about 3e-5 on every output group.

### Import reference code that expects the other framework

The reference package here imported PyTorch at module level for a helper,
used package names such as `utils` that collide with modules already
imported, and wrapped arrays in distributed helpers that assume a device
mesh. Stub what
the import needs, remove the stub before running, and replace mesh helpers
with the identity:

```python
sys.modules["torch"] = types.ModuleType("torch")   # top-level import only
sys.path.insert(0, reference_dir)
sys.modules.pop("utils", None)                     # take the reference's
import models.convnext as ref
del sys.modules["torch"]            # einops looks for torch when it runs
ref.enforce_ddp = lambda x: x       # no device mesh on a CPU host
```

**Why.** `einops` checks `sys.modules` for each framework when it dispatches
an operation, so a stub left in place breaks the reference's first
`rearrange`.

### Run the reference in fp32

Turn off the reference's reduced-precision options for the parity run (a
`use_bf16` flag in its model configuration, `dtype=jnp.float32` on its
modules) and compare fp32 against fp32. bf16 rounding is larger than the
errors a parity test is meant to find. Port the reference's own precision
choices as well: where it computes attention or adaptive normalization in
fp32 regardless of the training dtype, run those sections outside autocast
in the port.

## Converting weights

### Transpose layouts, rename parameters and assert every shape

Flax stores a convolution kernel as `(kh, kw, in, out)` and a dense kernel
as `(in, out)`; PyTorch stores `(out, in, kh, kw)` and `(out, in)`. Flax
calls a normalization layer's multiplier `scale` where PyTorch uses
`weight`. Parameters with a layout of their own, such as a global response
norm's `gamma` and `beta`, are copied unchanged.

```python
# PyTorch state_dict entry -> Flax parameter
if v.ndim == 4:
    v = np.transpose(v, (2, 3, 1, 0))      # OIHW -> HWIO
elif not is_layout_free(key):
    v = np.transpose(v, tuple(reversed(range(v.ndim))))
assert v.shape == param.shape, f"{key}: {v.shape} vs {param.shape}"

# Flax -> PyTorch
w = torch.from_numpy(np.ascontiguousarray(kernel.transpose(3, 2, 0, 1)))
```

Reuse the reference's own name-mapping rules (reimplement its regular
expressions in the parity script) instead of writing a new mapping, and pop
each source key as it is consumed so that unused weights show up at the
end.

### Flatten Flax checkpoints in the JAX environment, pack in the other

Read a `msgpack` checkpoint with `flax.serialization.msgpack_restore`,
flatten the nested dictionary to `/`-joined keys, keep only the subtree the
port needs, and save it as `.npz`. A second stage in the PyTorch environment
maps names and layouts into a `state_dict`. The parity test on the
converted weights then checks the conversion and the port together.

### Compare parameter counts first

Count the port's parameters and compare them with the paper's or the
reference's before any numerical test. Two generator sizes matched the
published counts exactly (133.9M and 464.5M), which rules out missing or
extra layers before any output is compared.

## Differences that parity tests found

### `jax.image.resize` antialiases by default; `F.interpolate` does not

`jax.image.resize(..., method="bilinear")` applies an antialiasing filter
when it downsamples (`antialias=True` is its default).
`torch.nn.functional.interpolate(mode="bilinear")` does not unless it is
given `antialias=True`. Pass it explicitly, and reproduce the reference's
preprocessing order exactly (here: map from [-1, 1] to [0, 1] without
clipping, normalize by the ImageNet mean and standard deviation, then
resize from 256 to 224).

**Evidence.** A pretrained feature network loaded from a public model hub
was at first left without a parity test and listed as a known deviation.
Its parity test, added later, found every one of 13 feature groups in
disagreement; with `antialias=True` all 13 matched within 3.4e-5.

### Default initializations differ between the frameworks

Flax's `nn.Dense` and `nn.Conv`, like other JAX layer libraries, initialize
biases to zero by default; PyTorch's `nn.Linear` and `nn.Conv2d` draw them
from $U(\pm 1/\sqrt{\text{fan\_in}})$. A port that copies the reference's
explicit initializations inherits PyTorch's defaults everywhere else. Zero
every bias the reference zeroes.

**Evidence.** A U-Net port scaled the last weight of each modulation layer
by 0.1, as the reference does, so that residual blocks start close to the
identity, but left that layer's bias at PyTorch's default
$U(\pm 0.0625)$: random per-channel offsets at initialization, the term the
scaling was meant to suppress. The same port had a bias on an attention
output projection that the reference lacks. To keep existing checkpoints
loadable, the fix kept that bias as a parameter, zero-initialized and
frozen, which is equivalent to having no bias.

### Statistics over the local batch depend on the per-device batch size

A quantity normalized over the rows of a local batch, such as batch-norm
statistics or a scale computed from a batch's pairwise distances, changes
with the number of rows each device holds. A port that runs the global
batch on fewer devices with gradient accumulation matches the reference
only if each accumulation chunk holds the reference's per-device batch.

**Evidence.** The reference computed its loss normalization over 4 rows per
host (128 rows over 32 hosts); the port processed chunks of 4 label rows to
match. When memory forced chunks of 2, the normalization averaged over 2
rows; the difference was accepted because every spatial token counted as a
row, which left thousands of rows per chunk.

## Matching the recipe

### Read the reference's training script defaults, not only its configs

Settings that the configuration files leave out come from the training
script's argument defaults. The reference's gradient-clipping threshold of
2.0 was one. Build a side-by-side table of the reference's and the port's
optimizer, schedule, EMA, loss and metric settings, and mark each deviation
as deliberate or pending. Check library defaults the same way: `optax.adamw`
applies weight decay to every parameter unless a `mask` is given, so a port
that exempts biases and norms from decay does not match a reference that
passes no mask.

### Use the reference's metric network and sanity-check it on real images

FID computed with torchvision's Inception weights and `transform_input` is
not the FID of the TensorFlow Inception network that published numbers
use; relative comparisons under one implementation hold, absolute numbers do
not transfer. For published comparisons, use the reference's metric network
(a TorchScript export of the TensorFlow Inception), its reference
statistics and its sampling protocol.

Before scoring a model, score real held-out images against the reference
statistics at two or more sample counts. FID is biased upward at small
sample counts, so the value must fall as the count grows.

**Evidence.** Real images scored 26.2 at 2,000 samples and 9.45 at 10,000.

### Parity thresholds

Compare each named output group separately, report the maximum error, and
list groups missing from either side. For feature maps, divide the maximum
absolute difference by the reference's maximum magnitude and require less
than 1e-3; for scalar losses and diagnostics, require an elementwise
relative error below 1e-4. Correct ports landed 20 to 30 times below these
thresholds, and the resize bug above failed them in every group.
