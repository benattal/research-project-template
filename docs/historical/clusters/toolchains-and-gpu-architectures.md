# Toolchains and GPU architectures

On clusters with an environment-module system such as Lmod, the compiler,
CUDA toolkit, Python and some Python packages come from modules, and a
virtualenv adds the rest. Compiled extensions then tie the torch wheel, the
CUDA toolkit and the GPU architecture together, and a node whose GPU is newer
than the installed wheels fails in ways that depend on which library
launches the kernel. The entries below cover loading modules, building wheels
and extensions, and running on mixed GPU generations.

## Modules and virtualenvs

### Load the launcher's modules before using the virtualenv

A virtualenv created from a module-provided Python works only with the same
module set loaded. An import error for a module-provided package outside a
job ("No module named faiss", "No module named 'numpy'") means the modules
are not loaded; it does not mean the work has to go through the scheduler.

**Why.** Packages that a module provides, such as NumPy and SciPy from a
SciPy-stack module or FAISS from its own module, reach the interpreter
through the `PYTHONPATH` that `module load` sets. They are not installed in
the virtualenv.

**How to apply.** Copy the module set from the batch launcher, which is the
maintained list, and run in a login shell:

```bash
bash -lc 'module load gcc/12.3 cuda/12.6 python/3.11 scipy-stack faiss
          venv/bin/python -m tools.probe --help'
```

**Evidence.** A four-minute GPU job waited hours in the queue after a missing
FAISS import on a login node was read as "needs a compute node".

### `module` exists only in login shells

Start every script that loads modules as a login shell: `#!/bin/bash -l` for
batch and background scripts, `bash -lc '...'` for one-off commands.

**Why.** The module system defines `module` as a shell function from the
login profile. A non-login shell (`nohup bash script.sh`, `bash -c`) does
not have it, and sourcing the module system's profile script from a
non-login shell did not define it on the systems in use. The failure
signature is `module: command not found` followed by import errors.

### Append to `PYTHONPATH`; never assign it

Add the repository root with `PYTHONPATH="$PWD:${PYTHONPATH:-}"`. An
assignment such as `PYTHONPATH=$PWD` drops the module paths.

**Why.** The module system puts its Python packages on `PYTHONPATH`. The
repository root is needed as well: `python -m pkg.mod` puts the working
directory on `sys.path`, but `python path/to/file.py` puts the file's own
directory there instead, so a script run by path cannot import the
repository's package without `PYTHONPATH`.

**Evidence.** After an assignment, `No module named 'numpy'` appeared after
other compiled packages had already imported, which pointed away from the
environment as the cause.

### Load one module set everywhere

Install scripts, batch launchers, interactive allocations and login-shell
commands must load the same module list, from one toolchain generation.
Python, compiler, CUDA, SciPy-stack and FAISS modules are built against each
other; an environment installed under one set and run under another fails
at import or at the first kernel. Keep the list in the batch launcher and
copy from there, not from old logs.

### A module-provided pip may refuse manylinux wheels

Some cluster Python builds disable the manylinux platform tags, so pip
refuses the binary wheels that PyPI publishes. Download the wheel with
explicit platform tags, rename it to the `linux_x86_64` tag, and install the
local file.

**Why.** The site configures pip for its own wheelhouse, whose wheels carry
`linux_x86_64` tags. `--isolated` and an explicit `--index-url` keep that
site configuration from redirecting the download.

**How to apply.** Try `manylinux_2_28` first, since newer releases publish
only that tag, and fall back to `manylinux2014`:

```bash
py=311
for plat in manylinux_2_28_x86_64 manylinux2014_x86_64; do
    pip download --isolated --index-url https://pypi.org/simple/ --no-deps \
        --platform "$plat" --python-version "$py" --implementation cp \
        --abi "cp$py" --only-binary :all: --dest "$WHEELS" "pkg==1.2.3" &&
        break
done
whl=$(ls -t "$WHEELS"/pkg-*.whl | head -1)
new=$(basename "$whl" |
      sed -E 's/-manylinux[^.]+(\.[^.]+)*\.whl/-linux_x86_64.whl/')
cp "$whl" "$WHEELS/$new" && pip install --no-deps "$WHEELS/$new"
```

Install the packages that the site wheelhouse carries from the wheelhouse
(`pip install --no-index ...`). Its pins then decide related versions: a
wheelhouse `torch-cluster` pinned to one torch minor release fixes the torch
version.

### Fetch sources on the login node; build inside a GPU job

Compute nodes on many clusters have no internet access, and extension builds
need a visible GPU or an explicit architecture list. Clone sources with
their submodules on a login node into the repository's vendored directory,
then compile in a GPU job that loads the same modules as the runs.

## Compiled extensions

### Match torch's CUDA version to the toolkit that builds the extension

An extension compiled against both torch and `nvcc` needs them at the same
CUDA major.minor version, for example torch cu128 with `nvcc` 12.8. Install
`torchvision` and other torch companions in the same `pip` command as torch,
then assert the versions.

**Why.** A later install of a package that depends on torch can resolve a
newer torch from PyPI with a different CUDA runtime, replacing the pinned
one without an error.

**How to apply.**

```bash
pip install "torch==2.7.*" "torchvision==0.22.*" \
    --index-url https://download.pytorch.org/whl/cu128
python - <<'EOF'
import torch
assert torch.__version__.startswith("2.7."), torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
EOF
```

**Evidence.** Installing `lpips` after torch pulled a torch built for
CUDA 13, and the tiny-cuda-nn build then refused to link against `nvcc` 12.8.

### Pin `setuptools<81` for extensions that import `pkg_resources`

Recent setuptools releases removed `pkg_resources`, and build scripts that
still import it fail with
`ModuleNotFoundError: No module named 'pkg_resources'`. With
`--no-build-isolation`, the build uses the environment's setuptools, so pin
it in the environment: `pip install "setuptools<81"`.

### Name the GPU architectures when the build host has no GPU

Set the extension's architecture list explicitly whenever the build may run
without a visible GPU (for tiny-cuda-nn, `TCNN_CUDA_ARCHITECTURES="86;89;90"`),
and check the wheel's CUDA version rather than its architecture list.

**Why.** Architecture detection queries the GPU. `torch.cuda.get_arch_list()`
returns an empty list when CUDA cannot initialize, as in a build job with no
GPU allocated, so a check on it passes or fails for the wrong reason.
`torch.version.cuda` does not depend on a device.

### Test that the build honored its precision flag

After building with a precision setting, such as tiny-cuda-nn's
`TCNN_HALF_PRECISION=0` for fp32, run one forward and backward pass and
assert the output type. A flag that the build ignored is otherwise invisible
until results drift.

```bash
python - <<'EOF'
import torch, tinycudann as tcnn
m = tcnn.NetworkWithInputEncoding(3, 3,
    {"otype": "HashGrid", "n_levels": 4, "n_features_per_level": 2,
     "log2_hashmap_size": 14, "base_resolution": 16, "per_level_scale": 2.0},
    {"otype": "CutlassMLP", "activation": "ReLU", "output_activation": "None",
     "n_neurons": 32, "n_hidden_layers": 2}).cuda()
y = m(torch.rand(8, 3, device="cuda"))
assert y.dtype == torch.float32, y.dtype
y.sum().backward()
EOF
```

### Standalone GPU images may lack the OptiX driver library

Cloud GPU images often install the compute driver without
`libnvoptix.so.1`, and OptiX-based renderers then fail to initialize their
CUDA backend. Check with `ldconfig -p | grep libnvoptix.so.1`, and install
the graphics component of the same driver branch:

```bash
major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader |
        head -1 | cut -d. -f1)
sudo apt-get install -y "libnvidia-gl-${major}-server" ||
    sudo apt-get install -y "libnvidia-gl-${major}"
```

## GPUs newer than the installed wheels

### A torch wheel runs only on the architectures it was built for

A torch build carries compiled kernels for a fixed list of GPU
architectures. On a newer GPU, every torch CUDA operation fails with

```text
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

Code that compiles its own kernels, such as a JIT compiler like DrJit, can
still run on the same node, so a mixed workload fails only in its torch
parts.

**How to apply.** Compare the wheel's list with the device:

```bash
python -c "import torch; print(torch.version.cuda, torch.cuda.get_arch_list(),
           torch.cuda.get_device_capability(0))"
```

**Evidence.** torch 2.5.1+cu121 lists `sm_50` through `sm_90`. On a
Blackwell GPU (compute capability 12.0), `torch.stack` failed while jobs
that used only DrJit ran. CUDA 12.8 builds of torch, first published for the
2.7 series as the cu128 wheels, carry `sm_120`.

### Check which library's kernels fail before rebuilding an extension

A compiled extension can run on a newer GPU from PTX even when torch cannot.
tiny-cuda-nn's build emits PTX (`-gencode=arch=compute_X,code=compute_X`)
as well as machine code for each listed architecture, and the driver
compiles that PTX for a newer GPU at load time.

**Why.** On a Blackwell node, the "no kernel image" failure came from torch
kernels that the extension called while constructing its network. Replacing
torch with a cu128 wheel was the fix. The extension needed a rebuild only
because it links against torch's ABI, and the system `nvcc` 12.0 sufficed
for that. The pip package `nvidia-cuda-nvcc-cu12` ships `ptxas` but not the
`nvcc` driver, so it cannot stand in for a newer toolkit.

**Evidence.** A tiny-cuda-nn build for architectures `86;89` ran on `sm_120`
under a 580-series driver once torch was replaced.

### Detect the GPU the job was given, not the node's first GPU

On a node with mixed GPU generations, read the assigned device's compute
capability through the CUDA runtime, and use the device name as a second
check.

**Why.** Where the cluster does not confine devices with cgroups,
`nvidia-smi` inside a job lists every GPU on the node, so
`nvidia-smi ... | head -1` can report a GPU of another generation. On one
driver release, the `compute_cap` query also returned a value that an
integer comparison rejected, and the guard was skipped without a message.
torch honors `CUDA_VISIBLE_DEVICES`, and reading a device property launches
no kernel, so the check is safe on any GPU.

**How to apply.**

```bash
read -r cc name <<< "$("$PY" -c 'import torch
c = torch.cuda.get_device_capability(0)
print(c[0] * 10 + c[1], torch.cuda.get_device_name(0).replace(" ", "_"))' \
    2>/dev/null)"
incompatible=0
{ [ -n "${cc:-}" ] && [ "$cc" -ge 100 ] 2>/dev/null; } && incompatible=1
[[ "${name,,}" == *blackwell* ]] && incompatible=1
```

**Evidence.** Ten jobs on one mixed node passed an `nvidia-smi` check and
later died with "no kernel image is available". Twelve more on another
driver passed because of the `compute_cap` comparison.

### Requeue off an incompatible GPU instead of excluding a mixed node

Exclude nodes whose GPUs are all incompatible. On a mixed node, let the job
requeue itself when it lands on an incompatible GPU, with a bound on the
number of requeues.

**Why.** Excluding a mixed node gives up its compatible GPUs. A requeue
returns the job to the queue, and the scheduler usually places it on another
GPU. The bound stops a run of bad placements from looping forever; past it,
the job continues and fails visibly rather than disappearing.

**How to apply.** The batch header needs `#SBATCH --requeue`. Then:

```bash
if [ "$incompatible" = 1 ]; then
    n=${SLURM_RESTART_COUNT:-0}
    if [ -n "${SLURM_JOB_ID:-}" ] && [ "$n" -lt 8 ]; then
        echo "incompatible GPU; requeue $((n + 1))" >&2
        scontrol requeue "$SLURM_JOB_ID" && exit 0
    fi
    echo "WARNING: incompatible GPU and requeues exhausted" >&2
fi
```

A second environment archive built for the new architecture, selected at
run time from the detected capability, removes the need to requeue. Give it
its own stage directory (see [environment staging](environment-staging.md)).

### Retry placement failures on node defects instead of excluding the node

When a node-specific defect makes a job fail within seconds with a known
message, such as a missing system interpreter, resubmit until the job lands
elsewhere rather than adding the node to `--exclude`. Exclusion suits a
permanent hardware limit; a defect should stay visible until it is fixed.

**Why.** An exclusion list outlives its reason and hides the defect. A retry
loop leaves every failed placement in the logs, where the failure rate stays
auditable. Distinct exit statuses and messages for each placement guard let
the loop tell a placement failure from a failed run.

**How to apply.**

```bash
for try in $(seq 1 "$MAX_TRIES"); do
    jid=$(sbatch --parsable --job-name="$NAME" launcher.sh "$@")
    while [ -n "$(squeue -h -j "$jid" 2>/dev/null)" ]; do
        sleep 20
        grep -qs "$GUARD_MESSAGE" "logs/${NAME}_${jid}.err" && break
    done
    grep -qs "$GUARD_MESSAGE" "logs/${NAME}_${jid}.err" || exit 0
    echo "job $jid hit the placement guard; retrying"
done
exit 1
```

### Refuse to train when the framework falls back to the CPU

Check the training device when training starts, and raise unless a CPU run
was requested explicitly.

**Why.** On a node with a broken GPU driver (NVML errors, a GPU fallen off
the bus), `torch.cuda.is_available()` returns false. A trainer that takes
its device from `accelerate`, or from `"cuda" if available else "cpu"`,
then trains on the CPU for the whole allocation without an error.

**How to apply.**

```python
device = accelerator.device
if os.environ.get("ALLOW_CPU") != "1" and (
        device.type != "cuda" or not torch.cuda.is_available()):
    raise RuntimeError(
        f"Refusing to train on device={device!r} "
        f"(torch.cuda.is_available()={torch.cuda.is_available()}); "
        "set ALLOW_CPU=1 for an intentional CPU run")
```

Set the override for smoke tests and import checks on hosts without GPUs.
Resubmit with the bad node in `--exclude`. In a pre-submitted `afterok`
chain, give the exclusion to every link: a link that fails on that node
stops the chain (see [long runs](long-runs-and-resumption.md)).

**Evidence.** A job placed on a node with a failed GPU ran at about 30 s
per iteration on the CPU until its allocation ended.
