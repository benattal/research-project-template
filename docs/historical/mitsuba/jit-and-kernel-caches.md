# JIT specialization and kernel caches

DrJit records array operations lazily and compiles each distinct kernel once,
so a training loop is fast only when its set of kernels stops growing after
warm-up and when compiled kernels survive between processes. The first half
of this file covers what makes DrJit compile a new kernel and how to catch
it. The second half covers the three on-disk caches a Mitsuba process uses
(DrJit's, OptiX's and the CUDA driver's) and the failures they cause when
shared.

## Specialization

### Make every changing scalar opaque

A Python number that enters a DrJit expression becomes a constant in the
generated kernel, so a value that changes every iteration compiles a new
kernel every iteration.

**Why.** DrJit identifies a kernel by a hash of its intermediate
representation, and an inlined literal is part of that representation. The
symptom is wall-clock time with an idle GPU, never an error.

**How to apply.** Wrap the value before it enters any expression:

```python
radius = dr.opaque(mi.Float, radius_value)      # schedule values, counters,
count = dr.opaque(mi.UInt32, live_count - 1)    # live counts, seeds, poses
```

For a scalar stored inside a persistent object, call `dr.make_opaque` on the
member after updating it. Candidates are learning-rate factors, schedule
values, iteration counters, emitter positions and cone angles that follow a
moving pose, per-step counts, and seeds. Update a persistent object's device
value instead of rebuilding the sampler, optimizer or schedule that owns it.

**Evidence.** A live-lane count used as a literal compiled about 45 fresh
kernels per step and raised a phase from 0.56 to 0.8 seconds; made opaque,
the steady step compiled none.

### Build opaque vectors from opaque components

`mi.Vector3f(*python_floats)` inlines three literals, and in DrJit 1.3.1
`dr.opaque(mi.Vector3f, (x, y, z))` with a Python tuple raises an error.
Build the vector from opaque scalars:

```python
offset = mi.Vector3f(*(dr.opaque(mi.Float, float(v)) for v in values))
```

**Evidence.** Object offsets written as `mi.Vector3f(*floats)` into a traced
visibility test recompiled a 17,233-operation OptiX kernel every step once the
pose started to move: steps went from 6.7–7.0 seconds to 15.9–17.7 seconds.
With component-wise opaque values the moving steps took 6.8–7.1 seconds,
compiled nothing after setup, and followed the same trajectory bit for bit.

### Keep wavefront widths in a small fixed set

Treat every new launch width as a potential new kernel.

**Why.** A width reaches a kernel as a literal whenever Python code uses it
as a number: `dr.width(x)` returns a Python `int`, and expressions such as
`index // spp` or a width passed into sampler seeding carry it into the
generated code. DrJit's own documentation for frozen functions warns that
the integer `dr.width()` returns may be merged into the generated code.

**How to apply.** Choose widths from a quantized set, typically powers of
two, and never let a width grow within one loop. Record the widths each call
site launches and warn when the set grows:

```python
_seen: dict[str, set[int]] = {}

def record_width(tag: str, n: int, budget: int = 6) -> None:
    widths = _seen.setdefault(tag, set())
    widths.add(int(n))
    if len(widths) == budget + 1:
        log(f"'{tag}' launched {len(widths)} widths: {sorted(widths)}")
```

A render legitimately has one partial tail chunk and may halve its chunk
after an out-of-memory retry, so a budget of about six widths per site is
enough.

**Evidence.** A render that re-derived its chunk width from free device
memory before every chunk compiled more than 800 kernels at about 1.1 seconds
each. Quantizing the width removed those compilations, but the 806-million-
ray render took 900 to 1,300 seconds before and after, so compilation was a
minority of its time. Measure the compile share before crediting a width fix
with a speedup.

### Detect recompilation with the kernel history

DrJit's kernel history reports, for every launch, whether the kernel was
compiled or reused, and is the instrument for all of the above.

**How to apply.** Enable the flag, then drain the history once per step;
each `dr.kernel_history()` call returns the launches since the previous call
and clears them:

```python
dr.set_flag(dr.JitFlag.KernelHistory, True)
dr.kernel_history()                                 # discard setup launches
seen = set()
for step in range(steps):
    run_step()
    history = dr.kernel_history()
    fresh = [e for e in history
             if not e.get("cache_hit") and not e.get("cache_disk")
             and e.get("codegen_time", 0.0) > 0.0]
    seen.update(str(e["hash"]) for e in history if "hash" in e)
    log(step, len(history), len(fresh), len(seen))
```

The fields used are `hash`, `size` (launch width), `operation_count`,
`uses_optix`, `recording_mode` (frozen-function record or replay),
`execution_time`, `codegen_time`, `backend_time` and `ir`. Times are in
milliseconds. OptiX launches are reported as uncached even when nothing was
compiled, hence the `codegen_time` test. `codegen_time` covers only DrJit's
generation of the IR, which happens even for cached kernels; `backend_time`
is the PTX or OptiX compilation that a cache miss adds. Record both: an audit
that records only `codegen_time` understates the cost of a recompilation.

A correctly structured loop reaches zero fresh compilations per step and a
constant count of cumulative unique hashes within a few steps. One or two
extra variants on the first scene-parameter update are normal: they differed
only in argument ordering tied to the first updated scene buffer.

**Evidence.** Over 1,021 steady steps of an inverse-rendering loop, 299 to
312 cumulative hashes (depending on the code version) stayed constant with no
fresh compilation.

### Diff the IR of a kernel that recompiles

When the same kernel recompiles every step, save its `ir` from the kernel
history on two consecutive steps and diff them. The structure is identical;
the differing constants name the value that should be opaque.

**Evidence.** Two consecutive recompilations of the 17,233-operation kernel
above differed only in four floating-point constants, the object offsets, and
the kernel hash.

### A warm disk cache hides specialization

Zero fresh compilations in the first steps of a run does not show that the
kernel set is invariant: the kernels may have been found in the on-disk
cache (`cache_disk`) because an earlier run with the same values compiled
them. Judge invariance by the cumulative hash count over steps where the
changing values actually change.

## On-disk caches

### A Mitsuba process uses three caches

Three layers compile and cache code, each in its own place:

- DrJit's kernel cache, in `$HOME/.drjit` in DrJit 1.3.1. The library reads
  no environment variable for this path (the binary contains none), so an
  exported `DRJIT_CACHE_DIR` has no effect. A cache written by another DrJit
  version triggers a message suggesting that the directory be wiped.
- OptiX's disk cache, placed by the driver at `OPTIX_CACHE_PATH`, with
  `OPTIX_CACHE_MAXSIZE` bounding its size; a size of 0 disables it.
- The CUDA driver's PTX compilation cache, at `CUDA_CACHE_PATH`, by default
  `~/.nv/ComputeCache`, with a size cap.

All three hold build products. DrJit kernels are content-addressed and OptiX
recompiles in well under a minute, so deleting them loses nothing but warm-up
time.

### Give every concurrent process its own OptiX and CUDA caches

Two processes writing one OptiX cache corrupt it.

**Why.** The OptiX disk cache is a database file that does not tolerate
concurrent writers. The failures are `OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR`
at startup, which aborts the interpreter (a test suite dies mid-run rather
than failing a test), and `OPTIX_ERROR_PIPELINE_LINK_ERROR`.

**How to apply.** Point each job at private directories on node-local disk
and remove them on exit:

```bash
root="${JOB_LOCAL_TMP:-/tmp/$USER}"      # guard: never expand to "/"
tag="${JOB_ID:-$$}"
export OPTIX_CACHE_PATH="$root/optix_cache_$tag"
export CUDA_CACHE_PATH="$root/cuda_cache_$tag"
mkdir -p "$OPTIX_CACHE_PATH" "$CUDA_CACHE_PATH"
trap 'rm -rf "$OPTIX_CACHE_PATH" "$CUDA_CACHE_PATH"' EXIT TERM INT
```

Guard the base directory with a default: an unguarded `${VAR}` that is unset
expands to an empty string, and the cache lands at `/optix_cache_<tag>`,
which the job cannot create. Append to an existing `EXIT` trap instead of
replacing it when the launcher arms its own. Runs that execute one after
another on the same GPU, such as a parameter sweep, can share one cache on
purpose so the sweep compiles once.

**Evidence.** Without cleanup, a shared `/tmp` accumulated 143 orphaned
cache directories (646 MB). Local launches without private caches died at
startup beside a running job on the same node.

### Keep the CUDA driver cache on local disk

The CUDA driver's default cache under the home directory sits on a network
filesystem on many clusters, and each driver recompilation against it is an
order of magnitude slower than against local disk.

**Evidence.** A driver recompilation took 0.9–2.0 seconds against the
network home cache and about 0.11 seconds locally. Moving the cache to local
disk alone cut an iteration from 1.23 seconds to 0.18 seconds. Two
concurrent jobs sharing the default cache also evict each other's entries
near its size cap.

On cloud GPU hosts whose persistent storage is a network mount, keep the
repository, outputs and logs on the mount and keep the virtual environment
and the kernel caches on local disk: both are reproducible and import far
faster there.

### The OptiX backend needs the driver's OptiX library

DrJit's CUDA variants load `libnvoptix.so.1`, which ships with the NVIDIA
driver's graphics components, not with the CUDA toolkit. On an image that
installs the toolkit and a compute-only driver, `cuda_ad_rgb` fails until
the driver's GL package of the matching version is installed (on Ubuntu,
`libnvidia-gl-<version>-server`). `DRJIT_LIBOPTIX_PATH` points DrJit at a
library in a non-standard location.

### OptiX pipeline link errors have more than one trigger

`OPTIX_ERROR_PIPELINE_LINK_ERROR` has been seen from two causes: concurrent
writers to one OptiX cache, and a single very large launch. A launch of about
67 million lanes failed at startup with this error on two GPUs in three
attempts, while the same program at about 16 million lanes ran. The cause of
the size dependence was not isolated. Private caches rule out the first
cause; if the error persists, split the launch into chunks.

### A process can wait indefinitely inside `optixPipelineCreate`

A training process was once found alive for hours with its GPU at 100%
utilization and no progress. Its main-thread native stack ran through
`jitc_eval` and `jitc_optix_compile` into the driver, and disassembly of the
installed DrJit binary placed the return address immediately after the call
through `optixPipelineCreate`. No Xid error was logged. The cause was not
found. When a process stops progressing, take a Python stack with
`py-spy dump` (read-only, does not stop the process) and a native stack before
restarting it, so the wait site is on record.

### Expect compilation in the first iterations

The first iterations of a run compile its kernels and are slower; this is
not a regression. In a small workload startup compilation took 2.35 seconds
and the first step of a new phase 1.56 seconds. Exclude the first few steps
of each phase from timing statistics, and report them separately.
