# Testing and profiling

A Mitsuba and DrJit project needs two kinds of measurement: tests and parity
gates that show the numbers are right, and profiles that show where time
goes. This file covers running tests on the GPU variant, the gates a
performance change must pass, and how to read DrJit, PyTorch and host
profiles without being misled by their own overhead.

## Test harness

### Select the CUDA variant only when none is set

A test module selects a Mitsuba variant only if none has been selected, and
selects the CUDA one:

```python
import mitsuba as mi
if mi.variant() is None:
    mi.set_variant("cuda_ad_rgb")
```

**Why.** An unconditional `mi.set_variant("llvm_ad_rgb")` at import time
aborts collection of the whole pytest run on any machine without
`libLLVM.so`, with "the LLVM backend is inactive because the LLVM shared
library could not be found", and an unconditional selection in one module
overrides the variant every other module expects. Mitsuba types such as
`mi.Float` belong to the active variant, so an application sets its variant
at the top of its entry point, before importing modules that use them. When
the LLVM backend is needed, `DRJIT_LIBLLVM_PATH` points DrJit at the
library.

### Results from the LLVM variant do not stand for the CUDA path

Run numerical tests, controls and per-lane diagnostics on the CUDA variant.
The CPU backend differs in arithmetic (`FastMath` lowering, fused
multiply-add contraction), in OptiX-specific code paths, and in which errors
appear.

**Evidence.** An integration test passed on the LLVM variant and failed on
the CUDA variant on the same assertion. A set of double-precision parity
failures caused by `FastMath` stayed hidden while their module selected the
LLVM variant and aborted collection on a machine without it.

### Bind an explicit free GPU and private kernel caches for test runs

Run the GPU suite with `CUDA_VISIBLE_DEVICES` set to a GPU no other job is
using, and with the per-process OptiX and CUDA caches of
[the kernel-cache notes](jit-and-kernel-caches.md). On the shared default
OptiX cache, a concurrent writer raises
`OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR`, which aborts the interpreter
mid-suite instead of failing one test.

### Skip, do not exit, at module level

A test module that raises `SystemExit` at import aborts the collection of its
whole directory. Use `pytest.skip("reason", allow_module_level=True)` for a
module that cannot run in the current environment.

## Acceptance gates for performance changes

### A speed change must leave the numbers unchanged

A performance change promises the same fits, the same gradient estimates and
the same optimization trajectories, not only the same estimator in
expectation. Sampling budgets, update counts, moving-average retention,
learning rates and schedules are part of the numerical method: a change that
alters any of them is a method change, however it is described. A speedup
with a changed gradient is a different estimator.

**How to apply.** Accept a change only with evidence proportional to what it
touches:

1. per-lane parity on production batches, run in process (bitwise where the
   change promises it; see [precision](precision.md));
2. matched-seed comparison of losses, gradients and weights against the
   spread of an unchanged repeat;
3. a trajectory gate from a fixed checkpoint over enough steps to see drift;
4. a stable kernel-hash set with zero fresh compilation after warm-up;
5. flat memory over a long run (see [memory](memory.md));
6. matched-hardware wall-clock time, stating which warm-up steps are
   excluded and whether compilation and file output are included.

### Run an unchanged repeat beside every candidate

Compare a candidate with a repeat of the unchanged code, run the same way,
not with exact equality. Without the repeat, rounding differences look like
regressions and sampling nondeterminism looks like the candidate's fault.

**Evidence.** A symbolic-loop candidate was first held back because its
gradients differed from the reference at step zero. An unchanged repeat later
showed the same kind of difference, traced to a nondeterministic sampling
table (see
[randomness and independence](randomness-and-independence.md)).
After that table was fixed, the candidate differed from the reference by
1.9e-6 in a normalized gradient, against 9.5e-7 for the unchanged repeat,
with every sample count identical, and was accepted.

### Gate long runs as well as short ones

Kernel reuse, memory and timing can fail late in a run. Run a long gate (on
the order of a thousand steps) and compare the median step time of its
second half with its first half; a ratio near one shows no late slowdown.

**Evidence.** Accepted gates held second-half to first-half median ratios of
0.98 to 1.01 over 256 and 1,024 steps.

## Profiling

### Audit timings are not throughput

The kernel history times every launch with CUDA events, and synchronized
phase timers wait for the device at each boundary. Both serialize work that
normally overlaps. Keep these instruments off by default behind an
environment switch, and report audited timings separately from ordinary
throughput.

**Evidence.** The same workload ran at a median of 1.81 seconds per step
with the kernel-history audit and phase timers on, and 1.61 seconds with
both off: the instrumentation cost about 12.6%.

### DrJit's kernel history does not see PyTorch or tiny-cuda-nn

Kernel-history times cover DrJit kernels, including OptiX launches, and
nothing that runs inside a `dr.wrap` call. Profile the network side with
`torch.profiler` (CUDA activity enabled), and do not subtract kernel-history
time from wall time to estimate host overhead.

**Evidence.** A 7.95-second step launched 10,803 DrJit kernels totaling 0.87
seconds of GPU time. The PyTorch profile added about 0.5 seconds per step of
network kernels (hash-grid backward and GEMMs). The GPU was busy for under
1.5 seconds of the step; the rest was host work around many small launches.
The PyTorch profiler wrote its trace and then crashed at interpreter exit in
that process, so export the trace explicitly (`export_chrome_trace`) before
exit.

### Tell host-bound from GPU-bound

Record process CPU time (`time.process_time()`, all threads) beside wall
time for each step. CPU time close to wall time indicates a host-bound step.
Launch counts per step and a one-hertz sample of GPU utilization complete the
picture.

### A host read absorbs the queued GPU time

A Python profiler attributes the time spent waiting for queued GPU work to
the first host read after the work was queued, not to the code that queued
it. Read a large cumulative time at a `.numpy()` call or a scalar
conversion as "the GPU work before this point", and look at the kernel
history for its cause.

**Evidence.** In a CPU profile of a gradient phase, 24 of 38 seconds sat in
two host reads: one reading a record at the start of a term and one reading
a mean at its end.

### Profiled Python cost is not a savings estimate

A profiler inflates the cost of many small Python calls. Remove such a cost
only after measuring the unprofiled gain.

**Evidence.** DrJit's traversal of dataclass inputs called
`typing.get_type_hints` 2,816 times over 16 steps, for 21.5 profiled
seconds. Pre-resolving the annotations changed the unprofiled step time from
4.44 to 4.48 seconds, no gain. Replacing the dataclasses with
`DRJIT_STRUCT` classes was not a drop-in change, because their
default-construction contract differs.

### Measure compilation before blaming it

A slow phase with suspected recompilation needs its fresh-compile count and
compile time per step measured with the kernel history (see
[JIT specialization](jit-and-kernel-caches.md)). Compilation can be real and
still a minor share of the time.

### Diagnostic materialization changes what it measures

Evaluating intermediate records at scope boundaries attributes lazy work to
the code that produced it, but it also changes kernel fusion, can evaluate
fields that would otherwise never be computed, and changes rounding. Keep
such breakdowns separate from throughput numbers and from numerical parity.

**Evidence.** Materializing records at scope boundaries changed a loop's
stable kernel count from 311 to 287.

### Bind host thread pools to the allocation

PyTorch, OpenBLAS and DrJit's thread pool (which Mitsuba and the LLVM backend
use) size themselves from the node's core count, not from the CPUs a
scheduler grants the process. Bind them at startup:

```python
count = int(os.environ.get("OMP_NUM_THREADS")
            or len(os.sched_getaffinity(0)))
torch.set_num_threads(count)        # OpenBLAS follows through OpenMP
dr.set_thread_count(count)
```

Have launchers export `OMP_NUM_THREADS` from the allocation. Write host
arithmetic on small matrices (a 4 by 4 projection of a point table)
elementwise rather than as a BLAS call, so it does not depend on the pool.

**Evidence.** On a 96-core node under an 8-CPU allocation, 96 spinning
threads made a per-step host table computation take 2 seconds instead of 80
milliseconds, two thirds of the step. On a 384-core node an unbound DrJit
process held about 390 threads against 22 to 29 when bound; a per-user
thread limit of 1,024 was then reached by a few processes, and further forks
failed with "Resource temporarily unavailable", killing unrelated tools.

### When wall time stalls outside GPU work, look at file writes

A step that takes far longer than its measured compute phases, with no fresh
compilation, is often blocked in a file write, especially on a network
filesystem.

**How to apply.** Keep synchronous writes to network storage off the
training loop: append records to a local file and let a background thread
publish ordered chunks to the remote file, with explicit drains at
checkpoints, at exit and on error, and with remote write errors propagated.
Remember that profilers write too: a phase timer that flushed a JSON file
after every phase added nine remote writes per step, outside the phases it
timed.

**Evidence.** One step took 12.1 seconds, of which 10.5 seconds were a single
153-byte `write()` to a metrics CSV on a network filesystem.

### Observe a stalled process from its parent

Linux lets a parent read its child's `/proc/<pid>/task/<tid>/syscall` and
`wchan` without ptrace; a child reading its parent's `syscall` gets `EPERM`.
A parent supervisor that samples these every 200 milliseconds, resolves file
descriptors from the syscall arguments through `/proc/<pid>/fd`, and writes
its observations to local disk localizes stalls without instrumenting the
program. A `request_wait_answer` wait channel is a FUSE filesystem request.
`py-spy dump` reads a Python stack without stopping the process. A native
debugger cannot stop a thread blocked in such a write until the write
returns, so a stack captured then shows the call after the stalled one.
