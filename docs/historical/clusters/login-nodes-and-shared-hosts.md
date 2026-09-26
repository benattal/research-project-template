# Login nodes and shared hosts

Login nodes, head nodes and multi-GPU hosts are shared with other users and
carry limits that compute allocations do not: which host may submit which
jobs, per-user thread ceilings, CPU-time limits, and no scheduler between one
user's process and another's. Compute nodes are shared too, between the jobs
placed on them. These entries cover submitting from the right host, staying
under thread ceilings, checking a GPU before using it, and cleaning up.

## GPU jobs may be accepted only from the GPU login node

On some clusters only one login node accepts GPU job submissions. From the
others, `sbatch` fails with

```text
GPU resources requested from a CPU login node; please submit from <gpu-login-node>
```

Check `hostname` before concluding that the cluster is down: `sinfo` can
show idle GPU nodes while the submitting host is the wrong one. When a
session lands on another login node, run the launcher on the GPU login node
over `ssh`:

```bash
ssh "$GPU_LOGIN" 'cd <repo> &&
    nohup setsid bash -l launcher.sh > logs/launch.log 2>&1 &'
```

Chain drivers that submit continuations must run there as well. When the GPU
login node goes down, it takes its local jobs and drivers with it, and GPU
submission stays blocked until it returns.

## Per-user thread ceilings make `fork` fail for everything you run

Some clusters cap the number of threads one user may hold on a login node
(1,024 on the systems these notes come from). At the cap, every new process
or thread fails, including the user's shell, editors, schedulers and
monitors.

**How to apply.** Count before every local launch, and keep the total near
70% of the cap (about 700 of 1,024) so compilers, editors and monitors can
still start:

```bash
ps -u "$USER" -o nlwp= | awk '{s += $1} END {print s}'
ps -u "$USER" -o pid,nlwp,etime,cmd --sort=-nlwp | head    # heaviest first
```

**Evidence.** At the cap, forks fail with `Resource temporarily unavailable`
(`EAGAIN`, raised as `BlockingIOError` in Python), and a library that sizes
its thread pool at startup raises the same error. Chain schedulers, a page
watcher and the interactive tool shell died this way.

## Thread pools size to the node, not to the allocation

Torch, OpenBLAS and DrJit size their host thread pools from the machine's
core count, not from the CPUs that a scheduler or affinity mask grants the
process. Bind them to the allocation at program start.

**Why.** On a large login node, one uncapped process holds about one thread
per core in each pool. Inside an allocation, the same pools oversubscribe
the granted CPUs, and their threads spin-wait against each other.

**How to apply.** Export the allocation in every launcher and bind once at
the application's entry point:

```bash
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
```

```python
import os
import drjit as dr
import torch

count = max(1, int(os.environ.get("OMP_NUM_THREADS")
                   or len(os.sched_getaffinity(0))))
torch.set_num_threads(count)   # reaches OpenBLAS through the OpenMP runtime
dr.set_thread_count(count)     # DrJit's pool, also used by Mitsuba
```

Code that does not bind its pools ignores `OMP_NUM_THREADS` for DrJit's
pool; run one such process at a time, and count threads before starting it.

**Evidence.** On a 384-core login node, one uncapped DrJit process held
about 390 threads; with `OMP_NUM_THREADS=8` it held 22 to 29. On a 96-core
compute node under an 8-CPU allocation, 96 spinning threads made one
host-side BLAS call take 2 s instead of 80 ms.

## Check allocation and live activity before a local GPU launch

Before starting a GPU process outside the scheduler, check that the
scheduler has not allocated the node and that no one is computing on the
device, then bind the process to that one device.

```bash
squeue -u "$USER"
[ -z "$(squeue -h -w "$(hostname -s)" -o %i)" ] ||
    { echo "node has scheduler allocations" >&2; exit 1; }
pids=$(nvidia-smi --id="$GPU" --query-compute-apps=pid \
       --format=csv,noheader,nounits)
util=$(nvidia-smi --id="$GPU" --query-gpu=utilization.gpu \
       --format=csv,noheader,nounits)
if [ -n "$pids" ] && ! [ "$util" -eq 0 ] 2>/dev/null; then
    echo "GPU $GPU busy or unknown ($pids, ${util}%)" >&2; exit 1
fi
export CUDA_VISIBLE_DEVICES=$GPU
```

A process that holds memory at 0% utilization is an idle reservation; an
unreadable utilization fails closed. Check again during long runs and before
each link of a chain, since the device can be claimed between processes. On
a host without a scheduler, `squeue` does not exist and `nvidia-smi` is the
only check.

## A head node that is also a compute node is not a free GPU

On clusters where the scheduler also places jobs on the head or login node,
an unallocated GPU process there competes with allocated jobs for GPU and
host memory. Route diagnostics through the scheduler on such hosts.

**Evidence.** A GPU probe started by hand on such a head node caused the
out-of-memory kill of another user's job.

## Clean up what you start

Record the PID of every background process and kill it by PID when its work
is done, including the sleeping children of polling loops.

**Why.** Unattended loops that start uncapped processes, duplicate loops,
and forgotten children are how a user reaches the thread ceiling. A
background waiter loop outlives a crashed session and keeps launching work.

**How to apply.** After any session crash, list your processes and kill the
leftovers by PID before relaunching anything ([shell safety](shell-safety.md)
explains why not by pattern):

```bash
ps -u "$USER" -o pid,ppid,etime,nlwp,cmd --sort=start_time
kill <pid> <pid>
```

## Inside a GPU allocation, select devices by identity

Tools that do not follow `CUDA_VISIBLE_DEVICES` need the allocated device
selected by UUID.

**Why.** The device cgroup can renumber the allocated GPU to ordinal 0, and
Vulkan and WebGPU loaders choose devices independently of CUDA's mask. In a
browser, the graphics process and the WebGPU compute path can open different
GPUs on a multi-GPU host.

**How to apply.** Require exactly one visible device, read its UUID and PCI
bus ID with `nvidia-smi --query-gpu=uuid,pci.bus_id --format=csv,noheader`,
select the Vulkan device from them, and confirm with `vulkaninfo --summary`
that the selected device's UUID matches.

## Jobs on one node share its network namespace and may lack a writable home

Derive a server port from the job ID, and never attach to a process already
listening on a default port: it may be another job's server. Point the XDG
directories at the job's temporary directory, since compute-node home
directories need not be writable and tools such as Chromium write their
configuration and crash databases there.

```bash
export PORT=$((20000 + SLURM_JOB_ID % 40000))
export XDG_CONFIG_HOME=$SLURM_TMPDIR/xdg/config
export XDG_CACHE_HOME=$SLURM_TMPDIR/xdg/cache
mkdir -p "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"
```

## Know whether a login node has GPUs before running anything there

On some clusters the login nodes have no GPU. On others a login node
carries several GPUs that every user shares, under per-process CPU-time and
per-user process limits. Check with `nvidia-smi` before running anything
that touches CUDA, and send training through the scheduler in both cases.
On a GPU login node, keep to smoke tests and sweeps of short runs, with one
BLAS thread per process and each process bound to one device.

**Why.** Without a GPU, a trainer fails at its first CUDA call, hangs at
import, or falls back to the CPU and runs until someone notices (see
[toolchains](toolchains-and-gpu-architectures.md#refuse-to-train-when-the-framework-falls-back-to-the-cpu)).
With GPUs, the CPU-time limit counts every thread of the process (see
[long runs](long-runs-and-resumption.md)), so thread count decides how long
a process lives.

**How to apply.** Package installs, downloads, import-only checks, queue
commands and editing are fine on any login node. For an intentional CPU run,
such as an import check, use the trainer's explicit CPU override rather
than removing its device check.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU
```

**Evidence.** On a login node with four H100s and 192 hardware threads, a
PyTorch process without thread caps started one thread per core, used its
3,600 s CPU-time allowance in minutes of wall time, and died with `SIGXCPU`
(exit status 152). Three processes per GPU on all four GPUs, plus their
data-loader workers, reached the per-user process limit (`ulimit -u` of
1,024), and runs died at startup with `RuntimeError: can't start new thread`.

## GPU request syntax and allowed GPU counts are site rules

Clusters differ in how a job requests GPUs and memory, and headers copied
from another cluster can be rejected. One cluster required
`--gpus-per-node=<type>:N` instead of `--gres=gpu:<type>:N`, required
`--account`, rejected any `--mem` request, and accepted only one GPU or a
multiple of four per job (its nodes held four GPUs each). Read the site's
job-submission page before writing headers, and validate the request in the
submitting script so that a bad count fails before it reaches the queue:

```bash
if ! [[ "$NGPU" =~ ^[0-9]+$ ]] || [ "$NGPU" -lt 1 ] ||
   { [ "$NGPU" -ne 1 ] && [ $((NGPU % 4)) -ne 0 ]; }; then
    echo "NGPU=$NGPU: this site accepts 1 GPU or a multiple of 4" >&2
    exit 2
fi
sbatch --account="$ACCOUNT" --gpus-per-node="h100:$NGPU" \
       --cpus-per-task=$((NGPU * 8)) job.sh
```

`torchrun --standalone` covers one node; a job larger than one node needs a
multi-node rendezvous.

**Evidence.** Two-GPU headers written before these rules applied remained
in several launch scripts and had to be overridden at every submission.
