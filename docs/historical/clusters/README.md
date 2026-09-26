# Clusters and shared hosts

Lessons from running GPU research jobs on SLURM clusters and on shared login
nodes: how the Python environment reaches a compute node, which per-job
state must stay private, how long runs survive time limits, and how outputs
get back and are cleaned up safely. Each file opens with the problem it
covers; each entry states the insight, the mechanism, how to apply it, and
the evidence where there is some.

| File | Subject |
|---|---|
| [environment-staging.md](environment-staging.md) | Shipping a virtualenv to node-local disk: one archive, a stage path keyed by archive identity, locking, import checks, a portable interpreter |
| [toolchains-and-gpu-architectures.md](toolchains-and-gpu-architectures.md) | Environment modules with virtualenvs, wheel and extension builds, GPUs newer than the installed torch wheels, and refusing a silent CPU fallback |
| [job-caches.md](job-caches.md) | Per-job OptiX, JIT, CUDA driver and bytecode caches on local disk, removed at exit; shared model-download caches on project storage |
| [code-deployment.md](code-deployment.md) | Which source a job imports: deployment checkouts, filesystem caching, hash gates, working trees edited mid-run, `SLURM_SUBMIT_DIR`, editable installs |
| [long-runs-and-resumption.md](long-runs-and-resumption.md) | Chaining runs past CPU-time and wall-clock limits through exact checkpoints and a validating driver, or pre-submitted `afterok` chains |
| [login-nodes-and-shared-hosts.md](login-nodes-and-shared-hosts.md) | Submission hosts, GPU-less login nodes, site GPU-request rules, thread ceilings, thread pools, GPU checks, process cleanup, and device selection |
| [output-retention-and-cleanup.md](output-retention-and-cleanup.md) | Log directories, copying outputs off the node, full quotas, pruning, manifests, and verified remote copies |
| [shell-safety.md](shell-safety.md) | `set -e` exits without output, trap composition, `pkill -f`, `bfs` time filters, and guarded deletes |
