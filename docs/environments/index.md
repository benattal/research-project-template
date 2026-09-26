# Environments and running the code

This is the authoritative environment and setup guide. Do not copy interpreter
paths, module sets, or node exclusions from old logs; use the maintained
installers and launchers named here.

TODO(template): fill in each section for the machines this project runs on,
and delete the sections that do not apply.

The standing compute authorization in the repository's `AGENTS.md` and
`CLAUDE.md` states how many local GPUs and scheduler jobs may
be used without further approval. Allocation and device-activity checks below
still apply.

## Before any GPU process

Importing an application entry point can allocate GPU state. Check allocation
and live activity first:

```bash
squeue -u "$USER"
squeue -w "$(hostname -s)"
nvidia-smi
```

On a host without SLURM, `squeue` does not exist and `nvidia-smi` is the only
allocation check. Training must not share a device allocated to another job
or used by another user's active workload. Small bounded diagnostics may use
an occupied device only under standing user permission. Do not replace a GPU
run with large CPU work.

Bind every job to one explicit device:

```bash
bash scripts/launchers/run_local.sh <gpu-index> -- python -m src.apps.example example-default
```

Static source reading, link checks, compilation, and non-GPU architecture and
unit tests do not require an application import.

### Process and thread budget

TODO(template): record the cluster's per-user thread ceiling, if any. Keep
normal operation well under it so a compiler, editor, or monitor can still
create workers. Check the count before each local launch:

```bash
ps -L -u "$USER" --no-headers | wc -l
```

Kill processes by PID, never by a `pkill -f` pattern.

## Repository checkout

Run commands from the repository root. Dataset paths, output paths, and
launchers are repository-relative. A cluster deployment is a separate checkout;
editing a local checkout does not change it.

## Local development environment

```bash
bash scripts/install/install_venv.sh
source venv/bin/activate
python -m pytest -q tests/architecture tests/unit
```

TODO(template): list system modules or CUDA versions the environment needs, and
how to load them in a login shell.

## Application commands

```bash
python -m src.apps.example example-default
python -m src.apps.example example-smoke --training.iterations 50
```

## SLURM clusters

```bash
mkdir -p logs
sbatch scripts/launchers/train_slurm.sh example example-default
```

TODO(template): partitions, GPU types, time limits, node exclusions and why,
how the environment reaches compute nodes, and how deployments are
synchronized and verified.

Login-node processes may have a CPU-time limit. A long local job runs as
consecutive links that each resume from the previous link's exact checkpoint.

## Reproducing a run

The preset name alone is not a complete run specification. Preserve:

- application entry and preset;
- command-line overrides;
- the saved `config.json` and `provenance.json`;
- dataset and target paths;
- checkpoint manifests;
- the base seed and named random-domain policy;
- environment and toolchain;
- hardware, for performance comparisons.

Diff a new `config.json` against the intended source run before interpreting a
numerical change. Use the lineage in
[production configurations](../production-configs.md), not timestamps or
convenient checkpoint filenames.

## Generated and ignored state

- `outputs/`: run artifacts and checkpoints (durable copy: `DATA.md`);
- `logs/`: scheduler and application logs (`logs/README.md` is the tracked
  placeholder);
- `data/resources/`: downloaded model weights and other resources;
- large assets under `data/`;
- `docs/pages/`: generated documentation and diagnostic websites;
- compilation caches and local virtual environments.

Do not force-add generated state. Downloaded assets may be absent in a clean
checkout without indicating missing source.
