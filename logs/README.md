# Logs

SLURM launchers write standard output and error here as `logs/%x_%j.out` and
`logs/%x_%j.err`; `scripts/launchers/run_local.sh` writes
`logs/local_<timestamp>_gpu<index>.log`. The directory is tracked so SLURM can
open those paths in a fresh checkout; generated log files remain ignored.

Logs are transient operational records, not experiment results. Durable run
artifacts belong under `outputs/` and in the storage described in `DATA.md`.
