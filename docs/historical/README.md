# Historical engineering notes

Durable lessons from earlier projects, grouped by the tool or task they
concern. Each note records something that cost real time to learn: a failure
signature, its mechanism, how to avoid it, and the measurement that
established it. They are written without reference to the project that
produced them, so they transfer to new work.

These notes are reference material, not binding guidance. A rule this project
adopts moves into the owning guide (`docs/implementation/`, `docs/style/`, or
the agent files) and is enforced there; the note stays as its rationale.

| Folder | Subject |
|---|---|
| [mitsuba/](mitsuba/README.md) | Mitsuba 3 and DrJit: randomness, JIT specialization, precision, memory, interop, testing |
| [real-time-rendering/](real-time-rendering/README.md) | Interactive native CUDA and WebGPU demos of a neural cache trained online |
| [clusters/](clusters/README.md) | SLURM jobs, environment staging, job caches, long runs, login nodes, cleanup |

## Adding a note

- Put it in the folder of the tool or task it concerns; start a new folder
  when no existing one fits, and list it above.
- One file per subject. Each entry states the insight first, then the
  mechanism, how to apply it, and the evidence.
- Keep it durable and anonymous: no project, dataset, or people names, no
  hosts, job IDs, paths, or dates. The operational history stays in the
  project's own trackers.
