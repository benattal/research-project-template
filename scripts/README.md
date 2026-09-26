# scripts/

Everything here is filed by **role**, not by campaign. Nothing sits loose at
the top level; if you are adding a script, put it in the directory whose
description below matches what the script *does*.

| Directory | What lives here |
|---|---|
| `launchers/` | Scheduler and environment adapters: `train_slurm.sh` for cluster jobs, `run_local.sh` for one explicit local GPU. |
| `probes/` | Investigation diagnostics: the scripts that answer "why is this number wrong". Split by subject once there are several. |
| `scoring/` | Metrics, comparisons, and audits over finished runs. |
| `regression/` | The A/B regression suite and the headline reproduction. See [experiments](../docs/implementation/experiments.md). |
| `verify/` | Correctness checks that assert rather than measure: `verify_*`, `validate_*`, environment checks, and smoke tests. |
| `figures/` | Figure, panel, and plot generators. Pure output producers; nothing imports them. |
| `pages/` | Results-page builders and `bundle.py`. See [results pages](../docs/implementation/results-pages.md). |
| `profiling/` | Timing and throughput measurement. |
| `install/` | Environment installers. |
| `infra/` | Documentation build, repository audits, and housekeeping. |
| `lib/` | Script-only helpers, such as the shell library `job_cache.sh`. Reusable Python belongs in `src/`. |
| `archive/` | Retired one-offs, grouped by the campaign that produced them. See [archive/README.md](archive/README.md). |

## Importing a shared module

Reusable behavior belongs in `src/`, not `scripts/lib/`. A directly executed
script may bootstrap the repository root with a depth-independent line, then
import the package:

```python
sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from src.core.utils.metrics import psnr
```

The same idiom gives the repository root, `__file__.rsplit("/scripts/", 1)[0]`,
and is preferred over counting `os.path.dirname` levels. Running a script as a
module from the root (`python -m scripts.probes.<name>`) needs no bootstrap.

## Canonical commands

Use these entry points for new work; do not clone a nearby probe or launcher.

| Workflow | Canonical entry point |
|---|---|
| cluster job | `sbatch scripts/launchers/train_slurm.sh <application> <preset> [overrides...]` |
| local GPU job | `bash scripts/launchers/run_local.sh <gpu> -- python -m src.apps.<application> <preset>` |
| source-tree audit | `python scripts/infra/check_source_tree.py` |
| documentation site | `bash scripts/infra/build_docs.sh` |
| page archive | `python scripts/pages/bundle.py docs/pages/<page>` |

TODO(template): add each new canonical workflow here when it is created.

## Conventions

- A script cited by a doc, a page, or the agent files has that citation
  updated in the same commit that moves it.
- Retiring a script means moving it to `archive/<campaign>-<yyyy-mm>/`, not
  deleting it: a retired probe is often the only record of how a published
  number was made.
- A new script may parse arguments and select artifacts, but model loading,
  metrics, reference selection, and image I/O belong in `src/`.
- Output cleanup is manifest-driven (see
  [experiments](../docs/implementation/experiments.md)): generate a manifest,
  review it, and delete only what it lists.
- Launchers source `lib/job_cache.sh` for per-job compilation caches that
  clean themselves up at exit.
