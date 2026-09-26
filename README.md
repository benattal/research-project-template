# TODO(template): Project name

TODO(template): one paragraph stating the research question, the method, and
what the repository demonstrates. Link the paper and project page when they
exist.

## Using this template

This repository is a starting point for a research code base: a layout, a set
of written conventions for code, prose, results and experiments, agent
guidance for coding assistants, and a small runnable example that shows the
conventions in code. Delete this section once the project is set up.

1. Create a repository from this template (on GitHub, **Use this template**),
   or clone it and replace the remote.
2. Find every customization point:

   ```bash
   grep -rn "TODO(template)" --exclude-dir=.git .
   ```

3. Replace the example application `src/apps/example/` with the project's
   first application. Update `CORE_CONCERNS` and `APPLICATIONS` in
   `scripts/infra/check_source_tree.py` and the matching docs
   (`docs/implementation/architecture.md`, `docs/implementation/apps/`,
   `docs/concepts/`).
4. Fill in the project model, the contribution paragraph, and the standing
   compute limits in `CLAUDE.md` and `AGENTS.md` (edit both together; a test
   checks that their shared sections match), the audience in
   `docs/style/writing.md`, the environment guide, and the storage
   placeholders in `DATA.md`.
5. Run the checks below.

### What is in it

| Piece | Where | Purpose |
|---|---|---|
| Agent guidance | `CLAUDE.md`, `AGENTS.md` | How coding agents work here: research workflow, architecture rules, numerical contracts, verification, shell safety |
| Style guides | `docs/style/` | Code organization, writing and math, results presentation |
| Documentation tree | `docs/` | Concepts, implementation, environments, papers, plans, trackers, archive, historical notes, production registry |
| Historical notes | `docs/historical/` | Durable, project-independent lessons on Mitsuba/DrJit, real-time rendering, and clusters |
| Source layout | `src/core/`, `src/apps/` | Shared contracts and one vertical slice per application |
| Example application | `src/apps/example/` | Frozen configs, a preset registry, explicit schedules, named random domains, run provenance |
| Script taxonomy | `scripts/` | Launchers, probes, scoring, verification, pages, infrastructure, each by role |
| Tests | `tests/{architecture,unit,integration}/` | Static structure gates, focused contracts, end-to-end runs |
| Data policy | `DATA.md`, `.claude/skills/upload-data/` | Outputs in object storage, retention, verify-before-cleanup |

## Repository layout

```text
src/            core/ shared contracts, apps/<application>/ vertical slices
data/           tracked small inputs; large assets ignored (DATA.md)
scripts/        role-based tools (scripts/README.md)
tests/          architecture/, unit/, integration/
docs/           the documentation site source (docs/index.md)
logs/           ignored logs; README keeps the directory
outputs/        ignored run artifacts
third_party/    baselines and vendored code
```

## Installation

```bash
bash scripts/install/install_venv.sh
source venv/bin/activate
```

See [environments](docs/environments/index.md) for cluster setup.

## Running

```bash
python -m src.apps.example example-default
python -m src.apps.example example-smoke --training.iterations 50
python -m src.apps.example --help            # lists the presets
sbatch scripts/launchers/train_slurm.sh example example-default
```

Each run writes `outputs/<preset>_<YYYYmmdd_HHMMSS>_<job>/` with the resolved
`config.json`, `provenance.json` (commit, dirty state, host), and its
artifacts.

## Checks

```bash
python scripts/infra/check_source_tree.py
python -m compileall -q src scripts tests
python -m pytest tests/architecture tests/unit
python -m pytest tests/integration
bash scripts/infra/build_docs.sh
```

## Documentation

Start at [docs/index.md](docs/index.md), or build the site with
`bash scripts/infra/build_docs.sh` and open
`docs/pages/documentation/index.html`.
