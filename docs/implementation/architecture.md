# Code architecture

The repository has a small set of shared concerns under `src/core/` and one
vertical slice per application under `src/apps/`. A file belongs where a
reader would look for its behavior, not where adding it was momentarily
convenient.

The mandatory naming, cohesion, layout, and review conventions are in
[code style and organization](../style/code.md).

```text
src/
  core/
    training/    schedules, run context and output provenance
    utils/       named random domains, metrics
  apps/
    example/     config, presets, entry, and training for one application
```

TODO(template): list the project's real core concerns (for example `model/`,
`data/`, `rendering/`) and applications. Update `CORE_CONCERNS` and
`APPLICATIONS` in `scripts/infra/check_source_tree.py` in the same change.

`third_party/` contains standalone references and vendored implementations.
Maintained source does not import them.

## Ownership rules

| Concern | Owner |
|---|---|
| Learning-rate schedules, run directories, saved provenance | `src/core/training/` |
| Named random domains, metrics with explicit conventions | `src/core/utils/` |
| Application configuration, presets, entry points, training loops, artifacts | `src/apps/<application>/` |
| Probes, scoring, launchers, pages, installation, and infrastructure | matching `scripts/` role directory |

There is no generic `common`, `helpers`, `runtime`, or `misc` root.

## Semantic cohesion and aesthetics

Long files are welcome when they keep one algorithm or lifecycle together. Do
not split code to meet a numeric line target. In particular, avoid sibling
families such as `trainer_loop.py`, `trainer_eval.py`, and
`trainer_common.py` when they share one state contract.

Names must explain themselves without parent-directory archaeology. Avoid
`core.py`, `base.py`, `common.py`, empty wrapper packages, singular/plural
collisions, and a file and a directory with the same normalized name. Code
layout is part of review: variable names, whitespace, phase ordering, and
visual balance matter.

The blocking checker is:

```bash
python scripts/infra/check_source_tree.py
```

It rejects unexpected core concerns and applications, excessive nesting,
generic names, prefix-fragment families, one-file packages, and confusing
collisions. `tests/architecture/test_source_tree.py` runs it.

## Application layout

Each application is a vertical slice:

| File | Owns |
|---|---|
| `config.py` | frozen dataclasses for every numerical parameter |
| `presets.py` | the registry of named, complete experiment configurations |
| `trainer.py` | the training lifecycle, artifacts, and `main()` |
| `__main__.py` | the command `python -m src.apps.<application>` |

Split a file into a subpackage (`config/`, `training/`, `models/`) only when
it holds several independent public contracts.

## Data-driven application behavior

Shared infrastructure represents one stable program whose variation is
described primarily by numerical and structural data: rates, budgets,
dimensions, schedules, paths, and parameters. Concrete components are
assembled before execution and expose stable callback interfaces. The shared
lifecycle consumes those values and callbacks without rewriting them or
dispatching on config modes inside the loop.

If a behavior requires extensive conditional control flow, mode-dependent
meanings for existing fields, or callbacks that replace major phases, treat
that difficulty as a design smell and revisit ownership before adding
structure. A different lifecycle may receive a clear entry script or workflow
inside its owning application. A new application is not an escape hatch for a
shared abstraction whose data no longer explains what it executes.

Changing a boolean mode into an enum, optional component, or tagged union does
not make mode-dependent control flow data-driven. A callback is appropriate
only when it supplies one bounded operation at an existing lifecycle point.

## Model lifecycle

Model code separates the physical or statistical quantity from the neural
machinery. Shared owners provide encodings, heads, optimizers, EMA,
checkpoint state, and coordinate mappings. Application code names the
quantity, its measure and support, and its supervision objective.

Checkpoint load is strict. Domain, mapping, support, output semantics, EMA
state, and app-specific manifests are persisted. Internal moves update all
callers directly; passive compatibility aliases are not retained.

## Randomness and performance

Every stochastic owner draws from an explicit named domain
(`src/core/utils/randomness.py`). Subsequent calls that should be independent
must not replay correlated samples. Fixed-seed replay is an explicit
diagnostic contract, not an accidental sampler lifetime.

Hot paths avoid host scalar reads, device-to-host transfers, checkpoint I/O,
profiling queries, and allocator flushing. A performance change passes
numerical parity, stable compilation, flat memory, and matched wall-clock
gates. See [randomness and performance](performance.md).

## Placement and clean-break gates

- Production source never imports probes, page builders, profiling scripts,
  or third-party baselines.
- Scripts are filed by role; nothing executable is loose at `scripts/` root.
- Generated pages live under ignored `docs/pages/`; source docs own adjacent
  assets.
- Dead branches, stale flags, forwarding modules, and migration aliases are
  deleted after reachability checks.
- `AGENTS.md` and `CLAUDE.md` change together.
- The source-tree checker, compileall, static tests, the documentation build,
  and the relevant GPU gates pass before promotion.
