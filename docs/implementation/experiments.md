# Experiment, reference, and regression workflows

Experiments are reproducible only when the code, configuration, random
domains, reference, metric, and artifact lineage are explicit. This document
describes the workflow; changing campaign state belongs in `docs/tracking/`.

## Run identity

Every run records, through `src/core/training/run_context.py`:

- application entry and preset;
- the complete resolved configuration including command-line overrides
  (`config.json`);
- dataset, target, and checkpoint paths;
- base seed and the named random domains in use;
- git commit and whether the working tree was dirty;
- host, scheduler job, and hardware metadata;
- phase timing, counters, and final artifacts.

Output directories are named
`outputs/{experiment}_{YYYYmmdd_HHMMSS}_{job-or-process-id}/`. Never infer a
method from the directory name alone; read its saved configuration and
checkpoint manifest.

## Canonical commands

Use application entry points for training and explicit secondary entries for
evaluation and references. Examples are in
[environments](../environments/index.md) and the application implementation
documents.

Role-specific launchers live under `scripts/launchers/`. Give every sweep arm a
unique experiment name by registering it as a preset. Do not add one-off
executable scripts under `data/` or the repository root.

## Reference construction

A ground-truth or reference artifact states:

- the represented component and its base measure;
- the dataset and evaluation contract;
- the estimator or solver and its budget;
- the seed and replicate policy;
- normalization and any smoothing, kernel, or bin width;
- output array and manifest paths.

A convergence claim requires independent budget rungs or replicates; rerunning
an identical fixed stream does not establish convergence. A reference must
contain the same components as the quantity it is compared against; a
truncated reference is a diagnostic, not production evidence.

## Metrics

Metrics are computed from full-precision arrays. Pin:

- component;
- reference array;
- peak or exposure convention;
- mask or pooling rule;
- resolution;
- sample budget.

Report quality and physical level separately where appropriate. PSNR, relative
MSE, correlation, total-mass ratio, and gradient agreement measure different
failures. Never derive a PSNR peak from a run-specific maximum.

Use [results presentation style](../style/results.md) for tables and panels.

## Regression suites

The regression workflow lives under `scripts/regression/`:

1. run a short, fixed-budget representative workflow;
2. compare its outputs to a pinned known-good artifact under matched seeds
   and configuration;
3. report absolute quality as context and A/B parity as the regression gate;
4. compare per-iteration timing only on matched hardware;
5. fail on missing artifacts rather than silently skipping a check.

When a deliberate numerical change moves the expected output, establish and
document the new reference through review. Do not repin merely to make a red
test green.

Architecture and unit tests cover structure and focused contracts; they do
not replace numerical model, gradient, checkpoint, or trajectory gates.

## Replicates and tuning

- Independent replicates vary the base seed while preserving the named-domain
  construction.
- Learning-rate comparisons tune each objective family independently and show
  that the selected optimum is inside the tested range.
- Sample and iteration budgets are matched to the claim being made.
- Diagnostic sampling cannot advance training streams.
- A performance arm reports numerical parity before a speedup.

## Checkpoints and lineage

Load checkpoints only through the preset that matches their mapping, support,
representation, target, and schema. Successful tensor loading is not
validation.

Current production selections are recorded in
[production configurations](../production-configs.md). Experiment documents
link that registry rather than copying run IDs or "best" tables.

## Outputs and cleanup

`outputs/` is ignored; its durable copy is described in `DATA.md`.
Machine-readable cleanup inventories live under `docs/tracking/data/`.

Before deleting outputs:

1. protect production lineages, active jobs, parent checkpoints, and unique
   incomplete evidence;
2. generate an exact manifest;
3. review the manifest;
4. execute deletion only from that manifest;
5. regenerate the post-cleanup inventory.

Never delete by broad glob, timestamp, or "failed-looking" name.

## Result pages

Page builders under `scripts/pages/` own artifact selection and generated
resources. See [results pages](results-pages.md). Generated pages are not
source and remain ignored under `docs/pages/`.
