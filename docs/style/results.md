# Results presentation style

Results must make numerical, visual, and provenance differences visible. A
panel that looks polished but hides scale, support, or sampling differences is
incorrect.

## Numerical source

- Compute metrics from full-precision arrays, not PNGs or screen captures.
- State the represented component: the total, or which term, split, or
  intermediate quantity is shown.
- Pin the base measure, resolution, sample budget, target, and reference
  lineage.
- Use one frozen reference per metric. Never score an arm against a reference
  it generated itself.
- Distinguish a conditional fit target from an independent oracle. A
  reference built from a learned model can validate fitting of that supplied
  target, but cannot independently validate the complete result.
- Pin PSNR peak or exposure conventions (`src/core/utils/metrics.py` requires
  the peak explicitly); do not derive them from one run's maximum.
- Report replicate count and seed policy. Nominal reruns with the same random
  stream are not independent evidence.
- Tune the learning rate per method when the objectives' curvature differs.

## Images and panels

- Use one reference-derived exposure or normalization across compared images.
- Never normalize each panel independently when brightness or magnitude is
  part of the claim.
- Label live versus EMA weights and every displayed component.
- Show a learned quantity as it is read by its consumer, with any factor the
  reader applies (a normalization, a material term) applied before display.
  Apply the same factors to references before forming differences, and record
  the displayed quantity in the page manifest.
- Keep reference and prediction positions consistent across figures.
- For signed quantities, use a symmetric zero-centered color scale and report
  its numeric range.
- For density or support diagnostics, show both spatial structure and total
  mass.
- Avoid clipped 8-bit images for quantitative comparisons.
- Each diagnostic page owns its generated images and metadata under
  `docs/pages/<page>/`.
- Treat each page directory as a standalone bundle: copy or derive every
  runtime asset beneath it and use page-relative URLs. Opening a copied
  `index.html` or extracting the page archive must not require the
  repository's `outputs/`, another generated page, or a server-root path.
- Prefer one compact, well-labeled viewer over a wall of repeated panels when
  results vary along several dimensions. Selectors must reflect the semantic
  hierarchy of the data and never offer combinations without an artifact.
- Visually audit every page after generation: default state, selector
  changes, narrow layout, captions, shared-scale claims, and reference and
  prediction ordering.

## Tables and claims

- Put units in headers and keep precision proportional to estimator noise.
- Report absolute quality and change from baseline separately.
- Pair mean values with dispersion or uncertainty when seeds or Monte Carlo
  batches vary.
- State known invalid or incomparable arms directly; do not let a footnote
  make an invalid number appear accepted.
- A performance table identifies hardware, software environment, warmup,
  measured phase, and whether compilation or artifact writing is included.
- A speedup is promotable only with numerical parity and flat-memory evidence.

## Provenance

- Generate published pages through the registered builder or artifact map
  rather than typing output paths from memory.
- Record source arrays, configs, checkpoints, and scoring commands in page
  metadata.
- Use the maintained [production registry](../production-configs.md) for
  lineage; timestamps and successful loading are not selection criteria.
- Put changing experiment status in `docs/tracking/`, durable conclusions in
  `docs/papers/provenance.md`, and presentation mechanics here.

Operational page-building instructions are in
[results pages](../implementation/results-pages.md), while metric and
reference contracts are in [experiments](../implementation/experiments.md).
