# Unbiased estimation from noisy data

Lessons from training models on corrupted measurements (Gaussian and Poisson
noise, Bernoulli pixel deletion, blur) when clean data is unavailable and
the loss must be unbiased for the clean one. The central tool is a set of
independent noisy views of each signal: products over distinct views have
the expectations of products of clean signals. The notes cover how to build
those products, what silently breaks them, how the common noise models
enter, when unbiasedness is worth its variance, and how to measure bias and
variance separately. Each file opens with the problem it covers; each entry
states the insight, the mechanism, how to apply it, and the evidence.

| File | Subject |
|---|---|
| [independent-views.md](independent-views.md) | Bilinear identities over distinct views, U- and V-statistics, excluding self-pairs, cross-fitting roles, affine maps, and noisy targets |
| [sources-of-bias.md](sources-of-bias.md) | Curvature of nonlinear stages, fitted and routed approximations, noisy fit locations, leaks through seeds and conditioning, non-identifying objectives, tuned corrections and clipping |
| [normalization-and-self-normalization.md](normalization-and-self-normalization.md) | Ratio estimators: unbiased numerators and labelled denominators, signed masses in reciprocals, floors, dead zones, scale matching, learned normalizers, averaging order and precision |
| [noise-models.md](noise-models.md) | Poisson scaling, Gaussian deconvolution, Bernoulli weighting, support normalization, disjoint pixel partitions and known linear operators |
| [bias-variance-trade-offs.md](bias-variance-trade-offs.md) | Where a plug-in beats an unbiased estimator, capture budgets, clean versus noisy deployment targets, unbounded unbiased losses, batch scaling and lost coverage |
| [measuring-bias-and-variance.md](measuring-bias-and-variance.md) | Frozen-model probes, one-ingredient swaps, clean ceilings, baselines, metric floors, paired seeds, identity tests and configuration validation |

For a new problem, start with [independent-views.md](independent-views.md)
for the construction and
[measuring-bias-and-variance.md](measuring-bias-and-variance.md) for the
controls, then read [sources-of-bias.md](sources-of-bias.md) before adding
any nonlinear stage.
