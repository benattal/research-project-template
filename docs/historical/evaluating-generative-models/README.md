# Evaluating generative models

Lessons from measuring whether a generator recovered the right
distribution: from 2D and 1D targets known in closed form, from exact
posteriors over a finite catalogue, and from image datasets (CelebA and
ImageNet) scored with FID and its companions. The recurring theme is that
one aggregate number hides the specific failure, so an evaluation needs
metrics chosen for the failures it must catch, references that give those
metrics a scale, and pictures that can veto them. Each file opens with the
problem it covers; each entry states the insight, the mechanism, how to
apply it, and the evidence.

| File | Subject |
|---|---|
| [choosing-metrics.md](choosing-metrics.md) | Families of population distances, binned versus transport metrics, support-tail diagnostics, per-sample versus population scores, and a metric reversal that changed a method of record |
| [toy-problems-as-instruments.md](toy-problems-as-instruments.md) | Exact targets, a scene suite that stresses one failure per scene, clean-data gates, oracle arms, bias-versus-variance probes, and diagnostics that repeat the training computation |
| [sample-size-and-seeds.md](sample-size-and-seeds.md) | Finite-sample floors, fixed sample counts and evaluators, seed spread and paired differences, converged versus best-over-run values, EMA weights, and checks that an evaluation ran |
| [image-metrics.md](image-metrics.md) | FID sanity bracketing, sample-count bias, extractor and protocol choices, grain versus blur, FID gains from sharper means, precision, recall, class consistency and guidance sweeps |
| [posterior-evaluation.md](posterior-evaluation.md) | Exact finite-catalogue posteriors, per-condition mean and dispersion, credible-set coverage, why PSNR rewards the posterior mean, and per-class label adherence |
| [reproducing-baselines.md](reproducing-baselines.md) | Component parity with a published method, tuned baselines and ceilings, matched steps, convergence and stability gates, bisecting failures, and retiring runs trained under bugs |
| [visual-inspection.md](visual-inspection.md) | Grid and density layouts that expose failures: fixed noise, several draws per condition, support overlays, live and EMA pairs, truthful trajectory pages and named rejection criteria |

Start with [choosing-metrics.md](choosing-metrics.md) and
[toy-problems-as-instruments.md](toy-problems-as-instruments.md) when
designing an evaluation. Read
[sample-size-and-seeds.md](sample-size-and-seeds.md) before comparing two
arms, and [posterior-evaluation.md](posterior-evaluation.md) for any
sampler conditioned on a measurement.
