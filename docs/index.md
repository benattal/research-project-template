# TODO(template): Project name

TODO(template): two or three sentences stating what the method does and why.

## Concepts

Start with the [concept map](concepts/index.md). Each application has one
concept document:

- [example](concepts/example.md).

## Implementation

- [Architecture](implementation/architecture.md)
- [Randomness and performance](implementation/performance.md)
- [Troubleshooting](implementation/troubleshooting.md)
- [Experiment workflows](implementation/experiments.md)
- [Results pages](implementation/results-pages.md)
- [Documentation build](implementation/documentation.md)

Application implementation maps:

- [example](implementation/apps/example.md).

The generated API Reference section documents the current `src.*` tree
without importing it.

## Style

- [Code style and organization](style/code.md)
- [Writing and math style](style/writing.md)
- [Results presentation style](style/results.md)

## Environments

[Environments and running the code](environments/index.md) is the
authoritative setup and launch guide.

## Historical notes

[Historical engineering notes](historical/README.md) collect durable,
project-independent lessons: [Mitsuba and DrJit](historical/mitsuba/README.md),
[real-time rendering](historical/real-time-rendering/README.md),
[clusters and shared hosts](historical/clusters/README.md),
[unbiased estimation](historical/unbiased-estimation/README.md),
[evaluating generative models](historical/evaluating-generative-models/README.md),
and [training engineering](historical/training-engineering/README.md).

## Papers, plans, and tracking

- [Papers and references](papers/README.md), with the
  [submission record](papers/submission/README.md)
- [Research provenance](papers/provenance.md)
- [Plans](plans/README.md)
- [Live tracking](tracking/README.md)
- [Production configurations](production-configs.md)
- [Archive](archive/README.md)

Changing operational state belongs in `docs/tracking/`, not in agent,
implementation, concept, or style documentation.
