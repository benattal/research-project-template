"""Named experiment presets of the example application.

The registry is data. Each preset is derived from the reference recipe with
``dataclasses.replace``; sweep arms come from a loop. The registry key is the
single owner of the experiment name, and no code branches on a preset name.
"""

from __future__ import annotations

from dataclasses import replace

from src.apps.example.config import ExampleExperimentConfig, ExampleTrainingConfig
from src.core.training.schedules import LearningRateSchedule


EXAMPLE_PRESETS: dict[str, tuple[str, ExampleExperimentConfig]] = {}


def _register(name: str, description: str, config: ExampleExperimentConfig) -> None:
    if name in EXAMPLE_PRESETS:
        raise ValueError(f"preset {name!r} registered twice")
    EXAMPLE_PRESETS[name] = (description, replace(config, experiment_name=name))


REFERENCE = ExampleExperimentConfig(experiment_name="", training=ExampleTrainingConfig())

_register(
    "example-default",
    "Reference recipe: 2,000 iterations, 100-iteration warmup, cosine anneal.",
    REFERENCE)

# A shorter horizon gets its own schedule rather than inheriting the
# reference warmup, which would fill a tenth of the phase.
_register(
    "example-smoke",
    "Smoke test: 100 iterations with a schedule built for that horizon.",
    replace(REFERENCE, training=replace(
        REFERENCE.training,
        iterations=100,
        log_interval=25,
        schedule=LearningRateSchedule(
            warmup_iters=10, peak_factor=1.0, min_factor=0.05))))

for _scale in (0.25, 0.5, 2.0, 4.0):
    _register(
        f"example-lr-x{_scale:g}",
        f"Learning-rate sweep arm: {_scale:g} times the reference rate.",
        replace(REFERENCE, training=replace(
            REFERENCE.training,
            learning_rate=REFERENCE.training.learning_rate * _scale)))
