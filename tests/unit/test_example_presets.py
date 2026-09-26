"""Example presets resolve to complete, independent configurations."""

from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from src.apps.example.presets import EXAMPLE_PRESETS, REFERENCE


def test_registry_key_owns_the_experiment_name() -> None:
    for name, (description, config) in EXAMPLE_PRESETS.items():
        assert config.experiment_name == name
        assert description


def test_every_preset_schedule_fits_its_horizon() -> None:
    for _, config in EXAMPLE_PRESETS.values():
        config.training.schedule.check_horizon(config.training.iterations)


def test_sweep_arms_change_only_the_learning_rate() -> None:
    reference = asdict(REFERENCE.training)
    arms = [name for name in EXAMPLE_PRESETS if name.startswith("example-lr-")]
    assert arms
    for name in arms:
        training = asdict(EXAMPLE_PRESETS[name][1].training)
        changed = {key for key in reference if training[key] != reference[key]}
        assert changed == {"learning_rate"}, (name, changed)


def test_shortening_the_horizon_requires_a_new_schedule() -> None:
    with pytest.raises(ValueError, match="construct a schedule"):
        replace(REFERENCE.training, iterations=50)


def test_configuration_is_immutable() -> None:
    _, config = EXAMPLE_PRESETS["example-default"]
    with pytest.raises(AttributeError):
        config.training.iterations = 1  # type: ignore[misc]
