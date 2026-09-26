"""End-to-end example runs: artifacts, reproducibility, and stream isolation."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from src.apps.example.presets import EXAMPLE_PRESETS
from src.apps.example.trainer import train


def _smoke(tmp_path, name: str, **training_overrides):
    _, config = EXAMPLE_PRESETS["example-smoke"]
    return replace(
        config,
        output_root=str(tmp_path / name),
        training=replace(config.training, **training_overrides))


def test_smoke_run_writes_its_artifacts_and_learns(tmp_path) -> None:
    result = train(_smoke(tmp_path, "run"))
    for name in ("config.json", "provenance.json", "metrics.json", "coefficients.json"):
        assert (result.run.output_dir / name).is_file(), name
    metrics = json.loads((result.run.output_dir / "metrics.json").read_text())
    assert metrics["psnr_peak"] == 2.0
    # The minibatch loss sits at the noise floor (noise_std squared) from the
    # first record on; the noise-free held-out error shows the fit improving.
    assert result.monitor_losses[-1] < result.monitor_losses[0]
    assert result.evaluation_psnr > 30.0


def test_repeated_run_reproduces_the_fit(tmp_path) -> None:
    first = train(_smoke(tmp_path, "first"))
    second = train(_smoke(tmp_path, "second"))
    np.testing.assert_array_equal(first.coefficients, second.coefficients)


def test_monitoring_does_not_perturb_training(tmp_path) -> None:
    monitored = train(_smoke(tmp_path, "monitored", monitor_samples=512))
    unmonitored = train(_smoke(tmp_path, "unmonitored", monitor_samples=0))
    assert monitored.monitor_losses and not unmonitored.monitor_losses
    np.testing.assert_array_equal(monitored.coefficients, unmonitored.coefficients)


def test_command_line_selects_a_preset_and_applies_overrides(tmp_path) -> None:
    pytest.importorskip("tyro")
    from src.apps.example.__main__ import main

    main(["example-smoke", "--output-root", str(tmp_path), "--training.iterations", "40"])
    (run_directory,) = tmp_path.iterdir()
    saved = json.loads((run_directory / "config.json").read_text())
    assert saved["experiment_name"] == "example-smoke"
    assert saved["training"]["iterations"] == 40
