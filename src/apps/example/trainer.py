"""Training lifecycle of the example application.

The example fits an orthonormal Legendre expansion to noisy samples of
``sin(pi x)`` on ``[-1, 1]`` by minibatch gradient descent, then scores the
fit on a uniform grid against the noise-free function. See
``docs/concepts/example.md`` for the mathematics.

Random domains: ``initialization``, ``training-inputs``, ``training-noise``
drive the fit; ``monitoring`` supplies held-out samples and never moves the
training streams.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.apps.example.config import ExampleExperimentConfig
from src.core.training.run_context import RunContext
from src.core.utils.metrics import psnr
from src.core.utils.randomness import RandomDomains


# Peak-to-peak range of sin(pi x) on [-1, 1]: the pinned PSNR peak.
TARGET_PEAK = 2.0


def target_function(x: np.ndarray) -> np.ndarray:
    """The noise-free function being fitted."""
    return np.sin(np.pi * x)


def legendre_features(x: np.ndarray, degree: int) -> np.ndarray:
    """Legendre polynomials scaled to unit second moment under U[-1, 1].

    Returns an array of shape ``(len(x), degree + 1)``. The scaling by
    ``sqrt(2k + 1)`` makes the least-squares Hessian close to the identity,
    so one learning rate suits every coefficient.
    """
    scale = np.sqrt(2.0 * np.arange(degree + 1) + 1.0)
    return np.polynomial.legendre.legvander(x, degree) * scale


@dataclass(frozen=True)
class ExampleResult:
    """What one run produced.

    Attributes:
        run: The run's directory and identity.
        coefficients: Final expansion coefficients, lowest degree first.
        training_losses: Minibatch mean squared error at each log interval.
        monitor_losses: Held-out mean squared error at each log interval.
        evaluation_psnr: PSNR of the final fit on the evaluation grid, in dB.
    """

    run: RunContext
    coefficients: np.ndarray
    training_losses: list[float]
    monitor_losses: list[float]
    evaluation_psnr: float


def train(config: ExampleExperimentConfig) -> ExampleResult:
    """Run one experiment and write its artifacts to a fresh run directory."""
    run = RunContext.create(config.experiment_name, config.output_root)
    run.save_provenance(config)
    domains = RandomDomains(config.seed)
    training = config.training
    degree = training.polynomial_degree

    coefficients = domains.stream("initialization").normal(0.0, 0.01, degree + 1)
    training_losses: list[float] = []
    monitor_losses: list[float] = []

    for iteration in range(training.iterations):
        inputs = domains.stream("training-inputs").uniform(-1.0, 1.0, training.batch_size)
        noise = domains.stream("training-noise").normal(0.0, training.noise_std, training.batch_size)
        observations = target_function(inputs) + noise

        features = legendre_features(inputs, degree)
        residual = features @ coefficients - observations
        gradient = features.T @ residual / training.batch_size
        rate = training.learning_rate * training.schedule.factor(iteration, training.iterations)
        coefficients = coefficients - rate * gradient

        if (iteration + 1) % training.log_interval == 0:
            training_losses.append(float(np.mean(residual * residual)))
            message = f"[{config.experiment_name}] {iteration + 1}/{training.iterations} "
            message += f"loss {training_losses[-1]:.5f}"
            if training.monitor_samples:
                held_out = domains.stream("monitoring").uniform(-1.0, 1.0, training.monitor_samples)
                error = legendre_features(held_out, degree) @ coefficients - target_function(held_out)
                monitor_losses.append(float(np.mean(error * error)))
                message += f"  held-out {monitor_losses[-1]:.5f}"
            print(message, flush=True)

    grid = np.linspace(-1.0, 1.0, training.evaluation_points)
    prediction = legendre_features(grid, degree) @ coefficients
    evaluation_psnr = psnr(prediction, target_function(grid), peak=TARGET_PEAK)

    run.write_json("coefficients.json", {"basis": "orthonormal-legendre", "coefficients": coefficients})
    run.write_json("metrics.json", {
        "evaluation_psnr_db": evaluation_psnr,
        "psnr_peak": TARGET_PEAK,
        "evaluation_points": training.evaluation_points,
        "training_losses": training_losses,
        "monitor_losses": monitor_losses,
    })
    print(f"[{config.experiment_name}] PSNR {evaluation_psnr:.2f} dB -> {run.output_dir}", flush=True)
    return ExampleResult(run, coefficients, training_losses, monitor_losses, evaluation_psnr)
