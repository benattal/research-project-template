"""Configuration of the example application.

Every numerical parameter has one field and one owner. The dataclasses are
frozen: runtime code reads them and records derived state in a ``RunContext``
instead of mutating them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.training.schedules import LearningRateSchedule


@dataclass(frozen=True)
class ExampleTrainingConfig:
    """Minibatch gradient descent on a polynomial fit.

    Attributes:
        iterations: Length of the training phase; also the schedule horizon.
        batch_size: Noisy samples per gradient estimate.
        learning_rate: Base rate, multiplied by ``schedule`` each iteration.
        schedule: The phase's complete learning-rate schedule.
        polynomial_degree: Highest degree of the orthonormal Legendre basis.
        noise_std: Standard deviation of the additive observation noise.
        evaluation_points: Size of the uniform grid the final fit is scored on.
        log_interval: Iterations between loss records.
        monitor_samples: Fresh held-out samples scored at each log interval,
            drawn from their own random domain; zero records no held-out loss.
    """

    iterations: int = 2000
    batch_size: int = 64
    learning_rate: float = 0.2
    schedule: LearningRateSchedule = field(
        default_factory=lambda: LearningRateSchedule(
            warmup_iters=100, peak_factor=1.0, min_factor=0.05))
    polynomial_degree: int = 5
    noise_std: float = 0.1
    evaluation_points: int = 257
    log_interval: int = 200
    monitor_samples: int = 256

    def __post_init__(self) -> None:
        for name in ("iterations", "batch_size", "evaluation_points", "log_interval"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.polynomial_degree < 0 or self.noise_std < 0.0 or self.monitor_samples < 0:
            raise ValueError(
                "polynomial_degree, noise_std and monitor_samples must be non-negative")
        self.schedule.check_horizon(self.iterations)


@dataclass(frozen=True)
class ExampleExperimentConfig:
    """One complete example experiment.

    Attributes:
        experiment_name: Set by the preset registry to the preset's name.
        seed: Base seed of every named random domain in the run.
        output_root: Directory that receives the run directory.
        training: The training phase.
    """

    experiment_name: str
    seed: int = 0
    output_root: str = "outputs"
    training: ExampleTrainingConfig = field(default_factory=ExampleTrainingConfig)
