"""Learning-rate schedules owned by each optimizer preset.

A schedule maps an iteration index within its phase to a multiplier of the
optimizer's base learning rate. Every preset constructs one explicitly; the
training loop reads it and nothing else rewrites, multiplies, or resets it
(see ``docs/style/code.md``, "Explicit optimization schedules").

The phase length is owned by the training configuration, not the schedule.
The schedule receives it at each call and validates once, at construction of
the owning configuration, that its fixed durations fit that phase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LearningRateSchedule:
    """Cosine warmup, an optional hold, then a cosine anneal to a floor.

    Attributes:
        warmup_iters: Iterations of cosine warmup from 0 to ``peak_factor``.
        hold_iters: Iterations held at ``peak_factor`` after the warmup.
        anneal_iters: Duration of the cosine anneal from ``peak_factor`` to
            ``min_factor``. Zero anneals over the remainder of the phase; a
            positive value keeps the envelope identical across phases of
            different length.
        peak_factor: Multiplier reached at the end of the warmup.
        min_factor: Multiplier held once the anneal ends. Equal to
            ``peak_factor`` for a constant schedule after warmup.
    """

    warmup_iters: int = 0
    hold_iters: int = 0
    anneal_iters: int = 0
    peak_factor: float = 1.0
    min_factor: float = 1.0

    def __post_init__(self) -> None:
        for name in ("warmup_iters", "hold_iters", "anneal_iters"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.min_factor <= self.peak_factor:
            raise ValueError(
                "expected 0 <= min_factor <= peak_factor, got "
                f"{self.min_factor} and {self.peak_factor}")

    def check_horizon(self, phase_iters: int) -> None:
        """Reject a phase too short to hold this schedule's fixed durations.

        A preset that shortens its phase must construct a schedule for the new
        horizon rather than inherit one built for a longer run.
        """
        fixed = self.warmup_iters + self.hold_iters + self.anneal_iters
        if phase_iters <= 0:
            raise ValueError("phase_iters must be positive")
        if fixed > phase_iters:
            raise ValueError(
                f"schedule needs {fixed} iterations of warmup, hold and "
                f"anneal but the phase has {phase_iters}; construct a "
                "schedule for this horizon")

    def factor(self, iteration: int, phase_iters: int) -> float:
        """Learning-rate multiplier at ``iteration`` of a ``phase_iters`` phase."""
        if iteration < self.warmup_iters:
            progress = iteration / self.warmup_iters
            return self.peak_factor * 0.5 * (1.0 - math.cos(math.pi * progress))
        anneal_start = self.warmup_iters + self.hold_iters
        if iteration < anneal_start:
            return self.peak_factor
        anneal_iters = self.anneal_iters or phase_iters - anneal_start
        if anneal_iters <= 0:
            return self.peak_factor
        # Clamped: past the anneal the raw cosine would rise again.
        progress = min((iteration - anneal_start) / anneal_iters, 1.0)
        span = self.peak_factor - self.min_factor
        return self.min_factor + span * 0.5 * (1.0 + math.cos(math.pi * progress))
