"""Learning-rate schedules have the documented shape and reject bad horizons."""

from __future__ import annotations

import math

import pytest

from src.core.training.schedules import LearningRateSchedule


def test_warmup_rises_from_zero_to_peak() -> None:
    schedule = LearningRateSchedule(warmup_iters=10, peak_factor=2.0, min_factor=0.5)
    assert schedule.factor(0, 100) == 0.0
    assert schedule.factor(5, 100) == pytest.approx(1.0)
    assert schedule.factor(10, 100) == pytest.approx(2.0)


def test_anneal_reaches_the_floor_at_the_end_of_the_phase() -> None:
    schedule = LearningRateSchedule(warmup_iters=10, peak_factor=1.0, min_factor=0.1)
    assert schedule.factor(99, 100) == pytest.approx(0.1, abs=1e-3)
    assert schedule.factor(100, 100) == pytest.approx(0.1)
    assert schedule.factor(1000, 100) == pytest.approx(0.1)


def test_hold_keeps_the_peak() -> None:
    schedule = LearningRateSchedule(warmup_iters=10, hold_iters=20, peak_factor=1.0, min_factor=0.0)
    assert all(schedule.factor(i, 100) == 1.0 for i in range(10, 30))
    assert schedule.factor(30, 100) == 1.0
    assert schedule.factor(31, 100) < 1.0


def test_fixed_anneal_is_identical_across_horizons() -> None:
    schedule = LearningRateSchedule(warmup_iters=10, anneal_iters=50, min_factor=0.2)
    short = [schedule.factor(i, 100) for i in range(100)]
    long = [schedule.factor(i, 1000) for i in range(100)]
    assert short == long


def test_constant_schedule_after_warmup() -> None:
    schedule = LearningRateSchedule(warmup_iters=4)
    assert [schedule.factor(i, 20) for i in range(4, 20)] == [1.0] * 16
    assert math.isclose(schedule.factor(2, 20), 0.5)


def test_horizon_too_short_for_the_schedule_is_rejected() -> None:
    schedule = LearningRateSchedule(warmup_iters=100, anneal_iters=50)
    schedule.check_horizon(150)
    with pytest.raises(ValueError, match="construct a schedule"):
        schedule.check_horizon(149)


def test_invalid_factors_are_rejected() -> None:
    with pytest.raises(ValueError):
        LearningRateSchedule(peak_factor=0.5, min_factor=1.0)
    with pytest.raises(ValueError):
        LearningRateSchedule(warmup_iters=-1)
