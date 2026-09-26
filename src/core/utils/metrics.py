"""Image and array metrics with explicit conventions.

Metrics are computed from full-precision arrays. The PSNR peak is always a
caller-supplied constant tied to the reference's known range, never a
per-run maximum, so arms scored with the same peak are comparable.
"""

from __future__ import annotations

import math

import numpy as np


def mean_squared_error(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Mean of the squared difference over every element."""
    if prediction.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: {prediction.shape} against {reference.shape}")
    difference = np.asarray(prediction, np.float64) - np.asarray(reference, np.float64)
    return float(np.mean(difference * difference))


def psnr(prediction: np.ndarray, reference: np.ndarray, *, peak: float) -> float:
    """Peak signal-to-noise ratio in decibels, for a pinned ``peak``."""
    if peak <= 0.0:
        raise ValueError("peak must be positive")
    error = mean_squared_error(prediction, reference)
    if error == 0.0:
        return math.inf
    return 10.0 * math.log10(peak * peak / error)
