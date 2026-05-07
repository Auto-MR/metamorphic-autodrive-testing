"""
Steering-specific evaluation metrics.
"""

import numpy as np


def steering_jitter(angles: np.ndarray) -> float:
    """Steering jitter = variance of first differences (smoothness)"""
    if len(angles) < 2:
        return 0.0
    diffs = np.diff(angles)
    return float(np.var(diffs))


def stability_index(angles: np.ndarray) -> float:
    """Overall stability (lower is better)"""
    return float(np.var(angles))


def mae_steering(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def max_deviation(angles: np.ndarray) -> float:
    """Maximum absolute steering angle (useful for straight road check)"""
    return float(np.max(np.abs(angles)))