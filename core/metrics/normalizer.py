"""
Output Normalizer — normalizes model outputs to [0, 1] for fair cross-model comparison.

Without normalization, different models produce different output ranges and
generalization scores cannot be compared fairly.

Usage:
    from core.metrics.normalizer import normalize_steering_output
    y_norm = normalize_steering_output(y)  # float in [-1,1] → [0,1]
"""

import numpy as np
from typing import List, Dict


STEERING_MIN = -1.0
STEERING_MAX = 1.0


def normalize_steering_output(y: float) -> float:
    """
    Normalize a steering output from [-1, 1] to [0, 1].
    Formula: y_norm = (y - min) / (max - min)
    """
    return float((y - STEERING_MIN) / (STEERING_MAX - STEERING_MIN))


def normalize_batch(outputs: List[float], min_val: float = None, max_val: float = None) -> List[float]:
    """
    Normalize a batch of outputs using min-max normalization.
    If min/max not provided, uses min/max of the batch itself.
    """
    arr = np.array(outputs, dtype=np.float32)
    lo = min_val if min_val is not None else arr.min()
    hi = max_val if max_val is not None else arr.max()
    if hi - lo < 1e-8:
        return [0.5] * len(outputs)
    return list((arr - lo) / (hi - lo))


def normalize_outputs_per_model(results_map: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """
    Normalize outputs separately per model, then return normalized versions.
    Enables fair comparison across architectures with different output ranges.
    """
    normalized = {}
    for model_name, outputs in results_map.items():
        normalized[model_name] = normalize_batch(outputs)
    return normalized
