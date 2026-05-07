"""
Noise models for systematic robustness testing.
Provides parameterized noise generators for images and sequences.
"""

import numpy as np


def gaussian_noise(data: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    return np.clip(data + np.random.normal(0, sigma, data.shape), 0, 1).astype(np.float32)


def salt_and_pepper(image: np.ndarray, prob: float = 0.02) -> np.ndarray:
    out = image.copy()
    rnd = np.random.rand(*image.shape[:2])
    out[rnd < prob / 2]     = 0.0
    out[rnd > 1 - prob / 2] = 1.0
    return out.astype(np.float32)


def uniform_noise(data: np.ndarray, low: float = -0.05, high: float = 0.05) -> np.ndarray:
    return np.clip(data + np.random.uniform(low, high, data.shape), 0, 1).astype(np.float32)


def impulse_noise(sequence: np.ndarray, prob: float = 0.1, sigma: float = 0.5) -> np.ndarray:
    """Random spike noise for sequence data."""
    out = sequence.copy()
    mask = np.random.rand(*sequence.shape) < prob
    out[mask] += np.random.normal(0, sigma, mask.sum())
    return out.astype(np.float32)
