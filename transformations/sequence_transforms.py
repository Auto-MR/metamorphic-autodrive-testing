"""
Sequence transformations for metamorphic testing of trajectory models.
All functions accept np.ndarray of shape (T, 2) and return same shape.
"""

import numpy as np


def shift_sequence(seq: np.ndarray, dx: float = 1.0, dy: float = 0.0) -> np.ndarray:
    """Translate all waypoints by (dx, dy)."""
    offset = np.array([dx, dy], dtype=np.float32)
    return (seq + offset).astype(np.float32)


def reverse_sequence(seq: np.ndarray) -> np.ndarray:
    """Reverse the temporal order of waypoints."""
    return seq[::-1].copy().astype(np.float32)


def add_sequence_noise(seq: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """Add Gaussian noise to each waypoint."""
    noise = np.random.normal(0, sigma, seq.shape).astype(np.float32)
    return (seq + noise).astype(np.float32)


def scale_sequence(seq: np.ndarray, factor: float = 1.1) -> np.ndarray:
    """Scale waypoint positions around centroid."""
    centroid = seq.mean(axis=0)
    return ((seq - centroid) * factor + centroid).astype(np.float32)


def rotate_sequence(seq: np.ndarray, angle_deg: float = 5.0) -> np.ndarray:
    """Rotate trajectory around centroid by angle_deg degrees."""
    theta = np.radians(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
    centroid = seq.mean(axis=0)
    return ((seq - centroid) @ R.T + centroid).astype(np.float32)
