"""
Image transformations for metamorphic testing of vision models.
All functions accept np.ndarray (H, W, 3) float32 in [0, 1] and return same.
"""

import numpy as np
import cv2


def flip_horizontal(image: np.ndarray) -> np.ndarray:
    """Mirror image left-to-right."""
    return image[:, ::-1].copy()


def flip_vertical(image: np.ndarray) -> np.ndarray:
    """Mirror image top-to-bottom."""
    return image[::-1, :].copy()


def adjust_brightness(image: np.ndarray, factor: float = 1.3) -> np.ndarray:
    """Scale pixel intensities by factor, clamp to [0, 1]."""
    return np.clip(image * factor, 0.0, 1.0).astype(np.float32)


def add_gaussian_noise(image: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    """Add zero-mean Gaussian noise."""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0).astype(np.float32)


def rotate_image(image: np.ndarray, angle: float = 5.0) -> np.ndarray:
    """Rotate image by `angle` degrees around center, no crop."""
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
    return rotated.astype(np.float32)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB to 3-channel grayscale (channels kept for compatibility)."""
    gray = np.mean(image, axis=-1, keepdims=True)
    return np.repeat(gray, 3, axis=-1).astype(np.float32)


def apply_rain(image: np.ndarray, intensity: float = 0.6, n_drops: int = 150) -> np.ndarray:
    """Simulate rainy day effect."""
    out = image.copy()
    h, w = image.shape[:2]
    
    # Add rain streaks
    for _ in range(n_drops):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        length = np.random.randint(8, 20)
        cv2.line(out, (x, y), (x + 2, y + length), (0.7, 0.7, 0.8), 1)
    
    # Add slight blue tint + darkening
    out = out * 0.85
    out[:, :, 2] = np.clip(out[:, :, 2] * 1.1, 0, 1)  # boost blue channel
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_snow(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """Simulate snowy / overcast day."""
    out = image.copy()
    h, w = image.shape[:2]
    
    # Add snowflakes
    num_snow = int(h * w * 0.015)
    coords = np.random.randint(0, [h, w], size=(num_snow, 2))
    for y, x in coords:
        size = np.random.randint(2, 5)
        cv2.circle(out, (x, y), size, (0.95, 0.95, 0.98), -1)
    
    # Cool color temperature + brightness reduction
    out = out * (0.75 + intensity * 0.2)
    out[:, :, 0] = np.clip(out[:, :, 0] * 1.05, 0, 1)  # slight blue tint
    out[:, :, 1] = np.clip(out[:, :, 1] * 0.95, 0, 1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_fog(image: np.ndarray, intensity: float = 0.45) -> np.ndarray:
    """Heavy fog / mist effect."""
    fog = np.ones_like(image) * 0.85
    return np.clip(image * (1 - intensity) + fog * intensity, 0.0, 1.0).astype(np.float32)