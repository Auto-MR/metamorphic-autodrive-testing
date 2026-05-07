"""
Geometric transformations for metamorphic testing.
"""

import numpy as np
import cv2


def scale_image(image: np.ndarray, factor: float = 1.1) -> np.ndarray:
    """Scale image by factor, then center-crop back to original size."""
    h, w = image.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    scaled = cv2.resize(image, (new_w, new_h))
    # Center crop
    y0 = (new_h - h) // 2
    x0 = (new_w - w) // 2
    return scaled[y0:y0 + h, x0:x0 + w].astype(np.float32)


def translate_image(image: np.ndarray, dx: int = 5, dy: int = 0) -> np.ndarray:
    """Translate image by (dx, dy) pixels with reflect border."""
    h, w = image.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, M, (w, h),
                          borderMode=cv2.BORDER_REFLECT).astype(np.float32)


def crop_center(image: np.ndarray, crop_ratio: float = 0.9) -> np.ndarray:
    """Center-crop then resize back to original dimensions."""
    h, w = image.shape[:2]
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    cropped = image[y0:y0 + ch, x0:x0 + cw]
    return cv2.resize(cropped, (w, h)).astype(np.float32)
