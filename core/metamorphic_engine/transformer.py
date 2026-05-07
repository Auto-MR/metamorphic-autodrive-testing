"""
Transformation dispatcher.
Maps transform name strings → callable transform functions.
"""

import numpy as np
from transformations.image_transforms    import apply_fog, apply_rain, apply_snow, flip_horizontal, adjust_brightness, add_gaussian_noise, rotate_image
from transformations.geometric_transforms import scale_image, translate_image
from transformations.sequence_transforms  import shift_sequence, reverse_sequence, add_sequence_noise


# ── Image transform registry ─────────────────────────────────────────────────

IMAGE_TRANSFORMS = {
    "flip":       flip_horizontal,
    "brightness": lambda img: adjust_brightness(img, factor=1.3),
    "noise":      lambda img: add_gaussian_noise(img, sigma=0.02),
    "rotate_5":   lambda img: rotate_image(img, angle=5),
    "rotate_neg5":lambda img: rotate_image(img, angle=-5),
    "scale_up":   lambda img: scale_image(img, factor=1.1),
    "translate":  lambda img: translate_image(img, dx=5, dy=0),
    # === NEW WEATHER TRANSFORMS ===
    "rain":         lambda img: apply_rain(img, intensity=0.6),
    "snow":         lambda img: apply_snow(img, intensity=0.5),
    "fog":          lambda img: apply_fog(img, intensity=0.45),
}

# ── Sequence transform registry ───────────────────────────────────────────────

SEQUENCE_TRANSFORMS = {
    "shift":       lambda seq: shift_sequence(seq, dx=1.0, dy=0.0),
    "reverse":     reverse_sequence,
    "noise":       lambda seq: add_sequence_noise(seq, sigma=0.05),
}


def get_transform(name: str, domain: str = "image"):
    """
    Retrieve a transform callable by name.

    Args:
        name:   key from IMAGE_TRANSFORMS or SEQUENCE_TRANSFORMS
        domain: "image" | "sequence"

    Returns:
        Callable transform function.
    """
    registry = IMAGE_TRANSFORMS if domain == "image" else SEQUENCE_TRANSFORMS
    if name not in registry:
        raise KeyError(f"Unknown transform '{name}' for domain '{domain}'. "
                       f"Available: {list(registry.keys())}")
    return registry[name]
