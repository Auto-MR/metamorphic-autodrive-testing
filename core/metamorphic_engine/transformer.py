"""
Transformer registry — maps string keys to transform functions.
Includes all single transforms + composed (Category D) transforms.
"""

import numpy as np
from typing import Callable


def get_transform(name: str, domain: str = "image") -> Callable:
    """Return a transform function by name."""

    if domain == "sequence":
        from transformations.sequence_transforms import (
            translate_sequence, add_sequence_noise, reverse_sequence
        )
        return {
            "translate": translate_sequence,
            "noise":     add_sequence_noise,
            "reverse":   reverse_sequence,
        }.get(name, lambda x: x)

    # Image transforms
    from transformations.image_transforms import (
        flip_horizontal, adjust_brightness, add_gaussian_noise,
        rotate_image, apply_rain, apply_snow, apply_fog
    )
    from transformations.geometric_transforms import (
        scale_image, translate_image, crop_center
    )
    from transformations.noise_models import gaussian_noise

    def composed_brightness_noise(img):
        img2 = adjust_brightness(img, factor=1.3)
        return add_gaussian_noise(img2, sigma=0.03)

    def composed_weather_blur(img):
        import cv2
        img2 = apply_fog(img, intensity=0.35)
        return cv2.GaussianBlur(img2, (5, 5), 0)

    def composed_shift_brightness(img):
        img2 = translate_image(img, dx=8, dy=0)
        return adjust_brightness(img2, factor=1.2)

    TRANSFORMS = {
        # Single — Category A
        "brightness":        lambda img: adjust_brightness(img, factor=1.3),
        "brightness_dark":   lambda img: adjust_brightness(img, factor=0.7),
        "contrast":          lambda img: adjust_brightness(img, factor=1.5),
        "noise":             lambda img: add_gaussian_noise(img, sigma=0.025),
        "rain":              apply_rain,
        "snow":              apply_snow,
        "fog":               apply_fog,
        # Single — Category B
        "flip":              flip_horizontal,
        "rotate_5":          lambda img: rotate_image(img, 5.0),
        "scale":             lambda img: scale_image(img, 1.1),
        "crop":              lambda img: crop_center(img, 0.9),
        # Single — Category C
        "translate":         lambda img: translate_image(img, dx=5, dy=0),
        "translate_large":   lambda img: translate_image(img, dx=12, dy=0),
        # Composed — Category D
        "composed_bright_noise":   composed_brightness_noise,
        "composed_weather_blur":   composed_weather_blur,
        "composed_shift_bright":   composed_shift_brightness,
    }

    if name not in TRANSFORMS:
        raise KeyError(f"Unknown transform '{name}'. Available: {list(TRANSFORMS.keys())}")
    return TRANSFORMS[name]


def list_transforms() -> list:
    """Return all registered transform names."""
    from transformations.image_transforms import (
        flip_horizontal, adjust_brightness, add_gaussian_noise, rotate_image
    )
    return [
        # Category A
        "brightness", "brightness_dark", "contrast", "noise", "rain", "snow", "fog",
        # Category B
        "flip", "rotate_5", "scale", "crop",
        # Category C
        "translate", "translate_large",
        # Category D
        "composed_bright_noise", "composed_weather_blur", "composed_shift_bright",
    ]
