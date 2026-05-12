"""
Parametric Sweep Testing — tests MR pass rates as transformation intensity varies.

Supervisor requirement:
  "Test a series of values, e.g. increment brightness from 0 to infinity —
   how does the pass rate differ?"

Each transform has a named intensity parameter swept across a range.
For each intensity level, we run the MR check on all test images and record
pass rate, mean delta, and violation count.

This produces:
  - Robustness Degradation Curve: pass_rate vs intensity
  - Critical Threshold: intensity where pass_rate first drops below target
  - Delta Growth Curve: how MR delta grows with intensity
  - Cross-model degradation comparison at each sweep level
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Tuple
import numpy as np


# ── Sweep result for a single intensity level ─────────────────────────────────

@dataclass
class SweepPoint:
    intensity:    float          # the parameter value (e.g. brightness factor)
    pass_rate:    float          # fraction of samples that passed
    mean_delta:   float          # average |y - y'| across samples
    max_delta:    float          # worst-case delta
    violations:   int            # number of samples that failed
    total:        int            # total samples tested
    passed:       int            # total samples that passed


@dataclass
class SweepResult:
    transform_name:  str
    parameter_name:  str         # e.g. "brightness_factor", "noise_sigma"
    mr_name:         str
    model_name:      str
    sweep_points:    List[SweepPoint] = field(default_factory=list)

    @property
    def intensities(self) -> List[float]:
        return [p.intensity for p in self.sweep_points]

    @property
    def pass_rates(self) -> List[float]:
        return [p.pass_rate for p in self.sweep_points]

    @property
    def mean_deltas(self) -> List[float]:
        return [p.mean_delta for p in self.sweep_points]

    @property
    def critical_threshold(self) -> Optional[float]:
        """Intensity where pass_rate first drops below 80%."""
        for pt in self.sweep_points:
            if pt.pass_rate < 0.80:
                return pt.intensity
        return None   # never violated

    @property
    def collapse_threshold(self) -> Optional[float]:
        """Intensity where pass_rate drops below 50%."""
        for pt in self.sweep_points:
            if pt.pass_rate < 0.50:
                return pt.intensity
        return None


# ── Parametric transform factories ────────────────────────────────────────────

def make_brightness_transform(factor: float) -> Callable:
    """factor in [0, ∞). factor=1.0 → no change. factor>1 → brighter."""
    def transform(image: np.ndarray) -> np.ndarray:
        return np.clip(image * factor, 0.0, 1.0).astype(np.float32)
    return transform


def make_noise_transform(sigma: float) -> Callable:
    """sigma in [0, ∞). sigma=0 → no noise."""
    def transform(image: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        return np.clip(image + noise, 0.0, 1.0).astype(np.float32)
    return transform


def make_blur_transform(kernel_size: int) -> Callable:
    """kernel_size: odd int ≥1. 1 → no blur."""
    def transform(image: np.ndarray) -> np.ndarray:
        import cv2
        k = max(1, kernel_size | 1)   # force odd
        return cv2.GaussianBlur(image, (k, k), 0).astype(np.float32)
    return transform


def make_fog_transform(intensity: float) -> Callable:
    """intensity in [0, 1]. 0 → no fog, 1 → completely white."""
    def transform(image: np.ndarray) -> np.ndarray:
        fog = np.ones_like(image) * 0.85
        return np.clip(image * (1 - intensity) + fog * intensity, 0.0, 1.0).astype(np.float32)
    return transform


def make_rotation_transform(angle_deg: float) -> Callable:
    """angle_deg in [0, 360). 0 → no rotation."""
    def transform(image: np.ndarray) -> np.ndarray:
        import cv2
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    return transform


def make_translation_transform(dx_pixels: float) -> Callable:
    """dx_pixels: lateral shift in pixels. 0 → no shift."""
    def transform(image: np.ndarray) -> np.ndarray:
        import cv2
        h, w = image.shape[:2]
        M = np.float32([[1, 0, dx_pixels], [0, 1, 0]])
        return cv2.warpAffine(image, M, (w, h),
                               borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    return transform


def make_contrast_transform(factor: float) -> Callable:
    """factor in [0, ∞). factor=1.0 → no change. Center around 0.5."""
    def transform(image: np.ndarray) -> np.ndarray:
        adjusted = (image - 0.5) * factor + 0.5
        return np.clip(adjusted, 0.0, 1.0).astype(np.float32)
    return transform


def make_rain_transform(n_drops: int) -> Callable:
    """n_drops: number of rain streaks. 0 → no rain."""
    def transform(image: np.ndarray) -> np.ndarray:
        import cv2
        out = image.copy()
        h, w = image.shape[:2]
        for _ in range(n_drops):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            length = np.random.randint(8, 20)
            cv2.line(out, (x, y), (x + 2, min(y + length, h - 1)), (0.7, 0.7, 0.8), 1)
        out = out * max(0.75, 1.0 - n_drops / 1000)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    return transform


# ── Sweep configuration catalogue ─────────────────────────────────────────────

SWEEP_CATALOGUE: Dict[str, Dict] = {
    "brightness": {
        "label":          "Brightness Factor",
        "unit":           "×",
        "baseline":       1.0,
        "sweep_values":   [0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0],
        "factory":        make_brightness_transform,
        "category":       "A",
        "description":    "Simulates day/night lighting changes and tunnel transitions",
    },
    "noise": {
        "label":          "Gaussian Noise σ",
        "unit":           "",
        "baseline":       0.0,
        "sweep_values":   [0.00, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
        "factory":        make_noise_transform,
        "category":       "A",
        "description":    "Simulates camera sensor noise and interference",
    },
    "blur": {
        "label":          "Blur Kernel Size",
        "unit":           "px",
        "baseline":       1,
        "sweep_values":   [1, 3, 5, 7, 9, 11, 15, 21, 31, 41, 51],
        "factory":        make_blur_transform,
        "category":       "A",
        "description":    "Simulates lens defocus, motion blur, and foggy windshield",
    },
    "fog": {
        "label":          "Fog Intensity",
        "unit":           "",
        "baseline":       0.0,
        "sweep_values":   [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.00],
        "factory":        make_fog_transform,
        "category":       "A",
        "description":    "Simulates foggy / misty weather conditions",
    },
    "rotation": {
        "label":          "Rotation Angle",
        "unit":           "°",
        "baseline":       0.0,
        "sweep_values":   [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 90],
        "factory":        make_rotation_transform,
        "category":       "B",
        "description":    "Simulates camera mounting angle variation",
    },
    "translation": {
        "label":          "Lateral Shift",
        "unit":           "px",
        "baseline":       0,
        "sweep_values":   [0, 2, 4, 6, 8, 10, 15, 20, 30, 40, 60, 80],
        "factory":        make_translation_transform,
        "category":       "C",
        "description":    "Simulates vehicle lateral position within lane",
    },
    "contrast": {
        "label":          "Contrast Factor",
        "unit":           "×",
        "baseline":       1.0,
        "sweep_values":   [0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0],
        "factory":        make_contrast_transform,
        "category":       "A",
        "description":    "Simulates overcast vs bright sunny conditions",
    },
    "rain": {
        "label":          "Rain Drop Count",
        "unit":           " drops",
        "baseline":       0,
        "sweep_values":   [0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 800],
        "factory":        make_rain_transform,
        "category":       "A",
        "description":    "Simulates light drizzle to heavy rainfall",
    },
}


# ── Core sweep engine ──────────────────────────────────────────────────────────

def run_parametric_sweep(
    adapter,
    mr,
    images:           List[np.ndarray],
    transform_name:   str,
    custom_values:    List[float] = None,
) -> SweepResult:
    """
    Sweep a single transform parameter across its range, testing the MR at each level.

    Args:
        adapter:        loaded model adapter
        mr:             MetamorphicRelation instance
        images:         list of test images
        transform_name: key in SWEEP_CATALOGUE
        custom_values:  override the default sweep_values if provided

    Returns:
        SweepResult with a SweepPoint per intensity level
    """
    if transform_name not in SWEEP_CATALOGUE:
        raise KeyError(
            f"Unknown sweep transform '{transform_name}'. "
            f"Available: {list(SWEEP_CATALOGUE.keys())}"
        )

    config = SWEEP_CATALOGUE[transform_name]
    values = custom_values if custom_values is not None else config["sweep_values"]
    factory = config["factory"]

    result = SweepResult(
        transform_name=transform_name,
        parameter_name=config["label"],
        mr_name=mr.name,
        model_name=adapter.model_name,
    )

    for intensity in values:
        transform_fn = factory(intensity)
        deltas = []
        passed_count = 0

        for img in images:
            # Normalise to float32 [0, 1] before applying the sweep transform.
            # All parametric transform factories expect float32 input and clip
            # to [0, 1] — passing a raw uint8 image would collapse every pixel
            # to 0 after the clip, making img2 meaningless.
            img_f = (img / 255.0).astype(np.float32) if img.max() > 1.0 else img.astype(np.float32)
            img2 = transform_fn(img_f)
            y1 = adapter.predict(img)   # adapter.predict handles its own preprocessing
            y2 = adapter.predict(img2)
            mr_result = mr.check(img, img2, y1, y2)
            if mr_result.delta is not None:
                deltas.append(mr_result.delta)
            if mr_result.passed:
                passed_count += 1

        n = len(images)
        result.sweep_points.append(SweepPoint(
            intensity=intensity,
            pass_rate=passed_count / n,
            mean_delta=float(np.mean(deltas)) if deltas else 0.0,
            max_delta=float(np.max(deltas))  if deltas else 0.0,
            violations=n - passed_count,
            total=n,
            passed=passed_count,
        ))

    return result


def run_cross_model_sweep(
    adapters:         Dict,
    mr,
    images:           List[np.ndarray],
    transform_name:   str,
    custom_values:    List[float] = None,
) -> Dict[str, SweepResult]:
    """
    Run parametric sweep across multiple models simultaneously.
    Returns { model_name: SweepResult }
    """
    return {
        model_name: run_parametric_sweep(adapter, mr, images, transform_name, custom_values)
        for model_name, adapter in adapters.items()
    }