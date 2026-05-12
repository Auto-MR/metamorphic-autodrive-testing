"""
Metamorphic Relations for Steering Angle Regression — 4 Categories.

Category A — Invariance MRs  (environmental changes → small output change)
Category B — Symmetry MRs    (structural input change → predictable output change)
Category C — Temporal MRs    (sequential frames → consistent output)
Category D — Composition MRs (combined transforms → stable output)
"""

import numpy as np
from core.metamorphic_engine.mr_base import MetamorphicRelation, MRResult


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY A — INVARIANCE MRs
# |f(x) - f(T(x))| < ε
# ─────────────────────────────────────────────────────────────────────────────

class BrightnessSteeringMR(MetamorphicRelation):
    """
    MR-A1: Brightness Invariance
    Lighting changes (day/night, tunnels) should not largely alter steering.
    Formula: |f(x) - f(T_brightness(x))| < ε
    """
    category = "A"

    def __init__(self, tolerance: float = None):
        super().__init__("brightness_steering", tolerance or 0.10)

    def check(self, x, x2, y, y2) -> MRResult:
        delta = float(abs(y - y2))
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name, passed=passed,
            expected=f"|Δθ| ≤ {self.tolerance:.3f}",
            actual=f"|Δθ| = {delta:.4f}", delta=delta,
            message="[A] Brightness invariance holds" if passed else
                    "[A] Steering sensitive to brightness — tunnel/night risk"
        )


class ContrastSteeringMR(MetamorphicRelation):
    """
    MR-A2: Contrast Invariance
    Contrast adjustments (overcast vs sunny) should not largely alter steering.
    Formula: |f(x) - f(T_contrast(x))| < ε
    """
    category = "A"

    def __init__(self, tolerance: float = None):
        super().__init__("contrast_steering", tolerance or 0.10)

    def check(self, x, x2, y, y2) -> MRResult:
        delta = float(abs(y - y2))
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name, passed=passed,
            expected=f"|Δθ| ≤ {self.tolerance:.3f}",
            actual=f"|Δθ| = {delta:.4f}", delta=delta,
            message="[A] Contrast invariance holds" if passed else
                    "[A] Steering sensitive to contrast changes"
        )


class NoiseRobustnessSteeringMR(MetamorphicRelation):
    """
    MR-A3: Gaussian Noise Robustness
    Small sensor noise should not destabilize steering.
    Formula: |f(x) - f(T_noise(x))| < ε
    """
    category = "A"

    def __init__(self, tolerance: float = None):
        super().__init__("noise_robustness_steering", tolerance or 0.12)

    def check(self, x, x2, y, y2) -> MRResult:
        delta = float(abs(y - y2))
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name, passed=passed,
            expected=f"|Δθ| ≤ {self.tolerance:.3f}",
            actual=f"|Δθ| = {delta:.4f}", delta=delta,
            message="[A] Noise robustness holds" if passed else
                    "[A] Steering unstable under sensor noise"
        )


class WeatherRobustnessMR(MetamorphicRelation):
    """
    MR-A4: Weather Robustness (fog, rain, snow)
    Mild degradation is expected; erratic behavior is not.
    Formula: |f(x) - f(T_weather(x))| < ε
    """
    category = "A"

    def __init__(self, tolerance: float = None):
        super().__init__("weather_robustness", tolerance or 0.20)

    def check(self, x, x2, y, y2) -> MRResult:
        delta = float(abs(y - y2))
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name, passed=passed,
            expected=f"|Δθ| ≤ {self.tolerance:.3f} (mild degradation OK)",
            actual=f"|Δθ| = {delta:.4f}", delta=delta,
            message="[A] Weather robustness OK" if passed else
                    "[A] Erratic steering under weather — safety risk"
        )


class OcclusionRobustnessMR(MetamorphicRelation):
    """
    MR-A5: Occlusion Robustness
    Partial occlusion should not catastrophically alter steering.
    Formula: |f(x) - f(T_occlude(x))| < ε
    """
    category = "A"

    def __init__(self, tolerance: float = None):
        super().__init__("occlusion_robustness", tolerance or 0.18)

    def check(self, x, x2, y, y2) -> MRResult:
        delta = float(abs(y - y2))
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name, passed=passed,
            expected=f"|Δθ| ≤ {self.tolerance:.3f}",
            actual=f"|Δθ| = {delta:.4f}", delta=delta,
            message="[A] Occlusion robustness OK" if passed else
                    "[A] Over-reliance on specific pixels — fragile"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY B — SYMMETRY MRs
# f(T(x)) ≈ -f(x)   (anti-symmetry)  or predictable sign relationship
# ─────────────────────────────────────────────────────────────────────────────

class FlipSymmetrySteeringMR(MetamorphicRelation):
    """
    MR-B1: Horizontal Flip Symmetry (CRITICAL — most important MR)
    Flipping the image should negate the steering angle.
    Formula: |f(flip(x)) + f(x)| < ε
    """
    category = "B"

    def __init__(self, tolerance: float = None):
        super().__init__("flip_symmetry_steering", tolerance or 0.10)

    def check(self, x, x2, y, y2) -> MRResult:
        delta = float(abs(y + y2))
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name, passed=passed,
            expected=f"|f(x) + f(flip(x))| ≤ {self.tolerance:.3f}",
            actual=f"|{y:.4f} + {y2:.4f}| = {delta:.4f}", delta=delta,
            message="[B] Flip anti-symmetry holds — model is unbiased" if passed else
                    "[B] DANGEROUS directional bias detected"
        )


class StraightRoadStabilityMR(MetamorphicRelation):
    """
    MR-B2: Straight Road Stability
    A perfectly straight road should produce near-zero steering.
    Formula: |f(x)| < ε  (no transformation needed — checks both y and y2)
    """
    category = "B"

    def __init__(self, tolerance: float = None):
        super().__init__("straight_road_stability", tolerance or 0.05)

    def check(self, x, x2, y, y2) -> MRResult:
        max_abs = max(abs(y), abs(y2))
        passed = max_abs <= self.tolerance
        return MRResult(
            mr_name=self.name, passed=passed,
            expected=f"max(|θ|, |θ'|) ≤ {self.tolerance:.3f} on straight road",
            actual=f"max({abs(y):.4f}, {abs(y2):.4f}) = {max_abs:.4f}", delta=max_abs,
            message="[B] Straight road stable" if passed else
                    "[B] Hallucinated drift on straight road"
        )


class PerspectiveScalingMR(MetamorphicRelation):
    """
    MR-B3: Perspective Scaling
    Zoom/crop should not change the steering decision significantly.
    Formula: |f(x) - f(scale(x))| < ε
    """
    category = "B"

    def __init__(self, tolerance: float = None):
        super().__init__("perspective_scaling", tolerance or 0.12)

    def check(self, x, x2, y, y2) -> MRResult:
        delta = float(abs(y - y2))
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name, passed=passed,
            expected=f"|Δθ| ≤ {self.tolerance:.3f}",
            actual=f"|Δθ| = {delta:.4f}", delta=delta,
            message="[B] Perspective scaling invariance holds" if passed else
                    "[B] Scale-dependent bias detected"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY C — TEMPORAL CONSISTENCY MRs
# |f(x_t) - f(x_{t+1})| < δ
# ─────────────────────────────────────────────────────────────────────────────

class TemporalSmoothnessMR(MetamorphicRelation):
    """
    MR-C1: Temporal Smoothness (VERY IMPORTANT)
    Steering should not jump drastically between consecutive frames.
    Formula: |f(x_t) - f(x_{t+1})| < δ
    """
    category = "C"

    def __init__(self, tolerance: float = None):
        super().__init__("temporal_smoothness", tolerance or 0.08)

    def check(self, x, x2, y, y2) -> MRResult:
        delta = float(abs(y - y2))
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name, passed=passed,
            expected=f"|θ_t - θ_{{t+1}}| ≤ {self.tolerance:.3f}",
            actual=f"|Δθ| = {delta:.4f}", delta=delta,
            message="[C] Temporal smoothness holds" if passed else
                    "[C] Sudden steering spike → unsafe"
        )


class ConstantCurvatureConsistencyMR(MetamorphicRelation):
    """
    MR-C2: Constant Curvature Consistency
    On constant curvature road, steering should remain stable across frames.
    Formula: |f(x_t) - f(x_{t+1})| < δ  on circular road
    """
    category = "C"

    def __init__(self, tolerance: float = None):
        super().__init__("constant_curvature_consistency", tolerance or 0.05)

    def check(self, x, x2, y, y2) -> MRResult:
        delta = float(abs(y - y2))
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name, passed=passed,
            expected=f"|Δθ| ≤ {self.tolerance:.3f} on constant bend",
            actual=f"|Δθ| = {delta:.4f}", delta=delta,
            message="[C] Curvature consistency OK" if passed else
                    "[C] Oscillating steering on constant bend"
        )


class SmallLaneShiftInvarianceMR(MetamorphicRelation):
    """
    MR-C3: Small Lane Shift Invariance
    Small lateral translations should cause only proportional small changes.
    Formula: |f(x) - f(translate(x))| < δ
    """
    category = "C"

    def __init__(self, tolerance: float = None):
        super().__init__("small_lane_shift_invariance", tolerance or 0.15)

    def check(self, x, x2, y, y2) -> MRResult:
        delta = float(abs(y - y2))
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name, passed=passed,
            expected=f"|Δθ| ≤ {self.tolerance:.3f}",
            actual=f"|Δθ| = {delta:.4f}", delta=delta,
            message="[C] Lane shift invariance holds" if passed else
                    "[C] Sudden sharp turn on minor shift → instability"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY D — COMPOSITION MRs
# Apply multiple transformations together → output remains stable
# ─────────────────────────────────────────────────────────────────────────────

class ComposedBrightnessNoiseMR(MetamorphicRelation):
    """
    MR-D1: Composed Brightness + Noise
    Combined transforms should not produce larger deviation than sum of parts.
    Formula: |f(x) - f(T_bright+noise(x))| < ε
    """
    category = "D"

    def __init__(self, tolerance: float = None):
        super().__init__("composed_brightness_noise", tolerance or 0.15)

    def check(self, x, x2, y, y2) -> MRResult:
        delta = float(abs(y - y2))
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name, passed=passed,
            expected=f"|Δθ| ≤ {self.tolerance:.3f} (brightness+noise)",
            actual=f"|Δθ| = {delta:.4f}", delta=delta,
            message="[D] Composed transform stability holds" if passed else
                    "[D] Composed transforms cause instability"
        )


class ComposedWeatherBlurMR(MetamorphicRelation):
    """
    MR-D2: Composed Weather + Blur
    Weather effect combined with blur should remain within degradation bounds.
    Formula: |f(x) - f(T_weather+blur(x))| < ε
    """
    category = "D"

    def __init__(self, tolerance: float = None):
        super().__init__("composed_weather_blur", tolerance or 0.22)

    def check(self, x, x2, y, y2) -> MRResult:
        delta = float(abs(y - y2))
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name, passed=passed,
            expected=f"|Δθ| ≤ {self.tolerance:.3f} (weather+blur)",
            actual=f"|Δθ| = {delta:.4f}", delta=delta,
            message="[D] Weather+blur stability holds" if passed else
                    "[D] Model fails combined weather+blur scenario"
        )


class LaneCenterShiftConsistencyMR(MetamorphicRelation):
    """
    MR-D3: Lane Center Shift Consistency (composed shift + brightness)
    Shifting vehicle position combined with lighting changes should allow correction.
    Formula: |f(x) - f(T_shift+bright(x))| < ε
    """
    category = "D"

    def __init__(self, tolerance: float = None):
        super().__init__("lane_center_shift_consistency", tolerance or 0.25)

    def check(self, x, x2, y, y2) -> MRResult:
        delta = float(abs(y - y2))
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name, passed=passed,
            expected=f"Smooth correction |Δθ| ≤ {self.tolerance:.3f}",
            actual=f"|Δθ| = {delta:.4f}", delta=delta,
            message="[D] Lane centering behavior correct" if passed else
                    "[D] No correction or dangerous overcorrection"
        )


# Alias kept for backward compatibility with older imports
SmoothnessSteeringMR = TemporalSmoothnessMR
