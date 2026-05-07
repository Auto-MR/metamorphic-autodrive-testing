"""
Metamorphic Relations for Steering Angle Regression model.
Covers all 10 MRs described.
"""

import numpy as np
from core.metamorphic_engine.mr_base import MetamorphicRelation, MRResult


class FlipSymmetrySteeringMR(MetamorphicRelation):
    """
    MR-1: Horizontal Flip Symmetry (CRITICAL)
    Horizontally flipping the image should negate the steering angle.
    """

    def __init__(self, tolerance: float = 0.10):
        super().__init__("flip_symmetry_steering", tolerance)

    def check(self, x, x2, y, y2) -> MRResult:
        delta = float(abs(y + y2))
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name,
            passed=passed,
            expected=f"|y + y_flip| <= {self.tolerance}",
            actual=f"|{y:.4f} + {y2:.4f}| = {delta:.4f}",
            delta=delta,
            message="Flip antisymmetry holds" if passed else "Model shows dangerous directional bias"
        )


class SmallLaneShiftInvarianceMR(MetamorphicRelation):
    """
    MR-2: Small Lane Shift Invariance
    Small lateral shifts should cause only small steering changes.
    """

    def __init__(self, tolerance: float = 0.15):
        super().__init__("small_lane_shift_invariance", tolerance)

    def check(self, x, x2, y, y2) -> MRResult:
        delta = float(abs(y - y2))
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name,
            passed=passed,
            expected=f"|Δθ| <= {self.tolerance}",
            actual=f"|Δθ| = {delta:.4f}",
            delta=delta,
            message="Small shift invariance holds" if passed else "Sudden sharp turn on minor shift → instability"
        )


class BrightnessSteeringMR(MetamorphicRelation):
    """
    MR-3: Brightness / Lighting Invariance
    """

    def __init__(self, tolerance: float = 0.10):
        super().__init__("brightness_steering", tolerance)

    def check(self, x, x2, y, y2) -> MRResult:
        delta = float(abs(y - y2))
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name,
            passed=passed,
            expected=f"|Δθ| <= {self.tolerance}",
            actual=f"diff = {delta:.4f}",
            delta=delta,
            message="Lighting invariance holds" if passed else "Steering sensitive to lighting (tunnels/night issues)"
        )


class WeatherRobustnessMR(MetamorphicRelation):
    """
    MR-4: Weather Robustness (fog, rain, blur)
    Expect slightly smoother/cautious steering, not erratic behavior.
    """

    def __init__(self, tolerance: float = 0.20):
        super().__init__("weather_robustness", tolerance)

    def check(self, x, x2, y, y2) -> MRResult:
        delta = float(abs(y - y2))
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name,
            passed=passed,
            expected=f"|Δθ| <= {self.tolerance} (mild degradation)",
            actual=f"diff = {delta:.4f}",
            delta=delta,
            message="Weather robustness OK" if passed else "Erratic/oscillating behavior under weather"
        )


class TemporalSmoothnessMR(MetamorphicRelation):
    """
    MR-5: Temporal Smoothness (VERY IMPORTANT)
    Steering should not jump drastically between consecutive frames.
    """

    def __init__(self, tolerance: float = 0.08):
        super().__init__("temporal_smoothness", tolerance)

    def check(self, x, x2, y, y2) -> MRResult:
        delta = float(abs(y - y2))
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name,
            passed=passed,
            expected=f"|θ_t - θ_{{t-1}}| <= {self.tolerance}",
            actual=f"|Δθ| = {delta:.4f}",
            delta=delta,
            message="Temporal smoothness holds" if passed else "Sudden steering spikes → unsafe"
        )


class ConstantCurvatureConsistencyMR(MetamorphicRelation):
    """
    MR-6: Constant Curvature Consistency
    On constant curvature (e.g. circular road), steering should be stable.
    """

    def __init__(self, tolerance: float = 0.05):
        super().__init__("constant_curvature_consistency", tolerance)

    def check(self, x, x2, y, y2) -> MRResult:
        delta = float(abs(y - y2))
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name,
            passed=passed,
            expected=f"|Δθ| <= {self.tolerance} on constant curve",
            actual=f"|Δθ| = {delta:.4f}",
            delta=delta,
            message="Curvature consistency OK" if passed else "Oscillating steering on constant bend"
        )


class OcclusionRobustnessMR(MetamorphicRelation):
    """
    MR-7: Occlusion Robustness
    Partial occlusion should not drastically change steering.
    """

    def __init__(self, tolerance: float = 0.18):
        super().__init__("occlusion_robustness", tolerance)

    def check(self, x, x2, y, y2) -> MRResult:
        delta = float(abs(y - y2))
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name,
            passed=passed,
            expected=f"|Δθ| <= {self.tolerance}",
            actual=f"diff = {delta:.4f}",
            delta=delta,
            message="Occlusion robustness OK" if passed else "Over-reliance on specific visible pixels"
        )


class LaneCenterShiftConsistencyMR(MetamorphicRelation):
    """
    MR-8: Lane Center Shift Consistency
    Shifting vehicle position inside lane should produce smooth correction.
    """

    def __init__(self, tolerance: float = 0.25):
        super().__init__("lane_center_shift_consistency", tolerance)

    def check(self, x, x2, y, y2) -> MRResult:
        delta = float(abs(y - y2))
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name,
            passed=passed,
            expected=f"Smooth correction |Δθ| <= {self.tolerance}",
            actual=f"|Δθ| = {delta:.4f}",
            delta=delta,
            message="Lane centering behavior correct" if passed else "No correction or dangerous overcorrection"
        )


class PerspectiveScalingMR(MetamorphicRelation):
    """
    MR-9: Perspective Scaling
    Zoom/crop should not change the steering decision significantly.
    """

    def __init__(self, tolerance: float = 0.12):
        super().__init__("perspective_scaling", tolerance)

    def check(self, x, x2, y, y2) -> MRResult:
        delta = float(abs(y - y2))
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name,
            passed=passed,
            expected=f"|Δθ| <= {self.tolerance}",
            actual=f"diff = {delta:.4f}",
            delta=delta,
            message="Scale invariance holds" if passed else "Scale bias detected"
        )


class StraightRoadStabilityMR(MetamorphicRelation):
    """
    MR-10: Straight Road Stability
    Perfectly aligned straight road → steering near zero.
    """

    def __init__(self, tolerance: float = 0.05):
        super().__init__("straight_road_stability", tolerance)

    def check(self, x, x2, y, y2) -> MRResult:
        # For straight road we usually test y ≈ 0 and y2 ≈ 0
        max_abs = max(abs(y), abs(y2))
        passed = max_abs <= self.tolerance
        return MRResult(
            mr_name=self.name,
            passed=passed,
            expected=f"|θ| <= {self.tolerance} on straight road",
            actual=f"max(|{y:.4f}|, |{y2:.4f}|) = {max_abs:.4f}",
            delta=max_abs,
            message="Straight road stable" if passed else "Hallucinated drift/oscillation on straight road"
        )