"""
MR Library — convenience factory for all metamorphic relations, organized by category.
"""

from metamorphic_relations.steering_mrs import (
    # Category A — Invariance
    BrightnessSteeringMR,
    ContrastSteeringMR,
    NoiseRobustnessSteeringMR,
    WeatherRobustnessMR,
    OcclusionRobustnessMR,
    # Category B — Symmetry
    FlipSymmetrySteeringMR,
    StraightRoadStabilityMR,
    PerspectiveScalingMR,
    # Category C — Temporal
    TemporalSmoothnessMR,
    ConstantCurvatureConsistencyMR,
    SmallLaneShiftInvarianceMR,
    # Category D — Composition
    ComposedBrightnessNoiseMR,
    ComposedWeatherBlurMR,
    LaneCenterShiftConsistencyMR,
)
from metamorphic_relations.cnn_mrs import BrightnessInvarianceMR, FlipSymmetryCNNMR, NoiseRobustnessMR
from metamorphic_relations.lstm_mrs import TranslationConsistencyMR, TimeShiftStabilityMR
from metamorphic_relations.cross_model_mrs import DepthSteeringConsistencyMR


# All steering MRs organized by category
STEERING_MRS_BY_CATEGORY = {
    "A": [
        BrightnessSteeringMR,
        ContrastSteeringMR,
        NoiseRobustnessSteeringMR,
        WeatherRobustnessMR,
        OcclusionRobustnessMR,
    ],
    "B": [
        FlipSymmetrySteeringMR,
        StraightRoadStabilityMR,
        PerspectiveScalingMR,
    ],
    "C": [
        TemporalSmoothnessMR,
        ConstantCurvatureConsistencyMR,
        SmallLaneShiftInvarianceMR,
    ],
    "D": [
        ComposedBrightnessNoiseMR,
        ComposedWeatherBlurMR,
        LaneCenterShiftConsistencyMR,
    ],
}

# All steering MRs flat
ALL_STEERING_MRS = [cls for mrs in STEERING_MRS_BY_CATEGORY.values() for cls in mrs]

# Full registry
ALL_MRS = {
    # Steering — Cat A
    "brightness_steering":          BrightnessSteeringMR,
    "contrast_steering":            ContrastSteeringMR,
    "noise_robustness_steering":    NoiseRobustnessSteeringMR,
    "weather_robustness":           WeatherRobustnessMR,
    "occlusion_robustness":         OcclusionRobustnessMR,
    # Steering — Cat B
    "flip_symmetry_steering":       FlipSymmetrySteeringMR,
    "straight_road_stability":      StraightRoadStabilityMR,
    "perspective_scaling":          PerspectiveScalingMR,
    # Steering — Cat C
    "temporal_smoothness":          TemporalSmoothnessMR,
    "constant_curvature_consistency": ConstantCurvatureConsistencyMR,
    "small_lane_shift_invariance":  SmallLaneShiftInvarianceMR,
    # Steering — Cat D
    "composed_brightness_noise":    ComposedBrightnessNoiseMR,
    "composed_weather_blur":        ComposedWeatherBlurMR,
    "lane_center_shift_consistency": LaneCenterShiftConsistencyMR,
    # CNN MRs
    "brightness_invariance_cnn":    BrightnessInvarianceMR,
    "flip_symmetry_cnn":            FlipSymmetryCNNMR,
    "noise_robustness_cnn":         NoiseRobustnessMR,
    # LSTM MRs
    "translation_consistency":      TranslationConsistencyMR,
    "time_shift_stability":         TimeShiftStabilityMR,
    # Cross-model
    "depth_steering_consistency":   DepthSteeringConsistencyMR,
}


def get_mr(name: str, **kwargs):
    """Instantiate an MR by name with optional constructor kwargs."""
    if name not in ALL_MRS:
        raise KeyError(f"Unknown MR '{name}'. Available: {list(ALL_MRS.keys())}")
    return ALL_MRS[name](**kwargs)


def get_steering_mrs(tolerance: float = None) -> list:
    """Return all steering MR instances with optional shared tolerance."""
    kwargs = {"tolerance": tolerance} if tolerance else {}
    return [cls(**kwargs) for cls in ALL_STEERING_MRS]


def get_mrs_by_category(category: str, tolerance: float = None) -> list:
    """Return MR instances for a specific category (A, B, C, D)."""
    classes = STEERING_MRS_BY_CATEGORY.get(category.upper(), [])
    kwargs = {"tolerance": tolerance} if tolerance else {}
    return [cls(**kwargs) for cls in classes]
