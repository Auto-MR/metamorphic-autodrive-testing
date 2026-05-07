"""
MR Library — convenience factory for common metamorphic relations.
Imports from metamorphic_relations/ and re-exports them here
so pipelines only need to import from one place.
"""

from metamorphic_relations.cnn_mrs      import BrightnessInvarianceMR, FlipSymmetryCNNMR, NoiseRobustnessMR
from metamorphic_relations.lstm_mrs     import TranslationConsistencyMR, TimeShiftStabilityMR
from metamorphic_relations.steering_mrs import FlipSymmetrySteeringMR, SmoothnessSteeringMR
from metamorphic_relations.cross_model_mrs import DepthSteeringConsistencyMR


ALL_MRS = {
    # CNN MRs
    "brightness_invariance": BrightnessInvarianceMR,
    "flip_symmetry_cnn":     FlipSymmetryCNNMR,
    "noise_robustness":      NoiseRobustnessMR,
    # LSTM MRs
    "translation_consistency": TranslationConsistencyMR,
    "time_shift_stability":    TimeShiftStabilityMR,
    # Steering MRs
    "flip_symmetry_steering":  FlipSymmetrySteeringMR,
    "smoothness_steering":     SmoothnessSteeringMR,
    # Cross-model MRs
    "depth_steering_consistency": DepthSteeringConsistencyMR,
}


def get_mr(name: str, **kwargs):
    """Instantiate an MR by name with optional constructor kwargs."""
    if name not in ALL_MRS:
        raise KeyError(f"Unknown MR '{name}'. Available: {list(ALL_MRS.keys())}")
    return ALL_MRS[name](**kwargs)
