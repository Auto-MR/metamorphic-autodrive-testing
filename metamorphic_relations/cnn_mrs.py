"""
Metamorphic Relations for the CNN Depth Estimation model.
"""

import numpy as np
from core.metamorphic_engine.mr_base import MetamorphicRelation, MRResult


class BrightnessInvarianceMR(MetamorphicRelation):
    """
    MR-CNN-1: Brightness Invariance
    Changing image brightness should not significantly alter the
    relative depth ordering — measured by rank correlation.

    check: corr(flatten(y), flatten(y2)) >= 1 - tolerance
    """

    def __init__(self, tolerance: float = 0.10):
        super().__init__("brightness_invariance", tolerance)

    def check(self, x, x2, y, y2) -> MRResult:
        from scipy.stats import spearmanr
        r, _ = spearmanr(y.flatten(), y2.flatten())
        delta = 1.0 - float(r)
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name,
            passed=passed,
            expected=f"rank_corr >= {1 - self.tolerance:.2f}",
            actual=f"rank_corr = {r:.4f}",
            delta=delta,
            message="Depth rank ordering preserved under brightness change" if passed
                    else "Depth ordering changed unexpectedly with brightness"
        )


class FlipSymmetryCNNMR(MetamorphicRelation):
    """
    MR-CNN-2: Flip Symmetry
    Horizontally flipping the input should produce a horizontally
    flipped depth map (depth values themselves are invariant to flip).

    check: mean_abs_diff(y, flip(y2)) <= tolerance
    """

    def __init__(self, tolerance: float = 0.05):
        super().__init__("flip_symmetry_cnn", tolerance)

    def check(self, x, x2, y, y2) -> MRResult:
        y2_flipped = y2[:, ::-1]
        delta = float(np.mean(np.abs(y - y2_flipped)))
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name,
            passed=passed,
            expected=f"mean_diff <= {self.tolerance}",
            actual=f"mean_diff = {delta:.4f}",
            delta=delta,
        )


class NoiseRobustnessMR(MetamorphicRelation):
    """
    MR-CNN-3: Noise Robustness
    Small Gaussian noise added to the input should not drastically
    change the depth map output.

    check: mean_abs_diff(y, y2) <= tolerance
    """

    def __init__(self, tolerance: float = 0.08):
        super().__init__("noise_robustness", tolerance)

    def check(self, x, x2, y, y2) -> MRResult:
        delta = float(np.mean(np.abs(y - y2)))
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name,
            passed=passed,
            expected=f"mean_diff <= {self.tolerance}",
            actual=f"mean_diff = {delta:.4f}",
            delta=delta,
        )
