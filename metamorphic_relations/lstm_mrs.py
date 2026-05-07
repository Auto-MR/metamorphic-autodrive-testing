"""
Metamorphic Relations for the LSTM Trajectory Prediction model.
"""

import numpy as np
from core.metamorphic_engine.mr_base import MetamorphicRelation, MRResult


class TranslationConsistencyMR(MetamorphicRelation):
    """
    MR-LSTM-1: Translation Consistency
    Shifting the entire input trajectory by (dx, dy) should shift
    the predicted trajectory by the same amount.

    check: mean_abs_diff(y2 - shift, y) <= tolerance
    """

    def __init__(self, tolerance: float = 0.5, shift: tuple = (1.0, 0.0)):
        super().__init__("translation_consistency", tolerance)
        self.shift = np.array(shift, dtype=np.float32)

    def check(self, x, x2, y, y2) -> MRResult:
        y2_corrected = y2 - self.shift
        delta = float(np.mean(np.abs(y - y2_corrected)))
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name,
            passed=passed,
            expected=f"translation_error <= {self.tolerance}",
            actual=f"translation_error = {delta:.4f}",
            delta=delta,
        )


class TimeShiftStabilityMR(MetamorphicRelation):
    """
    MR-LSTM-2: Time Shift Stability
    Reversing the input sequence should not produce wildly different
    predicted *lengths* (trajectory scale stability).

    check: abs(norm(y) - norm(y2)) / norm(y) <= tolerance
    """

    def __init__(self, tolerance: float = 0.5):
        super().__init__("time_shift_stability", tolerance)

    def check(self, x, x2, y, y2) -> MRResult:
        norm_y  = float(np.linalg.norm(y))
        norm_y2 = float(np.linalg.norm(y2))
        if norm_y < 1e-8:
            delta = 0.0
        else:
            delta = abs(norm_y - norm_y2) / norm_y
        passed = delta <= self.tolerance
        return MRResult(
            mr_name=self.name,
            passed=passed,
            expected=f"scale_diff <= {self.tolerance}",
            actual=f"scale_diff = {delta:.4f}",
            delta=delta,
        )
