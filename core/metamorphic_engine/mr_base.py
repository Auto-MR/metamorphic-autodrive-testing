"""
Metamorphic Relation Base Class.
Defines the contract every MR must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MRResult:
    mr_name:    str
    passed:     bool
    expected:   Any = None
    actual:     Any = None
    delta:      float = None
    message:    str = ""

    def __repr__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return (f"{status} | {self.mr_name} | "
                f"delta={self.delta:.4f}" if self.delta is not None else
                f"{status} | {self.mr_name} | {self.message}")


class MetamorphicRelation(ABC):
    """
    Abstract base for all metamorphic relations (MRs).

    Subclasses define:
      - name        : human-readable identifier
      - tolerance   : acceptable deviation between y and y_transformed
      - check(...)  : the actual relation logic
    """

    def __init__(self, name: str, tolerance: float = 0.1):
        self.name = name
        self.tolerance = tolerance

    @abstractmethod
    def check(
        self,
        input_x,
        transformed_x,
        output_y,
        output_y_transformed,
    ) -> MRResult:
        """
        Verify the metamorphic relation holds.

        Args:
            input_x             : original input
            transformed_x       : transformed input
            output_y            : model output on original input
            output_y_transformed: model output on transformed input

        Returns:
            MRResult with pass/fail and diagnostics
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', tol={self.tolerance})"
