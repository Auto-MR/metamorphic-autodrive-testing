"""
Cross-Model Metamorphic Relations.
Tests consistency constraints that span two different models.
"""

import numpy as np
from core.metamorphic_engine.mr_base import MetamorphicRelation, MRResult


class DepthSteeringConsistencyMR(MetamorphicRelation):
    """
    MR-CROSS-1: Depth–Steering Consistency
    When an image is flipped:
      - CNN depth map should also flip (left-right)
      - Steering angle should negate

    This MR receives:
      y  = (depth_map,  steering_angle)   for original image
      y2 = (depth_map2, steering_angle2)  for flipped image
    """

    def __init__(self, depth_tol: float = 0.05, steer_tol: float = 0.10):
        super().__init__("depth_steering_consistency")
        self.depth_tol = depth_tol
        self.steer_tol = steer_tol

    def check(self, x, x2, y, y2) -> MRResult:
        depth1, steer1 = y
        depth2, steer2 = y2

        # Depth: flipping x should flip depth map
        depth_delta = float(np.mean(np.abs(depth1 - depth2[:, ::-1])))
        steer_delta = float(abs(steer1 + steer2))

        depth_ok = depth_delta <= self.depth_tol
        steer_ok = steer_delta <= self.steer_tol
        passed   = depth_ok and steer_ok

        return MRResult(
            mr_name=self.name,
            passed=passed,
            delta=max(depth_delta, steer_delta),
            message=(
                f"depth_diff={depth_delta:.4f}({'OK' if depth_ok else 'FAIL'}), "
                f"steer_antisym={steer_delta:.4f}({'OK' if steer_ok else 'FAIL'})"
            )
        )
