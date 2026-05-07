"""
cross_model_tests.py — run cross-model metamorphic relations
that require outputs from multiple models on the same input.
"""

from typing import List
import numpy as np

from core.adapters.cnn_adapter     import CNNAdapter
from core.adapters.steering_adapter import SteeringAdapter
from metamorphic_relations.cross_model_mrs import DepthSteeringConsistencyMR
from core.metamorphic_engine.mr_base import MRResult
from transformations.image_transforms import flip_horizontal


def run_cross_model_flip_test(
    cnn_adapter:      CNNAdapter,
    steering_adapter: SteeringAdapter,
    inputs:           List[np.ndarray],
    tolerance_depth:  float = 0.05,
    tolerance_steer:  float = 0.10,
) -> List[MRResult]:
    """
    For each input image, check that:
      - CNN depth map flips when image is flipped
      - Steering angle negates when image is flipped
    """
    mr = DepthSteeringConsistencyMR(tolerance_depth, tolerance_steer)
    results = []

    for img in inputs:
        img_flip = flip_horizontal(img)

        y  = (cnn_adapter.predict(img),      steering_adapter.predict(img))
        y2 = (cnn_adapter.predict(img_flip), steering_adapter.predict(img_flip))

        result = mr.check(img, img_flip, y, y2)
        results.append(result)

    passed = sum(r.passed for r in results)
    print(f"[cross_model_flip] {passed}/{len(results)} passed")
    return results


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cnn = CNNAdapter();      cnn.load()
    steer = SteeringAdapter(); steer.load()
    imgs = [np.random.rand(64, 64, 3).astype(np.float32) for _ in range(5)]
    run_cross_model_flip_test(cnn, steer, imgs)
