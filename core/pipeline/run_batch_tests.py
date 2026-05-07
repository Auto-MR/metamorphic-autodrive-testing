"""
run_batch_tests.py — run all configured MRs over a batch of inputs.
"""

from typing import List, Dict
import numpy as np

from core.adapters.base_adapter import BaseAdapter
from core.metamorphic_engine.mr_base import MetamorphicRelation, MRResult
from core.metamorphic_engine.evaluator import MREvaluator


def run_batch(
    adapter:   BaseAdapter,
    mrs:       List[MetamorphicRelation],
    inputs:    List[np.ndarray],
    transforms: List[str],
    domain:    str = "image",
) -> Dict:
    """
    Run every (transform × MR) combination over all inputs.

    Returns:
        {
          "results":   flat list of MRResult,
          "summary":   aggregated pass-rate dict,
          "per_transform": { transform_name: summary_dict }
        }
    """
    evaluator = MREvaluator(adapter, mrs)

    all_results: List[MRResult] = []
    per_transform: Dict[str, dict] = {}

    for tf_name in transforms:
        results = evaluator.evaluate(inputs, tf_name, domain)
        all_results.extend(results)
        per_transform[tf_name] = evaluator.summary(results)
        print(f"  [{tf_name}] pass_rate = {per_transform[tf_name]['pass_rate']:.2%}")

    return {
        "results":       all_results,
        "summary":       evaluator.summary(all_results),
        "per_transform": per_transform,
    }


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from core.adapters.steering_adapter import SteeringAdapter
    from metamorphic_relations.steering_mrs import FlipSymmetrySteeringMR, SmoothnessSteeringMR

    adapter = SteeringAdapter(); adapter.load()
    mrs     = [FlipSymmetrySteeringMR(), SmoothnessSteeringMR()]
    inputs  = [np.random.rand(64, 64, 3).astype(np.float32) for _ in range(10)]

    report = run_batch(adapter, mrs, inputs, transforms=["flip", "noise"])
    print(report["summary"])
