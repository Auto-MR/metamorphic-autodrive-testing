"""
Evaluator — runs a set of MRs against a model+transform pair
and aggregates the results.
"""

from typing import List, Tuple
from core.metamorphic_engine.mr_base import MetamorphicRelation, MRResult
from core.adapters.base_adapter import BaseAdapter
from core.metamorphic_engine.transformer import get_transform


class MREvaluator:
    """
    Runs multiple metamorphic relations on a single model
    and collects structured results.
    """

    def __init__(self, adapter: BaseAdapter, mrs: List[MetamorphicRelation]):
        self.adapter = adapter
        self.mrs = mrs

    def evaluate(self, inputs: list, transform_name: str, domain: str = "image") -> List[MRResult]:
        """
        For each input sample, apply transform and check all MRs.

        Args:
            inputs:         list of input samples
            transform_name: key from transform registry
            domain:         "image" | "sequence"

        Returns:
            Flat list of MRResult objects (len = inputs × MRs)
        """
        transform = get_transform(transform_name, domain)
        results: List[MRResult] = []

        for x in inputs:
            x2  = transform(x)
            y1  = self.adapter.predict(x)
            y2  = self.adapter.predict(x2)

            for mr in self.mrs:
                result = mr.check(x, x2, y1, y2)
                results.append(result)

        return results

    def summary(self, results: List[MRResult]) -> dict:
        """Compute pass-rate summary from a list of MRResults."""
        total  = len(results)
        passed = sum(r.passed for r in results)
        by_mr  = {}
        for r in results:
            by_mr.setdefault(r.mr_name, {"passed": 0, "total": 0})
            by_mr[r.mr_name]["total"]  += 1
            by_mr[r.mr_name]["passed"] += int(r.passed)

        return {
            "total":      total,
            "passed":     passed,
            "failed":     total - passed,
            "pass_rate":  round(passed / total, 4) if total else 0.0,
            "by_mr":      by_mr,
        }
