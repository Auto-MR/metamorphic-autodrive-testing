"""
Robustness Scorer — computes per-model and global robustness scores.
Score = passed_MRs / total_MRs
"""

from typing import List, Dict
from core.metamorphic_engine.mr_base import MRResult


def compute_score(results: List[MRResult]) -> float:
    if not results:
        return 0.0
    return round(sum(r.passed for r in results) / len(results), 4)


def score_by_model(results_map: Dict[str, List[MRResult]]) -> Dict[str, float]:
    scores = {name: compute_score(res) for name, res in results_map.items()}
    all_results = [r for res in results_map.values() for r in res]
    scores["global"] = compute_score(all_results)
    return scores


def score_by_mr(results: List[MRResult]) -> Dict[str, float]:
    by_mr: Dict[str, List[MRResult]] = {}
    for r in results:
        by_mr.setdefault(r.mr_name, []).append(r)
    return {name: compute_score(res) for name, res in by_mr.items()}


def print_report(results_map: Dict[str, List[MRResult]]):
    model_scores = score_by_model(results_map)
    print("\n" + "="*50)
    print("  ROBUSTNESS SCORES")
    print("="*50)
    for name, score in model_scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {name:<30} {bar}  {score:.1%}")
    print("="*50 + "\n")
