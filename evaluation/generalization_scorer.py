"""
Generalization Scorer — the MAIN contribution of this project.

Computes the Generalization Score:
    GeneralizationScore(MR) = models_satisfying_MR / total_models

This measures how universally a metamorphic relation holds across
different ML architectures. High scores = generalizable MR.

Also computes:
  - Per-model robustness ranking
  - Per-category (A/B/C/D) generalization score
  - Cross-model failure patterns
"""

from typing import List, Dict, Tuple
import numpy as np
from core.metamorphic_engine.mr_base import MRResult


def compute_generalization_score(
    results_map: Dict[str, List[MRResult]]
) -> Dict[str, float]:
    """
    For each MR, compute the fraction of models that satisfy it.

    Args:
        results_map: { model_name: [MRResult, ...] }

    Returns:
        { mr_name: generalization_score (0.0 – 1.0) }

    Example output:
        { "brightness_steering": 1.0,   # all 5 models pass
          "flip_symmetry_steering": 0.8, # 4/5 models pass
          "weather_robustness": 0.6 }
    """
    # Collect pass/fail per MR per model
    mr_model_pass: Dict[str, Dict[str, bool]] = {}

    for model_name, results in results_map.items():
        # For each MR, aggregate across all samples for this model
        mr_results_per_mr: Dict[str, List[bool]] = {}
        for r in results:
            mr_results_per_mr.setdefault(r.mr_name, []).append(r.passed)

        for mr_name, passes in mr_results_per_mr.items():
            mr_model_pass.setdefault(mr_name, {})
            # A model "passes" this MR if majority of samples pass
            mr_model_pass[mr_name][model_name] = (sum(passes) / len(passes)) >= 0.5

    generalization_scores = {}
    total_models = len(results_map)

    for mr_name, model_passes in mr_model_pass.items():
        models_passing = sum(1 for passed in model_passes.values() if passed)
        generalization_scores[mr_name] = round(models_passing / total_models, 4)

    return generalization_scores


def compute_category_generalization(
    results_map: Dict[str, List[MRResult]],
    mr_categories: Dict[str, str]  # { mr_name: "A" | "B" | "C" | "D" }
) -> Dict[str, float]:
    """
    Compute average generalization score per MR category (A, B, C, D).

    Args:
        results_map:    { model_name: [MRResult, ...] }
        mr_categories:  { mr_name: category_letter }

    Returns:
        { "A": 0.95, "B": 0.80, "C": 0.75, "D": 0.60 }
    """
    gen_scores = compute_generalization_score(results_map)
    category_scores: Dict[str, List[float]] = {}

    for mr_name, score in gen_scores.items():
        cat = mr_categories.get(mr_name, "Unknown")
        category_scores.setdefault(cat, []).append(score)

    return {
        cat: round(float(np.mean(scores)), 4)
        for cat, scores in category_scores.items()
    }


def compute_cross_model_table(
    results_map: Dict[str, List[MRResult]]
) -> Dict[str, Dict[str, str]]:
    """
    Build a cross-model × MR pass/fail table.

    Returns:
        { mr_name: { model_name: "✓" | "✗" } }

    Example:
        {
          "brightness_steering": { "dave2": "✓", "gpr": "✓", "svr": "✓", "rf": "✓", "linear": "✓" },
          "flip_symmetry":       { "dave2": "✓", "gpr": "✓", "svr": "✓", "rf": "✗", "linear": "✓" },
        }
    """
    table: Dict[str, Dict[str, str]] = {}

    for model_name, results in results_map.items():
        mr_pass: Dict[str, List[bool]] = {}
        for r in results:
            mr_pass.setdefault(r.mr_name, []).append(r.passed)

        for mr_name, passes in mr_pass.items():
            table.setdefault(mr_name, {})
            majority_pass = (sum(passes) / len(passes)) >= 0.5
            table[mr_name][model_name] = "✓" if majority_pass else "✗"

    return table


def rank_models_by_robustness(
    results_map: Dict[str, List[MRResult]]
) -> List[Tuple[str, float]]:
    """
    Rank models from most to least robust, with their overall pass rate.

    Returns:
        [ ("dave2", 0.95), ("gpr", 0.85), ("svr", 0.80), ... ]
    """
    scores = []
    for model_name, results in results_map.items():
        if results:
            pass_rate = sum(r.passed for r in results) / len(results)
            scores.append((model_name, round(pass_rate, 4)))
    return sorted(scores, key=lambda x: x[1], reverse=True)


def identify_model_weaknesses(
    results_map: Dict[str, List[MRResult]]
) -> Dict[str, List[str]]:
    """
    Identify which MRs each model fails.

    Returns:
        { model_name: [ "mr_that_fails", ... ] }

    Example findings:
        { "gpr_steering": ["noise_robustness_steering", "composed_weather_blur"],
          "linear_steering": ["flip_symmetry_steering"] }
    """
    weaknesses: Dict[str, List[str]] = {}

    for model_name, results in results_map.items():
        mr_pass: Dict[str, List[bool]] = {}
        for r in results:
            mr_pass.setdefault(r.mr_name, []).append(r.passed)

        failed_mrs = [
            mr_name for mr_name, passes in mr_pass.items()
            if (sum(passes) / len(passes)) < 0.5
        ]
        weaknesses[model_name] = failed_mrs

    return weaknesses


def statistical_threshold(steering_range: float = 2.0, fraction: float = 0.05) -> float:
    """
    Compute a dataset-based statistical threshold.
    Formula: ε = fraction × steering_range
    Default: ε = 0.05 × 2.0 = 0.10

    This avoids hardcoding thresholds per model and improves generalization.
    """
    return round(fraction * steering_range, 4)


# ── MR category lookup (populated from steering_mrs.py classes) ───────────────

MR_CATEGORIES: Dict[str, str] = {
    # Category A — Invariance
    "brightness_steering":         "A",
    "contrast_steering":           "A",
    "noise_robustness_steering":   "A",
    "weather_robustness":          "A",
    "occlusion_robustness":        "A",
    # Category B — Symmetry
    "flip_symmetry_steering":      "B",
    "straight_road_stability":     "B",
    "perspective_scaling":         "B",
    # Category C — Temporal
    "temporal_smoothness":         "C",
    "constant_curvature_consistency": "C",
    "small_lane_shift_invariance": "C",
    # Category D — Composition
    "composed_brightness_noise":   "D",
    "composed_weather_blur":       "D",
    "lane_center_shift_consistency": "D",
}
