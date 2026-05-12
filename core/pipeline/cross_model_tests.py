"""
cross_model_tests.py — run metamorphic tests across ALL 5 model architectures
and compute generalization scores.

Models tested:
  1. DAVE-2 (CNN / Deep Learning)
  2. GPR (Gaussian Process)
  3. SVR (Support Vector Regression)
  4. Random Forest (Ensemble)
  5. Linear Regression (Baseline)
"""

from typing import List, Dict, Optional
import numpy as np

from core.adapters.steering_adapter import SteeringAdapter
from core.adapters.gpr_steering_adapter import GPRSteeringAdapter
from core.adapters.svr_steering_adapter import SVRSteeringAdapter
from core.adapters.rf_steering_adapter import RFSteeringAdapter
from core.adapters.linear_steering_adapter import LinearSteeringAdapter
from core.metamorphic_engine.mr_library import get_steering_mrs
from core.metamorphic_engine.mr_base import MRResult
from evaluation.generalization_scorer import (
    compute_generalization_score,
    compute_cross_model_table,
    rank_models_by_robustness,
    identify_model_weaknesses,
    MR_CATEGORIES,
)


def load_all_steering_adapters(
    dave2_path: str = None,
    gpr_path: str = None,
    svr_path: str = None,
    rf_path: str = None,
    linear_path: str = None,
) -> Dict:
    """
    Load all 5 model adapters. Each falls back to its stub if weights not found.
    Returns { model_name: loaded_adapter }
    """
    adapters = {}

    dave2 = SteeringAdapter(); dave2.load(dave2_path)
    adapters["dave2_cnn"] = dave2

    gpr = GPRSteeringAdapter(); gpr.load(gpr_path)
    adapters["gpr_steering"] = gpr

    svr = SVRSteeringAdapter(); svr.load(svr_path)
    adapters["svr_steering"] = svr

    rf = RFSteeringAdapter(); rf.load(rf_path)
    adapters["random_forest"] = rf

    lin = LinearSteeringAdapter(); lin.load(linear_path)
    adapters["linear_baseline"] = lin

    return adapters


def run_cross_model_evaluation(
    adapters: Dict,
    inputs: List[np.ndarray],
    tolerance: float = None,
    transforms_per_category: bool = True,
) -> Dict[str, List[MRResult]]:
    """
    Run all 14 MRs across all loaded model adapters.

    Args:
        adapters:               { model_name: BaseAdapter }
        inputs:                 list of test images
        tolerance:              shared tolerance (None = use per-MR defaults)
        transforms_per_category: apply the canonical transform for each MR category

    Returns:
        { model_name: [MRResult, ...] }
    """
    from core.metamorphic_engine.transformer import get_transform

    # Canonical transform mapping: MR name → transform name
    MR_TRANSFORM_MAP = {
        # Cat A
        "brightness_steering":        "brightness",
        "contrast_steering":          "contrast",
        "noise_robustness_steering":  "noise",
        "weather_robustness":         "fog",
        "occlusion_robustness":       "noise",
        # Cat B
        "flip_symmetry_steering":     "flip",
        "straight_road_stability":    "brightness",   # tests near-zero on straight
        "perspective_scaling":        "scale",
        # Cat C
        "temporal_smoothness":        "translate",
        "constant_curvature_consistency": "translate",
        "small_lane_shift_invariance": "translate",
        # Cat D
        "composed_brightness_noise":  "composed_bright_noise",
        "composed_weather_blur":      "composed_weather_blur",
        "lane_center_shift_consistency": "composed_shift_bright",
    }

    mrs = get_steering_mrs(tolerance)
    results_map: Dict[str, List[MRResult]] = {}

    for model_name, adapter in adapters.items():
        model_results = []
        for mr in mrs:
            tf_name = MR_TRANSFORM_MAP.get(mr.name, "brightness")
            transform = get_transform(tf_name, domain="image")
            for img in inputs:
                img2 = transform(img)
                y1 = adapter.predict(img)
                y2 = adapter.predict(img2)
                result = mr.check(img, img2, y1, y2)
                model_results.append(result)
        results_map[model_name] = model_results
        total = len(model_results)
        passed = sum(r.passed for r in model_results)
        print(f"  [{model_name}] {passed}/{total} passed ({passed/total:.1%})")

    return results_map


def print_generalization_report(results_map: Dict[str, List[MRResult]]):
    """Print the cross-model generalization table to console."""
    gen_scores = compute_generalization_score(results_map)
    table = compute_cross_model_table(results_map)
    ranking = rank_models_by_robustness(results_map)
    weaknesses = identify_model_weaknesses(results_map)

    model_names = list(results_map.keys())

    print("\n" + "="*70)
    print("  CROSS-MODEL GENERALIZATION TABLE")
    print("="*70)

    header = f"{'MR Name':<35}" + "".join(f"{m[:8]:<10}" for m in model_names) + "  Gen.Score"
    print(header)
    print("-" * len(header))

    for mr_name in sorted(gen_scores.keys()):
        row = f"{mr_name:<35}"
        for model in model_names:
            mark = table.get(mr_name, {}).get(model, " ")
            row += f"{mark:<10}"
        row += f"  {gen_scores[mr_name]:.0%}"
        cat = MR_CATEGORIES.get(mr_name, "?")
        row += f"  [Cat {cat}]"
        print(row)

    print("\n" + "="*70)
    print("  MODEL ROBUSTNESS RANKING")
    print("="*70)
    for rank, (model, score) in enumerate(ranking, 1):
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {rank}. {model:<25} {bar}  {score:.1%}")

    print("\n" + "="*70)
    print("  MODEL WEAKNESSES (MRs they fail)")
    print("="*70)
    for model, failed in weaknesses.items():
        if failed:
            print(f"  {model}: {', '.join(failed)}")
        else:
            print(f"  {model}: ✓ No weaknesses detected")

    avg_gen = np.mean(list(gen_scores.values()))
    print(f"\n  Average Generalization Score: {avg_gen:.1%}")
    print("="*70 + "\n")


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading all model adapters...")
    adapters = load_all_steering_adapters()

    print("\nGenerating test images...")
    imgs = []
    for i in range(5):
        x = np.linspace(0.1, 0.9, 64)
        shift = i * 0.05
        img = np.stack([
            np.tile(x + shift, (64, 1)),
            np.tile(x[::-1], (64, 1)),
            np.ones((64, 64)) * (0.3 + i * 0.05),
        ], axis=-1).clip(0, 1).astype(np.float32)
        imgs.append(img)

    print("\nRunning cross-model evaluation...")
    results_map = run_cross_model_evaluation(adapters, imgs)
    print_generalization_report(results_map)
