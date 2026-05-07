"""
Report Generator — produces a human-readable Markdown report
from a batch of MRResults.
"""

from typing import List, Dict
from datetime import datetime
from core.metamorphic_engine.mr_base import MRResult
from evaluation.robustness_scorer import score_by_model, score_by_mr


def generate_report(
    results_map:  Dict[str, List[MRResult]],
    output_path:  str = None,
) -> str:
    ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines   = [f"# Metamorphic Testing Report\n", f"Generated: {ts}\n\n---\n"]

    model_scores = score_by_model(results_map)
    lines.append("## Overall Robustness Scores\n")
    lines.append("| Model | Score |\n|---|---|\n")
    for name, score in model_scores.items():
        lines.append(f"| {name} | {score:.1%} |\n")

    for model_name, results in results_map.items():
        lines.append(f"\n## Model: `{model_name}`\n")
        mr_scores = score_by_mr(results)
        lines.append("| MR Name | Pass Rate | Failures |\n|---|---|---|\n")
        for mr_name, score in mr_scores.items():
            fails = sum(1 for r in results if r.mr_name == mr_name and not r.passed)
            lines.append(f"| {mr_name} | {score:.1%} | {fails} |\n")

    report = "".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"[ReportGenerator] Report saved → {output_path}")

    return report
