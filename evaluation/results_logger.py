"""
Results Logger — saves MRResult lists to JSON / CSV for later analysis.
"""

import json
import csv
from pathlib import Path
from typing import List
from datetime import datetime
from core.metamorphic_engine.mr_base import MRResult


def results_to_dicts(results: List[MRResult]) -> List[dict]:
    return [
        {
            "mr_name": r.mr_name,
            "passed":  r.passed,
            "delta":   r.delta,
            "message": r.message,
        }
        for r in results
    ]


def save_json(results: List[MRResult], path: str = None) -> str:
    path = path or f"data/processed/results_{_ts()}.json"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results_to_dicts(results), f, indent=2)
    print(f"[Logger] Saved {len(results)} results → {path}")
    return path


def save_csv(results: List[MRResult], path: str = None) -> str:
    path = path or f"data/processed/results_{_ts()}.csv"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = results_to_dicts(results)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Logger] Saved {len(results)} results → {path}")
    return path


def _ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")
