"""
run_single_test.py — run a single metamorphic test and return the result.
"""

import numpy as np
from core.adapters.base_adapter import BaseAdapter
from core.metamorphic_engine.mr_base import MetamorphicRelation, MRResult
from core.metamorphic_engine.transformer import get_transform


def run_test(
    adapter:        BaseAdapter,
    mr:             MetamorphicRelation,
    input_data:     np.ndarray,
    transform_name: str,
    domain:         str = "image",
) -> MRResult:
    """
    Run one metamorphic test: transform → predict × 2 → check relation.

    Args:
        adapter:        loaded model adapter
        mr:             metamorphic relation to verify
        input_data:     original input sample
        transform_name: key in transform registry
        domain:         "image" | "sequence"

    Returns:
        MRResult with pass/fail and diagnostics
    """
    transform = get_transform(transform_name, domain)

    transformed = transform(input_data)
    y_orig      = adapter.predict(input_data)
    y_trans     = adapter.predict(transformed)

    result = mr.check(input_data, transformed, y_orig, y_trans)

    print(f"[{adapter.model_name}] {result}")
    return result


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np
    from core.adapters.steering_adapter import SteeringAdapter
    from metamorphic_relations.steering_mrs import FlipSymmetrySteeringMR

    adapter = SteeringAdapter()
    adapter.load()

    mr    = FlipSymmetrySteeringMR(tolerance=0.1)
    image = np.random.rand(64, 64, 3).astype(np.float32)

    result = run_test(adapter, mr, image, transform_name="flip")
    print("PASS" if result.passed else "FAIL")
