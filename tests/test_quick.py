"""Repository smoke test for automated or local execution."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from quick_test import run_quick_test  # noqa: E402


def test_quick_example():
    metrics = run_quick_test()
    assert metrics["geoattribstack_rmse"] < metrics["equal_weight_rmse"]
