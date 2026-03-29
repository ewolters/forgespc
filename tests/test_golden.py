"""Golden file tests — calibrated against known statistical references.

These tests verify that ForgeSPC produces results within tolerance of
pre-computed reference values. Golden files copied from SVEND's
calibration system (CAL-001).
"""

import json
import math
import random
from pathlib import Path

import pytest

from forgespc.charts import (
    c_chart,
    calculate_summary,
    individuals_moving_range_chart,
    p_chart,
    xbar_r_chart,
)
from forgespc.capability import calculate_capability

GOLDEN_DIR = Path(__file__).parent / "golden"


def _load_golden(filename: str) -> dict:
    with open(GOLDEN_DIR / filename) as f:
        return json.load(f)


def _generate_normal(n: int, mean: float = 50, sigma: float = 2, seed: int = 42) -> list[float]:
    """Generate reproducible normal data matching golden file assumptions."""
    random.seed(seed)
    return [mean + random.gauss(0, sigma) for _ in range(n)]


class TestGoldenIMR:
    """I-MR chart golden tests (CAL-SPC-001, CAL-SPC-002)."""

    def test_cal_spc_001_stable(self):
        """Stable N(50,2) should produce in-control I-MR chart."""
        golden = _load_golden("spc_imr_cal_spc_001.json")
        data = _generate_normal(50, mean=50, sigma=2)

        result = individuals_moving_range_chart(data)
        expected = golden["expected"]

        # Check grand mean within tolerance
        exp_mean = expected["statistics.grand_mean"]
        assert abs(result.limits.cl - exp_mean["value"]) < exp_mean["tolerance"], \
            f"Grand mean {result.limits.cl:.4f} not within {exp_mean['tolerance']} of {exp_mean['value']}"

        # Check UCL within tolerance
        exp_ucl = expected["statistics.ucl"]
        assert abs(result.limits.ucl - exp_ucl["value"]) < exp_ucl["tolerance"]

        # Check OOC count
        exp_ooc = expected["statistics.n_ooc"]
        assert abs(len(result.out_of_control) - exp_ooc["value"]) <= exp_ooc["tolerance"]


class TestGoldenXbarR:
    """X-bar R chart golden tests (CAL-SPC-004, CAL-SPC-005)."""

    def test_cal_spc_004_stable_subgroups(self):
        """Stable subgroup data should produce in-control X-bar R chart."""
        golden = _load_golden("spc_xbar_r_cal_spc_004.json")
        random.seed(42)
        subgroups = [[50 + random.gauss(0, 2) for _ in range(5)] for _ in range(25)]

        result = xbar_r_chart(subgroups)
        expected = golden["expected"]

        # Grand mean
        if "statistics.grand_mean" in expected:
            exp = expected["statistics.grand_mean"]
            assert abs(result.limits.cl - exp["value"]) < exp["tolerance"]


class TestGoldenCapability:
    """Process capability golden tests (CAL-SPC-003, CAL-SPC-014)."""

    def test_cal_spc_003_capability(self):
        """Capability on N(50,2) with specs 44-56."""
        golden = _load_golden("spc_capability_cal_spc_003.json")
        data = _generate_normal(100, mean=50, sigma=2)

        cap = calculate_capability(data, usl=56.0, lsl=44.0)
        expected = golden["expected"]

        if "statistics.cp" in expected:
            exp = expected["statistics.cp"]
            assert abs(cap.cp - exp["value"]) < exp["tolerance"], \
                f"Cp={cap.cp:.3f}, expected {exp['value']} ± {exp['tolerance']}"

        if "statistics.cpk" in expected:
            exp = expected["statistics.cpk"]
            assert abs(cap.cpk - exp["value"]) < exp["tolerance"], \
                f"Cpk={cap.cpk:.3f}, expected {exp['value']} ± {exp['tolerance']}"


class TestGoldenPChart:
    """p-chart golden tests (CAL-SPC-008, CAL-SPC-009)."""

    def test_cal_spc_008_p_chart(self):
        """p-chart on defective data."""
        golden = _load_golden("spc_p_chart_cal_spc_008.json")
        expected = golden["expected"]

        # Generate p-chart data matching golden assumptions
        random.seed(42)
        n_samples = 20
        sample_size = 100
        defectives = [int(random.gauss(5, 2)) for _ in range(n_samples)]
        defectives = [max(0, d) for d in defectives]

        result = p_chart(defectives, sample_sizes=[sample_size] * n_samples)

        # Verify chart type
        assert result.chart_type == "p"
        # Verify limits are reasonable (within golden tolerance if available)
        if "statistics.p_bar" in expected:
            exp = expected["statistics.p_bar"]
            assert abs(result.limits.cl - exp["value"]) < exp["tolerance"]
