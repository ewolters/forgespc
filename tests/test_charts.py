"""Tests for SPC control chart computations."""

import random
import pytest
from forgespc.charts import (
    individuals_moving_range_chart,
    xbar_r_chart,
    p_chart,
    c_chart,
    calculate_summary,
)
from forgespc.models import ControlChartResult, ControlLimits


@pytest.fixture
def stable_data():
    """30 points from a stable process (mean=50, sigma=1)."""
    random.seed(42)
    return [50 + random.gauss(0, 1) for _ in range(30)]


@pytest.fixture
def subgroup_data():
    """25 subgroups of 5 from a stable process."""
    random.seed(42)
    return [[50 + random.gauss(0, 1) for _ in range(5)] for _ in range(25)]


class TestIMRChart:
    def test_returns_result(self, stable_data):
        result = individuals_moving_range_chart(stable_data)
        assert isinstance(result, ControlChartResult)
        assert result.chart_type == "I-MR"

    def test_control_limits_reasonable(self, stable_data):
        result = individuals_moving_range_chart(stable_data)
        assert result.limits.ucl > result.limits.cl > result.limits.lcl
        # For sigma=1 data, limits should be roughly mean ± 3
        assert 52 < result.limits.ucl < 54
        assert 46 < result.limits.lcl < 48

    def test_stable_process_in_control(self, stable_data):
        result = individuals_moving_range_chart(stable_data)
        assert result.in_control is True

    def test_out_of_control_detected(self):
        """Process with a shift should detect OOC points."""
        data = [50 + random.gauss(0, 1) for _ in range(20)]
        data += [55 + random.gauss(0, 1) for _ in range(10)]  # Shift!
        result = individuals_moving_range_chart(data)
        assert len(result.out_of_control) > 0

    def test_has_secondary_mr_chart(self, stable_data):
        result = individuals_moving_range_chart(stable_data)
        assert result.secondary_chart is not None
        assert result.secondary_chart.chart_type == "MR"

    def test_custom_control_limits(self, stable_data):
        result = individuals_moving_range_chart(
            stable_data,
            historical_mean=50.0,
            historical_sigma=1.0,
        )
        assert abs(result.limits.cl - 50.0) < 0.01

    def test_minimum_data(self):
        """Need at least 2 points for moving range."""
        result = individuals_moving_range_chart([1.0, 2.0, 3.0])
        assert isinstance(result, ControlChartResult)

    def test_to_dict(self, stable_data):
        result = individuals_moving_range_chart(stable_data)
        d = result.to_dict()
        assert "chart_type" in d
        assert "limits" in d
        assert d["chart_type"] == "I-MR"


class TestXbarRChart:
    def test_returns_result(self, subgroup_data):
        result = xbar_r_chart(subgroup_data)
        assert isinstance(result, ControlChartResult)
        assert result.chart_type == "X-bar R"

    def test_limits_reasonable(self, subgroup_data):
        result = xbar_r_chart(subgroup_data)
        assert result.limits.ucl > result.limits.cl > result.limits.lcl

    def test_has_range_chart(self, subgroup_data):
        result = xbar_r_chart(subgroup_data)
        assert result.secondary_chart is not None
        assert result.secondary_chart.chart_type == "R"


class TestPChart:
    def test_basic_p_chart(self):
        defectives = [3, 2, 5, 1, 4, 6, 2, 3, 1, 5]
        result = p_chart(defectives, sample_sizes=[100] * 10)
        assert result.chart_type == "p"
        assert result.limits.ucl > result.limits.cl > result.limits.lcl

    def test_p_chart_proportions(self):
        defectives = [5, 5, 5, 5, 5]
        result = p_chart(defectives, sample_sizes=[100] * 5)
        assert abs(result.limits.cl - 0.05) < 0.001


class TestCChart:
    def test_basic_c_chart(self):
        defects = [2, 3, 1, 4, 2, 5, 3, 1, 2, 4]
        result = c_chart(defects)
        assert result.chart_type == "c"
        assert result.limits.ucl > result.limits.cl
        assert result.limits.lcl >= 0  # Can't have negative defects


class TestSummary:
    def test_basic_summary(self, stable_data):
        s = calculate_summary(stable_data)
        assert s.n == 30
        assert 49 < s.mean < 51
        assert s.std_dev > 0
        assert s.min_val < s.max_val

    def test_summary_range(self, stable_data):
        s = calculate_summary(stable_data)
        assert abs(s.range_val - (s.max_val - s.min_val)) < 0.001
