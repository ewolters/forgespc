"""Tests for advanced SPC charts — CUSUM, EWMA, X-bar/S."""

import random
import numpy as np
import pytest

from forgespc.advanced import cusum_chart, ewma_chart, xbar_s_chart, CUSUMResult, EWMAResult
from forgespc.models import ControlChartResult


# ─── CUSUM ───────────────────────────────────────────────────────────────


class TestCUSUM:
    def test_stable_process_few_signals(self):
        """Stable N(50,2) should produce very few CUSUM signals."""
        random.seed(42)
        data = [50 + random.gauss(0, 2) for _ in range(50)]
        result = cusum_chart(data, target=50.0)
        assert isinstance(result, CUSUMResult)
        # CUSUM is sensitive — may get 0-2 false alarms on random data
        assert result.n_signals <= 3

    def test_shift_detected(self):
        """Mean shift 50→55 should trigger CUSUM signals."""
        random.seed(42)
        data = [50 + random.gauss(0, 2) for _ in range(30)]
        data += [55 + random.gauss(0, 2) for _ in range(20)]
        result = cusum_chart(data, target=50.0)
        assert result.n_signals > 0
        assert result.in_control is False
        assert len(result.signals_up) > 0

    def test_downward_shift(self):
        """Mean shift 50→45 should trigger downward signals."""
        random.seed(42)
        data = [50 + random.gauss(0, 2) for _ in range(30)]
        data += [45 + random.gauss(0, 2) for _ in range(20)]
        result = cusum_chart(data, target=50.0)
        assert len(result.signals_down) > 0

    def test_custom_parameters(self):
        random.seed(42)
        data = [50 + random.gauss(0, 2) for _ in range(50)]
        result = cusum_chart(data, target=50.0, k=1.0, h=8.0)
        assert result.k == 1.0
        assert result.h == 8.0

    def test_auto_target(self):
        """Target defaults to mean of data."""
        data = [100.0, 101.0, 99.0, 100.5]
        result = cusum_chart(data)
        assert abs(result.target - 100.125) < 0.01

    def test_to_chart_result(self):
        data = [50 + random.gauss(0, 2) for _ in range(30)]
        result = cusum_chart(data)
        cr = result.to_chart_result()
        assert isinstance(cr, ControlChartResult)
        assert cr.chart_type == "CUSUM"

    def test_cusum_state_lengths(self):
        data = [50.0] * 20
        result = cusum_chart(data)
        assert len(result.cusum_pos) == 20
        assert len(result.cusum_neg) == 20


# ─── EWMA ────────────────────────────────────────────────────────────────


class TestEWMA:
    def test_stable_process_in_control(self):
        """Stable N(50,2) should be in control."""
        random.seed(42)
        data = [50 + random.gauss(0, 2) for _ in range(50)]
        result = ewma_chart(data, target=50.0)
        assert isinstance(result, EWMAResult)
        assert result.in_control is True

    def test_shift_detected(self):
        """Mean shift should trigger EWMA out-of-control."""
        random.seed(42)
        data = [50 + random.gauss(0, 2) for _ in range(30)]
        data += [56 + random.gauss(0, 2) for _ in range(20)]
        result = ewma_chart(data, target=50.0)
        assert result.in_control is False
        assert len(result.out_of_control_indices) > 0

    def test_ewma_smoothing(self):
        """EWMA values should be smoother than raw data."""
        random.seed(42)
        data = [50 + random.gauss(0, 5) for _ in range(50)]
        result = ewma_chart(data, target=50.0, lambda_param=0.1)
        # EWMA std should be less than raw data std
        ewma_std = np.std(result.ewma_values)
        raw_std = np.std(data)
        assert ewma_std < raw_std

    def test_time_varying_limits(self):
        """UCL/LCL should start narrow and widen to steady state."""
        data = [50.0] * 20
        result = ewma_chart(data, target=50.0)
        assert result.ucl[0] < result.ucl[-1]  # Starts narrow
        # Should approach steady state
        assert abs(result.ucl[-1] - result.ucl_steady) < 0.01

    def test_to_chart_result(self):
        data = [50 + random.gauss(0, 2) for _ in range(30)]
        result = ewma_chart(data)
        cr = result.to_chart_result()
        assert cr.chart_type == "EWMA"


# ─── X-bar/S ─────────────────────────────────────────────────────────────


class TestXbarS:
    def test_stable_subgroups(self):
        """Stable subgroups should produce in-control chart."""
        random.seed(42)
        subgroups = [[50 + random.gauss(0, 2) for _ in range(10)] for _ in range(25)]
        result = xbar_s_chart(subgroups)
        assert isinstance(result, ControlChartResult)
        assert result.chart_type == "X-bar S"
        assert result.limits.ucl > result.limits.cl > result.limits.lcl

    def test_has_s_chart(self):
        random.seed(42)
        subgroups = [[50 + random.gauss(0, 2) for _ in range(10)] for _ in range(25)]
        result = xbar_s_chart(subgroups)
        assert result.secondary_chart is not None
        assert result.secondary_chart.chart_type == "S"

    def test_shift_detected(self):
        random.seed(42)
        stable = [[50 + random.gauss(0, 2) for _ in range(10)] for _ in range(15)]
        shifted = [[58 + random.gauss(0, 2) for _ in range(10)] for _ in range(10)]
        result = xbar_s_chart(stable + shifted)
        assert len(result.out_of_control) > 0

    def test_historical_limits(self):
        random.seed(42)
        subgroups = [[50 + random.gauss(0, 2) for _ in range(5)] for _ in range(25)]
        result = xbar_s_chart(subgroups, historical_mean=50.0, historical_sigma=2.0)
        assert abs(result.limits.cl - 50.0) < 0.01
