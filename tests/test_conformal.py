"""Tests for conformal SPC — distribution-free monitoring."""

import random
import numpy as np
import pytest

from forgespc.conformal import conformal_control, entropy_spc


class TestConformalControl:
    def test_stable_process(self):
        random.seed(42)
        data = [50 + random.gauss(0, 1) for _ in range(100)]
        result = conformal_control(data, alpha=0.05)
        assert result.n_calibration == 50
        assert result.n_monitoring == 50
        # Stable process: few OOC points (alpha * n_monitoring expected)
        assert result.n_ooc < 10

    def test_shift_detected(self):
        random.seed(42)
        data = [50 + random.gauss(0, 1) for _ in range(50)]
        data += [55 + random.gauss(0, 1) for _ in range(50)]
        result = conformal_control(data, alpha=0.05)
        assert result.n_ooc > 5  # Should catch the shift

    def test_non_normal_data(self):
        """Conformal works without normality assumption."""
        random.seed(42)
        # Exponential data (highly skewed)
        data = [random.expovariate(0.1) for _ in range(100)]
        result = conformal_control(data, alpha=0.05)
        # Should still produce valid results
        assert result.threshold > 0
        assert len(result.prediction_intervals) == 100

    def test_prediction_intervals(self):
        data = [50 + random.gauss(0, 1) for _ in range(100)]
        result = conformal_control(data)
        assert result.prediction_intervals is not None
        for lower, upper in result.prediction_intervals:
            assert lower < upper

    def test_too_few_observations(self):
        with pytest.raises(ValueError, match="at least 20"):
            conformal_control([1.0] * 10)


class TestEntropySPC:
    def test_stable_process(self):
        random.seed(42)
        data = [50 + random.gauss(0, 1) for _ in range(100)]
        result = entropy_spc(data, window_size=20)
        assert result.baseline_entropy > 0
        assert result.ucl > result.lcl

    def test_distribution_change_detected(self):
        """Entropy should detect distribution shape changes."""
        random.seed(42)
        # Normal → bimodal
        data = [50 + random.gauss(0, 1) for _ in range(60)]
        data += [random.choice([45, 55]) + random.gauss(0, 0.3) for _ in range(40)]
        result = entropy_spc(data, window_size=15)
        # Should detect the distributional change
        assert result.n_ooc > 0

    def test_variance_change_detected(self):
        """Entropy should detect spread changes even if mean is stable."""
        random.seed(42)
        data = [50 + random.gauss(0, 0.5) for _ in range(60)]
        data += [50 + random.gauss(0, 3.0) for _ in range(40)]  # Same mean, wider spread
        result = entropy_spc(data, window_size=15)
        assert result.n_ooc > 0

    def test_too_few_observations(self):
        with pytest.raises(ValueError):
            entropy_spc([1.0] * 10, window_size=20)
