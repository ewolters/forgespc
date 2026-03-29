"""Tests for process capability analysis."""

import random
import pytest
from forgespc.capability import calculate_capability


@pytest.fixture
def capable_data():
    """Capable process: Cpk > 1.33."""
    random.seed(42)
    return [50 + random.gauss(0, 0.5) for _ in range(100)]


@pytest.fixture
def marginal_data():
    """Marginal process: Cpk ~ 1.0."""
    random.seed(42)
    return [50 + random.gauss(0, 1.0) for _ in range(100)]


class TestCapability:
    def test_capable_process(self, capable_data):
        cap = calculate_capability(capable_data, usl=52.0, lsl=48.0)
        assert cap.cp > 1.0
        assert cap.cpk > 1.0
        assert cap.sigma_level > 3.0
        assert cap.yield_percent > 99.0

    def test_marginal_process(self, marginal_data):
        cap = calculate_capability(marginal_data, usl=53.0, lsl=47.0)
        assert 0.5 < cap.cpk < 2.0

    def test_incapable_process(self):
        """Wide variation relative to spec."""
        random.seed(42)
        data = [50 + random.gauss(0, 5) for _ in range(100)]
        cap = calculate_capability(data, usl=52.0, lsl=48.0)
        assert cap.cpk < 0.5
        assert cap.dpmo > 100000

    def test_centered_vs_shifted(self):
        """Cp should be same, Cpk should differ."""
        random.seed(42)
        centered = [50 + random.gauss(0, 0.5) for _ in range(100)]
        shifted = [51 + random.gauss(0, 0.5) for _ in range(100)]

        cap_c = calculate_capability(centered, usl=52.0, lsl=48.0)
        cap_s = calculate_capability(shifted, usl=52.0, lsl=48.0)

        # Cp should be similar (same spread)
        assert abs(cap_c.cp - cap_s.cp) < 0.5
        # Cpk should be worse for shifted
        assert cap_s.cpk < cap_c.cpk

    def test_has_interpretation(self, capable_data):
        cap = calculate_capability(capable_data, usl=52.0, lsl=48.0)
        assert cap.interpretation  # Non-empty string

    def test_to_dict(self, capable_data):
        cap = calculate_capability(capable_data, usl=52.0, lsl=48.0)
        d = cap.to_dict()
        assert "cp" in d
        assert "cpk" in d
        assert "dpmo" in d
        assert "sigma_level" in d

    def test_one_sided_spec(self):
        """Only USL specified."""
        random.seed(42)
        data = [50 + random.gauss(0, 1) for _ in range(50)]
        cap = calculate_capability(data, usl=53.0, lsl=47.0, target=50.0)
        assert cap.target == 50.0
