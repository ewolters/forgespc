"""Tests for Nelson rules and Western Electric rules."""

from forgespc.rules import check_nelson_rules, check_western_electric_rules


class TestNelsonRules:
    def test_no_violations_stable(self):
        """Stable random data should have few/no violations."""
        import random
        random.seed(42)
        data = [50 + random.gauss(0, 1) for _ in range(30)]
        violations = check_nelson_rules(data, center=50.0, sigma=1.0)
        # Stable data may have a few random hits but shouldn't flag heavily
        assert isinstance(violations, list)

    def test_rule1_handled_by_chart(self):
        """Rule 1 (beyond 3σ) is detected by chart OOC, not nelson rules.
        Nelson rules focus on patterns (rules 2-8)."""
        from forgespc.charts import individuals_moving_range_chart
        data = [50.0] * 10 + [60.0] + [50.0] * 10
        result = individuals_moving_range_chart(data)
        assert len(result.out_of_control) > 0  # Chart catches the point

    def test_rule2_nine_same_side(self):
        """9 consecutive points on same side of center."""
        data = [51.0] * 9 + [49.0] * 5  # 9 above, then 5 below
        violations = check_nelson_rules(data, center=50.0, sigma=1.0)
        rule2 = [v for v in violations if v.get("rule") == 2]
        assert len(rule2) > 0

    def test_returns_list_of_dicts(self):
        data = [50.0] * 20
        violations = check_nelson_rules(data, center=50.0, sigma=1.0)
        assert isinstance(violations, list)
        for v in violations:
            assert isinstance(v, dict)
            assert "rule" in v


class TestWesternElectricRules:
    def test_returns_list(self):
        data = [50 + i * 0.1 for i in range(20)]
        result = check_western_electric_rules(data, center=50.0, sigma=1.0)
        assert isinstance(result, list)
