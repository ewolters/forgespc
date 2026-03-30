"""Tests for Gage R&R (crossed, nested) and attribute agreement."""

import pytest


class TestHotellingTSquared:
    def test_in_control_process(self):
        from forgespc.gage import hotelling_t_squared_chart

        # 20 observations, 3 variables — stable process
        import random
        random.seed(42)
        data = [[10 + random.gauss(0, 0.5), 20 + random.gauss(0, 0.3), 5 + random.gauss(0, 0.2)]
                for _ in range(20)]
        result = hotelling_t_squared_chart(data)
        assert result.chart_type == "T-squared"
        assert result.limits.ucl > 0
        assert result.limits.lcl == 0.0
        assert len(result.data_points) == 20
        assert isinstance(result.in_control, bool)
        assert "Hotelling" in result.summary

    def test_out_of_control_detection(self):
        from forgespc.gage import hotelling_t_squared_chart

        # Stable process with one extreme outlier
        import random
        random.seed(42)
        data = [[10 + random.gauss(0, 0.1), 20 + random.gauss(0, 0.1)]
                for _ in range(30)]
        # Inject outlier
        data[15] = [50.0, 50.0]
        result = hotelling_t_squared_chart(data)
        assert len(result.out_of_control) >= 1
        ooc_indices = [p["index"] for p in result.out_of_control]
        assert 15 in ooc_indices

    def test_too_few_observations_raises(self):
        from forgespc.gage import hotelling_t_squared_chart

        data = [[1, 2, 3]]  # 1 obs, 3 vars — need at least 4
        with pytest.raises(ValueError, match="Need at least"):
            hotelling_t_squared_chart(data)

    def test_with_spec_limits(self):
        from forgespc.gage import hotelling_t_squared_chart

        import random
        random.seed(99)
        data = [[10 + random.gauss(0, 0.5), 5 + random.gauss(0, 0.3)]
                for _ in range(15)]
        result = hotelling_t_squared_chart(data, usl=20.0, lsl=0.0)
        assert result.limits.usl == 20.0
        assert result.limits.lsl == 0.0


class TestCrossedGageRR:
    def test_basic_crossed(self):
        from forgespc.gage import gage_rr_crossed

        # 3 parts, 2 operators, 2 replicates
        parts =      [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        operators =  ["A", "A", "B", "B", "A", "A", "B", "B", "A", "A", "B", "B"]
        measurements = [10.1, 10.2, 10.3, 10.1, 20.0, 20.1, 20.2, 19.9, 30.0, 30.1, 29.9, 30.0]

        result = gage_rr_crossed(parts, operators, measurements)
        assert result.assessment in ("Acceptable", "Marginal", "Unacceptable")
        assert result.n_parts == 3
        assert result.n_operators == 2
        assert result.n_replicates == 2
        assert result.n_total == 12
        assert result.var_total > 0
        assert abs(result.pct_contribution["Total"] - 100.0) < 0.1
        assert len(result.anova_table) >= 4  # Part, Operator, Interaction, Repeatability, Total
        assert result.ndc >= 1

    def test_good_gage_is_acceptable(self):
        from forgespc.gage import gage_rr_crossed

        # Parts vary a lot, measurement noise is small
        parts =      [1]*4 + [2]*4 + [3]*4 + [4]*4 + [5]*4
        operators =  ["A", "A", "B", "B"] * 5
        measurements = [
            10.00, 10.01, 10.00, 10.01,  # part 1
            20.00, 20.01, 20.00, 20.01,  # part 2
            30.00, 30.01, 30.00, 30.01,  # part 3
            40.00, 40.01, 40.00, 40.01,  # part 4
            50.00, 50.01, 50.00, 50.01,  # part 5
        ]
        result = gage_rr_crossed(parts, operators, measurements)
        assert result.assessment == "Acceptable"
        assert result.grr_percent < 10
        assert result.ndc >= 5

    def test_with_tolerance(self):
        from forgespc.gage import gage_rr_crossed

        parts =      [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        operators =  ["A", "A", "B", "B", "A", "A", "B", "B", "A", "A", "B", "B"]
        measurements = [10.1, 10.2, 10.3, 10.1, 20.0, 20.1, 20.2, 19.9, 30.0, 30.1, 29.9, 30.0]

        result = gage_rr_crossed(parts, operators, measurements, tolerance=25.0)
        assert result.tolerance == 25.0
        assert len(result.pct_tolerance) > 0
        assert "GRR" in result.pct_tolerance

    def test_plots_generated(self):
        from forgespc.gage import gage_rr_crossed

        parts =      [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        operators =  ["A", "A", "B", "B", "A", "A", "B", "B", "A", "A", "B", "B"]
        measurements = [10.1, 10.2, 10.3, 10.1, 20.0, 20.1, 20.2, 19.9, 30.0, 30.1, 29.9, 30.0]

        result = gage_rr_crossed(parts, operators, measurements)
        assert "components" in result.plots
        assert "by_part" in result.plots
        assert "by_operator" in result.plots
        assert "interaction" in result.plots

    def test_unbalanced_raises(self):
        from forgespc.gage import gage_rr_crossed

        parts =      [1, 1, 1, 2, 2]
        operators =  ["A", "A", "B", "A", "A"]
        measurements = [10.0, 10.1, 10.2, 20.0, 20.1]

        with pytest.raises(ValueError):
            gage_rr_crossed(parts, operators, measurements)


class TestNestedGageRR:
    def test_basic_nested(self):
        from forgespc.gage import gage_rr_nested

        # 2 operators, 3 parts each, 2 replicates
        measurements = [
            [[10.1, 10.2], [10.5, 10.4], [9.8, 9.9]],  # operator 1
            [[10.3, 10.1], [10.6, 10.7], [9.7, 9.8]],   # operator 2
        ]
        result = gage_rr_nested(measurements)
        assert result.assessment in ("Acceptable", "Marginal", "Unacceptable")
        assert result.pct_contribution["gage_rr"] >= 0
        assert result.pct_contribution["part_to_part"] >= 0
        assert abs(float(result.pct_contribution['gage_rr'] + result.pct_contribution['part_to_part']) - 100) < 1

    def test_acceptable_gage(self):
        from forgespc.gage import gage_rr_nested

        # Good gage — low repeatability, high part variation
        measurements = [
            [[10.0, 10.01], [20.0, 20.01], [30.0, 30.01]],
            [[10.5, 10.51], [20.5, 20.51], [30.5, 30.51]],
        ]
        result = gage_rr_nested(measurements)
        assert result.pct_contribution["part_to_part"] > 50


class TestAttributeAgreement:
    def test_perfect_agreement(self):
        from forgespc.gage import attribute_agreement

        ratings = [
            [1, 1, 0, 0, 1],  # appraiser 1
            [1, 1, 0, 0, 1],  # appraiser 2
        ]
        result = attribute_agreement(ratings)
        assert result["between_appraiser_pct"] == 100.0
        assert result["fleiss_kappa"] > 0.9

    def test_poor_agreement(self):
        from forgespc.gage import attribute_agreement

        ratings = [
            [1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 1, 0, 1, 0, 1],  # opposite
        ]
        result = attribute_agreement(ratings)
        assert result["between_appraiser_pct"] == 0.0
        assert result["fleiss_kappa"] < 0

    def test_vs_reference(self):
        from forgespc.gage import attribute_agreement

        ratings = [
            [1, 1, 0, 0, 1],
            [1, 0, 0, 0, 1],
        ]
        reference = [1, 1, 0, 0, 1]
        result = attribute_agreement(ratings, reference=reference)
        assert result["vs_reference"]["Appraiser 1"] == 100.0
        assert result["vs_reference"]["Appraiser 2"] == 80.0
