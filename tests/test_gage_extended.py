"""Tests for nested Gage R&R and attribute agreement."""

import pytest


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
