"""Golden file tests — calibrated against embedded reference data.

Each golden file is self-contained: input data + expected results.
No random seed dependencies, no ambient assumptions.
"""

import pytest
from forgespc.calibration import calibrate


class TestCalibrationService:
    def test_calibrate_passes_all(self):
        """All golden files with embedded data should pass calibration."""
        report = calibrate()
        assert report.total_checks > 0, "No checks ran"
        assert report.is_calibrated, (
            f"Calibration failed: {report.pass_rate:.0%} pass rate. "
            f"Failures: {[(f.case_id, f.metric, f.expected, f.actual) for f in report.failures]}"
        )

    def test_calibrate_check_count(self):
        """Should have at least 15 calibration checks."""
        report = calibrate()
        assert report.total_checks >= 15

    def test_calibrate_no_errors(self):
        """No errors during calibration (file read, computation, etc)."""
        report = calibrate()
        # CUSUM/EWMA/Xbar-S don't have embedded data yet — those skip, not error
        real_errors = [e for e in report.errors if "no embedded data" not in e and "unsupported" not in e]
        assert real_errors == [], f"Calibration errors: {real_errors}"

    def test_report_structure(self):
        """Report has all expected fields."""
        report = calibrate()
        assert hasattr(report, "version")
        assert hasattr(report, "pass_rate")
        assert hasattr(report, "is_calibrated")
        assert hasattr(report, "checks")
        assert hasattr(report, "failures")

    def test_individual_checks_have_metadata(self):
        """Each check has case_id, metric, expected, actual, tolerance."""
        report = calibrate()
        for check in report.checks:
            assert check.case_id
            assert check.metric
            assert isinstance(check.expected, (int, float))
            assert isinstance(check.actual, (int, float))
            assert isinstance(check.tolerance, (int, float))
            assert isinstance(check.passed, bool)
