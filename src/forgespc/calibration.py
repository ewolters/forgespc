"""Self-calibration service for ForgeSPC.

Any installation can verify its own accuracy by running calibration
against golden reference values. Golden files are self-contained —
they embed the input data AND expected results, so calibration is
deterministic and reproducible.

Usage:
    from forgespc.calibration import calibrate

    report = calibrate()
    print(f"Pass rate: {report.pass_rate:.0%}")
    print(f"Calibrated: {report.is_calibrated}")
    for f in report.failures:
        print(f"  FAIL: {f.case_id} {f.metric}: expected {f.expected} ± {f.tolerance}, got {f.actual}")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CalibrationCheck:
    """A single calibration check result."""

    case_id: str
    metric: str
    expected: float
    actual: float
    tolerance: float
    passed: bool


@dataclass
class CalibrationReport:
    """Full calibration report."""

    version: str
    total_checks: int = 0
    passed_checks: int = 0
    checks: list[CalibrationCheck] = field(default_factory=list)
    failures: list[CalibrationCheck] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed_checks / self.total_checks if self.total_checks else 0.0

    @property
    def is_calibrated(self) -> bool:
        return self.total_checks > 0 and len(self.failures) == 0


# Golden files shipped with the package
_GOLDEN_DIRS = [
    Path(__file__).parent / "golden",
    Path(__file__).parent.parent.parent.parent / "tests" / "golden",
]


def _find_golden_dir() -> Path | None:
    for d in _GOLDEN_DIRS:
        if d.exists() and list(d.glob("spc_*.json")):
            return d
    return None


def calibrate(golden_dir: str | Path | None = None) -> CalibrationReport:
    """Run self-calibration against golden reference files.

    Each golden file is self-contained: it includes the input data
    and the expected output values with tolerances. The calibration
    service feeds the embedded data through ForgeSPC and compares
    the results against the embedded expectations.

    Args:
        golden_dir: Path to golden files. Auto-detected if not provided.

    Returns:
        CalibrationReport with pass/fail for each metric.
    """
    from forgespc import __version__
    from forgespc.charts import individuals_moving_range_chart, xbar_r_chart, p_chart
    from forgespc.capability import calculate_capability

    report = CalibrationReport(version=__version__)

    gdir = Path(golden_dir) if golden_dir else _find_golden_dir()
    if gdir is None:
        report.errors.append("Golden files not found.")
        return report

    for golden_file in sorted(gdir.glob("spc_*.json")):
        try:
            with open(golden_file) as f:
                golden = json.load(f)
        except Exception as e:
            report.errors.append(f"Failed to read {golden_file.name}: {e}")
            continue

        case_id = golden.get("case_id", golden_file.stem)
        analysis_id = golden.get("analysis_id", "")
        expected = golden.get("expected", {})
        data = golden.get("data")
        config = golden.get("config", {})

        if data is None:
            report.errors.append(f"{case_id}: no embedded data, skipping")
            continue

        try:
            if analysis_id == "imr":
                result = individuals_moving_range_chart(data)
                _check(report, case_id, expected, "statistics.grand_mean", result.limits.cl)
                _check(report, case_id, expected, "statistics.ucl", result.limits.ucl)
                _check(report, case_id, expected, "statistics.lcl", result.limits.lcl)
                _check(report, case_id, expected, "statistics.n_ooc", float(len(result.out_of_control)))

            elif analysis_id == "xbar_r":
                result = xbar_r_chart(data)
                _check(report, case_id, expected, "statistics.grand_mean", result.limits.cl)
                _check(report, case_id, expected, "statistics.ucl", result.limits.ucl)
                if "statistics.n_ooc" in expected:
                    _check(report, case_id, expected, "statistics.n_ooc", float(len(result.out_of_control)))

            elif analysis_id == "capability":
                usl = config.get("usl", 56.0)
                lsl = config.get("lsl", 44.0)
                cap = calculate_capability(data, usl=usl, lsl=lsl)
                _check(report, case_id, expected, "statistics.cp", cap.cp)
                _check(report, case_id, expected, "statistics.cpk", cap.cpk)
                if "statistics.sigma_level" in expected:
                    _check(report, case_id, expected, "statistics.sigma_level", cap.sigma_level)

            elif analysis_id == "p_chart":
                sample_size = config.get("sample_size", 100)
                result = p_chart(data, sample_sizes=[sample_size] * len(data))
                _check(report, case_id, expected, "statistics.p_bar", result.limits.cl)
                if "statistics.ucl" in expected:
                    _check(report, case_id, expected, "statistics.ucl", result.limits.ucl)
                if "statistics.n_ooc" in expected:
                    _check(report, case_id, expected, "statistics.n_ooc", float(len(result.out_of_control)))

            else:
                report.errors.append(f"{case_id}: unsupported analysis_id '{analysis_id}'")

        except Exception as e:
            report.errors.append(f"{case_id}: {e}")

    return report


def _check(report: CalibrationReport, case_id: str, expected: dict, key: str, actual: float):
    """Check a single metric against its golden reference."""
    if key not in expected:
        return

    exp = expected[key]
    if isinstance(exp, dict):
        ref_value = exp["value"]
        tolerance = exp["tolerance"]
    else:
        ref_value = float(exp)
        tolerance = abs(ref_value * 0.05)

    passed = abs(actual - ref_value) <= tolerance
    check = CalibrationCheck(
        case_id=case_id,
        metric=key,
        expected=ref_value,
        actual=actual,
        tolerance=tolerance,
        passed=passed,
    )

    report.total_checks += 1
    report.checks.append(check)
    if passed:
        report.passed_checks += 1
    else:
        report.failures.append(check)
