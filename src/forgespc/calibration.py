"""Self-calibration service for ForgeSPC.

Any installation can verify its own accuracy by running calibration
against golden reference values. Returns a structured report.

Usage:
    from forgespc.calibration import calibrate

    report = calibrate()
    print(f"Pass rate: {report.pass_rate:.0%}")
    print(f"Failures: {report.failures}")
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CalibrationCheck:
    """A single calibration check result."""

    case_id: str
    metric: str
    expected: float
    actual: float
    tolerance: float
    passed: bool
    description: str = ""


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
    Path(__file__).parent / "golden",  # Shipped with package
    Path(__file__).parent.parent.parent.parent / "tests" / "golden",  # Dev layout
]


def _find_golden_dir() -> Path | None:
    for d in _GOLDEN_DIRS:
        if d.exists() and list(d.glob("spc_*.json")):
            return d
    return None


def _generate_normal(n: int, mean: float, sigma: float, seed: int = 42) -> list[float]:
    random.seed(seed)
    return [mean + random.gauss(0, sigma) for _ in range(n)]


def calibrate(golden_dir: str | Path | None = None) -> CalibrationReport:
    """Run self-calibration against golden reference files.

    Args:
        golden_dir: Path to golden files. Auto-detected if not provided.

    Returns:
        CalibrationReport with pass/fail for each metric.
    """
    from forgespc import __version__
    from forgespc.charts import individuals_moving_range_chart, xbar_r_chart
    from forgespc.capability import calculate_capability

    report = CalibrationReport(version=__version__)

    gdir = Path(golden_dir) if golden_dir else _find_golden_dir()
    if gdir is None:
        report.errors.append("Golden files not found. Install with tests or provide golden_dir.")
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

        # Generate data based on analysis type
        try:
            if analysis_id == "imr":
                data = _generate_normal(50, mean=50, sigma=2)
                result = individuals_moving_range_chart(data)
                _check_metric(report, case_id, expected, "statistics.grand_mean", result.limits.cl)
                _check_metric(report, case_id, expected, "statistics.ucl", result.limits.ucl)
                _check_metric(report, case_id, expected, "statistics.lcl", result.limits.lcl)
                _check_metric(report, case_id, expected, "statistics.n_ooc", float(len(result.out_of_control)))

            elif analysis_id == "xbar_r":
                random.seed(42)
                subgroups = [[50 + random.gauss(0, 2) for _ in range(5)] for _ in range(25)]
                result = xbar_r_chart(subgroups)
                _check_metric(report, case_id, expected, "statistics.grand_mean", result.limits.cl)
                _check_metric(report, case_id, expected, "statistics.ucl", result.limits.ucl)

            elif analysis_id == "capability":
                data = _generate_normal(100, mean=50, sigma=2)
                cap = calculate_capability(data, usl=56.0, lsl=44.0)
                _check_metric(report, case_id, expected, "statistics.cp", cap.cp)
                _check_metric(report, case_id, expected, "statistics.cpk", cap.cpk)
                _check_metric(report, case_id, expected, "statistics.sigma_level", cap.sigma_level)

        except Exception as e:
            report.errors.append(f"{case_id}: {e}")

    return report


def _check_metric(report: CalibrationReport, case_id: str, expected: dict, key: str, actual: float):
    """Check a single metric against its golden reference."""
    if key not in expected:
        return

    exp = expected[key]
    if isinstance(exp, dict):
        ref_value = exp["value"]
        tolerance = exp["tolerance"]
    else:
        ref_value = exp
        tolerance = abs(ref_value * 0.1)  # 10% default tolerance

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
