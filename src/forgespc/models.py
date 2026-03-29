"""Data models for SPC results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class ControlLimits:
    """Control limits for a control chart."""

    ucl: float  # Upper Control Limit
    cl: float  # Center Line
    lcl: float  # Lower Control Limit

    # Optional specification limits
    usl: float | None = None  # Upper Spec Limit
    lsl: float | None = None  # Lower Spec Limit


@dataclass
class ControlChartResult:
    """Result from control chart analysis."""

    chart_type: str
    data_points: list[float]
    limits: ControlLimits

    # Out of control points
    out_of_control: list[dict]  # [{index, value, reason}]

    # Run rules violations (Western Electric rules)
    run_violations: list[dict]  # [{rule, indices, description}]

    # Summary
    in_control: bool
    summary: str

    # For X-bar charts, also include R or S chart
    secondary_chart: Optional["ControlChartResult"] = None

    def to_dict(self) -> dict:
        result = asdict(self)
        if self.secondary_chart:
            result["secondary_chart"] = self.secondary_chart.to_dict()
        return result


@dataclass
class ProcessCapability:
    """Process capability analysis results."""

    # Short-term capability (within subgroup variation)
    cp: float  # Capability index
    cpk: float  # Capability index (centered)
    cpu: float  # Upper capability
    cpl: float  # Lower capability

    # Long-term performance (total variation)
    pp: float  # Performance index
    ppk: float  # Performance index (centered)
    ppu: float  # Upper performance
    ppl: float  # Lower performance

    # Sigma metrics
    sigma_within: float  # Within-subgroup std dev
    sigma_overall: float  # Overall std dev
    sigma_level: float  # Process sigma level (Z score)

    # Defect metrics
    dpmo: float  # Defects per million opportunities
    yield_percent: float  # Process yield %

    # Specs
    usl: float
    lsl: float
    target: float | None

    # Data summary
    mean: float
    n_samples: int

    # Interpretation
    interpretation: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StatisticalSummary:
    """Statistical summary of a dataset."""

    n: int
    mean: float
    median: float
    std_dev: float
    variance: float
    min_val: float
    max_val: float
    range_val: float
    q1: float
    q3: float
    iqr: float
    skewness: float
    kurtosis: float

    # Normality indicators
    anderson_darling: float | None = None
    is_normal: bool | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Statistical Functions
# =============================================================================


