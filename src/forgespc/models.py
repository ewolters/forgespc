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

    @property
    def n(self) -> int:
        return len(self.data_points)

    def to_render(self):
        """Emit a theme-neutral forgerender.ChartSpec for this control chart.

        Semantic roles (not literal colors) carry meaning; the consumer's theme
        maps role -> color. Depends only on forgerender, never on forgeviz.
        """
        from forgerender import (
            ROLE_CENTERLINE,
            ROLE_CONTROL_LIMIT,
            ROLE_DATA,
            ROLE_OUT_OF_CONTROL,
            ROLE_RUN_RULE,
            ROLE_SIGMA_ZONE,
            ROLE_SPEC_LIMIT,
            ChartSpec,
        )

        lim = self.limits
        x = list(range(1, len(self.data_points) + 1))
        spec = ChartSpec(
            title="Control Chart",
            subtitle=self.chart_type,
            chart_type="control_chart",
            x_axis={"label": "Sample", "grid": True},
            y_axis={"label": "Value", "grid": True},
        )
        spec.add_trace(x, self.data_points, name="Data", trace_type="line", color="", role=ROLE_DATA)

        ooc_idx = [p["index"] for p in self.out_of_control]
        if ooc_idx:
            spec.add_marker(ooc_idx, color="", label="Out of Control", role=ROLE_OUT_OF_CONTROL)

        spec.add_reference_line(lim.ucl, color="", label="UCL", role=ROLE_CONTROL_LIMIT)
        spec.add_reference_line(lim.cl, color="", dash="solid", label="CL", role=ROLE_CENTERLINE)
        spec.add_reference_line(lim.lcl, color="", label="LCL", role=ROLE_CONTROL_LIMIT)
        if lim.usl is not None:
            spec.add_reference_line(lim.usl, color="", dash="dotted", label="USL", role=ROLE_SPEC_LIMIT)
        if lim.lsl is not None:
            spec.add_reference_line(lim.lsl, color="", dash="dotted", label="LSL", role=ROLE_SPEC_LIMIT)

        if lim.ucl > lim.cl > lim.lcl:
            one_sigma = (lim.ucl - lim.cl) / 3
            spec.add_zone(lim.cl + 2 * one_sigma, lim.ucl, color="", role=ROLE_SIGMA_ZONE)
            spec.add_zone(lim.cl + one_sigma, lim.cl + 2 * one_sigma, color="", role=ROLE_SIGMA_ZONE)
            spec.add_zone(lim.cl - 2 * one_sigma, lim.cl - one_sigma, color="", role=ROLE_SIGMA_ZONE)
            spec.add_zone(lim.lcl, lim.cl - 2 * one_sigma, color="", role=ROLE_SIGMA_ZONE)

        run_idx = sorted({i for v in self.run_violations for i in v.get("indices", [])} - set(ooc_idx))
        if run_idx:
            spec.add_marker(run_idx, color="", symbol="diamond", label="Run Rule", role=ROLE_RUN_RULE)

        return spec

    @property
    def n_ooc(self) -> int:
        return len(self.out_of_control)

    @property
    def n_run_violations(self) -> int:
        return len(self.run_violations)

    @property
    def ucl(self) -> float:
        return self.limits.ucl

    @property
    def cl(self) -> float:
        return self.limits.cl

    @property
    def lcl(self) -> float:
        return self.limits.lcl

    def to_dict(self) -> dict:
        result = asdict(self)
        if self.secondary_chart:
            result["secondary_chart"] = self.secondary_chart.to_dict()
        # Add computed fields for forgenarr template compatibility
        result["n"] = self.n
        result["n_ooc"] = self.n_ooc
        result["n_run_violations"] = self.n_run_violations
        result["ucl"] = self.ucl
        result["cl"] = self.cl
        result["lcl"] = self.lcl
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

    # Defect metrics (within-subgroup sigma, consistent with Cp/Cpk)
    dpmo: float  # Defects per million opportunities (within sigma)
    yield_percent: float  # Process yield % (within sigma)

    # Specs
    usl: float
    lsl: float

    # Data summary
    mean: float
    n_samples: int

    # Interpretation
    interpretation: str

    # Optional
    target: float | None = None
    dpmo_overall: float | None = None  # DPMO using overall sigma (Pp/Ppk basis)
    yield_overall: float | None = None  # Yield using overall sigma
    bayesian_shadow: dict | None = None  # Bayesian capability results if requested

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


