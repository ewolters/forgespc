"""Data models for SPC results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from forgecore import ResultMixin


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
class ControlChartResult(ResultMixin):
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
        """Emit a theme-neutral forgecore.ChartSpec for this control chart.

        Semantic roles (not literal colors) carry meaning; the consumer's theme
        maps role -> color. Depends only on forgecore, never on forgeviz.
        """
        from forgecore import (
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

    def views(self):
        """The complete portrait: primary chart plus the paired secondary
        (I-MR, X-bar/R, X-bar/S) when the solver composed one."""
        specs = [self.to_render()]
        if self.secondary_chart:
            specs.append(self.secondary_chart.to_render())
        return specs

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
class ProcessCapability(ResultMixin):
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
    data: list[float] | None = None  # raw sample — views() draws from it (§5b)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def summary(self) -> str:
        return (
            f"Capability: Cp={self.cp:.2f} Cpk={self.cpk:.2f} "
            f"sigma={self.sigma_level:.2f} DPMO={self.dpmo:.0f}"
        )

    def capability(self) -> dict:
        """Capability-dialect view (forgecore.CAPABILITY vocabulary).

        Normalizes this result's field-name drift onto the shared tokens —
        e.g. sigma_level -> sigma — so any capability solver reads the same.
        """
        return {
            "usl": self.usl,
            "lsl": self.lsl,
            "cp": self.cp,
            "cpk": self.cpk,
            "sigma": self.sigma_level,
            "dpmo": self.dpmo,
        }

    def to_render(self):
        """Emit a theme-neutral forgecore.ChartSpec: fitted normal curve
        with spec limits and the process mean as semantic roles."""
        import math

        from forgecore import (
            ROLE_CENTERLINE,
            ROLE_DATA,
            ROLE_SPEC_LIMIT,
            ChartSpec,
        )

        sigma = self.sigma_overall or self.sigma_within
        lo = min(self.lsl, self.mean - 4 * sigma)
        hi = max(self.usl, self.mean + 4 * sigma)
        n = 80
        xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
        ys = [
            math.exp(-0.5 * ((x - self.mean) / sigma) ** 2)
            / (sigma * math.sqrt(2 * math.pi))
            for x in xs
        ]
        spec = ChartSpec(
            title="Process Capability",
            chart_type="capability",
            x_axis={"label": "Value", "grid": True},
            y_axis={"label": "Density", "grid": True},
        )
        spec.add_trace(xs, ys, name="Distribution", trace_type="area", color="", role=ROLE_DATA)
        spec.add_reference_line(self.mean, color="", dash="solid", label="Mean", role=ROLE_CENTERLINE)
        spec.add_reference_line(self.usl, color="", dash="dotted", label="USL", role=ROLE_SPEC_LIMIT)
        spec.add_reference_line(self.lsl, color="", dash="dotted", label="LSL", role=ROLE_SPEC_LIMIT)
        return spec

    def views(self):
        """The complete portrait: histogram + normal probability plot when the
        result carries its sample; the fitted-curve summary otherwise."""
        if not self.data:
            return [self.to_render()]
        return [self._histogram_view(), self._probability_view()]

    def _histogram_view(self):
        import math

        from forgecore import ROLE_CENTERLINE, ROLE_DATA, ROLE_SPEC_LIMIT, ChartSpec

        data = self.data
        n_bins = max(5, min(20, round(math.sqrt(len(data)))))
        lo, hi = min(data), max(data)
        width = (hi - lo) / n_bins or 1.0
        counts = [0] * n_bins
        for v in data:
            counts[min(int((v - lo) / width), n_bins - 1)] += 1
        centers = [lo + width * (i + 0.5) for i in range(n_bins)]

        spec = ChartSpec(
            title="Process Capability",
            subtitle=f"Cp={self.cp:.2f} Cpk={self.cpk:.2f}",
            chart_type="capability_histogram",
            x_axis={"label": "Value", "grid": True},
            y_axis={"label": "Count", "grid": True},
        )
        spec.add_trace(centers, counts, name="Sample", trace_type="bar", color="", role=ROLE_DATA)
        spec.add_reference_line(self.mean, color="", dash="solid", label="Mean", role=ROLE_CENTERLINE)
        spec.add_reference_line(self.usl, color="", dash="dotted", label="USL", role=ROLE_SPEC_LIMIT)
        spec.add_reference_line(self.lsl, color="", dash="dotted", label="LSL", role=ROLE_SPEC_LIMIT)
        if self.target is not None:
            spec.add_reference_line(self.target, color="", dash="dashed", label="Target", role=ROLE_CENTERLINE)
        return spec

    def _probability_view(self):
        from statistics import NormalDist

        from forgecore import ROLE_CENTERLINE, ROLE_DATA, ChartSpec

        sorted_d = sorted(self.data)
        n = len(sorted_d)
        nd = NormalDist()
        # Blom plotting positions
        theoretical = [nd.inv_cdf((i - 0.375) / (n + 0.25)) for i in range(1, n + 1)]

        spec = ChartSpec(
            title="Normal Probability Plot",
            chart_type="probability_plot",
            x_axis={"label": "Theoretical Quantiles", "grid": True},
            y_axis={"label": "Sample Quantiles", "grid": True},
        )
        spec.add_trace(theoretical, sorted_d, name="Data", trace_type="scatter", color="", role=ROLE_DATA)
        sigma = self.sigma_overall or self.sigma_within
        fit_y = [self.mean + sigma * t for t in (theoretical[0], theoretical[-1])]
        spec.add_trace([theoretical[0], theoretical[-1]], fit_y, name="Fit",
                       trace_type="line", color="", dash="dashed", role=ROLE_CENTERLINE)
        return spec


@dataclass
class StatisticalSummary(ResultMixin):
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


