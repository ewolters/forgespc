"""ForgeSPC — extracted from SVEND SPC engine.

Statistical Process Control computations. Pure Python stdlib.
No web framework, no database, no I/O dependencies.
"""

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Literal, Optional

from .constants import CONTROL_CHART_CONSTANTS, IMR_CONSTANTS
from .models import ControlChartResult, ControlLimits, StatisticalSummary
from .rules import check_nelson_rules, check_western_electric_rules


def calculate_summary(data: list[float]) -> StatisticalSummary:
    """Calculate comprehensive statistical summary."""
    n = len(data)
    if n < 2:
        raise ValueError("Need at least 2 data points")

    sorted_data = sorted(data)
    mean = statistics.mean(data)

    # Quartiles
    q1_idx = n // 4
    q3_idx = (3 * n) // 4
    q1 = sorted_data[q1_idx]
    q3 = sorted_data[q3_idx]

    # Variance and std dev
    variance = statistics.variance(data)
    std_dev = math.sqrt(variance)

    # Skewness (Fisher's)
    if std_dev > 0:
        skewness = sum((x - mean) ** 3 for x in data) / (n * std_dev**3)
    else:
        skewness = 0.0

    # Kurtosis (excess)
    if std_dev > 0:
        kurtosis = sum((x - mean) ** 4 for x in data) / (n * std_dev**4) - 3
    else:
        kurtosis = 0.0

    return StatisticalSummary(
        n=n,
        mean=mean,
        median=statistics.median(data),
        std_dev=std_dev,
        variance=variance,
        min_val=min(data),
        max_val=max(data),
        range_val=max(data) - min(data),
        q1=q1,
        q3=q3,
        iqr=q3 - q1,
        skewness=skewness,
        kurtosis=kurtosis,
    )


def z_to_dpmo(z: float) -> float:
    """Convert Z score to DPMO (using 1.5 sigma shift)."""
    # Approximate using normal CDF
    # P(defect) = 1 - Phi(z) for upper tail
    # With 1.5 sigma shift: use z - 1.5
    z_shifted = z - 1.5

    # Approximate normal CDF using error function approximation
    def norm_cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    # Two-tailed defect rate
    defect_rate = 1 - norm_cdf(z_shifted) + norm_cdf(-z_shifted - 3)  # Simplified
    defect_rate = max(0, min(1, 1 - norm_cdf(z_shifted)))  # Upper tail only for simplicity

    return defect_rate * 1_000_000


def dpmo_to_sigma(dpmo: float) -> float:
    """Convert DPMO to sigma level (with 1.5 shift)."""
    if dpmo <= 0:
        return 6.0  # Perfect
    if dpmo >= 1_000_000:
        return 0.0

    # Inverse normal approximation
    # Using Beasley-Springer-Moro algorithm approximation
    p = 1 - dpmo / 1_000_000

    if p <= 0:
        return 0.0
    if p >= 1:
        return 6.0

    # Approximate inverse normal
    a = [
        0,
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        0,
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        0,
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        0,
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        z = (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) / (
            (((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        z = (
            (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6])
            * q
            / (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1)
        )
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        z = -(((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) / (
            (((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1
        )

    # Add 1.5 sigma shift
    return z + 1.5


# =============================================================================
# Control Charts
# =============================================================================


def individuals_moving_range_chart(
    data: list[float],
    usl: float | None = None,
    lsl: float | None = None,
    historical_mean: float | None = None,
    historical_sigma: float | None = None,
) -> ControlChartResult:
    """
    Create I-MR (Individuals and Moving Range) control chart.

    Used for continuous data with subgroup size = 1.

    If historical_mean and/or historical_sigma are provided, they override
    the values calculated from data. This enables Phase 2 monitoring where
    limits are locked to a baseline period.
    """
    n = len(data)
    if n < 2:
        raise ValueError("Need at least 2 data points")

    # Calculate moving ranges
    moving_ranges = [abs(data[i] - data[i - 1]) for i in range(1, n)]

    # Center lines — use historical if provided
    x_bar = historical_mean if historical_mean is not None else statistics.mean(data)
    mr_bar = statistics.mean(moving_ranges)

    # Estimate sigma — use historical if provided
    sigma = historical_sigma if historical_sigma is not None else mr_bar / IMR_CONSTANTS["d2"]

    # Control limits for Individuals chart
    i_ucl = x_bar + 3 * sigma
    i_lcl = x_bar - 3 * sigma

    # Control limits for MR chart
    mr_ucl = IMR_CONSTANTS["D4"] * mr_bar
    mr_lcl = IMR_CONSTANTS["D3"] * mr_bar

    # Find out of control points (Individuals)
    out_of_control = []
    for i, val in enumerate(data):
        if val > i_ucl:
            out_of_control.append({"index": i, "value": val, "reason": "Above UCL"})
        elif val < i_lcl:
            out_of_control.append({"index": i, "value": val, "reason": "Below LCL"})

    # Check Western Electric rules
    run_violations = check_western_electric_rules(data, x_bar, sigma)

    in_control = len(out_of_control) == 0 and len(run_violations) == 0

    # Create MR chart result
    mr_out_of_control = []
    for i, val in enumerate(moving_ranges):
        if val > mr_ucl:
            mr_out_of_control.append({"index": i + 1, "value": val, "reason": "Above UCL"})

    mr_chart = ControlChartResult(
        chart_type="MR",
        data_points=moving_ranges,
        limits=ControlLimits(ucl=mr_ucl, cl=mr_bar, lcl=mr_lcl),
        out_of_control=mr_out_of_control,
        run_violations=[],
        in_control=len(mr_out_of_control) == 0,
        summary=f"MR Chart: Mean={mr_bar:.4f}, UCL={mr_ucl:.4f}",
    )

    using_historical = historical_mean is not None or historical_sigma is not None
    summary_parts = [
        f"I-MR Chart Analysis (n={n}){' [Historical limits]' if using_historical else ''}",
        f"Process Mean: {x_bar:.4f}{' (historical)' if historical_mean is not None else ''}",
        f"Sigma: {sigma:.4f}{' (historical)' if historical_sigma is not None else ' (estimated)'}",
        f"UCL: {i_ucl:.4f}, LCL: {i_lcl:.4f}",
    ]
    if out_of_control:
        summary_parts.append(f"Out of control: {len(out_of_control)} points")
    if run_violations:
        summary_parts.append(f"Run rule violations: {len(run_violations)}")
    if in_control:
        summary_parts.append("Process is IN CONTROL")
    else:
        summary_parts.append("Process is OUT OF CONTROL")

    return ControlChartResult(
        chart_type="I-MR",
        data_points=data,
        limits=ControlLimits(ucl=i_ucl, cl=x_bar, lcl=i_lcl, usl=usl, lsl=lsl),
        out_of_control=out_of_control,
        run_violations=run_violations,
        in_control=in_control,
        summary="\n".join(summary_parts),
        secondary_chart=mr_chart,
    )


def xbar_r_chart(
    subgroups: list[list[float]],
    usl: float | None = None,
    lsl: float | None = None,
    historical_mean: float | None = None,
    historical_sigma: float | None = None,
) -> ControlChartResult:
    """
    Create X-bar and R control chart.

    Used for continuous data with subgroups of size 2-10.

    If historical_mean and/or historical_sigma are provided, control limits
    use those instead of values computed from data. This enables Phase 2
    monitoring where limits are locked to a baseline period.
    """
    # Validate subgroups
    n_subgroups = len(subgroups)
    if n_subgroups < 2:
        raise ValueError("Need at least 2 subgroups")

    subgroup_size = len(subgroups[0])
    if not all(len(sg) == subgroup_size for sg in subgroups):
        raise ValueError("All subgroups must have the same size")

    if subgroup_size < 2 or subgroup_size > 10:
        raise ValueError("Subgroup size must be between 2 and 10 for X-bar R chart")

    constants = CONTROL_CHART_CONSTANTS[subgroup_size]

    # Calculate subgroup means and ranges
    subgroup_means = [statistics.mean(sg) for sg in subgroups]
    subgroup_ranges = [max(sg) - min(sg) for sg in subgroups]

    # Grand mean and average range — use historical if provided
    x_bar_bar = historical_mean if historical_mean is not None else statistics.mean(subgroup_means)
    r_bar = statistics.mean(subgroup_ranges)

    # Estimate sigma — use historical if provided
    sigma = historical_sigma if historical_sigma is not None else r_bar / constants["d2"]

    # Control limits for X-bar chart (use sigma-based if historical provided)
    if historical_sigma is not None:
        xbar_ucl = x_bar_bar + 3 * sigma / math.sqrt(subgroup_size)
        xbar_lcl = x_bar_bar - 3 * sigma / math.sqrt(subgroup_size)
    else:
        xbar_ucl = x_bar_bar + constants["A2"] * r_bar
        xbar_lcl = x_bar_bar - constants["A2"] * r_bar

    # Control limits for R chart
    r_ucl = constants["D4"] * r_bar
    r_lcl = constants["D3"] * r_bar

    # Find out of control points (X-bar)
    out_of_control = []
    for i, val in enumerate(subgroup_means):
        if val > xbar_ucl:
            out_of_control.append({"index": i, "value": val, "reason": "Above UCL"})
        elif val < xbar_lcl:
            out_of_control.append({"index": i, "value": val, "reason": "Below LCL"})

    # Check run rules
    run_violations = check_western_electric_rules(subgroup_means, x_bar_bar, sigma / math.sqrt(subgroup_size))

    # R chart out of control
    r_out_of_control = []
    for i, val in enumerate(subgroup_ranges):
        if val > r_ucl:
            r_out_of_control.append({"index": i, "value": val, "reason": "Above UCL"})
        elif val < r_lcl:
            r_out_of_control.append({"index": i, "value": val, "reason": "Below LCL"})

    r_chart = ControlChartResult(
        chart_type="R",
        data_points=subgroup_ranges,
        limits=ControlLimits(ucl=r_ucl, cl=r_bar, lcl=r_lcl),
        out_of_control=r_out_of_control,
        run_violations=[],
        in_control=len(r_out_of_control) == 0,
        summary=f"R Chart: R-bar={r_bar:.4f}, UCL={r_ucl:.4f}",
    )

    in_control = len(out_of_control) == 0 and len(run_violations) == 0 and len(r_out_of_control) == 0

    using_historical = historical_mean is not None or historical_sigma is not None
    summary_parts = [
        f"X-bar R Chart Analysis (k={n_subgroups}, n={subgroup_size}){' [Historical limits]' if using_historical else ''}",
        f"Grand Mean (X-bar-bar): {x_bar_bar:.4f}{' (historical)' if historical_mean is not None else ''}",
        f"Average Range (R-bar): {r_bar:.4f}",
        f"Sigma: {sigma:.4f}{' (historical)' if historical_sigma is not None else ' (estimated)'}",
        f"X-bar UCL: {xbar_ucl:.4f}, LCL: {xbar_lcl:.4f}",
    ]
    if in_control:
        summary_parts.append("Process is IN CONTROL")
    else:
        summary_parts.append("Process is OUT OF CONTROL")

    return ControlChartResult(
        chart_type="X-bar R",
        data_points=subgroup_means,
        limits=ControlLimits(ucl=xbar_ucl, cl=x_bar_bar, lcl=xbar_lcl, usl=usl, lsl=lsl),
        out_of_control=out_of_control,
        run_violations=run_violations,
        in_control=in_control,
        summary="\n".join(summary_parts),
        secondary_chart=r_chart,
    )


def p_chart(
    defectives: list[int],
    sample_sizes: list[int],
) -> ControlChartResult:
    """
    Create p-chart for proportion defective.

    Used for attribute data (pass/fail, defective/non-defective).
    """
    if len(defectives) != len(sample_sizes):
        raise ValueError("defectives and sample_sizes must have same length")

    n_samples = len(defectives)

    # Calculate proportions
    proportions = [d / n for d, n in zip(defectives, sample_sizes)]

    # Average proportion and sample size
    total_defectives = sum(defectives)
    total_inspected = sum(sample_sizes)
    p_bar = total_defectives / total_inspected
    n_bar = total_inspected / n_samples

    # Control limits (can vary by sample size, use average for simplicity)
    sigma_p = math.sqrt(p_bar * (1 - p_bar) / n_bar)
    ucl = p_bar + 3 * sigma_p
    lcl = max(0, p_bar - 3 * sigma_p)

    # Find out of control points
    out_of_control = []
    for i, (p, n) in enumerate(zip(proportions, sample_sizes)):
        # Use exact limits for each sample
        sigma_i = math.sqrt(p_bar * (1 - p_bar) / n)
        ucl_i = p_bar + 3 * sigma_i
        lcl_i = max(0, p_bar - 3 * sigma_i)

        if p > ucl_i:
            out_of_control.append({"index": i, "value": p, "reason": "Above UCL"})
        elif p < lcl_i:
            out_of_control.append({"index": i, "value": p, "reason": "Below LCL"})

    in_control = len(out_of_control) == 0

    summary_parts = [
        f"p-Chart Analysis (k={n_samples})",
        f"Average Proportion Defective (p-bar): {p_bar:.4f} ({p_bar * 100:.2f}%)",
        f"Total Defectives: {total_defectives} / {total_inspected}",
        f"UCL: {ucl:.4f}, LCL: {lcl:.4f}",
    ]
    if in_control:
        summary_parts.append("Process is IN CONTROL")
    else:
        summary_parts.append(f"Process is OUT OF CONTROL ({len(out_of_control)} points)")

    return ControlChartResult(
        chart_type="p",
        data_points=proportions,
        limits=ControlLimits(ucl=ucl, cl=p_bar, lcl=lcl),
        out_of_control=out_of_control,
        run_violations=[],
        in_control=in_control,
        summary="\n".join(summary_parts),
    )


def c_chart(
    defect_counts: list[int],
) -> ControlChartResult:
    """
    Create c-chart for count of defects per unit.

    Used when counting defects in same-sized units.
    """
    n_samples = len(defect_counts)

    # Average defect count
    c_bar = statistics.mean(defect_counts)

    # Control limits (Poisson-based)
    sigma_c = math.sqrt(c_bar)
    ucl = c_bar + 3 * sigma_c
    lcl = max(0, c_bar - 3 * sigma_c)

    # Find out of control points
    out_of_control = []
    for i, c in enumerate(defect_counts):
        if c > ucl:
            out_of_control.append({"index": i, "value": c, "reason": "Above UCL"})
        elif c < lcl:
            out_of_control.append({"index": i, "value": c, "reason": "Below LCL"})

    in_control = len(out_of_control) == 0

    summary_parts = [
        f"c-Chart Analysis (k={n_samples})",
        f"Average Defects (c-bar): {c_bar:.2f}",
        f"UCL: {ucl:.2f}, LCL: {lcl:.2f}",
    ]
    if in_control:
        summary_parts.append("Process is IN CONTROL")
    else:
        summary_parts.append(f"Process is OUT OF CONTROL ({len(out_of_control)} points)")

    return ControlChartResult(
        chart_type="c",
        data_points=[float(c) for c in defect_counts],
        limits=ControlLimits(ucl=ucl, cl=c_bar, lcl=lcl),
        out_of_control=out_of_control,
        run_violations=[],
        in_control=in_control,
        summary="\n".join(summary_parts),
    )


