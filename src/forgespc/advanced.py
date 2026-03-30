"""Advanced SPC charts — CUSUM, EWMA, X-bar/S.

These charts require numpy for efficient computation.
Install with: pip install forgespc[advanced]

CUSUM and EWMA are sensitive to small sustained shifts that
Shewhart charts miss. X-bar/S is preferred over X-bar/R for
subgroup sizes > 10.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import numpy as np
except ImportError:
    raise ImportError("numpy is required for advanced SPC charts. Install with: pip install forgespc[advanced]")

from .models import ControlChartResult, ControlLimits


# =============================================================================
# CUSUM — Cumulative Sum Chart
# =============================================================================


@dataclass
class CUSUMResult:
    """CUSUM chart result with full state for visualization."""

    cusum_pos: list[float]
    cusum_neg: list[float]
    signals_up: list[int]
    signals_down: list[int]
    target: float
    sigma: float
    k: float
    h: float
    n: int
    in_control: bool

    @property
    def n_signals(self) -> int:
        return len(self.signals_up) + len(self.signals_down)

    def to_chart_result(self) -> ControlChartResult:
        """Convert to standard ControlChartResult."""
        ooc = [
            {"index": int(i), "value": float(self.cusum_pos[i]), "reason": "CUSUM+ > h"}
            for i in self.signals_up
        ] + [
            {"index": int(i), "value": float(self.cusum_neg[i]), "reason": "CUSUM- > h"}
            for i in self.signals_down
        ]
        return ControlChartResult(
            chart_type="CUSUM",
            data_points=self.cusum_pos,
            limits=ControlLimits(ucl=self.h, cl=0.0, lcl=-self.h),
            out_of_control=ooc,
            run_violations=[],
            in_control=self.in_control,
            summary=f"CUSUM: {self.n_signals} signal(s), target={self.target:.4f}, k={self.k}, h={self.h}",
        )


def cusum_chart(
    data: list[float],
    target: float | None = None,
    k: float = 0.5,
    h: float = 5.0,
) -> CUSUMResult:
    """Compute CUSUM chart for detecting small sustained shifts.

    Args:
        data: Individual measurements.
        target: Target value. Defaults to mean of data.
        k: Slack value (allowance). Typically 0.5 sigma.
        h: Decision interval. Typically 4-5 sigma.

    Returns:
        CUSUMResult with full state for visualization.
    """
    arr = np.array(data, dtype=float)
    n = len(arr)

    if target is None:
        target = float(np.mean(arr))

    sigma = float(np.std(arr, ddof=1))
    if sigma == 0:
        sigma = 1.0

    z = (arr - target) / sigma

    cusum_pos = np.zeros(n)
    cusum_neg = np.zeros(n)

    for i in range(n):
        cusum_pos[i] = max(0, (cusum_pos[i - 1] if i > 0 else 0) + z[i] - k)
        cusum_neg[i] = max(0, (cusum_neg[i - 1] if i > 0 else 0) - z[i] - k)

    signals_up = np.where(cusum_pos > h)[0].tolist()
    signals_down = np.where(cusum_neg > h)[0].tolist()

    return CUSUMResult(
        cusum_pos=cusum_pos.tolist(),
        cusum_neg=cusum_neg.tolist(),
        signals_up=signals_up,
        signals_down=signals_down,
        target=target,
        sigma=sigma,
        k=k,
        h=h,
        n=n,
        in_control=(len(signals_up) + len(signals_down)) == 0,
    )


# =============================================================================
# EWMA — Exponentially Weighted Moving Average Chart
# =============================================================================


@dataclass
class EWMAResult:
    """EWMA chart result with full state for visualization."""

    ewma_values: list[float]
    ucl: list[float]  # Time-varying upper control limit
    lcl: list[float]  # Time-varying lower control limit
    ucl_steady: float  # Steady-state UCL
    lcl_steady: float  # Steady-state LCL
    target: float
    sigma: float
    lambda_param: float
    L: float
    n: int
    out_of_control_indices: list[int]
    in_control: bool

    def to_chart_result(self) -> ControlChartResult:
        ooc = [
            {"index": int(i), "value": float(self.ewma_values[i]), "reason": "EWMA outside limits"}
            for i in self.out_of_control_indices
        ]
        return ControlChartResult(
            chart_type="EWMA",
            data_points=self.ewma_values,
            limits=ControlLimits(ucl=self.ucl_steady, cl=self.target, lcl=self.lcl_steady),
            out_of_control=ooc,
            run_violations=[],
            in_control=self.in_control,
            summary=f"EWMA: {len(ooc)} OOC, lambda={self.lambda_param}, L={self.L}",
        )


def ewma_chart(
    data: list[float],
    target: float | None = None,
    lambda_param: float = 0.2,
    L: float = 3.0,
) -> EWMAResult:
    """Compute EWMA chart for detecting small shifts with memory.

    Args:
        data: Individual measurements.
        target: Target value. Defaults to mean of data.
        lambda_param: Smoothing parameter (0 < lambda <= 1). Smaller = more memory.
        L: Control limit width in sigma units.

    Returns:
        EWMAResult with full state for visualization.
    """
    arr = np.array(data, dtype=float)
    n = len(arr)

    if target is None:
        target = float(np.mean(arr))

    sigma = float(np.std(arr, ddof=1))
    if sigma == 0:
        sigma = 1.0

    # Compute EWMA values
    ewma = np.zeros(n)
    ewma[0] = lambda_param * arr[0] + (1 - lambda_param) * target
    for i in range(1, n):
        ewma[i] = lambda_param * arr[i] + (1 - lambda_param) * ewma[i - 1]

    # Time-varying control limits (approach steady state)
    factor = lambda_param / (2 - lambda_param)
    t = np.arange(1, n + 1)
    limit_width = L * sigma * np.sqrt(factor * (1 - (1 - lambda_param) ** (2 * t)))
    ucl = target + limit_width
    lcl = target - limit_width

    # Steady-state limits
    ucl_ss = target + L * sigma * np.sqrt(factor)
    lcl_ss = target - L * sigma * np.sqrt(factor)

    # OOC detection
    ooc_indices = [i for i in range(n) if ewma[i] > ucl[i] or ewma[i] < lcl[i]]

    return EWMAResult(
        ewma_values=ewma.tolist(),
        ucl=ucl.tolist(),
        lcl=lcl.tolist(),
        ucl_steady=float(ucl_ss),
        lcl_steady=float(lcl_ss),
        target=target,
        sigma=sigma,
        lambda_param=lambda_param,
        L=L,
        n=n,
        out_of_control_indices=ooc_indices,
        in_control=len(ooc_indices) == 0,
    )


# =============================================================================
# X-bar/S Chart — preferred for subgroups > 10
# =============================================================================


def xbar_s_chart(
    subgroups: list[list[float]],
    historical_mean: float | None = None,
    historical_sigma: float | None = None,
) -> ControlChartResult:
    """X-bar/S control chart using subgroup standard deviations.

    Preferred over X-bar/R when subgroup size > 10, or when more
    precise estimation of within-subgroup variation is needed.

    Args:
        subgroups: List of subgroup data (each subgroup is a list of floats).
        historical_mean: Use historical mean instead of computed.
        historical_sigma: Use historical sigma instead of computed.

    Returns:
        ControlChartResult with secondary S chart.
    """
    from .constants import CONTROL_CHART_CONSTANTS

    subgroup_means = [float(np.mean(sg)) for sg in subgroups]
    subgroup_stds = [float(np.std(sg, ddof=1)) for sg in subgroups]
    n_per_group = len(subgroups[0])

    grand_mean = historical_mean if historical_mean is not None else float(np.mean(subgroup_means))
    s_bar = float(np.mean(subgroup_stds))

    # Get constants for subgroup size
    consts = CONTROL_CHART_CONSTANTS.get(n_per_group, CONTROL_CHART_CONSTANTS[10])
    c4 = consts["c4"]
    A3 = consts["A3"]
    B3 = consts["B3"]
    B4 = consts["B4"]

    # Estimated sigma
    sigma_within = s_bar / c4 if historical_sigma is None else historical_sigma

    # X-bar limits
    x_ucl = grand_mean + A3 * s_bar
    x_lcl = grand_mean - A3 * s_bar

    # S chart limits
    s_ucl = B4 * s_bar
    s_lcl = B3 * s_bar

    # OOC detection for X-bar
    from .rules import check_nelson_rules

    x_ooc = []
    for i, mean in enumerate(subgroup_means):
        if mean > x_ucl or mean < x_lcl:
            reason = "above UCL" if mean > x_ucl else "below LCL"
            x_ooc.append({"index": i, "value": mean, "reason": reason})

    x_violations = check_nelson_rules(subgroup_means, grand_mean, sigma_within / (n_per_group ** 0.5))

    # OOC for S chart
    s_ooc = []
    for i, s in enumerate(subgroup_stds):
        if s > s_ucl or (s_lcl > 0 and s < s_lcl):
            reason = "above UCL" if s > s_ucl else "below LCL"
            s_ooc.append({"index": i, "value": s, "reason": reason})

    s_chart = ControlChartResult(
        chart_type="S",
        data_points=subgroup_stds,
        limits=ControlLimits(ucl=s_ucl, cl=s_bar, lcl=s_lcl),
        out_of_control=s_ooc,
        run_violations=[],
        in_control=len(s_ooc) == 0,
        summary=f"S chart: s_bar={s_bar:.4f}, UCL={s_ucl:.4f}",
    )

    return ControlChartResult(
        chart_type="X-bar S",
        data_points=subgroup_means,
        limits=ControlLimits(ucl=x_ucl, cl=grand_mean, lcl=x_lcl),
        out_of_control=x_ooc,
        run_violations=x_violations,
        in_control=len(x_ooc) == 0,
        summary=f"X-bar S: mean={grand_mean:.4f}, UCL={x_ucl:.4f}, LCL={x_lcl:.4f}",
        secondary_chart=s_chart,
    )
