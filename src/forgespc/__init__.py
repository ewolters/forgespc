"""ForgeSPC — Statistical Process Control engine for manufacturing.

Pure computation library. No web framework, no database, no I/O.
Takes numbers in, returns analysis results out.

Usage:
    from forgespc.charts import individuals_moving_range_chart, xbar_r_chart
    from forgespc.capability import calculate_capability
    from forgespc.rules import check_nelson_rules
    from forgespc.constants import CONTROL_CHART_CONSTANTS

    result = individuals_moving_range_chart(data)
    cap = calculate_capability(data, usl=25.02, lsl=24.98)

Advanced (requires numpy):
    from forgespc.gage import gage_rr_crossed, hotelling_t_squared_chart
"""

__version__ = "0.1.0"

__all__ = [
    # models
    "ControlLimits",
    "ControlChartResult",
    "ProcessCapability",
    "StatisticalSummary",
    # constants
    "CONTROL_CHART_CONSTANTS",
    # charts
    "individuals_moving_range_chart",
    "xbar_r_chart",
    "p_chart",
    "c_chart",
    "u_chart",
    "np_chart",
    "laney_p_chart",
    "laney_u_chart",
    "moving_average_chart",
    "zone_chart",
    # capability
    "calculate_capability",
    "degradation_capability",
    # rules
    "check_nelson_rules",
    "check_western_electric_rules",
    # advanced
    "cusum_chart",
    "ewma_chart",
    "xbar_s_chart",
    "mewma_chart",
    "generalized_variance_chart",
    # gage
    "gage_rr_crossed",
    "gage_rr_nested",
    "attribute_agreement",
    "hotelling_t_squared_chart",
    # bayesian
    "bayesian_capability",
    "bayesian_changepoint",
    "bayesian_control_chart",
    # conformal
    "conformal_control",
    "entropy_spc",
    # calibration
    "calibrate",
]
