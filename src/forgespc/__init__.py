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
