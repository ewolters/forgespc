"""ForgeSPC — extracted from SVEND SPC engine.

Statistical Process Control computations. Pure Python stdlib.
No web framework, no database, no I/O dependencies.
"""

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Literal, Optional

from .constants import CONTROL_CHART_CONSTANTS, IMR_CONSTANTS
from .models import ProcessCapability


def calculate_capability(
    data: list[float],
    usl: float,
    lsl: float,
    target: float | None = None,
    subgroup_size: int = 1,
) -> ProcessCapability:
    """
    Calculate process capability indices.

    Args:
        data: Measurement data (flat list)
        usl: Upper specification limit
        lsl: Lower specification limit
        target: Target value (defaults to midpoint of specs)
        subgroup_size: Size of rational subgroups for within-group sigma estimate
    """
    n = len(data)
    if n < 2:
        raise ValueError("Need at least 2 data points")

    if usl <= lsl:
        raise ValueError("USL must be greater than LSL")

    if target is None:
        target = (usl + lsl) / 2

    mean = statistics.mean(data)

    # Overall standard deviation
    sigma_overall = statistics.stdev(data)

    # Within-subgroup standard deviation estimate
    if subgroup_size == 1:
        # Use moving range method
        moving_ranges = [abs(data[i] - data[i - 1]) for i in range(1, n)]
        mr_bar = statistics.mean(moving_ranges)
        sigma_within = mr_bar / IMR_CONSTANTS["d2"]
    else:
        # Use pooled within-subgroup variance
        # Reshape data into subgroups
        n_subgroups = n // subgroup_size
        subgroups = [data[i * subgroup_size : (i + 1) * subgroup_size] for i in range(n_subgroups)]

        if subgroup_size in CONTROL_CHART_CONSTANTS:
            # Use R-bar method
            ranges = [max(sg) - min(sg) for sg in subgroups]
            r_bar = statistics.mean(ranges)
            sigma_within = r_bar / CONTROL_CHART_CONSTANTS[subgroup_size]["d2"]
        else:
            # Use pooled std dev
            within_vars = [statistics.variance(sg) for sg in subgroups if len(sg) > 1]
            sigma_within = math.sqrt(statistics.mean(within_vars)) if within_vars else sigma_overall

    # Specification width
    spec_width = usl - lsl

    # Short-term capability (Cp, Cpk)
    cp = spec_width / (6 * sigma_within) if sigma_within > 0 else 0
    cpu = (usl - mean) / (3 * sigma_within) if sigma_within > 0 else 0
    cpl = (mean - lsl) / (3 * sigma_within) if sigma_within > 0 else 0
    cpk = min(cpu, cpl)

    # Long-term performance (Pp, Ppk)
    pp = spec_width / (6 * sigma_overall) if sigma_overall > 0 else 0
    ppu = (usl - mean) / (3 * sigma_overall) if sigma_overall > 0 else 0
    ppl = (mean - lsl) / (3 * sigma_overall) if sigma_overall > 0 else 0
    ppk = min(ppu, ppl)

    # Sigma level (based on Cpk)
    sigma_level = 3 * cpk if cpk > 0 else 0

    # DPMO calculation
    # Using normal distribution approximation
    def norm_cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    z_upper = (usl - mean) / sigma_overall if sigma_overall > 0 else float("inf")
    z_lower = (mean - lsl) / sigma_overall if sigma_overall > 0 else float("inf")

    p_above_usl = 1 - norm_cdf(z_upper)
    p_below_lsl = norm_cdf(-z_lower)
    total_defect_rate = p_above_usl + p_below_lsl

    dpmo = total_defect_rate * 1_000_000
    yield_percent = (1 - total_defect_rate) * 100

    # Interpretation
    if cpk >= 2.0:
        interpretation = "Excellent: World-class capability (Six Sigma level)"
    elif cpk >= 1.67:
        interpretation = "Very Good: Capable process with good margin"
    elif cpk >= 1.33:
        interpretation = "Good: Process is capable"
    elif cpk >= 1.0:
        interpretation = "Marginal: Process barely meets specs, improvement needed"
    elif cpk >= 0.67:
        interpretation = "Poor: Process produces significant defects"
    else:
        interpretation = "Very Poor: Process is not capable, major improvement needed"

    return ProcessCapability(
        cp=cp,
        cpk=cpk,
        cpu=cpu,
        cpl=cpl,
        pp=pp,
        ppk=ppk,
        ppu=ppu,
        ppl=ppl,
        sigma_within=sigma_within,
        sigma_overall=sigma_overall,
        sigma_level=sigma_level,
        dpmo=dpmo,
        yield_percent=yield_percent,
        usl=usl,
        lsl=lsl,
        target=target,
        mean=mean,
        n_samples=n,
        interpretation=interpretation,
    )


# =============================================================================
# Helper Functions for API
# =============================================================================


def recommend_chart_type(
    data_type: Literal["continuous", "attribute"],
    subgroup_size: int = 1,
    attribute_type: Literal["defectives", "defects"] | None = None,
) -> str:
    """Recommend appropriate control chart type."""
    if data_type == "continuous":
        if subgroup_size == 1:
            return "I-MR"
        elif subgroup_size <= 10:
            return "X-bar R"
        else:
            return "X-bar S"
    else:  # attribute
        if attribute_type == "defectives":
            return "p" if subgroup_size > 1 else "np"
        else:  # defects
            return "c" if subgroup_size == 1 else "u"


