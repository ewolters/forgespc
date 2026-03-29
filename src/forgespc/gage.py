"""Gage R&R and multivariate SPC. Requires numpy.

Install with: pip install forgespc[advanced]
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


def hotelling_t_squared_chart(
    data: list[list[float]],
    alpha: float = 0.05,
    usl: float | None = None,
    lsl: float | None = None,
) -> ControlChartResult:
    """
    Hotelling's T² control chart for multivariate data.

    Each row in `data` is one observation with p measured variables.
    Computes the T² statistic for each observation and sets UCL based on F-distribution.

    Args:
        data: n × p matrix (list of lists), each inner list has p variable measurements
        alpha: significance level for control limit (default 0.05)

    Returns:
        ControlChartResult with T² values, UCL, and out-of-control points
    """
    import numpy as np
    from scipy import stats as scipy_stats

    X = np.array(data, dtype=float)
    n, p = X.shape

    if n < p + 1:
        raise ValueError(f"Need at least {p + 1} observations for {p} variables, got {n}")

    # Mean vector and covariance matrix
    x_bar = X.mean(axis=0)
    S = np.cov(X, rowvar=False, ddof=1)

    # Handle singular covariance (add small ridge if needed)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)

    # T² statistic for each observation
    t2_values = []
    for i in range(n):
        diff = X[i] - x_bar
        t2_i = float(diff @ S_inv @ diff)
        t2_values.append(t2_i)

    # Phase I control limit (using beta distribution for individual T²)
    # UCL = ((n-1)^2 / n) * Beta(alpha, p/2, (n-p-1)/2)
    # Equivalent F-based: UCL = p*(n+1)*(n-1) / (n*(n-p)) * F(alpha, p, n-p)
    if n > p:
        f_crit = scipy_stats.f.ppf(1 - alpha, p, n - p)
        ucl = p * (n + 1) * (n - 1) / (n * (n - p)) * f_crit
    else:
        ucl = max(t2_values) * 1.5  # Fallback if n <= p

    # Out-of-control points
    out_of_control = []
    for i, t2 in enumerate(t2_values):
        if t2 > ucl:
            # Identify which variable(s) contribute most
            diff = X[i] - x_bar
            contributions = []
            for j in range(p):
                contrib = diff[j] ** 2 * S_inv[j, j]
                contributions.append(contrib)
            max_contrib_idx = int(np.argmax(contributions))
            out_of_control.append(
                {
                    "index": i,
                    "value": round(t2, 4),
                    "reason": f"T²={t2:.2f} > UCL={ucl:.2f} (largest contributor: var {max_contrib_idx + 1})",
                }
            )

    in_control = len(out_of_control) == 0

    # Summary
    summary_parts = [
        "Hotelling's T² Control Chart",
        f"{'=' * 50}",
        f"Observations: {n}",
        f"Variables: {p}",
        f"UCL (α={alpha}): {ucl:.4f}",
        f"Mean T²: {np.mean(t2_values):.4f}",
        f"Max T²: {np.max(t2_values):.4f}",
        "",
        f"Out of control: {len(out_of_control)} point(s)",
        f"Process status: {'IN CONTROL' if in_control else 'OUT OF CONTROL'}",
    ]

    if out_of_control:
        summary_parts.append("")
        summary_parts.append("Out-of-control observations:")
        for pt in out_of_control[:10]:
            summary_parts.append(f"  Obs {pt['index'] + 1}: {pt['reason']}")

    # Variable means and std devs
    summary_parts.append("")
    summary_parts.append("Variable Statistics:")
    for j in range(p):
        summary_parts.append(f"  Var {j + 1}: mean={x_bar[j]:.4f}, std={np.sqrt(S[j, j]):.4f}")

    # Correlation matrix
    if p <= 6:
        corr = np.corrcoef(X, rowvar=False)
        summary_parts.append("")
        summary_parts.append("Correlation Matrix:")
        header = "       " + "  ".join(f"V{j + 1:5d}" for j in range(p))
        summary_parts.append(header)
        for i in range(p):
            row = f"  V{i + 1}  " + "  ".join(f"{corr[i, j]:6.3f}" for j in range(p))
            summary_parts.append(row)

    return ControlChartResult(
        chart_type="T-squared",
        data_points=t2_values,
        limits=ControlLimits(ucl=ucl, cl=float(np.mean(t2_values)), lcl=0.0, usl=usl, lsl=lsl),
        out_of_control=out_of_control,
        run_violations=[],  # Standard run rules don't apply to T²
        in_control=in_control,
        summary="\n".join(summary_parts),
    )


# =============================================================================
# Gage R&R (Measurement System Analysis)
# =============================================================================


@dataclass
class GageRRResult:
    """Gage R&R (Crossed) study results."""

    # Variance components
    var_repeatability: float
    var_reproducibility: float
    var_operator: float
    var_interaction: float
    var_grr: float
    var_part: float
    var_total: float

    # Percentage metrics
    pct_contribution: dict  # {source: %}
    pct_study_var: dict  # {source: %}
    pct_tolerance: dict  # {source: %} (empty if no tolerance)

    # Key metrics
    ndc: int  # Number of distinct categories
    grr_percent: float  # %Study Var for GRR (headline number)
    assessment: str  # Acceptable / Marginal / Unacceptable

    # ANOVA table
    anova_table: list  # [{source, df, ss, ms, f, p}, ...]
    interaction_significant: bool
    interaction_pooled: bool

    # Study info
    n_parts: int
    n_operators: int
    n_replicates: int
    n_total: int
    tolerance: float | None

    # Plot data (Plotly JSON)
    plots: dict

    def to_dict(self) -> dict:
        return asdict(self)


def gage_rr_crossed(
    parts: list,
    operators: list,
    measurements: list[float],
    tolerance: float | None = None,
    alpha_interaction: float = 0.25,
    study_var_k: float = 6.0,
) -> GageRRResult:
    """
    Perform a crossed Gage R&R study using the ANOVA method.

    Args:
        parts: Part identifier for each measurement.
        operators: Operator identifier for each measurement.
        measurements: Measurement values.
        tolerance: Specification tolerance (USL - LSL). If provided, %Tolerance is computed.
        alpha_interaction: Significance level for interaction test. If p > alpha,
                          interaction is pooled into repeatability. Default 0.25.
        study_var_k: Multiplier for study variation (6.0 = 99.73%). Default 6.0.

    Returns:
        GageRRResult with variance components, ANOVA table, assessment, and plots.
    """
    import numpy as np
    from scipy import stats as scipy_stats

    n = len(measurements)
    if n != len(parts) or n != len(operators):
        raise ValueError("parts, operators, and measurements must have the same length")
    if n < 4:
        raise ValueError("Need at least 4 measurements")

    # Identify unique parts and operators
    unique_parts = sorted(set(parts), key=lambda x: str(x))
    unique_operators = sorted(set(operators), key=lambda x: str(x))
    p_count = len(unique_parts)
    o_count = len(unique_operators)

    if p_count < 2:
        raise ValueError("Need at least 2 parts")
    if o_count < 2:
        raise ValueError("Need at least 2 operators")

    # Build cell structure: cells[part_idx][operator_idx] = [measurements]
    part_idx = {v: i for i, v in enumerate(unique_parts)}
    op_idx = {v: i for i, v in enumerate(unique_operators)}

    cells = [[[] for _ in range(o_count)] for _ in range(p_count)]
    for i in range(n):
        pi = part_idx[parts[i]]
        oi = op_idx[operators[i]]
        cells[pi][oi].append(measurements[i])

    # Check balanced design
    replicate_counts = set()
    for pi_list in cells:
        for cell_list in pi_list:
            if len(cell_list) == 0:
                raise ValueError("Unbalanced design: some part-operator combinations have no measurements")
            replicate_counts.add(len(cell_list))

    if len(replicate_counts) > 1:
        raise ValueError(
            f"Unbalanced design: replicate counts vary ({replicate_counts}). Gage R&R requires a balanced design."
        )

    r_count = replicate_counts.pop()
    if r_count < 2:
        raise ValueError("Need at least 2 replicates per part-operator combination")

    # Convert to numpy array: shape (p, o, r)
    data_array = np.array(cells, dtype=float)

    # Grand mean
    grand_mean = float(data_array.mean())

    # Marginal means
    part_means = data_array.mean(axis=(1, 2))  # (p,)
    op_means = data_array.mean(axis=(0, 2))  # (o,)
    cell_means = data_array.mean(axis=2)  # (p, o)

    # Sum of Squares
    ss_total = float(np.sum((data_array - grand_mean) ** 2))
    ss_part = float(o_count * r_count * np.sum((part_means - grand_mean) ** 2))
    ss_operator = float(p_count * r_count * np.sum((op_means - grand_mean) ** 2))
    ss_interaction = float(r_count * np.sum((cell_means - part_means[:, None] - op_means[None, :] + grand_mean) ** 2))
    ss_error = ss_total - ss_part - ss_operator - ss_interaction

    # Degrees of freedom
    df_part = p_count - 1
    df_operator = o_count - 1
    df_interaction = df_part * df_operator
    df_error = p_count * o_count * (r_count - 1)
    df_total = n - 1

    # Mean Squares
    ms_part = ss_part / df_part if df_part > 0 else 0.0
    ms_operator = ss_operator / df_operator if df_operator > 0 else 0.0
    ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0.0
    ms_error = ss_error / df_error if df_error > 0 else 0.0

    # F-test for interaction
    if df_interaction > 0 and ms_error > 0:
        f_interaction = ms_interaction / ms_error
        p_interaction = float(1 - scipy_stats.f.cdf(f_interaction, df_interaction, df_error))
    else:
        f_interaction = 0.0
        p_interaction = 1.0

    interaction_significant = p_interaction <= alpha_interaction
    interaction_pooled = not interaction_significant

    if interaction_pooled:
        # Pool interaction into error
        ss_error_pooled = ss_interaction + ss_error
        df_error_pooled = df_interaction + df_error
        ms_error_pooled = ss_error_pooled / df_error_pooled if df_error_pooled > 0 else 0.0

        # F-tests against pooled error
        f_part = ms_part / ms_error_pooled if ms_error_pooled > 0 else 0.0
        p_part = float(1 - scipy_stats.f.cdf(f_part, df_part, df_error_pooled)) if ms_error_pooled > 0 else 1.0
        f_operator = ms_operator / ms_error_pooled if ms_error_pooled > 0 else 0.0
        p_operator = (
            float(1 - scipy_stats.f.cdf(f_operator, df_operator, df_error_pooled)) if ms_error_pooled > 0 else 1.0
        )

        # Variance components (EMS method)
        var_repeatability = ms_error_pooled
        var_interaction = 0.0
        var_operator = max(0.0, (ms_operator - ms_error_pooled) / (p_count * r_count))
        var_part = max(0.0, (ms_part - ms_error_pooled) / (o_count * r_count))
    else:
        # Keep interaction term — test Part and Operator against interaction MS
        f_part = ms_part / ms_interaction if ms_interaction > 0 else 0.0
        p_part = float(1 - scipy_stats.f.cdf(f_part, df_part, df_interaction)) if ms_interaction > 0 else 1.0
        f_operator = ms_operator / ms_interaction if ms_interaction > 0 else 0.0
        p_operator = (
            float(1 - scipy_stats.f.cdf(f_operator, df_operator, df_interaction)) if ms_interaction > 0 else 1.0
        )

        # Variance components (EMS method)
        var_repeatability = ms_error
        var_interaction = max(0.0, (ms_interaction - ms_error) / r_count)
        var_operator = max(0.0, (ms_operator - ms_interaction) / (p_count * r_count))
        var_part = max(0.0, (ms_part - ms_interaction) / (o_count * r_count))

    var_reproducibility = var_operator + var_interaction
    var_grr = var_repeatability + var_reproducibility
    var_total = var_grr + var_part
    if var_total <= 0:
        var_total = 1e-10

    # Percentage Contribution (variance-based)
    pct_contribution = {
        "GRR": var_grr / var_total * 100,
        "Repeatability": var_repeatability / var_total * 100,
        "Reproducibility": var_reproducibility / var_total * 100,
        "Operator": var_operator / var_total * 100,
        "Operator x Part": var_interaction / var_total * 100,
        "Part-to-Part": var_part / var_total * 100,
        "Total": 100.0,
    }

    # %Study Variation (std dev-based)
    sigma_total = math.sqrt(var_total)
    pct_study_var = {}
    for source, var in [
        ("GRR", var_grr),
        ("Repeatability", var_repeatability),
        ("Reproducibility", var_reproducibility),
        ("Operator", var_operator),
        ("Operator x Part", var_interaction),
        ("Part-to-Part", var_part),
        ("Total", var_total),
    ]:
        pct_study_var[source] = math.sqrt(var) / sigma_total * 100 if sigma_total > 0 else 0.0

    # %Tolerance
    pct_tolerance = {}
    if tolerance and tolerance > 0:
        for source, var in [
            ("GRR", var_grr),
            ("Repeatability", var_repeatability),
            ("Reproducibility", var_reproducibility),
            ("Operator", var_operator),
            ("Operator x Part", var_interaction),
            ("Part-to-Part", var_part),
            ("Total", var_total),
        ]:
            pct_tolerance[source] = (study_var_k * math.sqrt(var)) / tolerance * 100

    # Number of Distinct Categories
    sigma_grr = math.sqrt(var_grr) if var_grr > 0 else 1e-10
    sigma_part = math.sqrt(var_part)
    ndc = max(1, int(1.41 * sigma_part / sigma_grr))

    # Assessment based on %StudyVar for GRR
    grr_pct = pct_study_var["GRR"]
    if grr_pct < 10:
        assessment = "Acceptable"
    elif grr_pct <= 30:
        assessment = "Marginal"
    else:
        assessment = "Unacceptable"

    # Build ANOVA table
    anova_table = [
        {
            "source": "Part",
            "df": df_part,
            "ss": round(ss_part, 6),
            "ms": round(ms_part, 6),
            "f": round(f_part, 4),
            "p": round(p_part, 4),
        },
        {
            "source": "Operator",
            "df": df_operator,
            "ss": round(ss_operator, 6),
            "ms": round(ms_operator, 6),
            "f": round(f_operator, 4),
            "p": round(p_operator, 4),
        },
        {
            "source": "Part x Operator",
            "df": df_interaction,
            "ss": round(ss_interaction, 6),
            "ms": round(ms_interaction, 6),
            "f": round(f_interaction, 4),
            "p": round(p_interaction, 4),
        },
    ]

    if interaction_pooled:
        anova_table.append(
            {
                "source": "Repeatability",
                "df": df_interaction + df_error,
                "ss": round(ss_interaction + ss_error, 6),
                "ms": round(ms_error_pooled, 6),
                "f": None,
                "p": None,
                "note": "Interaction pooled",
            }
        )
    else:
        anova_table.append(
            {
                "source": "Repeatability",
                "df": df_error,
                "ss": round(ss_error, 6),
                "ms": round(ms_error, 6),
                "f": None,
                "p": None,
            }
        )

    anova_table.append(
        {
            "source": "Total",
            "df": df_total,
            "ss": round(ss_total, 6),
            "ms": None,
            "f": None,
            "p": None,
        }
    )

    # Generate Plotly charts
    plots = _generate_grr_plots(
        data_array,
        unique_parts,
        unique_operators,
        pct_contribution,
        cell_means,
    )

    return GageRRResult(
        var_repeatability=round(var_repeatability, 8),
        var_reproducibility=round(var_reproducibility, 8),
        var_operator=round(var_operator, 8),
        var_interaction=round(var_interaction, 8),
        var_grr=round(var_grr, 8),
        var_part=round(var_part, 8),
        var_total=round(var_total, 8),
        pct_contribution={k: round(v, 2) for k, v in pct_contribution.items()},
        pct_study_var={k: round(v, 2) for k, v in pct_study_var.items()},
        pct_tolerance={k: round(v, 2) for k, v in pct_tolerance.items()},
        ndc=ndc,
        grr_percent=round(grr_pct, 2),
        assessment=assessment,
        anova_table=anova_table,
        interaction_significant=interaction_significant,
        interaction_pooled=interaction_pooled,
        n_parts=p_count,
        n_operators=o_count,
        n_replicates=r_count,
        n_total=n,
        tolerance=tolerance,
        plots=plots,
    )


def _generate_grr_plots(data_array, unique_parts, unique_operators, pct_contribution, cell_means):
    """Generate the 4 Plotly charts for Gage R&R."""
    p, o, r = data_array.shape
    part_labels = [str(x) for x in unique_parts]
    op_labels = [str(x) for x in unique_operators]

    # 1. Variance Components Bar Chart
    sources = ["GRR", "Repeatability", "Reproducibility", "Part-to-Part"]
    vals = [pct_contribution.get(s, 0) for s in sources]

    components_chart = {
        "traces": [
            {
                "type": "bar",
                "x": sources,
                "y": vals,
                "marker": {"color": ["#e74c3c", "#e67e22", "#f39c12", "#27ae60"]},
                "text": [f"{v:.1f}%" for v in vals],
                "textposition": "outside",
            }
        ],
        "layout": {
            "title": "% Contribution to Total Variation",
            "yaxis": {
                "title": "% Contribution",
                "range": [0, max(vals) * 1.2 if vals else 100],
            },
            "margin": {"t": 40, "b": 60},
        },
    }

    # 2. Measurement by Part (box plot)
    by_part_traces = []
    for pi in range(p):
        by_part_traces.append(
            {
                "type": "box",
                "y": data_array[pi, :, :].flatten().tolist(),
                "name": part_labels[pi],
                "boxpoints": "all",
                "jitter": 0.3,
                "pointpos": -1.8,
                "marker": {"size": 4},
            }
        )

    by_part_chart = {
        "traces": by_part_traces,
        "layout": {
            "title": "Measurement by Part",
            "yaxis": {"title": "Measurement"},
            "xaxis": {"title": "Part"},
            "showlegend": False,
            "margin": {"t": 40, "b": 60},
        },
    }

    # 3. Measurement by Operator (box plot)
    by_op_traces = []
    for oi in range(o):
        by_op_traces.append(
            {
                "type": "box",
                "y": data_array[:, oi, :].flatten().tolist(),
                "name": op_labels[oi],
                "boxpoints": "all",
                "jitter": 0.3,
                "pointpos": -1.8,
                "marker": {"size": 4},
            }
        )

    by_operator_chart = {
        "traces": by_op_traces,
        "layout": {
            "title": "Measurement by Operator",
            "yaxis": {"title": "Measurement"},
            "xaxis": {"title": "Operator"},
            "showlegend": False,
            "margin": {"t": 40, "b": 60},
        },
    }

    # 4. Part x Operator Interaction
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#e67e22", "#1abc9c"]
    interaction_traces = []
    for oi in range(o):
        interaction_traces.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "x": part_labels,
                "y": cell_means[:, oi].tolist(),
                "name": op_labels[oi],
                "line": {"color": colors[oi % len(colors)]},
                "marker": {"size": 8},
            }
        )

    interaction_chart = {
        "traces": interaction_traces,
        "layout": {
            "title": "Part x Operator Interaction",
            "yaxis": {"title": "Mean Measurement"},
            "xaxis": {"title": "Part"},
            "margin": {"t": 40, "b": 60},
        },
    }

    return {
        "components": components_chart,
        "by_part": by_part_chart,
        "by_operator": by_operator_chart,
        "interaction": interaction_chart,
    }


def gage_rr_nested(
    measurements: list[list[list[float]]],
    part_labels: list[str] | None = None,
    operator_labels: list[str] | None = None,
) -> "GageRRResult":
    """Nested (hierarchical) Gage R&R.

    Used when each operator measures DIFFERENT parts (not the same parts).
    Common in destructive testing where the part is consumed.

    Args:
        measurements: [operator][part][replicate] — 3D array
            Each operator measures their own set of parts.
        part_labels: optional part identifiers
        operator_labels: optional operator identifiers

    Returns:
        GageRRResult with variance components
    """
    import numpy as np

    n_operators = len(measurements)
    n_parts_per_operator = len(measurements[0])
    n_replicates = len(measurements[0][0])

    # Flatten for grand statistics
    all_values = []
    for op in measurements:
        for part in op:
            all_values.extend(part)
    grand_mean = np.mean(all_values)
    N = len(all_values)

    # Compute SS
    # SS_operator (between operators)
    operator_means = []
    for op in measurements:
        op_vals = [v for part in op for v in part]
        operator_means.append(np.mean(op_vals))

    n_per_operator = n_parts_per_operator * n_replicates
    ss_operator = n_per_operator * sum((m - grand_mean) ** 2 for m in operator_means)

    # SS_parts(operator) — parts nested within operators
    ss_parts_within = 0
    part_means_all = []
    for op_idx, op in enumerate(measurements):
        for part in op:
            part_mean = np.mean(part)
            part_means_all.append(part_mean)
            ss_parts_within += n_replicates * (part_mean - operator_means[op_idx]) ** 2

    # SS_repeatability (within part replicates)
    ss_repeat = 0
    for op in measurements:
        for part in op:
            part_mean = np.mean(part)
            for val in part:
                ss_repeat += (val - part_mean) ** 2

    # Degrees of freedom
    df_operator = n_operators - 1
    df_parts = n_operators * (n_parts_per_operator - 1)
    df_repeat = n_operators * n_parts_per_operator * (n_replicates - 1)

    # Mean squares
    ms_operator = ss_operator / df_operator if df_operator > 0 else 0
    ms_parts = ss_parts_within / df_parts if df_parts > 0 else 0
    ms_repeat = ss_repeat / df_repeat if df_repeat > 0 else 0

    # Variance components
    var_repeat = ms_repeat
    var_parts = max(0, (ms_parts - ms_repeat) / n_replicates)
    var_operator = max(0, (ms_operator - ms_parts) / (n_parts_per_operator * n_replicates))

    var_gage = var_repeat + var_operator  # reproducibility + repeatability
    var_total = var_gage + var_parts

    pct_gage = (var_gage / var_total * 100) if var_total > 0 else 0
    pct_repeat = (var_repeat / var_total * 100) if var_total > 0 else 0
    pct_operator = (var_operator / var_total * 100) if var_total > 0 else 0
    pct_parts = (var_parts / var_total * 100) if var_total > 0 else 0

    import math as _m
    grr_pct = _m.sqrt(var_gage / var_total) * 100 if var_total > 0 else 0
    ndc = max(1, int(1.41 * _m.sqrt(var_parts / var_gage))) if var_gage > 0 else 0

    if grr_pct < 10:
        assessment = "Acceptable"
    elif grr_pct < 30:
        assessment = "Marginal"
    else:
        assessment = "Unacceptable"

    return GageRRResult(
        var_repeatability=round(var_repeat, 6),
        var_reproducibility=round(var_operator, 6),
        var_operator=round(var_operator, 6),
        var_interaction=0.0,  # nested design has no interaction term
        var_grr=round(var_gage, 6),
        var_part=round(var_parts, 6),
        var_total=round(var_total, 6),
        pct_contribution={
            "gage_rr": round(pct_gage, 2),
            "repeatability": round(pct_repeat, 2),
            "reproducibility": round(pct_operator, 2),
            "part_to_part": round(pct_parts, 2),
        },
        pct_study_var={
            "gage_rr": round(grr_pct, 2),
            "repeatability": round(_m.sqrt(var_repeat / var_total) * 100 if var_total > 0 else 0, 2),
            "reproducibility": round(_m.sqrt(var_operator / var_total) * 100 if var_total > 0 else 0, 2),
            "part_to_part": round(_m.sqrt(var_parts / var_total) * 100 if var_total > 0 else 0, 2),
        },
        pct_tolerance={},
        ndc=ndc,
        grr_percent=round(grr_pct, 2),
        assessment=assessment,
        anova_table=[
            {"source": "Operator", "df": df_operator, "ss": round(ss_operator, 4), "ms": round(ms_operator, 4)},
            {"source": "Part(Operator)", "df": df_parts, "ss": round(ss_parts_within, 4), "ms": round(ms_parts, 4)},
            {"source": "Repeatability", "df": df_repeat, "ss": round(ss_repeat, 4), "ms": round(ms_repeat, 4)},
        ],
        interaction_significant=False,
        interaction_pooled=True,
        n_parts=n_operators * n_parts_per_operator,
        n_operators=n_operators,
        n_replicates=n_replicates,
        n_total=N,
        tolerance=None,
        plots={},
    )


def attribute_agreement(
    ratings: list[list[int]],
    reference: list[int] | None = None,
    appraiser_labels: list[str] | None = None,
) -> dict:
    """Attribute Agreement Analysis.

    Evaluates consistency of categorical judgments (pass/fail, go/no-go)
    across appraisers and against a reference standard.

    Args:
        ratings: [appraiser][sample] — each value is the category assigned
        reference: optional known-correct classification per sample
        appraiser_labels: optional names

    Returns:
        {within_appraiser, between_appraiser, vs_reference, kappa}
    """
    n_appraisers = len(ratings)
    n_samples = len(ratings[0])

    # Within-appraiser agreement (if repeated trials, ratings would be 3D)
    # For single-trial, within = 100% by definition

    # Between-appraiser agreement
    agreements = 0
    for s in range(n_samples):
        sample_ratings = [ratings[a][s] for a in range(n_appraisers)]
        if len(set(sample_ratings)) == 1:
            agreements += 1
    between_pct = agreements / n_samples * 100 if n_samples > 0 else 0

    # Vs reference (if provided)
    vs_ref = {}
    if reference:
        for a in range(n_appraisers):
            correct = sum(1 for s in range(n_samples) if ratings[a][s] == reference[s])
            label = appraiser_labels[a] if appraiser_labels and a < len(appraiser_labels) else f"Appraiser {a+1}"
            vs_ref[label] = round(correct / n_samples * 100, 1)

    # Fleiss' kappa (simplified for 2+ appraisers)
    categories = set()
    for a in range(n_appraisers):
        categories.update(ratings[a])
    categories = sorted(categories)

    # Count agreements per sample
    P_bar = 0
    for s in range(n_samples):
        sample_ratings = [ratings[a][s] for a in range(n_appraisers)]
        for cat in categories:
            n_j = sample_ratings.count(cat)
            P_bar += n_j * (n_j - 1)
    P_bar /= (n_samples * n_appraisers * (n_appraisers - 1)) if n_appraisers > 1 else 1

    # Expected agreement
    P_e = 0
    for cat in categories:
        p_j = sum(ratings[a].count(cat) for a in range(n_appraisers)) / (n_samples * n_appraisers)
        P_e += p_j ** 2

    kappa = (P_bar - P_e) / (1 - P_e) if P_e < 1 else 1.0

    return {
        "between_appraiser_pct": round(between_pct, 1),
        "vs_reference": vs_ref,
        "fleiss_kappa": round(kappa, 4),
        "kappa_interpretation": "excellent" if kappa > 0.75 else "fair_to_good" if kappa > 0.40 else "poor",
        "n_appraisers": n_appraisers,
        "n_samples": n_samples,
    }
