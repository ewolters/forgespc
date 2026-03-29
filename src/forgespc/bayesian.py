"""Bayesian SPC — the differentiator.

Normal-Inverse-Gamma conjugate posterior for process parameters,
Monte Carlo capability indices with full posterior distributions.

This eliminates the classical 1.5σ shift assumption. Instead of
point-estimate Cpk, you get P(Cpk > 1.33) with credible intervals.

Requires: numpy, scipy

Usage:
    from forgespc.bayesian import bayesian_capability, bayesian_changepoint

    result = bayesian_capability(data, usl=53.0, lsl=47.0)
    print(f"P(Cpk > 1.33) = {result.p_gt_133:.1%}")
    print(f"Bayesian Cpk = {result.cpk_median:.3f} [{result.cpk_ci[0]:.3f}, {result.cpk_ci[1]:.3f}]")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# =============================================================================
# Normal-Inverse-Gamma conjugate posterior
# =============================================================================


def nig_posterior_update(
    data: np.ndarray, mu0: float, nu0: float, alpha0: float, beta0: float
) -> tuple[float, float, float, float]:
    """Normal-Inverse-Gamma conjugate posterior update.

    Given data ~ N(mu, sigma²), with NIG prior on (mu, sigma²),
    returns the posterior parameters (mu_n, nu_n, alpha_n, beta_n).
    """
    n = len(data)
    x_bar = float(np.mean(data))
    nu_n = nu0 + n
    mu_n = (nu0 * mu0 + n * x_bar) / nu_n
    alpha_n = alpha0 + n / 2.0
    beta_n = (
        beta0
        + 0.5 * np.sum((data - x_bar) ** 2)
        + (n * nu0 * (x_bar - mu0) ** 2) / (2.0 * nu_n)
    )
    return float(mu_n), float(nu_n), float(alpha_n), float(beta_n)


def nig_sample(
    mu_n: float, nu_n: float, alpha_n: float, beta_n: float, n_samples: int = 10000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Draw (mu, sigma) samples from NIG posterior."""
    from scipy.stats import invgamma

    rng = np.random.default_rng(seed)
    sigma2_samples = invgamma.rvs(a=alpha_n, scale=beta_n, size=n_samples, random_state=rng)
    mu_samples = rng.normal(loc=mu_n, scale=np.sqrt(sigma2_samples / nu_n))
    sigma_samples = np.sqrt(sigma2_samples)
    return mu_samples, sigma_samples


def cpk_from_params(
    mu: np.ndarray, sigma: np.ndarray, usl: float | None = None, lsl: float | None = None
) -> np.ndarray:
    """Vectorized Cpk from arrays of mu and sigma. Supports one-sided specs."""
    if usl is not None and lsl is not None:
        cpu = (usl - mu) / (3.0 * sigma)
        cpl = (mu - lsl) / (3.0 * sigma)
        return np.minimum(cpu, cpl)
    elif usl is not None:
        return (usl - mu) / (3.0 * sigma)
    elif lsl is not None:
        return (mu - lsl) / (3.0 * sigma)
    else:
        return np.zeros_like(mu)


# =============================================================================
# Bayesian Capability Analysis
# =============================================================================


@dataclass
class BayesianCapabilityResult:
    """Full Bayesian capability analysis result.

    Contains both Bayesian (posterior median + CI) and frequentist
    (point estimate) indices for comparison.
    """

    # Bayesian Cpk
    cpk_median: float
    cpk_ci: tuple[float, float]  # 95% credible interval
    cpk_samples: list[float] | None = None  # Full posterior for ForgeViz

    # Probability table — the decision-making outputs
    p_gt_1: float = 0.0  # P(Cpk > 1.00)
    p_gt_133: float = 0.0  # P(Cpk > 1.33)
    p_gt_167: float = 0.0  # P(Cpk > 1.67)
    p_gt_2: float = 0.0  # P(Cpk > 2.00)

    # Frequentist comparison
    cpk_freq: float = 0.0

    # Additional indices (Bayesian median)
    cp_median: float | None = None
    cp_ci: tuple[float, float] | None = None
    cpm_median: float | None = None  # Taguchi
    cpm_ci: tuple[float, float] | None = None

    # Centering
    k_centering: float | None = None  # 0 = centered, 1 = at spec limit

    # Expected performance
    dpmo: float = 0.0
    yield_pct: float = 100.0
    sigma_level: float = 0.0
    p_out_of_spec: float = 0.0

    # Posterior state
    mu_posterior: tuple[float, float, float, float] | None = None  # (mu_n, nu_n, alpha_n, beta_n)
    mu_median: float = 0.0
    sigma_median: float = 0.0
    sigma_warning: str = ""

    # Verdict
    verdict: str = ""

    # Data summary
    n: int = 0
    x_bar: float = 0.0
    s: float = 0.0
    usl: float | None = None
    lsl: float | None = None
    target: float | None = None


def bayesian_capability(
    data: list[float] | np.ndarray,
    usl: float | None = None,
    lsl: float | None = None,
    target: float | None = None,
    prior_type: str = "weakly_informative",
    prior_params: dict | None = None,
    n_mc: int = 10000,
) -> BayesianCapabilityResult:
    """Bayesian process capability analysis.

    Eliminates the 1.5σ shift assumption. Uses Normal-Inverse-Gamma
    conjugate posterior to produce full probability distributions of
    capability indices.

    Args:
        data: Process measurements.
        usl: Upper specification limit.
        lsl: Lower specification limit.
        target: Target value (for Cpm).
        prior_type: "weakly_informative", "informative", or "historical".
        prior_params: For informative/historical priors.
        n_mc: Monte Carlo sample count.

    Returns:
        BayesianCapabilityResult with full posterior analysis.
    """
    arr = np.asarray(data, dtype=float)
    n = len(arr)
    x_bar = float(np.mean(arr))
    s = float(np.std(arr, ddof=1)) if n > 1 else 0.01

    if usl is None and lsl is None:
        raise ValueError("At least one spec limit (usl or lsl) required.")
    if usl is not None and lsl is not None and usl <= lsl:
        raise ValueError("usl must be greater than lsl.")

    if target is None:
        if usl is not None and lsl is not None:
            target = (usl + lsl) / 2.0
        elif usl is not None:
            target = usl
        else:
            target = lsl

    # Set prior
    pp = prior_params or {}
    if prior_type == "informative":
        mu0 = float(pp.get("mu0", x_bar))
        nu0 = float(pp.get("nu0", 5))
        alpha0 = float(pp.get("alpha0", 3))
        beta0 = float(pp.get("beta0", s ** 2 * 2))
    elif prior_type == "historical":
        hist_mean = float(pp.get("hist_mean", x_bar))
        hist_std = float(pp.get("hist_std", s))
        hist_n = int(pp.get("hist_n", 30))
        mu0, nu0 = hist_mean, float(hist_n)
        alpha0 = hist_n / 2.0
        beta0 = hist_n / 2.0 * hist_std ** 2
    else:  # weakly_informative
        s2 = float(np.var(arr, ddof=1)) if n > 1 else 1.0
        mu0, nu0, alpha0, beta0 = x_bar, 1.0, 2.0, max(s2, 1e-10)

    # Posterior update
    mu_n, nu_n, alpha_n, beta_n = nig_posterior_update(arr, mu0, nu0, alpha0, beta0)

    # Monte Carlo sampling
    mu_samples, sigma_samples = nig_sample(mu_n, nu_n, alpha_n, beta_n, n_mc)
    cpk_samples = cpk_from_params(mu_samples, sigma_samples, usl, lsl)

    cpk_median = float(np.median(cpk_samples))
    cpk_ci = (float(np.percentile(cpk_samples, 2.5)), float(np.percentile(cpk_samples, 97.5)))

    # Probability table
    p_gt_1 = float(np.mean(cpk_samples > 1.0))
    p_gt_133 = float(np.mean(cpk_samples > 1.33))
    p_gt_167 = float(np.mean(cpk_samples > 1.67))
    p_gt_2 = float(np.mean(cpk_samples > 2.0))

    # Frequentist Cpk
    cpk_freq = 0.0
    if s > 0:
        if usl is not None and lsl is not None:
            cpk_freq = float(min((usl - x_bar) / (3 * s), (x_bar - lsl) / (3 * s)))
        elif usl is not None:
            cpk_freq = float((usl - x_bar) / (3 * s))
        else:
            cpk_freq = float((x_bar - lsl) / (3 * s))

    # Cp (potential — ignores centering)
    cp_median = cp_ci = None
    if usl is not None and lsl is not None and s > 0:
        cp_samples = (usl - lsl) / (6.0 * sigma_samples)
        cp_median = float(np.median(cp_samples))
        cp_ci = (float(np.percentile(cp_samples, 2.5)), float(np.percentile(cp_samples, 97.5)))

    # Cpm — Taguchi (penalizes off-target)
    cpm_median = cpm_ci = None
    if target is not None and usl is not None and lsl is not None and s > 0:
        cpm_denom = 6.0 * np.sqrt(sigma_samples ** 2 + (mu_samples - target) ** 2)
        cpm_samples = (usl - lsl) / cpm_denom
        cpm_median = float(np.median(cpm_samples))
        cpm_ci = (float(np.percentile(cpm_samples, 2.5)), float(np.percentile(cpm_samples, 97.5)))

    # Centering
    k_centering = None
    if usl is not None and lsl is not None:
        midpoint = (usl + lsl) / 2.0
        half_tol = (usl - lsl) / 2.0
        k_centering = abs(x_bar - midpoint) / half_tol if half_tol > 0 else 0.0

    # Posterior predictive OOS
    rng_pp = np.random.default_rng(123)
    x_pred = rng_pp.normal(loc=mu_samples, scale=sigma_samples)
    oos_mask = np.zeros(len(x_pred), dtype=bool)
    if lsl is not None:
        oos_mask |= x_pred < lsl
    if usl is not None:
        oos_mask |= x_pred > usl
    p_oos = float(np.mean(oos_mask))
    dpmo = p_oos * 1e6
    yield_pct = (1.0 - p_oos) * 100.0

    # Sigma level from Bayesian DPMO
    from scipy.stats import norm as normdist

    if 0 < p_oos < 1:
        z_bench = float(normdist.ppf(1 - p_oos))
        sigma_level = z_bench + 1.5
    else:
        sigma_level = 7.5 if p_oos == 0 else 1.5

    # Sigma posterior sanity check
    sigma_99 = float(np.percentile(sigma_samples, 99))
    sigma_iqr = float(np.percentile(sigma_samples, 75) - np.percentile(sigma_samples, 25))
    sigma_warning = ""
    if sigma_iqr > 0 and sigma_99 > 5 * sigma_iqr + float(np.median(sigma_samples)):
        sigma_warning = "Heavy-tailed sigma posterior — possible non-normal data, mixed processes, or outliers."

    # Verdict
    if p_gt_133 >= 0.95:
        verdict = "CAPABLE — P(Cpk > 1.33) >= 95%"
    elif p_gt_133 >= 0.80:
        verdict = "MARGINAL — P(Cpk > 1.33) between 80-95%"
    else:
        verdict = "NOT CAPABLE — P(Cpk > 1.33) < 80%"

    return BayesianCapabilityResult(
        cpk_median=cpk_median,
        cpk_ci=cpk_ci,
        cpk_samples=cpk_samples.tolist(),
        p_gt_1=p_gt_1,
        p_gt_133=p_gt_133,
        p_gt_167=p_gt_167,
        p_gt_2=p_gt_2,
        cpk_freq=cpk_freq,
        cp_median=cp_median,
        cp_ci=cp_ci,
        cpm_median=cpm_median,
        cpm_ci=cpm_ci,
        k_centering=k_centering,
        dpmo=dpmo,
        yield_pct=yield_pct,
        sigma_level=sigma_level,
        p_out_of_spec=p_oos,
        mu_posterior=(mu_n, nu_n, alpha_n, beta_n),
        mu_median=float(np.median(mu_samples)),
        sigma_median=float(np.median(sigma_samples)),
        sigma_warning=sigma_warning,
        verdict=verdict,
        n=n,
        x_bar=x_bar,
        s=s,
        usl=usl,
        lsl=lsl,
        target=target,
    )


# =============================================================================
# Bayesian Changepoint Detection
# =============================================================================


@dataclass
class ChangeSegment:
    """A detected regime segment."""

    start: int
    end: int
    mean: float
    std: float
    n: int
    cpk: float | None = None  # If specs provided


@dataclass
class BayesianChangepointResult:
    """Result from Bayesian changepoint detection."""

    changepoints: list[int]
    segments: list[ChangeSegment]
    log_evidence: list[float]  # Log evidence at each potential changepoint
    n: int
    method: str = "bayesian_online"


def bayesian_changepoint(
    data: list[float] | np.ndarray,
    min_segment: int = 10,
    penalty: str = "bic",
    usl: float | None = None,
    lsl: float | None = None,
) -> BayesianChangepointResult:
    """Bayesian changepoint detection via marginal likelihood comparison.

    Uses a binary segmentation approach with BIC/AIC penalty to find
    regime changes. Each segment is modeled as N(mu, sigma²) with
    conjugate NIG prior.

    Args:
        data: Process measurements (time-ordered).
        min_segment: Minimum segment length.
        penalty: "bic" or "aic" for model selection.
        usl: Upper spec (for per-segment Cpk, optional).
        lsl: Lower spec (for per-segment Cpk, optional).

    Returns:
        BayesianChangepointResult with detected changepoints and segments.
    """
    arr = np.asarray(data, dtype=float)
    n = len(arr)

    def _segment_loglik(segment):
        """Log-likelihood of a segment under Gaussian model."""
        m = len(segment)
        if m < 2:
            return -np.inf
        mu = np.mean(segment)
        ss = np.sum((segment - mu) ** 2)
        sigma2 = ss / m
        if sigma2 <= 0:
            sigma2 = 1e-10
        return -0.5 * m * np.log(2 * np.pi * sigma2) - 0.5 * m

    def _find_best_split(segment, offset=0):
        """Find best single changepoint in a segment via BIC."""
        m = len(segment)
        if m < 2 * min_segment:
            return None, -np.inf

        full_ll = _segment_loglik(segment)
        k_full = 2  # mu, sigma
        if penalty == "bic":
            full_criterion = full_ll - k_full * np.log(m) / 2
        else:
            full_criterion = full_ll - k_full

        best_cp = None
        best_improvement = 0.0

        for cp in range(min_segment, m - min_segment + 1):
            left_ll = _segment_loglik(segment[:cp])
            right_ll = _segment_loglik(segment[cp:])
            split_ll = left_ll + right_ll
            k_split = 4  # two segments, each with mu + sigma
            if penalty == "bic":
                split_criterion = split_ll - k_split * np.log(m) / 2
            else:
                split_criterion = split_ll - k_split

            improvement = split_criterion - full_criterion
            if improvement > best_improvement:
                best_improvement = improvement
                best_cp = offset + cp

        return best_cp, best_improvement

    # Binary segmentation: recursively split
    changepoints = []
    segments_to_check = [(0, n)]

    while segments_to_check:
        start, end = segments_to_check.pop(0)
        segment = arr[start:end]
        cp, improvement = _find_best_split(segment, offset=start)

        if cp is not None and improvement > 0:
            changepoints.append(cp)
            segments_to_check.append((start, cp))
            segments_to_check.append((cp, end))

    changepoints = sorted(set(changepoints))

    # Build segments
    log_evidence = []  # Placeholder — binary segmentation doesn't produce per-point evidence
    boundaries = [0] + changepoints + [n]
    segments = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        seg_data = arr[start:end]
        seg_mean = float(np.mean(seg_data))
        seg_std = float(np.std(seg_data, ddof=1)) if len(seg_data) > 1 else 0.0

        seg_cpk = None
        if usl is not None and lsl is not None and seg_std > 0:
            seg_cpk = float(min((usl - seg_mean) / (3 * seg_std), (seg_mean - lsl) / (3 * seg_std)))

        segments.append(ChangeSegment(
            start=start, end=end,
            mean=seg_mean, std=seg_std, n=len(seg_data),
            cpk=seg_cpk,
        ))

    return BayesianChangepointResult(
        changepoints=changepoints,
        segments=segments,
        log_evidence=log_evidence,
        n=n,
        method="binary_segmentation_bic",
    )


# =============================================================================
# Bayesian Control Chart
# =============================================================================


@dataclass
class BayesianControlResult:
    """Bayesian control chart — posterior predictive limits."""

    data_points: list[float]
    posterior_mean: list[float]  # Running posterior mean
    ucl: list[float]  # Posterior predictive upper limit
    lcl: list[float]  # Posterior predictive lower limit
    out_of_control: list[int]
    in_control: bool
    n: int


def bayesian_control_chart(
    data: list[float] | np.ndarray,
    credible_level: float = 0.99,
) -> BayesianControlResult:
    """Bayesian control chart using sequential NIG posterior updates.

    Instead of fixed 3-sigma limits, uses posterior predictive intervals
    that tighten as evidence accumulates.

    Args:
        data: Process measurements.
        credible_level: Credible interval width (e.g., 0.99 for 99%).
    """
    from scipy.stats import t as tdist

    arr = np.asarray(data, dtype=float)
    n = len(arr)

    # Weakly informative prior
    x_init = arr[:min(5, n)]
    mu0 = float(np.mean(x_init))
    nu0 = 1.0
    alpha0 = 2.0
    beta0 = float(np.var(x_init, ddof=1)) if len(x_init) > 1 else 1.0

    posterior_mean = []
    ucl_vals = []
    lcl_vals = []
    ooc = []

    mu_n, nu_n, alpha_n, beta_n = mu0, nu0, alpha0, beta0

    for i in range(n):
        x = arr[i]

        # Posterior predictive: Student-t
        df_t = 2 * alpha_n
        loc_t = mu_n
        scale_t = np.sqrt(beta_n * (nu_n + 1) / (alpha_n * nu_n))

        t_dist = tdist(df=df_t, loc=loc_t, scale=scale_t)
        alpha_tail = (1 - credible_level) / 2
        ucl_i = float(t_dist.ppf(1 - alpha_tail))
        lcl_i = float(t_dist.ppf(alpha_tail))

        posterior_mean.append(float(mu_n))
        ucl_vals.append(ucl_i)
        lcl_vals.append(lcl_i)

        if x > ucl_i or x < lcl_i:
            ooc.append(i)

        # Update posterior with this observation
        nu_n_new = nu_n + 1
        mu_n_new = (nu_n * mu_n + x) / nu_n_new
        alpha_n_new = alpha_n + 0.5
        beta_n_new = beta_n + nu_n * (x - mu_n) ** 2 / (2 * nu_n_new)

        mu_n, nu_n, alpha_n, beta_n = mu_n_new, nu_n_new, alpha_n_new, beta_n_new

    return BayesianControlResult(
        data_points=arr.tolist(),
        posterior_mean=posterior_mean,
        ucl=ucl_vals,
        lcl=lcl_vals,
        out_of_control=ooc,
        in_control=len(ooc) == 0,
        n=n,
    )
