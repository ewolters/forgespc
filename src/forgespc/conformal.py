"""Conformal SPC — distribution-free process monitoring.

No normality assumption. Uses conformal prediction to construct
valid prediction intervals from any data distribution.

Nobody else has this in a production SPC tool.

Requires: numpy, scikit-learn

Usage:
    from forgespc.conformal import conformal_control, entropy_spc

    result = conformal_control(data, alpha=0.05)
    print(f"OOC points: {result.n_ooc}")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# =============================================================================
# Conformal Control Chart
# =============================================================================


@dataclass
class ConformalControlResult:
    """Conformal control chart result."""

    data_points: list[float]
    nonconformity_scores: list[float]
    threshold: float
    ooc_indices: list[int]
    in_control: bool
    n_calibration: int
    n_monitoring: int
    alpha: float  # False alarm rate
    prediction_intervals: list[tuple[float, float]] | None = None

    @property
    def n_ooc(self) -> int:
        return len(self.ooc_indices)


def conformal_control(
    data: list[float] | np.ndarray,
    alpha: float = 0.05,
    calibration_fraction: float = 0.5,
) -> ConformalControlResult:
    """Distribution-free control chart using conformal prediction.

    Phase I (calibration): Compute nonconformity scores from calibration data.
    Phase II (monitoring): Flag points whose scores exceed the (1-alpha) quantile.

    No normality assumption. Valid for any distribution with exchangeable data.

    Args:
        data: Process measurements (time-ordered).
        alpha: False alarm rate (e.g., 0.05 = 5% expected false alarms).
        calibration_fraction: Fraction of data used for calibration.
    """
    arr = np.asarray(data, dtype=float)
    n = len(arr)

    if n < 20:
        raise ValueError("Need at least 20 observations for conformal control chart.")

    n_cal = max(10, int(n * calibration_fraction))
    cal_data = arr[:n_cal]
    _ = arr[n_cal:]  # monitoring data (used by caller)

    # Nonconformity scores: |x - median| / MAD
    cal_median = float(np.median(cal_data))
    cal_mad = float(np.median(np.abs(cal_data - cal_median)))
    if cal_mad == 0:
        cal_mad = float(np.std(cal_data, ddof=1))
    if cal_mad == 0:
        cal_mad = 1.0

    cal_scores = np.abs(cal_data - cal_median) / cal_mad
    threshold = float(np.quantile(cal_scores, 1 - alpha))

    # Monitor
    all_scores = np.abs(arr - cal_median) / cal_mad
    ooc_indices = [int(i) for i in range(n_cal, n) if all_scores[i] > threshold]

    # Prediction intervals
    intervals = []
    for i in range(n):
        lower = cal_median - threshold * cal_mad
        upper = cal_median + threshold * cal_mad
        intervals.append((float(lower), float(upper)))

    return ConformalControlResult(
        data_points=arr.tolist(),
        nonconformity_scores=all_scores.tolist(),
        threshold=threshold,
        ooc_indices=ooc_indices,
        in_control=len(ooc_indices) == 0,
        n_calibration=n_cal,
        n_monitoring=n - n_cal,
        alpha=alpha,
        prediction_intervals=intervals,
    )


# =============================================================================
# Entropy SPC — information-theoretic monitoring
# =============================================================================


@dataclass
class EntropySPCResult:
    """Entropy-based SPC result."""

    entropy_values: list[float]
    baseline_entropy: float
    ucl: float
    lcl: float
    ooc_indices: list[int]
    in_control: bool
    window_size: int
    n: int

    @property
    def n_ooc(self) -> int:
        return len(self.ooc_indices)


def entropy_spc(
    data: list[float] | np.ndarray,
    window_size: int = 20,
    n_bins: int = 10,
    alpha: float = 0.05,
) -> EntropySPCResult:
    """Information-theoretic SPC using sliding-window Shannon entropy.

    Monitors the entropy (information content) of the process distribution.
    A shift in distribution — mean, spread, or shape — changes the entropy.

    Advantages over Shewhart: detects distributional changes that don't
    affect the mean (e.g., bimodality, spread changes, shape changes).

    Args:
        data: Process measurements.
        window_size: Sliding window for entropy computation.
        n_bins: Histogram bins for entropy estimation.
        alpha: False alarm rate for control limits.
    """
    arr = np.asarray(data, dtype=float)
    n = len(arr)

    if n < window_size * 2:
        raise ValueError(f"Need at least {window_size * 2} observations (2x window size).")

    # Compute sliding-window entropy
    entropies = []
    for i in range(n - window_size + 1):
        window = arr[i : i + window_size]
        # Histogram-based Shannon entropy
        counts, _ = np.histogram(window, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        h = float(-np.sum(probs * np.log2(probs)))
        entropies.append(h)

    ent_arr = np.array(entropies)

    # Use first half as calibration for limits
    n_cal = len(ent_arr) // 2
    cal_ent = ent_arr[:n_cal]
    cal_mean = float(np.mean(cal_ent))
    cal_std = float(np.std(cal_ent, ddof=1))

    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)
    ucl = cal_mean + z * cal_std
    lcl = cal_mean - z * cal_std

    ooc = [int(i + window_size - 1) for i, h in enumerate(entropies) if h > ucl or h < lcl]

    return EntropySPCResult(
        entropy_values=entropies,
        baseline_entropy=cal_mean,
        ucl=ucl,
        lcl=lcl,
        ooc_indices=ooc,
        in_control=len(ooc) == 0,
        window_size=window_size,
        n=n,
    )
