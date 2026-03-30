"""Tests for Bayesian SPC — the competitive differentiator."""

import random
import numpy as np

from forgespc.bayesian import (
    bayesian_capability,
    bayesian_changepoint,
    bayesian_control_chart,
    nig_posterior_update,
    nig_sample,
    BayesianCapabilityResult,
)


class TestNIGPosterior:
    def test_posterior_contracts_with_data(self):
        """More data = tighter posterior (higher nu_n, alpha_n)."""
        data_small = np.array([50.0, 51.0, 49.0])
        data_large = np.array([50.0 + random.gauss(0, 1) for _ in range(100)])

        _, nu_s, alpha_s, _ = nig_posterior_update(data_small, 50.0, 1.0, 2.0, 1.0)
        _, nu_l, alpha_l, _ = nig_posterior_update(data_large, 50.0, 1.0, 2.0, 1.0)

        assert nu_l > nu_s
        assert alpha_l > alpha_s

    def test_posterior_mean_converges_to_data(self):
        """With weak prior, posterior mean → sample mean."""
        data = np.array([100.0] * 50)
        mu_n, _, _, _ = nig_posterior_update(data, 0.0, 0.001, 0.001, 0.001)
        assert abs(mu_n - 100.0) < 1.0

    def test_sampling_produces_correct_shapes(self):
        mu_s, sigma_s = nig_sample(50.0, 10.0, 5.0, 2.0, n_samples=1000)
        assert len(mu_s) == 1000
        assert len(sigma_s) == 1000
        assert all(sigma_s > 0)  # Sigma must be positive


class TestBayesianCapability:
    def test_capable_process(self):
        random.seed(42)
        data = [50 + random.gauss(0, 0.5) for _ in range(100)]
        result = bayesian_capability(data, usl=52.0, lsl=48.0)

        assert isinstance(result, BayesianCapabilityResult)
        assert result.p_gt_133 > 0.8  # High probability of capable
        assert result.cpk_median > 1.0
        assert "NOT CAPABLE" not in result.verdict  # CAPABLE or MARGINAL

    def test_incapable_process(self):
        random.seed(42)
        data = [50 + random.gauss(0, 5) for _ in range(100)]
        result = bayesian_capability(data, usl=52.0, lsl=48.0)

        assert result.p_gt_133 < 0.1
        assert result.cpk_median < 0.5
        assert "NOT CAPABLE" in result.verdict

    def test_bayesian_vs_frequentist_agree_large_sample(self):
        """With lots of data, Bayesian and frequentist should converge."""
        random.seed(42)
        data = [50 + random.gauss(0, 1) for _ in range(500)]
        result = bayesian_capability(data, usl=53.0, lsl=47.0)

        diff = abs(result.cpk_median - result.cpk_freq)
        assert diff < 0.1, f"Bayesian ({result.cpk_median:.3f}) and freq ({result.cpk_freq:.3f}) diverge"

    def test_credible_interval_narrows_with_data(self):
        """More data = narrower CI."""
        random.seed(42)
        small = [50 + random.gauss(0, 1) for _ in range(20)]
        large = [50 + random.gauss(0, 1) for _ in range(200)]

        r_small = bayesian_capability(small, usl=53.0, lsl=47.0)
        r_large = bayesian_capability(large, usl=53.0, lsl=47.0)

        ci_width_small = r_small.cpk_ci[1] - r_small.cpk_ci[0]
        ci_width_large = r_large.cpk_ci[1] - r_large.cpk_ci[0]
        assert ci_width_large < ci_width_small

    def test_cpm_taguchi_penalizes_off_target(self):
        """Off-target process: Cpm < Cpk."""
        random.seed(42)
        data = [51.5 + random.gauss(0, 0.5) for _ in range(100)]  # Mean shifted from target
        result = bayesian_capability(data, usl=53.0, lsl=47.0, target=50.0)

        assert result.cpm_median is not None
        assert result.cpm_median < result.cpk_median  # Taguchi penalizes offset

    def test_centering_metric(self):
        """Off-center process should have k > 0."""
        data = [52.0 + random.gauss(0, 0.3) for _ in range(50)]  # Shifted high
        result = bayesian_capability(data, usl=53.0, lsl=47.0)
        assert result.k_centering is not None
        assert result.k_centering > 0.3  # Significantly off-center

    def test_one_sided_spec(self):
        """Only USL specified."""
        data = [50 + random.gauss(0, 1) for _ in range(50)]
        result = bayesian_capability(data, usl=53.0)
        assert result.cpk_median > 0

    def test_historical_prior(self):
        data = [50 + random.gauss(0, 1) for _ in range(30)]
        result = bayesian_capability(
            data, usl=53.0, lsl=47.0,
            prior_type="historical",
            prior_params={"hist_mean": 50.0, "hist_std": 1.0, "hist_n": 50},
        )
        assert result.cpk_median > 0

    def test_probability_table_sums(self):
        data = [50 + random.gauss(0, 1) for _ in range(100)]
        result = bayesian_capability(data, usl=53.0, lsl=47.0)
        # p_gt_2 <= p_gt_167 <= p_gt_133 <= p_gt_1
        assert result.p_gt_2 <= result.p_gt_167 + 0.01
        assert result.p_gt_167 <= result.p_gt_133 + 0.01
        assert result.p_gt_133 <= result.p_gt_1 + 0.01


class TestBayesianChangepoint:
    def test_detects_shift(self):
        random.seed(42)
        data = [50 + random.gauss(0, 1) for _ in range(50)]
        data += [60 + random.gauss(0, 1) for _ in range(50)]  # 10-sigma shift
        result = bayesian_changepoint(data)

        assert len(result.changepoints) > 0
        assert len(result.segments) >= 2

    def test_stable_process_few_changepoints(self):
        random.seed(42)
        data = [50 + random.gauss(0, 1) for _ in range(100)]
        result = bayesian_changepoint(data)
        assert len(result.changepoints) <= 3  # May get spurious ones

    def test_segments_cover_data(self):
        data = [50.0] * 30 + [55.0] * 30
        result = bayesian_changepoint(data)
        total_n = sum(s.n for s in result.segments)
        assert total_n == len(data)

    def test_per_segment_cpk(self):
        random.seed(42)
        data = [50 + random.gauss(0, 0.5) for _ in range(50)]
        data += [55 + random.gauss(0, 0.5) for _ in range(50)]
        result = bayesian_changepoint(data, usl=56.0, lsl=44.0)
        # At least one segment should have Cpk computed
        cpks = [s.cpk for s in result.segments if s.cpk is not None]
        assert len(cpks) > 0


class TestBayesianControlChart:
    def test_stable_in_control(self):
        random.seed(42)
        data = [50 + random.gauss(0, 1) for _ in range(50)]
        result = bayesian_control_chart(data)
        assert result.n == 50
        assert len(result.posterior_mean) == 50
        assert len(result.ucl) == 50

    def test_limits_tighten(self):
        """Posterior predictive limits should tighten as data accumulates."""
        data = [50 + random.gauss(0, 1) for _ in range(50)]
        result = bayesian_control_chart(data)
        early_width = result.ucl[5] - result.lcl[5]
        late_width = result.ucl[-1] - result.lcl[-1]
        assert late_width < early_width

    def test_shift_detected(self):
        random.seed(42)
        data = [50 + random.gauss(0, 1) for _ in range(30)]
        data += [58 + random.gauss(0, 1) for _ in range(20)]
        result = bayesian_control_chart(data)
        assert len(result.out_of_control) > 0
