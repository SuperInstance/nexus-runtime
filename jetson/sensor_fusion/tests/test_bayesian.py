"""Tests for bayesian.py — BayesEstimate, BayesianEstimator."""

import math
import pytest

from jetson.sensor_fusion.bayesian import BayesEstimate, BayesianEstimator


# ===================================================================
# BayesEstimate tests
# ===================================================================

class TestBayesEstimate:
    def test_creation(self):
        e = BayesEstimate(mean=5.0, variance=2.0, confidence_interval=(3.0, 7.0), samples=10)
        assert e.mean == 5.0
        assert e.variance == 2.0
        assert e.confidence_interval == (3.0, 7.0)
        assert e.samples == 10

    def test_default_samples(self):
        e = BayesEstimate(mean=0.0, variance=1.0, confidence_interval=(-1.0, 1.0))
        assert e.samples == 1

    def test_fields_are_mutable(self):
        e = BayesEstimate(mean=0.0, variance=1.0, confidence_interval=(0.0, 0.0))
        e.mean = 99.0
        assert e.mean == 99.0


# ===================================================================
# BayesianEstimator tests
# ===================================================================

class TestBayesianComputePosterior:
    def test_equal_precision(self):
        be = BayesianEstimator()
        pm, pv = be.compute_posterior(prior_mean=0.0, prior_var=1.0, obs_mean=10.0, obs_var=1.0)
        assert pm == pytest.approx(5.0, abs=1e-10)
        assert pv == pytest.approx(0.5, abs=1e-10)

    def test_very_certain_prior(self):
        be = BayesianEstimator()
        pm, pv = be.compute_posterior(prior_mean=5.0, prior_var=0.01, obs_mean=100.0, obs_var=100.0)
        assert pm == pytest.approx(5.0, abs=0.1)
        assert pv < 0.01

    def test_very_certain_observation(self):
        be = BayesianEstimator()
        pm, pv = be.compute_posterior(prior_mean=100.0, prior_var=100.0, obs_mean=5.0, obs_var=0.01)
        assert pm == pytest.approx(5.0, abs=0.1)
        assert pv < 0.01

    def test_same_mean(self):
        be = BayesianEstimator()
        pm, pv = be.compute_posterior(prior_mean=7.0, prior_var=2.0, obs_mean=7.0, obs_var=3.0)
        assert pm == pytest.approx(7.0, abs=1e-10)
        assert pv < 2.0

    def test_posterior_var_smaller_than_both(self):
        be = BayesianEstimator()
        _, pv = be.compute_posterior(prior_mean=0.0, prior_var=5.0, obs_mean=0.0, obs_var=5.0)
        assert pv < 5.0

    def test_posterior_var_always_smaller_than_prior(self):
        be = BayesianEstimator()
        _, pv = be.compute_posterior(prior_mean=0.0, prior_var=10.0, obs_mean=0.0, obs_var=100.0)
        assert pv < 10.0


class TestBayesianUpdatePrior:
    def test_update_prior_is_compute_posterior(self):
        be = BayesianEstimator()
        r1 = be.update_prior(0.0, 1.0, 10.0, 1.0)
        r2 = be.compute_posterior(0.0, 1.0, 10.0, 1.0)
        assert r1[0] == pytest.approx(r2[0], abs=1e-12)
        assert r1[1] == pytest.approx(r2[1], abs=1e-12)


class TestBayesianComputeConfidence:
    def test_95_percent_ci(self):
        be = BayesianEstimator()
        ci = be.compute_confidence(posterior_mean=0.0, posterior_var=1.0, confidence_level=0.95)
        assert abs(ci[1] - ci[0]) > 0.0
        assert abs(ci[0] + ci[1]) < 1e-10  # symmetric around mean

    def test_99_percent_wider_than_95(self):
        be = BayesianEstimator()
        ci95 = be.compute_confidence(0.0, 1.0, 0.95)
        ci99 = be.compute_confidence(0.0, 1.0, 0.99)
        assert (ci99[1] - ci99[0]) > (ci95[1] - ci95[0])

    def test_zero_variance(self):
        be = BayesianEstimator()
        ci = be.compute_confidence(5.0, 0.0, 0.95)
        assert ci == (5.0, 5.0)

    def test_50_percent_ci(self):
        be = BayesianEstimator()
        ci = be.compute_confidence(0.0, 1.0, 0.5)
        assert ci[0] < ci[1]
        assert ci[0] < 0.0
        assert ci[1] > 0.0

    def test_confidence_level_zero(self):
        be = BayesianEstimator()
        ci = be.compute_confidence(0.0, 1.0, 0.0)
        assert ci == (0.0, 0.0)

    def test_confidence_level_one(self):
        be = BayesianEstimator()
        ci = be.compute_confidence(0.0, 1.0, 1.0)
        assert ci[0] <= -10.0  # very wide
        assert ci[1] >= 10.0

    def test_high_variance_wider(self):
        be = BayesianEstimator()
        ci1 = be.compute_confidence(0.0, 1.0, 0.95)
        ci10 = be.compute_confidence(0.0, 10.0, 0.95)
        assert (ci10[1] - ci10[0]) > (ci1[1] - ci1[0])


class TestBayesianSequentialUpdate:
    def test_single_observation(self):
        be = BayesianEstimator()
        est = be.sequential_update(0.0, 10.0, [5.0])
        assert est.mean > 0.0
        assert est.mean < 5.0
        assert est.samples == 1

    def test_multiple_observations(self):
        be = BayesianEstimator()
        true_val = 10.0
        obs = [true_val + i * 0.1 for i in range(-5, 6)]
        est = be.sequential_update(0.0, 100.0, obs, obs_variances=[1.0] * len(obs))
        assert abs(est.mean - true_val) < 1.0

    def test_sequential_variance_decreases(self):
        be = BayesianEstimator()
        obs = [5.0, 5.0, 5.0, 5.0, 5.0]
        est = be.sequential_update(0.0, 100.0, obs, obs_variances=[1.0] * 5)
        assert est.variance < 1.0

    def test_no_obs_variances_uses_posterior_var(self):
        be = BayesianEstimator()
        est = be.sequential_update(0.0, 5.0, [5.0])
        assert est.variance < 5.0

    def test_confidence_interval_in_result(self):
        be = BayesianEstimator()
        est = be.sequential_update(0.0, 10.0, [5.0])
        ci = est.confidence_interval
        assert ci[0] < est.mean
        assert ci[1] > est.mean

    def test_empty_observations(self):
        be = BayesianEstimator()
        est = be.sequential_update(5.0, 3.0, [])
        assert est.mean == pytest.approx(5.0, abs=1e-10)
        assert est.variance == pytest.approx(3.0, abs=1e-10)
        assert est.samples == 0


class TestBayesianPredict:
    def test_predict_adds_noise(self):
        be = BayesianEstimator()
        pm, pv = be.predict(5.0, 2.0, 3.0)
        assert pm == pytest.approx(5.0, abs=1e-10)
        assert pv == pytest.approx(5.0, abs=1e-10)

    def test_predict_zero_noise(self):
        be = BayesianEstimator()
        pm, pv = be.predict(5.0, 2.0, 0.0)
        assert pm == pytest.approx(5.0, abs=1e-10)
        assert pv == pytest.approx(2.0, abs=1e-10)

    def test_predict_mean_unchanged(self):
        be = BayesianEstimator()
        pm, _ = be.predict(42.0, 100.0, 50.0)
        assert pm == pytest.approx(42.0, abs=1e-10)


class TestBayesianComputeEvidence:
    def test_perfect_agreement(self):
        be = BayesianEstimator()
        ev = be.compute_evidence(obs_mean=5.0, obs_var=1.0, prior_mean=5.0, prior_var=1.0)
        assert ev > 0.0

    def test_large_disagreement_lower_evidence(self):
        be = BayesianEstimator()
        ev_close = be.compute_evidence(obs_mean=5.0, obs_var=1.0, prior_mean=5.0, prior_var=1.0)
        ev_far = be.compute_evidence(obs_mean=100.0, obs_var=1.0, prior_mean=5.0, prior_var=1.0)
        assert ev_close > ev_far

    def test_evidence_is_positive(self):
        be = BayesianEstimator()
        ev = be.compute_evidence(0.0, 1.0, 0.0, 1.0)
        assert ev > 0.0

    def test_evidence_with_zero_variances(self):
        be = BayesianEstimator()
        ev = be.compute_evidence(obs_mean=5.0, obs_var=0.0, prior_mean=5.0, prior_var=0.0)
        assert ev > 0.0

    def test_higher_certainty_increases_evidence(self):
        be = BayesianEstimator()
        ev1 = be.compute_evidence(obs_mean=5.0, obs_var=10.0, prior_mean=5.0, prior_var=10.0)
        ev2 = be.compute_evidence(obs_mean=5.0, obs_var=0.1, prior_mean=5.0, prior_var=0.1)
        assert ev2 > ev1


class TestBayesianZScore:
    def test_z_score_95(self):
        be = BayesianEstimator()
        z = be._z_score(0.95)
        assert abs(z - 1.96) < 0.01

    def test_z_score_99(self):
        be = BayesianEstimator()
        z = be._z_score(0.99)
        assert abs(z - 2.576) < 0.02

    def test_z_score_50(self):
        be = BayesianEstimator()
        z = be._z_score(0.5)
        assert abs(z - 0.674) < 0.01
