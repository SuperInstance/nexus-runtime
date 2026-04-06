"""Tests for remaining_life module — 30+ tests."""

import math
import pytest
from jetson.maintenance.remaining_life import (
    RULEstimate,
    RemainingLifeEstimator,
)


@pytest.fixture
def estimator():
    return RemainingLifeEstimator()


# ── RULEstimate ──────────────────────────────────────────────

class TestRULEstimate:
    def test_defaults(self):
        r = RULEstimate()
        assert r.median == 0.0
        assert r.lower_bound == 0.0
        assert r.upper_bound == 0.0
        assert r.confidence == 0.0
        assert r.method == ""

    def test_custom(self):
        r = RULEstimate(median=100.0, lower_bound=80.0, upper_bound=120.0,
                         confidence=0.9, method="linear")
        assert r.median == 100.0
        assert r.method == "linear"


# ── estimate_linear ──────────────────────────────────────────

class TestEstimateLinear:
    def test_basic(self, estimator):
        rul = estimator.estimate_linear(100.0, 10.0, 0.0)
        assert rul.method == "linear"
        assert rul.median == 10.0
        assert rul.lower_bound == 8.0
        assert rul.upper_bound == 12.0

    def test_zero_rate_returns_inf(self, estimator):
        rul = estimator.estimate_linear(100.0, 0.0, 0.0)
        assert rul.median == float("inf")

    def test_negative_rate_returns_inf(self, estimator):
        rul = estimator.estimate_linear(100.0, -5.0, 0.0)
        assert rul.median == float("inf")

    def test_already_below_threshold(self, estimator):
        rul = estimator.estimate_linear(5.0, 10.0, 10.0)
        assert rul.median == 0.0

    def test_exact_threshold(self, estimator):
        rul = estimator.estimate_linear(50.0, 10.0, 50.0)
        assert rul.median == 0.0

    def test_confidence_value(self, estimator):
        rul = estimator.estimate_linear(100.0, 10.0, 0.0)
        assert rul.confidence == 0.7

    def test_bounds_are_positive(self, estimator):
        rul = estimator.estimate_linear(5.0, 100.0, 0.0)
        assert rul.lower_bound >= 0.0

    def test_large_values(self, estimator):
        rul = estimator.estimate_linear(100000.0, 1.0, 0.0)
        assert rul.median == 100000.0


# ── estimate_exponential ─────────────────────────────────────

class TestEstimateExponential:
    def test_basic(self, estimator):
        rul = estimator.estimate_exponential(100.0, 0.1, 10.0)
        assert rul.method == "exponential"
        assert rul.median > 0.0
        assert rul.upper_bound > rul.median

    def test_zero_rate_returns_inf(self, estimator):
        rul = estimator.estimate_exponential(100.0, 0.0, 10.0)
        assert rul.median == float("inf")

    def test_current_below_threshold(self, estimator):
        rul = estimator.estimate_exponential(5.0, 0.1, 10.0)
        assert rul.median == 0.0

    def test_current_equals_threshold(self, estimator):
        rul = estimator.estimate_exponential(10.0, 0.1, 10.0)
        assert rul.median == 0.0

    def test_zero_threshold(self, estimator):
        rul = estimator.estimate_exponential(100.0, 0.1, 0.0)
        assert rul.median == 0.0

    def test_confidence(self, estimator):
        rul = estimator.estimate_exponential(100.0, 0.1, 10.0)
        assert rul.confidence == 0.75

    def test_negative_rate_returns_inf(self, estimator):
        rul = estimator.estimate_exponential(100.0, -0.1, 10.0)
        assert rul.median == float("inf")

    def test_fast_degradation(self, estimator):
        rul = estimator.estimate_exponential(100.0, 1.0, 10.0)
        rul2 = estimator.estimate_exponential(100.0, 0.01, 10.0)
        assert rul.median < rul2.median


# ── estimate_weibull ─────────────────────────────────────────

class TestEstimateWeibull:
    def test_basic(self, estimator):
        rul = estimator.estimate_weibull(shape=2.0, scale=100.0, current_age=50.0)
        assert rul.method == "weibull"
        assert rul.median >= 0.0

    def test_zero_shape(self, estimator):
        rul = estimator.estimate_weibull(shape=0.0, scale=100.0, current_age=0.0)
        assert rul.confidence == 0.0

    def test_negative_shape(self, estimator):
        rul = estimator.estimate_weibull(shape=-1.0, scale=100.0, current_age=0.0)
        assert rul.confidence == 0.0

    def test_zero_scale(self, estimator):
        rul = estimator.estimate_weibull(shape=2.0, scale=0.0, current_age=0.0)
        assert rul.confidence == 0.0

    def test_age_exceeds_mean(self, estimator):
        rul = estimator.estimate_weibull(shape=2.0, scale=50.0, current_age=1000.0)
        assert rul.median == 0.0

    def test_confidence(self, estimator):
        rul = estimator.estimate_weibull(shape=2.0, scale=100.0, current_age=10.0)
        assert rul.confidence == 0.65

    def test_bounds(self, estimator):
        rul = estimator.estimate_weibull(shape=2.0, scale=100.0, current_age=10.0)
        assert rul.lower_bound >= 0.0
        assert rul.upper_bound >= rul.median

    def test_different_shapes(self, estimator):
        rul1 = estimator.estimate_weibull(shape=1.0, scale=100.0, current_age=0.0)
        rul2 = estimator.estimate_weibull(shape=3.0, scale=100.0, current_age=0.0)
        # Both should have valid median
        assert rul1.median > 0.0
        assert rul2.median > 0.0


# ── ensemble_estimate ────────────────────────────────────────

class TestEnsembleEstimate:
    def test_empty(self, estimator):
        rul = estimator.ensemble_estimate([])
        assert rul.method == "ensemble"
        assert rul.median == 0.0

    def test_single_estimate(self, estimator):
        e = RULEstimate(median=100.0, lower_bound=80.0, upper_bound=120.0,
                         confidence=0.7, method="linear")
        rul = estimator.ensemble_estimate([e])
        assert rul.median == 100.0

    def test_equal_weights(self, estimator):
        e1 = RULEstimate(median=100.0, lower_bound=80.0, upper_bound=120.0,
                          confidence=0.7, method="linear")
        e2 = RULEstimate(median=200.0, lower_bound=160.0, upper_bound=240.0,
                          confidence=0.75, method="exponential")
        rul = estimator.ensemble_estimate([e1, e2])
        assert rul.median == 150.0
        assert "linear" in rul.method
        assert "exponential" in rul.method

    def test_custom_weights(self, estimator):
        e1 = RULEstimate(median=100.0, confidence=0.7, method="linear")
        e2 = RULEstimate(median=200.0, confidence=0.75, method="exp")
        rul = estimator.ensemble_estimate([e1, e2], weights=[3.0, 1.0])
        assert rul.median == 125.0

    def test_zero_weights(self, estimator):
        e1 = RULEstimate(median=100.0, method="a")
        rul = estimator.ensemble_estimate([e1], weights=[0.0])
        assert rul.median == 0.0

    def test_confidence_averaged(self, estimator):
        e1 = RULEstimate(median=100.0, confidence=0.6, method="a")
        e2 = RULEstimate(median=100.0, confidence=0.8, method="b")
        rul = estimator.ensemble_estimate([e1, e2])
        assert abs(rul.confidence - 0.7) < 1e-9

    def test_lower_bound_non_negative(self, estimator):
        e1 = RULEstimate(median=1.0, lower_bound=-5.0, upper_bound=7.0,
                          confidence=1.0, method="a")
        rul = estimator.ensemble_estimate([e1])
        assert rul.lower_bound >= 0.0


# ── update_with_reading ──────────────────────────────────────

class TestUpdateWithReading:
    def test_perfect_reading(self, estimator):
        base = RULEstimate(median=100.0, lower_bound=80.0, upper_bound=120.0,
                           confidence=0.7, method="linear")
        updated = estimator.update_with_reading(base, 50.0, 50.0)
        assert updated.median == base.median

    def test_worse_reading(self, estimator):
        base = RULEstimate(median=100.0, lower_bound=80.0, upper_bound=120.0,
                           confidence=0.7, method="linear")
        updated = estimator.update_with_reading(base, 25.0, 50.0)
        assert updated.median < base.median

    def test_better_reading(self, estimator):
        base = RULEstimate(median=100.0, lower_bound=80.0, upper_bound=120.0,
                           confidence=0.7, method="linear")
        updated = estimator.update_with_reading(base, 75.0, 50.0)
        # capped at 1.0 ratio -> median unchanged
        assert updated.median >= base.median

    def test_zero_expected(self, estimator):
        base = RULEstimate(median=100.0, method="linear")
        updated = estimator.update_with_reading(base, 10.0, 0.0)
        assert updated.median == base.median

    def test_confidence_increases(self, estimator):
        base = RULEstimate(median=100.0, confidence=0.5, method="linear",
                           lower_bound=80.0, upper_bound=120.0)
        updated = estimator.update_with_reading(base, 50.0, 50.0)
        assert updated.confidence >= base.confidence

    def test_method_preserved(self, estimator):
        base = RULEstimate(median=100.0, method="weibull",
                           lower_bound=80.0, upper_bound=120.0)
        updated = estimator.update_with_reading(base, 50.0, 50.0)
        assert "weibull" in updated.method
