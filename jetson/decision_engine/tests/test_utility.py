"""Tests for utility.py — UtilityTheory, UtilityFunction, RiskAttitude."""

import math
import pytest

from jetson.decision_engine.utility import (
    UtilityFunction, RiskAttitude, UtilityTheory,
)


# ============================================================
# RiskAttitude enum
# ============================================================

class TestRiskAttitude:
    def test_enum_values(self):
        assert RiskAttitude.RISK_AVERSE.value == "risk_averse"
        assert RiskAttitude.RISK_NEUTRAL.value == "risk_neutral"
        assert RiskAttitude.RISK_SEEKING.value == "risk_seeking"

    def test_enum_members(self):
        assert len(RiskAttitude) == 3


# ============================================================
# UtilityFunction dataclass
# ============================================================

class TestUtilityFunction:
    def test_defaults(self):
        uf = UtilityFunction(type="exponential")
        assert uf.type == "exponential"
        assert uf.parameters == {}
        assert uf.domain == (0.0, float("inf"))

    def test_custom(self):
        uf = UtilityFunction(type="power", parameters={"a": 0.5}, domain=(1, 100))
        assert uf.type == "power"
        assert uf.parameters["a"] == pytest.approx(0.5)
        assert uf.domain == (1, 100)


# ============================================================
# UtilityTheory.exponential_utility
# ============================================================

class TestExponentialUtility:
    def test_zero_risk_aversion(self):
        # As risk aversion → 0, U(x) ≈ risk_aversion * x
        u = UtilityTheory.exponential_utility(10, 0.001)
        assert u > 0

    def test_high_risk_aversion(self):
        # For CARA exponential, higher ra saturates faster to 1.0
        u_low = UtilityTheory.exponential_utility(10, 0.1)
        u_high = UtilityTheory.exponential_utility(10, 2.0)
        # Both positive; u_high is closer to 1.0 due to saturation
        assert u_high > u_low
        assert u_high < 1.0

    def test_positive_value(self):
        u = UtilityTheory.exponential_utility(5, 1.0)
        assert 0 < u < 1

    def test_zero_value(self):
        u = UtilityTheory.exponential_utility(0, 1.0)
        assert u == pytest.approx(0.0)

    def test_negative_value(self):
        u = UtilityTheory.exponential_utility(-5, 1.0)
        assert u < 0

    def test_increasing(self):
        u1 = UtilityTheory.exponential_utility(1, 1.0)
        u2 = UtilityTheory.exponential_utility(2, 1.0)
        assert u2 > u1

    def test_unit_risk_aversion(self):
        u = UtilityTheory.exponential_utility(1, 1.0)
        assert u == pytest.approx(1.0 - math.exp(-1.0))


# ============================================================
# UtilityTheory.power_utility
# ============================================================

class TestPowerUtility:
    def test_risk_neutral(self):
        # risk_aversion=0 → U(x) = x
        u = UtilityTheory.power_utility(10, 0.0)
        assert u == pytest.approx(10.0)

    def test_log_case(self):
        # risk_aversion=1 → U(x) = ln(x)
        u = UtilityTheory.power_utility(math.e, 1.0)
        assert u == pytest.approx(1.0)

    def test_risk_averse(self):
        u1 = UtilityTheory.power_utility(10, 0.5)
        u2 = UtilityTheory.power_utility(10, 0.0)
        # Risk averse: less utility than linear
        assert u1 < u2

    def test_zero_value_log(self):
        u = UtilityTheory.power_utility(0, 1.0)
        assert u == float("-inf")

    def test_negative_value(self):
        # For exponent > 0, negative values give -inf
        u = UtilityTheory.power_utility(-1, 0.5)
        assert u == float("-inf")

    def test_increasing(self):
        u1 = UtilityTheory.power_utility(5, 0.5)
        u2 = UtilityTheory.power_utility(10, 0.5)
        assert u2 > u1

    def test_high_risk_aversion(self):
        u = UtilityTheory.power_utility(10, 2.0)
        # 10^(-1) / (-1) = -0.1
        assert u == pytest.approx(-0.1)


# ============================================================
# UtilityTheory.compute_certainty_equivalent
# ============================================================

class TestCertaintyEquivalent:
    def test_ce_risk_neutral(self):
        # For linear utility, CE = EV
        outcomes = [0, 100]
        probs = [0.5, 0.5]
        ce = UtilityTheory.compute_certainty_equivalent(
            lambda x: x, outcomes, probs
        )
        assert ce == pytest.approx(50.0, abs=1e-3)

    def test_ce_risk_averse(self):
        outcomes = [0, 100]
        probs = [0.5, 0.5]
        u = lambda x: 1 - math.exp(-0.01 * x)
        ce = UtilityTheory.compute_certainty_equivalent(u, outcomes, probs)
        # CE < EV for risk averse
        assert ce < 50.0
        assert ce > 0

    def test_ce_certain_outcome(self):
        outcomes = [50]
        probs = [1.0]
        ce = UtilityTheory.compute_certainty_equivalent(
            lambda x: x, outcomes, probs
        )
        assert ce == pytest.approx(50.0, abs=1e-3)

    def test_ce_empty(self):
        ce = UtilityTheory.compute_certainty_equivalent(lambda x: x, [], [])
        assert ce == 0.0

    def test_ce_normalizes_probabilities(self):
        outcomes = [0, 100]
        probs = [1, 1]  # sum = 2
        ce = UtilityTheory.compute_certainty_equivalent(
            lambda x: x, outcomes, probs
        )
        assert ce == pytest.approx(50.0, abs=1e-3)

    def test_ce_high_risk_aversion(self):
        outcomes = [0, 200]
        probs = [0.5, 0.5]
        u_low = lambda x: 1 - math.exp(-0.01 * x)
        u_high = lambda x: 1 - math.exp(-0.05 * x)
        ce_low = UtilityTheory.compute_certainty_equivalent(u_low, outcomes, probs)
        ce_high = UtilityTheory.compute_certainty_equivalent(u_high, outcomes, probs)
        assert ce_low > ce_high  # Higher risk aversion → lower CE


# ============================================================
# UtilityTheory.compute_risk_premium
# ============================================================

class TestRiskPremium:
    def test_risk_averse_positive_premium(self):
        rp = UtilityTheory.compute_risk_premium(40.0, 50.0)
        assert rp == pytest.approx(10.0)

    def test_risk_neutral_zero_premium(self):
        rp = UtilityTheory.compute_risk_premium(50.0, 50.0)
        assert rp == pytest.approx(0.0)

    def test_risk_seeking_negative_premium(self):
        rp = UtilityTheory.compute_risk_premium(60.0, 50.0)
        assert rp == pytest.approx(-10.0)

    def test_premium_formula(self):
        ev = 100.0
        ce = 80.0
        assert UtilityTheory.compute_risk_premium(ce, ev) == pytest.approx(20.0)


# ============================================================
# UtilityTheory.compare_lotteries
# ============================================================

class TestCompareLotteries:
    def test_risk_neutral_same_ev(self):
        la = ([0, 100], [0.5, 0.5])
        lb = ([50], [1.0])
        result = UtilityTheory.compare_lotteries(la, lb, RiskAttitude.RISK_NEUTRAL)
        assert result == "tie"

    def test_risk_averse_prefers_certain(self):
        la = ([50], [1.0])
        lb = ([0, 100], [0.5, 0.5])
        result = UtilityTheory.compare_lotteries(la, lb, RiskAttitude.RISK_AVERSE)
        assert result == "a"

    def test_risk_seeking_prefers_risky(self):
        la = ([50], [1.0])
        lb = ([0, 100], [0.5, 0.5])
        result = UtilityTheory.compare_lotteries(la, lb, RiskAttitude.RISK_SEEKING)
        assert result == "b"

    def test_different_ev_neutral(self):
        la = ([100], [1.0])
        lb = ([50], [1.0])
        result = UtilityTheory.compare_lotteries(la, lb, RiskAttitude.RISK_NEUTRAL)
        assert result == "a"

    def test_normalizes_probabilities(self):
        la = ([100], [2.0])
        lb = ([50], [1.0])
        result = UtilityTheory.compare_lotteries(la, lb, RiskAttitude.RISK_NEUTRAL)
        assert result == "a"

    def test_three_outcome_lottery(self):
        la = ([0, 50, 100], [0.25, 0.50, 0.25])
        lb = ([50], [1.0])
        result = UtilityTheory.compare_lotteries(la, lb, RiskAttitude.RISK_NEUTRAL)
        # EV of a = 0*0.25 + 50*0.5 + 100*0.25 = 50; EV of b = 50
        assert result == "tie"


# ============================================================
# UtilityTheory.construct_utility_function
# ============================================================

class TestConstructUtilityFunction:
    def test_construct_exponential(self):
        points = [
            (0, 0.0),
            (1, 0.5),
            (2, 0.75),
            (3, 0.875),
        ]
        uf = UtilityTheory.construct_utility_function(points)
        assert uf.type == "exponential"
        assert "risk_aversion" in uf.parameters
        assert uf.parameters["risk_aversion"] > 0

    def test_construct_linear_fallback(self):
        # Points that don't fit exponential well
        points = [
            (0, 0.0),
            (10, 10.0),
            (20, 5.0),  # not monotonic
        ]
        uf = UtilityTheory.construct_utility_function(points)
        assert uf.type == "linear"

    def test_construct_single_point(self):
        points = [(5, 0.5)]
        uf = UtilityTheory.construct_utility_function(points)
        assert uf.type == "linear"

    def test_construct_empty_points(self):
        uf = UtilityTheory.construct_utility_function([])
        assert uf.type == "linear"

    def test_construct_domain(self):
        points = [(1, 0.0), (5, 0.5), (10, 1.0)]
        uf = UtilityTheory.construct_utility_function(points)
        assert uf.domain[0] == pytest.approx(1.0)
        assert uf.domain[1] == pytest.approx(10.0)

    def test_construct_perfect_exponential(self):
        b = 0.5
        points = [(x, 1 - math.exp(-b * x)) for x in [1, 2, 3, 4, 5]]
        uf = UtilityTheory.construct_utility_function(points)
        assert uf.type == "exponential"
        assert abs(uf.parameters["risk_aversion"] - b) < 0.1

    def test_construct_r_squared_in_params(self):
        points = [(1, 0.0), (2, 0.5), (3, 0.75), (4, 0.9)]
        uf = UtilityTheory.construct_utility_function(points)
        if uf.type == "exponential":
            assert "r_squared" in uf.parameters
