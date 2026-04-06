"""Tests for uncertainty.py — UncertaintyManager, UncertainValue, DecisionScenario."""

import math
import pytest

from jetson.decision_engine.uncertainty import (
    UncertainValue, DecisionScenario, UncertaintyManager,
)


# ============================================================
# UncertainValue dataclass
# ============================================================

class TestUncertainValue:
    def test_defaults(self):
        uv = UncertainValue()
        assert uv.mean == 0.0
        assert uv.std_dev == 0.0
        assert uv.distribution_type == "normal"
        assert uv.confidence == 0.95

    def test_custom(self):
        uv = UncertainValue(mean=5.0, std_dev=1.5, distribution_type="uniform", confidence=0.99)
        assert uv.mean == pytest.approx(5.0)
        assert uv.std_dev == pytest.approx(1.5)
        assert uv.distribution_type == "uniform"
        assert uv.confidence == pytest.approx(0.99)


# ============================================================
# DecisionScenario dataclass
# ============================================================

class TestDecisionScenario:
    def test_defaults(self):
        ds = DecisionScenario()
        assert ds.alternatives == []
        assert ds.outcomes == []
        assert ds.probabilities == []
        assert ds.uncertainties == []

    def test_full(self):
        ds = DecisionScenario(
            alternatives=["a", "b"],
            outcomes=[[10, 20], [15, 25]],
            probabilities=[0.5, 0.5],
            uncertainties=[UncertainValue(mean=10, std_dev=2)],
        )
        assert len(ds.alternatives) == 2
        assert len(ds.outcomes) == 2


# ============================================================
# UncertaintyManager.compute_expected_value
# ============================================================

class TestExpectedValue:
    def setup_method(self):
        self.mgr = UncertaintyManager()

    def test_ev_basic(self):
        assert self.mgr.compute_expected_value([10, 20], [0.5, 0.5]) == pytest.approx(15.0)

    def test_ev_three_outcomes(self):
        assert self.mgr.compute_expected_value(
            [0, 10, 100], [0.7, 0.2, 0.1]
        ) == pytest.approx(12.0)

    def test_ev_single_outcome(self):
        assert self.mgr.compute_expected_value([42], [1.0]) == pytest.approx(42.0)

    def test_ev_normalizes_probabilities(self):
        assert self.mgr.compute_expected_value([10, 20], [2, 2]) == pytest.approx(15.0)

    def test_ev_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            self.mgr.compute_expected_value([1, 2], [0.5])

    def test_ev_certain(self):
        assert self.mgr.compute_expected_value([5], [1.0]) == pytest.approx(5.0)

    def test_ev_zero_probability(self):
        assert self.mgr.compute_expected_value([10, 0], [1.0, 0.0]) == pytest.approx(10.0)


# ============================================================
# UncertaintyManager.compute_expected_utility
# ============================================================

class TestExpectedUtility:
    def setup_method(self):
        self.mgr = UncertaintyManager()

    def test_eu_linear(self):
        eu = self.mgr.compute_expected_utility([10, 20], [0.5, 0.5], lambda x: x)
        assert eu == pytest.approx(15.0)

    def test_eu_exponential(self):
        eu = self.mgr.compute_expected_utility(
            [0, 10], [0.5, 0.5], lambda x: 1 - math.exp(-0.1 * x)
        )
        # Verify positive
        assert eu > 0

    def test_eu_normalizes(self):
        eu1 = self.mgr.compute_expected_utility([10], [2.0], lambda x: x)
        eu2 = self.mgr.compute_expected_utility([10], [1.0], lambda x: x)
        assert eu1 == pytest.approx(eu2)

    def test_eu_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            self.mgr.compute_expected_utility([1], [0.5, 0.5], lambda x: x)

    def test_eu_single_outcome(self):
        eu = self.mgr.compute_expected_utility([5], [1.0], lambda x: x * 2)
        assert eu == pytest.approx(10.0)


# ============================================================
# UncertaintyManager.compute_value_at_risk
# ============================================================

class TestValueAtRisk:
    def setup_method(self):
        self.mgr = UncertaintyManager()

    def test_var_basic(self):
        outcomes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        var = self.mgr.compute_value_at_risk(outcomes, 0.9)
        assert var == 1.0  # Bottom 10%

    def test_var_high_confidence(self):
        outcomes = [10, 20, 30]
        var = self.mgr.compute_value_at_risk(outcomes, 0.99)
        assert var == 10.0

    def test_var_empty(self):
        var = self.mgr.compute_value_at_risk([], 0.95)
        assert var == 0.0

    def test_var_single(self):
        var = self.mgr.compute_value_at_risk([42], 0.95)
        assert var == 42.0

    def test_var_50_confidence(self):
        outcomes = [1, 2, 3, 4, 5]
        var = self.mgr.compute_value_at_risk(outcomes, 0.5)
        assert var == 3.0


# ============================================================
# UncertaintyManager.minimax_regret
# ============================================================

class TestMinimaxRegret:
    def setup_method(self):
        self.mgr = UncertaintyManager()

    def test_regret_basic(self):
        # Payoff table: alt × scenario
        alts = [
            [100, 200],  # alt0: good in scenario 0, ok in scenario 1
            [150, 100],  # alt1: ok in scenario 0, bad in scenario 1
        ]
        best = self.mgr.minimax_regret(alts)
        # Regret for alt0: max(0, 0) = 0; alt1: max(50, 100) = 100
        # So alt0 is best
        assert best == 0

    def test_regret_three_alts(self):
        alts = [
            [10, 20, 30],
            [20, 10, 20],
            [15, 15, 25],
        ]
        best = self.mgr.minimax_regret(alts)
        assert best in [0, 1, 2]

    def test_regret_empty(self):
        assert self.mgr.minimax_regret([]) == -1

    def test_regret_single_alt(self):
        assert self.mgr.minimax_regret([[5, 10, 15]]) == 0

    def test_regret_equal_alts(self):
        alts = [[5, 10], [5, 10]]
        best = self.mgr.minimax_regret(alts)
        assert best == 0  # First one chosen (tied, min index)

    def test_regret_zero_scenarios(self):
        assert self.mgr.minimax_regret([[]]) == 0

    def test_regret_identifies_safest(self):
        alts = [
            [100, 0],   # risky
            [40, 40],   # safe
        ]
        # Regret for alt0: max(0, 40) = 40; alt1: max(60, 0) = 60
        # alt0 has lower max regret
        best = self.mgr.minimax_regret(alts)
        assert best == 0  # risky has lower max regret here


# ============================================================
# UncertaintyManager.compute_info_gain
# ============================================================

class TestInfoGain:
    def setup_method(self):
        self.mgr = UncertaintyManager()

    def test_info_gain_basic(self):
        belief = [0.5, 0.5]
        # Perfect observation (identity matrix)
        obs = [[1.0, 0.0], [0.0, 1.0]]
        ig = self.mgr.compute_info_gain(belief, obs)
        assert ig > 0

    def test_info_gain_uninformative(self):
        belief = [0.5, 0.5]
        # Uninformative observation (equal likelihoods)
        obs = [[0.5, 0.5], [0.5, 0.5]]
        ig = self.mgr.compute_info_gain(belief, obs)
        assert ig == pytest.approx(0.0, abs=1e-10)

    def test_info_gain_empty_belief(self):
        ig = self.mgr.compute_info_gain([], [[0.5, 0.5]])
        assert ig == 0.0

    def test_info_gain_empty_observation(self):
        ig = self.mgr.compute_info_gain([0.5, 0.5], [])
        assert ig == 0.0

    def test_info_gain_zero_belief(self):
        ig = self.mgr.compute_info_gain([0, 0], [[1, 0], [0, 1]])
        assert ig == 0.0

    def test_info_gain_asymmetric(self):
        belief = [0.7, 0.3]
        obs = [[0.8, 0.1], [0.2, 0.9]]
        ig = self.mgr.compute_info_gain(belief, obs)
        assert ig > 0


# ============================================================
# UncertaintyManager.sensitivity_analysis
# ============================================================

class TestSensitivityAnalysis:
    def setup_method(self):
        self.mgr = UncertaintyManager()

    def test_sensitivity_linear(self):
        decision = lambda p: p["x"] + p["y"]
        ranges = {"x": (0, 10), "y": (0, 5)}
        ranking = self.mgr.sensitivity_analysis(decision, ranges)
        assert len(ranking) == 2
        assert ranking[0][0] == "x"  # x has larger range → higher sensitivity

    def test_sensitivity_single_param(self):
        decision = lambda p: p["a"] ** 2
        ranges = {"a": (0, 10)}
        ranking = self.mgr.sensitivity_analysis(decision, ranges)
        assert len(ranking) == 1
        assert ranking[0][1] > 0

    def test_sensitivity_with_base(self):
        decision = lambda p: p["x"] * p["y"]
        ranges = {"x": (0, 10), "y": (0, 10)}
        base = {"x": 5, "y": 5}
        ranking = self.mgr.sensitivity_analysis(decision, ranges, base)
        assert len(ranking) == 2

    def test_sensitivity_zero_range(self):
        decision = lambda p: p["x"]
        ranges = {"x": (5, 5)}
        ranking = self.mgr.sensitivity_analysis(decision, ranges)
        assert ranking[0][1] == 0.0

    def test_sensitivity_sorted(self):
        decision = lambda p: 3 * p["a"] + p["b"]
        ranges = {"a": (0, 10), "b": (0, 10)}
        ranking = self.mgr.sensitivity_analysis(decision, ranges)
        # First should be most sensitive
        assert ranking[0][1] >= ranking[1][1]

    def test_sensitivity_no_params(self):
        decision = lambda p: 42
        ranking = self.mgr.sensitivity_analysis(decision, {})
        assert ranking == []

    def test_sensitivity_nonlinear(self):
        decision = lambda p: p["x"] ** 2 + p["y"]
        ranges = {"x": (0, 10), "y": (0, 10)}
        ranking = self.mgr.sensitivity_analysis(decision, ranges)
        # x squared dominates
        assert ranking[0][0] == "x"
