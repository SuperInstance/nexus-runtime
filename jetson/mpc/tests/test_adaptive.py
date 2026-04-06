"""Tests for jetson.mpc.adaptive — 25 tests."""
import math
import pytest
from jetson.mpc.adaptive import (
    HorizonConfig,
    PerformanceMetrics,
    AdaptedParameters,
    AdaptiveController,
)


# ---- Data classes ----

class TestAdaptiveDataClasses:
    def test_horizon_config_defaults(self):
        cfg = HorizonConfig()
        assert cfg.min_horizon == 5
        assert cfg.max_horizon == 50
        assert cfg.adaptation_rate == pytest.approx(0.1)

    def test_horizon_config_custom(self):
        cfg = HorizonConfig(
            min_horizon=3, max_horizon=30,
            adaptation_rate=0.2,
            performance_thresholds={"tracking_error": 0.5},
        )
        assert cfg.min_horizon == 3
        assert cfg.max_horizon == 30
        assert cfg.performance_thresholds["tracking_error"] == 0.5

    def test_performance_metrics_defaults(self):
        pm = PerformanceMetrics()
        assert pm.tracking_error == 0.0
        assert pm.constraint_violations == 0.0
        assert pm.computation_time == 0.0
        assert pm.cost == 0.0

    def test_performance_metrics_custom(self):
        pm = PerformanceMetrics(
            tracking_error=1.5,
            constraint_violations=0.3,
            computation_time=0.02,
            cost=42.0,
        )
        assert pm.tracking_error == 1.5
        assert pm.cost == 42.0

    def test_adapted_parameters_defaults(self):
        ap = AdaptedParameters()
        assert ap.horizon == 10
        assert ap.dt == pytest.approx(0.1)
        assert ap.weights == []

    def test_adapted_parameters_custom(self):
        ap = AdaptedParameters(horizon=20, dt=0.05, weights=[1.0, 2.0])
        assert ap.horizon == 20
        assert ap.weights == [1.0, 2.0]


# ---- AdaptiveController ----

class TestAdaptiveController:
    def setup_method(self):
        self.ctrl = AdaptiveController()

    def test_initial_state(self):
        assert self.ctrl.current_horizon == 10
        assert self.ctrl.current_dt == pytest.approx(0.1)
        assert self.ctrl.current_weights == []
        assert self.ctrl.history == []

    def test_select_horizon_good_performance(self):
        cfg = HorizonConfig(min_horizon=5, max_horizon=20,
                            performance_thresholds={"tracking_error": 1.0})
        pm = PerformanceMetrics(tracking_error=0.1, computation_time=0.001)
        h = self.ctrl.select_horizon(pm, cfg)
        assert cfg.min_horizon <= h <= cfg.max_horizon

    def test_select_horizon_poor_tracking(self):
        cfg = HorizonConfig(min_horizon=5, max_horizon=50,
                            performance_thresholds={"tracking_error": 0.5})
        pm = PerformanceMetrics(tracking_error=5.0)
        h = self.ctrl.select_horizon(pm, cfg)
        assert h >= self.ctrl.current_horizon

    def test_select_horizon_slow_computation(self):
        cfg = HorizonConfig(min_horizon=5, max_horizon=50,
                            performance_thresholds={"computation_time": 0.001})
        pm = PerformanceMetrics(tracking_error=0.01, computation_time=1.0)
        h = self.ctrl.select_horizon(pm, cfg)
        assert h <= self.ctrl.current_horizon

    def test_select_horizon_constraint_violation(self):
        cfg = HorizonConfig(min_horizon=5, max_horizon=50,
                            performance_thresholds={"constraint_violations": 0.01})
        pm = PerformanceMetrics(constraint_violations=5.0)
        h = self.ctrl.select_horizon(pm, cfg)
        assert h >= self.ctrl.current_horizon

    def test_select_horizon_respects_bounds(self):
        cfg = HorizonConfig(min_horizon=8, max_horizon=12)
        pm = PerformanceMetrics(tracking_error=100.0)
        for _ in range(50):
            h = self.ctrl.select_horizon(pm, cfg)
            assert 8 <= h <= 12

    def test_select_horizon_updates_history(self):
        cfg = HorizonConfig()
        pm = PerformanceMetrics()
        self.ctrl.select_horizon(pm, cfg)
        assert len(self.ctrl.history) == 1

    def test_adapt_weights_empty(self):
        result = self.ctrl.adapt_weights(PerformanceMetrics(), [])
        assert result == []

    def test_adapt_weights_increases_on_error(self):
        weights = [1.0, 1.0]
        result = self.ctrl.adapt_weights(
            PerformanceMetrics(tracking_error=5.0),
            weights,
        )
        assert len(result) == 2
        assert self.ctrl.current_weights == result

    def test_adapt_weights_normalised(self):
        weights = [1.0, 1.0, 1.0]
        result = self.ctrl.adapt_weights(
            PerformanceMetrics(tracking_error=1.0),
            weights,
        )
        avg = sum(result) / len(result)
        assert avg == pytest.approx(1.0)

    def test_adapt_weights_preserves_length(self):
        weights = [1.0] * 10
        result = self.ctrl.adapt_weights(PerformanceMetrics(), weights)
        assert len(result) == 10

    def test_should_replan_error(self):
        result = self.ctrl.should_replan(
            PerformanceMetrics(tracking_error=5.0),
            {"tracking_error": 1.0},
        )
        assert result is True

    def test_should_replan_violation(self):
        result = self.ctrl.should_replan(
            PerformanceMetrics(constraint_violations=5.0),
            {"constraint_violations": 0.1},
        )
        assert result is True

    def test_should_replan_cost(self):
        result = self.ctrl.should_replan(
            PerformanceMetrics(cost=500.0),
            {"cost": 100.0},
        )
        assert result is True

    def test_should_replan_good(self):
        result = self.ctrl.should_replan(
            PerformanceMetrics(tracking_error=0.01, constraint_violations=0.0,
                               cost=1.0),
            {"tracking_error": 1.0, "constraint_violations": 0.1, "cost": 100.0},
        )
        assert result is False

    def test_should_replan_empty_thresholds(self):
        result = self.ctrl.should_replan(PerformanceMetrics(), {})
        assert result is False

    def test_compute_optimal_dt_basic(self):
        dt = self.ctrl.compute_optimal_dt(1.0, 10)
        assert 0.01 <= dt <= 1.0
        assert self.ctrl.current_dt == dt

    def test_compute_optimal_dt_high_complexity(self):
        dt = self.ctrl.compute_optimal_dt(100.0, 10)
        assert dt < 0.1  # should be small for high complexity

    def test_compute_optimal_dt_large_horizon(self):
        dt = self.ctrl.compute_optimal_dt(0.1, 50)
        assert dt >= 0.01  # larger horizon allows bigger dt

    def test_tune_parameters_empty_history(self):
        bounds = {"horizon": (5, 20), "dt": (0.01, 0.5)}
        result = self.ctrl.tune_parameters([], bounds)
        assert result["horizon"] == pytest.approx(12.5)  # midpoint
        assert result["dt"] > 0

    def test_tune_parameters_with_history(self):
        bounds = {"horizon": (5, 20), "dt": (0.01, 0.5), "weight_scale": (0.1, 5.0)}
        history = [
            PerformanceMetrics(tracking_error=2.0, constraint_violations=0.5,
                               computation_time=0.01),
        ]
        result = self.ctrl.tune_parameters(history, bounds)
        assert "horizon" in result
        assert "dt" in result
        assert "weight_scale" in result

    def test_tune_parameters_clamped(self):
        bounds = {"horizon": (5, 20)}
        history = [PerformanceMetrics(tracking_error=100.0)]
        result = self.ctrl.tune_parameters(history, bounds)
        assert 5 <= result["horizon"] <= 20

    def test_tune_parameters_high_error(self):
        bounds = {"horizon": (5, 50)}
        history = [PerformanceMetrics(tracking_error=100.0)]
        result = self.ctrl.tune_parameters(history, bounds)
        assert result["horizon"] > 5  # should increase horizon for high error
