"""Tests for degradation module — 30+ tests."""

import math
import pytest
from jetson.maintenance.degradation import (
    DegradationModel,
    DegradationCurve,
    DegradationModeler,
)


@pytest.fixture
def modeler():
    return DegradationModeler()


@pytest.fixture
def linear_data():
    """Perfectly linear degradation: 100 -> 55 over t=0..9."""
    times = list(range(10))
    values = [100.0 - i * 5.0 for i in times]
    return [float(t) for t in times], values


@pytest.fixture
def exponential_data():
    """Exponential decay: 100 * exp(-0.1 * t)."""
    times = list(range(10))
    values = [100.0 * math.exp(-0.1 * t) for t in times]
    return [float(t) for t in times], values


# ── DegradationModel ─────────────────────────────────────────

class TestDegradationModel:
    def test_defaults(self):
        dm = DegradationModel()
        assert dm.model_type == ""
        assert dm.parameters == {}
        assert dm.fit_quality == 0.0

    def test_custom(self):
        dm = DegradationModel(model_type="linear",
                               parameters={"slope": -5.0}, fit_quality=0.95)
        assert dm.parameters["slope"] == -5.0


# ── DegradationCurve ─────────────────────────────────────────

class TestDegradationCurve:
    def test_defaults(self):
        dc = DegradationCurve()
        assert dc.time_points == []
        assert dc.health_values == []
        assert dc.model is None

    def test_with_model(self):
        dc = DegradationCurve(
            time_points=[0.0, 1.0],
            health_values=[100.0, 95.0],
            model=DegradationModel(model_type="linear"),
        )
        assert len(dc.time_points) == 2


# ── fit_linear ───────────────────────────────────────────────

class TestFitLinear:
    def test_perfect_fit(self, modeler, linear_data):
        times, values = linear_data
        curve = modeler.fit_linear(times, values)
        assert curve.model is not None
        assert curve.model.model_type == "linear"
        assert curve.model.fit_quality > 0.99

    def test_slope_and_intercept(self, modeler, linear_data):
        times, values = linear_data
        curve = modeler.fit_linear(times, values)
        assert abs(curve.model.parameters["slope"] - (-5.0)) < 1e-9
        assert abs(curve.model.parameters["intercept"] - 100.0) < 1e-9

    def test_noisy_data(self, modeler):
        import random
        random.seed(42)
        times = [float(i) for i in range(20)]
        values = [100.0 - 5.0 * t + random.gauss(0, 1.0) for t in times]
        curve = modeler.fit_linear(times, values)
        assert curve.model.fit_quality > 0.95

    def test_empty_data(self, modeler):
        curve = modeler.fit_linear([], [])
        assert curve.model.fit_quality == 0.0

    def test_single_point(self, modeler):
        curve = modeler.fit_linear([1.0], [50.0])
        assert curve.model.fit_quality == 0.0

    def test_constant_values(self, modeler):
        times = [float(i) for i in range(5)]
        values = [50.0] * 5
        curve = modeler.fit_linear(times, values)
        assert abs(curve.model.parameters["slope"]) < 1e-9
        # R² for constant: should be 1.0 (or near)
        assert curve.model.fit_quality >= 0.99

    def test_data_preserved(self, modeler, linear_data):
        times, values = linear_data
        curve = modeler.fit_linear(times, values)
        assert curve.time_points == times
        assert curve.health_values == values


# ── fit_exponential ──────────────────────────────────────────

class TestFitExponential:
    def test_good_fit(self, modeler, exponential_data):
        times, values = exponential_data
        curve = modeler.fit_exponential(times, values)
        assert curve.model is not None
        assert curve.model.model_type == "exponential"
        assert curve.model.fit_quality > 0.99

    def test_parameters(self, modeler, exponential_data):
        times, values = exponential_data
        curve = modeler.fit_exponential(times, values)
        params = curve.model.parameters
        assert abs(params["a"] - 100.0) < 1.0
        assert abs(params["b"] - (-0.1)) < 0.01

    def test_negative_values(self, modeler):
        times = [float(i) for i in range(5)]
        values = [-1.0] * 5
        curve = modeler.fit_exponential(times, values)
        assert curve.model.fit_quality == 0.0

    def test_zero_values(self, modeler):
        times = [float(i) for i in range(5)]
        values = [0.0] * 5
        curve = modeler.fit_exponential(times, values)
        assert curve.model.fit_quality == 0.0

    def test_empty_data(self, modeler):
        curve = modeler.fit_exponential([], [])
        assert curve.model.fit_quality == 0.0

    def test_single_point(self, modeler):
        curve = modeler.fit_exponential([1.0], [10.0])
        # Should produce something even if low quality
        assert curve.model.model_type == "exponential"

    def test_data_preserved(self, modeler, exponential_data):
        times, values = exponential_data
        curve = modeler.fit_exponential(times, values)
        assert curve.time_points == times
        assert curve.health_values == values


# ── fit_weibull ──────────────────────────────────────────────

class TestFitWeibull:
    def test_degradation_shape(self, modeler):
        times = [float(i) for i in range(1, 21)]
        values = [100.0 * math.exp(-0.05 * t) for t in times]
        curve = modeler.fit_weibull(times, values)
        assert curve.model is not None
        assert curve.model.model_type == "weibull"
        assert curve.model.fit_quality >= 0.0

    def test_has_parameters(self, modeler):
        times = [float(i) for i in range(1, 11)]
        values = [100.0 - i * 5.0 for i in range(1, 11)]
        curve = modeler.fit_weibull(times, values)
        assert "k" in curve.model.parameters
        assert "lambda" in curve.model.parameters

    def test_empty_data(self, modeler):
        curve = modeler.fit_weibull([], [])
        assert curve.model.fit_quality == 0.0

    def test_single_point(self, modeler):
        curve = modeler.fit_weibull([1.0], [50.0])
        assert curve.model.fit_quality == 0.0

    def test_two_points(self, modeler):
        curve = modeler.fit_weibull([0.0, 10.0], [100.0, 50.0])
        assert curve.model is not None
        assert curve.model.model_type == "weibull"

    def test_data_preserved(self, modeler):
        times = [float(i) for i in range(5)]
        values = [100.0 - i * 10.0 for i in range(5)]
        curve = modeler.fit_weibull(times, values)
        assert curve.time_points == times
        assert curve.health_values == values

    def test_all_same_values(self, modeler):
        times = [float(i) for i in range(10)]
        values = [50.0] * 10
        curve = modeler.fit_weibull(times, values)
        # rng will be 0, all normalized to 0 -> special handling
        assert curve.model is not None


# ── select_best_model ────────────────────────────────────────

class TestSelectBestModel:
    def test_selects_highest_r2(self, modeler, linear_data):
        times, values = linear_data
        c_lin = modeler.fit_linear(times, values)
        c_exp = modeler.fit_exponential(times, values)
        best = modeler.select_best_model([c_lin, c_exp])
        assert best.model.fit_quality >= c_lin.model.fit_quality
        assert best.model.fit_quality >= c_exp.model.fit_quality

    def test_single_curve(self, modeler, linear_data):
        times, values = linear_data
        c = modeler.fit_linear(times, values)
        best = modeler.select_best_model([c])
        assert best.model.model_type == "linear"

    def test_empty_list(self, modeler):
        best = modeler.select_best_model([])
        assert best.time_points == []
        assert best.model is None

    def test_none_models_skipped(self, modeler):
        curves = [DegradationCurve(), DegradationCurve(time_points=[1.0], health_values=[1.0])]
        best = modeler.select_best_model(curves)
        assert best is not None


# ── predict_future ───────────────────────────────────────────

class TestPredictFuture:
    def test_linear_extrapolation(self, modeler, linear_data):
        times, values = linear_data
        curve = modeler.fit_linear(times, values)
        future = [10.0, 11.0, 12.0]
        predicted = modeler.predict_future(curve, future)
        # Should continue declining: 50, 45, 40
        assert len(predicted) == 3
        assert abs(predicted[0] - 50.0) < 1e-6
        assert abs(predicted[1] - 45.0) < 1e-6

    def test_exponential_extrapolation(self, modeler, exponential_data):
        times, values = exponential_data
        curve = modeler.fit_exponential(times, values)
        future = [10.0, 11.0]
        predicted = modeler.predict_future(curve, future)
        assert len(predicted) == 2
        # Values should be positive and decreasing
        assert all(p > 0 for p in predicted)
        assert predicted[0] > predicted[1]

    def test_no_model_returns_zeros(self, modeler):
        curve = DegradationCurve()
        predicted = modeler.predict_future(curve, [1.0, 2.0])
        assert predicted == [0.0, 0.0]

    def test_empty_future(self, modeler, linear_data):
        times, values = linear_data
        curve = modeler.fit_linear(times, values)
        predicted = modeler.predict_future(curve, [])
        assert predicted == []

    def test_weibull_prediction(self, modeler):
        times = [float(i) for i in range(1, 11)]
        values = [100.0 - i * 5.0 for i in range(1, 11)]
        curve = modeler.fit_weibull(times, values)
        predicted = modeler.predict_future(curve, [20.0])
        assert len(predicted) == 1
        assert isinstance(predicted[0], float)


# ── compute_residuals ────────────────────────────────────────

class TestComputeResiduals:
    def test_perfect_fit(self, modeler):
        obs = [10.0, 20.0, 30.0]
        pred = [10.0, 20.0, 30.0]
        res = modeler.compute_residuals(obs, pred)
        assert all(abs(r) < 1e-9 for r in res)

    def test_offset(self, modeler):
        obs = [10.0, 20.0, 30.0]
        pred = [12.0, 22.0, 32.0]
        res = modeler.compute_residuals(obs, pred)
        assert all(abs(r - (-2.0)) < 1e-9 for r in res)

    def test_length_mismatch_raises(self, modeler):
        with pytest.raises(ValueError):
            modeler.compute_residuals([1.0, 2.0], [1.0])

    def test_empty(self, modeler):
        res = modeler.compute_residuals([], [])
        assert res == []

    def test_single_point(self, modeler):
        res = modeler.compute_residuals([5.0], [3.0])
        assert res == [2.0]
