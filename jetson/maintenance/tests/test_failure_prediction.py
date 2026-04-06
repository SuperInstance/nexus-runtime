"""Tests for failure_prediction module — 30+ tests."""

import math
import pytest
from jetson.maintenance.failure_prediction import (
    TimeSeriesPoint,
    PredictionResult,
    FailurePredictor,
)


@pytest.fixture
def predictor():
    return FailurePredictor()


@pytest.fixture
def linear_declining():
    """Values decreasing linearly from 100 to ~55."""
    return [
        TimeSeriesPoint(timestamp=float(i), value=100.0 - i * 5.0)
        for i in range(10)
    ]


@pytest.fixture
def stable_readings():
    return [
        TimeSeriesPoint(timestamp=float(i), value=50.0 + (i % 3) * 0.01)
        for i in range(20)
    ]


# ── TimeSeriesPoint ──────────────────────────────────────────

class TestTimeSeriesPoint:
    def test_defaults(self):
        p = TimeSeriesPoint(timestamp=1.0, value=42.0)
        assert p.metadata == {}

    def test_with_metadata(self):
        p = TimeSeriesPoint(timestamp=1.0, value=42.0, metadata={"loc": "engine"})
        assert p.metadata["loc"] == "engine"


# ── PredictionResult ─────────────────────────────────────────

class TestPredictionResult:
    def test_defaults(self):
        r = PredictionResult()
        assert r.predicted_failure_time is None
        assert r.confidence == 0.0
        assert r.contributing_factors == []

    def test_custom(self):
        r = PredictionResult(
            predicted_failure_time=100.0, confidence=0.9,
            contributing_factors=["declining_trend"],
        )
        assert r.confidence == 0.9


# ── analyze_trend ────────────────────────────────────────────

class TestAnalyzeTrend:
    def test_empty_readings(self, predictor):
        res = predictor.analyze_trend([])
        assert res.confidence == 0.0
        assert res.predicted_failure_time is None

    def test_single_reading(self, predictor):
        res = predictor.analyze_trend([TimeSeriesPoint(0.0, 50.0)])
        assert res.confidence == 0.0

    def test_declining_trend(self, predictor, linear_declining):
        res = predictor.analyze_trend(linear_declining, failure_threshold=0.0)
        assert res.predicted_failure_time is not None
        assert res.predicted_failure_time > 9.0  # after last reading
        assert "declining_trend" in res.contributing_factors

    def test_stable_data(self, predictor, stable_readings):
        res = predictor.analyze_trend(stable_readings)
        assert res.confidence < 0.5  # very flat slope

    def test_rising_trend(self, predictor):
        readings = [TimeSeriesPoint(float(i), 10.0 + i * 5.0) for i in range(10)]
        res = predictor.analyze_trend(readings, failure_threshold=200.0)
        assert res.predicted_failure_time is not None
        assert res.predicted_failure_time > 9.0

    def test_high_confidence_linear(self, predictor):
        # Perfect linear data
        readings = [TimeSeriesPoint(float(i), 100.0 - i * 10.0) for i in range(10)]
        res = predictor.analyze_trend(readings)
        assert res.confidence > 0.99

    def test_failure_time_none_when_past(self, predictor):
        readings = [TimeSeriesPoint(float(i), 10.0 + i * 5.0) for i in range(10)]
        # threshold below current range
        res = predictor.analyze_trend(readings, failure_threshold=-100.0)
        assert res.predicted_failure_time is None


# ── detect_anomalies ─────────────────────────────────────────

class TestDetectAnomalies:
    def test_empty(self, predictor):
        assert predictor.detect_anomalies([]) == []

    def test_no_anomalies(self, predictor, stable_readings):
        result = predictor.detect_anomalies(stable_readings, sensitivity=3.0)
        assert len(result) == 0

    def test_obvious_anomaly(self, predictor):
        readings = [TimeSeriesPoint(float(i), 50.0) for i in range(10)]
        readings.append(TimeSeriesPoint(10.0, 500.0))
        result = predictor.detect_anomalies(readings, sensitivity=2.0)
        assert len(result) >= 1
        assert result[0]["value"] == 500.0

    def test_anomaly_structure(self, predictor):
        readings = [TimeSeriesPoint(float(i), 50.0) for i in range(10)]
        readings.append(TimeSeriesPoint(10.0, 500.0))
        result = predictor.detect_anomalies(readings, sensitivity=2.0)
        anom = result[0]
        assert "index" in anom
        assert "timestamp" in anom
        assert "value" in anom
        assert "z_score" in anom
        assert "mean" in anom
        assert "std" in anom

    def test_low_sensitivity_catches_more(self, predictor):
        readings = [TimeSeriesPoint(float(i), 50.0) for i in range(20)]
        readings.append(TimeSeriesPoint(20.0, 60.0))
        high_s = predictor.detect_anomalies(readings, sensitivity=3.0)
        low_s = predictor.detect_anomalies(readings, sensitivity=1.0)
        assert len(low_s) >= len(high_s)

    def test_single_reading_no_anomaly(self, predictor):
        result = predictor.detect_anomalies(
            [TimeSeriesPoint(0.0, 42.0)], sensitivity=2.0
        )
        assert result == []

    def test_all_same_values(self, predictor):
        readings = [TimeSeriesPoint(float(i), 42.0) for i in range(10)]
        result = predictor.detect_anomalies(readings, sensitivity=2.0)
        assert len(result) == 0


# ── compute_failure_probability ──────────────────────────────

class TestComputeFailureProbability:
    def test_empty(self, predictor):
        assert predictor.compute_failure_probability([]) == 0.0

    def test_linear_model(self, predictor, linear_declining):
        prob = predictor.compute_failure_probability(linear_declining, model="linear")
        assert 0.0 <= prob <= 1.0

    def test_threshold_model_high_deviation(self, predictor):
        # Many stable readings plus one outlier -> high z-score
        readings = [TimeSeriesPoint(float(i), 50.0) for i in range(20)]
        readings.append(TimeSeriesPoint(20.0, 200.0))
        prob = predictor.compute_failure_probability(readings, model="threshold")
        assert prob > 0.5

    def test_threshold_model_low_deviation(self, predictor, stable_readings):
        prob = predictor.compute_failure_probability(stable_readings, model="threshold")
        assert prob < 0.5

    def test_single_reading_linear(self, predictor):
        prob = predictor.compute_failure_probability(
            [TimeSeriesPoint(0.0, 50.0)], model="linear",
        )
        assert prob == 0.0


# ── multi_sensor_fusion ──────────────────────────────────────

class TestMultiSensorFusion:
    def test_empty(self, predictor):
        res = predictor.multi_sensor_fusion({})
        assert res.confidence == 0.0

    def test_single_sensor(self, predictor):
        preds = {"s1": PredictionResult(predicted_failure_time=100.0, confidence=0.8)}
        res = predictor.multi_sensor_fusion(preds)
        assert res.predicted_failure_time == 100.0
        assert res.confidence == 0.8

    def test_multiple_sensors_equal(self, predictor):
        preds = {
            "s1": PredictionResult(predicted_failure_time=100.0, confidence=0.8),
            "s2": PredictionResult(predicted_failure_time=200.0, confidence=0.6),
        }
        res = predictor.multi_sensor_fusion(preds)
        assert res.predicted_failure_time == 150.0
        assert res.confidence == 0.7

    def test_weighted_fusion(self, predictor):
        preds = {
            "s1": PredictionResult(predicted_failure_time=100.0, confidence=0.8),
            "s2": PredictionResult(predicted_failure_time=200.0, confidence=0.6),
        }
        weights = {"s1": 3.0, "s2": 1.0}
        res = predictor.multi_sensor_fusion(preds, weights)
        # weighted avg: (3*100 + 1*200) / 4 = 125
        assert abs(res.predicted_failure_time - 125.0) < 1e-6

    def test_none_failure_times_skipped(self, predictor):
        preds = {
            "s1": PredictionResult(predicted_failure_time=None, confidence=0.5),
            "s2": PredictionResult(predicted_failure_time=200.0, confidence=0.5),
        }
        res = predictor.multi_sensor_fusion(preds)
        assert res.predicted_failure_time == 200.0

    def test_factors_merged(self, predictor):
        preds = {
            "s1": PredictionResult(confidence=0.5, contributing_factors=["a"]),
            "s2": PredictionResult(confidence=0.5, contributing_factors=["b", "a"]),
        }
        res = predictor.multi_sensor_fusion(preds)
        assert "a" in res.contributing_factors
        assert "b" in res.contributing_factors


# ── compute_confidence_interval ──────────────────────────────

class TestComputeConfidenceInterval:
    def test_basic(self, predictor):
        lo, hi = predictor.compute_confidence_interval(100.0, 5.0)
        assert lo < 100.0 < hi

    def test_symmetric(self, predictor):
        lo, hi = predictor.compute_confidence_interval(50.0, 2.0)
        assert abs((50.0 - lo) - (hi - 50.0)) < 1e-9

    def test_custom_z(self, predictor):
        lo1, hi1 = predictor.compute_confidence_interval(0.0, 1.0, z_multiplier=1.0)
        lo2, hi2 = predictor.compute_confidence_interval(0.0, 1.0, z_multiplier=3.0)
        assert (hi2 - lo2) > (hi1 - lo1)

    def test_zero_uncertainty(self, predictor):
        lo, hi = predictor.compute_confidence_interval(42.0, 0.0)
        assert lo == hi == 42.0
