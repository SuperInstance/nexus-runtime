"""Tests for health_scoring module — 30+ tests."""

import pytest
from jetson.maintenance.health_scoring import (
    SensorReading,
    EquipmentHealth,
    HealthScorer,
)


@pytest.fixture
def scorer():
    return HealthScorer()


@pytest.fixture
def good_readings():
    return [
        SensorReading(timestamp=i, sensor_id="temp", value=50.0 + (i % 3) * 0.1, unit="C")
        for i in range(10)
    ]


@pytest.fixture
def bad_readings():
    return [
        SensorReading(timestamp=i, sensor_id="temp", value=50.0 + i * 5.0, unit="C")
        for i in range(10)
    ]


# ── SensorReading ────────────────────────────────────────────

class TestSensorReading:
    def test_creation_defaults(self):
        r = SensorReading(timestamp=1.0, sensor_id="s1", value=42.0)
        assert r.unit == ""
        assert r.quality == 1.0

    def test_creation_full(self):
        r = SensorReading(timestamp=1.0, sensor_id="s1", value=42.0, unit="V", quality=0.9)
        assert r.quality == 0.9
        assert r.unit == "V"

    def test_timestamp_float(self):
        r = SensorReading(timestamp=100.5, sensor_id="s1", value=0.0)
        assert isinstance(r.timestamp, float)


# ── EquipmentHealth ──────────────────────────────────────────

class TestEquipmentHealth:
    def test_defaults(self):
        eh = EquipmentHealth(equipment_id="EQ1")
        assert eh.overall_score == 1.0
        assert eh.subsystem_scores == {}
        assert eh.trend == "stable"
        assert eh.last_updated == 0.0

    def test_custom(self):
        eh = EquipmentHealth(
            equipment_id="EQ2", overall_score=0.5,
            subsystem_scores={"motor": 0.6}, trend="degrading",
            last_updated=100.0,
        )
        assert eh.overall_score == 0.5
        assert eh.trend == "degrading"


# ── compute_health_score ─────────────────────────────────────

class TestComputeHealthScore:
    def test_empty_readings_returns_one(self, scorer):
        assert scorer.compute_health_score([], 50.0) == 1.0

    def test_perfect_readings(self, scorer, good_readings):
        score = scorer.compute_health_score(good_readings, 50.0)
        assert score > 0.9

    def test_degraded_readings(self, scorer, bad_readings):
        score = scorer.compute_health_score(bad_readings, 50.0)
        assert score < 0.8

    def test_custom_thresholds(self, scorer):
        readings = [SensorReading(i, "s", 51.0) for i in range(5)]
        score = scorer.compute_health_score(
            readings, 50.0,
            thresholds={"warning": 2.0, "critical": 8.0},
        )
        # 51 is within warning threshold of 2.0
        assert score > 0.95

    def test_critical_deviation(self, scorer):
        readings = [SensorReading(i, "s", 100.0) for i in range(5)]
        score = scorer.compute_health_score(readings, 50.0)
        assert score < 0.5

    def test_quality_reduces_score(self, scorer):
        # Use a value within warning range so scores are non-zero
        readings = [SensorReading(0, "s", 55.0, quality=0.5)]
        score_full = scorer.compute_health_score(
            [SensorReading(0, "s", 55.0)], 50.0
        )
        score_half = scorer.compute_health_score(readings, 50.0)
        assert score_half < score_full

    def test_score_bounded_zero_one(self, scorer):
        readings = [SensorReading(i, "s", -1000.0) for i in range(100)]
        score = scorer.compute_health_score(readings, 50.0)
        assert 0.0 <= score <= 1.0

    def test_zero_baseline(self, scorer):
        readings = [SensorReading(i, "s", 0.01) for i in range(5)]
        score = scorer.compute_health_score(readings, 0.0)
        assert 0.0 <= score <= 1.0


# ── compute_subsystem_health ─────────────────────────────────

class TestComputeSubsystemHealth:
    def test_empty_subsystems(self, scorer):
        result = scorer.compute_subsystem_health({})
        assert result == {}

    def test_single_subsystem(self, scorer):
        readings = [SensorReading(i, "temp", 50.0) for i in range(5)]
        result = scorer.compute_subsystem_health({"motor": readings})
        assert "motor" in result
        assert result["motor"] > 0.9

    def test_multiple_subsystems(self, scorer):
        r1 = [SensorReading(i, "t1", 50.0) for i in range(5)]
        r2 = [SensorReading(i, "t2", 10.0) for i in range(5)]
        result = scorer.compute_subsystem_health({"a": r1, "b": r2})
        assert len(result) == 2

    def test_empty_readings_subsystem(self, scorer):
        result = scorer.compute_subsystem_health({"empty": []})
        assert result["empty"] == 1.0


# ── detect_degradation_trend ─────────────────────────────────

class TestDetectDegradationTrend:
    def test_insufficient_data(self, scorer):
        direction, slope = scorer.detect_degradation_trend([])
        assert direction == "stable"
        assert slope == 0.0

    def test_single_reading(self, scorer):
        r = [SensorReading(0, "s", 50.0)]
        direction, slope = scorer.detect_degradation_trend(r)
        assert direction == "stable"

    def test_degrading_trend(self, scorer):
        readings = [SensorReading(i, "s", 100.0 - i * 10.0) for i in range(10)]
        direction, slope = scorer.detect_degradation_trend(readings)
        assert direction == "degrading"
        assert slope < 0

    def test_improving_trend(self, scorer):
        readings = [SensorReading(i, "s", 10.0 + i * 10.0) for i in range(10)]
        direction, slope = scorer.detect_degradation_trend(readings)
        assert direction == "improving"
        assert slope > 0

    def test_stable_trend(self, scorer):
        readings = [SensorReading(i, "s", 50.0) for i in range(10)]
        direction, slope = scorer.detect_degradation_trend(readings)
        assert direction == "stable"
        assert abs(slope) < 1e-9

    def test_custom_window(self, scorer):
        readings = [SensorReading(i, "s", 100.0 - i * 10.0) for i in range(20)]
        direction, slope = scorer.detect_degradation_trend(readings, window=3)
        assert direction == "degrading"

    def test_small_values(self, scorer):
        readings = [SensorReading(i, "s", 0.001 - i * 0.0001) for i in range(10)]
        direction, _ = scorer.detect_degradation_trend(readings)
        assert direction == "degrading"


# ── aggregate_subsystems ─────────────────────────────────────

class TestAggregateSubsystems:
    def test_empty(self, scorer):
        assert scorer.aggregate_subsystems({}) == 1.0

    def test_equal_weights(self, scorer):
        scores = {"a": 0.8, "b": 0.6}
        result = scorer.aggregate_subsystems(scores)
        assert abs(result - 0.7) < 1e-9

    def test_custom_weights(self, scorer):
        scores = {"a": 1.0, "b": 0.0}
        weights = {"a": 3.0, "b": 1.0}
        result = scorer.aggregate_subsystems(scores, weights)
        assert result == 0.75

    def test_missing_weight_key(self, scorer):
        scores = {"a": 0.5, "b": 0.5}
        weights = {"a": 1.0}
        result = scorer.aggregate_subsystems(scores, weights)
        # b gets weight 0 so falls back to equal average
        assert result == 0.5

    def test_zero_weights(self, scorer):
        scores = {"a": 0.5}
        result = scorer.aggregate_subsystems(scores, {"a": 0.0})
        assert result == 0.5

    def test_bounded_result(self, scorer):
        result = scorer.aggregate_subsystems({"a": 1.2, "b": -0.3})
        assert 0.0 <= result <= 1.0


# ── generate_health_report ───────────────────────────────────

class TestGenerateHealthReport:
    def test_healthy_report(self, scorer):
        eh = EquipmentHealth(equipment_id="EQ1", overall_score=0.95)
        report = scorer.generate_health_report(eh)
        assert report["status"] == "healthy"
        assert report["equipment_id"] == "EQ1"

    def test_warning_report(self, scorer):
        eh = EquipmentHealth(equipment_id="EQ1", overall_score=0.6)
        report = scorer.generate_health_report(eh)
        assert report["status"] == "warning"

    def test_critical_report(self, scorer):
        eh = EquipmentHealth(equipment_id="EQ1", overall_score=0.3)
        report = scorer.generate_health_report(eh)
        assert report["status"] == "critical"

    def test_report_has_recommendations(self, scorer):
        eh = EquipmentHealth(
            equipment_id="EQ1", overall_score=0.3, trend="degrading",
            subsystem_scores={"motor": 0.2},
        )
        report = scorer.generate_health_report(eh)
        assert len(report["recommendations"]) > 0

    def test_report_structure(self, scorer):
        eh = EquipmentHealth(equipment_id="EQ1", last_updated=99.0)
        report = scorer.generate_health_report(eh)
        assert "overall_score" in report
        assert "trend" in report
        assert "subsystem_scores" in report
        assert "last_updated" in report
