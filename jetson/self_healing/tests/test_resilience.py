"""Tests for resilience module — 38 tests."""

import pytest

from jetson.self_healing.resilience import (
    MetricTrend,
    ResilienceMetric,
    ResilienceReport,
    SystemResilience,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def resilience():
    return SystemResilience()


@pytest.fixture
def sample_metrics():
    return [
        ResilienceMetric(name="availability", value=99.5, target=99.9),
        ResilienceMetric(name="mttr", value=45.0, target=30.0),
        ResilienceMetric(name="throughput", value=1000.0, target=1200.0),
    ]


# ── ResilienceMetric ─────────────────────────────────────────────────────

class TestResilienceMetric:
    def test_fields(self):
        m = ResilienceMetric(name="test", value=42.0, target=100.0)
        assert m.name == "test"
        assert m.value == 42.0
        assert m.target == 100.0
        assert m.trend == MetricTrend.UNKNOWN

    def test_timestamp_auto(self):
        import time
        before = time.time()
        m = ResilienceMetric(name="x", value=1.0)
        after = time.time()
        assert before <= m.timestamp <= after

    def test_unit_field(self):
        m = ResilienceMetric(name="x", value=1.0, unit="ms")
        assert m.unit == "ms"


# ── MTTR / MTBF ──────────────────────────────────────────────────────────

class TestMTTR:
    def test_compute_mttr_basic(self, resilience):
        history = [
            {"time_to_recover": 10.0},
            {"time_to_recover": 20.0},
            {"time_to_recover": 30.0},
        ]
        mttr = resilience.compute_mean_time_to_recovery(history)
        assert mttr == 20.0

    def test_compute_mttr_empty(self, resilience):
        assert resilience.compute_mean_time_to_recovery([]) == 0.0

    def test_compute_mttr_missing_key(self, resilience):
        history = [{"other": 1.0}]
        assert resilience.compute_mean_time_to_recovery(history) == 0.0

    def test_compute_mttr_single(self, resilience):
        assert resilience.compute_mean_time_to_recovery([{"time_to_recover": 5.0}]) == 5.0


class TestMTBF:
    def test_compute_mtbf_basic(self, resilience):
        history = [
            {"timestamp": 0.0},
            {"timestamp": 100.0},
            {"timestamp": 300.0},
        ]
        mtbf = resilience.compute_mean_time_between_failures(history)
        assert mtbf == 150.0  # (100 + 200) / 2

    def test_compute_mtbf_single_event(self, resilience):
        assert resilience.compute_mean_time_between_failures([{"timestamp": 0.0}]) == 0.0

    def test_compute_mtbf_empty(self, resilience):
        assert resilience.compute_mean_time_between_failures([]) == 0.0

    def test_compute_mtbf_unsorted(self, resilience):
        history = [
            {"timestamp": 300.0},
            {"timestamp": 0.0},
            {"timestamp": 100.0},
        ]
        mtbf = resilience.compute_mean_time_between_failures(history)
        assert mtbf == 150.0

    def test_compute_mtbf_missing_timestamps(self, resilience):
        history = [{"other": 1.0}, {"timestamp": 5.0}]
        assert resilience.compute_mean_time_between_failures(history) == 0.0


class TestAvailability:
    def test_high_availability(self, resilience):
        avail = resilience.compute_availability(mtbf=99900.0, mttr=100.0)
        assert avail > 99.0
        assert avail < 100.0

    def test_perfect_availability(self, resilience):
        assert resilience.compute_availability(100.0, 0.0) == 100.0

    def test_zero_mtbf(self, resilience):
        # MTBF=0 means system never stays up → availability = 0
        assert resilience.compute_availability(0.0, 100.0) == 0.0

    def test_both_zero(self, resilience):
        assert resilience.compute_availability(0.0, 0.0) == 100.0

    def test_50_percent(self, resilience):
        avail = resilience.compute_availability(100.0, 100.0)
        assert abs(avail - 50.0) < 0.01


# ── Fault tolerance ──────────────────────────────────────────────────────

class TestFaultTolerance:
    def test_no_failures(self, resilience):
        assert resilience.compute_fault_tolerance(0, 10) == 1.0

    def test_all_failed(self, resilience):
        assert resilience.compute_fault_tolerance(10, 10) == 0.0

    def test_half_failed(self, resilience):
        tol = resilience.compute_fault_tolerance(5, 10)
        assert abs(tol - 0.25) < 0.01

    def test_one_failed(self, resilience):
        tol = resilience.compute_fault_tolerance(1, 10)
        assert abs(tol - 0.81) < 0.01

    def test_zero_total(self, resilience):
        assert resilience.compute_fault_tolerance(0, 0) == 1.0


# ── Redundancy ───────────────────────────────────────────────────────────

class TestRedundancy:
    def test_all_redundant(self, resilience):
        components = [
            {"redundancy_count": 2, "critical": True},
            {"redundancy_count": 1, "critical": False},
        ]
        score = resilience.compute_redundancy_level(components)
        assert score == 1.0  # both redundant, bonus for critical

    def test_no_redundancy(self, resilience):
        components = [
            {"redundancy_count": 0, "critical": True},
            {"redundancy_count": 0, "critical": False},
        ]
        score = resilience.compute_redundancy_level(components)
        assert score == 0.0

    def test_partial_redundancy(self, resilience):
        components = [
            {"redundancy_count": 1, "critical": True},
            {"redundancy_count": 0, "critical": True},
        ]
        score = resilience.compute_redundancy_level(components)
        assert 0.0 < score < 1.0

    def test_empty_components(self, resilience):
        assert resilience.compute_redundancy_level([]) == 0.0

    def test_non_critical_redundant(self, resilience):
        components = [
            {"redundancy_count": 1, "critical": False},
            {"redundancy_count": 0, "critical": False},
        ]
        score = resilience.compute_redundancy_level(components)
        assert score == 0.5


# ── Resilience index ─────────────────────────────────────────────────────

class TestResilienceIndex:
    def test_empty_metrics(self, resilience):
        assert resilience.compute_resilience_index([]) == 0.0

    def test_perfect_metrics(self, resilience):
        metrics = [
            ResilienceMetric("a", 100.0, 100.0),
            ResilienceMetric("b", 50.0, 50.0),
        ]
        idx = resilience.compute_resilience_index(metrics)
        assert abs(idx - 100.0) < 0.01

    def test_half_metrics(self, resilience):
        metrics = [
            ResilienceMetric("a", 50.0, 100.0),
        ]
        idx = resilience.compute_resilience_index(metrics)
        assert abs(idx - 50.0) < 0.01

    def test_improving_trend_boost(self, resilience):
        metrics = [
            ResilienceMetric("a", 50.0, 100.0, trend=MetricTrend.IMPROVING),
        ]
        idx = resilience.compute_resilience_index(metrics)
        assert idx > 50.0

    def test_degrading_trend_penalty(self, resilience):
        metrics = [
            ResilienceMetric("a", 50.0, 100.0, trend=MetricTrend.DEGRADING),
        ]
        idx = resilience.compute_resilience_index(metrics)
        assert idx < 50.0

    def test_over_target_capped(self, resilience):
        metrics = [
            ResilienceMetric("a", 200.0, 100.0),
        ]
        idx = resilience.compute_resilience_index(metrics)
        assert idx <= 100.0


# ── Report generation ────────────────────────────────────────────────────

class TestReport:
    def test_generate_report_basic(self, resilience, sample_metrics):
        report = resilience.generate_resilience_report(sample_metrics)
        assert report.overall_score > 0
        assert report.availability > 0

    def test_generate_report_with_history(self, resilience, sample_metrics):
        failures = [
            {"timestamp": 0.0},
            {"timestamp": 100.0},
        ]
        recoveries = [
            {"time_to_recover": 10.0},
            {"time_to_recover": 15.0},
        ]
        report = resilience.generate_resilience_report(
            sample_metrics, failure_history=failures, recovery_history=recoveries
        )
        assert report.mttr == 12.5
        assert report.mtbf == 100.0

    def test_report_has_recommendations(self, resilience):
        metrics = [ResilienceMetric("availability", 95.0, 99.9)]
        report = resilience.generate_resilience_report(metrics)
        assert len(report.recommendations) > 0

    def test_report_healthy_recommendations(self, resilience):
        metrics = [ResilienceMetric("availability", 99.99, 99.9)]
        report = resilience.generate_resilience_report(
            metrics,
            failure_history=[{"timestamp": 0.0}, {"timestamp": 86400.0}],
            recovery_history=[{"time_to_recover": 1.0}, {"time_to_recover": 1.0}],
            components=[{"redundancy_count": 1, "critical": True}],
        )
        assert "acceptable" in report.recommendations[0].lower()

    def test_report_fault_tolerance_with_components(self, resilience):
        components = [
            {"failed": True},
            {"failed": False},
        ]
        metrics = [ResilienceMetric("x", 50.0, 100.0)]
        report = resilience.generate_resilience_report(metrics, components=components)
        assert report.fault_tolerance < 1.0


# ── Metric tracking ──────────────────────────────────────────────────────

class TestMetricTracking:
    def test_record_metric(self, resilience):
        m = ResilienceMetric("cpu", 75.0)
        resilience.record_metric(m)
        assert resilience.get_metric_trend("cpu") == MetricTrend.UNKNOWN

    def test_get_metric_trend_improving(self, resilience):
        for i in range(10):
            resilience.record_metric(ResilienceMetric("metric", float(i), timestamp=float(i)))
        trend = resilience.get_metric_trend("metric")
        assert trend == MetricTrend.IMPROVING

    def test_get_metric_trend_degrading(self, resilience):
        for i in range(10):
            resilience.record_metric(ResilienceMetric("metric", float(10 - i), timestamp=float(i)))
        trend = resilience.get_metric_trend("metric")
        assert trend == MetricTrend.DEGRADING

    def test_get_metric_trend_stable(self, resilience):
        for i in range(10):
            resilience.record_metric(ResilienceMetric("metric", 50.0, timestamp=float(i)))
        trend = resilience.get_metric_trend("metric")
        assert trend == MetricTrend.STABLE

    def test_get_metric_trend_unknown_not_found(self, resilience):
        assert resilience.get_metric_trend("nonexistent") == MetricTrend.UNKNOWN

    def test_record_failure_and_recovery(self, resilience):
        resilience.record_failure({"timestamp": 1.0})
        resilience.record_recovery({"time_to_recover": 5.0})
        # Should use internal events
        report = resilience.generate_resilience_report([ResilienceMetric("x", 50.0, 100.0)])
        assert report.mttr == 5.0

    def test_metric_history_capped(self, resilience):
        for i in range(600):
            resilience.record_metric(ResilienceMetric("m", float(i), timestamp=float(i)))
        # Should not raise, just cap; use larger window for reliable trend detection
        trend = resilience.get_metric_trend("m", window=100)
        assert trend == MetricTrend.IMPROVING
