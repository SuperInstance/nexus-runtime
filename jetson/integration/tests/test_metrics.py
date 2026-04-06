"""Tests for MetricCollector — Phase 5 Round 10."""

import time
import math
import pytest
from jetson.integration.metrics import Metric, MetricCollector


@pytest.fixture
def collector():
    return MetricCollector()


@pytest.fixture
def populated_collector(collector):
    for i in range(10):
        collector.record_gauge("cpu", float(i), "%")
        collector.record_gauge("mem", 100.0 - float(i), "MB")
    return collector


# === Metric ===

class TestMetric:
    def test_default(self):
        m = Metric(name="x", value=1.0)
        assert m.name == "x"
        assert m.value == 1.0
        assert m.unit == ""
        assert m.tags == {}

    def test_full(self):
        before = time.time()
        m = Metric(name="y", value=42.0, unit="ms",
                   tags={"host": "rpi"})
        after = time.time()
        assert m.unit == "ms"
        assert m.tags["host"] == "rpi"
        assert before <= m.timestamp <= after

    def test_timestamp_auto(self):
        before = time.time()
        m = Metric(name="t", value=0.0)
        after = time.time()
        assert before <= m.timestamp <= after


# === Recording ===

class TestRecording:
    def test_record(self, collector):
        m = Metric(name="a", value=1.0)
        collector.record(m)
        assert len(collector.get_metric("a")) == 1

    def test_record_multiple(self, collector):
        for i in range(5):
            collector.record(Metric(name="a", value=float(i)))
        assert len(collector.get_metric("a")) == 5

    def test_record_gauge(self, collector):
        collector.record_gauge("temp", 72.5, "F")
        metrics = collector.get_metric("temp")
        assert len(metrics) == 1
        assert metrics[0].value == 72.5
        assert metrics[0].unit == "F"

    def test_record_gauge_tags(self, collector):
        collector.record_gauge("lat", 10.0, "ms", {"region": "us"})
        m = collector.get_metric("lat")[0]
        assert m.tags["region"] == "us"

    def test_record_counter(self, collector):
        collector.record_counter("requests")
        collector.record_counter("requests", 5)
        metrics = collector.get_metric("requests")
        assert metrics[0].value == 1.0
        assert metrics[1].value == 6.0

    def test_record_histogram(self, collector):
        collector.record_histogram("latency", 0.5, "s")
        collector.record_histogram("latency", 1.0, "s")
        metrics = collector.get_metric("latency")
        assert len(metrics) == 2
        assert metrics[1].value == 1.0

    def test_record_histogram_tags(self, collector):
        collector.record_histogram("resp", 200, "ms", {"path": "/api"})
        m = collector.get_metric("resp")[0]
        assert m.tags["path"] == "/api"

    def test_record_gauge_default_unit(self, collector):
        collector.record_gauge("x", 1.0)
        m = collector.get_metric("x")[0]
        assert m.unit == ""

    def test_record_gauge_default_tags(self, collector):
        collector.record_gauge("x", 1.0)
        m = collector.get_metric("x")[0]
        assert m.tags == {}


# === Queries ===

class TestQueries:
    def test_get_metric(self, collector):
        collector.record_gauge("a", 1.0)
        collector.record_gauge("a", 2.0)
        assert len(collector.get_metric("a")) == 2

    def test_get_metric_missing(self, collector):
        assert collector.get_metric("ghost") == []

    def test_get_all_metrics(self, populated_collector):
        all_m = populated_collector.get_all_metrics()
        assert "cpu" in all_m
        assert "mem" in all_m

    def test_get_metric_names(self, populated_collector):
        names = populated_collector.get_metric_names()
        assert "cpu" in names
        assert "mem" in names

    def test_empty_collector(self, collector):
        assert collector.get_all_metrics() == {}
        assert collector.get_metric_names() == []


# === Aggregation ===

class TestAggregation:
    def test_aggregate_mean(self, populated_collector):
        val = populated_collector.aggregate("cpu", "mean")
        assert val == pytest.approx(4.5)

    def test_aggregate_sum(self, populated_collector):
        val = populated_collector.aggregate("cpu", "sum")
        assert val == pytest.approx(45.0)

    def test_aggregate_min(self, populated_collector):
        val = populated_collector.aggregate("cpu", "min")
        assert val == 0.0

    def test_aggregate_max(self, populated_collector):
        val = populated_collector.aggregate("cpu", "max")
        assert val == 9.0

    def test_aggregate_count(self, populated_collector):
        val = populated_collector.aggregate("cpu", "count")
        assert val == 10.0

    def test_aggregate_last(self, populated_collector):
        val = populated_collector.aggregate("cpu", "last")
        assert val == 9.0

    def test_aggregate_unknown(self, populated_collector):
        val = populated_collector.aggregate("cpu", "unknown_agg")
        assert val is None

    def test_aggregate_empty(self, collector):
        assert collector.aggregate("ghost", "mean") is None

    def test_aggregate_window(self, collector):
        now = time.time()
        collector.record_gauge("w", 1.0)
        # Insert metric with old timestamp
        old_m = Metric(name="w", value=999.0, timestamp=now - 1000)
        collector.record(old_m)
        # Only recent should be counted
        val = collector.aggregate("w", "sum", window=10.0)
        assert val == 1.0


# === Statistics ===

class TestStatistics:
    def test_basic_stats(self, populated_collector):
        stats = populated_collector.compute_statistics("cpu")
        assert stats["mean"] == pytest.approx(4.5)
        assert stats["min"] == 0.0
        assert stats["max"] == 9.0
        assert stats["count"] == 10.0

    def test_std(self, populated_collector):
        stats = populated_collector.compute_statistics("cpu")
        # std of 0..9 = sqrt((n^2-1)/12) ≈ sqrt(8.25) ≈ 2.872
        assert stats["std"] is not None
        assert stats["std"] > 0

    def test_percentiles(self, populated_collector):
        stats = populated_collector.compute_statistics("cpu")
        assert stats["p50"] is not None
        assert stats["p95"] is not None
        assert stats["p99"] is not None

    def test_empty_stats(self, collector):
        stats = collector.compute_statistics("ghost")
        assert stats["mean"] is None
        assert stats["count"] == 0
        assert stats["min"] is None

    def test_single_value(self, collector):
        collector.record_gauge("s", 5.0)
        stats = collector.compute_statistics("s")
        assert stats["mean"] == 5.0
        assert stats["std"] == 0.0
        assert stats["p50"] == 5.0

    def test_constant_values(self, collector):
        for _ in range(5):
            collector.record_gauge("c", 3.0)
        stats = collector.compute_statistics("c")
        assert stats["std"] == 0.0
        assert stats["p50"] == 3.0


# === Reset ===

class TestReset:
    def test_reset_single(self, collector):
        collector.record_gauge("a", 1.0)
        collector.record_gauge("b", 2.0)
        collector.reset_metrics("a")
        assert collector.get_metric("a") == []
        assert len(collector.get_metric("b")) == 1

    def test_reset_all(self, collector):
        collector.record_gauge("a", 1.0)
        collector.record_gauge("b", 2.0)
        collector.reset_metrics()
        assert collector.get_all_metrics() == {}

    def test_reset_missing(self, collector):
        # Should not raise
        collector.reset_metrics("ghost")

    def test_reset_clears_counters(self, collector):
        collector.record_counter("req", 10)
        collector.reset_metrics("req")
        collector.record_counter("req", 1)
        metrics = collector.get_metric("req")
        assert metrics[0].value == 1.0  # Counter was reset


# === Export ===

class TestExport:
    def test_export_text(self, populated_collector):
        text = populated_collector.export_metrics("text")
        assert "NEXUS Metrics Export" in text
        assert "cpu" in text

    def test_export_json(self, populated_collector):
        import json
        text = populated_collector.export_metrics("json")
        data = json.loads(text)
        assert "cpu" in data
        assert isinstance(data["cpu"], list)

    def test_export_json_structure(self, populated_collector):
        import json
        text = populated_collector.export_metrics("json")
        data = json.loads(text)
        entry = data["cpu"][0]
        assert "value" in entry
        assert "unit" in entry
        assert "timestamp" in entry
        assert "tags" in entry

    def test_export_text_empty(self, collector):
        text = collector.export_metrics("text")
        assert "NEXUS Metrics Export" in text

    def test_export_json_empty(self, collector):
        import json
        text = collector.export_metrics("json")
        data = json.loads(text)
        assert data == {}

    def test_export_text_unit(self, collector):
        collector.record_gauge("t", 72.0, "F")
        text = collector.export_metrics("text")
        assert "F" in text

    def test_export_text_tags(self, collector):
        collector.record_gauge("r", 100, "ms", {"host": "node1"})
        text = collector.export_metrics("text")
        assert "host=node1" in text
