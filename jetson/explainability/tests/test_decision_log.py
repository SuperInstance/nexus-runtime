"""Tests for decision logging module."""

import math
import pytest
import time
from jetson.explainability.decision_log import DecisionRecord, DecisionLog


# --- Fixtures ---

@pytest.fixture
def sample_record():
    return DecisionRecord(
        timestamp=time.time(),
        decision_type="navigation",
        input_state={"position": [0, 0], "heading": 90},
        output_action={"thrust": 0.5, "rudder": -10},
        confidence=0.95,
        reasoning="Navigating to waypoint",
        model_version="2.1.0",
        metadata={"entity_id": "vessel_1"},
    )


@pytest.fixture
def log():
    return DecisionLog(max_size=100)


# --- DecisionRecord tests ---

class TestDecisionRecord:
    def test_creation(self):
        r = DecisionRecord(time.time(), "nav", {}, {}, 0.9, "reason")
        assert r.decision_type == "nav"
        assert r.confidence == 0.9

    def test_confidence_clamped_high(self):
        r = DecisionRecord(time.time(), "nav", {}, {}, 1.5, "r")
        assert r.confidence == 1.0

    def test_confidence_clamped_low(self):
        r = DecisionRecord(time.time(), "nav", {}, {}, -0.5, "r")
        assert r.confidence == 0.0

    def test_timestamp_default(self):
        r = DecisionRecord(0, "nav", {}, {}, 0.9, "r")
        assert r.timestamp > 0

    def test_metadata_default(self):
        r = DecisionRecord(time.time(), "nav", {}, {}, 0.9, "r")
        assert r.metadata == {}

    def test_model_version_default(self):
        r = DecisionRecord(time.time(), "nav", {}, {}, 0.9, "r")
        assert r.model_version == "1.0.0"

    def test_all_fields(self, sample_record):
        assert sample_record.decision_type == "navigation"
        assert sample_record.reasoning == "Navigating to waypoint"
        assert sample_record.metadata["entity_id"] == "vessel_1"


# --- DecisionLog tests ---

class TestDecisionLog:
    def test_log_decision(self, log, sample_record):
        log.log_decision(sample_record)
        assert log.size == 1

    def test_log_multiple_decisions(self, log):
        for i in range(10):
            r = DecisionRecord(time.time() + i, "nav", {}, {}, 0.9, f"reason {i}")
            log.log_decision(r)
        assert log.size == 10

    def test_max_size_eviction(self):
        log = DecisionLog(max_size=5)
        for i in range(10):
            r = DecisionRecord(time.time() + i, "nav", {}, {}, 0.9, f"r{i}")
            log.log_decision(r)
        assert log.size == 5

    def test_query_by_type(self, log):
        for i in range(5):
            r = DecisionRecord(time.time() + i, "nav", {}, {}, 0.9, f"r{i}")
            log.log_decision(r)
        r = DecisionRecord(time.time() + 6, "safety", {}, {}, 0.8, "safety")
        log.log_decision(r)
        result = log.query_by_type("nav")
        assert len(result) == 5
        assert all(r.decision_type == "nav" for r in result)

    def test_query_by_type_limit(self, log):
        for i in range(10):
            r = DecisionRecord(time.time() + i, "nav", {}, {}, 0.9, f"r{i}")
            log.log_decision(r)
        result = log.query_by_type("nav", limit=3)
        assert len(result) == 3

    def test_query_by_type_empty(self, log):
        result = log.query_by_type("nonexistent")
        assert result == []

    def test_query_by_type_sorted_recent(self, log):
        for i in range(5):
            r = DecisionRecord(time.time() + i, "nav", {}, {}, 0.9, f"r{i}")
            log.log_decision(r)
        result = log.query_by_type("nav")
        assert result[0].timestamp > result[-1].timestamp

    def test_query_by_time(self, log):
        base = 1000.0
        for i in range(5):
            r = DecisionRecord(base + i * 10, "nav", {}, {}, 0.9, f"r{i}")
            log.log_decision(r)
        result = log.query_by_time(1015, 1035)
        timestamps = [r.timestamp for r in result]
        assert 1020 in timestamps
        assert 1030 in timestamps
        assert 1010 not in timestamps

    def test_query_by_time_no_results(self, log):
        r = DecisionRecord(1000, "nav", {}, {}, 0.9, "r")
        log.log_decision(r)
        result = log.query_by_time(2000, 3000)
        assert result == []

    def test_query_by_time_empty_log(self, log):
        result = log.query_by_time(0, 1000)
        assert result == []

    def test_get_decision_chain(self, log, sample_record):
        for i in range(5):
            r = DecisionRecord(
                time.time() + i,
                "nav",
                {},
                {},
                0.9,
                f"step {i}",
                metadata={"entity_id": "vessel_1"},
            )
            log.log_decision(r)
        chain = log.get_decision_chain("vessel_1")
        assert len(chain) == 5
        # Should be chronological
        for i in range(len(chain) - 1):
            assert chain[i].timestamp <= chain[i + 1].timestamp

    def test_get_decision_chain_no_entity(self, log):
        chain = log.get_decision_chain("nonexistent")
        assert chain == []

    def test_compute_decision_frequency(self, log):
        base = 1000.0
        for i in range(10):
            r = DecisionRecord(base + i, "nav", {}, {}, 0.9, "r")
            log.log_decision(r)
        freq = log.compute_decision_frequency("nav")
        assert freq == pytest.approx(10.0 / 9.0, abs=0.01)

    def test_compute_decision_frequency_no_records(self, log):
        freq = log.compute_decision_frequency("nav")
        assert freq == 0.0

    def test_compute_decision_frequency_zero_duration(self, log):
        r1 = DecisionRecord(1000, "nav", {}, {}, 0.9, "r")
        r2 = DecisionRecord(1000, "nav", {}, {}, 0.9, "r")
        log.log_decision(r1)
        log.log_decision(r2)
        freq = log.compute_decision_frequency("nav")
        assert freq == 2.0

    def test_detect_anomalies_low_confidence(self, log):
        for _ in range(10):
            r = DecisionRecord(time.time(), "nav", {}, {}, 0.95, "normal")
            log.log_decision(r)
        r = DecisionRecord(time.time(), "nav", {}, {}, 0.1, "anomaly")
        log.log_decision(r)
        anomalies = log.detect_anomalies()
        assert len(anomalies) > 0

    def test_detect_anomalies_no_anomalies(self, log):
        for _ in range(10):
            r = DecisionRecord(time.time(), "nav", {}, {}, 0.9, "normal")
            log.log_decision(r)
        anomalies = log.detect_anomalies()
        assert len(anomalies) == 0

    def test_detect_anomalies_empty_log(self, log):
        anomalies = log.detect_anomalies()
        assert anomalies == []

    def test_detect_anomalies_custom_list(self, log):
        records = [
            DecisionRecord(1000, "nav", {}, {}, 0.95, "normal"),
            DecisionRecord(1001, "nav", {}, {}, 0.93, "normal"),
            DecisionRecord(1002, "nav", {}, {}, 0.97, "normal"),
            DecisionRecord(1003, "nav", {}, {}, 0.94, "normal"),
            DecisionRecord(1004, "nav", {}, {}, 0.96, "normal"),
            DecisionRecord(1005, "nav", {}, {}, 0.92, "normal"),
            DecisionRecord(1006, "nav", {}, {}, 0.05, "anomaly"),
        ]
        anomalies = log.detect_anomalies(records)
        assert len(anomalies) >= 1

    def test_export_log_json(self, log, sample_record):
        log.log_decision(sample_record)
        exported = log.export_log("json")
        assert '"decision_type": "navigation"' in exported
        assert '"confidence": 0.95' in exported

    def test_export_log_text(self, log, sample_record):
        log.log_decision(sample_record)
        exported = log.export_log("text")
        assert "navigation" in exported
        assert "0.950" in exported

    def test_export_log_invalid_format(self, log):
        with pytest.raises(ValueError, match="Unsupported export format"):
            log.export_log("xml")

    def test_export_log_empty(self, log):
        exported = log.export_log("json")
        assert exported == "[]"

    def test_compute_statistics_basic(self, log, sample_record):
        log.log_decision(sample_record)
        stats = log.compute_statistics()
        assert stats["total_decisions"] == 1
        assert stats["avg_confidence"] == 0.95

    def test_compute_statistics_multi(self, log):
        for i in range(5):
            r = DecisionRecord(time.time(), "nav", {}, {}, 0.5 + i * 0.1, "r")
            log.log_decision(r)
        stats = log.compute_statistics()
        assert stats["total_decisions"] == 5
        assert stats["avg_confidence"] == pytest.approx(0.7, abs=0.01)
        assert stats["min_confidence"] == 0.5
        assert stats["max_confidence"] == 0.9

    def test_compute_statistics_empty(self, log):
        stats = log.compute_statistics()
        assert stats["total_decisions"] == 0
        assert stats["avg_confidence"] == 0.0

    def test_compute_statistics_with_period(self, log):
        now = time.time()
        old = DecisionRecord(now - 100, "nav", {}, {}, 0.9, "old")
        recent = DecisionRecord(now - 1, "nav", {}, {}, 0.5, "recent")
        log.log_decision(old)
        log.log_decision(recent)
        stats = log.compute_statistics(period=10)
        assert stats["total_decisions"] == 1

    def test_compute_statistics_decision_types(self, log):
        r1 = DecisionRecord(1000, "nav", {}, {}, 0.9, "r")
        r2 = DecisionRecord(1001, "safety", {}, {}, 0.8, "r")
        r3 = DecisionRecord(1002, "nav", {}, {}, 0.7, "r")
        log.log_decision(r1)
        log.log_decision(r2)
        log.log_decision(r3)
        stats = log.compute_statistics()
        assert stats["decision_types"]["nav"] == 2
        assert stats["decision_types"]["safety"] == 1

    def test_compute_statistics_model_versions(self, log):
        r1 = DecisionRecord(1000, "nav", {}, {}, 0.9, "r", model_version="1.0")
        r2 = DecisionRecord(1001, "nav", {}, {}, 0.9, "r", model_version="2.0")
        log.log_decision(r1)
        log.log_decision(r2)
        stats = log.compute_statistics()
        assert stats["model_versions"]["1.0"] == 1
        assert stats["model_versions"]["2.0"] == 1

    def test_size_property(self, log):
        assert log.size == 0
        log.log_decision(DecisionRecord(time.time(), "nav", {}, {}, 0.9, "r"))
        assert log.size == 1

    def test_records_property(self, log, sample_record):
        log.log_decision(sample_record)
        records = log.records
        assert len(records) == 1
        assert records[0].decision_type == "navigation"

    def test_records_is_copy(self, log, sample_record):
        log.log_decision(sample_record)
        records = log.records
        records.clear()
        assert log.size == 1  # original unchanged

    def test_entity_chain_mixed_entities(self, log):
        for i in range(3):
            r = DecisionRecord(
                time.time() + i, "nav", {}, {}, 0.9, "r",
                metadata={"entity_id": "v1"},
            )
            log.log_decision(r)
        for i in range(2):
            r = DecisionRecord(
                time.time() + 10 + i, "nav", {}, {}, 0.9, "r",
                metadata={"entity_id": "v2"},
            )
            log.log_decision(r)
        assert len(log.get_decision_chain("v1")) == 3
        assert len(log.get_decision_chain("v2")) == 2

    def test_log_decision_without_entity_id(self, log):
        r = DecisionRecord(time.time(), "nav", {}, {}, 0.9, "r")
        log.log_decision(r)
        assert log.size == 1
