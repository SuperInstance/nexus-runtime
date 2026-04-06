"""Tests for attack_detection module."""

import time
import pytest
from jetson.security.attack_detection import (
    AnomalyRecord,
    AttackSignature,
    CommandInjector,
    InjectionMatch,
    InjectionType,
    SensorAnomalyDetector,
    Severity,
)


# ── SensorAnomalyDetector ──────────────────────────────────────────

class TestSensorAnomalyDetectorConstruction:
    def test_construct(self):
        d = SensorAnomalyDetector()
        assert d.get_anomaly_log() == []

    def test_log_initially_empty(self):
        d = SensorAnomalyDetector()
        assert len(d.get_anomaly_log()) == 0


class TestCheckRange:
    def test_normal_value_returns_none(self):
        d = SensorAnomalyDetector()
        assert d.check_range("temp", 25.0, (0.0, 100.0)) is None

    def test_low_violation(self):
        d = SensorAnomalyDetector()
        r = d.check_range("temp", -5.0, (0.0, 100.0))
        assert r is not None
        assert r.anomaly_type == "range_violation"
        assert r.sensor_id == "temp"
        assert r.value == -5.0

    def test_high_violation(self):
        d = SensorAnomalyDetector()
        r = d.check_range("pressure", 200.0, (0.0, 150.0))
        assert r is not None
        assert r.severity == Severity.HIGH

    def test_boundary_low_ok(self):
        d = SensorAnomalyDetector()
        assert d.check_range("x", 0.0, (0.0, 100.0)) is None

    def test_boundary_high_ok(self):
        d = SensorAnomalyDetector()
        assert d.check_range("x", 100.0, (0.0, 100.0)) is None

    def test_boundary_just_below(self):
        d = SensorAnomalyDetector()
        r = d.check_range("x", -0.001, (0.0, 100.0))
        assert r is not None

    def test_negative_range(self):
        d = SensorAnomalyDetector()
        assert d.check_range("depth", -10.0, (-50.0, 0.0)) is None

    def test_negative_range_violation(self):
        d = SensorAnomalyDetector()
        r = d.check_range("depth", -60.0, (-50.0, 0.0))
        assert r is not None

    def test_custom_timestamp(self):
        d = SensorAnomalyDetector()
        ts = 1000.0
        r = d.check_range("t", 999.0, (0.0, 10.0), timestamp=ts)
        assert r is not None
        assert r.timestamp == ts

    def test_violation_logged(self):
        d = SensorAnomalyDetector()
        d.check_range("t", 999.0, (0.0, 10.0))
        assert len(d.get_anomaly_log()) == 1

    def test_multiple_violations(self):
        d = SensorAnomalyDetector()
        d.check_range("a", 200, (0, 100))
        d.check_range("b", -5, (0, 100))
        d.check_range("c", 50, (0, 100))
        assert len(d.get_anomaly_log()) == 2


class TestCheckRate:
    def test_normal_rate(self):
        d = SensorAnomalyDetector()
        assert d.check_rate("alt", 11.0, 10.0, 5.0) is None

    def test_rate_violation(self):
        d = SensorAnomalyDetector()
        r = d.check_rate("alt", 20.0, 10.0, 5.0)
        assert r is not None
        assert r.anomaly_type == "rate_violation"

    def test_rate_exact(self):
        d = SensorAnomalyDetector()
        assert d.check_rate("x", 15.0, 10.0, 5.0) is None

    def test_rate_negative_prev(self):
        d = SensorAnomalyDetector()
        r = d.check_rate("x", 10.0, -10.0, 15.0)
        assert r is not None
        assert r.value == 20.0

    def test_rate_severity_medium(self):
        d = SensorAnomalyDetector()
        r = d.check_rate("x", 100, 0, 1)
        assert r is not None
        assert r.severity == Severity.MEDIUM

    def test_rate_logged(self):
        d = SensorAnomalyDetector()
        d.check_rate("x", 100, 0, 1)
        assert len(d.get_anomaly_log()) == 1

    def test_rate_custom_timestamp(self):
        d = SensorAnomalyDetector()
        ts = 500.0
        r = d.check_rate("x", 100, 0, 1, timestamp=ts)
        assert r.timestamp == ts


class TestCheckConsistency:
    def test_consistent_sensors(self):
        d = SensorAnomalyDetector()
        r = d.check_consistency("a", 10.0, {"b": 10.0, "c": 9.5}, tolerance=1.0)
        assert r is None

    def test_inconsistent_sensor(self):
        d = SensorAnomalyDetector()
        r = d.check_consistency("a", 10.0, {"b": 20.0}, tolerance=1.0)
        assert r is not None
        assert r.anomaly_type == "consistency_violation"

    def test_empty_correlated(self):
        d = SensorAnomalyDetector()
        assert d.check_consistency("a", 10.0, {}) is None

    def test_custom_tolerance(self):
        d = SensorAnomalyDetector()
        r = d.check_consistency("a", 10.0, {"b": 15.0}, tolerance=10.0)
        assert r is None

    def test_first_correlated_violates(self):
        d = SensorAnomalyDetector()
        # diff between a=10.0 and b=5.0 is 5.0 > tolerance 3.0 -> violation
        r = d.check_consistency("a", 10.0, {"b": 5.0, "c": 10.0}, tolerance=3.0)
        assert r is not None
        assert r.anomaly_type == "consistency_violation"

    def test_consistency_logged(self):
        d = SensorAnomalyDetector()
        d.check_consistency("a", 10.0, {"b": 100.0}, tolerance=1.0)
        assert len(d.get_anomaly_log()) == 1


class TestDetectJamming:
    def test_no_jamming(self):
        d = SensorAnomalyDetector()
        assert d.detect_jamming(80.0, 2.0) is False  # SNR=40

    def test_jamming_detected(self):
        d = SensorAnomalyDetector()
        assert d.detect_jamming(5.0, 10.0) is True  # SNR=0.5

    def test_custom_threshold(self):
        d = SensorAnomalyDetector()
        # SNR = 50/10 = 5, threshold=3, 5>=3 so no jamming
        assert d.detect_jamming(50.0, 10.0, threshold=3.0) is False

    def test_jamming_custom_high_threshold(self):
        d = SensorAnomalyDetector()
        # SNR = 2, threshold = 5 -> jamming
        assert d.detect_jamming(2.0, 1.0, threshold=5.0) is True

    def test_zero_noise_floor(self):
        d = SensorAnomalyDetector()
        # noise_floor = 0 -> edge case
        assert d.detect_jamming(0.0, 0.0) is False  # signal=0, not > 0

    def test_zero_noise_with_signal(self):
        d = SensorAnomalyDetector()
        assert d.detect_jamming(5.0, 0.0) is True  # signal > 0 with 0 noise


class TestDetectSpoofing:
    def test_no_spoofing(self):
        d = SensorAnomalyDetector()
        model = {"temp": (0, 100), "pressure": (90, 110)}
        readings = {"temp": 25, "pressure": 100}
        assert d.detect_spoofing(readings, model) == 0.0

    def test_full_spoofing(self):
        d = SensorAnomalyDetector()
        model = {"temp": (0, 50), "pressure": (90, 110)}
        readings = {"temp": 999, "pressure": -999}
        prob = d.detect_spoofing(readings, model)
        assert prob == 1.0

    def test_partial_spoofing(self):
        d = SensorAnomalyDetector()
        model = {"a": (0, 10), "b": (0, 10), "c": (0, 10)}
        readings = {"a": 5, "b": 999, "c": 999}
        prob = d.detect_spoofing(readings, model)
        assert 0.0 < prob < 1.0
        assert abs(prob - 2 / 3) < 1e-9

    def test_empty_readings(self):
        d = SensorAnomalyDetector()
        assert d.detect_spoofing({}, {"a": (0, 10)}) == 0.0

    def test_empty_model(self):
        d = SensorAnomalyDetector()
        assert d.detect_spoofing({"a": 999}, {}) == 0.0

    def test_extra_keys_in_readings(self):
        d = SensorAnomalyDetector()
        model = {"a": (0, 10)}
        readings = {"a": 5, "b": 999}  # b not in model
        assert d.detect_spoofing(readings, model) == 0.0


class TestClearLog:
    def test_clear_log(self):
        d = SensorAnomalyDetector()
        d.check_range("x", 999, (0, 10))
        d.check_rate("x", 999, 0, 1)
        assert len(d.get_anomaly_log()) == 2
        d.clear_log()
        assert len(d.get_anomaly_log()) == 0


# ── CommandInjector ─────────────────────────────────────────────────

class TestCommandInjectorConstruction:
    def test_construct(self):
        ci = CommandInjector()
        assert ci is not None


class TestScanCommand:
    def test_clean_command(self):
        ci = CommandInjector()
        assert ci.scan_command("move forward 10") == []

    def test_sql_injection_detected(self):
        ci = CommandInjector()
        matches = ci.scan_command("SELECT * FROM sensors")
        assert len(matches) > 0
        assert any(m.injection_type == InjectionType.SQL for m in matches)

    def test_shell_injection_detected(self):
        ci = CommandInjector()
        matches = ci.scan_command("run; rm -rf /")
        assert len(matches) > 0
        assert any(m.injection_type == InjectionType.SHELL for m in matches)

    def test_bytecode_injection_detected(self):
        ci = CommandInjector()
        matches = ci.scan_command("__import__('os').system('ls')")
        assert len(matches) > 0
        assert any(m.injection_type == InjectionType.BYTECODE for m in matches)

    def test_lfi_detected(self):
        ci = CommandInjector()
        matches = ci.scan_command("read ../../etc/passwd")
        assert len(matches) > 0
        assert any(m.injection_type == InjectionType.LFI for m in matches)

    def test_command_pipe_detected(self):
        ci = CommandInjector()
        matches = ci.scan_command("data | cat")
        assert len(matches) > 0
        assert any(m.injection_type == InjectionType.COMMAND for m in matches)

    def test_forbidden_patterns(self):
        ci = CommandInjector()
        matches = ci.scan_command("normal command", forbidden_patterns=["dangerous"])
        assert len(matches) == 0
        matches = ci.scan_command("dangerous command", forbidden_patterns=["dangerous"])
        assert len(matches) > 0

    def test_multiple_matches(self):
        ci = CommandInjector()
        matches = ci.scan_command("SELECT * FROM x; DROP TABLE y; __import__('os')")
        types = {m.injection_type for m in matches}
        assert InjectionType.SQL in types
        assert InjectionType.BYTECODE in types

    def test_match_position(self):
        ci = CommandInjector()
        matches = ci.scan_command("normal DROP TABLE")
        assert len(matches) > 0
        for m in matches:
            assert m.position >= 0

    def test_empty_command(self):
        ci = CommandInjector()
        assert ci.scan_command("") == []


class TestClassifyInjection:
    def test_clean_classified_none(self):
        ci = CommandInjector()
        assert ci.classify_injection("navigate to waypoint") == InjectionType.NONE

    def test_sql_classified(self):
        ci = CommandInjector()
        assert ci.classify_injection("SELECT * FROM x") == InjectionType.SQL

    def test_shell_classified(self):
        ci = CommandInjector()
        # Use backtick shell injection (needs closing backtick for regex)
        result = ci.classify_injection("`rm -rf /`")
        assert result == InjectionType.SHELL

    def test_bytecode_classified(self):
        ci = CommandInjector()
        assert ci.classify_injection("exec(code)") == InjectionType.BYTECODE


class TestSanitizeCommand:
    def test_clean_unchanged(self):
        ci = CommandInjector()
        assert ci.sanitize_command("navigate forward 10") == "navigate forward 10"

    def test_sql_stripped(self):
        ci = CommandInjector()
        result = ci.sanitize_command("SELECT * FROM sensors")
        # Should be sanitized — no SELECT left
        assert "SELECT" not in result

    def test_with_allowed_ops(self):
        ci = CommandInjector()
        # Use non-SQL command to test allowed_ops filter without SQL stripping
        result = ci.sanitize_command("read sensor data", allowed_ops=["read", "sensor"])
        assert "read" in result
        assert "sensor" in result
        assert "data" not in result  # 'data' not in allowed_ops

    def test_shell_stripped(self):
        ci = CommandInjector()
        result = ci.sanitize_command("cmd; rm -rf /")
        assert "rm" not in result

    def test_empty_after_filter(self):
        ci = CommandInjector()
        result = ci.sanitize_command("DROP TABLE", allowed_ops=["SELECT"])
        assert result == ""

    def test_lfi_stripped(self):
        ci = CommandInjector()
        result = ci.sanitize_command("read ../../etc/passwd")
        assert "../" not in result


class TestValidateCommandStructure:
    def test_valid_command(self):
        ci = CommandInjector()
        schema = {"required_keys": ["move"], "max_length": 100}
        result = ci.validate_command_structure("move forward", schema)
        assert result["valid"] is True

    def test_missing_required_key(self):
        ci = CommandInjector()
        schema = {"required_keys": ["emergency"]}
        result = ci.validate_command_structure("normal cmd", schema)
        assert result["valid"] is False

    def test_too_long(self):
        ci = CommandInjector()
        schema = {"max_length": 5}
        result = ci.validate_command_structure("a very long command", schema)
        assert result["valid"] is False

    def test_forbidden_keys(self):
        ci = CommandInjector()
        schema = {"forbidden_keys": ["shutdown"]}
        result = ci.validate_command_structure("system shutdown", schema)
        assert result["valid"] is False

    def test_allowed_keys_filter(self):
        ci = CommandInjector()
        schema = {"allowed_keys": ["read", "sensor"]}
        result = ci.validate_command_structure("read sensor", schema)
        assert result["valid"] is True

    def test_disallowed_token(self):
        ci = CommandInjector()
        schema = {"allowed_keys": ["read"]}
        result = ci.validate_command_structure("read delete", schema)
        assert result["valid"] is False

    def test_empty_schema(self):
        ci = CommandInjector()
        result = ci.validate_command_structure("anything", {})
        assert result["valid"] is True


# ── Dataclass Tests ─────────────────────────────────────────────────

class TestAnomalyRecord:
    def test_create(self):
        r = AnomalyRecord(
            timestamp=1.0, sensor_id="t", anomaly_type="range",
            severity=Severity.HIGH, value=100.0, expected_range=(0, 10),
        )
        assert r.sensor_id == "t"
        assert r.value == 100.0

    def test_expected_range_tuple(self):
        r = AnomalyRecord(
            timestamp=0, sensor_id="x", anomaly_type="y",
            severity=Severity.LOW, value=5, expected_range=(0, 10),
        )
        assert r.expected_range == (0, 10)


class TestAttackSignature:
    def test_create(self):
        s = AttackSignature(
            name="sql_injection",
            pattern_description="SELECT pattern",
            detection_threshold=0.9,
            severity=Severity.CRITICAL,
        )
        assert s.name == "sql_injection"
        assert s.detection_threshold == 0.9
