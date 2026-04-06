"""Tests for fault_detector module — 38 tests."""

import time

import pytest

from jetson.self_healing.fault_detector import (
    DegradationReport,
    FaultCategory,
    FaultDetector,
    FaultEvent,
    FaultSeverity,
    HealthIndicator,
    IndicatorStatus,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def detector():
    return FaultDetector()


@pytest.fixture
def healthy_indicator():
    return HealthIndicator(
        component="nav_engine",
        metric_name="cpu_usage",
        value=45.0,
        normal_range=(20.0, 80.0),
    )


@pytest.fixture
def warning_indicator():
    return HealthIndicator(
        component="sensor_hub",
        metric_name="latency_ms",
        value=108.0,
        normal_range=(50.0, 100.0),
    )


@pytest.fixture
def critical_indicator():
    return HealthIndicator(
        component="gps_module",
        metric_name="signal_strength",
        value=-50.0,
        normal_range=(10.0, 90.0),
    )


# ── HealthIndicator ───────────────────────────────────────────────────────

class TestHealthIndicator:
    def test_evaluate_healthy_within_range(self):
        ind = HealthIndicator("c1", "m1", 50.0, (10.0, 90.0))
        assert ind.evaluate() == IndicatorStatus.HEALTHY

    def test_evaluate_healthy_at_boundary(self):
        ind = HealthIndicator("c1", "m1", 90.0, (10.0, 90.0))
        # 90 + 10% margin = 99 → still healthy
        assert ind.evaluate() == IndicatorStatus.HEALTHY

    def test_evaluate_warning_near_boundary(self):
        ind = HealthIndicator("c1", "m1", 105.0, (10.0, 90.0))
        # margin = 8, warning zone up to 90+16=106
        assert ind.evaluate() == IndicatorStatus.WARNING

    def test_evaluate_critical_far_outside(self):
        ind = HealthIndicator("c1", "m1", 200.0, (10.0, 90.0))
        assert ind.evaluate() == IndicatorStatus.CRITICAL

    def test_evaluate_critical_below_range(self):
        ind = HealthIndicator("c1", "m1", -100.0, (10.0, 90.0))
        assert ind.evaluate() == IndicatorStatus.CRITICAL

    def test_dataclass_fields(self, healthy_indicator):
        assert healthy_indicator.component == "nav_engine"
        assert healthy_indicator.metric_name == "cpu_usage"
        assert healthy_indicator.value == 45.0
        assert healthy_indicator.normal_range == (20.0, 80.0)

    def test_default_status_unknown(self):
        ind = HealthIndicator("c", "m", 0)
        assert ind.status == IndicatorStatus.UNKNOWN

    def test_evaluate_zero_range(self):
        ind = HealthIndicator("c", "m", 5.0, (0.0, 0.0))
        # margin = 0 → only exact match is healthy
        assert ind.evaluate() == IndicatorStatus.CRITICAL


# ── FaultEvent ────────────────────────────────────────────────────────────

class TestFaultEvent:
    def test_default_id_generated(self):
        event = FaultEvent()
        assert len(event.id) == 12

    def test_custom_fields(self):
        event = FaultEvent(
            component="motor_ctrl",
            fault_type="overheating",
            severity=FaultSeverity.HIGH,
            symptoms=["temp > 90C"],
        )
        assert event.component == "motor_ctrl"
        assert event.fault_type == "overheating"
        assert event.severity == FaultSeverity.HIGH

    def test_context_dict(self):
        event = FaultEvent(context={"key": "value"})
        assert event.context["key"] == "value"

    def test_timestamp_auto_set(self):
        before = time.time()
        event = FaultEvent()
        after = time.time()
        assert before <= event.timestamp <= after


# ── FaultSeverity ─────────────────────────────────────────────────────────

class TestFaultSeverity:
    def test_ordering(self):
        assert FaultSeverity.NONE < FaultSeverity.LOW
        assert FaultSeverity.LOW < FaultSeverity.MEDIUM
        assert FaultSeverity.MEDIUM < FaultSeverity.HIGH
        assert FaultSeverity.HIGH < FaultSeverity.CRITICAL

    def test_ge(self):
        assert FaultSeverity.HIGH >= FaultSeverity.MEDIUM

    def test_le(self):
        assert FaultSeverity.LOW <= FaultSeverity.MEDIUM


# ── FaultDetector ─────────────────────────────────────────────────────────

class TestFaultDetector:
    def test_register_indicator(self, detector, healthy_indicator):
        detector.register_health_indicator(healthy_indicator)
        indicators = detector.check_indicators()
        assert len(indicators) == 1

    def test_register_multiple_indicators(self, detector):
        detector.register_health_indicator(HealthIndicator("c1", "m1", 50.0, (10, 90)))
        detector.register_health_indicator(HealthIndicator("c2", "m2", 30.0, (5, 50)))
        assert len(detector.check_indicators()) == 2

    def test_register_updates_existing(self, detector):
        ind1 = HealthIndicator("c1", "m1", 50.0, (10, 90))
        ind2 = HealthIndicator("c1", "m1", 60.0, (10, 90))
        detector.register_health_indicator(ind1)
        detector.register_health_indicator(ind2)
        assert len(detector.check_indicators()) == 1

    def test_check_indicators_re_evaluates(self, detector):
        ind = HealthIndicator("c1", "m1", 200.0, (10, 90))
        detector.register_health_indicator(ind)
        results = detector.check_indicators()
        assert results[0].status == IndicatorStatus.CRITICAL

    def test_detect_fault_no_fault(self, detector, healthy_indicator):
        fault = detector.detect_fault([healthy_indicator])
        assert fault is None

    def test_detect_fault_warning(self, detector, warning_indicator):
        fault = detector.detect_fault([warning_indicator])
        assert fault is not None
        assert fault.severity == FaultSeverity.MEDIUM

    def test_detect_fault_critical(self, detector, critical_indicator):
        fault = detector.detect_fault([critical_indicator])
        assert fault is not None
        assert fault.severity == FaultSeverity.HIGH

    def test_detect_fault_picks_worst(self, detector):
        warning = HealthIndicator("c1", "m1", 150.0, (10, 90))
        healthy = HealthIndicator("c2", "m2", 50.0, (10, 90))
        fault = detector.detect_fault([healthy, warning])
        assert fault is not None
        assert fault.severity == FaultSeverity.HIGH

    def test_detect_fault_records_history(self, detector, critical_indicator):
        detector.register_health_indicator(critical_indicator)
        detector.detect_fault([critical_indicator])
        history = detector.get_fault_history()
        assert len(history) == 1

    def test_classify_thermal(self, detector):
        fault = FaultEvent(fault_type="cpu_temp_overheating", component="thermal_sensor")
        assert detector.classify_fault(fault) == FaultCategory.THERMAL

    def test_classify_memory(self, detector):
        fault = FaultEvent(fault_type="memory_leak", component="app")
        assert detector.classify_fault(fault) == FaultCategory.MEMORY

    def test_classify_network(self, detector):
        fault = FaultEvent(fault_type="network_timeout", component="comms")
        assert detector.classify_fault(fault) == FaultCategory.NETWORK

    def test_classify_sensor(self, detector):
        fault = FaultEvent(fault_type="gps_signal_lost", component="navigation")
        assert detector.classify_fault(fault) == FaultCategory.SENSOR

    def test_classify_power(self, detector):
        fault = FaultEvent(fault_type="battery_drain", component="power_sys")
        assert detector.classify_fault(fault) == FaultCategory.POWER

    def test_classify_hardware(self, detector):
        fault = FaultEvent(fault_type="disk_io_error", component="storage")
        assert detector.classify_fault(fault) == FaultCategory.HARDWARE

    def test_classify_software(self, detector):
        fault = FaultEvent(fault_type="firmware_crash", component="ctrl")
        assert detector.classify_fault(fault) == FaultCategory.SOFTWARE

    def test_classify_unknown(self, detector):
        fault = FaultEvent(fault_type="something_weird", component="mystery")
        assert detector.classify_fault(fault) == FaultCategory.UNKNOWN

    def test_compute_severity_multi_indicator(self, detector):
        fault = FaultEvent(context={"indicator_count": 5})
        assert detector.compute_severity(fault) == FaultSeverity.CRITICAL

    def test_compute_severity_high_deviation(self, detector):
        fault = FaultEvent(context={"value": 200.0, "normal_range": (0, 100)})
        assert detector.compute_severity(fault) >= FaultSeverity.HIGH

    def test_compute_severity_low_deviation(self, detector):
        fault = FaultEvent(context={"value": 103.0, "normal_range": (0, 100)})
        assert detector.compute_severity(fault) <= FaultSeverity.MEDIUM

    def test_detect_degradation_insufficient_data(self, detector):
        result = detector.detect_degradation([])
        assert result is None

    def test_detect_degradation_two_points(self, detector):
        history = [
            HealthIndicator("c1", "m1", 100.0, (0, 200), timestamp=0),
            HealthIndicator("c1", "m1", 101.0, (0, 200), timestamp=1),
        ]
        assert detector.detect_degradation(history) is None

    def test_detect_degradation_degrading(self, detector):
        history = [
            HealthIndicator("c1", "m1", 100.0, (0, 200), timestamp=0),
            HealthIndicator("c1", "m1", 80.0, (0, 200), timestamp=1),
            HealthIndicator("c1", "m1", 60.0, (0, 200), timestamp=2),
        ]
        result = detector.detect_degradation(history)
        assert result is not None
        assert result.trend == "degrading"

    def test_detect_degradation_improving(self, detector):
        history = [
            HealthIndicator("c1", "m1", 50.0, (0, 200), timestamp=0),
            HealthIndicator("c1", "m1", 70.0, (0, 200), timestamp=1),
            HealthIndicator("c1", "m1", 90.0, (0, 200), timestamp=2),
        ]
        result = detector.detect_degradation(history)
        assert result is not None
        assert result.trend == "improving"

    def test_detect_degradation_stable(self, detector):
        history = [
            HealthIndicator("c1", "m1", 100.0, (0, 200), timestamp=0),
            HealthIndicator("c1", "m1", 100.0, (0, 200), timestamp=1),
            HealthIndicator("c1", "m1", 100.0, (0, 200), timestamp=2),
        ]
        result = detector.detect_degradation(history)
        assert result is not None
        assert result.trend == "stable"

    def test_get_fault_history_all(self, detector):
        for i in range(5):
            ind = HealthIndicator(f"c{i}", f"m{i}", -999, (0, 100))
            detector.register_health_indicator(ind)
            detector.detect_fault([ind])
        history = detector.get_fault_history(limit=3)
        assert len(history) == 3

    def test_get_fault_history_filtered(self, detector):
        ind1 = HealthIndicator("comp_a", "m1", -999, (0, 100))
        ind2 = HealthIndicator("comp_b", "m2", -999, (0, 100))
        detector.register_health_indicator(ind1)
        detector.register_health_indicator(ind2)
        detector.detect_fault([ind1])
        detector.detect_fault([ind2])
        history = detector.get_fault_history(component="comp_a")
        assert len(history) == 1

    def test_clear_fault_history(self, detector):
        ind = HealthIndicator("c1", "m1", -999, (0, 100))
        detector.detect_fault([ind])
        detector.clear_fault_history()
        assert len(detector.get_fault_history()) == 0

    def test_get_indicator_history(self, detector):
        ind1 = HealthIndicator("c1", "m1", 50.0, (0, 100), timestamp=1)
        ind2 = HealthIndicator("c1", "m1", 60.0, (0, 100), timestamp=2)
        detector.register_health_indicator(ind1)
        detector.register_health_indicator(ind2)
        history = detector.get_indicator_history("c1", "m1")
        assert len(history) == 2

    def test_anomaly_threshold_setter(self, detector):
        detector.anomaly_threshold = 5.0
        assert detector.anomaly_threshold == 5.0
        detector.anomaly_threshold = -1.0
        assert detector.anomaly_threshold == 0.1

    def test_indicator_history_capped(self, detector):
        for i in range(1100):
            ind = HealthIndicator("c1", "m1", float(i), (0, 2000), timestamp=float(i))
            detector.register_health_indicator(ind)
        history = detector.get_indicator_history("c1", "m1")
        assert len(history) <= 1000
