"""Tests for SystemHealthMonitor — Phase 5 Round 10."""

import time
import pytest
from jetson.integration.health import (
    HealthStatus,
    SubsystemHealth,
    SystemHealthMonitor,
)


@pytest.fixture
def monitor():
    return SystemHealthMonitor()


# === HealthStatus ===

class TestHealthStatusEnum:
    def test_values(self):
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"

    def test_count(self):
        assert len(HealthStatus) == 4

    def test_members_are_strings(self):
        for h in HealthStatus:
            assert isinstance(h.value, str)


# === SubsystemHealth ===

class TestSubsystemHealth:
    def test_default(self):
        h = SubsystemHealth(name="x")
        assert h.name == "x"
        assert h.status == HealthStatus.UNKNOWN
        assert h.last_check == 0.0
        assert h.details == ""
        assert h.response_time_ms == 0.0
        assert h.history == []
        assert h.check_timestamps == []

    def test_full(self):
        h = SubsystemHealth(name="y", status=HealthStatus.HEALTHY,
                            details="all good", response_time_ms=5.5)
        assert h.status == HealthStatus.HEALTHY
        assert h.details == "all good"
        assert h.response_time_ms == 5.5


# === Registration ===

class TestRegistration:
    def test_register(self, monitor):
        monitor.register_subsystem("a")
        assert monitor.get_health("a") is not None

    def test_register_duplicate(self, monitor):
        monitor.register_subsystem("a")
        monitor.register_subsystem("a")  # idempotent
        assert len(monitor.get_registered_subsystems()) == 1

    def test_unregister(self, monitor):
        monitor.register_subsystem("a")
        assert monitor.unregister_subsystem("a") is True
        assert monitor.get_health("a") is None

    def test_unregister_missing(self, monitor):
        assert monitor.unregister_subsystem("ghost") is False

    def test_get_registered(self, monitor):
        for name in ["a", "b", "c"]:
            monitor.register_subsystem(name)
        assert set(monitor.get_registered_subsystems()) == {"a", "b", "c"}

    def test_get_registered_empty(self, monitor):
        assert monitor.get_registered_subsystems() == []


# === Update ===

class TestUpdateHealth:
    def test_update_creates(self, monitor):
        monitor.update_health("new_sub", HealthStatus.HEALTHY)
        assert monitor.get_health("new_sub") is not None

    def test_update_status(self, monitor):
        monitor.register_subsystem("a")
        monitor.update_health("a", HealthStatus.HEALTHY)
        assert monitor.get_health("a").status == HealthStatus.HEALTHY

    def test_update_details(self, monitor):
        monitor.update_health("a", HealthStatus.DEGRADED, details="slow")
        assert monitor.get_health("a").details == "slow"

    def test_update_response_time(self, monitor):
        monitor.update_health("a", HealthStatus.HEALTHY, response_time_ms=42.0)
        assert monitor.get_health("a").response_time_ms == 42.0

    def test_update_history(self, monitor):
        monitor.update_health("a", HealthStatus.HEALTHY)
        monitor.update_health("a", HealthStatus.DEGRADED)
        h = monitor.get_health("a")
        assert len(h.history) == 2

    def test_update_timestamp(self, monitor):
        before = time.time()
        monitor.update_health("a", HealthStatus.HEALTHY)
        after = time.time()
        h = monitor.get_health("a")
        assert before <= h.last_check <= after


# === Overall Health ===

class TestOverallHealth:
    def test_empty(self, monitor):
        assert monitor.get_overall_health() == HealthStatus.UNKNOWN

    def test_all_healthy(self, monitor):
        for n in ["a", "b"]:
            monitor.update_health(n, HealthStatus.HEALTHY)
        assert monitor.get_overall_health() == HealthStatus.HEALTHY

    def test_one_degraded(self, monitor):
        monitor.update_health("a", HealthStatus.HEALTHY)
        monitor.update_health("b", HealthStatus.DEGRADED)
        assert monitor.get_overall_health() == HealthStatus.DEGRADED

    def test_one_unhealthy(self, monitor):
        monitor.update_health("a", HealthStatus.HEALTHY)
        monitor.update_health("b", HealthStatus.UNHEALTHY)
        assert monitor.get_overall_health() == HealthStatus.UNHEALTHY

    def test_all_unknown(self, monitor):
        monitor.update_health("a", HealthStatus.UNKNOWN)
        assert monitor.get_overall_health() == HealthStatus.UNKNOWN


# === Uptime ===

class TestUptime:
    def test_no_data(self, monitor):
        assert monitor.compute_uptime("ghost") == 0.0

    def test_all_healthy(self, monitor):
        for _ in range(5):
            monitor.update_health("a", HealthStatus.HEALTHY)
        uptime = monitor.compute_uptime("a", period=3600)
        assert uptime == 1.0

    def test_partial_healthy(self, monitor):
        monitor.update_health("a", HealthStatus.HEALTHY)
        monitor.update_health("a", HealthStatus.HEALTHY)
        monitor.update_health("a", HealthStatus.UNHEALTHY)
        uptime = monitor.compute_uptime("a", period=3600)
        assert uptime == pytest.approx(2.0 / 3.0)

    def test_zero_period(self, monitor):
        monitor.update_health("a", HealthStatus.HEALTHY)
        uptime = monitor.compute_uptime("a", period=0.0)
        assert uptime == 0.0


# === Degradation Detection ===

class TestDegradation:
    def test_insufficient_data(self, monitor):
        monitor.update_health("a", HealthStatus.HEALTHY)
        degraded, reason = monitor.detect_degradation("a")
        assert degraded is False
        assert "insufficient" in reason

    def test_no_degradation(self, monitor):
        for _ in range(5):
            monitor.update_health("a", HealthStatus.HEALTHY)
        degraded, reason = monitor.detect_degradation("a")
        assert degraded is False
        assert reason == "ok"

    def test_degraded_detected(self, monitor):
        for _ in range(3):
            monitor.update_health("a", HealthStatus.DEGRADED)
        degraded, reason = monitor.detect_degradation("a")
        assert degraded is True
        assert "degraded" in reason

    def test_unhealthy_detected(self, monitor):
        for _ in range(4):
            monitor.update_health("a", HealthStatus.UNHEALTHY)
        degraded, reason = monitor.detect_degradation("a")
        assert degraded is True

    def test_missing_subsystem(self, monitor):
        degraded, reason = monitor.detect_degradation("ghost")
        assert degraded is False
        assert "not found" in reason

    def test_custom_history(self, monitor):
        history = [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.DEGRADED, HealthStatus.DEGRADED]
        degraded, _ = monitor.detect_degradation("x", history=history)
        assert degraded is True


# === Health Report ===

class TestHealthReport:
    def test_empty_report(self, monitor):
        report = monitor.get_health_report()
        assert report["total"] == 0
        assert report["overall"] == "unknown"

    def test_report_counts(self, monitor):
        monitor.update_health("a", HealthStatus.HEALTHY)
        monitor.update_health("b", HealthStatus.DEGRADED)
        monitor.update_health("c", HealthStatus.UNHEALTHY)
        report = monitor.get_health_report()
        assert report["total"] == 3
        assert report["healthy"] == 1
        assert report["degraded"] == 1
        assert report["unhealthy"] == 1

    def test_report_subsystem_details(self, monitor):
        monitor.update_health("a", HealthStatus.HEALTHY, response_time_ms=10.0)
        report = monitor.get_health_report()
        sub = report["subsystems"]["a"]
        assert sub["status"] == "healthy"
        assert sub["response_time_ms"] == 10.0

    def test_report_check_count(self, monitor):
        monitor.update_health("a", HealthStatus.HEALTHY)
        monitor.update_health("a", HealthStatus.DEGRADED)
        report = monitor.get_health_report()
        assert report["subsystems"]["a"]["check_count"] == 2


# === History Limit ===

class TestHistoryLimit:
    def test_set_limit(self, monitor):
        monitor.set_history_limit(5)
        for _ in range(10):
            monitor.update_health("a", HealthStatus.HEALTHY)
        h = monitor.get_health("a")
        assert len(h.history) == 5

    def test_default_limit_large(self, monitor):
        for _ in range(500):
            monitor.update_health("a", HealthStatus.HEALTHY)
        h = monitor.get_health("a")
        assert len(h.history) == 500
