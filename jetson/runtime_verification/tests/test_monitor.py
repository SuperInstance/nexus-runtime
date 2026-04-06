"""Tests for monitor module — 38+ tests."""

import time
import pytest
from jetson.runtime_verification.monitor import (
    MonitorEvent,
    MonitorRule,
    RuntimeMonitor,
    Alert,
)


# ---------- MonitorEvent dataclass tests ----------

class TestMonitorEvent:
    def test_create_minimal(self):
        e = MonitorEvent(timestamp=1.0, source="sensor", event_type="reading")
        assert e.timestamp == 1.0
        assert e.source == "sensor"
        assert e.event_type == "reading"
        assert e.value is None
        assert e.metadata is None

    def test_create_full(self):
        e = MonitorEvent(
            timestamp=2.0,
            source="motor",
            event_type="overheat",
            value=105.5,
            metadata={"sensor_id": "M1"},
        )
        assert e.value == 105.5
        assert e.metadata["sensor_id"] == "M1"

    def test_current_timestamp(self):
        e = MonitorEvent(timestamp=time.time(), source="x", event_type="y")
        assert isinstance(e.timestamp, float)


# ---------- MonitorRule dataclass tests ----------

class TestMonitorRule:
    def test_create_minimal(self):
        r = MonitorRule(name="high_temp", condition_fn=lambda e: True)
        assert r.name == "high_temp"
        assert r.threshold == 0.0
        assert r.severity == "warning"
        assert r.cooldown == 0.0

    def test_create_full(self):
        r = MonitorRule(
            name="critical_temp",
            condition_fn=lambda e: e.value > 100,
            threshold=100.0,
            severity="critical",
            cooldown=5.0,
        )
        assert r.severity == "critical"
        assert r.cooldown == 5.0

    def test_condition_fn_true(self):
        r = MonitorRule(name="t", condition_fn=lambda e: e.value > 5)
        e = MonitorEvent(timestamp=0, source="s", event_type="t", value=10)
        assert r.condition_fn(e) is True

    def test_condition_fn_false(self):
        r = MonitorRule(name="t", condition_fn=lambda e: e.value > 5)
        e = MonitorEvent(timestamp=0, source="s", event_type="t", value=3)
        assert r.condition_fn(e) is False


# ---------- RuntimeMonitor tests ----------

class TestRuntimeMonitor:
    def setup_method(self):
        self.mon = RuntimeMonitor()

    def test_add_rule(self):
        r = MonitorRule(name="r1", condition_fn=lambda e: True)
        self.mon.add_rule(r)
        assert "r1" in self.mon._rules

    def test_add_multiple_rules(self):
        for i in range(5):
            self.mon.add_rule(MonitorRule(name=f"r{i}", condition_fn=lambda e: True))
        assert len(self.mon._rules) == 5

    def test_add_rule_initializes_trigger_count(self):
        r = MonitorRule(name="new_rule", condition_fn=lambda e: True)
        self.mon.add_rule(r)
        assert self.mon._trigger_counts["new_rule"] == 0

    def test_process_event_no_rules(self):
        e = MonitorEvent(timestamp=0, source="s", event_type="t", value=42)
        result = self.mon.process_event(e)
        assert result == []

    def test_process_event_no_trigger(self):
        self.mon.add_rule(
            MonitorRule(name="high", condition_fn=lambda e: e.value > 100)
        )
        e = MonitorEvent(timestamp=0, source="s", event_type="t", value=50)
        result = self.mon.process_event(e)
        assert result == []

    def test_process_event_triggers(self):
        self.mon.add_rule(
            MonitorRule(name="high", condition_fn=lambda e: e.value > 100)
        )
        e = MonitorEvent(timestamp=0, source="s", event_type="t", value=150)
        result = self.mon.process_event(e)
        assert len(result) == 1
        assert result[0].rule_name == "high"

    def test_process_event_multiple_rules(self):
        self.mon.add_rule(
            MonitorRule(name="r1", condition_fn=lambda e: e.value > 10)
        )
        self.mon.add_rule(
            MonitorRule(name="r2", condition_fn=lambda e: e.value > 50)
        )
        e = MonitorEvent(timestamp=0, source="s", event_type="t", value=100)
        result = self.mon.process_event(e)
        assert len(result) == 2

    def test_process_event_stores_history(self):
        e = MonitorEvent(timestamp=0, source="s", event_type="t")
        self.mon.process_event(e)
        assert len(self.mon._event_history) == 1

    def test_process_event_increments_trigger_count(self):
        self.mon.add_rule(
            MonitorRule(name="r1", condition_fn=lambda e: True)
        )
        self.mon.process_event(MonitorEvent(0, "s", "t"))
        assert self.mon._trigger_counts["r1"] == 1

    def test_evaluate_condition_true(self):
        rule = MonitorRule(name="r", condition_fn=lambda e: e.value > 5)
        e = MonitorEvent(0, "s", "t", value=10)
        assert self.mon.evaluate_condition(e, rule) is True

    def test_evaluate_condition_false(self):
        rule = MonitorRule(name="r", condition_fn=lambda e: e.value > 5)
        e = MonitorEvent(0, "s", "t", value=3)
        assert self.mon.evaluate_condition(e, rule) is False

    def test_evaluate_condition_exception(self):
        rule = MonitorRule(name="r", condition_fn=lambda e: e.nonexistent)  # type: ignore
        e = MonitorEvent(0, "s", "t")
        assert self.mon.evaluate_condition(e, rule) is False

    def test_cooldown_prevents_rapid_triggers(self):
        self.mon.add_rule(
            MonitorRule(name="cd", condition_fn=lambda e: True, cooldown=10.0)
        )
        e = MonitorEvent(0, "s", "t")
        result1 = self.mon.process_event(e)
        result2 = self.mon.process_event(e)
        assert len(result1) == 1
        assert len(result2) == 0

    def test_cooldown_allows_after_period(self):
        self.mon.add_rule(
            MonitorRule(name="cd", condition_fn=lambda e: True, cooldown=0.0)
        )
        e = MonitorEvent(0, "s", "t")
        result1 = self.mon.process_event(e)
        result2 = self.mon.process_event(e)
        assert len(result1) == 1
        assert len(result2) == 1

    def test_compute_trigger_frequency(self):
        self.mon.add_rule(MonitorRule(name="f", condition_fn=lambda e: True))
        self.mon.process_event(MonitorEvent(0, "s", "t"))
        self.mon.process_event(MonitorEvent(0, "s", "t"))
        freq = self.mon.compute_trigger_frequency("f", 60.0)
        assert freq == pytest.approx(2.0 / 60.0)

    def test_compute_trigger_frequency_zero(self):
        freq = self.mon.compute_trigger_frequency("nonexistent", 60.0)
        assert freq == 0.0

    def test_compute_trigger_frequency_zero_window(self):
        self.mon.add_rule(MonitorRule(name="f", condition_fn=lambda e: True))
        freq = self.mon.compute_trigger_frequency("f", 0.0)
        assert freq == 0.0

    def test_compute_trigger_frequency_negative_window(self):
        freq = self.mon.compute_trigger_frequency("f", -10.0)
        assert freq == 0.0

    def test_get_active_alerts_empty(self):
        alerts = self.mon.get_active_alerts()
        assert alerts == []

    def test_get_active_alerts_with_alerts(self):
        self.mon.add_rule(MonitorRule(name="a", condition_fn=lambda e: True))
        self.mon.process_event(MonitorEvent(0, "s", "t"))
        alerts = self.mon.get_active_alerts()
        assert len(alerts) == 1

    def test_get_active_alerts_excludes_acknowledged(self):
        self.mon.add_rule(MonitorRule(name="a", condition_fn=lambda e: True))
        alerts = self.mon.process_event(MonitorEvent(0, "s", "t"))
        self.mon.acknowledge_alert(alerts[0].alert_id)
        active = self.mon.get_active_alerts()
        assert active == []

    def test_acknowledge_alert_true(self):
        self.mon.add_rule(MonitorRule(name="a", condition_fn=lambda e: True))
        alerts = self.mon.process_event(MonitorEvent(0, "s", "t"))
        assert self.mon.acknowledge_alert(alerts[0].alert_id) is True

    def test_acknowledge_alert_false(self):
        assert self.mon.acknowledge_alert("nonexistent-id") is False

    def test_compute_monitor_health_perfect(self):
        health = self.mon.compute_monitor_health()
        assert health == 100.0

    def test_compute_monitor_health_with_info_alert(self):
        self.mon.add_rule(
            MonitorRule(name="info_r", condition_fn=lambda e: True, severity="info")
        )
        self.mon.process_event(MonitorEvent(0, "s", "t"))
        health = self.mon.compute_monitor_health()
        assert health == pytest.approx(99.5)

    def test_compute_monitor_health_with_critical_alert(self):
        self.mon.add_rule(
            MonitorRule(name="crit_r", condition_fn=lambda e: True, severity="critical")
        )
        self.mon.process_event(MonitorEvent(0, "s", "t"))
        health = self.mon.compute_monitor_health()
        assert health == pytest.approx(90.0)

    def test_compute_monitor_health_acked_no_penalty(self):
        self.mon.add_rule(
            MonitorRule(name="crit_r", condition_fn=lambda e: True, severity="critical")
        )
        alerts = self.mon.process_event(MonitorEvent(0, "s", "t"))
        self.mon.acknowledge_alert(alerts[0].alert_id)
        health = self.mon.compute_monitor_health()
        assert health == 100.0

    def test_compute_monitor_health_floor_at_zero(self):
        self.mon.add_rule(
            MonitorRule(name="cr", condition_fn=lambda e: True, severity="critical")
        )
        for _ in range(15):
            self.mon.process_event(MonitorEvent(0, "s", "t"))
        health = self.mon.compute_monitor_health()
        assert health >= 0.0

    def test_alert_has_uuid(self):
        self.mon.add_rule(MonitorRule(name="r", condition_fn=lambda e: True))
        alerts = self.mon.process_event(MonitorEvent(0, "s", "t"))
        assert isinstance(alerts[0].alert_id, str)
        assert len(alerts[0].alert_id) > 0

    def test_alert_has_timestamp(self):
        before = time.time()
        self.mon.add_rule(MonitorRule(name="r", condition_fn=lambda e: True))
        alerts = self.mon.process_event(MonitorEvent(0, "s", "t"))
        after = time.time()
        assert before <= alerts[0].timestamp <= after

    def test_alert_references_event(self):
        self.mon.add_rule(MonitorRule(name="r", condition_fn=lambda e: True))
        e = MonitorEvent(0, "sensor_A", "overheat", value=105)
        alerts = self.mon.process_event(e)
        assert alerts[0].event.source == "sensor_A"
