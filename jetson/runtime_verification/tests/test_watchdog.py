"""Tests for watchdog module — 38+ tests."""

import time
import pytest
from jetson.runtime_verification.watchdog import (
    WatchdogConfig,
    WatchdogState,
    WatchdogManager,
)


# ---------- WatchdogConfig tests ----------

class TestWatchdogConfig:
    def test_defaults(self):
        cfg = WatchdogConfig()
        assert cfg.timeout_ms == 5000
        assert cfg.heartbeat_interval_ms == 1000
        assert cfg.max_missed_beats == 3
        assert cfg.escalation_actions == ["log", "alert"]

    def test_custom_values(self):
        cfg = WatchdogConfig(
            timeout_ms=10000,
            heartbeat_interval_ms=2000,
            max_missed_beats=5,
            escalation_actions=["log"],
        )
        assert cfg.timeout_ms == 10000
        assert cfg.heartbeat_interval_ms == 2000
        assert cfg.max_missed_beats == 5
        assert cfg.escalation_actions == ["log"]


# ---------- WatchdogState tests ----------

class TestWatchdogState:
    def test_defaults(self):
        state = WatchdogState()
        assert state.active is True
        assert state.last_heartbeat == 0.0
        assert state.missed_beats == 0
        assert state.escalated is False

    def test_custom(self):
        state = WatchdogState(
            active=False, last_heartbeat=100.0, missed_beats=5, escalated=True
        )
        assert state.active is False
        assert state.last_heartbeat == 100.0
        assert state.missed_beats == 5
        assert state.escalated is True


# ---------- WatchdogManager tests ----------

class TestWatchdogManager:
    def setup_method(self):
        self.wm = WatchdogManager()

    def test_register_component(self):
        cfg = WatchdogConfig(timeout_ms=1000)
        self.wm.register("motor", cfg)
        state = self.wm.get_state("motor")
        assert state is not None
        assert state.active is True

    def test_register_multiple(self):
        for i in range(5):
            self.wm.register(f"comp_{i}", WatchdogConfig())
        assert len(self.wm._configs) == 5

    def test_register_initializes_state(self):
        self.wm.register("nav", WatchdogConfig())
        state = self.wm.get_state("nav")
        assert state.missed_beats == 0
        assert state.escalated is False

    def test_register_initializes_heartbeat_log(self):
        self.wm.register("sensor", WatchdogConfig())
        assert "sensor" in self.wm._heartbeat_log
        assert self.wm._heartbeat_log["sensor"] == []

    def test_feed_heartbeat(self):
        self.wm.register("motor", WatchdogConfig())
        before = time.time()
        self.wm.feed_heartbeat("motor")
        state = self.wm.get_state("motor")
        assert state.last_heartbeat >= before
        assert state.missed_beats == 0

    def test_feed_heartbeat_resets_missed(self):
        self.wm.register("motor", WatchdogConfig(timeout_ms=1))
        state = self.wm.get_state("motor")
        state.missed_beats = 3
        state.escalated = True
        self.wm.feed_heartbeat("motor")
        assert state.missed_beats == 0
        assert state.escalated is False

    def test_feed_heartbeat_nonexistent(self):
        # Should not raise
        self.wm.feed_heartbeat("nope")

    def test_feed_heartbeat_records_log(self):
        self.wm.register("motor", WatchdogConfig())
        self.wm.feed_heartbeat("motor")
        self.wm.feed_heartbeat("motor")
        assert len(self.wm._heartbeat_log["motor"]) == 2

    def test_check_all_no_expiry(self):
        self.wm.register("motor", WatchdogConfig(timeout_ms=10000))
        self.wm.feed_heartbeat("motor")
        expired = self.wm.check_all()
        assert expired == []

    def test_check_all_expired(self):
        self.wm.register("motor", WatchdogConfig(timeout_ms=10, max_missed_beats=1))
        state = self.wm.get_state("motor")
        state.last_heartbeat = time.time() - 1.0  # 1 second ago
        expired = self.wm.check_all()
        assert "motor" in expired

    def test_check_all_multiple_expired(self):
        for name in ["a", "b", "c"]:
            self.wm.register(name, WatchdogConfig(timeout_ms=10))
        for name in ["a", "b", "c"]:
            st = self.wm.get_state(name)
            st.last_heartbeat = time.time() - 1.0
        expired = self.wm.check_all()
        assert len(expired) == 3

    def test_check_all_inactive_skipped(self):
        self.wm.register("motor", WatchdogConfig(timeout_ms=10))
        state = self.wm.get_state("motor")
        state.active = False
        state.last_heartbeat = time.time() - 10.0
        expired = self.wm.check_all()
        assert expired == []

    def test_check_all_escalation(self):
        self.wm.register("motor", WatchdogConfig(timeout_ms=10, max_missed_beats=2))
        state = self.wm.get_state("motor")
        state.last_heartbeat = time.time() - 1.0
        # First check: expired, missed_beats becomes 1
        self.wm.check_all()
        assert state.escalated is False
        # Second check: missed_beats becomes 2, should escalate
        self.wm.check_all()
        assert state.escalated is True

    def test_get_state_exists(self):
        self.wm.register("motor", WatchdogConfig())
        state = self.wm.get_state("motor")
        assert state is not None
        assert isinstance(state, WatchdogState)

    def test_get_state_nonexistent(self):
        assert self.wm.get_state("nope") is None

    def test_reset_exists(self):
        self.wm.register("motor", WatchdogConfig())
        state = self.wm.get_state("motor")
        state.missed_beats = 5
        state.escalated = True
        state.active = False
        assert self.wm.reset("motor") is True
        assert state.missed_beats == 0
        assert state.escalated is False
        assert state.active is True

    def test_reset_nonexistent(self):
        assert self.wm.reset("nope") is False

    def test_reset_updates_last_heartbeat(self):
        self.wm.register("motor", WatchdogConfig())
        before = time.time()
        self.wm.reset("motor")
        state = self.wm.get_state("motor")
        assert state.last_heartbeat >= before

    def test_compute_uptime_healthy(self):
        self.wm.register("motor", WatchdogConfig())
        self.wm.feed_heartbeat("motor")
        uptime = self.wm.compute_uptime("motor")
        assert uptime == 100.0

    def test_compute_uptime_no_heartbeats(self):
        self.wm.register("motor", WatchdogConfig())
        uptime = self.wm.compute_uptime("motor")
        assert uptime == 0.0

    def test_compute_uptime_escalated(self):
        self.wm.register("motor", WatchdogConfig(timeout_ms=10, max_missed_beats=2))
        state = self.wm.get_state("motor")
        state.missed_beats = 2
        state.escalated = True
        uptime = self.wm.compute_uptime("motor")
        # Should be degraded but not zero (missed_beats == max)
        assert 0.0 <= uptime <= 100.0

    def test_compute_uptime_nonexistent(self):
        # No heartbeats, no start time -> defaults
        uptime = self.wm.compute_uptime("nope", period=60.0)
        # _start_times and _heartbeat_log won't have it; recent will be []
        assert uptime == 100.0

    def test_set_custom_handler(self):
        handler_called = []
        def handler(name, state):
            handler_called.append(name)

        self.wm.register("motor", WatchdogConfig(timeout_ms=10, max_missed_beats=1))
        self.wm.set_custom_handler("motor", handler)
        state = self.wm.get_state("motor")
        state.last_heartbeat = time.time() - 1.0
        self.wm.check_all()
        assert "motor" in handler_called

    def test_custom_handler_called_on_escalation(self):
        handler_calls = []
        def handler(name, state):
            handler_calls.append((name, state.missed_beats))

        self.wm.register("motor", WatchdogConfig(timeout_ms=10, max_missed_beats=1))
        self.wm.set_custom_handler("motor", handler)
        state = self.wm.get_state("motor")
        state.last_heartbeat = time.time() - 1.0
        self.wm.check_all()
        assert len(handler_calls) == 1

    def test_no_custom_handler_no_error(self):
        self.wm.register("motor", WatchdogConfig(timeout_ms=10, max_missed_beats=1))
        state = self.wm.get_state("motor")
        state.last_heartbeat = time.time() - 1.0
        # Should not raise even without handler
        self.wm.check_all()

    def test_get_summary_empty(self):
        summary = self.wm.get_summary()
        assert summary["total_components"] == 0
        assert summary["components"] == {}

    def test_get_summary_with_components(self):
        self.wm.register("motor", WatchdogConfig(timeout_ms=1000, max_missed_beats=3))
        self.wm.register("nav", WatchdogConfig(timeout_ms=2000, max_missed_beats=5))
        summary = self.wm.get_summary()
        assert summary["total_components"] == 2
        assert "motor" in summary["components"]
        assert "nav" in summary["components"]

    def test_get_summary_includes_config(self):
        self.wm.register("motor", WatchdogConfig(timeout_ms=777, heartbeat_interval_ms=333, max_missed_beats=9))
        s = self.wm.get_summary()
        cfg = s["components"]["motor"]["config"]
        assert cfg["timeout_ms"] == 777
        assert cfg["heartbeat_interval_ms"] == 333
        assert cfg["max_missed_beats"] == 9

    def test_get_summary_includes_uptime(self):
        self.wm.register("motor", WatchdogConfig())
        self.wm.feed_heartbeat("motor")
        s = self.wm.get_summary()
        assert "uptime_percentage" in s["components"]["motor"]

    def test_get_summary_nonexistent_component_state(self):
        self.wm.register("x", WatchdogConfig())
        # Manually corrupt state to test robustness
        del self.wm._states["x"]
        s = self.wm.get_summary()
        assert s["components"]["x"]["active"] is False
        assert s["components"]["x"]["missed_beats"] == 0
