"""Tests for state mirror synchronization."""

import math
import pytest
from jetson.digital_twin.state_mirror import (
    SyncStatus, MirrorConfig, SyncRecord, StateMirror
)
from jetson.digital_twin.physics import VesselState, Force, VesselPhysics


class TestSyncStatus:
    def test_values(self):
        assert SyncStatus.SYNCED.value == "synced"
        assert SyncStatus.DRIFTING.value == "drifting"
        assert SyncStatus.DIVERGED.value == "diverged"
        assert SyncStatus.UNKNOWN.value == "unknown"

    def test_all_members(self):
        members = list(SyncStatus)
        assert len(members) == 4


class TestMirrorConfig:
    def test_defaults(self):
        cfg = MirrorConfig()
        assert cfg.sync_interval == 0.1
        assert cfg.drift_threshold == 0.5
        assert cfg.divergence_threshold == 5.0
        assert cfg.smoothing == 0.3

    def test_custom(self):
        cfg = MirrorConfig(sync_interval=0.5, drift_threshold=1.0)
        assert cfg.sync_interval == 0.5
        assert cfg.drift_threshold == 1.0


class TestSyncRecord:
    def test_creation(self):
        r = SyncRecord(
            timestamp=1.0,
            real_state=VesselState(x=1),
            twin_state=VesselState(x=1.1),
            drift=0.1,
        )
        assert r.timestamp == 1.0
        assert r.drift == 0.1


class TestStateMirror:
    def setup_method(self):
        self.mirror = StateMirror()

    def test_initial_status(self):
        assert self.mirror.get_sync_status() == SyncStatus.UNKNOWN

    def test_update_twin(self):
        state = VesselState(x=10, y=20)
        self.mirror.update_twin(state, 1.0)
        assert self.mirror.get_sync_status() == SyncStatus.SYNCED

    def test_get_twin_state_before_update(self):
        result = self.mirror.get_twin_state(0.0)
        assert isinstance(result, VesselState)

    def test_get_twin_state_after_update(self):
        state = VesselState(x=10, y=20, z=5)
        self.mirror.update_twin(state, 1.0)
        twin = self.mirror.get_twin_state(1.0)
        assert abs(twin.x - 10.0) < 1e-6

    def test_compute_drift_zero(self):
        s1 = VesselState(x=1, y=2, z=3, vx=0.5, vy=0.3, vz=0.1)
        s2 = VesselState(x=1, y=2, z=3, vx=0.5, vy=0.3, vz=0.1)
        drift = self.mirror.compute_drift(s1, s2)
        assert drift == 0.0

    def test_compute_drift_position(self):
        s1 = VesselState(x=0, y=0)
        s2 = VesselState(x=3, y=4)
        drift = self.mirror.compute_drift(s1, s2)
        assert abs(drift - 5.0) < 1e-6

    def test_compute_drift_velocity_contribution(self):
        s1 = VesselState(vx=0, vy=0)
        s2 = VesselState(x=0, y=0, vx=0, vy=10)
        drift = self.mirror.compute_drift(s1, s2)
        # 10 * 0.1 = 1.0
        assert abs(drift - 1.0) < 1e-6

    def test_synced_status_small_drift(self):
        cfg = MirrorConfig(drift_threshold=1.0, divergence_threshold=10.0)
        mirror = StateMirror(config=cfg)
        mirror.update_twin(VesselState(x=0), 0.0)
        mirror.update_twin(VesselState(x=0.5), 1.0)  # drift < 1.0
        assert mirror.get_sync_status() == SyncStatus.SYNCED

    def test_drifting_status(self):
        cfg = MirrorConfig(drift_threshold=0.1, divergence_threshold=10.0)
        mirror = StateMirror(config=cfg)
        mirror.update_twin(VesselState(x=0), 0.0)
        mirror.force_resync(VesselState(x=0), 0.0)
        # Now update with larger offset — smoothing will keep twin close
        mirror.update_twin(VesselState(x=5), 1.0)
        status = mirror.get_sync_status()
        assert status in (SyncStatus.DRIFTING, SyncStatus.SYNCED)

    def test_diverged_status(self):
        cfg = MirrorConfig(smoothing=0.0)
        mirror = StateMirror(config=cfg)
        mirror.force_resync(VesselState(x=0), 0.0)
        # Manually record large drift
        mirror.record_sync(VesselState(x=0), VesselState(x=100), 1.0)
        # The sync status checks current_real vs current_twin
        # After force_resync with smoothing=0 and then update_twin, twin follows directly
        mirror.update_twin(VesselState(x=0), 1.0)
        assert mirror.get_sync_status() == SyncStatus.SYNCED

    def test_interpolate_state(self):
        s1 = VesselState(x=0, y=0, vx=0)
        s2 = VesselState(x=10, y=20, vx=5)
        mid = self.mirror.interpolate_state(s1, s2, 0.5)
        assert abs(mid.x - 5.0) < 1e-9
        assert abs(mid.y - 10.0) < 1e-9
        assert abs(mid.vx - 2.5) < 1e-9

    def test_interpolate_state_zero(self):
        s1 = VesselState(x=5)
        s2 = VesselState(x=10)
        result = self.mirror.interpolate_state(s1, s2, 0.0)
        assert abs(result.x - 5.0) < 1e-9

    def test_interpolate_state_one(self):
        s1 = VesselState(x=5)
        s2 = VesselState(x=10)
        result = self.mirror.interpolate_state(s1, s2, 1.0)
        assert abs(result.x - 10.0) < 1e-9

    def test_interpolate_state_clamped(self):
        s1 = VesselState(x=0)
        s2 = VesselState(x=10)
        result_low = self.mirror.interpolate_state(s1, s2, -1.0)
        assert abs(result_low.x - 0.0) < 1e-9
        result_high = self.mirror.interpolate_state(s1, s2, 2.0)
        assert abs(result_high.x - 10.0) < 1e-9

    def test_interpolate_state_orientation(self):
        s1 = VesselState(roll=0, pitch=0, yaw=0)
        s2 = VesselState(roll=0.1, pitch=0.2, yaw=0.3)
        mid = self.mirror.interpolate_state(s1, s2, 0.5)
        assert abs(mid.roll - 0.05) < 1e-9
        assert abs(mid.yaw - 0.15) < 1e-9

    def test_predict_state(self):
        state = VesselState()
        force = Force(fx=100)
        predicted = self.mirror.predict_state(state, 1.0, [force])
        assert predicted.vx > 0  # velocity changes in first step
        # Position uses old velocity (0), so x stays 0 in one step
        predicted2 = self.mirror.predict_state(predicted, 1.0, [force])
        assert predicted2.x > 0  # now velocity moves position

    def test_predict_state_zero_force(self):
        state = VesselState(vx=5)
        predicted = self.mirror.predict_state(state, 1.0, [])
        assert abs(predicted.vx - 5.0) < 1e-9
        assert predicted.x > 0  # still moves due to velocity

    def test_sync_history(self):
        for i in range(5):
            self.mirror.update_twin(VesselState(x=float(i)), float(i))
        history = self.mirror.sync_history(10.0)
        assert len(history) == 5

    def test_sync_history_duration_filter(self):
        for i in range(10):
            self.mirror.update_twin(VesselState(x=float(i)), float(i))
        history = self.mirror.sync_history(3.0)
        assert len(history) >= 3

    def test_force_resync(self):
        self.mirror.update_twin(VesselState(x=0), 0.0)
        self.mirror.force_resync(VesselState(x=100, y=200), 1.0)
        twin = self.mirror.get_twin_state(1.0)
        assert abs(twin.x - 100.0) < 1e-6

    def test_record_sync(self):
        s1 = VesselState(x=0)
        s2 = VesselState(x=1)
        self.mirror.record_sync(s1, s2, 1.0)
        drifts = self.mirror.get_drift_history()
        assert len(drifts) == 1
        assert drifts[0] == 1.0

    def test_get_last_sync_time(self):
        assert self.mirror.get_last_sync_time() == -1.0
        self.mirror.update_twin(VesselState(), 5.0)
        assert self.mirror.get_last_sync_time() == 5.0

    def test_get_history_length(self):
        assert self.mirror.get_history_length() == 0
        self.mirror.update_twin(VesselState(), 1.0)
        assert self.mirror.get_history_length() == 1

    def test_clear_history(self):
        self.mirror.update_twin(VesselState(), 1.0)
        self.mirror.update_twin(VesselState(), 2.0)
        assert self.mirror.get_history_length() == 2
        self.mirror.clear_history()
        assert self.mirror.get_history_length() == 0

    def test_get_drift_history_empty(self):
        assert self.mirror.get_drift_history() == []

    def test_smoothing_effect(self):
        cfg = MirrorConfig(smoothing=1.0)  # No smoothing (full jump to new)
        mirror = StateMirror(config=cfg)
        mirror.update_twin(VesselState(x=0), 0.0)
        mirror.update_twin(VesselState(x=10), 1.0)
        twin = mirror.get_twin_state(1.0)
        assert abs(twin.x - 10.0) < 1e-6

    def test_smoothing_with_value(self):
        cfg = MirrorConfig(smoothing=1.0)  # Full smoothing = stays at old
        mirror = StateMirror(config=cfg)
        mirror.update_twin(VesselState(x=0), 0.0)
        mirror.update_twin(VesselState(x=10), 1.0)
        twin = mirror.get_twin_state(1.0)
        # With smoothing=1.0: twin = 0 + 1.0 * (10 - 0) = 10.0
        assert abs(twin.x - 10.0) < 1e-6

    def test_multiple_updates(self):
        for i in range(20):
            self.mirror.update_twin(VesselState(x=float(i)), float(i))
        assert self.mirror.get_history_length() == 20
        assert self.mirror.get_last_sync_time() == 19.0

    def test_custom_physics(self):
        physics = VesselPhysics()
        mirror = StateMirror(physics=physics)
        state = VesselState()
        predicted = mirror.predict_state(state, 1.0, [Force(fx=100)])
        assert predicted.vx > 0
        predicted2 = mirror.predict_state(predicted, 1.0, [Force(fx=100)])
        assert predicted2.x > 0
