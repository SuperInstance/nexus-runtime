"""Tests for autopilot / pilot system."""

import pytest

from jetson.navigation.geospatial import Coordinate
from jetson.navigation.pilot import (
    Autopilot, PilotCommand, PilotMode, VesselStateInternal,
)
from jetson.navigation.collision import CollisionThreat, Severity
from jetson.navigation.waypoint import Waypoint, WaypointManager


def make_state(lat, lon, speed, heading, drift_speed=0.0, drift_heading=0.0):
    return VesselStateInternal(
        position=Coordinate(latitude=lat, longitude=lon),
        speed=speed, heading=heading,
        drift_speed=drift_speed, drift_heading=drift_heading,
    )


class TestPilotMode:
    def test_mode_values(self):
        assert PilotMode.MANUAL.value == 0
        assert PilotMode.AUTOPILOT.value == 1
        assert PilotMode.WAYPOINT_FOLLOWING.value == 2
        assert PilotMode.STATION_KEEPING.value == 3
        assert PilotMode.EMERGENCY.value == 4

    def test_all_modes_exist(self):
        modes = list(PilotMode)
        assert len(modes) == 5


class TestPilotCommand:
    def test_default_command(self):
        cmd = PilotCommand(thrust=0.5, rudder=0.0, target_speed=2.0, target_heading=90.0)
        assert cmd.thrust == 0.5
        assert cmd.rudder == 0.0
        assert cmd.target_speed == 2.0
        assert cmd.target_heading == 90.0
        assert cmd.mode == PilotMode.AUTOPILOT

    def test_custom_mode(self):
        cmd = PilotCommand(0.0, 0.0, 0.0, 0.0, mode=PilotMode.EMERGENCY)
        assert cmd.mode == PilotMode.EMERGENCY


class TestVesselStateInternal:
    def test_create_state(self):
        state = make_state(37.0, -122.0, 5.0, 90.0)
        assert state.speed == 5.0
        assert state.heading == 90.0
        assert state.drift_speed == 0.0

    def test_with_drift(self):
        state = make_state(37.0, -122.0, 5.0, 90.0, drift_speed=1.0, drift_heading=180.0)
        assert state.drift_speed == 1.0
        assert state.drift_heading == 180.0


class TestAutopilotInit:
    def test_default_mode(self):
        ap = Autopilot()
        assert ap.get_mode() == PilotMode.AUTOPILOT

    def test_set_mode(self):
        ap = Autopilot()
        ap.set_mode(PilotMode.WAYPOINT_FOLLOWING)
        assert ap.get_mode() == PilotMode.WAYPOINT_FOLLOWING

    def test_set_mode_emergency(self):
        ap = Autopilot()
        ap.set_mode(PilotMode.EMERGENCY)
        assert ap.get_mode() == PilotMode.EMERGENCY

    def test_set_mode_manual(self):
        ap = Autopilot()
        ap.set_mode(PilotMode.MANUAL)
        assert ap.get_mode() == PilotMode.MANUAL


class TestComputeControl:
    def test_heading_correction_right(self):
        ap = Autopilot()
        current = make_state(0, 0, 2.0, 0.0)
        target = make_state(0, 0, 2.0, 45.0)
        cmd = ap.compute_control(current, target)
        assert cmd.rudder > 0  # Starboard to turn right

    def test_heading_correction_left(self):
        ap = Autopilot()
        current = make_state(0, 0, 2.0, 45.0)
        target = make_state(0, 0, 2.0, 0.0)
        cmd = ap.compute_control(current, target)
        assert cmd.rudder < 0  # Port to turn left

    def test_speed_increase(self):
        ap = Autopilot()
        current = make_state(0, 0, 1.0, 0.0)
        target = make_state(0, 0, 5.0, 0.0)
        cmd = ap.compute_control(current, target)
        assert cmd.thrust > 0

    def test_speed_decrease(self):
        ap = Autopilot()
        current = make_state(0, 0, 5.0, 0.0)
        target = make_state(0, 0, 1.0, 0.0)
        cmd = ap.compute_control(current, target)
        assert cmd.thrust < 0

    def test_on_target(self):
        ap = Autopilot()
        current = make_state(0, 0, 2.0, 90.0)
        target = make_state(0, 0, 2.0, 90.0)
        cmd = ap.compute_control(current, target)
        assert abs(cmd.thrust) < 0.01
        assert abs(cmd.rudder) < 0.01

    def test_thrust_bounded(self):
        ap = Autopilot()
        current = make_state(0, 0, 0.0, 0.0)
        target = make_state(0, 0, 100.0, 0.0)
        cmd = ap.compute_control(current, target)
        assert -1.0 <= cmd.thrust <= 1.0

    def test_rudder_bounded(self):
        ap = Autopilot()
        current = make_state(0, 0, 2.0, 0.0)
        target = make_state(0, 0, 2.0, 180.0)
        cmd = ap.compute_control(current, target)
        assert -1.0 <= cmd.rudder <= 1.0

    def test_large_heading_error(self):
        """Test heading normalization for 350->10 (should be +20, not -340)."""
        ap = Autopilot()
        current = make_state(0, 0, 2.0, 350.0)
        target = make_state(0, 0, 2.0, 10.0)
        cmd = ap.compute_control(current, target)
        assert cmd.rudder > 0  # Small turn right


class TestHoldPosition:
    def test_no_drift(self):
        ap = Autopilot()
        pos = Coordinate(latitude=0, longitude=0)
        cmd = ap.hold_position(pos, (0.0, 0.0))
        assert cmd.thrust == 0.0
        assert cmd.rudder == 0.0
        assert cmd.mode == PilotMode.STATION_KEEPING

    def test_with_drift(self):
        ap = Autopilot()
        pos = Coordinate(latitude=0, longitude=0)
        cmd = ap.hold_position(pos, (2.0, 90.0))
        assert cmd.thrust > 0
        assert cmd.target_heading == pytest.approx(270.0, abs=1.0)

    def test_counter_drift_direction(self):
        ap = Autopilot()
        pos = Coordinate(latitude=0, longitude=0)
        cmd = ap.hold_position(pos, (1.0, 0.0))
        assert cmd.target_heading == pytest.approx(180.0, abs=1.0)

    def test_small_drift_ignored(self):
        ap = Autopilot()
        pos = Coordinate(latitude=0, longitude=0)
        cmd = ap.hold_position(pos, (0.005, 90.0))
        assert cmd.thrust == 0.0

    def test_thrust_proportional_to_drift(self):
        ap = Autopilot()
        pos = Coordinate(latitude=0, longitude=0)
        cmd1 = ap.hold_position(pos, (1.0, 90.0))
        cmd2 = ap.hold_position(pos, (3.0, 90.0))
        assert cmd2.thrust > cmd1.thrust


class TestFollowWaypoints:
    def test_empty_manager(self):
        ap = Autopilot()
        state = make_state(0, 0, 2.0, 0.0)
        wm = WaypointManager()
        cmd = ap.follow_waypoints(state, wm)
        assert cmd.thrust == 0.0

    def test_single_waypoint(self):
        ap = Autopilot()
        state = make_state(0, 0, 2.0, 0.0)
        wm = WaypointManager()
        wm.add_waypoint(Waypoint(id="w1", latitude=0.1, longitude=0, speed=3.0))
        cmd = ap.follow_waypoints(state, wm)
        assert isinstance(cmd, PilotCommand)

    def test_all_reached(self):
        ap = Autopilot()
        state = make_state(0, 0, 2.0, 0.0)
        wm = WaypointManager()
        wm.add_waypoint(Waypoint(id="w1", latitude=0.0, longitude=0.0,
                                 acceptance_radius=1000.0))
        cmd = ap.follow_waypoints(state, wm)
        assert cmd.thrust == 0.0

    def test_multiple_waypoints(self):
        ap = Autopilot()
        state = make_state(0, 0, 2.0, 0.0)
        wm = WaypointManager()
        wm.add_waypoint(Waypoint(id="w1", latitude=0.1, longitude=0))
        wm.add_waypoint(Waypoint(id="w2", latitude=0.2, longitude=0))
        cmd = ap.follow_waypoints(state, wm)
        assert isinstance(cmd, PilotCommand)

    def test_heading_towards_waypoint(self):
        ap = Autopilot()
        state = make_state(0, 0, 2.0, 0.0)
        wm = WaypointManager()
        wm.add_waypoint(Waypoint(id="w1", latitude=0.1, longitude=0))
        cmd = ap.follow_waypoints(state, wm)
        # Should aim roughly north
        assert 0 <= cmd.target_heading < 360


class TestEmergencyStop:
    def test_emergency_command(self):
        ap = Autopilot()
        cmd = ap.emergency_stop()
        assert cmd.thrust == 0.0
        assert cmd.rudder == 0.0
        assert cmd.target_speed == 0.0
        assert cmd.mode == PilotMode.EMERGENCY

    def test_emergency_from_any_state(self):
        ap = Autopilot()
        state = make_state(0, 0, 10.0, 90.0)
        cmd = ap.emergency_stop()
        assert cmd.thrust == 0.0


class TestSmoothControl:
    def test_first_command_unchanged(self):
        ap = Autopilot()
        cmd = PilotCommand(thrust=0.5, rudder=0.3, target_speed=3.0, target_heading=90.0)
        smoothed = ap.smooth_control(None, cmd)
        assert smoothed.thrust == pytest.approx(0.5)
        assert smoothed.rudder == pytest.approx(0.3)

    def test_rate_limiting_thrust(self):
        ap = Autopilot()
        prev = PilotCommand(thrust=0.0, rudder=0.0, target_speed=2.0, target_heading=90.0)
        new = PilotCommand(thrust=1.0, rudder=0.0, target_speed=2.0, target_heading=90.0)
        smoothed = ap.smooth_control(prev, new)
        assert abs(smoothed.thrust - 0.0) <= ap._max_thrust_rate + 1e-10
        assert smoothed.thrust < 1.0

    def test_rate_limiting_rudder(self):
        ap = Autopilot()
        prev = PilotCommand(thrust=0.0, rudder=0.0, target_speed=2.0, target_heading=90.0)
        new = PilotCommand(thrust=0.0, rudder=1.0, target_speed=2.0, target_heading=90.0)
        smoothed = ap.smooth_control(prev, new)
        assert abs(smoothed.rudder - 0.0) <= ap._max_rudder_rate + 1e-10
        assert smoothed.rudder < 1.0

    def test_rate_limiting_heading(self):
        ap = Autopilot()
        prev = PilotCommand(thrust=0.0, rudder=0.0, target_speed=2.0, target_heading=0.0)
        new = PilotCommand(thrust=0.0, rudder=0.0, target_speed=2.0, target_heading=90.0)
        smoothed = ap.smooth_control(prev, new)
        assert smoothed.target_heading < 90.0

    def test_rate_limiting_speed(self):
        ap = Autopilot()
        prev = PilotCommand(thrust=0.0, rudder=0.0, target_speed=1.0, target_heading=0.0)
        new = PilotCommand(thrust=0.0, rudder=0.0, target_speed=5.0, target_heading=0.0)
        smoothed = ap.smooth_control(prev, new)
        assert smoothed.target_speed < 5.0

    def test_negative_thrust_rate(self):
        ap = Autopilot()
        prev = PilotCommand(thrust=0.5, rudder=0.0, target_speed=2.0, target_heading=90.0)
        new = PilotCommand(thrust=-0.5, rudder=0.0, target_speed=2.0, target_heading=90.0)
        smoothed = ap.smooth_control(prev, new)
        assert smoothed.thrust > -0.5

    def test_output_bounded(self):
        ap = Autopilot()
        prev = PilotCommand(thrust=0.0, rudder=0.0, target_speed=0.0, target_heading=0.0)
        new = PilotCommand(thrust=0.0, rudder=0.0, target_speed=10.0, target_heading=350.0)
        smoothed = ap.smooth_control(prev, new)
        assert -1.0 <= smoothed.thrust <= 1.0
        assert -1.0 <= smoothed.rudder <= 1.0
        assert smoothed.target_speed >= 0.0


class TestSetMaxRates:
    def test_set_thrust_rate(self):
        ap = Autopilot()
        ap.set_max_rates(thrust_rate=0.5)
        assert ap._max_thrust_rate == 0.5

    def test_set_rudder_rate(self):
        ap = Autopilot()
        ap.set_max_rates(rudder_rate=0.3)
        assert ap._max_rudder_rate == 0.3

    def test_set_heading_rate(self):
        ap = Autopilot()
        ap.set_max_rates(heading_rate=10.0)
        assert ap._max_heading_rate == 10.0

    def test_set_speed_rate(self):
        ap = Autopilot()
        ap.set_max_rates(speed_rate=0.5)
        assert ap._max_speed_rate == 0.5

    def test_set_multiple_rates(self):
        ap = Autopilot()
        ap.set_max_rates(thrust_rate=0.2, rudder_rate=0.1)
        assert ap._max_thrust_rate == 0.2
        assert ap._max_rudder_rate == 0.1

    def test_none_does_not_change(self):
        ap = Autopilot()
        original = ap._max_thrust_rate
        ap.set_max_rates()  # No arguments
        assert ap._max_thrust_rate == original


class TestAvoidCollision:
    def test_no_threats(self):
        ap = Autopilot()
        state = make_state(0, 0, 5.0, 90.0)
        result = ap.avoid_collision(state, [])
        assert result is None

    def test_single_threat(self):
        ap = Autopilot()
        state = make_state(0, 0, 5.0, 0.0)
        threat = CollisionThreat(
            vessel_id="t", position=Coordinate(0, 0.001),
            velocity=(0, -5), distance=100.0,
            tcpa=10.0, dcpa=10.0, severity=Severity.HIGH,
        )
        result = ap.avoid_collision(state, [threat])
        assert result is not None
        assert isinstance(result, PilotCommand)

    def test_multiple_threats_most_severe(self):
        ap = Autopilot()
        state = make_state(0, 0, 5.0, 0.0)
        threats = [
            CollisionThreat(
                vessel_id=f"t{i}", position=Coordinate(0, 0.001 * (i + 1)),
                velocity=(0, -5), distance=100.0,
                tcpa=10.0, dcpa=10.0,
                severity=Severity.LOW if i > 0 else Severity.CRITICAL,
            )
            for i in range(3)
        ]
        result = ap.avoid_collision(state, threats)
        assert result is not None

    def test_evasive_reduces_speed(self):
        ap = Autopilot()
        state = make_state(0, 0, 5.0, 0.0)
        threat = CollisionThreat(
            vessel_id="t", position=Coordinate(0, 0.001),
            velocity=(0, -5), distance=50.0,
            tcpa=5.0, dcpa=5.0, severity=Severity.CRITICAL,
        )
        result = ap.avoid_collision(state, [threat])
        assert result.target_speed < 5.0
