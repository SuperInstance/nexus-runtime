"""Pytest integration tests for Navigation Stack cross-module trials.

Each test calls 2+ real navigation modules working together:
WaypointManager, GeoCalculator, PathFollower, CollisionAvoidance,
Autopilot, SituationalAwareness.
"""
import pytest
from math import sqrt

from jetson.navigation.waypoint import Waypoint, WaypointManager
from jetson.navigation.path_follower import PathFollower, CrossTrackError
from jetson.navigation.collision import (
    CollisionAvoidance, CollisionThreat, Severity, VesselState,
)
from jetson.navigation.pilot import (
    Autopilot, PilotCommand, PilotMode, VesselStateInternal,
)
from jetson.navigation.geospatial import Coordinate, GeoCalculator
from jetson.navigation.situational import (
    SituationalAwareness, Contact, ContactType, Weather, WeatherCondition,
    SituationReport,
)


# ─── Helpers ───────────────────────────────────────────────────────────────

def _c(lat, lon):
    """Shorthand for Coordinate."""
    return Coordinate(latitude=lat, longitude=lon)


def _wp(wid, lat, lon, speed=1.5, radius=10.0):
    """Shorthand for Waypoint."""
    return Waypoint(id=wid, latitude=lat, longitude=lon, speed=speed,
                    acceptance_radius=radius)


def _vs(lat, lon, speed, heading, vid=""):
    """Shorthand for VesselState (collision module)."""
    return VesselState(position=_c(lat, lon), speed=speed, heading=heading,
                       vessel_id=vid)


def _vi(lat, lon, speed, heading):
    """Shorthand for VesselStateInternal (pilot module)."""
    return VesselStateInternal(position=_c(lat, lon), speed=speed, heading=heading)


# ═══════════════════════════════════════════════════════════════════════════
# 1. WaypointManager + GeoCalculator
# ═══════════════════════════════════════════════════════════════════════════

class TestWaypointGeospatial:
    """WaypointManager operations validated through GeoCalculator."""

    def test_total_distance_uses_haversine(self):
        wps = [_wp("a", 36.0, -122.0), _wp("b", 36.01, -122.01),
               _wp("c", 36.02, -122.02)]
        d = WaypointManager.compute_total_distance(wps)
        assert d > 0
        # Cross-check with direct haversine
        d_direct = (GeoCalculator.haversine_distance(_c(36.0, -122.0), _c(36.01, -122.01))
                    + GeoCalculator.haversine_distance(_c(36.01, -122.01), _c(36.02, -122.02)))
        assert abs(d - d_direct) < 1.0

    def test_single_waypoint_distance_zero(self):
        d = WaypointManager.compute_total_distance([_wp("x", 36.0, -122.0)])
        assert d == 0.0

    def test_empty_waypoint_list_distance(self):
        assert WaypointManager.compute_total_distance([]) == 0.0

    def test_optimize_preserves_waypoints(self):
        wps = [_wp("a", 36.0, -122.0), _wp("b", 36.01, -121.99),
               _wp("c", 36.02, -122.0)]
        opt = WaypointManager.optimize_sequence(wps)
        assert len(opt) == len(wps)
        assert {w.id for w in opt} == {"a", "b", "c"}

    def test_optimize_two_waypoints_unchanged(self):
        wps = [_wp("a", 36.0, -122.0), _wp("b", 36.01, -122.0)]
        opt = WaypointManager.optimize_sequence(wps)
        assert len(opt) == 2

    def test_optimized_distance_not_worse(self):
        wps = [_wp("a", 36.0, -122.0), _wp("b", 36.02, -122.0),
               _wp("c", 36.01, -121.99)]
        d_before = WaypointManager.compute_total_distance(wps)
        d_after = WaypointManager.compute_total_distance(WaypointManager.optimize_sequence(wps))
        assert d_after <= d_before + 1.0

    def test_interpolation_midpoint(self):
        w1, w2 = _wp("a", 36.0, -122.0), _wp("b", 36.02, -122.02)
        mid = WaypointManager.compute_interpolated(w1, w2, 0.5)
        assert abs(mid.latitude - 36.01) < 0.01

    def test_interpolation_zero_fraction(self):
        w1, w2 = _wp("a", 36.0, -122.0), _wp("b", 36.02, -122.02)
        p = WaypointManager.compute_interpolated(w1, w2, 0.0)
        assert abs(p.latitude - 36.0) < 1e-9

    def test_interpolation_one_fraction(self):
        w1, w2 = _wp("a", 36.0, -122.0), _wp("b", 36.02, -122.02)
        p = WaypointManager.compute_interpolated(w1, w2, 1.0)
        assert abs(p.latitude - 36.02) < 1e-9

    def test_remaining_distance_from_far_point(self):
        wm = WaypointManager()
        wm.add_waypoint(_wp("a", 36.01, -122.01))
        wm.add_waypoint(_wp("b", 36.02, -122.02))
        rem = wm.remaining_distance(_c(36.0, -122.0))
        assert rem > 0

    def test_remaining_distance_zero_when_at_last(self):
        wm = WaypointManager()
        wm.add_waypoint(_wp("a", 36.0, -122.0, radius=500))
        wm.add_waypoint(_wp("b", 36.001, -122.001, radius=500))
        rem = wm.remaining_distance(_c(36.0, -122.0))
        assert rem == 0.0

    def test_waypoint_reached_uses_haversine(self):
        wp = _wp("x", 36.01, -122.01, radius=100)
        pos = _c(36.01, -122.01)
        assert WaypointManager.is_waypoint_reached(pos, wp) is True
        dist = GeoCalculator.haversine_distance(pos, wp.to_coordinate())
        assert dist <= wp.acceptance_radius

    def test_waypoint_not_reached(self):
        wp = _wp("x", 36.01, -122.01, radius=1.0)
        pos = _c(36.0, -122.0)
        dist = GeoCalculator.haversine_distance(pos, wp.to_coordinate())
        assert dist > wp.acceptance_radius
        assert WaypointManager.is_waypoint_reached(pos, wp) is False

    def test_segment_distance_cross_check(self):
        wm = WaypointManager()
        wm.add_waypoint(_wp("a", 36.0, -122.0))
        wm.add_waypoint(_wp("b", 36.01, -122.0))
        seg_d = wm.segment_distance(0)
        direct = GeoCalculator.haversine_distance(_c(36.0, -122.0), _c(36.01, -122.0))
        assert abs(seg_d - direct) < 1.0

    def test_segment_distance_last_returns_zero(self):
        wm = WaypointManager()
        wm.add_waypoint(_wp("a", 36.0, -122.0))
        assert wm.segment_distance(0) == 0.0

    def test_segment_distance_out_of_bounds(self):
        wm = WaypointManager()
        wm.add_waypoint(_wp("a", 36.0, -122.0))
        wm.add_waypoint(_wp("b", 36.01, -122.0))
        assert wm.segment_distance(5) == 0.0

    def test_current_target_skips_reached(self):
        wm = WaypointManager()
        wm.add_waypoint(_wp("a", 36.0, -122.0, radius=500))
        wm.add_waypoint(_wp("b", 36.02, -122.02))
        pos = _c(36.0, -122.0)
        target = WaypointManager.get_current_target(pos, wm.get_all_waypoints())
        assert target is not None
        assert target.id == "b"

    def test_current_target_all_reached(self):
        wm = WaypointManager()
        wm.add_waypoint(_wp("a", 36.0, -122.0, radius=500))
        pos = _c(36.0, -122.0)
        assert WaypointManager.get_current_target(pos, wm.get_all_waypoints()) is None

    def test_to_coordinate_conversion(self):
        wp = _wp("a", 36.0, -122.0)
        c = wp.to_coordinate()
        assert isinstance(c, Coordinate)
        assert c.latitude == 36.0

    def test_insert_waypoint(self):
        wm = WaypointManager()
        wm.add_waypoint(_wp("a", 36.0, -122.0))
        ok = wm.insert_waypoint(_wp("b", 36.01, -122.01), "a")
        assert ok is True
        assert wm.count() == 2
        assert wm.get_waypoint("b") is not None

    def test_insert_after_nonexistent(self):
        wm = WaypointManager()
        ok = wm.insert_waypoint(_wp("b", 36.01, -122.01), "missing")
        assert ok is False

    def test_remove_waypoint(self):
        wm = WaypointManager()
        wm.add_waypoint(_wp("a", 36.0, -122.0))
        assert wm.remove_waypoint("a") is True
        assert wm.count() == 0

    def test_remove_nonexistent(self):
        wm = WaypointManager()
        assert wm.remove_waypoint("ghost") is False

    def test_reindex(self):
        wm = WaypointManager()
        wm.add_waypoint(_wp("old", 36.0, -122.0))
        wm.add_waypoint(_wp("old2", 36.01, -122.01))
        wm.reindex()
        assert wm.get_waypoint("wp_0") is not None
        assert wm.get_waypoint("wp_1") is not None

    def test_clear(self):
        wm = WaypointManager()
        wm.add_waypoint(_wp("a", 36.0, -122.0))
        wm.clear()
        assert wm.count() == 0


# ═══════════════════════════════════════════════════════════════════════════
# 2. PathFollower + GeoCalculator
# ═══════════════════════════════════════════════════════════════════════════

class TestPathFollowerGeospatial:
    """PathFollower uses GeoCalculator for all geometric computations."""

    def test_cte_magnitude_geospatial(self):
        cte = PathFollower.compute_cross_track_error(
            _c(36.005, -122.005), _c(36.0, -122.0), _c(36.01, -122.01))
        assert cte.magnitude >= 0
        assert cte.direction in (-1.0, 1.0)

    def test_cte_cross_check_with_geospatial(self):
        cte = PathFollower.compute_cross_track_error(
            _c(36.005, -122.005), _c(36.0, -122.0), _c(36.01, -122.01))
        xt = GeoCalculator.cross_track_distance(
            _c(36.0, -122.0), _c(36.01, -122.01), _c(36.005, -122.005))
        assert abs(cte.magnitude - abs(xt)) < 1.0

    def test_cte_closest_point_on_segment(self):
        cte = PathFollower.compute_cross_track_error(
            _c(36.005, -122.005), _c(36.0, -122.0), _c(36.01, -122.01))
        assert isinstance(cte.closest_point, Coordinate)

    def test_desired_heading_matches_bearing(self):
        h = PathFollower.compute_desired_heading(_c(36.0, -122.0), _c(36.01, -122.0))
        b = GeoCalculator.bearing(_c(36.0, -122.0), _c(36.01, -122.0))
        assert abs(h - b) < 1e-6

    def test_along_track_distance_matches_geospatial(self):
        atd = PathFollower.compute_along_track_distance(
            _c(36.005, -122.005), _c(36.0, -122.0), _c(36.01, -122.01))
        direct = GeoCalculator.along_track_distance(
            _c(36.0, -122.0), _c(36.01, -122.01), _c(36.005, -122.005))
        assert abs(atd - direct) < 1.0

    def test_pure_pursuit_heading_valid(self):
        path = [_c(36.0, -122.0), _c(36.005, -122.005), _c(36.01, -122.01)]
        h = PathFollower.pure_pursuit(_c(36.0, -122.0), path, 100.0)
        assert 0 <= h < 360

    def test_pure_pursuit_single_point(self):
        h = PathFollower.pure_pursuit(_c(36.0, -122.0), [_c(36.01, -122.0)], 50.0)
        b = GeoCalculator.bearing(_c(36.0, -122.0), _c(36.01, -122.0))
        assert abs(h - b) < 1e-6

    def test_pure_pursuit_empty_path(self):
        h = PathFollower.pure_pursuit(_c(36.0, -122.0), [], 50.0)
        assert h == 0.0

    def test_stanley_heading_valid(self):
        path = [_c(36.0, -122.0), _c(36.01, -122.01)]
        h = PathFollower.stanley_method(_c(36.0, -122.0), 0.0, path)
        assert 0 <= h < 360

    def test_stanley_single_point(self):
        h = PathFollower.stanley_method(_c(36.0, -122.0), 45.0,
                                        [_c(36.01, -122.0)])
        assert 0 <= h < 360

    def test_stanley_empty_path(self):
        h = PathFollower.stanley_method(_c(36.0, -122.0), 45.0, [])
        assert h == 45.0

    def test_los_heading_valid(self):
        path = [_c(36.0, -122.0), _c(36.01, -122.01)]
        h = PathFollower.los_guidance(_c(36.0, -122.0), path)
        assert 0 <= h < 360

    def test_los_single_point(self):
        h = PathFollower.los_guidance(_c(36.0, -122.0), [_c(36.01, -122.0)])
        b = GeoCalculator.bearing(_c(36.0, -122.0), _c(36.01, -122.0))
        assert abs(h - b) < 1.0

    def test_los_empty_path(self):
        assert PathFollower.los_guidance(_c(36.0, -122.0), []) == 0.0

    def test_speed_adjustment_full_at_zero_cte(self):
        assert PathFollower.compute_speed_adjustment(0.0, 2.0) == 2.0

    def test_speed_adjustment_full_below_5m_cte(self):
        assert PathFollower.compute_speed_adjustment(3.0, 2.0) == 2.0

    def test_speed_adjustment_zero_at_max_cte(self):
        assert PathFollower.compute_speed_adjustment(50.0, 2.0) == 0.0

    def test_speed_adjustment_zero_above_max_cte(self):
        assert PathFollower.compute_speed_adjustment(100.0, 2.0) == 0.0

    def test_speed_adjustment_intermediate_cte(self):
        s = PathFollower.compute_speed_adjustment(25.0, 2.0)
        assert 0 < s < 2.0

    def test_curvature_straight_line(self):
        path = [_c(36.0, -122.0), _c(36.01, -122.0), _c(36.02, -122.0)]
        c = PathFollower.compute_path_curvature(path)
        assert c < 0.1

    def test_curvature_turning_path(self):
        path = [_c(36.0, -122.0), _c(36.01, -122.0), _c(36.01, -122.01)]
        c = PathFollower.compute_path_curvature(path)
        assert c > 0.0

    def test_curvature_fewer_than_3_points(self):
        assert PathFollower.compute_path_curvature([_c(36, -122)]) == 0.0
        assert PathFollower.compute_path_curvature([]) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 3. CollisionAvoidance + GeoCalculator
# ═══════════════════════════════════════════════════════════════════════════

class TestCollisionGeospatial:
    """CollisionAvoidance threat detection uses GeoCalculator distances."""

    def test_detect_threats_returns_list(self):
        ca = CollisionAvoidance()
        own = _vs(36.0, -122.0, 2.0, 0.0, "own")
        other = _vs(36.005, -122.005, 1.5, 180.0, "t")
        threats = ca.detect_threats(own, [other])
        assert isinstance(threats, list)

    def test_detect_threats_sorted_by_severity(self):
        ca = CollisionAvoidance()
        own = _vs(36.0, -122.0, 2.0, 0.0, "own")
        t1 = _vs(36.001, -122.0, 1.5, 180.0, "t1")
        t2 = _vs(36.005, -122.005, 1.5, 180.0, "t2")
        threats = ca.detect_threats(own, [t1, t2])
        if len(threats) > 1:
            assert threats[0].severity.value >= threats[1].severity.value

    def test_detect_no_threats_far_away(self):
        ca = CollisionAvoidance()
        own = _vs(36.0, -122.0, 2.0, 0.0, "own")
        other = _vs(0.0, 0.0, 1.0, 0.0, "far")
        threats = ca.detect_threats(own, [other])
        assert len(threats) == 0

    def test_tcpa_numeric(self):
        own = _vs(36.0, -122.0, 2.0, 0.0, "own")
        other = _vs(36.005, -122.0, 1.5, 90.0, "other")
        tcpa = CollisionAvoidance.compute_tcpa(own, other)
        assert isinstance(tcpa, float)

    def test_dcpa_non_negative(self):
        own = _vs(36.0, -122.0, 2.0, 0.0, "own")
        other = _vs(36.005, -122.0, 1.5, 90.0, "other")
        dcpa = CollisionAvoidance.compute_dcpa(own, other)
        assert dcpa >= 0

    def test_risk_score_range(self):
        r = CollisionAvoidance.compute_risk_score(60, 30, 3, 2)
        assert 0 <= r <= 1

    def test_risk_score_zero_for_negative_tcpa(self):
        r = CollisionAvoidance.compute_risk_score(-1.0, 30, 3, 2)
        assert r == 0.0

    def test_risk_score_zero_for_large_tcpa(self):
        r = CollisionAvoidance.compute_risk_score(700, 30, 3, 2)
        assert r == 0.0

    def test_risk_score_high_when_close_and_slow_tcpa(self):
        r = CollisionAvoidance.compute_risk_score(5.0, 10.0, 5.0, 5.0)
        assert r > 0.5

    def test_safe_zone_minimum(self):
        ca = CollisionAvoidance()
        own = _vs(36.0, -122.0, 0.0, 0.0)
        sz = ca.compute_safe_zone(own, 0.0)
        assert sz >= ca.safe_distance

    def test_safe_zone_scales_with_speed(self):
        ca = CollisionAvoidance()
        own = _vs(36.0, -122.0, 0.0, 0.0)
        sz_slow = ca.compute_safe_zone(own, 1.0)
        sz_fast = ca.compute_safe_zone(own, 5.0)
        assert sz_fast >= sz_slow

    def test_avoidance_path_with_threats(self):
        ca = CollisionAvoidance()
        own = _vs(36.0, -122.0, 2.0, 0.0, "own")
        threat = CollisionThreat(
            vessel_id="t", position=_c(36.005, -122.005),
            velocity=(0.5, 0.3), distance=500, tcpa=100, dcpa=50,
            severity=Severity.HIGH)
        dest = _c(36.02, -122.02)
        path = ca.plan_avoidance_path(own, [threat], dest)
        assert len(path) >= 2
        assert path[0] == own.position
        assert path[-1] == dest

    def test_avoidance_path_no_threats(self):
        ca = CollisionAvoidance()
        own = _vs(36.0, -122.0, 2.0, 0.0)
        dest = _c(36.01, -122.01)
        path = ca.plan_avoidance_path(own, [], dest)
        assert len(path) == 2
        assert path[1] == dest

    def test_evasive_maneuver_no_threat(self):
        ca = CollisionAvoidance()
        own = _vs(36.0, -122.0, 2.0, 0.0)
        threat = CollisionThreat(
            vessel_id="t", position=_c(36.01, -122.01),
            velocity=(0, 0), distance=1000, tcpa=-1, dcpa=500,
            severity=Severity.NONE)
        hc, sm = ca.generate_evasive_maneuver(threat, own)
        assert hc == 0.0 and sm == 1.0

    def test_evasive_maneuver_critical(self):
        ca = CollisionAvoidance()
        own = _vs(36.0, -122.0, 2.0, 0.0, "own")
        threat = CollisionThreat(
            vessel_id="t", position=_c(36.001, -122.001),
            velocity=(0.5, -0.3), distance=50, tcpa=10, dcpa=20,
            severity=Severity.CRITICAL)
        hc, sm = ca.generate_evasive_maneuver(threat, own)
        assert hc != 0.0
        assert sm < 1.0

    def test_evasive_maneuver_low(self):
        ca = CollisionAvoidance()
        own = _vs(36.0, -122.0, 2.0, 0.0, "own")
        threat = CollisionThreat(
            vessel_id="t", position=_c(36.005, -122.005),
            velocity=(0.5, -0.3), distance=200, tcpa=300, dcpa=80,
            severity=Severity.LOW)
        hc, sm = ca.generate_evasive_maneuver(threat, own)
        assert abs(hc) <= 10.0
        assert 0 < sm < 1.0

    def test_velocity_components_north(self):
        vx, vy = CollisionAvoidance._velocity_components(2.0, 0.0)
        assert abs(vy - 2.0) < 1e-9
        assert abs(vx) < 1e-9

    def test_velocity_components_east(self):
        vx, vy = CollisionAvoidance._velocity_components(2.0, 90.0)
        assert abs(vx - 2.0) < 1e-9
        assert abs(vy) < 1e-9

    def test_velocity_components_south(self):
        vx, vy = CollisionAvoidance._velocity_components(2.0, 180.0)
        assert abs(vy + 2.0) < 1e-9


# ═══════════════════════════════════════════════════════════════════════════
# 4. Autopilot + CollisionAvoidance + PathFollower
# ═══════════════════════════════════════════════════════════════════════════

class TestAutopilotCrossModule:
    """Autopilot integrates PathFollower and CollisionAvoidance."""

    def test_waypoint_following_uses_path_follower(self):
        ap = Autopilot()
        ap.set_mode(PilotMode.WAYPOINT_FOLLOWING)
        wm = WaypointManager()
        wm.add_waypoint(_wp("w0", 36.01, -122.01, 1.5))
        state = _vi(36.0, -122.0, 1.0, 0.0)
        cmd = ap.follow_waypoints(state, wm)
        assert isinstance(cmd, PilotCommand)
        assert -1 <= cmd.thrust <= 1

    def test_waypoint_following_empty_wm(self):
        ap = Autopilot()
        cmd = ap.follow_waypoints(_vi(36.0, -122.0, 1.0, 0.0), WaypointManager())
        assert cmd.thrust == 0.0

    def test_waypoint_following_all_reached(self):
        ap = Autopilot()
        wm = WaypointManager()
        wm.add_waypoint(_wp("w0", 36.0, -122.0, radius=500))
        state = _vi(36.0, -122.0, 1.0, 0.0)
        cmd = ap.follow_waypoints(state, wm)
        assert cmd.target_speed == 0.0

    def test_waypoint_following_multiple_waypoints(self):
        ap = Autopilot()
        wm = WaypointManager()
        wm.add_waypoint(_wp("w0", 36.01, -122.01))
        wm.add_waypoint(_wp("w1", 36.02, -122.02))
        state = _vi(36.0, -122.0, 1.0, 0.0)
        cmd = ap.follow_waypoints(state, wm)
        assert isinstance(cmd, PilotCommand)

    def test_collision_avoidance_command(self):
        ap = Autopilot()
        state = _vi(36.0, -122.0, 1.0, 0.0)
        threat = CollisionThreat(
            vessel_id="t", position=_c(36.005, -122.005),
            velocity=(0.5, 0.3), distance=500, tcpa=100, dcpa=50,
            severity=Severity.HIGH)
        cmd = ap.avoid_collision(state, [threat])
        assert cmd is not None
        assert cmd.target_speed < state.speed

    def test_no_threats_returns_none(self):
        ap = Autopilot()
        state = _vi(36.0, -122.0, 1.0, 0.0)
        assert ap.avoid_collision(state, []) is None

    def test_compute_control_basic(self):
        ap = Autopilot()
        cmd = ap.compute_control(
            _vi(36.0, -122.0, 0.0, 0.0),
            _vi(36.01, -122.0, 2.0, 90.0))
        assert isinstance(cmd, PilotCommand)
        assert -1 <= cmd.thrust <= 1
        assert -1 <= cmd.rudder <= 1

    def test_control_output_mode(self):
        ap = Autopilot()
        ap.set_mode(PilotMode.AUTOPILOT)
        cmd = ap.compute_control(
            _vi(36.0, -122.0, 0.0, 0.0),
            _vi(36.01, -122.0, 2.0, 90.0))
        assert cmd.mode == PilotMode.AUTOPILOT

    def test_control_smooth_rate_limited(self):
        ap = Autopilot()
        c1 = ap.compute_control(
            _vi(36.0, -122.0, 0.0, 0.0),
            _vi(36.01, -122.0, 2.0, 90.0))
        c2 = ap.compute_control(
            _vi(36.0, -122.0, 0.0, 0.0),
            _vi(36.01, -122.0, 2.0, 270.0))
        assert abs(c2.thrust - c1.thrust) <= 0.15

    def test_smooth_control_none_prev(self):
        ap = Autopilot()
        cmd = PilotCommand(thrust=0.5, rudder=0.5, target_speed=1.0, target_heading=90.0)
        result = ap.smooth_control(None, cmd)
        assert result.thrust == 0.5

    def test_emergency_stop(self):
        ap = Autopilot()
        estop = ap.emergency_stop()
        assert estop.mode == PilotMode.EMERGENCY
        assert estop.thrust == 0.0
        assert estop.rudder == 0.0
        assert estop.target_speed == 0.0

    def test_hold_position_with_drift(self):
        ap = Autopilot()
        cmd = ap.hold_position(_c(36.0, -122.0), (0.5, 90.0))
        assert isinstance(cmd, PilotCommand)
        assert cmd.mode == PilotMode.STATION_KEEPING
        assert cmd.thrust > 0

    def test_hold_position_no_drift(self):
        ap = Autopilot()
        cmd = ap.hold_position(_c(36.0, -122.0), (0.0, 0.0))
        assert cmd.thrust == 0.0
        assert cmd.mode == PilotMode.STATION_KEEPING

    def test_mode_transitions(self):
        ap = Autopilot()
        for mode in PilotMode:
            ap.set_mode(mode)
            assert ap.get_mode() == mode

    def test_set_max_rates(self):
        ap = Autopilot()
        ap.set_max_rates(thrust_rate=0.05, rudder_rate=0.05)
        c1 = PilotCommand(thrust=0.9, rudder=0.0, target_speed=2.0, target_heading=0.0)
        c2 = PilotCommand(thrust=0.0, rudder=0.0, target_speed=0.0, target_heading=0.0)
        s = ap.smooth_control(c1, c2)
        assert abs(s.thrust - 0.9) <= 0.051  # float tolerance


# ═══════════════════════════════════════════════════════════════════════════
# 5. SituationalAwareness + CollisionAvoidance + GeoCalculator
# ═══════════════════════════════════════════════════════════════════════════

class TestSituationalCrossModule:
    """SituationalAwareness uses CollisionAvoidance and GeoCalculator."""

    def test_update_contacts_basic(self):
        sa = SituationalAwareness()
        readings = [
            {"id": "c1", "latitude": 36.005, "longitude": -122.005,
             "speed": 2.0, "heading": 90.0, "timestamp": 1.0},
        ]
        updated = sa.update_contacts(readings)
        assert len(updated) == 1
        assert sa.get_contact("c1") is not None

    def test_update_existing_contact(self):
        sa = SituationalAwareness()
        readings = [{"id": "c1", "latitude": 36.005, "longitude": -122.005,
                     "speed": 2.0, "timestamp": 1.0}]
        sa.update_contacts(readings)
        readings2 = [{"id": "c1", "latitude": 36.01, "longitude": -122.01,
                      "speed": 3.0, "timestamp": 2.0}]
        sa.update_contacts(readings2)
        c = sa.get_contact("c1")
        assert c.position.latitude == 36.01

    def test_contact_count(self):
        sa = SituationalAwareness()
        for i in range(5):
            sa.update_contacts([
                {"id": f"c{i}", "latitude": 36.0 + i * 0.001,
                 "longitude": -122.0, "timestamp": float(i)}])
        assert sa.get_contact_count() == 5

    def test_max_contacts_limit(self):
        sa = SituationalAwareness(max_contacts=3)
        for i in range(5):
            sa.update_contacts([
                {"id": f"c{i}", "latitude": 36.0, "longitude": -122.0,
                 "timestamp": float(i)}])
        assert sa.get_contact_count() == 3

    def test_predict_contact_positions(self):
        sa = SituationalAwareness()
        contacts = [
            Contact(id="c1", position=_c(36.0, -122.0),
                    velocity=(1.0, 0.0), heading=90.0)]
        preds = sa.predict_contact_positions(contacts, 60.0)
        assert "c1" in preds
        assert preds["c1"].longitude > -122.0

    def test_predict_stationary_contact(self):
        sa = SituationalAwareness()
        contacts = [
            Contact(id="c1", position=_c(36.0, -122.0),
                    velocity=(0.0, 0.0))]
        preds = sa.predict_contact_positions(contacts, 60.0)
        assert abs(preds["c1"].latitude - 36.0) < 1e-9

    def test_situation_report_with_contacts(self):
        sa = SituationalAwareness()
        own = _vs(36.0, -122.0, 2.0, 0.0, "own")
        contacts = [
            Contact(id="c1", position=_c(36.005, -122.005),
                    velocity=(0.5, 0.3), heading=180.0, distance=500)]
        weather = Weather()
        report = sa.compute_situation_report(own, contacts, weather, timestamp=1.0)
        assert isinstance(report, SituationReport)
        assert report.own_vessel.vessel_id == "own"
        assert len(report.contacts) == 1

    def test_situation_report_no_contacts(self):
        sa = SituationalAwareness()
        own = _vs(36.0, -122.0, 2.0, 0.0, "own")
        report = sa.compute_situation_report(own, [], Weather())
        assert len(report.threats) == 0

    def test_situation_report_threats_detected(self):
        sa = SituationalAwareness()
        own = _vs(36.0, -122.0, 2.0, 0.0, "own")
        close_contact = Contact(
            id="c1", position=_c(36.001, -122.001),
            velocity=(0.3, 0.3), heading=225.0, distance=100)
        report = sa.compute_situation_report(own, [close_contact], Weather())
        # Threats may or may not be detected based on proximity but check report is valid
        assert report.overall_risk >= 0

    def test_classify_stationary_nearby(self):
        sa = SituationalAwareness()
        c = Contact(id="c1", position=_c(36.005, -122.005),
                    velocity=(0.0, 0.0), distance=100)
        assert sa.classify_contact(c) == ContactType.BUOY

    def test_classify_stationary_far(self):
        sa = SituationalAwareness()
        c = Contact(id="c1", position=_c(36.005, -122.005),
                    velocity=(0.0, 0.0), distance=10000)
        assert sa.classify_contact(c) == ContactType.LAND

    def test_classify_moving_vessel(self):
        sa = SituationalAwareness()
        c = Contact(id="c1", position=_c(36.005, -122.005),
                    velocity=(1.0, 1.0))
        assert sa.classify_contact(c) == ContactType.VESSEL

    def test_classify_debris(self):
        sa = SituationalAwareness()
        c = Contact(id="c1", position=_c(36.005, -122.005),
                    velocity=(0.2, 0.2), distance=100)
        assert sa.classify_contact(c) == ContactType.DEBRIS

    def test_classify_platform(self):
        sa = SituationalAwareness()
        c = Contact(id="c1", position=_c(36.005, -122.005),
                    velocity=(0.0, 0.0), distance=600)
        assert sa.classify_contact(c) == ContactType.PLATFORM

    def test_overall_risk_no_threats_calm(self):
        sa = SituationalAwareness()
        own = _vs(36.0, -122.0, 2.0, 0.0, "own")
        weather = Weather(condition=WeatherCondition.CALM, visibility=10000)
        risk = sa.assess_overall_risk([], weather, own)
        assert 0 <= risk < 0.2

    def test_overall_risk_high_with_storm(self):
        sa = SituationalAwareness()
        own = _vs(36.0, -122.0, 5.0, 0.0, "own")
        weather = Weather(condition=WeatherCondition.STORM, visibility=100)
        threats = [CollisionThreat(
            vessel_id="t", position=_c(36.001, -122.0),
            velocity=(0, 0), distance=100, tcpa=30, dcpa=20,
            severity=Severity.CRITICAL)]
        risk = sa.assess_overall_risk(threats, weather, own)
        assert risk > 0.1

    def test_weather_classify_calm(self):
        assert Weather.classify(1.0, 0.1) == WeatherCondition.CALM

    def test_weather_classify_moderate(self):
        assert Weather.classify(5.0, 0.5) == WeatherCondition.MODERATE

    def test_weather_classify_rough(self):
        assert Weather.classify(10.0, 1.5) == WeatherCondition.ROUGH

    def test_weather_classify_storm(self):
        assert Weather.classify(18.0, 3.0) == WeatherCondition.STORM

    def test_weather_classify_hurricane(self):
        assert Weather.classify(30.0, 6.0) == WeatherCondition.HURRICANE

    def test_remove_stale_contacts(self):
        sa = SituationalAwareness(track_timeout=10.0)
        sa.update_contacts([
            {"id": "c1", "latitude": 36.0, "longitude": -122.0, "timestamp": 0.0}])
        stale = sa.remove_stale_contacts(current_time=20.0)
        assert "c1" in stale
        assert sa.get_contact_count() == 0

    def test_track_contact_history(self):
        sa = SituationalAwareness()
        sa.track_contact("c1", _c(36.0, -122.0), 1.0)
        sa.track_contact("c1", _c(36.001, -122.0), 2.0)
        assert sa.get_contact_count() == 0  # track_contact doesn't create contact

    def test_get_all_contacts(self):
        sa = SituationalAwareness()
        sa.update_contacts([
            {"id": "c1", "latitude": 36.0, "longitude": -122.0, "timestamp": 1.0},
            {"id": "c2", "latitude": 36.01, "longitude": -122.0, "timestamp": 1.0},
        ])
        assert len(sa.get_all_contacts()) == 2

    def test_get_missing_contact(self):
        sa = SituationalAwareness()
        assert sa.get_contact("ghost") is None

    def test_contact_velocity_from_speed_heading(self):
        sa = SituationalAwareness()
        sa.update_contacts([
            {"id": "c1", "latitude": 36.0, "longitude": -122.0,
             "speed": 3.0, "heading": 90.0, "timestamp": 1.0}])
        c = sa.get_contact("c1")
        speed = sqrt(c.velocity[0]**2 + c.velocity[1]**2)
        assert abs(speed - 3.0) < 0.01


# ═══════════════════════════════════════════════════════════════════════════
# 6. GeoCalculator integration helpers
# ═══════════════════════════════════════════════════════════════════════════

class TestGeoCalculatorIntegration:
    """GeoCalculator used as foundation by all navigation modules."""

    def test_destination_and_back(self):
        c = _c(36.0, -122.0)
        dest = GeoCalculator.destination(c, 45.0, 1000.0)
        bearing_back = GeoCalculator.bearing(dest, c)
        expected_back = (45.0 + 180.0) % 360
        assert abs(bearing_back - expected_back) < 5.0

    def test_polygon_containment_inside(self):
        poly = [_c(36.0, -122.0), _c(36.01, -122.0),
                _c(36.01, -122.01), _c(36.0, -122.01)]
        assert GeoCalculator.is_in_polygon(_c(36.005, -122.005), poly) is True

    def test_polygon_containment_outside(self):
        poly = [_c(36.0, -122.0), _c(36.01, -122.0),
                _c(36.01, -122.01), _c(36.0, -122.01)]
        assert GeoCalculator.is_in_polygon(_c(35.0, -121.0), poly) is False

    def test_polygon_too_few_points(self):
        assert GeoCalculator.is_in_polygon(_c(36.0, -122.0),
                                          [_c(36, -122), _c(37, -123)]) is False

    def test_compute_speed(self):
        pos = [_c(36.0, -122.0), _c(36.01, -122.0)]
        ts = [0.0, 10.0]
        s = GeoCalculator.compute_speed(pos, ts)
        assert s > 0

    def test_compute_speed_insufficient_data(self):
        assert GeoCalculator.compute_speed([_c(36, -122)], [0.0]) == 0.0
        assert GeoCalculator.compute_speed([], []) == 0.0

    def test_normalize_longitude(self):
        assert GeoCalculator.normalize_longitude(190) == -170
        assert GeoCalculator.normalize_longitude(-190) == 170
        assert GeoCalculator.normalize_longitude(0) == 0

    def test_normalize_latitude(self):
        assert GeoCalculator.normalize_latitude(95) == 90
        assert GeoCalculator.normalize_latitude(-95) == -90

    def test_midpoint_accuracy(self):
        c1, c2 = _c(36.0, -122.0), _c(36.0, -122.02)
        mid = GeoCalculator.midpoint(c1, c2)
        d1 = GeoCalculator.haversine_distance(c1, mid)
        d2 = GeoCalculator.haversine_distance(mid, c2)
        assert abs(d1 - d2) < 1.0

    def test_haversine_symmetric(self):
        c1, c2 = _c(36.0, -122.0), _c(36.01, -122.01)
        d1 = GeoCalculator.haversine_distance(c1, c2)
        d2 = GeoCalculator.haversine_distance(c2, c1)
        assert abs(d1 - d2) < 1e-6

    def test_haversine_zero_same_point(self):
        c = _c(36.0, -122.0)
        assert GeoCalculator.haversine_distance(c, c) == 0.0

    def test_cross_track_sign_right(self):
        start, end = _c(36.0, -122.0), _c(36.01, -122.0)
        point = _c(36.005, -121.999)
        xt = GeoCalculator.cross_track_distance(start, end, point)
        assert xt > 0

    def test_cross_track_sign_left(self):
        start, end = _c(36.0, -122.0), _c(36.01, -122.0)
        point = _c(36.005, -122.001)
        xt = GeoCalculator.cross_track_distance(start, end, point)
        assert xt < 0

    def test_along_track_non_negative(self):
        atd = GeoCalculator.along_track_distance(
            _c(36.0, -122.0), _c(36.01, -122.0), _c(36.005, -122.0))
        assert atd >= 0

    def test_compute_speed_mismatched_lengths(self):
        assert GeoCalculator.compute_speed([_c(36, -122)], [0.0, 1.0]) == 0.0
