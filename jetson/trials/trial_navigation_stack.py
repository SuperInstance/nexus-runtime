"""Trial 1: Navigation Stack — Nav + Collision + Geospatial + Pilot integration.

Tests cross-module interaction between waypoint management, path following,
collision avoidance, autopilot, and geospatial calculations.
"""

from math import radians

from jetson.navigation.waypoint import Waypoint, WaypointManager
from jetson.navigation.path_follower import PathFollower, CrossTrackError
from jetson.navigation.collision import (
    CollisionAvoidance, CollisionThreat, Severity, VesselState,
)
from jetson.navigation.pilot import (
    Autopilot, PilotCommand, PilotMode, VesselStateInternal,
)
from jetson.navigation.geospatial import Coordinate, GeoCalculator


def _make_coordinate(lat, lon):
    return Coordinate(latitude=lat, longitude=lon)


def _make_waypoint(wp_id, lat, lon, speed=1.5):
    return Waypoint(id=wp_id, latitude=lat, longitude=lon, speed=speed)


def _make_vessel_state(lat, lon, speed, heading, vid=""):
    return VesselState(
        position=_make_coordinate(lat, lon),
        speed=speed, heading=heading, vessel_id=vid,
    )


def _make_internal_state(lat, lon, speed, heading):
    return VesselStateInternal(
        position=_make_coordinate(lat, lon), speed=speed, heading=heading,
    )


def run_trial():
    """Run all navigation stack integration tests. Returns True if all pass."""
    passed = 0
    failed = 0
    total = 0

    def check(name, condition):
        nonlocal passed, failed, total
        total += 1
        if condition:
            passed += 1
        else:
            failed += 1

    # --- 1. Waypoint + Geospatial: waypoints created with geospatial coords ---
    wm = WaypointManager()
    wm.add_waypoint(_make_waypoint("wp0", 36.0, -122.0))
    wm.add_waypoint(_make_waypoint("wp1", 36.01, -122.01))
    wm.add_waypoint(_make_waypoint("wp2", 36.02, -122.02))
    check("waypoint_count_3", wm.count() == 3)
    check("waypoint_distance_positive", WaypointManager.compute_total_distance(wm.get_all_waypoints()) > 0)

    # --- 2. Geospatial: haversine between waypoints ---
    c1 = _make_coordinate(36.0, -122.0)
    c2 = _make_coordinate(36.01, -122.01)
    dist = GeoCalculator.haversine_distance(c1, c2)
    check("haversine_reasonable", 100 < dist < 2000)

    # --- 3. Geospatial: bearing between coordinates ---
    bearing = GeoCalculator.bearing(c1, c2)
    check("bearing_in_range", 0 <= bearing < 360)

    # --- 4. Geospatial: destination point ---
    dest = GeoCalculator.destination(c1, 0, 1000)
    check("destination_not_same", abs(dest.latitude - c1.latitude) > 0.001 or abs(dest.longitude - c1.longitude) > 0.001)

    # --- 5. Geospatial: midpoint ---
    mid = GeoCalculator.midpoint(c1, c2)
    check("midpoint_between", abs(mid.latitude - (c1.latitude + c2.latitude) / 2) < 0.1)

    # --- 6. PathFollower + Geospatial: cross-track error ---
    cte = PathFollower.compute_cross_track_error(
        _make_coordinate(36.005, -122.005), c1, c2,
    )
    check("cte_has_magnitude", cte.magnitude >= 0)
    check("cte_has_direction", cte.direction in (-1.0, 1.0))

    # --- 7. PathFollower + Geospatial: along-track distance ---
    atd = PathFollower.compute_along_track_distance(
        _make_coordinate(36.005, -122.005), c1, c2,
    )
    check("atd_non_negative", atd >= 0)

    # --- 8. PathFollower + Geospatial: desired heading ---
    dh = PathFollower.compute_desired_heading(c1, c2)
    check("desired_heading_valid", 0 <= dh < 360)

    # --- 9. PathFollower + Geospatial: pure pursuit ---
    path = [c1, c2, _make_coordinate(36.02, -122.02)]
    heading_pp = PathFollower.pure_pursuit(_make_coordinate(36.0, -122.0), path, 50.0)
    check("pure_pursuit_heading_valid", 0 <= heading_pp < 360)

    # --- 10. PathFollower + Geospatial: stanley method ---
    heading_stanley = PathFollower.stanley_method(
        _make_coordinate(36.0, -122.0), 0.0, path,
    )
    check("stanley_heading_valid", 0 <= heading_stanley < 360)

    # --- 11. PathFollower + Geospatial: LOS guidance ---
    heading_los = PathFollower.los_guidance(
        _make_coordinate(36.0, -122.0), path,
    )
    check("los_heading_valid", 0 <= heading_los < 360)

    # --- 12. PathFollower: speed adjustment from CTE ---
    speed = PathFollower.compute_speed_adjustment(2.0, 1.5)
    check("speed_adjust_cte_small", speed == 1.5)
    speed_high_cte = PathFollower.compute_speed_adjustment(60.0, 1.5)
    check("speed_adjust_cte_large", speed_high_cte == 0.0)

    # --- 13. PathFollower: path curvature ---
    curvature = PathFollower.compute_path_curvature(path)
    check("curvature_non_negative", curvature >= 0)

    # --- 14. CollisionAvoidance + Geospatial: detect threats ---
    ca = CollisionAvoidance()
    own = _make_vessel_state(36.0, -122.0, 2.0, 0.0, "own")
    others = [_make_vessel_state(36.005, -122.005, 1.5, 180.0, "other1")]
    threats = ca.detect_threats(own, others)
    check("detect_threats_list", isinstance(threats, list))

    # --- 15. CollisionAvoidance + Geospatial: TCPA ---
    tcpa = CollisionAvoidance.compute_tcpa(own, others[0])
    check("tcpa_numeric", isinstance(tcpa, (int, float)))

    # --- 16. CollisionAvoidance + Geospatial: DCPA ---
    dcpa = CollisionAvoidance.compute_dcpa(own, others[0])
    check("dcpa_non_negative", dcpa >= 0)

    # --- 17. CollisionAvoidance: risk score ---
    risk = CollisionAvoidance.compute_risk_score(100.0, 50.0, 2.0, 1.5)
    check("risk_in_range", 0 <= risk <= 1)

    # --- 18. CollisionAvoidance: safe zone ---
    sz = ca.compute_safe_zone(own, 2.0)
    check("safe_zone_positive", sz >= 100)

    # --- 19. CollisionAvoidance + Geospatial: avoidance path ---
    threat = CollisionThreat(
        vessel_id="t1", position=_make_coordinate(36.005, -122.005),
        velocity=(0.5, 0.3), distance=500, tcpa=100, dcpa=50,
        severity=Severity.HIGH,
    )
    avoidance_path = ca.plan_avoidance_path(own, [threat], _make_coordinate(36.02, -122.02))
    check("avoidance_path_has_dest", len(avoidance_path) >= 2)
    check("avoidance_path_starts_at_own", avoidance_path[0] == own.position)

    # --- 20. Autopilot: compute control ---
    ap = Autopilot()
    current = _make_internal_state(36.0, -122.0, 1.0, 45.0)
    target = _make_internal_state(36.01, -122.0, 1.5, 90.0)
    cmd = ap.compute_control(current, target)
    check("pilot_command_type", isinstance(cmd, PilotCommand))
    check("pilot_thrust_range", -1 <= cmd.thrust <= 1)
    check("pilot_rudder_range", -1 <= cmd.rudder <= 1)

    # --- 21. Autopilot: waypoint following ---
    ap.set_mode(PilotMode.WAYPOINT_FOLLOWING)
    wm2 = WaypointManager()
    wm2.add_waypoint(_make_waypoint("w0", 36.01, -122.01, 1.5))
    wm2.add_waypoint(_make_waypoint("w1", 36.02, -122.02, 1.5))
    wp_cmd = ap.follow_waypoints(current, wm2)
    check("wp_follow_command", isinstance(wp_cmd, PilotCommand))

    # --- 22. Autopilot: emergency stop ---
    estop = ap.emergency_stop()
    check("estop_zero_thrust", estop.thrust == 0.0)
    check("estop_zero_rudder", estop.rudder == 0.0)
    check("estop_mode", estop.mode == PilotMode.EMERGENCY)

    # --- 23. Autopilot: station keeping ---
    hold = ap.hold_position(_make_coordinate(36.0, -122.0), (0.5, 90.0))
    check("hold_command_type", isinstance(hold, PilotCommand))

    # --- 24. Autopilot + CollisionAvoidance: collision avoidance command ---
    collision_cmd = ap.avoid_collision(current, [threat])
    check("collision_avoidance_returns_cmd", collision_cmd is not None)
    if collision_cmd:
        check("collision_cmd_thrust", -1 <= collision_cmd.thrust <= 1)

    # --- 25. Autopilot + CollisionAvoidance: no threats = None ---
    no_threats = ap.avoid_collision(current, [])
    check("no_threats_returns_none", no_threats is None)

    # --- 26. Autopilot: smooth control ---
    ap2 = Autopilot()
    cmd1 = PilotCommand(thrust=0.8, rudder=0.6, target_speed=2.0, target_heading=90.0)
    cmd2 = PilotCommand(thrust=0.2, rudder=-0.8, target_speed=0.5, target_heading=270.0)
    smooth = ap2.smooth_control(None, cmd1)
    check("smooth_first_call", smooth.thrust == cmd1.thrust)
    smooth2 = ap2.smooth_control(cmd1, cmd2)
    check("smooth_rate_limited", abs(smooth2.thrust - cmd1.thrust) <= 0.1)

    # --- 27. Autopilot: set mode ---
    ap.set_mode(PilotMode.STATION_KEEPING)
    check("mode_changed", ap.get_mode() == PilotMode.STATION_KEEPING)

    # --- 28. Waypoint + Geospatial: interpolation ---
    wpa = _make_waypoint("a", 36.0, -122.0)
    wpb = _make_waypoint("b", 36.02, -122.02)
    interp = WaypointManager.compute_interpolated(wpa, wpb, 0.5)
    check("interp_lat", abs(interp.latitude - 36.01) < 0.01)

    # --- 29. Waypoint + Geospatial: remaining distance ---
    rem = wm2.remaining_distance(_make_coordinate(36.0, -122.0))
    check("remaining_dist_positive", rem > 0)

    # --- 30. Waypoint: current target ---
    target_wp = WaypointManager.get_current_target(
        _make_coordinate(36.0, -122.0), wm2.get_all_waypoints(),
    )
    check("current_target_exists", target_wp is not None)

    # --- 31. Waypoint: reached check ---
    reached = WaypointManager.is_waypoint_reached(
        _make_coordinate(36.01, -122.01), _make_waypoint("w0", 36.01, -122.01),
    )
    check("waypoint_reached", reached is True)

    # --- 32. Waypoint: optimize sequence ---
    wps = [
        _make_waypoint("a", 36.0, -122.0),
        _make_waypoint("b", 36.01, -121.99),
        _make_waypoint("c", 36.02, -122.0),
    ]
    optimized = WaypointManager.optimize_sequence(wps)
    check("optimize_same_count", len(optimized) == 3)

    # --- 33. Geospatial: polygon containment ---
    polygon = [
        _make_coordinate(36.0, -122.0),
        _make_coordinate(36.01, -122.0),
        _make_coordinate(36.01, -122.01),
        _make_coordinate(36.0, -122.01),
    ]
    inside = GeoCalculator.is_in_polygon(
        _make_coordinate(36.005, -122.005), polygon,
    )
    check("polygon_inside", inside is True)
    outside = GeoCalculator.is_in_polygon(
        _make_coordinate(35.0, -121.0), polygon,
    )
    check("polygon_outside", outside is False)

    # --- 34. Geospatial: speed computation ---
    positions = [_make_coordinate(36.0, -122.0), _make_coordinate(36.01, -122.0)]
    timestamps = [0.0, 10.0]
    speed = GeoCalculator.compute_speed(positions, timestamps)
    check("speed_computed", speed > 0)

    # --- 35. Geospatial: normalize ---
    check("normalize_lon", GeoCalculator.normalize_longitude(190) == -170)
    check("normalize_lat", GeoCalculator.normalize_latitude(95) == 90)

    # --- 36. Autopilot + CollisionAvoidance: evasive maneuver integration ---
    own_vs = _make_vessel_state(36.0, -122.0, 2.0, 0.0, "own")
    closing = _make_vessel_state(36.002, -122.0, 1.8, 180.0, "closing")
    ca2 = CollisionAvoidance(warning_distance=500, critical_distance=100)
    threats2 = ca2.detect_threats(own_vs, [closing])
    if threats2:
        heading_chg, speed_mult = ca2.generate_evasive_maneuver(threats2[0], own_vs)
        check("evasive_heading_nonzero", heading_chg != 0.0)
        check("evasive_speed_valid", 0 < speed_mult <= 1.0)

    # --- 37. Waypoint: insert and remove ---
    wm3 = WaypointManager()
    wm3.add_waypoint(_make_waypoint("w0", 36.0, -122.0))
    wm3.insert_waypoint(_make_waypoint("w1", 36.01, -122.01), "w0")
    check("insert_after", wm3.count() == 2)
    wm3.remove_waypoint("w1")
    check("remove_success", wm3.count() == 1)
    check("remove_nonexistent", wm3.remove_waypoint("nonexistent") is False)

    # --- 38. Waypoint: segment distance ---
    wm4 = WaypointManager()
    wm4.add_waypoint(_make_waypoint("a", 36.0, -122.0))
    wm4.add_waypoint(_make_waypoint("b", 36.01, -122.01))
    seg_dist = wm4.segment_distance(0)
    check("seg_dist_positive", seg_dist > 0)

    return failed == 0
