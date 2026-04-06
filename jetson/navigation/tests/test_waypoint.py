"""Tests for waypoint management module."""

import pytest

from jetson.navigation.geospatial import Coordinate
from jetson.navigation.waypoint import Waypoint, WaypointManager


def make_waypoint(wp_id, lat, lon, **kwargs):
    return Waypoint(id=wp_id, latitude=lat, longitude=lon, **kwargs)


class TestWaypoint:
    def test_create_waypoint_defaults(self):
        wp = Waypoint(id="wp1", latitude=37.0, longitude=-122.0)
        assert wp.id == "wp1"
        assert wp.speed == 1.5
        assert wp.heading is None
        assert wp.acceptance_radius == 10.0

    def test_create_waypoint_custom(self):
        wp = Waypoint(id="wp2", latitude=34.0, longitude=-118.0,
                      speed=3.0, heading=45.0, acceptance_radius=25.0)
        assert wp.speed == 3.0
        assert wp.heading == 45.0
        assert wp.acceptance_radius == 25.0

    def test_to_coordinate(self):
        wp = Waypoint(id="wp1", latitude=37.0, longitude=-122.0)
        coord = wp.to_coordinate()
        assert isinstance(coord, Coordinate)
        assert coord.latitude == 37.0
        assert coord.longitude == -122.0


class TestWaypointManagerInit:
    def test_initial_state(self):
        wm = WaypointManager()
        assert wm.count() == 0
        assert wm.get_all_waypoints() == []

    def test_clear(self):
        wm = WaypointManager()
        wm.add_waypoint(make_waypoint("a", 0, 0))
        wm.add_waypoint(make_waypoint("b", 1, 1))
        assert wm.count() == 2
        wm.clear()
        assert wm.count() == 0


class TestWaypointManagerAdd:
    def test_add_single(self):
        wm = WaypointManager()
        wp = make_waypoint("wp1", 37.0, -122.0)
        wm.add_waypoint(wp)
        assert wm.count() == 1

    def test_add_multiple(self):
        wm = WaypointManager()
        for i in range(5):
            wm.add_waypoint(make_waypoint(f"wp{i}", i, -i))
        assert wm.count() == 5

    def test_add_preserves_order(self):
        wm = WaypointManager()
        wm.add_waypoint(make_waypoint("a", 0, 0))
        wm.add_waypoint(make_waypoint("b", 1, 1))
        wps = wm.get_all_waypoints()
        assert wps[0].id == "a"
        assert wps[1].id == "b"


class TestWaypointManagerInsert:
    def test_insert_after_existing(self):
        wm = WaypointManager()
        wm.add_waypoint(make_waypoint("a", 0, 0))
        wm.add_waypoint(make_waypoint("c", 2, 2))
        result = wm.insert_waypoint(make_waypoint("b", 1, 1), after_id="a")
        assert result is True
        wps = wm.get_all_waypoints()
        assert wps[0].id == "a"
        assert wps[1].id == "b"
        assert wps[2].id == "c"

    def test_insert_after_nonexistent(self):
        wm = WaypointManager()
        wm.add_waypoint(make_waypoint("a", 0, 0))
        result = wm.insert_waypoint(make_waypoint("x", 5, 5), after_id="nonexistent")
        assert result is False
        assert wm.count() == 1

    def test_insert_at_end(self):
        wm = WaypointManager()
        wm.add_waypoint(make_waypoint("a", 0, 0))
        result = wm.insert_waypoint(make_waypoint("b", 1, 1), after_id="a")
        assert result is True
        assert wm.get_all_waypoints()[-1].id == "b"


class TestWaypointManagerRemove:
    def test_remove_existing(self):
        wm = WaypointManager()
        wm.add_waypoint(make_waypoint("a", 0, 0))
        wm.add_waypoint(make_waypoint("b", 1, 1))
        result = wm.remove_waypoint("a")
        assert result is True
        assert wm.count() == 1
        assert wm.get_all_waypoints()[0].id == "b"

    def test_remove_nonexistent(self):
        wm = WaypointManager()
        result = wm.remove_waypoint("nonexistent")
        assert result is False

    def test_remove_from_empty(self):
        wm = WaypointManager()
        result = wm.remove_waypoint("a")
        assert result is False


class TestWaypointManagerGet:
    def test_get_existing(self):
        wm = WaypointManager()
        wp = make_waypoint("wp1", 37.0, -122.0)
        wm.add_waypoint(wp)
        retrieved = wm.get_waypoint("wp1")
        assert retrieved is not None
        assert retrieved.id == "wp1"

    def test_get_nonexistent(self):
        wm = WaypointManager()
        assert wm.get_waypoint("missing") is None

    def test_get_all_returns_copy(self):
        wm = WaypointManager()
        wm.add_waypoint(make_waypoint("a", 0, 0))
        wps1 = wm.get_all_waypoints()
        wm.add_waypoint(make_waypoint("b", 1, 1))
        wps2 = wm.get_all_waypoints()
        assert len(wps1) != len(wps2)


class TestInterpolate:
    def test_interpolate_zero(self):
        wp1 = make_waypoint("a", 0.0, 0.0)
        wp2 = make_waypoint("b", 10.0, 10.0)
        result = WaypointManager.compute_interpolated(wp1, wp2, 0.0)
        assert result.latitude == pytest.approx(0.0)
        assert result.longitude == pytest.approx(0.0)

    def test_interpolate_one(self):
        wp1 = make_waypoint("a", 0.0, 0.0)
        wp2 = make_waypoint("b", 10.0, 10.0)
        result = WaypointManager.compute_interpolated(wp1, wp2, 1.0)
        assert result.latitude == pytest.approx(10.0)
        assert result.longitude == pytest.approx(10.0)

    def test_interpolate_midpoint(self):
        wp1 = make_waypoint("a", 0.0, 0.0)
        wp2 = make_waypoint("b", 10.0, 20.0)
        result = WaypointManager.compute_interpolated(wp1, wp2, 0.5)
        assert result.latitude == pytest.approx(5.0)
        assert result.longitude == pytest.approx(10.0)

    def test_interpolate_clamp_high(self):
        wp1 = make_waypoint("a", 0.0, 0.0)
        wp2 = make_waypoint("b", 10.0, 10.0)
        result = WaypointManager.compute_interpolated(wp1, wp2, 2.0)
        assert result.latitude == pytest.approx(10.0)
        assert result.longitude == pytest.approx(10.0)

    def test_interpolate_clamp_low(self):
        wp1 = make_waypoint("a", 0.0, 0.0)
        wp2 = make_waypoint("b", 10.0, 10.0)
        result = WaypointManager.compute_interpolated(wp1, wp2, -1.0)
        assert result.latitude == pytest.approx(0.0)
        assert result.longitude == pytest.approx(0.0)


class TestComputeTotalDistance:
    def test_empty_waypoints(self):
        assert WaypointManager.compute_total_distance([]) == 0.0

    def test_single_waypoint(self):
        wps = [make_waypoint("a", 0, 0)]
        assert WaypointManager.compute_total_distance(wps) == 0.0

    def test_two_waypoints(self):
        wps = [
            make_waypoint("a", 0.0, 0.0),
            make_waypoint("b", 1.0, 0.0),
        ]
        dist = WaypointManager.compute_total_distance(wps)
        assert 110000 < dist < 112000  # ~1 degree latitude

    def test_multiple_waypoints(self):
        wps = [
            make_waypoint("a", 0.0, 0.0),
            make_waypoint("b", 1.0, 0.0),
            make_waypoint("c", 1.0, 1.0),
        ]
        dist = WaypointManager.compute_total_distance(wps)
        assert dist > 220000


class TestGetCurrentTarget:
    def test_no_waypoints(self):
        pos = Coordinate(latitude=0.0, longitude=0.0)
        assert WaypointManager.get_current_target(pos, []) is None

    def test_all_reached(self):
        pos = Coordinate(latitude=0.5, longitude=0.5)
        wps = [make_waypoint("a", 0.5, 0.5, acceptance_radius=100.0)]
        assert WaypointManager.get_current_target(pos, wps) is None

    def test_first_not_reached(self):
        pos = Coordinate(latitude=0.0, longitude=0.0)
        wps = [make_waypoint("a", 5.0, 5.0, acceptance_radius=1.0)]
        target = WaypointManager.get_current_target(pos, wps)
        assert target is not None
        assert target.id == "a"

    def test_second_target(self):
        pos = Coordinate(latitude=0.0, longitude=0.0)
        wps = [
            make_waypoint("a", 0.0, 0.0, acceptance_radius=100.0),
            make_waypoint("b", 5.0, 5.0, acceptance_radius=1.0),
        ]
        target = WaypointManager.get_current_target(pos, wps)
        assert target is not None
        assert target.id == "b"


class TestIsWaypointReached:
    def test_reached(self):
        pos = Coordinate(latitude=0.0, longitude=0.0)
        wp = make_waypoint("a", 0.0, 0.0, acceptance_radius=10.0)
        assert WaypointManager.is_waypoint_reached(pos, wp) is True

    def test_not_reached(self):
        pos = Coordinate(latitude=0.0, longitude=0.0)
        wp = make_waypoint("a", 5.0, 0.0, acceptance_radius=1.0)
        assert WaypointManager.is_waypoint_reached(pos, wp) is False

    def test_edge_of_radius(self):
        pos = Coordinate(latitude=0.0, longitude=0.0)
        # 1 degree ~ 111 km, 0.0001 degree ~ 11.1 m
        wp = make_waypoint("a", 0.0001, 0.0, acceptance_radius=12.0)
        assert WaypointManager.is_waypoint_reached(pos, wp) is True


class TestOptimizeSequence:
    def test_empty_list(self):
        result = WaypointManager.optimize_sequence([])
        assert result == []

    def test_single_waypoint(self):
        wps = [make_waypoint("a", 0, 0)]
        result = WaypointManager.optimize_sequence(wps)
        assert len(result) == 1
        assert result[0].id == "a"

    def test_two_waypoints(self):
        wps = [
            make_waypoint("a", 0, 0),
            make_waypoint("b", 1, 1),
        ]
        result = WaypointManager.optimize_sequence(wps)
        assert len(result) == 2

    def test_improves_distance(self):
        wps = [
            make_waypoint("a", 0, 0),
            make_waypoint("b", 0, 5),
            make_waypoint("c", 0, 4),
            make_waypoint("d", 0, 3),
            make_waypoint("e", 0, 2),
            make_waypoint("f", 0, 1),
        ]
        original_dist = WaypointManager.compute_total_distance(wps)
        optimized = WaypointManager.optimize_sequence(wps)
        optimized_dist = WaypointManager.compute_total_distance(optimized)
        assert optimized_dist <= original_dist

    def test_preserves_first_waypoint(self):
        wps = [
            make_waypoint("a", 0, 0),
            make_waypoint("b", 0, 5),
            make_waypoint("c", 0, 1),
        ]
        result = WaypointManager.optimize_sequence(wps)
        assert result[0].id == "a"


class TestReindex:
    def test_reindex_all(self):
        wm = WaypointManager()
        wm.add_waypoint(make_waypoint("old1", 0, 0))
        wm.add_waypoint(make_waypoint("old2", 1, 1))
        wm.add_waypoint(make_waypoint("old3", 2, 2))
        wm.reindex()
        wps = wm.get_all_waypoints()
        assert wps[0].id == "wp_0"
        assert wps[1].id == "wp_1"
        assert wps[2].id == "wp_2"


class TestSegmentDistance:
    def test_valid_segment(self):
        wm = WaypointManager()
        wm.add_waypoint(make_waypoint("a", 0, 0))
        wm.add_waypoint(make_waypoint("b", 1, 0))
        dist = wm.segment_distance(0)
        assert 110000 < dist < 112000

    def test_last_waypoint(self):
        wm = WaypointManager()
        wm.add_waypoint(make_waypoint("a", 0, 0))
        assert wm.segment_distance(0) == 0.0

    def test_out_of_bounds(self):
        wm = WaypointManager()
        wm.add_waypoint(make_waypoint("a", 0, 0))
        wm.add_waypoint(make_waypoint("b", 1, 0))
        assert wm.segment_distance(5) == 0.0
        assert wm.segment_distance(-1) == 0.0


class TestRemainingDistance:
    def test_all_reached(self):
        wm = WaypointManager()
        wm.add_waypoint(make_waypoint("a", 0, 0, acceptance_radius=1000))
        pos = Coordinate(latitude=0, longitude=0)
        assert wm.remaining_distance(pos) == 0.0

    def test_none_reached(self):
        wm = WaypointManager()
        wm.add_waypoint(make_waypoint("a", 1, 0, acceptance_radius=1))
        wm.add_waypoint(make_waypoint("b", 2, 0, acceptance_radius=1))
        pos = Coordinate(latitude=0, longitude=0)
        dist = wm.remaining_distance(pos)
        assert dist > 0

    def test_empty_waypoints(self):
        wm = WaypointManager()
        pos = Coordinate(latitude=0, longitude=0)
        assert wm.remaining_distance(pos) == 0.0
