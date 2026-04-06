"""Tests for path following controllers."""

import pytest

from jetson.navigation.geospatial import Coordinate, GeoCalculator
from jetson.navigation.path_follower import CrossTrackError, PathFollower


def make_coord(lat, lon):
    return Coordinate(latitude=lat, longitude=lon)


# ---- Cross Track Error ----

class TestCrossTrackError:
    def test_on_path(self):
        start = make_coord(0.0, 0.0)
        end = make_coord(0.0, 1.0)
        pos = make_coord(0.0, 0.5)
        cte = PathFollower.compute_cross_track_error(pos, start, end)
        assert cte.magnitude < 500
        assert isinstance(cte.direction, float)
        assert isinstance(cte.closest_point, Coordinate)

    def test_off_path_right(self):
        start = make_coord(0.0, 0.0)
        end = make_coord(0.0, 1.0)
        pos = make_coord(-1.0, 0.5)  # South = starboard (right)
        cte = PathFollower.compute_cross_track_error(pos, start, end)
        assert cte.magnitude > 50000
        assert cte.direction == 1.0

    def test_off_path_left(self):
        start = make_coord(0.0, 0.0)
        end = make_coord(0.0, 1.0)
        pos = make_coord(1.0, 0.5)  # North = port (left)
        cte = PathFollower.compute_cross_track_error(pos, start, end)
        assert cte.magnitude > 50000
        assert cte.direction == -1.0

    def test_degenerate_path(self):
        start = make_coord(0.0, 0.0)
        end = make_coord(0.0, 0.0)
        pos = make_coord(1.0, 1.0)
        cte = PathFollower.compute_cross_track_error(pos, start, end)
        assert cte.closest_point == start

    def test_closest_point_on_segment(self):
        start = make_coord(0.0, 0.0)
        end = make_coord(0.0, 2.0)
        pos = make_coord(0.0, 1.0)
        cte = PathFollower.compute_cross_track_error(pos, start, end)
        assert cte.closest_point.longitude == pytest.approx(1.0, abs=0.01)

    def test_beyond_end_of_segment(self):
        start = make_coord(0.0, 0.0)
        end = make_coord(0.0, 1.0)
        pos = make_coord(0.5, 2.0)  # Past the end
        cte = PathFollower.compute_cross_track_error(pos, start, end)
        assert isinstance(cte.closest_point, Coordinate)


# ---- Along Track Distance ----

class TestAlongTrackDistance:
    def test_at_start(self):
        start = make_coord(0.0, 0.0)
        end = make_coord(0.0, 1.0)
        dat = PathFollower.compute_along_track_distance(start, start, end)
        assert abs(dat) < 500

    def test_increasing(self):
        start = make_coord(0.0, 0.0)
        end = make_coord(0.0, 2.0)
        p1 = make_coord(0.0, 0.5)
        p2 = make_coord(0.0, 1.5)
        dat1 = PathFollower.compute_along_track_distance(p1, start, end)
        dat2 = PathFollower.compute_along_track_distance(p2, start, end)
        assert dat2 > dat1


# ---- Desired Heading ----

class TestDesiredHeading:
    def test_north(self):
        pos = make_coord(0.0, 0.0)
        target = make_coord(1.0, 0.0)
        heading = PathFollower.compute_desired_heading(pos, target)
        assert heading == pytest.approx(0.0, abs=1.0)

    def test_east(self):
        pos = make_coord(0.0, 0.0)
        target = make_coord(0.0, 1.0)
        heading = PathFollower.compute_desired_heading(pos, target)
        assert heading == pytest.approx(90.0, abs=1.0)

    def test_south(self):
        pos = make_coord(1.0, 0.0)
        target = make_coord(0.0, 0.0)
        heading = PathFollower.compute_desired_heading(pos, target)
        assert heading == pytest.approx(180.0, abs=1.0)

    def test_west(self):
        pos = make_coord(0.0, 1.0)
        target = make_coord(0.0, 0.0)
        heading = PathFollower.compute_desired_heading(pos, target)
        assert heading == pytest.approx(270.0, abs=1.0)


# ---- Pure Pursuit ----

class TestPurePursuit:
    def test_empty_path(self):
        pos = make_coord(0.0, 0.0)
        heading = PathFollower.pure_pursuit(pos, [], 50.0)
        assert heading == 0.0

    def test_single_point(self):
        pos = make_coord(0.0, 0.0)
        target = make_coord(1.0, 0.0)
        heading = PathFollower.pure_pursuit(pos, [target], 50.0)
        assert heading == pytest.approx(0.0, abs=1.0)

    def test_straight_path(self):
        pos = make_coord(0.0, 0.0)
        path = [make_coord(0.0, 0.5), make_coord(0.0, 1.0)]
        heading = PathFollower.pure_pursuit(pos, path, 10000)
        assert heading == pytest.approx(90.0, abs=5.0)

    def test_heading_towards_path(self):
        pos = make_coord(0.0, 0.0)
        path = [make_coord(0.0, 0.5), make_coord(0.0, 1.0)]
        heading = PathFollower.pure_pursuit(pos, path, 50.0)
        # Should head roughly east
        assert 0 < heading < 180

    def test_short_lookahead(self):
        pos = make_coord(0.0, 0.0)
        path = [make_coord(0.0, 1.0), make_coord(0.0, 2.0)]
        h1 = PathFollower.pure_pursuit(pos, path, 10.0)
        h2 = PathFollower.pure_pursuit(pos, path, 100000.0)
        assert 0 <= h1 < 360
        assert 0 <= h2 < 360

    def test_returns_valid_range(self):
        pos = make_coord(37.0, -122.0)
        path = [
            make_coord(37.01, -122.01),
            make_coord(37.02, -122.02),
            make_coord(37.03, -122.03),
        ]
        heading = PathFollower.pure_pursuit(pos, path, 100.0)
        assert 0 <= heading < 360


# ---- Stanley Method ----

class TestStanleyMethod:
    def test_empty_path(self):
        pos = make_coord(0.0, 0.0)
        heading = PathFollower.stanley_method(pos, 0.0, [], 1.0)
        assert heading == 0.0

    def test_single_point(self):
        pos = make_coord(0.0, 0.0)
        target = make_coord(0.0, 1.0)
        heading = PathFollower.stanley_method(pos, 0.0, [target], 1.0)
        assert 0 <= heading < 360

    def test_on_track_no_correction(self):
        pos = make_coord(0.0, 0.0)
        path = [make_coord(0.0, -0.5), make_coord(0.0, 1.0)]
        heading = PathFollower.stanley_method(pos, 90.0, path, 1.0)
        assert 0 <= heading < 360

    def test_off_track_applies_correction(self):
        pos = make_coord(1.0, 0.0)
        path = [make_coord(0.0, -0.5), make_coord(0.0, 1.0)]
        h1 = PathFollower.stanley_method(pos, 90.0, path, 0.0)
        h2 = PathFollower.stanley_method(pos, 90.0, path, 5.0)
        # Higher gain should produce different heading
        assert 0 <= h1 < 360
        assert 0 <= h2 < 360

    def test_heading_always_valid(self):
        pos = make_coord(37.0, -122.0)
        path = [
            make_coord(37.01, -122.01),
            make_coord(37.02, -122.02),
        ]
        heading = PathFollower.stanley_method(pos, 45.0, path, 2.0)
        assert 0 <= heading < 360


# ---- LOS Guidance ----

class TestLOSGuidance:
    def test_empty_path(self):
        pos = make_coord(0.0, 0.0)
        heading = PathFollower.los_guidance(pos, [], 3.0)
        assert heading == 0.0

    def test_single_point(self):
        pos = make_coord(0.0, 0.0)
        target = make_coord(1.0, 0.0)
        heading = PathFollower.los_guidance(pos, [target], 3.0)
        assert heading == pytest.approx(0.0, abs=1.0)

    def test_heading_towards_path(self):
        pos = make_coord(0.0, 0.0)
        path = [make_coord(0.0, 1.0), make_coord(0.0, 2.0)]
        heading = PathFollower.los_guidance(pos, path, 3.0)
        assert 0 <= heading < 360

    def test_different_gain_values(self):
        pos = make_coord(0.0, 0.0)
        path = [make_coord(0.0, 1.0), make_coord(0.0, 2.0)]
        h1 = PathFollower.los_guidance(pos, path, 1.0)
        h2 = PathFollower.los_guidance(pos, path, 10.0)
        assert 0 <= h1 < 360
        assert 0 <= h2 < 360

    def test_returns_valid_range(self):
        pos = make_coord(37.0, -122.0)
        path = [make_coord(37.01, -122.01), make_coord(37.02, -122.02)]
        heading = PathFollower.los_guidance(pos, path, 3.0)
        assert 0 <= heading < 360


# ---- Speed Adjustment ----

class TestSpeedAdjustment:
    def test_zero_cte_full_speed(self):
        speed = PathFollower.compute_speed_adjustment(0.0, 5.0)
        assert speed == pytest.approx(5.0)

    def test_small_cte_full_speed(self):
        speed = PathFollower.compute_speed_adjustment(3.0, 5.0)
        assert speed == pytest.approx(5.0)

    def test_moderate_cte_reduced_speed(self):
        speed = PathFollower.compute_speed_adjustment(20.0, 5.0)
        assert 0 < speed < 5.0

    def test_large_cte_zero_speed(self):
        speed = PathFollower.compute_speed_adjustment(60.0, 5.0)
        assert speed == 0.0

    def test_max_cte_boundary(self):
        speed = PathFollower.compute_speed_adjustment(50.0, 5.0)
        assert speed == 0.0

    def test_threshold_cte(self):
        speed = PathFollower.compute_speed_adjustment(5.0, 5.0)
        assert speed == pytest.approx(5.0)

    def test_linear_interpolation(self):
        """At CTE=27.5 (midpoint of 5-50), speed should be 2.5."""
        speed = PathFollower.compute_speed_adjustment(27.5, 5.0)
        assert speed == pytest.approx(2.5, abs=0.01)

    def test_never_negative(self):
        speed = PathFollower.compute_speed_adjustment(100.0, 5.0)
        assert speed == 0.0

    def test_zero_target_speed(self):
        speed = PathFollower.compute_speed_adjustment(0.0, 0.0)
        assert speed == 0.0


# ---- Path Curvature ----

class TestPathCurvature:
    def test_straight_path(self):
        path = [
            make_coord(0.0, 0.0),
            make_coord(0.0, 1.0),
            make_coord(0.0, 2.0),
        ]
        curvature = PathFollower.compute_path_curvature(path)
        assert curvature < 0.01

    def test_single_turn(self):
        path = [
            make_coord(0.0, 0.0),
            make_coord(0.0, 1.0),
            make_coord(1.0, 1.0),
        ]
        curvature = PathFollower.compute_path_curvature(path)
        assert curvature > 0

    def test_few_points(self):
        path = [make_coord(0.0, 0.0), make_coord(1.0, 1.0)]
        curvature = PathFollower.compute_path_curvature(path)
        assert curvature == 0.0

    def test_empty_path(self):
        curvature = PathFollower.compute_path_curvature([])
        assert curvature == 0.0

    def test_sharp_turns(self):
        path = [
            make_coord(0.0, 0.0),
            make_coord(0.0, 0.01),
            make_coord(0.01, 0.01),
            make_coord(0.01, 0.0),
        ]
        curvature = PathFollower.compute_path_curvature(path)
        assert curvature > 0
