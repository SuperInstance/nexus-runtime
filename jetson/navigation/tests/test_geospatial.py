"""Tests for geospatial calculations module."""

import math
import pytest

from jetson.navigation.geospatial import Coordinate, GeoCalculator


# ---- Coordinate ----

class TestCoordinate:
    def test_create_coordinate(self):
        c = Coordinate(latitude=37.7749, longitude=-122.4194)
        assert c.latitude == pytest.approx(37.7749, abs=1e-6)
        assert c.longitude == pytest.approx(-122.4194, abs=1e-6)

    def test_coordinate_is_frozen(self):
        c = Coordinate(latitude=0.0, longitude=0.0)
        with pytest.raises(AttributeError):
            c.latitude = 1.0

    def test_coordinate_equality(self):
        c1 = Coordinate(latitude=37.0, longitude=-122.0)
        c2 = Coordinate(latitude=37.0, longitude=-122.0)
        assert c1 == c2

    def test_coordinate_inequality(self):
        c1 = Coordinate(latitude=37.0, longitude=-122.0)
        c2 = Coordinate(latitude=38.0, longitude=-122.0)
        assert c1 != c2

    def test_coordinate_hash(self):
        c1 = Coordinate(latitude=37.0, longitude=-122.0)
        c2 = Coordinate(latitude=37.0, longitude=-122.0)
        assert hash(c1) == hash(c2)

    def test_coordinate_repr(self):
        c = Coordinate(latitude=37.0, longitude=-122.0)
        r = repr(c)
        assert "37.0" in r
        assert "-122.0" in r


# ---- Haversine Distance ----

class TestHaversineDistance:
    def test_same_point(self):
        c = Coordinate(latitude=37.7749, longitude=-122.4194)
        assert GeoCalculator.haversine_distance(c, c) == pytest.approx(0.0, abs=1.0)

    def test_known_distance_sf_la(self):
        """San Francisco to Los Angeles ~559 km."""
        sf = Coordinate(latitude=37.7749, longitude=-122.4194)
        la = Coordinate(latitude=34.0522, longitude=-118.2437)
        dist = GeoCalculator.haversine_distance(sf, la)
        assert 550000 < dist < 570000  # meters

    def test_known_distance_london_paris(self):
        """London to Paris ~343 km."""
        london = Coordinate(latitude=51.5074, longitude=-0.1278)
        paris = Coordinate(latitude=48.8566, longitude=2.3522)
        dist = GeoCalculator.haversine_distance(london, paris)
        assert 330000 < dist < 360000

    def test_equator_distance(self):
        """1 degree of longitude at equator ~111 km."""
        c1 = Coordinate(latitude=0.0, longitude=0.0)
        c2 = Coordinate(latitude=0.0, longitude=1.0)
        dist = GeoCalculator.haversine_distance(c1, c2)
        assert 110000 < dist < 112000

    def test_north_south_distance(self):
        """1 degree of latitude ~111 km everywhere."""
        c1 = Coordinate(latitude=0.0, longitude=0.0)
        c2 = Coordinate(latitude=1.0, longitude=0.0)
        dist = GeoCalculator.haversine_distance(c1, c2)
        assert 110000 < dist < 112000

    def test_very_short_distance(self):
        c1 = Coordinate(latitude=37.7749, longitude=-122.4194)
        c2 = Coordinate(latitude=37.77491, longitude=-122.4194)
        dist = GeoCalculator.haversine_distance(c1, c2)
        assert 0 < dist < 100

    def test_antipodal_points(self):
        """Antipodal points should be ~20,000 km apart."""
        c1 = Coordinate(latitude=0.0, longitude=0.0)
        c2 = Coordinate(latitude=0.0, longitude=180.0)
        dist = GeoCalculator.haversine_distance(c1, c2)
        assert 19900000 < dist < 20100000

    def test_negative_coordinates(self):
        c1 = Coordinate(latitude=-33.8688, longitude=151.2093)  # Sydney
        c2 = Coordinate(latitude=-37.8136, longitude=144.9631)  # Melbourne
        dist = GeoCalculator.haversine_distance(c1, c2)
        assert 700000 < dist < 750000

    def test_symmetry(self):
        c1 = Coordinate(latitude=40.0, longitude=-74.0)
        c2 = Coordinate(latitude=51.5, longitude=-0.1)
        assert GeoCalculator.haversine_distance(c1, c2) == pytest.approx(
            GeoCalculator.haversine_distance(c2, c1), abs=1.0
        )


# ---- Bearing ----

class TestBearing:
    def test_bearing_north(self):
        """Due north should be 0 degrees."""
        c1 = Coordinate(latitude=0.0, longitude=0.0)
        c2 = Coordinate(latitude=1.0, longitude=0.0)
        bearing = GeoCalculator.bearing(c1, c2)
        assert bearing == pytest.approx(0.0, abs=1.0)

    def test_bearing_east(self):
        """Due east should be ~90 degrees."""
        c1 = Coordinate(latitude=0.0, longitude=0.0)
        c2 = Coordinate(latitude=0.0, longitude=1.0)
        bearing = GeoCalculator.bearing(c1, c2)
        assert bearing == pytest.approx(90.0, abs=1.0)

    def test_bearing_south(self):
        """Due south should be ~180 degrees."""
        c1 = Coordinate(latitude=1.0, longitude=0.0)
        c2 = Coordinate(latitude=0.0, longitude=0.0)
        bearing = GeoCalculator.bearing(c1, c2)
        assert bearing == pytest.approx(180.0, abs=1.0)

    def test_bearing_west(self):
        """Due west should be ~270 degrees."""
        c1 = Coordinate(latitude=0.0, longitude=1.0)
        c2 = Coordinate(latitude=0.0, longitude=0.0)
        bearing = GeoCalculator.bearing(c1, c2)
        assert bearing == pytest.approx(270.0, abs=1.0)

    def test_bearing_range(self):
        c1 = Coordinate(latitude=37.0, longitude=-122.0)
        c2 = Coordinate(latitude=38.0, longitude=-121.0)
        bearing = GeoCalculator.bearing(c1, c2)
        assert 0 <= bearing < 360

    def test_same_point_bearing(self):
        c = Coordinate(latitude=37.0, longitude=-122.0)
        bearing = GeoCalculator.bearing(c, c)
        assert 0 <= bearing < 360


# ---- Destination ----

class TestDestination:
    def test_destination_north(self):
        """Moving 111 km north should change latitude by ~1 degree."""
        c = Coordinate(latitude=0.0, longitude=0.0)
        dest = GeoCalculator.destination(c, 0.0, 111000)
        assert dest.latitude == pytest.approx(1.0, abs=0.01)
        assert dest.longitude == pytest.approx(0.0, abs=0.01)

    def test_destination_east_at_equator(self):
        c = Coordinate(latitude=0.0, longitude=0.0)
        dest = GeoCalculator.destination(c, 90.0, 111320)
        assert dest.latitude == pytest.approx(0.0, abs=0.01)
        assert dest.longitude == pytest.approx(1.0, abs=0.01)

    def test_destination_zero_distance(self):
        c = Coordinate(latitude=37.0, longitude=-122.0)
        dest = GeoCalculator.destination(c, 45.0, 0.0)
        assert dest.latitude == pytest.approx(c.latitude, abs=1e-10)
        assert dest.longitude == pytest.approx(c.longitude, abs=1e-10)

    def test_destination_round_trip(self):
        """Destination should be consistent with haversine + bearing."""
        c1 = Coordinate(latitude=37.7749, longitude=-122.4194)
        brg = GeoCalculator.bearing(c1, Coordinate(latitude=34.0522, longitude=-118.2437))
        dist = GeoCalculator.haversine_distance(c1, Coordinate(latitude=34.0522, longitude=-118.2437))
        dest = GeoCalculator.destination(c1, brg, dist)
        assert dest.latitude == pytest.approx(34.0522, abs=0.1)
        assert dest.longitude == pytest.approx(-118.2437, abs=0.1)


# ---- Midpoint ----

class TestMidpoint:
    def test_midpoint_equator(self):
        c1 = Coordinate(latitude=0.0, longitude=0.0)
        c2 = Coordinate(latitude=0.0, longitude=10.0)
        mid = GeoCalculator.midpoint(c1, c2)
        assert mid.latitude == pytest.approx(0.0, abs=0.01)
        assert mid.longitude == pytest.approx(5.0, abs=0.01)

    def test_midpoint_symmetry(self):
        c1 = Coordinate(latitude=37.0, longitude=-122.0)
        c2 = Coordinate(latitude=40.0, longitude=-74.0)
        mid1 = GeoCalculator.midpoint(c1, c2)
        mid2 = GeoCalculator.midpoint(c2, c1)
        assert mid1.latitude == pytest.approx(mid2.latitude, abs=1e-10)
        assert mid1.longitude == pytest.approx(mid2.longitude, abs=1e-10)

    def test_midpoint_same_point(self):
        c = Coordinate(latitude=37.0, longitude=-122.0)
        mid = GeoCalculator.midpoint(c, c)
        assert mid.latitude == pytest.approx(c.latitude, abs=1e-10)
        assert mid.longitude == pytest.approx(c.longitude, abs=1e-10)


# ---- Is In Polygon ----

class TestIsInPolygon:
    def test_point_inside_triangle(self):
        polygon = [
            Coordinate(latitude=0.0, longitude=0.0),
            Coordinate(latitude=0.0, longitude=1.0),
            Coordinate(latitude=1.0, longitude=0.0),
        ]
        point = Coordinate(latitude=0.2, longitude=0.2)
        assert GeoCalculator.is_in_polygon(point, polygon) is True

    def test_point_outside_triangle(self):
        polygon = [
            Coordinate(latitude=0.0, longitude=0.0),
            Coordinate(latitude=0.0, longitude=1.0),
            Coordinate(latitude=1.0, longitude=0.0),
        ]
        point = Coordinate(latitude=2.0, longitude=2.0)
        assert GeoCalculator.is_in_polygon(point, polygon) is False

    def test_point_on_vertex(self):
        polygon = [
            Coordinate(latitude=0.0, longitude=0.0),
            Coordinate(latitude=0.0, longitude=1.0),
            Coordinate(latitude=1.0, longitude=0.0),
        ]
        point = Coordinate(latitude=0.0, longitude=0.0)
        # On vertex behavior depends on ray casting implementation
        result = GeoCalculator.is_in_polygon(point, polygon)
        assert isinstance(result, bool)

    def test_insufficient_polygon_vertices(self):
        polygon = [
            Coordinate(latitude=0.0, longitude=0.0),
            Coordinate(latitude=1.0, longitude=0.0),
        ]
        point = Coordinate(latitude=0.5, longitude=0.0)
        assert GeoCalculator.is_in_polygon(point, polygon) is False

    def test_point_inside_square(self):
        polygon = [
            Coordinate(latitude=0.0, longitude=0.0),
            Coordinate(latitude=0.0, longitude=2.0),
            Coordinate(latitude=2.0, longitude=2.0),
            Coordinate(latitude=2.0, longitude=0.0),
        ]
        point = Coordinate(latitude=1.0, longitude=1.0)
        assert GeoCalculator.is_in_polygon(point, polygon) is True

    def test_point_outside_square(self):
        polygon = [
            Coordinate(latitude=0.0, longitude=0.0),
            Coordinate(latitude=0.0, longitude=2.0),
            Coordinate(latitude=2.0, longitude=2.0),
            Coordinate(latitude=2.0, longitude=0.0),
        ]
        point = Coordinate(latitude=3.0, longitude=3.0)
        assert GeoCalculator.is_in_polygon(point, polygon) is False

    def test_empty_polygon(self):
        point = Coordinate(latitude=0.0, longitude=0.0)
        assert GeoCalculator.is_in_polygon(point, []) is False

    def test_concave_polygon(self):
        polygon = [
            Coordinate(latitude=0.0, longitude=0.0),
            Coordinate(latitude=0.0, longitude=3.0),
            Coordinate(latitude=1.0, longitude=1.0),
            Coordinate(latitude=3.0, longitude=3.0),
            Coordinate(latitude=3.0, longitude=0.0),
        ]
        point = Coordinate(latitude=0.5, longitude=0.5)
        assert GeoCalculator.is_in_polygon(point, polygon) is True


# ---- Compute Speed ----

class TestComputeSpeed:
    def test_zero_speed(self):
        positions = [Coordinate(latitude=0.0, longitude=0.0)] * 3
        timestamps = [0.0, 1.0, 2.0]
        assert GeoCalculator.compute_speed(positions, timestamps) == pytest.approx(0.0, abs=0.01)

    def test_known_speed(self):
        """1 degree latitude in 1111 seconds -> ~100 m/s at 111 km."""
        positions = [
            Coordinate(latitude=0.0, longitude=0.0),
            Coordinate(latitude=1.0, longitude=0.0),
        ]
        timestamps = [0.0, 1111.0]
        speed = GeoCalculator.compute_speed(positions, timestamps)
        assert 90 < speed < 110  # m/s

    def test_insufficient_positions(self):
        positions = [Coordinate(latitude=0.0, longitude=0.0)]
        timestamps = [0.0]
        assert GeoCalculator.compute_speed(positions, timestamps) == 0.0

    def test_empty_positions(self):
        assert GeoCalculator.compute_speed([], []) == 0.0

    def test_mismatched_lengths(self):
        positions = [Coordinate(latitude=0.0, longitude=0.0)]
        timestamps = [0.0, 1.0]
        assert GeoCalculator.compute_speed(positions, timestamps) == 0.0

    def test_zero_time_interval(self):
        positions = [
            Coordinate(latitude=0.0, longitude=0.0),
            Coordinate(latitude=1.0, longitude=0.0),
        ]
        timestamps = [0.0, 0.0]
        assert GeoCalculator.compute_speed(positions, timestamps) == 0.0

    def test_multiple_segments(self):
        positions = [
            Coordinate(latitude=0.0, longitude=0.0),
            Coordinate(latitude=0.5, longitude=0.0),
            Coordinate(latitude=1.0, longitude=0.0),
        ]
        timestamps = [0.0, 500.0, 1000.0]
        speed = GeoCalculator.compute_speed(positions, timestamps)
        assert speed > 0


# ---- Cross Track Distance ----

class TestCrossTrackDistance:
    def test_on_path(self):
        """Point on the path should have near-zero cross-track distance."""
        c1 = Coordinate(latitude=0.0, longitude=0.0)
        c2 = Coordinate(latitude=0.0, longitude=1.0)
        midpoint = Coordinate(latitude=0.0, longitude=0.5)
        dxt = GeoCalculator.cross_track_distance(c1, c2, midpoint)
        assert abs(dxt) < 100  # small error

    def test_right_of_path(self):
        c1 = Coordinate(latitude=0.0, longitude=0.0)
        c2 = Coordinate(latitude=0.0, longitude=1.0)
        point = Coordinate(latitude=-0.5, longitude=0.5)  # South = starboard (right)
        dxt = GeoCalculator.cross_track_distance(c1, c2, point)
        assert dxt > 0

    def test_left_of_path(self):
        c1 = Coordinate(latitude=0.0, longitude=0.0)
        c2 = Coordinate(latitude=0.0, longitude=1.0)
        point = Coordinate(latitude=0.5, longitude=0.5)  # North = port (left)
        dxt = GeoCalculator.cross_track_distance(c1, c2, point)
        assert dxt < 0


# ---- Along Track Distance ----

class TestAlongTrackDistance:
    def test_start_point(self):
        c1 = Coordinate(latitude=0.0, longitude=0.0)
        c2 = Coordinate(latitude=0.0, longitude=1.0)
        dat = GeoCalculator.along_track_distance(c1, c2, c1)
        assert abs(dat) < 100

    def test_midpoint_along_track(self):
        c1 = Coordinate(latitude=0.0, longitude=0.0)
        c2 = Coordinate(latitude=0.0, longitude=1.0)
        midpoint = Coordinate(latitude=0.0, longitude=0.5)
        total = GeoCalculator.haversine_distance(c1, c2)
        dat = GeoCalculator.along_track_distance(c1, c2, midpoint)
        assert abs(dat - total / 2) < 5000


# ---- Normalize ----

class TestNormalize:
    def test_normalize_longitude_positive(self):
        assert GeoCalculator.normalize_longitude(190.0) == pytest.approx(-170.0)

    def test_normalize_longitude_negative(self):
        assert GeoCalculator.normalize_longitude(-200.0) == pytest.approx(160.0)

    def test_normalize_longitude_in_range(self):
        assert GeoCalculator.normalize_longitude(45.0) == pytest.approx(45.0)

    def test_normalize_latitude_clamp(self):
        assert GeoCalculator.normalize_latitude(100.0) == 90.0

    def test_normalize_latitude_negative_clamp(self):
        assert GeoCalculator.normalize_latitude(-100.0) == -90.0

    def test_normalize_latitude_in_range(self):
        assert GeoCalculator.normalize_latitude(45.0) == 45.0


# ---- Earth Radius ----

class TestConstants:
    def test_earth_radius(self):
        assert GeoCalculator.EARTH_RADIUS == pytest.approx(6_371_000, abs=1)
