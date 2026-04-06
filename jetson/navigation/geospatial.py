"""Geographic calculations for marine navigation.

Pure Python implementation of haversine distance, bearing calculations,
destination point computation, and polygon containment checks.
"""

from dataclasses import dataclass
from math import (
    acos, asin, atan2, cos, degrees, hypot, pi, radians, sin, sqrt,
)


EARTH_RADIUS_METERS = 6_371_000.0  # Mean Earth radius in meters


@dataclass(frozen=True)
class Coordinate:
    """Geographic coordinate (latitude, longitude) in decimal degrees."""
    latitude: float
    longitude: float


class GeoCalculator:
    """Geographic computation utilities for marine navigation."""

    EARTH_RADIUS = EARTH_RADIUS_METERS

    @staticmethod
    def haversine_distance(c1: Coordinate, c2: Coordinate) -> float:
        """Compute great-circle distance between two coordinates in meters.

        Uses the haversine formula for spherical earth approximation.
        """
        lat1, lon1 = radians(c1.latitude), radians(c1.longitude)
        lat2, lon2 = radians(c2.latitude), radians(c2.longitude)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        return EARTH_RADIUS_METERS * c

    @staticmethod
    def bearing(c1: Coordinate, c2: Coordinate) -> float:
        """Compute initial bearing from c1 to c2 in degrees (0-360).

        Returns the bearing in degrees from true north, clockwise.
        """
        lat1, lon1 = radians(c1.latitude), radians(c1.longitude)
        lat2, lon2 = radians(c2.latitude), radians(c2.longitude)
        dlon = lon2 - lon1
        x = sin(dlon) * cos(lat2)
        y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
        bearing_rad = atan2(x, y)
        bearing_deg = degrees(bearing_rad)
        return (bearing_deg + 360) % 360

    @staticmethod
    def destination(c: Coordinate, bearing_deg: float, distance_m: float) -> Coordinate:
        """Compute destination point given start, bearing, and distance.

        Args:
            c: Starting coordinate.
            bearing_deg: Bearing in degrees from true north.
            distance_m: Distance in meters.

        Returns:
            The destination Coordinate.
        """
        lat1 = radians(c.latitude)
        lon1 = radians(c.longitude)
        bearing = radians(bearing_deg)
        angular_dist = distance_m / EARTH_RADIUS_METERS
        lat2 = asin(
            sin(lat1) * cos(angular_dist) +
            cos(lat1) * sin(angular_dist) * cos(bearing)
        )
        lon2 = lon1 + atan2(
            sin(bearing) * sin(angular_dist) * cos(lat1),
            cos(angular_dist) - sin(lat1) * sin(lat2)
        )
        return Coordinate(latitude=degrees(lat2), longitude=degrees(lon2))

    @staticmethod
    def midpoint(c1: Coordinate, c2: Coordinate) -> Coordinate:
        """Compute the midpoint between two coordinates.

        Returns a Coordinate at the great-circle midpoint.
        """
        lat1, lon1 = radians(c1.latitude), radians(c1.longitude)
        lat2, lon2 = radians(c2.latitude), radians(c2.longitude)
        dlon = lon2 - lon1
        bx = cos(lat2) * cos(dlon)
        by = cos(lat2) * sin(dlon)
        lat_m = atan2(
            sin(lat1) + sin(lat2),
            sqrt((cos(lat1) + bx) ** 2 + by ** 2)
        )
        lon_m = lon1 + atan2(by, cos(lat1) + bx)
        return Coordinate(latitude=degrees(lat_m), longitude=degrees(lon_m))

    @staticmethod
    def is_in_polygon(point: Coordinate, polygon: list) -> bool:
        """Check if a point is inside a polygon using ray-casting algorithm.

        Args:
            point: The coordinate to test.
            polygon: List of Coordinate objects forming a closed polygon.

        Returns:
            True if the point is inside the polygon.
        """
        n = len(polygon)
        if n < 3:
            return False
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i].longitude, polygon[i].latitude
            xj, yj = polygon[j].longitude, polygon[j].latitude
            xp, yp = point.longitude, point.latitude
            if ((yi > yp) != (yj > yp)) and \
               (xp < (xj - xi) * (yp - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    @staticmethod
    def compute_speed(positions: list, timestamps: list) -> float:
        """Compute average speed from a series of positions and timestamps.

        Args:
            positions: List of Coordinate objects.
            timestamps: List of timestamps in seconds (monotonically increasing).

        Returns:
            Average speed in m/s, or 0.0 if insufficient data.
        """
        if len(positions) < 2 or len(timestamps) < 2:
            return 0.0
        if len(positions) != len(timestamps):
            return 0.0
        total_dist = 0.0
        total_time = 0.0
        for i in range(1, len(positions)):
            total_dist += GeoCalculator.haversine_distance(positions[i - 1], positions[i])
            total_time += timestamps[i] - timestamps[i - 1]
        if total_time <= 0:
            return 0.0
        return total_dist / total_time

    @staticmethod
    def cross_track_distance(
        start: Coordinate, end: Coordinate, point: Coordinate
    ) -> float:
        """Compute cross-track distance from point to great-circle arc.

        Positive = right (starboard) of the path direction, negative = left (port).
        Returns distance in meters.
        """
        d13 = GeoCalculator.haversine_distance(start, point) / EARTH_RADIUS_METERS
        bearing_13 = radians(GeoCalculator.bearing(start, point))
        bearing_12 = radians(GeoCalculator.bearing(start, end))
        return asin(sin(d13) * sin(bearing_13 - bearing_12)) * EARTH_RADIUS_METERS

    @staticmethod
    def along_track_distance(
        start: Coordinate, end: Coordinate, point: Coordinate
    ) -> float:
        """Compute along-track distance from start to the closest point on the path.

        Returns distance in meters from start along the great-circle arc.
        """
        d13 = GeoCalculator.haversine_distance(start, point) / EARTH_RADIUS_METERS
        bearing_13 = radians(GeoCalculator.bearing(start, point))
        bearing_12 = radians(GeoCalculator.bearing(start, end))
        dxt = asin(sin(d13) * sin(bearing_13 - bearing_12))
        cos_dxt = cos(dxt)
        if abs(cos_dxt) < 1e-10:
            return 0.0
        val = cos(d13) / cos_dxt
        val = max(-1.0, min(1.0, val))
        dat = acos(val)
        return dat * EARTH_RADIUS_METERS

    @staticmethod
    def normalize_longitude(lon: float) -> float:
        """Normalize longitude to [-180, 180] range."""
        return (lon + 540) % 360 - 180

    @staticmethod
    def normalize_latitude(lat: float) -> float:
        """Clamp latitude to [-90, 90] range."""
        return max(-90.0, min(90.0, lat))
