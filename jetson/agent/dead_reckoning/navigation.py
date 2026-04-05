"""NEXUS Dead Reckoning - Navigation Math.

Haversine distance, bearing calculations, coordinate conversions,
and geodetic helpers for marine navigation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class NavigationMath:
    """Static utility methods for marine navigation calculations.

    All distance calculations use the WGS-84 Earth radius of 6,371,000 m.
    Angles are in degrees for public API, radians internally.
    """

    EARTH_RADIUS_M = 6_371_000.0  # mean Earth radius in meters

    @staticmethod
    def haversine_distance(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate great-circle distance in meters between two points.

        Args:
            lat1: Latitude of point 1 (degrees).
            lon1: Longitude of point 1 (degrees).
            lat2: Latitude of point 2 (degrees).
            lon2: Longitude of point 2 (degrees).

        Returns:
            Distance in meters.
        """
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return NavigationMath.EARTH_RADIUS_M * c

    @staticmethod
    def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate initial bearing from point 1 to point 2 (degrees, 0-360).

        Args:
            lat1: Latitude of start point (degrees).
            lon1: Longitude of start point (degrees).
            lat2: Latitude of end point (degrees).
            lon2: Longitude of end point (degrees).

        Returns:
            Bearing in degrees [0, 360).
        """
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dlambda = math.radians(lon2 - lon1)

        x = math.sin(dlambda) * math.cos(phi2)
        y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)

        theta = math.atan2(x, y)

        return (math.degrees(theta) + 360) % 360

    @staticmethod
    def destination_point(
        lat: float, lon: float, bearing_deg: float, distance_m: float
    ) -> tuple[float, float]:
        """Calculate destination point given start, bearing, and distance.

        Args:
            lat: Start latitude (degrees).
            lon: Start longitude (degrees).
            bearing_deg: Bearing in degrees.
            distance_m: Distance in meters.

        Returns:
            Tuple of (latitude, longitude) in degrees.
        """
        phi1 = math.radians(lat)
        lambda1 = math.radians(lon)
        theta = math.radians(bearing_deg)

        R = NavigationMath.EARTH_RADIUS_M
        phi2 = math.asin(
            math.sin(phi1) * math.cos(distance_m / R)
            + math.cos(phi1) * math.sin(distance_m / R) * math.cos(theta)
        )
        lambda2 = lambda1 + math.atan2(
            math.sin(theta) * math.sin(distance_m / R) * math.cos(phi1),
            math.cos(distance_m / R) - math.sin(phi1) * math.sin(phi2),
        )

        return math.degrees(phi2), math.degrees(lambda2)

    @staticmethod
    def cross_track_distance(
        lat1: float, lon1: float,
        lat2: float, lon2: float,
        lat3: float, lon3: float,
    ) -> float:
        """Calculate cross-track distance: distance from point 3 to
        the great-circle path defined by points 1 -> 2.

        Positive = point 3 is to the right of the path,
        negative = to the left.

        Returns:
            Distance in meters.
        """
        d13 = NavigationMath.haversine_distance(lat1, lon1, lat3, lon3)
        bearing_12 = math.radians(NavigationMath.bearing(lat1, lon1, lat2, lon2))
        bearing_13 = math.radians(NavigationMath.bearing(lat1, lon1, lat3, lon3))

        d_xt = math.asin(
            math.sin(d13 / NavigationMath.EARTH_RADIUS_M)
            * math.sin(bearing_13 - bearing_12)
        ) * NavigationMath.EARTH_RADIUS_M

        return d_xt

    @staticmethod
    def along_track_distance(
        lat1: float, lon1: float,
        lat2: float, lon2: float,
        lat3: float, lon3: float,
    ) -> float:
        """Calculate along-track distance: distance from point 1 to the
        closest point on the great-circle path from 1 to 2.

        Returns:
            Distance in meters.
        """
        d13 = NavigationMath.haversine_distance(lat1, lon1, lat3, lon3)
        d_xt = NavigationMath.cross_track_distance(lat1, lon1, lat2, lon2, lat3, lon3)

        d_at = math.acos(
            math.cos(d13 / NavigationMath.EARTH_RADIUS_M)
            / math.cos(d_xt / NavigationMath.EARTH_RADIUS_M)
        ) * NavigationMath.EARTH_RADIUS_M

        # Guard against numerical errors
        if math.isnan(d_at):
            return 0.0

        return d_at

    @staticmethod
    def meters_to_degrees_lat(meters: float) -> float:
        """Convert meters to degrees of latitude (approximate)."""
        return meters / 111_320.0

    @staticmethod
    def meters_to_degrees_lon(meters: float, latitude: float) -> float:
        """Convert meters to degrees of longitude at a given latitude."""
        return meters / (111_320.0 * math.cos(math.radians(latitude)))

    @staticmethod
    def degrees_lat_to_meters(degrees: float) -> float:
        """Convert degrees of latitude to meters (approximate)."""
        return degrees * 111_320.0

    @staticmethod
    def degrees_lon_to_meters(degrees: float, latitude: float) -> float:
        """Convert degrees of longitude to meters at a given latitude."""
        return degrees * 111_320.0 * math.cos(math.radians(latitude))

    @staticmethod
    def normalize_angle(angle_deg: float) -> float:
        """Normalize an angle to [0, 360) degrees."""
        return angle_deg % 360

    @staticmethod
    def angular_difference(a: float, b: float) -> float:
        """Smallest signed angle from b to a in degrees (-180, 180]."""
        diff = (a - b) % 360
        if diff > 180:
            diff -= 360
        return diff

    @staticmethod
    def midpoint(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
        """Calculate the midpoint between two points."""
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        lambda1 = math.radians(lon1)
        lambda2 = math.radians(lon2)

        bx = math.cos(phi2) * math.cos(lambda2 - lambda1)
        by = math.cos(phi2) * math.sin(lambda2 - lambda1)

        phi_m = math.atan2(
            math.sin(phi1) + math.sin(phi2),
            math.sqrt((math.cos(phi1) + bx) ** 2 + by ** 2),
        )
        lambda_m = lambda1 + math.atan2(by, math.cos(phi1) + bx)

        return math.degrees(phi_m), math.degrees(lambda_m)


@dataclass
class VelocityVector:
    """2D velocity vector with north/east components in m/s."""

    north: float = 0.0  # m/s northward
    east: float = 0.0   # m/s eastward

    @property
    def speed(self) -> float:
        """Scalar speed in m/s."""
        return math.sqrt(self.north ** 2 + self.east ** 2)

    @property
    def heading(self) -> float:
        """Heading in degrees [0, 360)."""
        if self.speed < 1e-9:
            return 0.0
        h = math.degrees(math.atan2(self.east, self.north))
        return h % 360

    @classmethod
    def from_speed_heading(cls, speed: float, heading_deg: float) -> VelocityVector:
        """Create from speed (m/s) and heading (degrees from north)."""
        h = math.radians(heading_deg)
        return cls(
            north=speed * math.cos(h),
            east=speed * math.sin(h),
        )
