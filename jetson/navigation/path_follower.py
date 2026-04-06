"""Path following controllers for autonomous marine navigation.

Implements Pure Pursuit, Stanley, and LOS (Line-of-Sight) guidance
algorithms with cross-track error computation and speed adjustment.
"""

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, pi, radians, sin, sqrt
from typing import List, Optional, Tuple

from .geospatial import Coordinate, GeoCalculator


@dataclass
class CrossTrackError:
    """Cross-track error from the desired path."""
    magnitude: float    # Distance off path in meters
    direction: float    # Direction of error: +1 = right, -1 = left
    closest_point: Coordinate


class PathFollower:
    """Path following controllers for marine vehicles."""

    @staticmethod
    def compute_cross_track_error(
        position: Coordinate, path_start: Coordinate, path_end: Coordinate
    ) -> CrossTrackError:
        """Compute cross-track error from position relative to path segment.

        Returns a CrossTrackError with magnitude, direction, and closest point.
        """
        dxt = GeoCalculator.cross_track_distance(path_start, path_end, position)
        dat = GeoCalculator.along_track_distance(path_start, path_end, position)
        direction = 1.0 if dxt >= 0 else -1.0
        magnitude = abs(dxt)

        # Compute the closest point on the path segment
        total_path_dist = GeoCalculator.haversine_distance(path_start, path_end)
        if total_path_dist < 1e-6:
            closest_point = path_start
        else:
            fraction = max(0.0, min(1.0, dat / total_path_dist))
            lat = path_start.latitude + fraction * (path_end.latitude - path_start.latitude)
            lon = path_start.longitude + fraction * (path_end.longitude - path_start.longitude)
            closest_point = Coordinate(latitude=lat, longitude=lon)

        return CrossTrackError(
            magnitude=magnitude,
            direction=direction,
            closest_point=closest_point
        )

    @staticmethod
    def compute_along_track_distance(
        position: Coordinate, path_start: Coordinate, path_end: Coordinate
    ) -> float:
        """Compute how far along the path segment the position is.

        Returns distance in meters from path_start along the path.
        """
        return GeoCalculator.along_track_distance(path_start, path_end, position)

    @staticmethod
    def compute_desired_heading(
        position: Coordinate, target: Coordinate
    ) -> float:
        """Compute desired heading from position to target.

        Returns heading in degrees [0, 360).
        """
        return GeoCalculator.bearing(position, target)

    @staticmethod
    def pure_pursuit(
        position: Coordinate,
        path_points: List[Coordinate],
        lookahead: float
    ) -> float:
        """Pure pursuit guidance algorithm.

        Finds the lookahead point on the path and returns desired heading.

        Args:
            position: Current vessel position.
            path_points: List of path waypoints.
            lookahead: Lookahead distance in meters.

        Returns:
            Desired heading in degrees [0, 360).
        """
        if not path_points:
            return 0.0
        if len(path_points) == 1:
            return GeoCalculator.bearing(position, path_points[0])

        # Find the closest path segment
        best_dist = float('inf')
        closest_seg_idx = 0
        for i in range(len(path_points) - 1):
            cte = PathFollower.compute_cross_track_error(
                position, path_points[i], path_points[i + 1]
            )
            seg_dist = GeoCalculator.haversine_distance(position, cte.closest_point)
            if seg_dist < best_dist:
                best_dist = seg_dist
                closest_seg_idx = i

        # Find lookahead point on the closest segment
        seg_start = path_points[closest_seg_idx]
        seg_end = path_points[closest_seg_idx + 1]
        seg_dist = GeoCalculator.haversine_distance(seg_start, seg_end)

        if seg_dist < 1e-6:
            return GeoCalculator.bearing(position, seg_start)

        # Try to find a point at lookahead distance
        # Walk along path from closest segment
        accumulated = 0.0
        for i in range(closest_seg_idx, len(path_points) - 1):
            s = path_points[i]
            e = path_points[i + 1]
            d = GeoCalculator.haversine_distance(s, e)
            remaining = lookahead - accumulated
            if remaining <= d:
                frac = remaining / d if d > 0 else 0
                lat = s.latitude + frac * (e.latitude - s.latitude)
                lon = s.longitude + frac * (e.longitude - s.longitude)
                lookahead_point = Coordinate(latitude=lat, longitude=lon)
                return GeoCalculator.bearing(position, lookahead_point)
            accumulated += d

        # Lookahead exceeds path, aim for last point
        return GeoCalculator.bearing(position, path_points[-1])

    @staticmethod
    def stanley_method(
        position: Coordinate,
        heading: float,
        path_points: List[Coordinate],
        gain: float = 1.0
    ) -> float:
        """Stanley controller for path following.

        Combines heading error with cross-track error correction.

        Args:
            position: Current vessel position.
            heading: Current vessel heading in degrees.
            path_points: List of path waypoints.
            gain: Cross-track error gain parameter.

        Returns:
            Corrected heading in degrees [0, 360).
        """
        if len(path_points) < 2:
            if path_points:
                return GeoCalculator.bearing(position, path_points[0])
            return heading

        # Find nearest path segment
        best_dist = float('inf')
        closest_seg_idx = 0
        for i in range(len(path_points) - 1):
            cte = PathFollower.compute_cross_track_error(
                position, path_points[i], path_points[i + 1]
            )
            seg_dist = GeoCalculator.haversine_distance(position, cte.closest_point)
            if seg_dist < best_dist:
                best_dist = seg_dist
                closest_seg_idx = i

        # Heading error toward the next waypoint on path
        target_wp = path_points[min(closest_seg_idx + 1, len(path_points) - 1)]
        desired_heading = GeoCalculator.bearing(position, target_wp)
        heading_error = desired_heading - heading
        # Normalize to [-180, 180]
        heading_error = (heading_error + 180) % 360 - 180

        # Cross-track error correction
        cte = PathFollower.compute_cross_track_error(
            position,
            path_points[closest_seg_idx],
            path_points[closest_seg_idx + 1]
        )
        cte_correction = degrees(atan2(gain * cte.magnitude * cte.direction, 1.0))

        corrected = heading + heading_error + cte_correction
        return corrected % 360

    @staticmethod
    def los_guidance(
        position: Coordinate,
        path_points: List[Coordinate],
        lookahead_gain: float = 3.0
    ) -> float:
        """Line-of-Sight (LOS) guidance algorithm.

        Computes desired heading using a lookahead distance proportional
        to the distance to the path endpoint.

        Args:
            position: Current vessel position.
            path_points: List of path waypoints.
            lookahead_gain: Multiplier for LOS distance (Delta > 1).

        Returns:
            Desired heading in degrees [0, 360).
        """
        if not path_points:
            return 0.0
        if len(path_points) == 1:
            return GeoCalculator.bearing(position, path_points[0])

        # Find closest path segment and determine lookahead point
        total_remaining = 0.0
        for i in range(len(path_points) - 1):
            total_remaining += GeoCalculator.haversine_distance(
                path_points[i], path_points[i + 1]
            )

        lookahead_dist = lookahead_gain * max(total_remaining, 10.0)
        lookahead_dist = min(lookahead_dist, total_remaining)

        # Use pure pursuit with dynamic lookahead
        return PathFollower.pure_pursuit(position, path_points, lookahead_dist)

    @staticmethod
    def compute_speed_adjustment(cte: float, target_speed: float) -> float:
        """Adjust speed based on cross-track error.

        Reduces speed proportionally to cross-track error magnitude.

        Args:
            cte: Cross-track error magnitude in meters.
            target_speed: Desired speed in m/s.

        Returns:
            Adjusted speed in m/s (never negative).
        """
        # Speed reduction factor: full speed for CTE < 5m, linear decrease
        max_cte = 50.0  # Maximum CTE for which speed can be zero
        if cte >= max_cte:
            return 0.0
        if cte <= 5.0:
            return target_speed
        # Linear interpolation between full speed and zero
        factor = 1.0 - (cte - 5.0) / (max_cte - 5.0)
        return max(0.0, target_speed * factor)

    @staticmethod
    def compute_path_curvature(path_points: List[Coordinate]) -> float:
        """Compute average curvature of a path.

        Returns average turning rate in degrees per meter.
        """
        if len(path_points) < 3:
            return 0.0

        total_turn = 0.0
        total_dist = 0.0
        for i in range(1, len(path_points) - 1):
            b1 = GeoCalculator.bearing(path_points[i - 1], path_points[i])
            b2 = GeoCalculator.bearing(path_points[i], path_points[i + 1])
            turn = abs(b2 - b1)
            if turn > 180:
                turn = 360 - turn
            d = GeoCalculator.haversine_distance(path_points[i - 1], path_points[i + 1])
            total_turn += turn
            total_dist += d

        if total_dist < 1e-6:
            return 0.0
        return total_turn / total_dist
