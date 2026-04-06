"""Waypoint management for autonomous marine navigation.

Provides waypoint storage, sequencing, interpolation, and optimization
using 2-opt TSP algorithm for route planning.
"""

from dataclasses import dataclass, field
import random
from typing import List, Optional, Tuple

from .geospatial import Coordinate, GeoCalculator


@dataclass
class Waypoint:
    """A navigation waypoint with position and motion parameters."""
    id: str
    latitude: float
    longitude: float
    speed: float = 1.5          # Target speed in m/s
    heading: Optional[float] = None  # Target heading (degrees), None = auto
    acceptance_radius: float = 10.0  # Radius in meters to consider reached

    def to_coordinate(self) -> Coordinate:
        return Coordinate(latitude=self.latitude, longitude=self.longitude)


class WaypointManager:
    """Manages waypoint sequences for mission planning."""

    def __init__(self):
        self._waypoints: List[Waypoint] = []
        self._next_id_counter: int = 0

    def add_waypoint(self, waypoint: Waypoint) -> None:
        """Add a waypoint to the end of the sequence."""
        self._waypoints.append(waypoint)

    def insert_waypoint(self, waypoint: Waypoint, after_id: str) -> bool:
        """Insert a waypoint after the waypoint with the given ID.

        Returns True if insertion succeeded, False if after_id not found.
        """
        for i, wp in enumerate(self._waypoints):
            if wp.id == after_id:
                self._waypoints.insert(i + 1, waypoint)
                return True
        return False

    def remove_waypoint(self, waypoint_id: str) -> bool:
        """Remove a waypoint by ID.

        Returns True if removed, False if not found.
        """
        for i, wp in enumerate(self._waypoints):
            if wp.id == waypoint_id:
                self._waypoints.pop(i)
                return True
        return False

    def get_waypoint(self, waypoint_id: str) -> Optional[Waypoint]:
        """Retrieve a waypoint by ID."""
        for wp in self._waypoints:
            if wp.id == waypoint_id:
                return wp
        return None

    def get_all_waypoints(self) -> List[Waypoint]:
        """Return a copy of all waypoints."""
        return list(self._waypoints)

    def clear(self) -> None:
        """Remove all waypoints."""
        self._waypoints.clear()

    def count(self) -> int:
        """Return the number of waypoints."""
        return len(self._waypoints)

    @staticmethod
    def compute_interpolated(
        wp1: Waypoint, wp2: Waypoint, fraction: float
    ) -> Coordinate:
        """Interpolate between two waypoints at a given fraction [0, 1].

        Returns the interpolated Coordinate.
        """
        fraction = max(0.0, min(1.0, fraction))
        c1 = wp1.to_coordinate()
        c2 = wp2.to_coordinate()
        lat = c1.latitude + fraction * (c2.latitude - c1.latitude)
        lon = c1.longitude + fraction * (c2.longitude - c1.longitude)
        return Coordinate(latitude=lat, longitude=lon)

    @staticmethod
    def compute_total_distance(waypoints: List[Waypoint]) -> float:
        """Compute total path distance through waypoints using haversine.

        Returns distance in meters.
        """
        if len(waypoints) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(waypoints)):
            c1 = waypoints[i - 1].to_coordinate()
            c2 = waypoints[i].to_coordinate()
            total += GeoCalculator.haversine_distance(c1, c2)
        return total

    @staticmethod
    def get_current_target(
        position: Coordinate, waypoints: List[Waypoint]
    ) -> Optional[Waypoint]:
        """Get the next unreached waypoint target.

        Returns the first waypoint not yet reached based on acceptance radius.
        Returns None if all waypoints are reached or list is empty.
        """
        for wp in waypoints:
            coord = wp.to_coordinate()
            dist = GeoCalculator.haversine_distance(position, coord)
            if dist > wp.acceptance_radius:
                return wp
        return None

    @staticmethod
    def is_waypoint_reached(
        position: Coordinate, waypoint: Waypoint
    ) -> bool:
        """Check if a position is within the waypoint's acceptance radius."""
        coord = waypoint.to_coordinate()
        dist = GeoCalculator.haversine_distance(position, coord)
        return dist <= waypoint.acceptance_radius

    @staticmethod
    def optimize_sequence(waypoints: List[Waypoint]) -> List[Waypoint]:
        """Optimize waypoint visit order using 2-opt TSP algorithm.

        Minimizes total travel distance by iteratively improving the route.
        Uses the first waypoint as a fixed starting point.
        Returns a new list of waypoints in optimized order.
        """
        if len(waypoints) <= 2:
            return list(waypoints)

        n = len(waypoints)
        route = list(range(n))
        # Fix the first waypoint as start
        best_distance = WaypointManager._route_distance(route, waypoints)
        improved = True
        iterations = 0
        max_iterations = n * n * 2

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    new_distance = WaypointManager._route_distance(new_route, waypoints)
                    if new_distance < best_distance:
                        route = new_route
                        best_distance = new_distance
                        improved = True

        return [waypoints[i] for i in route]

    @staticmethod
    def _route_distance(route: List[int], waypoints: List[Waypoint]) -> float:
        """Compute total distance for a route (list of indices)."""
        dist = 0.0
        for i in range(len(route) - 1):
            c1 = waypoints[route[i]].to_coordinate()
            c2 = waypoints[route[i + 1]].to_coordinate()
            dist += GeoCalculator.haversine_distance(c1, c2)
        return dist

    def reindex(self) -> None:
        """Reassign IDs to all waypoints sequentially."""
        for i, wp in enumerate(self._waypoints):
            wp.id = f"wp_{i}"

    def segment_distance(self, index: int) -> float:
        """Compute distance from waypoint at index to the next one.

        Returns 0.0 if at the last waypoint or index out of bounds.
        """
        if index < 0 or index >= len(self._waypoints) - 1:
            return 0.0
        c1 = self._waypoints[index].to_coordinate()
        c2 = self._waypoints[index + 1].to_coordinate()
        return GeoCalculator.haversine_distance(c1, c2)

    def remaining_distance(self, position: Coordinate) -> float:
        """Compute remaining distance from position through all unreached waypoints."""
        remaining = []
        for wp in self._waypoints:
            if not self.is_waypoint_reached(position, wp):
                remaining.append(wp)
        if not remaining:
            return 0.0
        dist_to_first = GeoCalculator.haversine_distance(position, remaining[0].to_coordinate())
        return dist_to_first + self.compute_total_distance(remaining)
