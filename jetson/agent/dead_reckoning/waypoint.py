"""NEXUS Dead Reckoning - Waypoint Navigator.

Sequential waypoint navigation with Haversine-based bearing and
distance calculations, arrival detection, and autopilot heading commands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .navigation import NavigationMath


class WaypointStatus(Enum):
    """Status of waypoint navigation."""

    IDLE = auto()
    NAVIGATING = auto()
    ARRIVED = auto()
    ALL_COMPLETE = auto()
    OFF_COURSE = auto()


@dataclass
class Waypoint:
    """A single navigation waypoint."""

    latitude: float = 0.0
    longitude: float = 0.0
    arrival_radius_m: float = 20.0  # default arrival radius
    name: str = ""

    @property
    def is_valid(self) -> bool:
        return -90 <= self.latitude <= 90 and -180 <= self.longitude <= 180


@dataclass
class WaypointProgress:
    """Progress information for the current waypoint leg."""

    current_waypoint_idx: int = 0
    total_waypoints: int = 0
    distance_to_waypoint_m: float = 0.0
    bearing_to_waypoint_deg: float = 0.0
    cross_track_error_m: float = 0.0
    total_distance_m: float = 0.0
    distance_traveled_m: float = 0.0
    percent_complete: float = 0.0


@dataclass
class HeadingCommand:
    """Autopilot heading command."""

    desired_heading_deg: float = 0.0
    desired_speed_ms: float = 0.0
    turn_direction: str = ""  # "port", "starboard", ""
    heading_error_deg: float = 0.0
    is_arrival: bool = False


class WaypointNavigator:
    """Sequential waypoint navigation system.

    Accepts a list of waypoints and generates heading commands
    for the autopilot. Detects waypoint arrival and automatically
    progresses to the next waypoint.

    Features:
    - Haversine distance and bearing to next waypoint
    - Arrival detection within configurable radius
    - Cross-track error calculation
    - Heading command generation for autopilot
    - Sequential waypoint progression

    Usage:
        nav = WaypointNavigator()
        nav.set_waypoints([
            Waypoint(32.5, -117.1, name="WP1"),
            Waypoint(32.6, -117.0, name="WP2"),
        ])
        command = nav.update(lat=32.48, lon=-117.12, heading=45.0, speed=5.0)
        if nav.status == WaypointStatus.ARRIVED:
            nav.advance()
    """

    def __init__(
        self,
        default_arrival_radius_m: float = 20.0,
        off_course_threshold_m: float = 100.0,
        max_speed_ms: float = 10.0,
    ) -> None:
        self.default_arrival_radius = default_arrival_radius_m
        self.off_course_threshold = off_course_threshold_m
        self.max_speed = max_speed_ms

        self._waypoints: list[Waypoint] = []
        self._current_idx: int = 0
        self._status: WaypointStatus = WaypointStatus.IDLE
        self._last_position: tuple[float, float] = (0.0, 0.0)
        self._distance_traveled_m: float = 0.0
        self._total_distance_m: float = 0.0
        self._arrival_count: int = 0
        self._completed: bool = False

    @property
    def status(self) -> WaypointStatus:
        return self._status

    @property
    def current_waypoint(self) -> Optional[Waypoint]:
        if 0 <= self._current_idx < len(self._waypoints):
            return self._waypoints[self._current_idx]
        return None

    @property
    def current_index(self) -> int:
        return self._current_idx

    @property
    def total_waypoints(self) -> int:
        return len(self._waypoints)

    @property
    def is_complete(self) -> bool:
        return self._completed

    @property
    def distance_traveled(self) -> float:
        return self._distance_traveled_m

    @property
    def arrival_count(self) -> int:
        return self._arrival_count

    def set_waypoints(self, waypoints: list[Waypoint]) -> None:
        """Set the waypoint list and begin navigation.

        Args:
            waypoints: List of Waypoint objects to navigate.
        """
        if not waypoints:
            self._waypoints = []
            self._status = WaypointStatus.IDLE
            self._completed = True
            return

        self._waypoints = list(waypoints)
        # Apply default arrival radius to waypoints without one
        for wp in self._waypoints:
            if wp.arrival_radius_m <= 0:
                wp.arrival_radius_m = self.default_arrival_radius

        self._current_idx = 0
        self._status = WaypointStatus.NAVIGATING
        self._distance_traveled_m = 0.0
        self._completed = False
        self._arrival_count = 0

        # Calculate total route distance
        self._total_distance_m = 0.0
        for i in range(len(self._waypoints) - 1):
            self._total_distance_m += NavigationMath.haversine_distance(
                self._waypoints[i].latitude, self._waypoints[i].longitude,
                self._waypoints[i + 1].latitude, self._waypoints[i + 1].longitude,
            )

    def update(
        self,
        lat: float,
        lon: float,
        heading: float = 0.0,
        speed: float = 0.0,
    ) -> HeadingCommand:
        """Update navigator with current position and get heading command.

        Args:
            lat: Current latitude in degrees.
            lon: Current longitude in degrees.
            heading: Current heading in degrees.
            speed: Current speed in m/s.

        Returns:
            HeadingCommand for the autopilot.
        """
        if not self._waypoints or self._completed:
            self._status = WaypointStatus.ALL_COMPLETE if self._completed else WaypointStatus.IDLE
            return HeadingCommand(is_arrival=False)

        wp = self._waypoints[self._current_idx]
        if wp is None:
            return HeadingCommand(is_arrival=False)

        # Distance and bearing to waypoint
        dist = NavigationMath.haversine_distance(lat, lon, wp.latitude, wp.longitude)
        bearing = NavigationMath.bearing(lat, lon, wp.latitude, wp.longitude)

        # Cross-track error
        if self._current_idx > 0:
            prev = self._waypoints[self._current_idx - 1]
            xte = NavigationMath.cross_track_distance(
                prev.latitude, prev.longitude,
                wp.latitude, wp.longitude,
                lat, lon,
            )
        else:
            xte = 0.0

        # Track distance traveled
        if self._last_position != (0.0, 0.0):
            seg_dist = NavigationMath.haversine_distance(
                self._last_position[0], self._last_position[1], lat, lon
            )
            self._distance_traveled_m += seg_dist

        self._last_position = (lat, lon)

        # Check arrival
        if dist <= wp.arrival_radius_m:
            self._status = WaypointStatus.ARRIVED
            self._arrival_count += 1
            return HeadingCommand(
                desired_heading_deg=heading,
                desired_speed_ms=0.0,
                heading_error_deg=0.0,
                is_arrival=True,
            )

        # Check off-course
        if abs(xte) > self.off_course_threshold:
            self._status = WaypointStatus.OFF_COURSE
        else:
            self._status = WaypointStatus.NAVIGATING

        # Compute heading command
        heading_error = NavigationMath.angular_difference(bearing, heading)
        turn_direction = ""
        if heading_error > 5:
            turn_direction = "starboard"
        elif heading_error < -5:
            turn_direction = "port"

        # Speed: slow down near waypoint
        desired_speed = min(self.max_speed, speed)
        if dist < wp.arrival_radius_m * 5:
            desired_speed = max(1.0, desired_speed * (dist / (wp.arrival_radius_m * 5)))

        self._status = WaypointStatus.NAVIGATING

        return HeadingCommand(
            desired_heading_deg=bearing,
            desired_speed_ms=desired_speed,
            turn_direction=turn_direction,
            heading_error_deg=heading_error,
            is_arrival=False,
        )

    def advance(self) -> bool:
        """Advance to the next waypoint.

        Returns:
            True if advanced, False if already at last waypoint.
        """
        if self._current_idx < len(self._waypoints) - 1:
            self._current_idx += 1
            self._status = WaypointStatus.NAVIGATING
            return True
        else:
            self._completed = True
            self._status = WaypointStatus.ALL_COMPLETE
            return False

    def skip_to(self, index: int) -> bool:
        """Skip to a specific waypoint by index.

        Args:
            index: Waypoint index to skip to.

        Returns:
            True if valid, False if index out of range.
        """
        if 0 <= index < len(self._waypoints):
            self._current_idx = index
            self._status = WaypointStatus.NAVIGATING
            return True
        return False

    def get_progress(self) -> WaypointProgress:
        """Return current navigation progress."""
        if not self._waypoints:
            return WaypointProgress()

        wp = self._waypoints[self._current_idx] if self._current_idx < len(self._waypoints) else None
        lat, lon = self._last_position

        if wp:
            dist = NavigationMath.haversine_distance(lat, lon, wp.latitude, wp.longitude)
            bearing = NavigationMath.bearing(lat, lon, wp.latitude, wp.longitude)
        else:
            dist = 0.0
            bearing = 0.0

        pct = (self._distance_traveled_m / self._total_distance_m * 100) if self._total_distance_m > 0 else 0.0

        return WaypointProgress(
            current_waypoint_idx=self._current_idx,
            total_waypoints=len(self._waypoints),
            distance_to_waypoint_m=dist,
            bearing_to_waypoint_deg=bearing,
            total_distance_m=self._total_distance_m,
            distance_traveled_m=self._distance_traveled_m,
            percent_complete=min(100.0, pct),
        )

    def reset(self) -> None:
        """Reset navigator to idle state."""
        self._waypoints = []
        self._current_idx = 0
        self._status = WaypointStatus.IDLE
        self._last_position = (0.0, 0.0)
        self._distance_traveled_m = 0.0
        self._total_distance_m = 0.0
        self._arrival_count = 0
        self._completed = False
