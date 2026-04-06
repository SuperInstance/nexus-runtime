"""
Maritime Zone Management Module.

Manages maritime zones (territorial waters, ports, anchorages, traffic
separation schemes, exclusion zones, caution areas, pilotage areas).
Checks vessel permissions, computes optimal routes, and detects violations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ZoneType(Enum):
    """Types of maritime zones."""
    TERRITORIAL = "territorial"
    PORT = "port"
    ANCHORAGE = "anchorage"
    TRAFFIC_SEPARATION = "traffic_separation"
    EXCLUSION = "exclusion"
    CAUTION = "caution"
    PILOTAGE = "pilotage"


@dataclass
class MaritimeZone:
    """Represents a maritime zone with boundary and restrictions."""
    name: str
    zone_type: ZoneType
    boundary: List[Tuple[float, float]]  # polygon vertices [(lat, lon), ...]
    restrictions: List[str] = field(default_factory=list)
    max_speed: float = float('inf')      # knots
    entry_requirements: List[str] = field(default_factory=list)
    active: bool = True


@dataclass
class ZoneViolation:
    """Represents a zone violation event."""
    zone_name: str
    zone_type: ZoneType
    violation_type: str  # 'entry', 'speed', 'requirements'
    position: Tuple[float, float]
    severity: str  # 'low', 'medium', 'high', 'critical'


@dataclass
class VesselInfo:
    """Information about a vessel for zone checking."""
    mmsi: int
    vessel_type: str = "cargo"
    speed: float = 0.0
    length: float = 100.0
    has_pilot: bool = False
    has_clearance: bool = False
    position: Tuple[float, float] = (0.0, 0.0)


@dataclass
class RoutePoint:
    """A point on a computed route."""
    lat: float
    lon: float
    heading: float = 0.0
    in_zone: Optional[str] = None


@dataclass
class Route:
    """A computed maritime route."""
    waypoints: List[RoutePoint]
    total_distance: float = 0.0
    estimated_time: float = 0.0
    avoids_zones: List[str] = field(default_factory=list)


class ZoneManager:
    """Manages maritime zones and vessel zone compliance."""

    def __init__(self) -> None:
        self._zones: Dict[str, MaritimeZone] = {}

    def add_zone(self, zone: MaritimeZone) -> None:
        """Register a new maritime zone."""
        self._zones[zone.name] = zone

    def remove_zone(self, name: str) -> bool:
        """Remove a zone by name. Returns True if found and removed."""
        if name in self._zones:
            del self._zones[name]
            return True
        return False

    def get_zone(self, name: str) -> Optional[MaritimeZone]:
        """Get a zone by name."""
        return self._zones.get(name)

    def list_zones(self) -> List[MaritimeZone]:
        """List all registered zones."""
        return list(self._zones.values())

    def check_position(
        self, position: Tuple[float, float]
    ) -> List[MaritimeZone]:
        """
        Check which zones overlap with a given position.
        Returns list of zones the position falls within.
        """
        overlapping = []
        for zone in self._zones.values():
            if not zone.active:
                continue
            if self._point_in_polygon(position, zone.boundary):
                overlapping.append(zone)
        return overlapping

    def check_entry_permission(
        self, vessel: VesselInfo, zone: MaritimeZone
    ) -> Tuple[bool, str]:
        """
        Check if a vessel has permission to enter a zone.
        Returns (allowed: bool, reason: str).
        """
        if not zone.active:
            return (True, "Zone inactive.")

        # Exclusion zones are always restricted
        if zone.zone_type == ZoneType.EXCLUSION:
            return (False, f"Entry prohibited: {zone.name} is an exclusion zone.")

        # Check clearance requirements
        if "clearance_required" in zone.restrictions and not vessel.has_clearance:
            return (False, f"Clearance required for {zone.name}.")

        # Check pilotage requirements
        if zone.zone_type == ZoneType.PILOTAGE and not vessel.has_pilot:
            return (False, f"Pilot required for {zone.name}.")

        # Check vessel type restrictions
        allowed_types = [r.split(":")[1] for r in zone.restrictions if r.startswith("allowed_types:")]
        if allowed_types and vessel.vessel_type not in allowed_types:
            return (False, f"Vessel type '{vessel.vessel_type}' not permitted in {zone.name}.")

        # Check speed limit
        if vessel.speed > zone.max_speed:
            return (True, f"Speed warning: {vessel.speed}kts exceeds limit {zone.max_speed}kts in {zone.name}.")

        return (True, f"Entry permitted for {zone.name}.")

    def compute_entry_requirements(
        self,
        position: Tuple[float, float],
        vessel_type: str,
    ) -> List[str]:
        """
        Compute entry requirements for all zones at a given position.
        Returns list of requirement descriptions.
        """
        zones = self.check_position(position)
        requirements = []

        for zone in zones:
            if zone.zone_type == ZoneType.PILOTAGE:
                requirements.append(f"Pilot required in {zone.name}")
            if "clearance_required" in zone.restrictions:
                requirements.append(f"Port clearance required for {zone.name}")
            if zone.max_speed < float('inf'):
                requirements.append(f"Speed limit {zone.max_speed}kts in {zone.name}")
            for req in zone.entry_requirements:
                requirements.append(f"{req} (zone: {zone.name})")

        return requirements

    def compute_optimal_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        zones: Optional[List[MaritimeZone]] = None,
    ) -> Route:
        """
        Compute a route from start to end that avoids restricted zones.
        Uses waypoint-based pathfinding with zone avoidance.
        """
        if zones is None:
            zones = list(self._zones.values())

        restricted_zones = [z for z in zones if z.zone_type == ZoneType.EXCLUSION and z.active]

        # Generate waypoints along direct path
        waypoints = self._generate_waypoints(start, end)

        # Check each waypoint and deflect around restricted zones
        final_waypoints = []
        avoids = []
        for wp_lat, wp_lon in waypoints:
            wp_pos = (wp_lat, wp_lon)
            in_restricted = False
            for zone in restricted_zones:
                if self._point_in_polygon(wp_pos, zone.boundary):
                    in_restricted = True
                    if zone.name not in avoids:
                        avoids.append(zone.name)
                    # Deflect waypoint outside the zone
                    deflected = self._deflect_around_zone(wp_pos, zone)
                    final_waypoints.append(RoutePoint(
                        lat=deflected[0],
                        lon=deflected[1],
                        heading=0.0,
                        in_zone=None,
                    ))
                    break
            if not in_restricted:
                # Check which non-restricted zones this point is in
                zone_name = None
                for zone in zones:
                    if zone.active and self._point_in_polygon(wp_pos, zone.boundary):
                        zone_name = zone.name
                        break
                final_waypoints.append(RoutePoint(
                    lat=wp_lat,
                    lon=wp_lon,
                    heading=0.0,
                    in_zone=zone_name,
                ))

        # Compute heading for each waypoint
        for i, wp in enumerate(final_waypoints):
            if i < len(final_waypoints) - 1:
                next_wp = final_waypoints[i + 1]
                wp.heading = self._compute_bearing(
                    (wp.lat, wp.lon), (next_wp.lat, next_wp.lon)
                )

        # Compute total distance
        total_dist = 0.0
        for i in range(len(final_waypoints) - 1):
            p1 = (final_waypoints[i].lat, final_waypoints[i].lon)
            p2 = (final_waypoints[i + 1].lat, final_waypoints[i + 1].lon)
            total_dist += self._haversine_distance(p1, p2)

        return Route(
            waypoints=final_waypoints,
            total_distance=round(total_dist, 2),
            estimated_time=round(total_dist / 12.0, 2),  # assume 12 knots
            avoids_zones=avoids,
        )

    def detect_zone_violation(
        self,
        vessel_position: Tuple[float, float],
        vessel_type: str = "cargo",
    ) -> List[ZoneViolation]:
        """
        Detect any zone violations for a vessel at a given position.
        Checks for entry into restricted zones and type violations.
        """
        violations = []
        zones = self.check_position(vessel_position)

        for zone in zones:
            # Check exclusion zone entry
            if zone.zone_type == ZoneType.EXCLUSION:
                violations.append(ZoneViolation(
                    zone_name=zone.name,
                    zone_type=zone.zone_type,
                    violation_type="entry",
                    position=vessel_position,
                    severity="critical",
                ))

            # Check vessel type restrictions
            allowed_types = [r.split(":")[1] for r in zone.restrictions if r.startswith("allowed_types:")]
            if allowed_types and vessel_type not in allowed_types:
                violations.append(ZoneViolation(
                    zone_name=zone.name,
                    zone_type=zone.zone_type,
                    violation_type="requirements",
                    position=vessel_position,
                    severity="high",
                ))

        return violations

    @staticmethod
    def _point_in_polygon(
        point: Tuple[float, float],
        polygon: List[Tuple[float, float]],
    ) -> bool:
        """Ray-casting algorithm for point-in-polygon test."""
        if len(polygon) < 3:
            return False

        x, y = point
        n = len(polygon)
        inside = False
        j = n - 1

        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            if ((yi > y) != (yj > y)) and \
               (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i

        return inside

    @staticmethod
    def _deflect_around_zone(
        point: Tuple[float, float],
        zone: MaritimeZone,
    ) -> Tuple[float, float]:
        """Deflect a point to the nearest edge of a zone's boundary."""
        if not zone.boundary:
            return point

        min_dist = float('inf')
        best_point = point

        for i in range(len(zone.boundary)):
            p1 = zone.boundary[i]
            p2 = zone.boundary[(i + 1) % len(zone.boundary)]
            edge_mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            dist = math.sqrt((point[0] - edge_mid[0]) ** 2 + (point[1] - edge_mid[1]) ** 2)

            if dist < min_dist:
                min_dist = dist
                # Move point slightly outside the edge
                offset = 0.005  # ~0.3nm
                dx = point[0] - edge_mid[0]
                dy = point[1] - edge_mid[1]
                length = math.sqrt(dx * dx + dy * dy)
                if length > 0:
                    best_point = (
                        edge_mid[0] + (dx / length) * offset,
                        edge_mid[1] + (dy / length) * offset,
                    )
                else:
                    best_point = (p2[0] + offset, p2[1] + offset)

        return best_point

    @staticmethod
    def _compute_bearing(
        start: Tuple[float, float],
        end: Tuple[float, float],
    ) -> float:
        """Compute initial bearing from start to end in degrees."""
        lat1, lon1 = math.radians(start[0]), math.radians(start[1])
        lat2, lon2 = math.radians(end[0]), math.radians(end[1])

        dlon = lon2 - lon1
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        bearing = math.atan2(x, y)
        return math.degrees(bearing) % 360

    @staticmethod
    def _haversine_distance(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> float:
        """Compute distance between two points in nautical miles."""
        R = 3440.065  # Earth radius in nm
        lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
        lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    @staticmethod
    def _generate_waypoints(
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_points: int = 20,
    ) -> List[Tuple[float, float]]:
        """Generate equally spaced waypoints between start and end."""
        waypoints = []
        for i in range(num_points + 1):
            t = i / num_points
            lat = start[0] + t * (end[0] - start[0])
            lon = start[1] + t * (end[1] - start[1])
            waypoints.append((lat, lon))
        return waypoints
