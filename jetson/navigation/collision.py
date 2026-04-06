"""Collision avoidance system for autonomous marine navigation.

Implements COLREGS-inspired threat detection, TCPA/DCPA calculations,
risk scoring, evasive maneuver generation, and avoidance path planning.
"""

from dataclasses import dataclass, field
from math import acos, cos, degrees, hypot, pi, radians, sin, sqrt
from enum import Enum
from typing import List, Optional, Tuple

from .geospatial import Coordinate, GeoCalculator


class Severity(Enum):
    """Collision threat severity levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class VesselState:
    """State of a vessel for collision analysis."""
    position: Coordinate
    speed: float          # m/s
    heading: float        # degrees (0-360)
    vessel_id: str = ""


@dataclass
class CollisionThreat:
    """Detected collision threat."""
    vessel_id: str
    position: Coordinate
    velocity: Tuple[float, float]  # (east_mps, north_mps)
    distance: float         # meters
    tcpa: float             # seconds (time to closest point of approach)
    dcpa: float             # meters (distance at closest point of approach)
    severity: Severity


class CollisionAvoidance:
    """Collision avoidance system for marine vehicles."""

    def __init__(
        self,
        safe_distance: float = 100.0,
        warning_distance: float = 300.0,
        critical_distance: float = 50.0,
        max_tcpa: float = 600.0,  # 10 minutes
    ):
        self.safe_distance = safe_distance
        self.warning_distance = warning_distance
        self.critical_distance = critical_distance
        self.max_tcpa = max_tcpa

    def detect_threats(
        self, own_vessel: VesselState, others: List[VesselState]
    ) -> List[CollisionThreat]:
        """Detect collision threats from other vessels.

        Returns a list of detected threats sorted by severity (highest first).
        """
        threats = []
        for other in others:
            tcpa = self.compute_tcpa(own_vessel, other)
            dcpa = self.compute_dcpa(own_vessel, other)
            dist = GeoCalculator.haversine_distance(
                own_vessel.position, other.position
            )

            # Compute velocity components
            other_vel = self._velocity_components(other.speed, other.heading)

            risk_score = self.compute_risk_score(tcpa, dcpa, own_vessel.speed, other.speed)
            severity = self._risk_to_severity(risk_score, dist)

            if severity != Severity.NONE:
                threats.append(CollisionThreat(
                    vessel_id=other.vessel_id,
                    position=other.position,
                    velocity=other_vel,
                    distance=dist,
                    tcpa=tcpa,
                    dcpa=dcpa,
                    severity=severity,
                ))

        threats.sort(key=lambda t: t.severity.value, reverse=True)
        return threats

    @staticmethod
    def compute_tcpa(own: VesselState, other: VesselState) -> float:
        """Compute Time to Closest Point of Approach.

        Returns TCPA in seconds. Negative values mean vessels are separating.
        """
        # Relative velocity components
        own_vx, own_vy = CollisionAvoidance._velocity_components_static(
            own.speed, own.heading
        )
        other_vx, other_vy = CollisionAvoidance._velocity_components_static(
            other.speed, other.heading
        )
        # Relative velocity of other w.r.t. own
        rel_vx = other_vx - own_vx
        rel_vy = other_vy - own_vy

        # Relative position
        dx = other.position.longitude - own.position.longitude
        dy = other.position.latitude - own.position.latitude

        # Convert relative position to approximate meters
        lat_rad = radians(own.position.latitude)
        dx_m = dx * 111320.0 * cos(lat_rad)
        dy_m = dy * 110540.0

        # TCPA = -dot(relative_pos, relative_vel) / |relative_vel|^2
        rel_speed_sq = rel_vx ** 2 + rel_vy ** 2
        if rel_speed_sq < 1e-10:
            return float('inf')

        tcpa = -(dx_m * rel_vx + dy_m * rel_vy) / rel_speed_sq
        return tcpa if tcpa > 0 else -1.0

    @staticmethod
    def compute_dcpa(own: VesselState, other: VesselState) -> float:
        """Compute Distance at Closest Point of Approach.

        Returns DCPA in meters.
        """
        tcpa = CollisionAvoidance.compute_tcpa(own, other)
        if tcpa < 0:
            # Already separating; current distance is closest
            return GeoCalculator.haversine_distance(
                own.position, other.position
            )

        own_vx, own_vy = CollisionAvoidance._velocity_components_static(
            own.speed, own.heading
        )
        other_vx, other_vy = CollisionAvoidance._velocity_components_static(
            other.speed, other.heading
        )
        # Relative velocity of other w.r.t. own
        rel_vx = other_vx - own_vx
        rel_vy = other_vy - own_vy

        dx = other.position.longitude - own.position.longitude
        dy = other.position.latitude - own.position.latitude
        lat_rad = radians(own.position.latitude)
        dx_m = dx * 111320.0 * cos(lat_rad)
        dy_m = dy * 110540.0

        # Position at CPA
        cpa_x = dx_m + rel_vx * tcpa
        cpa_y = dy_m + rel_vy * tcpa

        return sqrt(cpa_x ** 2 + cpa_y ** 2)

    @staticmethod
    def compute_risk_score(
        tcpa: float, dcpa: float, own_speed: float, other_speed: float
    ) -> float:
        """Compute collision risk score [0, 1].

        Higher values indicate greater collision risk.
        """
        if tcpa < 0:
            return 0.0
        if tcpa > 600:
            return 0.0

        # DCPA risk: inversely proportional
        dcpa_risk = 1.0 / (1.0 + (dcpa / 50.0) ** 2)

        # TCPA risk: increases as time decreases
        if tcpa < 1e-6:
            tcpa_risk = 1.0
        else:
            tcpa_risk = 1.0 / (1.0 + (tcpa / 120.0) ** 2)

        # Speed factor: faster vessels = higher risk
        speed_factor = min(1.0, (own_speed + other_speed) / 20.0)

        risk = 0.5 * dcpa_risk + 0.35 * tcpa_risk + 0.15 * speed_factor
        return min(1.0, max(0.0, risk))

    def generate_evasive_maneuver(
        self, threat: CollisionThreat, own_vessel: VesselState
    ) -> Tuple[float, float]:
        """Generate evasive maneuver (heading_change, speed_change).

        Returns:
            Tuple of (heading_change_degrees, speed_multiplier).
            Positive heading change = turn to starboard.
        """
        if threat.severity == Severity.NONE:
            return (0.0, 1.0)

        # Determine which side the threat is approaching from
        bearing_to_threat = GeoCalculator.bearing(
            own_vessel.position, threat.position
        )
        relative_bearing = (bearing_to_threat - own_vessel.heading + 360) % 360

        if threat.severity == Severity.CRITICAL:
            # Hard turn away from threat
            if relative_bearing <= 180:
                heading_change = -60.0  # Turn to port
            else:
                heading_change = 60.0   # Turn to starboard
            speed_mult = 0.5
        elif threat.severity == Severity.HIGH:
            if relative_bearing <= 180:
                heading_change = -35.0
            else:
                heading_change = 35.0
            speed_mult = 0.7
        elif threat.severity == Severity.MEDIUM:
            if relative_bearing <= 180:
                heading_change = -20.0
            else:
                heading_change = 20.0
            speed_mult = 0.85
        else:  # LOW
            heading_change = -10.0 if relative_bearing <= 180 else 10.0
            speed_mult = 0.9

        return (heading_change, speed_mult)

    def compute_safe_zone(
        self, own_vessel: VesselState, speed: float, reaction_time: float = 30.0
    ) -> float:
        """Compute safe zone radius based on speed and reaction time.

        Args:
            own_vessel: Current vessel state.
            speed: Vessel speed in m/s.
            reaction_time: Reaction time in seconds.

        Returns:
            Safe zone radius in meters.
        """
        stopping_distance = speed * reaction_time
        safe_zone = max(self.safe_distance, stopping_distance * 1.5)
        return safe_zone

    def plan_avoidance_path(
        self,
        own_vessel: VesselState,
        threats: List[CollisionThreat],
        destination: Coordinate
    ) -> List[Coordinate]:
        """Plan a path around detected threats to reach destination.

        Returns a list of waypoints (including current position and destination).
        """
        path = [own_vessel.position]

        if not threats:
            path.append(destination)
            return path

        # Sort threats by distance
        sorted_threats = sorted(threats, key=lambda t: t.distance)

        for threat in sorted_threats:
            # Compute avoidance waypoint perpendicular to the threat bearing
            bearing_to_threat = GeoCalculator.bearing(
                own_vessel.position, threat.position
            )
            # Deflect 90 degrees to starboard
            avoidance_bearing = (bearing_to_threat + 90) % 360
            avoidance_dist = self.safe_distance * 1.5
            avoidance_point = GeoCalculator.destination(
                threat.position, avoidance_bearing, avoidance_dist
            )
            path.append(avoidance_point)

        path.append(destination)
        return path

    @staticmethod
    def _velocity_components(speed: float, heading: float) -> Tuple[float, float]:
        """Convert speed and heading to velocity components (east, north) in m/s."""
        heading_rad = radians(heading)
        vx = speed * sin(heading_rad)   # East component
        vy = speed * cos(heading_rad)   # North component
        return (vx, vy)

    @staticmethod
    def _velocity_components_static(speed: float, heading: float) -> Tuple[float, float]:
        """Convert speed and heading to velocity components (east, north) in m/s."""
        heading_rad = radians(heading)
        vx = speed * sin(heading_rad)
        vy = speed * cos(heading_rad)
        return (vx, vy)

    def _risk_to_severity(self, risk_score: float, distance: float) -> Severity:
        """Convert risk score and distance to severity level."""
        if risk_score < 0.15 or distance > self.warning_distance:
            return Severity.NONE
        if risk_score < 0.3:
            return Severity.LOW
        if risk_score < 0.5:
            return Severity.MEDIUM
        if risk_score < 0.7 or distance <= self.critical_distance:
            return Severity.HIGH
        return Severity.CRITICAL
