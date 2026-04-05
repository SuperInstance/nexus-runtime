"""NEXUS Dead Reckoning - Intention Broadcasting.

"Where am I going?" protocol for vessel-to-vessel coordination.
Broadcasts position, destination, ETA, and confidence.
Receives other vessels' intentions for collision risk assessment.

Wire protocol message types (0x30-0x33):
  0x30 POSITION_REPORT    - Own vessel position and velocity
  0x31 INTENTION_BROADCAST - Destination, ETA, confidence
  0x32 COLLISION_WARNING  - CPA-based collision risk alert
  0x33 WAYPOINT_COMMAND   - Waypoint list for coordinated nav

Payload format: struct-packed binary for wire efficiency.
"""

from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from .navigation import NavigationMath


# ===================================================================
# Wire Protocol Message Types (0x30-0x33)
# ===================================================================


class DRMessageType(IntEnum):
    """Dead reckoning wire protocol message types."""

    POSITION_REPORT = 0x30
    INTENTION_BROADCAST = 0x31
    COLLISION_WARNING = 0x32
    WAYPOINT_COMMAND = 0x33


class DRCriticality(IntEnum):
    """Message criticality for dead reckoning messages."""

    TELEMETRY = 0
    COORDINATION = 1
    SAFETY = 2


class DRDirection(IntEnum):
    """Message direction."""

    V2V = 0  # Vessel to Vessel
    V2S = 1  # Vessel to Shore
    S2V = 2  # Shore to Vessel
    BOTH = 3


# Metadata: (direction, criticality)
DR_MSG_TYPE_INFO: dict[int, tuple[int, int]] = {
    DRMessageType.POSITION_REPORT: (DRDirection.BOTH, DRCriticality.TELEMETRY),
    DRMessageType.INTENTION_BROADCAST: (DRDirection.V2V, DRCriticality.COORDINATION),
    DRMessageType.COLLISION_WARNING: (DRDirection.BOTH, DRCriticality.SAFETY),
    DRMessageType.WAYPOINT_COMMAND: (DRDirection.S2V, DRCriticality.COORDINATION),
}


# Payload format strings for struct.pack/unpack
# Position report: vessel_id(8s) lat(f) lon(f) heading(f) speed(f) vn(f) ve(f) confidence(f) timestamp(I)
_POS_FMT = ">8sfffffffI"  # 36 bytes
_POS_SIZE = struct.calcsize(_POS_FMT)

# Intention broadcast: vessel_id(8s) lat(f) lon(f) dest_lat(f) dest_lon(f) eta_seconds(f) confidence(f) timestamp(I)
_INT_FMT = ">8sffffffI"   # 32 bytes
_INT_SIZE = struct.calcsize(_INT_FMT)

# Collision warning: vessel_id(8s) other_id(8s) cpa_dist(f) cpa_time(f) risk_level(B) timestamp(I)
_COL_FMT = ">8s8sffBI"    # 25 bytes
_COL_SIZE = struct.calcsize(_COL_FMT)

# Waypoint command: vessel_id(8s) num_waypoints(B) wp0_lat(f) wp0_lon(f) ...
_WP_HEADER = ">8sB"       # 9 bytes


@dataclass
class IntentionMessage:
    """A vessel's intention broadcast: position + destination + ETA + confidence."""

    vessel_id: str = ""
    current_lat: float = 0.0
    current_lon: float = 0.0
    dest_lat: float = 0.0
    dest_lon: float = 0.0
    eta_seconds: float = 0.0
    confidence: float = 0.0
    heading: float = 0.0
    speed: float = 0.0
    timestamp_ms: int = 0

    @property
    def is_valid(self) -> bool:
        return self.vessel_id and self.confidence > 0.01

    def encode_position_report(self) -> bytes:
        """Encode as POSITION_REPORT (0x30) payload."""
        vid = self.vessel_id.encode("utf-8")[:8].ljust(8, b"\x00")
        ts = self.timestamp_ms if self.timestamp_ms else int(time.time() * 1000)
        return struct.pack(
            _POS_FMT,
            vid,
            self.current_lat, self.current_lon,
            self.heading, self.speed,
            0.0, 0.0,  # vn, ve placeholder
            self.confidence,
            ts & 0xFFFFFFFF,
        )

    def encode_intention_broadcast(self) -> bytes:
        """Encode as INTENTION_BROADCAST (0x31) payload."""
        vid = self.vessel_id.encode("utf-8")[:8].ljust(8, b"\x00")
        ts = self.timestamp_ms if self.timestamp_ms else int(time.time() * 1000)
        return struct.pack(
            _INT_FMT,
            vid,
            self.current_lat, self.current_lon,
            self.dest_lat, self.dest_lon,
            self.eta_seconds, self.confidence,
            ts & 0xFFFFFFFF,
        )

    @classmethod
    def decode_position_report(cls, payload: bytes) -> IntentionMessage:
        """Decode a POSITION_REPORT payload."""
        if len(payload) < _POS_SIZE:
            raise ValueError(f"Payload too short: {len(payload)} < {_POS_SIZE}")
        vid, lat, lon, hdg, spd, _vn, _ve, conf, ts = struct.unpack(_POS_FMT, payload[:_POS_SIZE])
        return cls(
            vessel_id=vid.rstrip(b"\x00").decode("utf-8", errors="replace"),
            current_lat=lat, current_lon=lon,
            heading=hdg, speed=spd,
            confidence=conf,
            timestamp_ms=ts,
        )

    @classmethod
    def decode_intention_broadcast(cls, payload: bytes) -> IntentionMessage:
        """Decode an INTENTION_BROADCAST payload."""
        if len(payload) < _INT_SIZE:
            raise ValueError(f"Payload too short: {len(payload)} < {_INT_SIZE}")
        vid, lat, lon, dlat, dlon, eta, conf, ts = struct.unpack(_INT_FMT, payload[:_INT_SIZE])
        return cls(
            vessel_id=vid.rstrip(b"\x00").decode("utf-8", errors="replace"),
            current_lat=lat, current_lon=lon,
            dest_lat=dlat, dest_lon=dlon,
            eta_seconds=eta, confidence=conf,
            timestamp_ms=ts,
        )


@dataclass
class CollisionAssessment:
    """Result of a CPA (Closest Point of Approach) collision risk assessment."""

    vessel_id: str = ""
    other_vessel_id: str = ""
    cpa_distance_m: float = float("inf")   # distance at CPA
    cpa_time_s: float = float("inf")       # time to CPA in seconds
    risk_level: int = 0                     # 0=none, 1=caution, 2=warning, 3=danger
    tcpa: float = float("inf")              # time to CPA
    bearing_to_other: float = 0.0           # degrees

    @property
    def is_risk(self) -> bool:
        return self.risk_level >= 2

    @property
    def risk_label(self) -> str:
        labels = {0: "NONE", 1: "CAUTION", 2: "WARNING", 3: "DANGER"}
        return labels.get(self.risk_level, "UNKNOWN")

    def encode_collision_warning(self) -> bytes:
        """Encode as COLLISION_WARNING (0x32) payload."""
        vid = self.vessel_id.encode("utf-8")[:8].ljust(8, b"\x00")
        oid = self.other_vessel_id.encode("utf-8")[:8].ljust(8, b"\x00")
        ts = int(time.time() * 1000) & 0xFFFFFFFF
        return struct.pack(
            _COL_FMT,
            vid, oid,
            self.cpa_distance_m, self.cpa_time_s,
            self.risk_level,
            ts,
        )

    @classmethod
    def decode_collision_warning(cls, payload: bytes) -> CollisionAssessment:
        """Decode a COLLISION_WARNING payload."""
        if len(payload) < _COL_SIZE:
            raise ValueError(f"Payload too short: {len(payload)} < {_COL_SIZE}")
        vid, oid, cpa_dist, cpa_time, risk, ts = struct.unpack(_COL_FMT, payload[:_COL_SIZE])
        return cls(
            vessel_id=vid.rstrip(b"\x00").decode("utf-8", errors="replace"),
            other_vessel_id=oid.rstrip(b"\x00").decode("utf-8", errors="replace"),
            cpa_distance_m=cpa_dist,
            cpa_time_s=cpa_time,
            risk_level=risk,
        )


class CPAAlgorithm:
    """Closest Point of Approach (CPA) algorithm for collision risk.

    Computes the minimum distance between two vessels on their
    predicted courses, and the time at which it occurs.

    Uses relative velocity vectors projected onto the line
    connecting the two vessels.
    """

    def __init__(
        self,
        warning_distance_m: float = 500.0,
        danger_distance_m: float = 200.0,
        caution_distance_m: float = 1000.0,
        max_prediction_time_s: float = 600.0,
    ) -> None:
        self.warning_distance = warning_distance_m
        self.danger_distance = danger_distance_m
        self.caution_distance = caution_distance_m
        self.max_prediction_time = max_prediction_time_s

    def compute_cpa(
        self,
        own_lat: float, own_lon: float,
        own_speed: float, own_heading: float,
        other_lat: float, other_lon: float,
        other_speed: float, other_heading: float,
    ) -> CollisionAssessment:
        """Compute CPA between own vessel and another vessel.

        Args:
            own_lat, own_lon: Own position (degrees).
            own_speed: Own speed in m/s.
            own_heading: Own heading in degrees.
            other_lat, other_lon: Other vessel position (degrees).
            other_speed: Other vessel speed in m/s.
            other_heading: Other vessel heading in degrees.

        Returns:
            CollisionAssessment with CPA distance, time, and risk level.
        """
        # Convert to local Cartesian (approximate)
        # Use own position as origin
        dx = (other_lon - own_lon) * 111320.0 * math.cos(math.radians(own_lat))
        dy = (other_lat - own_lat) * 111320.0

        # Velocity components in m/s (north, east)
        own_h = math.radians(own_heading)
        other_h = math.radians(other_heading)
        own_vn = own_speed * math.cos(own_h)
        own_ve = own_speed * math.sin(own_h)
        other_vn = other_speed * math.cos(other_h)
        other_ve = other_speed * math.sin(other_h)

        # Relative velocity (own - other in terms of closing)
        rel_vx = other_ve - own_ve
        rel_vy = other_vn - own_vn

        # Relative position vector
        rx = dx
        ry = dy

        # CPA calculation
        # CPA occurs when d/dt(range^2) = 0
        # range^2 = (rx + rel_vx*t)^2 + (ry + rel_vy*t)^2
        # d(range^2)/dt = 2*(rx + rel_vx*t)*rel_vx + 2*(ry + rel_vy*t)*rel_vy = 0
        # t_cpa = -(rx*rel_vx + ry*rel_vy) / (rel_vx^2 + rel_vy^2)

        rel_speed_sq = rel_vx ** 2 + rel_vy ** 2

        if rel_speed_sq < 1e-6:
            # Vessels moving at same velocity — current distance is CPA
            cpa_dist = math.sqrt(rx ** 2 + ry ** 2)
            cpa_time = float("inf")
        else:
            t_cpa = -(rx * rel_vx + ry * rel_vy) / rel_speed_sq
            cpa_time = t_cpa

            # If CPA is in the past, current distance is minimum
            if t_cpa < 0:
                cpa_dist = math.sqrt(rx ** 2 + ry ** 2)
                cpa_time = 0.0
            elif t_cpa > self.max_prediction_time:
                cpa_dist = math.sqrt(
                    (rx + rel_vx * self.max_prediction_time) ** 2
                    + (ry + rel_vy * self.max_prediction_time) ** 2
                )
                cpa_time = self.max_prediction_time
            else:
                cpx = rx + rel_vx * t_cpa
                cpy = ry + rel_vy * t_cpa
                cpa_dist = math.sqrt(cpx ** 2 + cpy ** 2)

        # Bearing to other vessel
        bearing = math.degrees(math.atan2(dx, dy)) % 360

        # Determine risk level
        if cpa_dist <= self.danger_distance and cpa_time <= 300:
            risk = 3  # DANGER
        elif cpa_dist <= self.warning_distance and cpa_time <= 300:
            risk = 2  # WARNING
        elif cpa_dist <= self.caution_distance and cpa_time <= 300:
            risk = 1  # CAUTION
        else:
            risk = 0  # NONE

        return CollisionAssessment(
            cpa_distance_m=cpa_dist,
            cpa_time_s=cpa_time,
            risk_level=risk,
            tcpa=cpa_time,
            bearing_to_other=bearing,
        )

    def assess_intention(
        self,
        own: IntentionMessage,
        other: IntentionMessage,
    ) -> CollisionAssessment:
        """Assess collision risk between two vessels' intentions.

        Args:
            own: Own vessel's intention.
            other: Other vessel's intention.

        Returns:
            CollisionAssessment.
        """
        result = self.compute_cpa(
            own.current_lat, own.current_lon,
            own.speed, own.heading,
            other.current_lat, other.current_lon,
            other.speed, other.heading,
        )
        result.vessel_id = own.vessel_id
        result.other_vessel_id = other.vessel_id
        return result


class IntentionBroadcaster:
    """Intention broadcasting system for vessel-to-vessel coordination.

    Broadcasts own vessel's position, destination, ETA, and confidence.
    Receives other vessels' intentions and computes collision risks.

    Usage:
        broadcaster = IntentionBroadcaster(vessel_id="VESSEL_A")
        broadcaster.update_own(
            lat=32.0, lon=-117.0, heading=45.0, speed=5.0,
            dest_lat=32.5, dest_lon=-116.5, confidence=0.9,
        )
        msg = broadcaster.get_broadcast()
        # Receive other vessel's message
        broadcaster.receive_intention(other_msg)
        risks = broadcaster.get_collision_risks()
    """

    def __init__(
        self,
        vessel_id: str = "",
        cpa_warning_m: float = 500.0,
        cpa_danger_m: float = 200.0,
    ) -> None:
        self.vessel_id = vessel_id
        self._cpa = CPAAlgorithm(
            warning_distance_m=cpa_warning_m,
            danger_distance_m=cpa_danger_m,
        )
        self._own_intention = IntentionMessage(vessel_id=vessel_id)
        self._known_vessels: dict[str, IntentionMessage] = {}
        self._collision_risks: list[CollisionAssessment] = []

    def update_own(
        self,
        lat: float,
        lon: float,
        heading: float = 0.0,
        speed: float = 0.0,
        dest_lat: float = 0.0,
        dest_lon: float = 0.0,
        eta_seconds: float = 0.0,
        confidence: float = 1.0,
    ) -> IntentionMessage:
        """Update own vessel's position and intention.

        Args:
            lat: Current latitude.
            lon: Current longitude.
            heading: Current heading in degrees.
            speed: Current speed in m/s.
            dest_lat: Destination latitude.
            dest_lon: Destination longitude.
            eta_seconds: Estimated time to arrival in seconds.
            confidence: Position confidence (0-1).

        Returns:
            Updated IntentionMessage.
        """
        self._own_intention = IntentionMessage(
            vessel_id=self.vessel_id,
            current_lat=lat, current_lon=lon,
            dest_lat=dest_lat, dest_lon=dest_lon,
            eta_seconds=eta_seconds,
            confidence=confidence,
            heading=heading,
            speed=speed,
            timestamp_ms=int(time.time() * 1000),
        )

        # Re-assess all collision risks
        self._assess_all_risks()

        return self._own_intention

    def get_broadcast(self) -> IntentionMessage:
        """Get own vessel's intention for broadcasting."""
        return self._own_intention

    def receive_intention(self, msg: IntentionMessage) -> CollisionAssessment | None:
        """Receive another vessel's intention and assess collision risk.

        Args:
            msg: Other vessel's intention message.

        Returns:
            CollisionAssessment if risk detected, None otherwise.
        """
        if not msg.is_valid:
            return None

        self._known_vessels[msg.vessel_id] = msg
        assessment = self._cpa.assess_intention(self._own_intention, msg)
        self._update_risks(assessment)
        return assessment if assessment.risk_level > 0 else None

    def receive_position_report(self, payload: bytes) -> IntentionMessage | None:
        """Receive and decode a position report (type 0x30)."""
        try:
            msg = IntentionMessage.decode_position_report(payload)
            self._known_vessels[msg.vessel_id] = msg
            return msg
        except (ValueError, struct.error):
            return None

    def receive_intention_broadcast(self, payload: bytes) -> IntentionMessage | None:
        """Receive and decode an intention broadcast (type 0x31)."""
        try:
            msg = IntentionMessage.decode_intention_broadcast(payload)
            return self.receive_intention(msg)
        except (ValueError, struct.error):
            return None

    def get_collision_risks(self) -> list[CollisionAssessment]:
        """Return all active collision risks sorted by severity."""
        return sorted(self._collision_risks, key=lambda r: -r.risk_level)

    def get_known_vessels(self) -> dict[str, IntentionMessage]:
        """Return all known vessel intentions."""
        return dict(self._known_vessels)

    def _assess_all_risks(self) -> None:
        """Re-assess collision risks with all known vessels."""
        self._collision_risks.clear()
        for msg in self._known_vessels.values():
            assessment = self._cpa.assess_intention(self._own_intention, msg)
            if assessment.risk_level > 0:
                self._update_risks(assessment)

    def _update_risks(self, assessment: CollisionAssessment) -> None:
        """Update collision risks list."""
        # Remove existing assessment for this vessel pair
        self._collision_risks = [
            r for r in self._collision_risks
            if not (
                r.vessel_id == assessment.vessel_id
                and r.other_vessel_id == assessment.other_vessel_id
            )
        ]
        if assessment.risk_level > 0:
            self._collision_risks.append(assessment)
