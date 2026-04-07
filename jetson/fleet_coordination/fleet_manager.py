"""Fleet state management — vessel registration, status tracking, anomaly detection."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class VesselHealth(Enum):
    NOMINAL = "nominal"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"


@dataclass
class VesselStatus:
    """Real-time status of a single vessel in the fleet."""
    vessel_id: str
    position: Tuple[float, float] = (0.0, 0.0)
    heading: float = 0.0           # degrees, 0=North
    speed: float = 0.0             # knots
    fuel: float = 100.0            # percentage 0-100
    health: float = 1.0            # 0.0 (dead) to 1.0 (perfect)
    trust_score: float = 1.0       # 0.0 to 1.0
    available: bool = True
    current_task: Optional[str] = None
    last_heartbeat: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def distance_to(self, other: VesselStatus) -> float:
        dx = self.position[0] - other.position[0]
        dy = self.position[1] - other.position[1]
        return math.sqrt(dx * dx + dy * dy)

    def bearing_to(self, other: VesselStatus) -> float:
        dx = other.position[0] - self.position[0]
        dy = other.position[1] - self.position[1]
        angle = math.degrees(math.atan2(dx, dy)) % 360
        return angle


@dataclass
class FleetState:
    """Snapshot of the entire fleet at a point in time."""
    vessels: List[VesselStatus] = field(default_factory=list)
    tasks: List[Any] = field(default_factory=list)
    connectivity_graph: Dict[str, List[str]] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)


@dataclass
class AnomalyRecord:
    """Record of a detected fleet anomaly."""
    anomaly_type: str
    vessel_id: Optional[str]
    description: str
    severity: float  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)


class FleetManager:
    """Central manager for fleet state, vessel lifecycle, and anomaly detection."""

    def __init__(self) -> None:
        self._vessels: Dict[str, VesselStatus] = {}
        self._tasks: List[Any] = []
        self._connectivity: Dict[str, List[str]] = {}
        self._anomaly_history: List[AnomalyRecord] = []
        self._last_updated: float = time.time()

    # ------------------------------------------------------------------ CRUD
    def register_vessel(self, vessel_info: Dict[str, Any]) -> VesselStatus:
        """Register a new vessel in the fleet."""
        vid = vessel_info.get("vessel_id")
        if vid is None:
            raise ValueError("vessel_id is required")
        if vid in self._vessels:
            raise ValueError(f"Vessel {vid} already registered")

        status = VesselStatus(
            vessel_id=vid,
            position=tuple(vessel_info.get("position", (0.0, 0.0))),
            heading=vessel_info.get("heading", 0.0),
            speed=vessel_info.get("speed", 0.0),
            fuel=vessel_info.get("fuel", 100.0),
            health=vessel_info.get("health", 1.0),
            trust_score=vessel_info.get("trust_score", 1.0),
            available=vessel_info.get("available", True),
            metadata=vessel_info.get("metadata", {}),
        )
        self._vessels[vid] = status
        self._connectivity[vid] = []
        self._touch()
        logger.info("Vessel registered: %s (fleet size now %d)", vid, len(self._vessels))
        return status

    def deregister_vessel(self, vessel_id: str) -> bool:
        """Remove a vessel from the fleet. Returns True if found."""
        if vessel_id not in self._vessels:
            return False
        del self._vessels[vessel_id]
        logger.info("Vessel deregistered: %s", vessel_id)
        # Clean connectivity references
        if vessel_id in self._connectivity:
            del self._connectivity[vessel_id]
        for neighbours in self._connectivity.values():
            if vessel_id in neighbours:
                neighbours.remove(vessel_id)
        self._touch()
        return True

    def update_vessel_status(self, vessel_id: str, status: Dict[str, Any]) -> bool:
        """Partial-update fields on an existing vessel. Returns True on success."""
        if vessel_id not in self._vessels:
            return False
        v = self._vessels[vessel_id]
        if "position" in status:
            v.position = tuple(status["position"])
        if "heading" in status:
            v.heading = status["heading"]
        if "speed" in status:
            v.speed = status["speed"]
        if "fuel" in status:
            v.fuel = max(0.0, min(100.0, status["fuel"]))
        if "health" in status:
            v.health = max(0.0, min(1.0, status["health"]))
        if "trust_score" in status:
            v.trust_score = max(0.0, min(1.0, status["trust_score"]))
        if "available" in status:
            v.available = status["available"]
        if "current_task" in status:
            v.current_task = status["current_task"]
        if "metadata" in status:
            v.metadata.update(status["metadata"])
        v.last_heartbeat = time.time()
        self._touch()
        return True

    # ----------------------------------------------------------------- Query
    def get_vessel(self, vessel_id: str) -> Optional[VesselStatus]:
        return self._vessels.get(vessel_id)

    def get_all_vessels(self) -> List[VesselStatus]:
        return list(self._vessels.values())

    def get_available_vessels(self) -> List[VesselStatus]:
        return [v for v in self._vessels.values() if v.available]

    # ---------------------------------------------------------------- Health
    def compute_fleet_health(self) -> float:
        """Aggregate health score across all vessels, 0-1."""
        if not self._vessels:
            return 0.0
        total = sum(v.health for v in self._vessels.values())
        return total / len(self._vessels)

    def get_fleet_snapshot(self) -> FleetState:
        return FleetState(
            vessels=list(self._vessels.values()),
            tasks=list(self._tasks),
            connectivity_graph=dict(self._connectivity),
            last_updated=self._last_updated,
        )

    # -------------------------------------------------------------- Anomaly
    def detect_anomalies(self) -> List[AnomalyRecord]:
        """Run anomaly detection across the fleet and return records."""
        anomalies: List[AnomalyRecord] = []
        now = time.time()

        for vid, v in self._vessels.items():
            # Low fuel anomaly
            if v.fuel < 15.0:
                anomalies.append(AnomalyRecord(
                    anomaly_type="low_fuel",
                    vessel_id=vid,
                    description=f"Vessel {vid} fuel critically low: {v.fuel:.1f}%",
                    severity=max(0.0, 1.0 - v.fuel / 15.0),
                    details={"fuel": v.fuel},
                ))

            # Health degradation anomaly
            if v.health < 0.5:
                anomalies.append(AnomalyRecord(
                    anomaly_type="health_degradation",
                    vessel_id=vid,
                    description=f"Vessel {vid} health degraded: {v.health:.2f}",
                    severity=max(0.0, 1.0 - v.health),
                    details={"health": v.health},
                ))

            # Low trust anomaly
            if v.trust_score < 0.3:
                anomalies.append(AnomalyRecord(
                    anomaly_type="low_trust",
                    vessel_id=vid,
                    description=f"Vessel {vid} trust score low: {v.trust_score:.2f}",
                    severity=max(0.0, 1.0 - v.trust_score),
                    details={"trust_score": v.trust_score},
                ))

            # Stale heartbeat anomaly (> 300 s stale)
            age = now - v.last_heartbeat
            if age > 300:
                anomalies.append(AnomalyRecord(
                    anomaly_type="stale_heartbeat",
                    vessel_id=vid,
                    description=f"Vessel {vid} heartbeat stale: {age:.0f}s",
                    severity=min(1.0, age / 600),
                    details={"age_seconds": age},
                ))

        # Proximity anomaly: vessels within 50 m
        vessel_list = list(self._vessels.values())
        for i in range(len(vessel_list)):
            for j in range(i + 1, len(vessel_list)):
                dist = vessel_list[i].distance_to(vessel_list[j])
                if 0 < dist < 50:
                    anomalies.append(AnomalyRecord(
                        anomaly_type="proximity_warning",
                        vessel_id=None,
                        description=(
                            f"Vessels {vessel_list[i].vessel_id} and "
                            f"{vessel_list[j].vessel_id} dangerously close: "
                            f"{dist:.1f}m"
                        ),
                        severity=max(0.0, 1.0 - dist / 50),
                        details={
                            "vessel_a": vessel_list[i].vessel_id,
                            "vessel_b": vessel_list[j].vessel_id,
                            "distance": dist,
                        },
                    ))

        self._anomaly_history.extend(anomalies)
        if anomalies:
            logger.warning(
                "Detected %d fleet anomalies: %s",
                len(anomalies),
                ", ".join(a.anomaly_type for a in anomalies),
            )
        return anomalies

    def get_anomaly_history(self) -> List[AnomalyRecord]:
        return list(self._anomaly_history)

    # --------------------------------------------------------- Connectivity
    def add_connection(self, vessel_a: str, vessel_b: str) -> bool:
        if vessel_a not in self._vessels or vessel_b not in self._vessels:
            return False
        if vessel_b not in self._connectivity[vessel_a]:
            self._connectivity[vessel_a].append(vessel_b)
        if vessel_a not in self._connectivity[vessel_b]:
            self._connectivity[vessel_b].append(vessel_a)
        self._touch()
        return True

    def remove_connection(self, vessel_a: str, vessel_b: str) -> bool:
        if vessel_a in self._connectivity and vessel_b in self._connectivity[vessel_a]:
            self._connectivity[vessel_a].remove(vessel_b)
        if vessel_b in self._connectivity and vessel_a in self._connectivity[vessel_b]:
            self._connectivity[vessel_b].remove(vessel_a)
        return True

    # --------------------------------------------------------- Internal
    def _touch(self) -> None:
        self._last_updated = time.time()
