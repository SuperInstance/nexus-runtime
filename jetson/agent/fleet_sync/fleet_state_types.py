"""
NEXUS Fleet State Types — Extended types for marine fleet coordination.

Defines shared state that fleets need to sync:
- Vessel positions (lat, lon, heading, speed)
- Trust scores (per vessel, per subsystem)
- Task assignments (who's doing what)
- Safety alerts (active emergency states)
- Resource allocations (who needs fuel, supplies)
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    COLLISION_RISK = "collision_risk"
    GROUNDING_RISK = "grounding_risk"
    MAN_OVERBOARD = "man_overboard"
    FIRE = "fire"
    FLOODING = "flooding"
    ENGINE_FAILURE = "engine_failure"
    NAVIGATION_ERROR = "navigation_error"
    COMMUNICATIONS_LOSS = "communications_loss"
    WEATHER_WARNING = "weather_warning"
    LOW_FUEL = "low_fuel"
    MEDICAL_EMERGENCY = "medical_emergency"
    SECURITY_ALERT = "security_alert"
    EQUIPMENT_MALFUNCTION = "equipment_malfunction"


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ResourceType(Enum):
    FUEL = "fuel"
    WATER = "water"
    FOOD = "food"
    MEDICAL_SUPPLIES = "medical_supplies"
    REPAIR_PARTS = "repair_parts"
    BATTERY = "battery"


@dataclass
class VesselPosition:
    """GPS position and motion state of a vessel."""
    vessel_id: str
    latitude: float = 0.0       # degrees, -90 to 90
    longitude: float = 0.0      # degrees, -180 to 180
    heading: float = 0.0        # degrees, 0 to 360
    speed_knots: float = 0.0    # nautical miles per hour
    altitude: float = 0.0       # meters (for UAVs)
    timestamp: float = 0.0
    gps_fix: str = "none"       # none, 2d, 3d, dgps

    def distance_to(self, other: "VesselPosition") -> float:
        """Haversine distance in nautical miles."""
        R = 3440.065  # Earth radius in nautical miles
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    def bearing_to(self, other: "VesselPosition") -> float:
        """Bearing in degrees from self to other."""
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)
        dlon = lon2 - lon1
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
        bearing = math.atan2(x, y)
        return (math.degrees(bearing) + 360) % 360

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vessel_id": self.vessel_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "heading": self.heading,
            "speed_knots": self.speed_knots,
            "altitude": self.altitude,
            "timestamp": self.timestamp,
            "gps_fix": self.gps_fix,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VesselPosition":
        return cls(
            vessel_id=data["vessel_id"],
            latitude=data.get("latitude", 0.0),
            longitude=data.get("longitude", 0.0),
            heading=data.get("heading", 0.0),
            speed_knots=data.get("speed_knots", 0.0),
            altitude=data.get("altitude", 0.0),
            timestamp=data.get("timestamp", 0.0),
            gps_fix=data.get("gps_fix", "none"),
        )

    def is_valid(self) -> bool:
        return (-90 <= self.latitude <= 90 and -180 <= self.longitude <= 180
                and 0 <= self.heading <= 360 and self.speed_knots >= 0)


@dataclass
class SafetyAlert:
    """A safety alert from a vessel."""
    alert_id: str
    vessel_id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str = ""
    timestamp: float = 0.0
    acknowledged_by: str = ""
    resolved: bool = False
    resolved_by: str = ""
    resolved_at: float = 0.0
    affected_vessels: List[str] = field(default_factory=list)

    @property
    def priority(self) -> int:
        """Higher number = higher priority for conflict resolution."""
        severity_order = {
            AlertSeverity.EMERGENCY: 100,
            AlertSeverity.CRITICAL: 75,
            AlertSeverity.WARNING: 50,
            AlertSeverity.INFO: 25,
        }
        return severity_order.get(self.severity, 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "vessel_id": self.vessel_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "acknowledged_by": self.acknowledged_by,
            "resolved": self.resolved,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at,
            "affected_vessels": self.affected_vessels,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SafetyAlert":
        return cls(
            alert_id=data["alert_id"],
            vessel_id=data["vessel_id"],
            alert_type=AlertType(data["alert_type"]),
            severity=AlertSeverity(data["severity"]),
            message=data.get("message", ""),
            timestamp=data.get("timestamp", 0.0),
            acknowledged_by=data.get("acknowledged_by", ""),
            resolved=data.get("resolved", False),
            resolved_by=data.get("resolved_by", ""),
            resolved_at=data.get("resolved_at", 0.0),
            affected_vessels=data.get("affected_vessels", []),
        )


@dataclass
class ResourceLevel:
    """Resource level for a vessel."""
    vessel_id: str
    resource_type: ResourceType
    current_level: float = 100.0  # percentage 0-100
    capacity: float = 100.0
    consumption_rate: float = 0.0  # per hour
    timestamp: float = 0.0
    status: str = "normal"  # normal, low, critical, empty

    @property
    def remaining_hours(self) -> float:
        if self.consumption_rate <= 0:
            return float("inf")
        return (self.current_level / 100.0) * self.capacity / self.consumption_rate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vessel_id": self.vessel_id,
            "resource_type": self.resource_type.value,
            "current_level": self.current_level,
            "capacity": self.capacity,
            "consumption_rate": self.consumption_rate,
            "timestamp": self.timestamp,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceLevel":
        return cls(
            vessel_id=data["vessel_id"],
            resource_type=ResourceType(data["resource_type"]),
            current_level=data.get("current_level", 100.0),
            capacity=data.get("capacity", 100.0),
            consumption_rate=data.get("consumption_rate", 0.0),
            timestamp=data.get("timestamp", 0.0),
            status=data.get("status", "normal"),
        )


@dataclass
class TaskAssignment:
    """Task assignment in the fleet."""
    task_id: str
    description: str = ""
    assigned_to: str = ""
    priority: int = 5           # 1=highest, 10=lowest
    status: TaskStatus = TaskStatus.PENDING
    created_by: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    due_at: float = 0.0
    estimated_duration_hours: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "assigned_to": self.assigned_to,
            "priority": self.priority,
            "status": self.status.value,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "due_at": self.due_at,
            "estimated_duration_hours": self.estimated_duration_hours,
            "dependencies": self.dependencies,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskAssignment":
        return cls(
            task_id=data["task_id"],
            description=data.get("description", ""),
            assigned_to=data.get("assigned_to", ""),
            priority=data.get("priority", 5),
            status=TaskStatus(data.get("status", "pending")),
            created_by=data.get("created_by", ""),
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
            due_at=data.get("due_at", 0.0),
            estimated_duration_hours=data.get("estimated_duration_hours", 0.0),
            dependencies=data.get("dependencies", []),
        )


@dataclass
class SubsystemTrustScore:
    """Per-vessel, per-subsystem trust score."""
    vessel_id: str
    subsystem: str              # steering, engine, navigation, payload, communication
    score: float = 0.0          # 0.0 to 1.0
    autonomy_level: int = 0     # 0-5
    last_updated: float = 0.0
    events_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vessel_id": self.vessel_id,
            "subsystem": self.subsystem,
            "score": self.score,
            "autonomy_level": self.autonomy_level,
            "last_updated": self.last_updated,
            "events_count": self.events_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubsystemTrustScore":
        return cls(
            vessel_id=data["vessel_id"],
            subsystem=data["subsystem"],
            score=data.get("score", 0.0),
            autonomy_level=data.get("autonomy_level", 0),
            last_updated=data.get("last_updated", 0.0),
            events_count=data.get("events_count", 0),
        )
