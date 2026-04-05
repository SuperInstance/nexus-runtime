"""NEXUS Edge Heartbeat — Vessel State Tracking.

VesselState dataclass and VesselStateManager for tracking the current
state of a NEXUS edge vessel. Integrates sensor readings, trust scores,
and mission queue status.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field


@dataclass
class VesselState:
    """Current state of a NEXUS edge vessel.

    Aggregates sensor health, trust scores, mission queue, and uptime
    into a single state object updated every heartbeat cycle.
    """

    vessel_id: str = "nexus-vessel-001"
    uptime_seconds: float = 0.0
    trust_scores: dict[str, float] = field(default_factory=dict)
    autonomy_level: int = 0
    sensor_status: dict[str, str] = field(default_factory=dict)
    current_mission: str | None = None
    pending_missions: list[str] = field(default_factory=list)
    last_heartbeat: float = 0.0
    error_count: int = 0
    mission_count: int = 0
    mission_failures: int = 0


class VesselStateManager:
    """Manages vessel state lifecycle.

    Provides update methods for sensor readings and trust scores,
    generates status reports, and serializes state to JSON.
    """

    def __init__(self, vessel_id: str = "nexus-vessel-001") -> None:
        self._start_time: float = time.time()
        self.state = VesselState(
            vessel_id=vessel_id,
            last_heartbeat=time.time(),
        )

    @property
    def vessel_id(self) -> str:
        return self.state.vessel_id

    def update_from_sensors(self, readings: dict[str, str]) -> None:
        """Update sensor status from a dict of sensor_id -> status.

        Valid status values: 'ok', 'degraded', 'offline'.
        Invalid values are silently ignored.

        Args:
            readings: Mapping of sensor IDs to their status strings.
        """
        valid_statuses = {"ok", "degraded", "offline"}
        for sensor_id, status in readings.items():
            if status in valid_statuses:
                self.state.sensor_status[sensor_id] = status

    def update_from_trust(self, trust_scores: dict[str, float]) -> None:
        """Update trust scores from subsystem-level trust map.

        Also recalculates the overall autonomy level as the minimum
        autonomy level across all subsystems with non-zero trust.

        Args:
            trust_scores: Mapping of subsystem name -> trust score (0.0-1.0).
        """
        self.state.trust_scores.update(trust_scores)

    def set_autonomy_level(self, level: int) -> None:
        """Manually set the overall autonomy level (0-5).

        Args:
            level: Autonomy level, clamped to [0, 5].
        """
        self.state.autonomy_level = max(0, min(5, level))

    def set_current_mission(self, mission: str | None) -> None:
        """Set the currently executing mission.

        Args:
            mission: Mission identifier, or None if idle.
        """
        self.state.current_mission = mission

    def set_pending_missions(self, missions: list[str]) -> None:
        """Set the queue of pending missions.

        Args:
            missions: List of mission identifiers waiting to execute.
        """
        self.state.pending_missions = list(missions)

    def record_heartbeat(self) -> None:
        """Record a heartbeat timestamp and update uptime."""
        now = time.time()
        self.state.last_heartbeat = now
        self.state.uptime_seconds = now - self._start_time

    def record_mission_complete(self, success: bool = True) -> None:
        """Record a mission execution result.

        Args:
            success: Whether the mission succeeded.
        """
        self.state.mission_count += 1
        if not success:
            self.state.mission_failures += 1

    def record_error(self) -> None:
        """Increment the error counter."""
        self.state.error_count += 1

    def get_status_report(self) -> dict:
        """Generate a comprehensive status report dict.

        Returns:
            Dictionary with vessel health summary, suitable for
            JSON serialization and git-agent .agent/status reporting.
        """
        total_sensors = len(self.state.sensor_status)
        ok_sensors = sum(1 for s in self.state.sensor_status.values() if s == "ok")
        degraded_sensors = sum(1 for s in self.state.sensor_status.values() if s == "degraded")
        offline_sensors = sum(1 for s in self.state.sensor_status.values() if s == "offline")

        avg_trust = 0.0
        if self.state.trust_scores:
            avg_trust = sum(self.state.trust_scores.values()) / len(self.state.trust_scores)

        min_trust = min(self.state.trust_scores.values()) if self.state.trust_scores else 0.0
        max_trust = max(self.state.trust_scores.values()) if self.state.trust_scores else 0.0

        # Determine overall status level
        # Only consider sensor/trust thresholds when data exists
        has_trust = len(self.state.trust_scores) > 0
        if (total_sensors > 0 and offline_sensors > 0) or (has_trust and min_trust < 0.2):
            status_level = "ALERT"
        elif (total_sensors > 0 and degraded_sensors > 0) or (has_trust and min_trust < 0.4):
            status_level = "ATTENTION"
        elif self.state.current_mission is not None:
            status_level = "EXECUTING"
        else:
            status_level = "ALL_CLEAR"

        return {
            "vessel_id": self.state.vessel_id,
            "status_level": status_level,
            "uptime_seconds": round(self.state.uptime_seconds, 1),
            "last_heartbeat": self.state.last_heartbeat,
            "autonomy_level": self.state.autonomy_level,
            "current_mission": self.state.current_mission,
            "pending_missions": len(self.state.pending_missions),
            "missions_completed": self.state.mission_count,
            "mission_failures": self.state.mission_failures,
            "error_count": self.state.error_count,
            "sensors": {
                "total": total_sensors,
                "ok": ok_sensors,
                "degraded": degraded_sensors,
                "offline": offline_sensors,
                "detail": dict(self.state.sensor_status),
            },
            "trust": {
                "average": round(avg_trust, 4),
                "min": round(min_trust, 4),
                "max": round(max_trust, 4),
                "by_subsystem": {
                    k: round(v, 4) for k, v in self.state.trust_scores.items()
                },
            },
        }

    def to_json(self) -> str:
        """Serialize the full VesselState to a JSON string."""
        state_dict = {
            "vessel_id": self.state.vessel_id,
            "uptime_seconds": round(self.state.uptime_seconds, 3),
            "trust_scores": {k: round(v, 4) for k, v in self.state.trust_scores.items()},
            "autonomy_level": self.state.autonomy_level,
            "sensor_status": dict(self.state.sensor_status),
            "current_mission": self.state.current_mission,
            "pending_missions": self.state.pending_missions,
            "last_heartbeat": self.state.last_heartbeat,
            "error_count": self.state.error_count,
            "mission_count": self.state.mission_count,
            "mission_failures": self.state.mission_failures,
        }
        return json.dumps(state_dict, indent=2)

    def from_json(self, json_str: str) -> None:
        """Restore VesselState from a JSON string.

        Args:
            json_str: JSON-serialized VesselState.
        """
        data = json.loads(json_str)
        self.state = VesselState(
            vessel_id=data.get("vessel_id", self.state.vessel_id),
            uptime_seconds=data.get("uptime_seconds", 0.0),
            trust_scores=data.get("trust_scores", {}),
            autonomy_level=data.get("autonomy_level", 0),
            sensor_status=data.get("sensor_status", {}),
            current_mission=data.get("current_mission"),
            pending_missions=data.get("pending_missions", []),
            last_heartbeat=data.get("last_heartbeat", 0.0),
            error_count=data.get("error_count", 0),
            mission_count=data.get("mission_count", 0),
            mission_failures=data.get("mission_failures", 0),
        )
        # Keep the start_time consistent with uptime
        if self.state.uptime_seconds > 0:
            self._start_time = time.time() - self.state.uptime_seconds

    def reset(self) -> None:
        """Reset vessel state to initial values (preserving vessel_id)."""
        vid = self.state.vessel_id
        now = time.time()
        self._start_time = now
        self.state = VesselState(
            vessel_id=vid,
            last_heartbeat=now,
        )
