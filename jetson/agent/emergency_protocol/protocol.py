"""Emergency Protocol — Detection and response for NEXUS vessel emergencies.

Emergency levels:
    GREEN:  All systems normal
    YELLOW: Non-critical anomaly detected (sensor degradation, minor trust loss)
    ORANGE: Significant issue (sensor failure, trust below threshold, mission timeout)
    RED:    Critical emergency (E-Stop, safety violation, communication loss, trust collapse)

Actions per level:
    GREEN:  Normal operations
    YELLOW: Log event, increase monitoring frequency
    ORANGE: Alert captain, reduce autonomy, prepare contingency
    RED:    HALT all autonomous operations, alert fleet, create GitHub Issue
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EmergencyLevel(str, Enum):
    """Emergency severity levels."""

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED = "RED"


class IncidentCategory(str, Enum):
    """Categories of incidents that can trigger emergency levels."""

    SENSOR = "SENSOR"
    TRUST = "TRUST"
    SAFETY = "SAFETY"
    COMMUNICATION = "COMMUNICATION"
    MISSION = "MISSION"


# ── Thresholds ────────────────────────────────────────────────────

DEFAULT_THRESHOLDS: dict[str, float] = {
    "trust_yellow": 0.60,   # Trust score at or below → YELLOW
    "trust_orange": 0.35,   # Trust score at or below → ORANGE
    "trust_red": 0.15,      # Trust score at or below → RED
    "sensor_stale_seconds": 30.0,   # Sensor data older than this → YELLOW
    "comm_timeout_seconds": 120.0,  # No comms for this long → ORANGE
    "comm_dead_seconds": 300.0,     # No comms for this long → RED
    "mission_overrun_fraction": 1.5,  # Mission exceeding expected * fraction → ORANGE
    "rapid_trust_loss_rate": 0.2,     # Trust loss per assessment cycle → ORANGE
}


# ── Data classes ──────────────────────────────────────────────────

@dataclass
class Incident:
    """Record of a single emergency incident."""

    id: str
    level: str  # GREEN, YELLOW, ORANGE, RED
    category: str  # SENSOR, TRUST, SAFETY, COMMUNICATION, MISSION
    description: str
    timestamp: float
    vessel_state: dict = field(default_factory=dict)
    trust_scores: dict = field(default_factory=dict)
    auto_actions_taken: list[str] = field(default_factory=list)
    resolution: str | None = None
    resolved_at: float | None = None


@dataclass
class EmergencyAssessment:
    """Result of an emergency assessment cycle."""

    vessel_id: str
    timestamp: float
    previous_level: str
    current_level: str
    incidents_detected: list[Incident] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)
    reason: str = ""
    trust_snapshot: dict = field(default_factory=dict)

    @property
    def level_changed(self) -> bool:
        return self.previous_level != self.current_level

    @property
    def escalated(self) -> bool:
        """True if emergency level increased."""
        level_order = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}
        return level_order.get(self.current_level, 0) > level_order.get(
            self.previous_level, 0
        )


@dataclass
class EscalationResult:
    """Result of escalating an incident."""

    incident_id: str
    level: str
    issue_created: bool = False
    issue_number: int = 0
    trust_reduced: bool = False
    fleet_notified: bool = False
    commit_hash: str = ""
    error: str = ""


@dataclass
class DeescalationResult:
    """Result of de-escalating an incident."""

    incident_id: str
    previous_level: str
    new_level: str
    issue_closed: bool = False
    trust_restored: bool = False
    operations_resumed: bool = False
    error: str = ""


# ── Helper ────────────────────────────────────────────────────────

def generate_incident_id() -> str:
    """Generate a unique incident identifier."""
    short_uuid = uuid.uuid4().hex[:8]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"INC-{ts}-{short_uuid}"


# ── Main Emergency Protocol ──────────────────────────────────────

class EmergencyProtocol:
    """Detects and responds to emergency situations on NEXUS vessels.

    Monitors vessel state, trust scores, and sensor status to determine
    the current emergency level. Coordinates automated responses and
    fleet-wide alerts via the git-agent bridge.

    Usage:
        protocol = EmergencyProtocol(vessel_id="vessel-001", bridge=bridge)
        assessment = protocol.assess(vessel_state, trust_scores, sensor_status)
        if assessment.escalated:
            protocol.escalate(assessment.incidents_detected[0])
    """

    def __init__(
        self,
        vessel_id: str,
        bridge: Any | None = None,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self.vessel_id = vessel_id
        self.bridge = bridge
        self.current_level = EmergencyLevel.GREEN.value
        self.incident_count = 0
        self.incident_history: list[Incident] = []
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self._monitoring_multiplier = 1.0  # Increased during emergencies

    def assess(
        self,
        vessel_state: dict,
        trust_scores: dict,
        sensor_status: dict,
    ) -> EmergencyAssessment:
        """Evaluate current vessel state and determine emergency level.

        Runs all detectors and aggregates findings to determine the
        highest emergency level.

        Args:
            vessel_state: Dict with vessel state information (safety_state,
                last_comm_time, mission_start, expected_duration, etc.)
            trust_scores: Dict mapping subsystem names to trust scores (0.0-1.0).
            sensor_status: Dict with sensor readings and health info.

        Returns:
            EmergencyAssessment with current level, incidents, and actions.
        """
        from .detectors import (
            CommunicationLossDetector,
            MissionTimeoutDetector,
            SafetyViolationDetector,
            SensorFailureDetector,
            TrustCollapseDetector,
        )

        now = time.time()
        previous_level = self.current_level
        all_incidents: list[Incident] = []

        # ── Run all detectors ─────────────────────────────────────
        # Sensor failure
        sensor_detector = SensorFailureDetector()
        incidents = sensor_detector.detect(sensor_status, self.thresholds)
        all_incidents.extend(incidents)

        # Trust collapse
        trust_detector = TrustCollapseDetector()
        trust_history = [
            {subsystem: score}
            for subsystem, score in trust_scores.items()
        ]
        incidents = trust_detector.detect(trust_scores, trust_history)
        all_incidents.extend(incidents)

        # Safety violation
        safety_detector = SafetyViolationDetector()
        safety_state = vessel_state.get("safety_state", {})
        incidents = safety_detector.detect(safety_state)
        all_incidents.extend(incidents)

        # Communication loss
        comm_detector = CommunicationLossDetector()
        last_comm = vessel_state.get("last_comm_time", now)
        incidents = comm_detector.detect(last_comm, self.thresholds)
        all_incidents.extend(incidents)

        # Mission timeout
        mission_detector = MissionTimeoutDetector()
        mission_start = vessel_state.get("mission_start", now)
        expected_duration = vessel_state.get("expected_duration", float("inf"))
        incidents = mission_detector.detect(mission_start, expected_duration)
        all_incidents.extend(incidents)

        # ── Determine highest emergency level ─────────────────────
        level_order = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}
        new_level = EmergencyLevel.GREEN.value

        if all_incidents:
            for incident in all_incidents:
                if level_order.get(incident.level, 0) > level_order.get(new_level, 0):
                    new_level = incident.level

        # Store incidents and update state
        self.current_level = new_level
        actions_taken: list[str] = []

        for incident in all_incidents:
            self.incident_count += 1
            incident.vessel_state = vessel_state
            incident.trust_scores = trust_scores
            self.incident_history.append(incident)
            actions_taken.extend(self._auto_actions_for_level(incident.level))

        # Update monitoring frequency
        if new_level in ("YELLOW", "ORANGE"):
            self._monitoring_multiplier = 2.0
        elif new_level == "RED":
            self._monitoring_multiplier = 4.0
        else:
            self._monitoring_multiplier = 1.0

        reason = self._build_assessment_reason(all_incidents, new_level, previous_level)

        return EmergencyAssessment(
            vessel_id=self.vessel_id,
            timestamp=now,
            previous_level=previous_level,
            current_level=new_level,
            incidents_detected=all_incidents,
            actions_taken=actions_taken,
            reason=reason,
            trust_snapshot=dict(trust_scores),
        )

    def escalate(self, incident: Incident) -> EscalationResult:
        """Escalate emergency: create GitHub Issue, notify fleet, reduce trust.

        Performs automated escalation actions based on incident level:
        - ORANGE: Alert captain, reduce autonomy level, prepare contingency
        - RED: Halt operations, create GitHub Issue, reduce trust, notify fleet

        Args:
            incident: The Incident to escalate.

        Returns:
            EscalationResult with details of actions taken.
        """
        from .response import EmergencyResponder

        responder = EmergencyResponder(vessel_id=self.vessel_id, bridge=self.bridge)
        result = EscalationResult(
            incident_id=incident.id,
            level=incident.level,
        )

        if incident.level == "RED":
            red_result = responder.respond_red(incident)
            result.issue_created = red_result.get("issue_created", False)
            result.issue_number = red_result.get("issue_number", 0)
            result.trust_reduced = red_result.get("trust_reduced", False)
            result.fleet_notified = red_result.get("fleet_notified", False)
            result.commit_hash = red_result.get("commit_hash", "")
            result.error = red_result.get("error", "")
            incident.auto_actions_taken = red_result.get("actions", [])

        elif incident.level == "ORANGE":
            orange_result = responder.respond_orange(incident)
            result.trust_reduced = orange_result.get("trust_reduced", False)
            result.fleet_notified = orange_result.get("fleet_notified", False)
            incident.auto_actions_taken = orange_result.get("actions", [])

        elif incident.level == "YELLOW":
            yellow_result = responder.respond_yellow(incident)
            incident.auto_actions_taken = yellow_result.get("actions", [])

        return result

    def deescalate(self, incident_id: str, resolution: str) -> DeescalationResult:
        """De-escalate: close Issue, restore trust, resume operations.

        Finds the incident by ID, marks it as resolved, and performs
        recovery actions.

        Args:
            incident_id: The ID of the incident to de-escalate.
            resolution: Human-readable description of how it was resolved.

        Returns:
            DeescalationResult with details of recovery actions.
        """
        result = DeescalationResult(
            incident_id=incident_id,
            previous_level=self.current_level,
            new_level=self.current_level,
        )

        # Find the incident
        incident = None
        for inc in self.incident_history:
            if inc.id == incident_id:
                incident = inc
                break

        if incident is None:
            result.error = f"Incident {incident_id} not found"
            return result

        # Mark as resolved
        incident.resolution = resolution
        incident.resolved_at = time.time()

        # Determine new level from remaining unresolved incidents
        level_order = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}
        unresolved = [
            inc for inc in self.incident_history
            if inc.resolution is None and inc.id != incident_id
        ]

        if unresolved:
            max_level = max(
                level_order.get(inc.level, 0) for inc in unresolved
            )
            new_level = next(
                (lvl for lvl, ord_val in level_order.items() if ord_val == max_level),
                "GREEN",
            )
        else:
            new_level = "GREEN"

        result.new_level = new_level
        self.current_level = new_level

        # Restore monitoring frequency
        if new_level == "GREEN":
            self._monitoring_multiplier = 1.0
        elif new_level == "YELLOW":
            self._monitoring_multiplier = 2.0

        result.operations_resumed = new_level in ("GREEN", "YELLOW")
        result.trust_restored = incident.level in ("RED", "ORANGE")

        return result

    def get_incident_report(self) -> dict:
        """Generate full incident report for fleet review.

        Returns:
            Dict with vessel info, current level, and all incident history.
        """
        return {
            "vessel_id": self.vessel_id,
            "current_level": self.current_level,
            "incident_count": self.incident_count,
            "monitoring_multiplier": self._monitoring_multiplier,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "incidents": [
                {
                    "id": inc.id,
                    "level": inc.level,
                    "category": inc.category,
                    "description": inc.description,
                    "timestamp": inc.timestamp,
                    "timestamp_iso": datetime.fromtimestamp(
                        inc.timestamp, tz=timezone.utc
                    ).isoformat(),
                    "resolved": inc.resolution is not None,
                    "resolution": inc.resolution,
                    "auto_actions_taken": inc.auto_actions_taken,
                }
                for inc in self.incident_history
            ],
        }

    @property
    def monitoring_multiplier(self) -> float:
        """Current monitoring frequency multiplier."""
        return self._monitoring_multiplier

    def _auto_actions_for_level(self, level: str) -> list[str]:
        """Return list of automatic actions for a given emergency level."""
        actions_map = {
            "GREEN": [],
            "YELLOW": [
                "log_incident",
                "increase_monitoring",
            ],
            "ORANGE": [
                "log_incident",
                "increase_monitoring",
                "alert_captain",
                "reduce_autonomy",
                "prepare_contingency",
            ],
            "RED": [
                "halt_autonomous_ops",
                "set_actuators_safe",
                "create_github_issue",
                "commit_incident",
                "reduce_trust",
                "notify_fleet",
                "start_watchdog",
            ],
        }
        return actions_map.get(level, [])

    def _build_assessment_reason(
        self,
        incidents: list[Incident],
        new_level: str,
        previous_level: str,
    ) -> str:
        """Build human-readable reason string for the assessment."""
        if not incidents:
            return "All systems nominal"
        categories = [inc.category for inc in incidents if inc.level == new_level]
        unique_categories = list(dict.fromkeys(categories))
        direction = "escalated" if new_level != previous_level else "maintained"
        return f"Emergency {direction} to {new_level}: {', '.join(unique_categories)}"
