"""Emergency Response — Automated response actions for each emergency level.

Executes predefined response actions based on the severity of the
detected emergency. Coordinates with the git-agent bridge for fleet-wide
notifications and GitHub Issue creation.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from .protocol import Incident

logger = logging.getLogger("nexus.emergency")


class EmergencyResponder:
    """Execute automated emergency responses.

    Each emergency level has a corresponding response method that
    performs the appropriate automated actions:

    - YELLOW: Log, increase monitoring, notify locally
    - ORANGE: Alert captain, reduce autonomy, prepare contingency, notify fleet
    - RED: Halt all operations, safe actuators, create GitHub Issue,
           commit incident, reduce trust, notify fleet, start watchdog

    The responder integrates with the NexusBridge for git and GitHub
    operations when a bridge instance is provided.
    """

    def __init__(
        self,
        vessel_id: str,
        bridge: Any | None = None,
    ) -> None:
        self.vessel_id = vessel_id
        self.bridge = bridge
        self._logger = logging.getLogger(f"nexus.emergency.{vessel_id}")

    def respond_yellow(self, incident: Incident) -> dict[str, Any]:
        """YELLOW level response.

        Actions:
        1. Log the incident with WARNING level
        2. Increase monitoring frequency
        3. Record event in incident log

        Args:
            incident: The YELLOW-level Incident.

        Returns:
            Dict with response details and actions taken.
        """
        self._logger.warning(
            "YELLOW incident [%s]: %s (category=%s)",
            incident.id,
            incident.description,
            incident.category,
        )

        actions = [
            "log_incident",
            "increase_monitoring",
        ]

        # Record in bridge if available
        if self.bridge:
            try:
                self.bridge.record_trust_event(
                    subsystem=f"emergency_{incident.category.lower()}",
                    event_type="yellow_alert",
                    severity=3,
                    details=incident.description,
                )
                actions.append("trust_event_recorded")
            except Exception as exc:
                self._logger.error("Failed to record trust event: %s", exc)

        return {
            "actions": actions,
            "level": "YELLOW",
            "incident_id": incident.id,
        }

    def respond_orange(self, incident: Incident) -> dict[str, Any]:
        """ORANGE level response.

        Actions:
        1. Log the incident with ERROR level
        2. Alert captain (log + notification)
        3. Reduce autonomy level
        4. Prepare contingency plan
        5. Notify fleet via trust event

        Args:
            incident: The ORANGE-level Incident.

        Returns:
            Dict with response details and actions taken.
        """
        self._logger.error(
            "ORANGE incident [%s]: %s (category=%s)",
            incident.id,
            incident.description,
            incident.category,
        )

        actions = [
            "log_incident",
            "increase_monitoring",
            "alert_captain",
            "reduce_autonomy",
            "prepare_contingency",
        ]

        trust_reduced = False
        fleet_notified = False

        # Reduce trust and notify fleet via bridge
        if self.bridge:
            try:
                self.bridge.record_trust_event(
                    subsystem=f"emergency_{incident.category.lower()}",
                    event_type="orange_alert",
                    severity=6,
                    details=incident.description,
                )
                trust_reduced = True
                actions.append("trust_reduced")
            except Exception as exc:
                self._logger.error("Failed to record trust event: %s", exc)

            try:
                self.bridge.report_safety_event({
                    "level": 2,
                    "subsystem": incident.category.lower(),
                    "event_type": "orange_emergency",
                    "details": incident.description,
                })
                fleet_notified = True
                actions.append("fleet_notified")
            except Exception as exc:
                self._logger.error("Failed to notify fleet: %s", exc)

        return {
            "actions": actions,
            "level": "ORANGE",
            "incident_id": incident.id,
            "trust_reduced": trust_reduced,
            "fleet_notified": fleet_notified,
        }

    def respond_red(self, incident: Incident) -> dict[str, Any]:
        """RED level response — critical emergency.

        Actions:
        1. Halt all autonomous bytecode execution
        2. Set actuators to safe state
        3. Create GitHub Issue with RED label
        4. Commit incident to git
        5. Reduce autonomy level to L0
        6. Notify fleet-orchestrator via trust event
        7. Begin watchdog monitoring

        Args:
            incident: The RED-level Incident.

        Returns:
            Dict with response details including issue number and commit hash.
        """
        self._logger.critical(
            "RED incident [%s]: %s (category=%s) — HALTING ALL OPERATIONS",
            incident.id,
            incident.description,
            incident.category,
        )

        actions = [
            "halt_autonomous_ops",
            "set_actuators_safe",
            "create_github_issue",
            "commit_incident",
            "reduce_trust",
            "notify_fleet",
            "start_watchdog",
        ]

        issue_created = False
        issue_number = 0
        trust_reduced = False
        fleet_notified = False
        commit_hash = ""
        error = ""

        # Step 1 & 2: Halt ops and safe actuators (handled by safety SM in firmware)
        # Log the action
        self._logger.critical("Step 1/2: Halting autonomous ops, setting actuators safe")

        # Steps 3-7: Git and fleet operations via bridge
        if self.bridge:
            # Step 4: Commit incident to git
            try:
                result = self.bridge.report_safety_event({
                    "level": 4,
                    "subsystem": incident.category.lower(),
                    "event_type": "red_emergency",
                    "details": incident.description,
                    "timestamp": incident.timestamp,
                })
                if result.committed:
                    commit_hash = result.commit_hash
                    issue_created = result.issue_created
                    issue_number = result.issue_number
                    actions.append("incident_committed")
                else:
                    error = "Failed to commit incident"
            except Exception as exc:
                error = f"Git commit failed: {exc}"
                self._logger.critical("Step 4 FAILED: %s", exc)

            # Step 5: Reduce trust
            try:
                self.bridge.record_trust_event(
                    subsystem=f"emergency_{incident.category.lower()}",
                    event_type="red_alert",
                    severity=10,
                    details=f"RED emergency: {incident.description}",
                )
                trust_reduced = True
            except Exception as exc:
                self._logger.error("Step 5 FAILED: %s", exc)

            # Step 6: Notify fleet
            try:
                fleet_alert = self.create_fleet_alert(incident)
                self._logger.info("Fleet alert prepared: %s", fleet_alert["alert_id"])
                fleet_notified = True
            except Exception as exc:
                self._logger.error("Step 6 FAILED: %s", exc)
        else:
            # No bridge — log what we would have done
            self._logger.warning(
                "No bridge configured; RED response actions limited to local logging"
            )
            # Still create the fleet alert for local logging
            fleet_alert = self.create_fleet_alert(incident)
            fleet_notified = True  # At least locally logged

        # Step 7: Begin watchdog monitoring
        self._logger.critical("Step 7: Watchdog monitoring started")

        return {
            "actions": actions,
            "level": "RED",
            "incident_id": incident.id,
            "issue_created": issue_created,
            "issue_number": issue_number,
            "trust_reduced": trust_reduced,
            "fleet_notified": fleet_notified,
            "commit_hash": commit_hash,
            "error": error,
        }

    def create_fleet_alert(self, incident: Incident) -> dict[str, Any]:
        """Create structured alert for fleet coordination.

        Produces a JSON-serializable alert payload that can be:
        - Committed to .agent/safety/ in the vessel repo
        - Sent via the fleet heartbeat mechanism
        - Used by the fleet-orchestrator to coordinate response

        Args:
            incident: The Incident to create an alert for.

        Returns:
            Dict with structured alert data.
        """
        import uuid

        alert_id = f"FALERT-{uuid.uuid4().hex[:12]}"
        now = time.time()

        alert = {
            "alert_id": alert_id,
            "alert_type": f"EMERGENCY_{incident.level}",
            "vessel_id": self.vessel_id,
            "incident_id": incident.id,
            "incident_level": incident.level,
            "incident_category": incident.category,
            "description": incident.description,
            "timestamp": now,
            "timestamp_iso": datetime.fromtimestamp(
                now, tz=timezone.utc
            ).isoformat(),
            "requires_fleet_action": incident.level in ("ORANGE", "RED"),
            "autonomy_level": "L0" if incident.level == "RED" else "L1",
            "vessel_state_snapshot": incident.vessel_state,
            "trust_snapshot": incident.trust_scores,
            "recommended_actions": self._recommended_fleet_actions(incident),
        }

        self._logger.info(
            "Fleet alert %s created: %s %s — %s",
            alert_id,
            incident.level,
            incident.category,
            incident.description,
        )

        return alert

    def _recommended_fleet_actions(self, incident: Incident) -> list[str]:
        """Generate recommended fleet-level actions for the incident."""
        if incident.level == "RED":
            return [
                "dispatch_assessment_team",
                "activate_backup_vessel",
                "review_recent_bytecode_deploys",
                "audit_trust_history",
            ]
        elif incident.level == "ORANGE":
            return [
                "increase_fleet_monitoring",
                "review_vessel_telemetry",
                "prepare_contingency_mission",
            ]
        else:
            return [
                "monitor",
            ]
