"""NEXUS Orchestrator — System status aggregation.

Collects status from all subsystems (trust, safety, heartbeat, bridge,
skill system, emergency protocol) and presents a unified SystemStatus view.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.nexus_orchestrator.orchestrator import NexusOrchestrator


@dataclass
class SystemStatus:
    """Comprehensive system status snapshot.

    Aggregated from all NEXUS subsystems into a single view for
    telemetry, git commits, and fleet coordination.
    """

    vessel_id: str = ""
    uptime_seconds: float = 0.0
    trust_scores: dict[str, float] = field(default_factory=dict)
    autonomy_level: int = 0
    safety_status: str = "GREEN"  # GREEN, YELLOW, ORANGE, RED
    loaded_skills: list[str] = field(default_factory=list)
    pending_missions: int = 0
    completed_missions: int = 0
    last_heartbeat: float = 0.0
    bridge_connected: bool = False
    total_trust_events: int = 0
    bytecode_deployment_count: int = 0
    emergency_incident_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)


class StatusAggregator:
    """Aggregates status from all subsystems.

    Pulls current state from the orchestrator's subsystems and produces
    a unified SystemStatus snapshot.
    """

    def collect(self, orchestrator: NexusOrchestrator) -> SystemStatus:
        """Collect status from all orchestrator subsystems.

        Args:
            orchestrator: The running NexusOrchestrator instance.

        Returns:
            SystemStatus snapshot.
        """
        now = time.time()

        # Trust scores
        trust_scores: dict[str, float] = {}
        total_events = 0
        if orchestrator.trust_engine is not None:
            all_subsystems = orchestrator.trust_engine.get_all_scores()
            for name, st in all_subsystems.items():
                trust_scores[name] = st.trust_score
                total_events += st.total_windows

        # Autonomy level = minimum across subsystems
        autonomy_level = 0
        if orchestrator.trust_engine is not None:
            levels = [
                orchestrator.trust_engine.get_autonomy_level(name)
                for name in trust_scores
            ]
            if levels:
                autonomy_level = min(levels)

        # Safety status from emergency protocol
        safety_status = "GREEN"
        if orchestrator.emergency_protocol is not None:
            safety_status = orchestrator.emergency_protocol.current_level

        # Loaded skills
        loaded_skills: list[str] = []
        if orchestrator.skill_registry is not None:
            loaded_skills = orchestrator.skill_registry.list_names()

        # Heartbeat info
        pending_missions = 0
        completed_missions = 0
        last_heartbeat = 0.0
        uptime = 0.0
        if orchestrator.heartbeat is not None:
            hb_status = orchestrator.heartbeat.get_status()
            pending_missions = hb_status.get("pending_missions", 0)
            last_heartbeat = now  # approximate
            uptime = now - orchestrator._start_time
            vessel_state = orchestrator.heartbeat.vessel_state
            completed_missions = vessel_state.mission_count

        # Bridge
        bridge_connected = False
        if orchestrator.bridge is not None:
            try:
                bridge_status = orchestrator.bridge.get_status()
                bridge_connected = bridge_status.connected
            except Exception:
                bridge_connected = False

        return SystemStatus(
            vessel_id=orchestrator.vessel_id,
            uptime_seconds=round(uptime, 2),
            trust_scores={k: round(v, 6) for k, v in trust_scores.items()},
            autonomy_level=autonomy_level,
            safety_status=safety_status,
            loaded_skills=loaded_skills,
            pending_missions=pending_missions,
            completed_missions=completed_missions,
            last_heartbeat=last_heartbeat,
            bridge_connected=bridge_connected,
            total_trust_events=total_events,
            bytecode_deployment_count=orchestrator._deployment_count,
            emergency_incident_count=(
                orchestrator.emergency_protocol.incident_count
                if orchestrator.emergency_protocol is not None
                else 0
            ),
        )

    def to_json(self, status: SystemStatus) -> str:
        """Serialize SystemStatus to a JSON string.

        Args:
            status: The SystemStatus to serialize.

        Returns:
            JSON string.
        """
        return json.dumps(status.to_dict(), indent=2)

    def to_git_commit_message(self, status: SystemStatus) -> str:
        """Generate a git commit message from system status.

        Format:
            STATUS: vessel=<id> trust=avg:0.55 level=L2
                    safety=GREEN missions=3/1 skills=2 uptime=3600s

        Args:
            status: The SystemStatus to format.

        Returns:
            Formatted commit message string.
        """
        avg_trust = 0.0
        if status.trust_scores:
            avg_trust = sum(status.trust_scores.values()) / len(status.trust_scores)

        lines = [
            f"STATUS: vessel={status.vessel_id}",
            f"  trust_avg={avg_trust:.4f} level=L{status.autonomy_level}",
            f"  safety={status.safety_status}",
            f"  missions={status.completed_missions} done, {status.pending_missions} pending",
            f"  skills={len(status.loaded_skills)} loaded",
            f"  deployments={status.bytecode_deployment_count}",
            f"  incidents={status.emergency_incident_count}",
            f"  uptime={status.uptime_seconds:.0f}s",
            f"  bridge={'connected' if status.bridge_connected else 'disconnected'}",
        ]
        return "\n".join(lines)
