"""NEXUS Edge Heartbeat — 5-phase vessel heartbeat cycle.

Connects NEXUS edge runtime to git-agent coordination.
Each heartbeat cycle: PERCEIVE -> THINK -> ACT -> REMEMBER -> NOTIFY

Usage:
    from agent.edge_heartbeat import EdgeHeartbeat, HeartbeatConfig
    from agent.edge_heartbeat.config import load_config

    hb = EdgeHeartbeat(config_path="/path/to/vessel_config.json")
    result = hb.run_once()
"""

from agent.edge_heartbeat.heartbeat import (
    EdgeHeartbeat,
    HeartbeatPhase,
    HeartbeatResult,
    PhaseResult,
)
from agent.edge_heartbeat.mission_runner import (
    Mission,
    MissionResult,
    MissionRunner,
    MissionStatus,
    MissionType,
)
from agent.edge_heartbeat.vessel_state import (
    VesselState,
    VesselStateManager,
)

__all__ = [
    # Core
    "EdgeHeartbeat",
    "HeartbeatResult",
    "HeartbeatPhase",
    "PhaseResult",
    # Config
    "HeartbeatConfig",
    # Vessel state
    "VesselState",
    "VesselStateManager",
    # Missions
    "Mission",
    "MissionResult",
    "MissionRunner",
    "MissionStatus",
    "MissionType",
]
