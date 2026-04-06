"""Fleet Coordination — Phase 5 Round 6 of NEXUS.

Provides fleet state management, task orchestration, distributed consensus,
resource management, and fleet communication patterns for autonomous
surface vessel operations.
"""

from .fleet_manager import VesselStatus, FleetState, FleetManager
from .task_orchestration import FleetTask, TaskOrchestrator
from .consensus import Proposal, Vote, ConsensusResult, ConsensusProtocol
from .resource_mgmt import FleetResource, ResourceAllocation, FleetResourceManager
from .communication import (
    FleetMessage, BroadcastResult, FleetCommunication
)

__all__ = [
    "VesselStatus", "FleetState", "FleetManager",
    "FleetTask", "TaskOrchestrator",
    "Proposal", "Vote", "ConsensusResult", "ConsensusProtocol",
    "FleetResource", "ResourceAllocation", "FleetResourceManager",
    "FleetMessage", "BroadcastResult", "FleetCommunication",
]
