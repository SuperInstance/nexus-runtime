"""NEXUS Orchestrator — Integration harness wiring all modules together.

Central coordinator that connects:
  - Edge Heartbeat (mission execution cycle)
  - git-agent Bridge (fleet coordination)
  - Rosetta Stone (intent translation)
  - Skill System (behavior loading)
  - Safety Validator (bytecode verification)
  - Trust Engine (autonomy gating)
  - Emergency Protocol (safety monitoring)

Usage:
    from agent.nexus_orchestrator import NexusOrchestrator

    orch = NexusOrchestrator(config_path="/opt/nexus-runtime/configs/vessel.json")
    orch.start()
    result = orch.process_natural_language_command("read sensor 3")
    print(result)
    orch.stop()
"""

from agent.nexus_orchestrator.orchestrator import (
    NexusOrchestrator,
    CommandResult,
    SkillLoadResult,
    EmergencyResult,
    SimulationResult,
    ComparisonResult,
)
from agent.nexus_orchestrator.system_status import (
    SystemStatus,
    StatusAggregator,
)
from agent.nexus_orchestrator.simulation import (
    MissionSimulator,
    SimStep,
)

__all__ = [
    "NexusOrchestrator",
    "CommandResult",
    "SkillLoadResult",
    "EmergencyResult",
    "SimulationResult",
    "ComparisonResult",
    "SystemStatus",
    "StatusAggregator",
    "MissionSimulator",
    "SimStep",
]
