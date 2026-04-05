"""NEXUS git-agent bridge — Python library for NEXUS ↔ git-agent coordination.

Translates between NEXUS Wire Protocol operations and git-agent-style
coordination (git operations, GitHub API). Runs directly on the Jetson
alongside the NEXUS Python SDK, keeping the edge stack pure Python.

Modules:
    NexusBridge         — Main bridge coordinator
    BytecodeDeployer    — Bytecode validation and deployment
    TelemetryIngester   — Sensor data batching to git commits
    TrustSync           — INCREMENTS trust to git audit trail
    EquipmentManifest   — HAL capabilities declaration

Usage:
    from nexus_bridge import NexusBridge

    bridge = NexusBridge(
        vessel_id="vessel-001",
        repo_path="/path/to/vessel/repo",
    )
    result = bridge.deploy_bytecode(
        bytecode=compiled_bytecode,
        source_reflex="heading-hold-v2",
        provenance={"author": "agent-flux"},
    )
"""

from .bridge import (
    NexusBridge,
    BridgeStatus,
    DeployResult,
    SafetyResult,
    MissionResult,
)
from .bytecode_deployer import (
    BytecodeDeployer,
    ValidationResult,
    ValidationReport,
)
from .telemetry_ingester import (
    TelemetryIngester,
    TelemetryResult,
    TelemetryBatch,
    SensorReading,
)
from .trust_sync import (
    TrustSync,
    TrustResult,
    TrustRecord,
)
from .equipment_manifest import (
    EquipmentManifest,
    SensorEntry,
    ActuatorEntry,
)

__all__ = [
    # Main bridge
    "NexusBridge",
    "BridgeStatus",
    "DeployResult",
    "SafetyResult",
    "MissionResult",
    # Bytecode deployment
    "BytecodeDeployer",
    "ValidationResult",
    "ValidationReport",
    # Telemetry
    "TelemetryIngester",
    "TelemetryResult",
    "TelemetryBatch",
    "SensorReading",
    # Trust
    "TrustSync",
    "TrustResult",
    "TrustRecord",
    # Equipment manifest
    "EquipmentManifest",
    "SensorEntry",
    "ActuatorEntry",
]

__version__ = "0.1.0"
