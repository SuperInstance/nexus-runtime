"""
NEXUS Runtime Verification Module

Provides safety invariant checking, design-by-contract verification,
runtime monitoring, temporal logic checking, and watchdog timer management.
"""

from jetson.runtime_verification.invariants import (
    Invariant,
    Violation,
    InvariantChecker,
)
from jetson.runtime_verification.contracts import (
    Contract,
    ContractResult,
    ContractChecker,
)
from jetson.runtime_verification.monitor import (
    MonitorEvent,
    MonitorRule,
    RuntimeMonitor,
    Alert,
)
from jetson.runtime_verification.temporal import (
    TemporalFormula,
    TemporalLogicChecker,
)
from jetson.runtime_verification.watchdog import (
    WatchdogConfig,
    WatchdogState,
    WatchdogManager,
)

__all__ = [
    # Invariants
    "Invariant",
    "Violation",
    "InvariantChecker",
    # Contracts
    "Contract",
    "ContractResult",
    "ContractChecker",
    # Monitor
    "MonitorEvent",
    "MonitorRule",
    "RuntimeMonitor",
    "Alert",
    # Temporal
    "TemporalFormula",
    "TemporalLogicChecker",
    # Watchdog
    "WatchdogConfig",
    "WatchdogState",
    "WatchdogManager",
]
