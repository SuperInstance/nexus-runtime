"""
NEXUS — Distributed Intelligence Platform for Marine Robotics.

A comprehensive framework for autonomous marine systems featuring:
- Bytecode VM for deterministic sensor processing
- Wire protocol for reliable underwater communications
- Trust engine for multi-agent cooperation
- Autonomous Agent Behavior (AAB) framework
- Fleet orchestration and role management
"""

__version__ = "0.2.1"
__author__ = "NEXUS Team"

from nexus.exceptions import (
    NexusError, VMError, VMHaltError, VMStackOverflow, VMStackUnderflow,
    InvalidOpcode, VMMemoryError, VMDivisionByZero,
    WireError, FrameDecodeError, CRCMismatchError, FrameTooLargeError,
    TrustError, InsufficientTrust, AgentNotFoundError,
    NodeError, InvalidTransitionError, ServiceNotFoundError,
    SafetyViolationError,
)
