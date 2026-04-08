"""NEXUS exception hierarchy for structured error handling across all modules."""

from __future__ import annotations


class NexusError(Exception):
    """Base exception for all NEXUS errors."""
    pass


# --- VM Errors ---

class VMError(NexusError):
    """Base for VM-related errors."""
    pass

class VMHaltError(VMError):
    """VM was halted unexpectedly."""
    pass

class VMStackOverflow(VMError):
    """Stack overflow in the VM."""
    pass

class VMStackUnderflow(VMError):
    """Stack underflow in the VM."""
    pass

class InvalidOpcode(VMError):
    """Unknown or invalid opcode encountered."""
    pass

class VMMemoryError(VMError):
    """Memory access out of bounds."""
    pass

class VMDivisionByZero(VMError):
    """Division by zero in VM."""
    pass


# --- Wire Protocol Errors ---

class WireError(NexusError):
    """Base for wire protocol errors."""
    pass

class FrameDecodeError(WireError):
    """Error decoding a wire frame."""
    pass

class CRCMismatchError(WireError):
    """CRC check failed on a frame."""
    pass

class FrameTooLargeError(WireError):
    """Frame exceeds maximum size."""
    pass


# --- Trust Engine Errors ---

class TrustError(NexusError):
    """Base for trust engine errors."""
    pass

class InsufficientTrust(TrustError):
    """Trust score too low for requested operation."""
    pass

class AgentNotFoundError(TrustError):
    """Agent not found in trust network."""
    pass


# --- Core / Node Errors ---

class NodeError(NexusError):
    """Base for node lifecycle errors."""
    pass

class InvalidTransitionError(NodeError):
    """Invalid state transition attempted."""
    pass

class ServiceNotFoundError(NexusError):
    """Service not found in registry."""
    pass


# --- Safety Errors ---

class SafetyViolationError(NexusError):
    """A safety constraint was violated."""
    pass
