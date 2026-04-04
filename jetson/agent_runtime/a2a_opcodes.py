"""NEXUS Agent Runtime - A2A opcode definitions and interpreter.

29 A2A opcodes for agent-to-agent communication.
All A2A opcodes are NOP on ESP32 firmware (backward compatible).
Interpreted on Jetson agent runtime.

Categories:
  Intent (0x20-0x29): 10 opcodes
  Agent Communication (0x30-0x35): 6 opcodes
  Capability Negotiation (0x40-0x45): 6 opcodes
  Safety Augmentation (0x50-0x56): 7 opcodes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class A2ACategory(IntEnum):
    """A2A opcode categories."""
    INTENT = 1
    COMMUNICATION = 2
    CAPABILITY = 3
    SAFETY = 4


# ===================================================================
# Complete 29 A2A Opcode Definitions
# ===================================================================

# Intent opcodes (0x20-0x29) — 10 opcodes
DECLARE_INTENT = 0x20
ASSERT_GOAL = 0x21
VERIFY_OUTCOME = 0x22
EXPLAIN_FAILURE = 0x23
SET_PRIORITY = 0x24
REQUEST_RESOURCE = 0x25
RELEASE_RESOURCE = 0x26
CANCEL_INTENT = 0x27
UPDATE_GOAL = 0x28
QUERY_STATUS = 0x29

# Agent Communication (0x30-0x35) — 6 opcodes
TELL = 0x30
ASK = 0x31
DELEGATE = 0x32
REPORT_STATUS = 0x33
REQUEST_OVERRIDE = 0x34
BROADCAST = 0x35

# Capability Negotiation (0x40-0x45) — 6 opcodes
REQUIRE_CAPABILITY = 0x40
DECLARE_SENSOR_NEED = 0x41
DECLARE_ACTUATOR_USE = 0x42
CHECK_AVAILABILITY = 0x43
RESERVE_RESOURCE = 0x44
REPORT_CAPABILITY = 0x45

# Safety Augmentation (0x50-0x56) — 7 opcodes
TRUST_CHECK = 0x50
AUTONOMY_LEVEL_ASSERT = 0x51
SAFE_BOUNDARY = 0x52
RATE_LIMIT = 0x53
EMERGENCY_CLAIM = 0x54
RELEASE_CLAIM = 0x55
VERIFY_AUTHORITY = 0x56


# ===================================================================
# Full opcode registry (29 entries)
# ===================================================================

A2A_OPCODES: dict[int, dict] = {
    # Intent opcodes (0x20-0x29)
    0x20: {"name": "DECLARE_INTENT", "category": A2ACategory.INTENT,
           "desc": "Declare high-level intent for code block"},
    0x21: {"name": "ASSERT_GOAL", "category": A2ACategory.INTENT,
           "desc": "Assert a goal that should be achieved"},
    0x22: {"name": "VERIFY_OUTCOME", "category": A2ACategory.INTENT,
           "desc": "Verify expected outcome was achieved"},
    0x23: {"name": "EXPLAIN_FAILURE", "category": A2ACategory.INTENT,
           "desc": "Provide explanation for a failure"},
    0x24: {"name": "SET_PRIORITY", "category": A2ACategory.INTENT,
           "desc": "Set execution priority level"},
    0x25: {"name": "REQUEST_RESOURCE", "category": A2ACategory.INTENT,
           "desc": "Request a shared resource"},
    0x26: {"name": "RELEASE_RESOURCE", "category": A2ACategory.INTENT,
           "desc": "Release a previously acquired resource"},
    0x27: {"name": "CANCEL_INTENT", "category": A2ACategory.INTENT,
           "desc": "Cancel a previously declared intent"},
    0x28: {"name": "UPDATE_GOAL", "category": A2ACategory.INTENT,
           "desc": "Update parameters of an active goal"},
    0x29: {"name": "QUERY_STATUS", "category": A2ACategory.INTENT,
           "desc": "Query status of an intent or goal"},
    # Agent Communication (0x30-0x35)
    0x30: {"name": "TELL", "category": A2ACategory.COMMUNICATION,
           "desc": "Send information to another agent"},
    0x31: {"name": "ASK", "category": A2ACategory.COMMUNICATION,
           "desc": "Query another agent for information"},
    0x32: {"name": "DELEGATE", "category": A2ACategory.COMMUNICATION,
           "desc": "Delegate a task to another agent"},
    0x33: {"name": "REPORT_STATUS", "category": A2ACategory.COMMUNICATION,
           "desc": "Report current status to supervisor"},
    0x34: {"name": "REQUEST_OVERRIDE", "category": A2ACategory.COMMUNICATION,
           "desc": "Request manual override from human"},
    0x35: {"name": "BROADCAST", "category": A2ACategory.COMMUNICATION,
           "desc": "Broadcast information to all agents"},
    # Capability Negotiation (0x40-0x45)
    0x40: {"name": "REQUIRE_CAPABILITY", "category": A2ACategory.CAPABILITY,
           "desc": "Assert required capability to execute"},
    0x41: {"name": "DECLARE_SENSOR_NEED", "category": A2ACategory.CAPABILITY,
           "desc": "Declare sensor requirements"},
    0x42: {"name": "DECLARE_ACTUATOR_USE", "category": A2ACategory.CAPABILITY,
           "desc": "Declare actuator usage intent"},
    0x43: {"name": "CHECK_AVAILABILITY", "category": A2ACategory.CAPABILITY,
           "desc": "Check resource availability"},
    0x44: {"name": "RESERVE_RESOURCE", "category": A2ACategory.CAPABILITY,
           "desc": "Reserve a specific resource"},
    0x45: {"name": "REPORT_CAPABILITY", "category": A2ACategory.CAPABILITY,
           "desc": "Report current capability status"},
    # Safety Augmentation (0x50-0x56)
    0x50: {"name": "TRUST_CHECK", "category": A2ACategory.SAFETY,
           "desc": "Check trust level before proceeding"},
    0x51: {"name": "AUTONOMY_LEVEL_ASSERT", "category": A2ACategory.SAFETY,
           "desc": "Assert required autonomy level"},
    0x52: {"name": "SAFE_BOUNDARY", "category": A2ACategory.SAFETY,
           "desc": "Define a safe operating boundary"},
    0x53: {"name": "RATE_LIMIT", "category": A2ACategory.SAFETY,
           "desc": "Rate-limit a recurring operation"},
    0x54: {"name": "EMERGENCY_CLAIM", "category": A2ACategory.SAFETY,
           "desc": "Claim emergency control of subsystem"},
    0x55: {"name": "RELEASE_CLAIM", "category": A2ACategory.SAFETY,
           "desc": "Release emergency claim"},
    0x56: {"name": "VERIFY_AUTHORITY", "category": A2ACategory.SAFETY,
           "desc": "Verify authority to perform action"},
}

A2A_NAME_TO_OPCODE: dict[str, int] = {
    info["name"]: opcode for opcode, info in A2A_OPCODES.items()
}


@dataclass
class ActionResult:
    """Result of interpreting an A2A opcode."""

    opcode: int
    success: bool = True
    message: str = ""
    data: dict = field(default_factory=dict)
    blocked: bool = False


@dataclass
class AgentContext:
    """Runtime context for A2A opcode interpretation."""

    agent_id: str = "agent-0"
    trust_level: float = 0.0
    autonomy_level: int = 0
    capabilities: set[str] = field(default_factory=set)
    resources: dict[str, bool] = field(default_factory=dict)
    claimed_subsystems: set[str] = field(default_factory=set)
    log: list[str] = field(default_factory=list)


def interpret_a2a_opcode(
    opcode: int,
    operands: tuple,
    metadata: list,
    agent_context: AgentContext,
) -> ActionResult:
    """Interpret an A2A opcode."""
    op1, op2 = operands if len(operands) >= 2 else (operands[0] if operands else 0, 0)
    info = A2A_OPCODES.get(opcode)

    if info is None:
        return ActionResult(opcode=opcode, success=False, message=f"Unknown A2A opcode: 0x{opcode:02X}")

    name = info["name"]
    category = info["category"]

    if category == A2ACategory.INTENT:
        return _interpret_intent(name, opcode, op1, op2, metadata, agent_context)
    elif category == A2ACategory.COMMUNICATION:
        return _interpret_communication(name, opcode, op1, op2, metadata, agent_context)
    elif category == A2ACategory.CAPABILITY:
        return _interpret_capability(name, opcode, op1, op2, metadata, agent_context)
    elif category == A2ACategory.SAFETY:
        return _interpret_safety(name, opcode, op1, op2, metadata, agent_context)

    return ActionResult(opcode=opcode, success=False, message=f"Unhandled category for {name}")


def _interpret_intent(name, opcode, op1, op2, metadata, ctx):
    if name == "DECLARE_INTENT":
        ctx.log.append(f"Intent declared: op1={op1}")
        return ActionResult(opcode=opcode, success=True, message="Intent declared")
    elif name == "ASSERT_GOAL":
        ctx.log.append(f"Goal asserted: op1={op1}")
        return ActionResult(opcode=opcode, success=True, message="Goal asserted")
    elif name == "VERIFY_OUTCOME":
        success = bool(op1)
        ctx.log.append(f"Outcome verified: {'pass' if success else 'fail'}")
        return ActionResult(opcode=opcode, success=success,
                          message="Outcome matches" if success else "Outcome mismatch")
    elif name == "EXPLAIN_FAILURE":
        ctx.log.append(f"Failure explanation recorded: op1={op1}")
        return ActionResult(opcode=opcode, success=True, message="Failure explanation recorded")
    elif name == "SET_PRIORITY":
        ctx.log.append(f"Priority set to: {op1}")
        return ActionResult(opcode=opcode, success=True,
                          message=f"Priority set to {op1}", data={"priority": op1})
    elif name == "REQUEST_RESOURCE":
        rid = str(op1)
        if rid in ctx.resources and ctx.resources[rid]:
            return ActionResult(opcode=opcode, success=True, message=f"Resource {rid} granted")
        return ActionResult(opcode=opcode, success=False, blocked=True,
                          message=f"Resource {rid} unavailable")
    elif name == "RELEASE_RESOURCE":
        rid = str(op1)
        ctx.resources.pop(rid, None)
        return ActionResult(opcode=opcode, success=True, message=f"Resource {rid} released")
    elif name == "CANCEL_INTENT":
        ctx.log.append(f"Intent cancelled: op1={op1}")
        return ActionResult(opcode=opcode, success=True, message="Intent cancelled")
    elif name == "UPDATE_GOAL":
        ctx.log.append(f"Goal updated: op1={op1}")
        return ActionResult(opcode=opcode, success=True, message="Goal updated")
    elif name == "QUERY_STATUS":
        return ActionResult(opcode=opcode, success=True, message="Status queried")
    return ActionResult(opcode=opcode, success=False, message=f"Unhandled intent: {name}")


def _interpret_communication(name, opcode, op1, op2, metadata, ctx):
    if name == "TELL":
        ctx.log.append(f"TELL: target={op1}")
        return ActionResult(opcode=opcode, success=True, message="Information sent")
    elif name == "ASK":
        ctx.log.append(f"ASK: target={op1}")
        return ActionResult(opcode=opcode, success=True, message="Query sent")
    elif name == "DELEGATE":
        ctx.log.append(f"DELEGATE: task={op1} to agent={op2}")
        return ActionResult(opcode=opcode, success=True, message="Task delegated")
    elif name == "REPORT_STATUS":
        ctx.log.append(f"STATUS: level={op1}")
        return ActionResult(opcode=opcode, success=True, message=f"Status reported (level {op1})")
    elif name == "REQUEST_OVERRIDE":
        ctx.log.append(f"OVERRIDE requested: reason={op1}")
        return ActionResult(opcode=opcode, success=True, blocked=True, message="Override requested")
    elif name == "BROADCAST":
        ctx.log.append(f"BROADCAST: msg_id={op1}")
        return ActionResult(opcode=opcode, success=True, message="Broadcast sent")
    return ActionResult(opcode=opcode, success=False, message=f"Unhandled comm: {name}")


def _interpret_capability(name, opcode, op1, op2, metadata, ctx):
    if name == "REQUIRE_CAPABILITY":
        ctx.log.append(f"REQUIRE_CAPABILITY: cap_id={op1}")
        return ActionResult(opcode=opcode, success=True, message="Capability check recorded")
    elif name == "DECLARE_SENSOR_NEED":
        ctx.log.append(f"SENSOR_NEED: sensor_id={op1}")
        return ActionResult(opcode=opcode, success=True, message=f"Sensor need declared: {op1}")
    elif name == "DECLARE_ACTUATOR_USE":
        ctx.log.append(f"ACTUATOR_USE: actuator_id={op1}")
        return ActionResult(opcode=opcode, success=True, message=f"Actuator use declared: {op1}")
    elif name == "CHECK_AVAILABILITY":
        rid = str(op1)
        available = ctx.resources.get(rid, False)
        return ActionResult(opcode=opcode, success=available,
                          message=f"Resource {rid}: {'available' if available else 'unavailable'}")
    elif name == "RESERVE_RESOURCE":
        rid = str(op1)
        if rid in ctx.resources and ctx.resources[rid]:
            return ActionResult(opcode=opcode, success=False, message=f"Resource {rid} already reserved")
        ctx.resources[rid] = True
        return ActionResult(opcode=opcode, success=True, message=f"Resource {rid} reserved")
    elif name == "REPORT_CAPABILITY":
        ctx.log.append(f"CAPABILITY_REPORT: cap_id={op1}")
        return ActionResult(opcode=opcode, success=True, message="Capability reported")
    return ActionResult(opcode=opcode, success=False, message=f"Unhandled cap: {name}")


def _interpret_safety(name, opcode, op1, op2, metadata, ctx):
    if name == "TRUST_CHECK":
        required = op1 / 255.0
        sufficient = ctx.trust_level >= required
        if not sufficient:
            return ActionResult(opcode=opcode, success=False, blocked=True,
                              message=f"Trust check failed: {ctx.trust_level:.3f} < {required:.3f}")
        return ActionResult(opcode=opcode, success=True,
                          message=f"Trust check passed: {ctx.trust_level:.3f} >= {required:.3f}")
    elif name == "AUTONOMY_LEVEL_ASSERT":
        required_level = op1
        sufficient = ctx.autonomy_level >= required_level
        if not sufficient:
            return ActionResult(opcode=opcode, success=False, blocked=True,
                              message=f"Autonomy insufficient: {ctx.autonomy_level} < {required_level}")
        return ActionResult(opcode=opcode, success=True,
                          message=f"Autonomy level OK: {ctx.autonomy_level} >= {required_level}")
    elif name == "SAFE_BOUNDARY":
        ctx.log.append(f"SAFE_BOUNDARY: boundary_id={op1}")
        return ActionResult(opcode=opcode, success=True, message="Safe boundary defined")
    elif name == "RATE_LIMIT":
        ctx.log.append(f"RATE_LIMIT: max={op1}/window={op2}")
        return ActionResult(opcode=opcode, success=True, message=f"Rate limit set: {op1} per {op2}ms")
    elif name == "EMERGENCY_CLAIM":
        subsystem = str(op1)
        ctx.claimed_subsystems.add(subsystem)
        ctx.log.append(f"EMERGENCY_CLAIM: {subsystem}")
        return ActionResult(opcode=opcode, success=True, message=f"Emergency claim on {subsystem}")
    elif name == "RELEASE_CLAIM":
        subsystem = str(op1)
        ctx.claimed_subsystems.discard(subsystem)
        ctx.log.append(f"RELEASE_CLAIM: {subsystem}")
        return ActionResult(opcode=opcode, success=True, message=f"Claim released on {subsystem}")
    elif name == "VERIFY_AUTHORITY":
        required_role = op1
        has_authority = ctx.autonomy_level >= required_role
        if not has_authority:
            return ActionResult(opcode=opcode, success=False, blocked=True,
                              message=f"Authority insufficient: {ctx.autonomy_level} < {required_role}")
        return ActionResult(opcode=opcode, success=True, message=f"Authority verified for role {required_role}")
    return ActionResult(opcode=opcode, success=False, message=f"Unhandled safety: {name}")


def get_a2a_opcodes() -> dict[int, str]:
    return {opcode: info["name"] for opcode, info in A2A_OPCODES.items()}


def get_a2a_opcode_count() -> int:
    return len(A2A_OPCODES)


def is_a2a_opcode(opcode: int) -> bool:
    return opcode in A2A_OPCODES


def a2a_opcode_name(opcode: int) -> str | None:
    info = A2A_OPCODES.get(opcode)
    return info["name"] if info else None
