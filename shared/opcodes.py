"""NEXUS Bytecode VM - Complete opcode definitions.

Core opcodes (0x00-0x1F): 32 opcodes for the Reflex VM on ESP32-S3.
A2A opcodes (0x20-0x56): 29 opcodes for agent-to-agent communication.
A2A opcodes are NOP on existing ESP32 firmware (backward compatible).
"""

from __future__ import annotations


# ===================================================================
# Core Opcodes (0x00-0x1F) - 32 opcodes
# ===================================================================

# Stack operations (0x00-0x07)
NOP: int = 0x00
PUSH_I8: int = 0x01
PUSH_I16: int = 0x02
PUSH_F32: int = 0x03
POP: int = 0x04
DUP: int = 0x05
SWAP: int = 0x06
ROT: int = 0x07

# Arithmetic (0x08-0x10)
ADD_F: int = 0x08
SUB_F: int = 0x09
MUL_F: int = 0x0A
DIV_F: int = 0x0B
NEG_F: int = 0x0C
ABS_F: int = 0x0D
MIN_F: int = 0x0E
MAX_F: int = 0x0F
CLAMP_F: int = 0x10

# Comparison (0x11-0x15)
EQ_F: int = 0x11
LT_F: int = 0x12
GT_F: int = 0x13
LTE_F: int = 0x14
GTE_F: int = 0x15

# Logic (0x16-0x19)
AND_B: int = 0x16
OR_B: int = 0x17
XOR_B: int = 0x18
NOT_B: int = 0x19

# I/O (0x1A-0x1C)
READ_PIN: int = 0x1A
WRITE_PIN: int = 0x1B
READ_TIMER_MS: int = 0x1C

# Control flow (0x1D-0x1F)
JUMP: int = 0x1D
JUMP_IF_FALSE: int = 0x1E
JUMP_IF_TRUE: int = 0x1F

# ===================================================================
# A2A Opcodes (0x20-0x56) - 29 opcodes
# All NOP on ESP32 firmware (backward compatible).
# ===================================================================

# Intent opcodes (0x20-0x26)
DECLARE_INTENT: int = 0x20
ASSERT_GOAL: int = 0x21
VERIFY_OUTCOME: int = 0x22
EXPLAIN_FAILURE: int = 0x23

# Agent Communication (0x30-0x34)
TELL: int = 0x30
ASK: int = 0x31
DELEGATE: int = 0x32
REPORT_STATUS: int = 0x33
REQUEST_OVERRIDE: int = 0x34

# Capability Negotiation (0x40-0x44)
REQUIRE_CAPABILITY: int = 0x40
DECLARE_SENSOR_NEED: int = 0x41
DECLARE_ACTUATOR_USE: int = 0x42

# Safety Augmentation (0x50-0x56)
TRUST_CHECK: int = 0x50
AUTONOMY_LEVEL_ASSERT: int = 0x51
SAFE_BOUNDARY: int = 0x52
RATE_LIMIT: int = 0x53

# ===================================================================
# Opcode name registry
# ===================================================================

OPCODE_NAMES: dict[int, str] = {
    # Core
    0x00: "NOP",
    0x01: "PUSH_I8",
    0x02: "PUSH_I16",
    0x03: "PUSH_F32",
    0x04: "POP",
    0x05: "DUP",
    0x06: "SWAP",
    0x07: "ROT",
    0x08: "ADD_F",
    0x09: "SUB_F",
    0x0A: "MUL_F",
    0x0B: "DIV_F",
    0x0C: "NEG_F",
    0x0D: "ABS_F",
    0x0E: "MIN_F",
    0x0F: "MAX_F",
    0x10: "CLAMP_F",
    0x11: "EQ_F",
    0x12: "LT_F",
    0x13: "GT_F",
    0x14: "LTE_F",
    0x15: "GTE_F",
    0x16: "AND_B",
    0x17: "OR_B",
    0x18: "XOR_B",
    0x19: "NOT_B",
    0x1A: "READ_PIN",
    0x1B: "WRITE_PIN",
    0x1C: "READ_TIMER_MS",
    0x1D: "JUMP",
    0x1E: "JUMP_IF_FALSE",
    0x1F: "JUMP_IF_TRUE",
    # A2A Intent
    0x20: "DECLARE_INTENT",
    0x21: "ASSERT_GOAL",
    0x22: "VERIFY_OUTCOME",
    0x23: "EXPLAIN_FAILURE",
    # A2A Communication
    0x30: "TELL",
    0x31: "ASK",
    0x32: "DELEGATE",
    0x33: "REPORT_STATUS",
    0x34: "REQUEST_OVERRIDE",
    # A2A Capability
    0x40: "REQUIRE_CAPABILITY",
    0x41: "DECLARE_SENSOR_NEED",
    0x42: "DECLARE_ACTUATOR_USE",
    # A2A Safety
    0x50: "TRUST_CHECK",
    0x51: "AUTONOMY_LEVEL_ASSERT",
    0x52: "SAFE_BOUNDARY",
    0x53: "RATE_LIMIT",
}

# Reverse lookup: name -> opcode
OPCODE_VALUES: dict[str, int] = {v: k for k, v in OPCODE_NAMES.items()}

OPCODE_CORE_COUNT: int = 32
OPCODE_A2A_COUNT: int = 29
OPCODE_TOTAL_COUNT: int = OPCODE_CORE_COUNT + OPCODE_A2A_COUNT


def opcode_name(opcode: int) -> str:
    """Return the mnemonic name for an opcode, or 'UNKNOWN'."""
    return OPCODE_NAMES.get(opcode, "UNKNOWN")


def is_core_opcode(opcode: int) -> bool:
    """Check if opcode is a core VM opcode (0x00-0x1F)."""
    return opcode <= 0x1F


def is_a2a_opcode(opcode: int) -> bool:
    """Check if opcode is an A2A opcode (0x20-0x56)."""
    return 0x20 <= opcode <= 0x56


def is_valid_opcode(opcode: int) -> bool:
    """Check if opcode is known."""
    return opcode in OPCODE_NAMES
