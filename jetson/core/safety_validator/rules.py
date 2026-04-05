"""NEXUS Safety Validator — Safety rule definitions.

Defines:
  - TRUST_OPCODE_MATRIX: which opcodes are allowed at each trust level (L0–L5)
  - FORBIDDEN_PATTERNS: patterns that are always disallowed regardless of trust
  - SAFETY_CRITICAL_PINS: pin numbers protected by hardware safety interlocks
  - Stack effect table, instruction size, syscall IDs
  - Configuration constants for the pipeline
"""

from __future__ import annotations

# ===================================================================
# Instruction format constants
# ===================================================================

INSTR_SIZE = 8  # Every instruction is exactly 8 bytes

# Flag bits
FLAGS_HAS_IMMEDIATE = 0x01
FLAGS_IS_FLOAT = 0x02
FLAGS_EXTENDED_CLAMP = 0x04
FLAGS_IS_CALL = 0x08
FLAGS_SYSCALL = 0x80

# Syscall IDs (opcode=NOP, flags=SYSCALL, operand2=syscall_id)
SYSCALL_HALT = 0x01
SYSCALL_PID_COMPUTE = 0x02
SYSCALL_RECORD_SNAPSHOT = 0x03
SYSCALL_EMIT_EVENT = 0x04

# ===================================================================
# Core opcodes (0x00–0x1F)
# ===================================================================

OP_NOP = 0x00
OP_PUSH_I8 = 0x01
OP_PUSH_I16 = 0x02
OP_PUSH_F32 = 0x03
OP_POP = 0x04
OP_DUP = 0x05
OP_SWAP = 0x06
OP_ROT = 0x07
OP_ADD_F = 0x08
OP_SUB_F = 0x09
OP_MUL_F = 0x0A
OP_DIV_F = 0x0B
OP_NEG_F = 0x0C
OP_ABS_F = 0x0D
OP_MIN_F = 0x0E
OP_MAX_F = 0x0F
OP_CLAMP_F = 0x10
OP_EQ_F = 0x11
OP_LT_F = 0x12
OP_GT_F = 0x13
OP_LTE_F = 0x14
OP_GTE_F = 0x15
OP_AND_B = 0x16
OP_OR_B = 0x17
OP_XOR_B = 0x18
OP_NOT_B = 0x19
OP_READ_PIN = 0x1A
OP_WRITE_PIN = 0x1B
OP_READ_TIMER_MS = 0x1C
OP_JUMP = 0x1D
OP_JUMP_IF_FALSE = 0x1E
OP_JUMP_IF_TRUE = 0x1F

# ===================================================================
# A2A opcodes (0x20–0x56) — all NOP on ESP32 firmware
# ===================================================================

OP_DECLARE_INTENT = 0x20
OP_ASSERT_GOAL = 0x21
OP_VERIFY_OUTCOME = 0x22
OP_EXPLAIN_FAILURE = 0x23
OP_TELL = 0x30
OP_ASK = 0x31
OP_DELEGATE = 0x32
OP_REPORT_STATUS = 0x33
OP_REQUEST_OVERRIDE = 0x34
OP_REQUIRE_CAPABILITY = 0x40
OP_DECLARE_SENSOR_NEED = 0x41
OP_DECLARE_ACTUATOR_USE = 0x42
OP_TRUST_CHECK = 0x50
OP_AUTONOMY_LEVEL_ASSERT = 0x51
OP_SAFE_BOUNDARY = 0x52
OP_RATE_LIMIT = 0x53

# ===================================================================
# Stack effects (net change in stack depth per opcode)
# Positive = pushes, Negative = pops, 0 = no net change
# ===================================================================

STACK_EFFECTS: dict[int, int] = {
    OP_NOP: 0,
    OP_PUSH_I8: 1,
    OP_PUSH_I16: 1,
    OP_PUSH_F32: 1,
    OP_POP: -1,
    OP_DUP: 1,       # pops 0, pushes 1 (copies top)
    OP_SWAP: 0,
    OP_ROT: 0,
    OP_ADD_F: -1,    # pops 2, pushes 1
    OP_SUB_F: -1,
    OP_MUL_F: -1,
    OP_DIV_F: -1,
    OP_NEG_F: 0,     # pops 1, pushes 1
    OP_ABS_F: 0,
    OP_MIN_F: -1,    # pops 2, pushes 1
    OP_MAX_F: -1,
    OP_CLAMP_F: 0,   # pops 1, pushes 1
    OP_EQ_F: -1,     # pops 2, pushes 1
    OP_LT_F: -1,
    OP_GT_F: -1,
    OP_LTE_F: -1,
    OP_GTE_F: -1,
    OP_AND_B: -1,
    OP_OR_B: -1,
    OP_XOR_B: -1,
    OP_NOT_B: 0,
    OP_READ_PIN: 1,   # pushes pin value
    OP_WRITE_PIN: -1, # pops value to write
    OP_READ_TIMER_MS: 1,
    OP_JUMP: 0,
    OP_JUMP_IF_FALSE: -1,  # pops condition
    OP_JUMP_IF_TRUE: -1,   # pops condition
}

# A2A opcodes have no stack effect on ESP32
for _op in range(0x20, 0x57):
    STACK_EFFECTS.setdefault(_op, 0)

# ===================================================================
# All valid opcodes
# ===================================================================

ALL_VALID_OPCODES: set[int] = set(STACK_EFFECTS.keys())

# ===================================================================
# Trust Level → Allowed Opcodes Matrix
#
# L0 (Disabled): read-only sensor ops + computation (no I/O writes, no control flow)
# L1 (Advisory): + conditional branches (JUMP_IF_FALSE/TRUE)
# L2 (Supervised): + WRITE_PIN to non-safety actuators, unconditional JUMP
# L3 (Semi-Autonomous): + CALL/RET (subroutine capability)
# L4 (High Autonomy): + all I/O pins including safety-critical
# L5 (Full Autonomy): + SYSCALL (HALT, PID_COMPUTE, full system access)
# ===================================================================

L0_OPCODES: set[int] = {
    OP_NOP,
    OP_PUSH_I8, OP_PUSH_I16, OP_PUSH_F32,
    OP_POP, OP_DUP, OP_SWAP, OP_ROT,
    OP_ADD_F, OP_SUB_F, OP_MUL_F, OP_DIV_F,
    OP_NEG_F, OP_ABS_F, OP_MIN_F, OP_MAX_F, OP_CLAMP_F,
    OP_EQ_F, OP_LT_F, OP_GT_F, OP_LTE_F, OP_GTE_F,
    OP_AND_B, OP_OR_B, OP_XOR_B, OP_NOT_B,
    OP_READ_PIN, OP_READ_TIMER_MS,
}

L1_OPCODES: set[int] = L0_OPCODES | {
    OP_JUMP_IF_FALSE, OP_JUMP_IF_TRUE,
}

L2_OPCODES: set[int] = L1_OPCODES | {
    OP_WRITE_PIN,
    OP_JUMP,
}

L3_OPCODES: set[int] = L2_OPCODES.copy()
# CALL = JUMP with FLAGS_IS_CALL (same JUMP opcode, no new opcodes)
# RET = JUMP with operand2=0xFFFFFFFF (same JUMP opcode)

L4_OPCODES: set[int] = L3_OPCODES | {
    # WRITE_PIN to safety-critical pins is now allowed
    # (same opcode, but pin restrictions are relaxed at L4)
    OP_READ_PIN,  # already included, but explicit for clarity
}

# L5 allows all opcodes
L5_OPCODES: set[int] = ALL_VALID_OPCODES.copy()

TRUST_OPCODE_MATRIX: dict[int, set[int]] = {
    0: L0_OPCODES,
    1: L1_OPCODES,
    2: L2_OPCODES,
    3: L3_OPCODES,
    4: L4_OPCODES,
    5: L5_OPCODES,
}

# ===================================================================
# Forbidden patterns (checked regardless of trust level)
# ===================================================================

FORBIDDEN_PATTERNS: list[dict[str, str]] = [
    {
        "name": "unmatched_call",
        "description": "CALL without matching RET (call stack imbalance)",
    },
    {
        "name": "jump_to_zero",
        "description": "JUMP targeting instruction address 0 (infinite loop risk)",
    },
    {
        "name": "excessive_nop",
        "description": "More than 100 consecutive NOP instructions",
    },
    {
        "name": "write_to_safety_pin_low_trust",
        "description": "WRITE_PIN to safety-critical pins below trust level 4",
    },
    {
        "name": "syscall_low_trust",
        "description": "SYSCALL (HALT/PID/etc.) below trust level 5",
    },
]

# ===================================================================
# Safety-critical pin definitions
# Pin 0: E-Stop input
# Pin 1: Watchdog timer feed
# Pin 2: Heartbeat output
# Pin 3: Power enable (motor controller)
# ===================================================================

SAFETY_CRITICAL_PINS: set[int] = {0, 1, 2, 3}

# ===================================================================
# Pipeline configuration defaults
# ===================================================================

DEFAULT_MAX_INSTRUCTIONS = 1024
DEFAULT_MAX_STACK_DEPTH = 64       # Pipeline uses 64; reflex validator uses 16
DEFAULT_MAX_CALL_DEPTH = 8
DEFAULT_MAX_NOP_SEQUENCE = 100
DEFAULT_MAX_PROGRAM_SIZE = DEFAULT_MAX_INSTRUCTIONS * INSTR_SIZE  # 8192 bytes

# PID safe ranges (for SYSCALL PID_COMPUTE)
PID_KP_RANGE = (0.0, 10.0)
PID_KI_RANGE = (0.0, 5.0)
PID_KD_RANGE = (0.0, 5.0)
PID_SETPOINT_RANGE = (-1000.0, 1000.0)

# Actuator physical ranges (for CLAMP_F validation)
ACTUATOR_SAFE_RANGES: dict[int, tuple[float, float]] = {
    # Pin: (min, max)
    4: (-90.0, 90.0),    # Rudder angle
    5: (-100.0, 100.0),  # Throttle
    6: (0.0, 360.0),     # Heading setpoint
    7: (0.0, 1.0),       # Normalized output
}
DEFAULT_CLAMP_RANGE = (-1000.0, 1000.0)
