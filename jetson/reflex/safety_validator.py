"""NEXUS Reflex Compiler - Safety validation.

Static analysis for reflex bytecode:
  - Stack depth analysis
  - Jump bounds checking
  - Cycle budget verification
  - Actuator clamping verification (CLAMP_F before WRITE_PIN)
  - NaN/Infinity guard verification
"""

from __future__ import annotations

import math
import struct

from reflex.bytecode_emitter import (
    INSTR_SIZE,
    unpack_instruction,
)


# Opcode categories for stack effect analysis
# Positive = pushes, Negative = pops, 0 = no change
STACK_EFFECTS: dict[int, int] = {
    0x00: 0,   # NOP
    0x01: 1,   # PUSH_I8
    0x02: 1,   # PUSH_I16
    0x03: 1,   # PUSH_F32
    0x04: -1,  # POP
    0x05: 1,   # DUP
    0x06: 0,   # SWAP
    0x07: 0,   # ROT
    0x08: -1,  # ADD_F
    0x09: -1,  # SUB_F
    0x0A: -1,  # MUL_F
    0x0B: -1,  # DIV_F
    0x0C: 0,   # NEG_F
    0x0D: 0,   # ABS_F
    0x0E: -1,  # MIN_F
    0x0F: -1,  # MAX_F
    0x10: 0,   # CLAMP_F (pops 1, pushes 1, net 0)
    0x11: -1,  # EQ_F
    0x12: -1,  # LT_F
    0x13: -1,  # GT_F
    0x14: -1,  # LTE_F
    0x15: -1,  # GTE_F
    0x16: -1,  # AND_B
    0x17: -1,  # OR_B
    0x18: -1,  # XOR_B
    0x19: 0,   # NOT_B
    0x1A: 1,   # READ_PIN
    0x1B: -1,  # WRITE_PIN
    0x1C: 1,   # READ_TIMER_MS
    0x1D: 0,   # JUMP
    0x1E: -1,  # JUMP_IF_FALSE
    0x1F: -1,  # JUMP_IF_TRUE
}

# A2A opcodes have no stack effect
for op in range(0x20, 0x57):
    STACK_EFFECTS[op] = 0

MAX_STACK_DEPTH = 16
MAX_CYCLE_BUDGET = 1000  # Max instructions per reflex cycle
CLAMP_F_OPCODE = 0x10
WRITE_PIN_OPCODE = 0x1B
JUMP_OPCODE = 0x1D
JUMP_IF_FALSE_OPCODE = 0x1E
JUMP_IF_TRUE_OPCODE = 0x1F


class SafetyValidator:
    """Reflex safety validator."""

    def __init__(self, max_stack: int = MAX_STACK_DEPTH,
                 max_cycles: int = MAX_CYCLE_BUDGET) -> None:
        self.max_stack = max_stack
        self.max_cycles = max_cycles

    def validate_bytecode(self, bytecode: bytes) -> list[str]:
        """Validate compiled bytecode for safety violations.

        Args:
            bytecode: Compiled bytecode (must be multiple of 8 bytes).

        Returns:
            List of safety violations (empty if safe).
        """
        errors: list[str] = []

        if len(bytecode) % INSTR_SIZE != 0:
            errors.append(f"Bytecode length {len(bytecode)} is not a multiple of {INSTR_SIZE}")

        n_instr = len(bytecode) // INSTR_SIZE
        if n_instr > self.max_cycles:
            errors.append(f"Instruction count {n_instr} exceeds cycle budget {self.max_cycles}")

        # Stack depth analysis
        stack_depth = self.check_stack_depth(bytecode)
        if stack_depth is None:
            errors.append("Stack underflow detected")
        elif stack_depth > self.max_stack:
            errors.append(f"Max stack depth {stack_depth} exceeds limit {self.max_stack}")

        # Jump bounds checking
        jump_errors = self._check_jump_bounds(bytecode)
        errors.extend(jump_errors)

        # CLAMP_F before WRITE_PIN check
        clamp_errors = self._check_clamp_before_write(bytecode)
        errors.extend(clamp_errors)

        # NaN guard check (for PUSH_F32 with NaN/Inf)
        nan_errors = self._check_nan_values(bytecode)
        errors.extend(nan_errors)

        return errors

    def check_stack_depth(self, bytecode: bytes) -> int | None:
        """Compute maximum stack depth.

        Args:
            bytecode: Compiled bytecode.

        Returns:
            Maximum stack depth, or None if underflow detected.
        """
        n_instr = len(bytecode) // INSTR_SIZE
        max_depth = 0
        current = 0

        for i in range(n_instr):
            opcode, flags, op1, op2 = unpack_instruction(bytecode, i * INSTR_SIZE)
            effect = STACK_EFFECTS.get(opcode, 0)
            current += effect

            if current < 0:
                return None  # Underflow
            if current > max_depth:
                max_depth = current

        return max_depth

    def _check_jump_bounds(self, bytecode: bytes) -> list[str]:
        """Check all jump targets are within bounds."""
        errors: list[str] = []
        n_instr = len(bytecode) // INSTR_SIZE

        for i in range(n_instr):
            opcode, flags, op1, op2 = unpack_instruction(bytecode, i * INSTR_SIZE)
            if opcode in (JUMP_OPCODE, JUMP_IF_FALSE_OPCODE, JUMP_IF_TRUE_OPCODE):
                target = op1
                if target >= n_instr:
                    errors.append(
                        f"Jump at instruction {i} targets {target}, "
                        f"but only {n_instr} instructions exist"
                    )

        return errors

    def _check_clamp_before_write(self, bytecode: bytes) -> list[str]:
        """Check that every WRITE_PIN is preceded by CLAMP_F."""
        errors: list[str] = []
        n_instr = len(bytecode) // INSTR_SIZE

        for i in range(n_instr):
            opcode, flags, op1, op2 = unpack_instruction(bytecode, i * INSTR_SIZE)
            if opcode == WRITE_PIN_OPCODE:
                # Look backwards for CLAMP_F (not necessarily immediately preceding)
                found_clamp = False
                for j in range(i - 1, -1, -1):
                    prev_op, _, _, _ = unpack_instruction(bytecode, j * INSTR_SIZE)
                    if prev_op == WRITE_PIN_OPCODE:
                        break  # Found another WRITE_PIN, stop searching
                    if prev_op == CLAMP_F_OPCODE:
                        found_clamp = True
                        break
                if not found_clamp:
                    errors.append(
                        f"WRITE_PIN at instruction {i} not preceded by CLAMP_F"
                    )

        return errors

    def _check_nan_values(self, bytecode: bytes) -> list[str]:
        """Check for NaN/Infinity in PUSH_F32 immediate values."""
        errors: list[str] = []
        n_instr = len(bytecode) // INSTR_SIZE

        for i in range(n_instr):
            opcode, flags, op1, op2 = unpack_instruction(bytecode, i * INSTR_SIZE)
            if opcode == 0x03 and (flags & 0x02):  # PUSH_F32 with IS_FLOAT flag
                value = struct.unpack("<f", struct.pack("<I", op2))[0]
                if math.isnan(value):
                    errors.append(f"PUSH_F32 at instruction {i} has NaN value")
                elif math.isinf(value):
                    errors.append(f"PUSH_F32 at instruction {i} has Infinity value")

        return errors
