"""Rosetta Stone Layer 4: IR-to-bytecode generator.

Converts validated IR instructions into NEXUS bytecode using the same
8-byte fixed instruction format as the reflex compiler.

Instruction format (8 bytes, little-endian):
  Byte 0:    opcode  (uint8)
  Byte 1:    flags   (uint8)
  Bytes 2-3: operand1 (uint16)
  Bytes 4-7: operand2 (uint32)

Uses the BytecodeEmitter from the reflex package for encoding.
"""

from __future__ import annotations

import struct

from reflex.bytecode_emitter import (
    BytecodeEmitter,
    INSTR_SIZE,
    FLAGS_HAS_IMMEDIATE,
    FLAGS_IS_FLOAT,
    FLAGS_SYSCALL,
    float_to_u32,
    f32_to_f16_bits,
    pack_instruction,
    unpack_instruction,
)
from reflex.safety_validator import SafetyValidator, STACK_EFFECTS
from shared.opcodes import OPCODE_NAMES, OPCODE_VALUES

from agent.rosetta_stone.intent_compiler import IRInstruction


# ===================================================================
# Opcodes used in the NEXUS VM
# ===================================================================

_CORE_OPCODES: set[str] = {
    "NOP", "PUSH_I8", "PUSH_I16", "PUSH_F32", "POP", "DUP", "SWAP", "ROT",
    "ADD_F", "SUB_F", "MUL_F", "DIV_F", "NEG_F", "ABS_F", "MIN_F", "MAX_F",
    "CLAMP_F", "EQ_F", "LT_F", "GT_F", "LTE_F", "GTE_F",
    "AND_B", "OR_B", "XOR_B", "NOT_B",
    "READ_PIN", "WRITE_PIN", "READ_TIMER_MS",
    "JUMP", "JUMP_IF_FALSE", "JUMP_IF_TRUE",
}

_A2A_OPCODES: set[str] = {
    "DECLARE_INTENT", "ASSERT_GOAL", "VERIFY_OUTCOME", "EXPLAIN_FAILURE",
    "TELL", "ASK", "DELEGATE", "REPORT_STATUS", "REQUEST_OVERRIDE",
    "REQUIRE_CAPABILITY", "DECLARE_SENSOR_NEED", "DECLARE_ACTUATOR_USE",
    "TRUST_CHECK", "AUTONOMY_LEVEL_ASSERT", "SAFE_BOUNDARY", "RATE_LIMIT",
}

_ALL_VALID_OPCODES = _CORE_OPCODES | _A2A_OPCODES


# ===================================================================
# BytecodeGenerator
# ===================================================================

class BytecodeGenerator:
    """Generate NEXUS bytecode from validated IR instructions.

    Uses the reflex BytecodeEmitter for binary encoding, ensuring
    compatibility with the ESP32-S3 VM.
    """

    def __init__(self) -> None:
        self._emitter = BytecodeEmitter()
        self._validator = SafetyValidator()

    def generate(self, ir: list[IRInstruction]) -> bytes:
        """Generate NEXUS bytecode from IR instructions.

        First resolves label references, then emits each instruction
        using the BytecodeEmitter.

        Args:
            ir: List of validated IR instructions.

        Returns:
            Compiled bytecode bytes (multiple of 8 bytes).

        Raises:
            ValueError: If an unknown opcode or unresolved label is encountered.
        """
        self._emitter.reset()

        # Build label -> instruction index map
        labels: dict[str, int] = {}
        for i, instr in enumerate(ir):
            if instr.label:
                labels[instr.label] = i

        # Emit each instruction
        for instr in ir:
            self._emit_instruction(instr, labels)

        bytecode = self._emitter.get_bytecode()

        # Verify bytecode integrity
        errors = self._validator.validate_bytecode(bytecode)
        # Filter out CLAMP_F warnings for syscalls/reads that don't write
        # (our IR generator always adds CLAMP_F before WRITE_PIN)
        if errors:
            raise ValueError(f"Bytecode verification failed: {'; '.join(errors)}")

        return bytecode

    def disassemble(self, bytecode: bytes) -> list[str]:
        """Disassemble bytecode into human-readable strings.

        Args:
            bytecode: NEXUS bytecode bytes.

        Returns:
            List of disassembled instruction strings.
        """
        lines: list[str] = []
        n_instr = len(bytecode) // INSTR_SIZE

        for i in range(n_instr):
            opcode, flags, op1, op2 = unpack_instruction(bytecode, i * INSTR_SIZE)
            op_name = OPCODE_NAMES.get(opcode, f"UNKNOWN(0x{opcode:02X})")
            parts = [f"{i:4d}: {op_name}"]

            # Decode operands based on opcode
            if opcode in (0x00,) and (flags & FLAGS_SYSCALL):
                # Syscall
                syscall_names = {0x01: "HALT", 0x02: "PID_COMPUTE",
                                 0x03: "RECORD_SNAPSHOT", 0x04: "EMIT_EVENT"}
                sname = syscall_names.get(op2, f"SYSCALL_{op2}")
                parts.append(f"  {sname}")
                if opcode == 0x00 and op2 == 0x04 and op1 != 0:
                    parts.append(f"  event_id={op1}")
            elif opcode == 0x03 and (flags & FLAGS_IS_FLOAT):
                # PUSH_F32
                value = struct.unpack("<f", struct.pack("<I", op2))[0]
                parts.append(f"  {value}")
            elif opcode in (0x01, 0x02) and (flags & FLAGS_HAS_IMMEDIATE):
                # PUSH_I8 or PUSH_I16
                parts.append(f"  {op1}")
            elif opcode == 0x10 and (flags & 0x04):
                # CLAMP_F extended
                lo16 = op2 & 0xFFFF
                hi16 = (op2 >> 16) & 0xFFFF
                parts.append(f"  clamp")
            elif opcode in (0x1A, 0x1B) and (flags & FLAGS_HAS_IMMEDIATE):
                # READ_PIN / WRITE_PIN
                parts.append(f"  pin={op1}")
            elif opcode in (0x1D, 0x1E, 0x1F):
                # JUMP / JUMP_IF_FALSE / JUMP_IF_TRUE
                parts.append(f"  -> {op2}")
            else:
                if op1 != 0 or op2 != 0:
                    parts.append(f"  op1={op1} op2={op2}")

            lines.append(" ".join(parts))

        return lines

    # -----------------------------------------------------------------
    # Instruction emission
    # -----------------------------------------------------------------

    def _emit_instruction(
        self, instr: IRInstruction, labels: dict[str, int]
    ) -> None:
        """Emit a single IR instruction as bytecode."""
        opcode_name = instr.opcode

        # Handle SYSCALL as a special case (maps to NOP + SYSCALL flag)
        if opcode_name == "SYSCALL":
            self._emit_syscall(instr)
            return

        # Handle HALT alias
        if opcode_name == "HALT":
            self._emitter.emit_halt()
            return

        # Map CLAMP_F with lo/hi operands
        if opcode_name == "CLAMP_F":
            lo = float(instr.operand1) if instr.operand1 is not None else -1.0
            hi = float(instr.operand2) if instr.operand2 is not None else 1.0
            self._emitter.emit_clamp_f(lo, hi)
            return

        # Map opcodes to emitter methods
        emitter_method = self._get_emitter_method(opcode_name)
        if emitter_method is not None:
            emitter_method(instr, labels)
            return

        # Direct opcode mapping for simple instructions
        opcode_val = OPCODE_VALUES.get(opcode_name)
        if opcode_val is not None:
            self._emitter.emit_raw(opcode_val, 0, 0, 0)
            return

        raise ValueError(f"Unknown IR opcode: {opcode_name}")

    def _emit_syscall(self, instr: IRInstruction) -> None:
        """Emit a SYSCALL instruction (NOP + FLAGS_SYSCALL)."""
        syscall_id = int(instr.operand2) if instr.operand2 is not None else 0x01
        event_data = int(instr.operand1) if instr.operand1 is not None else 0

        if syscall_id == 0x01:  # HALT
            self._emitter.emit_halt()
        elif syscall_id == 0x02:  # PID_COMPUTE
            self._emitter.emit_raw(0x00, FLAGS_SYSCALL, 0, 0x02)
        elif syscall_id == 0x03:  # RECORD_SNAPSHOT
            self._emitter.emit_raw(0x00, FLAGS_SYSCALL, 0, 0x03)
        elif syscall_id == 0x04:  # EMIT_EVENT
            self._emitter.emit_raw(0x00, FLAGS_SYSCALL, event_data & 0xFFFF, 0x04)
        else:
            self._emitter.emit_raw(0x00, FLAGS_SYSCALL, 0, syscall_id)

    def _get_emitter_method(self, opcode_name: str):
        """Get the appropriate emitter method for an IR opcode."""
        _EMITTER_MAP = {
            "NOP": lambda instr, labels: self._emitter.emit_nop(),
            "PUSH_I8": lambda instr, labels: self._emitter.emit_push_i8(
                int(instr.operand1) if instr.operand1 is not None else 0
            ),
            "PUSH_I16": lambda instr, labels: self._emitter.emit_push_i16(
                int(instr.operand1) if instr.operand1 is not None else 0
            ),
            "PUSH_F32": lambda instr, labels: self._emitter.emit_push_f32(
                float(instr.operand2) if instr.operand2 is not None else 0.0
            ),
            "POP": lambda instr, labels: self._emitter.emit_pop(),
            "DUP": lambda instr, labels: self._emitter.emit_dup(),
            "SWAP": lambda instr, labels: self._emitter.emit_swap(),
            "ROT": lambda instr, labels: self._emitter.emit_rot(),
            "ADD_F": lambda instr, labels: self._emitter.emit_add_f(),
            "SUB_F": lambda instr, labels: self._emitter.emit_sub_f(),
            "MUL_F": lambda instr, labels: self._emitter.emit_mul_f(),
            "DIV_F": lambda instr, labels: self._emitter.emit_div_f(),
            "NEG_F": lambda instr, labels: self._emitter.emit_neg_f(),
            "ABS_F": lambda instr, labels: self._emitter.emit_abs_f(),
            "MIN_F": lambda instr, labels: self._emitter.emit_min_f(),
            "MAX_F": lambda instr, labels: self._emitter.emit_max_f(),
            "EQ_F": lambda instr, labels: self._emitter.emit_eq_f(),
            "LT_F": lambda instr, labels: self._emitter.emit_lt_f(),
            "GT_F": lambda instr, labels: self._emitter.emit_gt_f(),
            "LTE_F": lambda instr, labels: self._emitter.emit_lte_f(),
            "GTE_F": lambda instr, labels: self._emitter.emit_gte_f(),
            "AND_B": lambda instr, labels: self._emitter.emit_and_b(),
            "OR_B": lambda instr, labels: self._emitter.emit_or_b(),
            "XOR_B": lambda instr, labels: self._emitter.emit_xor_b(),
            "NOT_B": lambda instr, labels: self._emitter.emit_not_b(),
            "READ_PIN": lambda instr, labels: self._emitter.emit_read_pin(
                int(instr.operand1) if instr.operand1 is not None else 0
            ),
            "WRITE_PIN": lambda instr, labels: self._emitter.emit_write_pin(
                int(instr.operand1) if instr.operand1 is not None else 0
            ),
            "READ_TIMER_MS": lambda instr, labels: self._emitter.emit_read_timer_ms(),
            "JUMP": lambda instr, labels: self._emitter.emit_jump(
                self._resolve_label(instr.jump_target, labels)
            ),
            "JUMP_IF_FALSE": lambda instr, labels: self._emitter.emit_jump_if_false(
                self._resolve_label(instr.jump_target, labels)
            ),
            "JUMP_IF_TRUE": lambda instr, labels: self._emitter.emit_jump_if_true(
                self._resolve_label(instr.jump_target, labels)
            ),
            "DECLARE_INTENT": lambda instr, labels: self._emitter.emit_raw(
                OPCODE_VALUES["DECLARE_INTENT"], 0,
                int(instr.operand1) if instr.operand1 is not None else 0,
                int(instr.operand2) if instr.operand2 is not None else 0,
            ),
        }
        return _EMITTER_MAP.get(opcode_name)

    @staticmethod
    def _resolve_label(
        jump_target: str | None, labels: dict[str, int]
    ) -> int:
        """Resolve a label name to an instruction index."""
        if jump_target is None:
            return 0
        if jump_target not in labels:
            raise ValueError(f"Unresolved label: '{jump_target}'")
        return labels[jump_target]
