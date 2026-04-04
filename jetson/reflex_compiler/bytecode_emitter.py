"""NEXUS Reflex Compiler - Binary bytecode emitter.

Emits 8-byte fixed-length instructions from an intermediate
representation into raw binary bytecode.

Instruction format (8 bytes, little-endian):
  Byte 0:    opcode  (uint8)
  Byte 1:    flags   (uint8)
  Bytes 2-3: operand1 (uint16)
  Bytes 4-7: operand2 (uint32)
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ===================================================================
# Instruction format helpers
# ===================================================================

INSTR_SIZE = 8

# Flag constants
FLAGS_HAS_IMMEDIATE = 0x01
FLAGS_IS_FLOAT = 0x02
FLAGS_EXTENDED_CLAMP = 0x04
FLAGS_IS_CALL = 0x08
FLAGS_SYSCALL = 0x80

# Syscall IDs
SYSCALL_HALT = 0x01
SYSCALL_PID_COMPUTE = 0x02
SYSCALL_RECORD_SNAPSHOT = 0x03
SYSCALL_EMIT_EVENT = 0x04


def pack_instruction(opcode: int, flags: int = 0, operand1: int = 0,
                     operand2: int = 0) -> bytes:
    """Pack an 8-byte instruction."""
    return struct.pack("<BBHI", opcode, flags, operand1, operand2)


def unpack_instruction(data: bytes, offset: int = 0) -> tuple[int, int, int, int]:
    """Unpack an 8-byte instruction into (opcode, flags, operand1, operand2)."""
    opcode, flags, operand1, operand2 = struct.unpack_from("<BBHI", data, offset)
    return opcode, flags, operand1, operand2


def float_to_u32(value: float) -> int:
    """Convert a float32 to its uint32 representation."""
    return struct.unpack("<I", struct.pack("<f", value))[0]


def u32_to_float(value: int) -> float:
    """Convert a uint32 to float32."""
    return struct.unpack("<f", struct.pack("<I", value))[0]


class BytecodeEmitter:
    """Binary bytecode emitter."""

    def __init__(self) -> None:
        self._instructions: list[bytes] = []

    def emit_raw(self, opcode: int, flags: int = 0, operand1: int = 0,
                 operand2: int = 0) -> None:
        """Emit a raw 8-byte instruction."""
        self._instructions.append(pack_instruction(opcode, flags, operand1, operand2))

    def emit_nop(self) -> None:
        """Emit a NOP instruction."""
        self.emit_raw(0x00, 0x00, 0, 0)

    def emit_push_i8(self, value: int) -> None:
        """Emit PUSH_I8 with immediate value in operand1 low byte."""
        self.emit_raw(0x01, FLAGS_HAS_IMMEDIATE, value & 0xFF, 0)

    def emit_push_i16(self, value: int) -> None:
        """Emit PUSH_I16 with immediate value in operand1."""
        self.emit_raw(0x02, FLAGS_HAS_IMMEDIATE, value & 0xFFFF, 0)

    def emit_push_f32(self, value: float) -> None:
        """Emit PUSH_F32 with float value encoded in operand2."""
        self.emit_raw(0x03, FLAGS_IS_FLOAT, 0, float_to_u32(value))

    def emit_pop(self) -> None:
        """Emit POP instruction."""
        self.emit_raw(0x04, 0x00, 0, 0)

    def emit_dup(self) -> None:
        """Emit DUP instruction."""
        self.emit_raw(0x05, 0x00, 0, 0)

    def emit_swap(self) -> None:
        """Emit SWAP instruction."""
        self.emit_raw(0x06, 0x00, 0, 0)

    def emit_rot(self) -> None:
        """Emit ROT instruction."""
        self.emit_raw(0x07, 0x00, 0, 0)

    def emit_add_f(self) -> None:
        """Emit ADD_F instruction."""
        self.emit_raw(0x08, 0x00, 0, 0)

    def emit_sub_f(self) -> None:
        """Emit SUB_F instruction."""
        self.emit_raw(0x09, 0x00, 0, 0)

    def emit_mul_f(self) -> None:
        """Emit MUL_F instruction."""
        self.emit_raw(0x0A, 0x00, 0, 0)

    def emit_div_f(self) -> None:
        """Emit DIV_F instruction."""
        self.emit_raw(0x0B, 0x00, 0, 0)

    def emit_neg_f(self) -> None:
        """Emit NEG_F instruction."""
        self.emit_raw(0x0C, 0x00, 0, 0)

    def emit_abs_f(self) -> None:
        """Emit ABS_F instruction."""
        self.emit_raw(0x0D, 0x00, 0, 0)

    def emit_min_f(self) -> None:
        """Emit MIN_F instruction."""
        self.emit_raw(0x0E, 0x00, 0, 0)

    def emit_max_f(self) -> None:
        """Emit MAX_F instruction."""
        self.emit_raw(0x0F, 0x00, 0, 0)

    def emit_clamp_f(self, lo: float, hi: float) -> None:
        """Emit CLAMP_F with lo/hi bounds (extended encoding)."""
        # Standard: operand2 = float_as_uint32 (single bound)
        # Extended: operand2 = (hi_u16 << 16) | lo_u16
        lo_i16 = max(-32768, min(32767, int(lo * 100)))  # Scale to 0.01 precision
        hi_i16 = max(-32768, min(32767, int(hi * 100)))
        operand2 = ((hi_i16 & 0xFFFF) << 16) | (lo_i16 & 0xFFFF)
        self.emit_raw(0x10, FLAGS_EXTENDED_CLAMP, 0, operand2)

    def emit_eq_f(self) -> None:
        """Emit EQ_F instruction."""
        self.emit_raw(0x11, 0x00, 0, 0)

    def emit_lt_f(self) -> None:
        """Emit LT_F instruction."""
        self.emit_raw(0x12, 0x00, 0, 0)

    def emit_gt_f(self) -> None:
        """Emit GT_F instruction."""
        self.emit_raw(0x13, 0x00, 0, 0)

    def emit_lte_f(self) -> None:
        """Emit LTE_F instruction."""
        self.emit_raw(0x14, 0x00, 0, 0)

    def emit_gte_f(self) -> None:
        """Emit GTE_F instruction."""
        self.emit_raw(0x15, 0x00, 0, 0)

    def emit_and_b(self) -> None:
        """Emit AND_B instruction."""
        self.emit_raw(0x16, 0x00, 0, 0)

    def emit_or_b(self) -> None:
        """Emit OR_B instruction."""
        self.emit_raw(0x17, 0x00, 0, 0)

    def emit_xor_b(self) -> None:
        """Emit XOR_B instruction."""
        self.emit_raw(0x18, 0x00, 0, 0)

    def emit_not_b(self) -> None:
        """Emit NOT_B instruction."""
        self.emit_raw(0x19, 0x00, 0, 0)

    def emit_read_pin(self, pin: int) -> None:
        """Emit READ_PIN instruction."""
        self.emit_raw(0x1A, FLAGS_HAS_IMMEDIATE, pin & 0xFFFF, 0)

    def emit_write_pin(self, pin: int) -> None:
        """Emit WRITE_PIN instruction."""
        self.emit_raw(0x1B, FLAGS_HAS_IMMEDIATE, pin & 0xFFFF, 0)

    def emit_read_timer_ms(self) -> None:
        """Emit READ_TIMER_MS instruction."""
        self.emit_raw(0x1C, 0x00, 0, 0)

    def emit_jump(self, target: int) -> None:
        """Emit JUMP instruction."""
        self.emit_raw(0x1D, 0x00, target & 0xFFFF, 0)

    def emit_jump_if_false(self, target: int) -> None:
        """Emit JUMP_IF_FALSE instruction."""
        self.emit_raw(0x1E, 0x00, target & 0xFFFF, 0)

    def emit_jump_if_true(self, target: int) -> None:
        """Emit JUMP_IF_TRUE instruction."""
        self.emit_raw(0x1F, 0x00, target & 0xFFFF, 0)

    def emit_halt(self) -> None:
        """Emit HALT (NOP + SYSCALL + syscall_id=1)."""
        self.emit_raw(0x00, FLAGS_SYSCALL, 0, SYSCALL_HALT)

    def get_bytecode(self) -> bytes:
        """Return the compiled bytecode."""
        return b"".join(self._instructions)

    def instruction_count(self) -> int:
        """Return the number of instructions emitted."""
        return len(self._instructions)

    def reset(self) -> None:
        """Reset the emitter state."""
        self._instructions.clear()

    def current_offset(self) -> int:
        """Return the current byte offset (for jump target calculation)."""
        return len(self._instructions) * INSTR_SIZE
