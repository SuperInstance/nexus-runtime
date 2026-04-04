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

import math
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


def f32_to_f16_bits(value: float) -> int:
    """Convert a float32 to IEEE 754 float16 bit pattern.

    Matches the C firmware's f16_to_f32_bits() decoder in reverse.
    float16: 1 sign bit, 5 exponent bits (bias 15), 10 mantissa bits.
    """
    if math.isnan(value) or math.isinf(value):
        f32_bits = float_to_u32(value)
        sign = (f32_bits >> 31) & 1
        if math.isnan(value):
            return (sign << 15) | 0x7E00 | 1  # quiet NaN
        else:
            return (sign << 15) | 0x7C00  # Inf

    if value == 0.0:
        f32_bits = float_to_u32(value)
        sign = (f32_bits >> 31) & 1
        return sign << 15

    sign = 0
    if value < 0:
        sign = 1
        value = -value

    f32_bits = float_to_u32(value)
    f32_exp = (f32_bits >> 23) & 0xFF
    f32_mant = f32_bits & 0x7FFFFF

    # float32 exponent bias = 127, float16 exponent bias = 15
    f16_exp = f32_exp - 127 + 15

    if f32_exp == 0:
        # float32 denormal — become float16 denormal
        return sign << 15
    elif f32_exp == 255:
        return (sign << 15) | 0x7C00

    if f16_exp >= 31:
        # Overflow to Inf
        return (sign << 15) | 0x7C00

    if f16_exp <= 0:
        # Underflow to denormal
        shift = 1 - f16_exp
        mant = (0x400 | (f32_mant >> 13)) >> shift
        round_bit = (0x400 | (f32_mant >> 13)) & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        if round_bit > half or (round_bit == half and (mant & 1)):
            mant += 1
        return (sign << 15) | (mant & 0x3FF)

    # Normal case: 10-bit mantissa from 23-bit float32 mantissa
    mant = (f32_mant + 0x1000) >> 13  # round to nearest even
    if mant >= 0x400:
        mant = 0x400
        f16_exp += 1
        if f16_exp >= 31:
            return (sign << 15) | 0x7C00

    return (sign << 15) | ((f16_exp & 0x1F) << 10) | (mant & 0x3FF)


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
        """Emit CLAMP_F with lo/hi bounds as IEEE float16 in operand2.

        Matches the C firmware's CLAMP_F decoder which uses f16_to_f32_bits():
          lower 16 bits of operand2 = lo as float16 bit pattern
          upper 16 bits of operand2 = hi as float16 bit pattern
        """
        lo16 = f32_to_f16_bits(lo) & 0xFFFF
        hi16 = f32_to_f16_bits(hi) & 0xFFFF
        operand2 = (hi16 << 16) | lo16
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

    def emit_jump(self, target: int, is_call: bool = False) -> None:
        """Emit JUMP instruction.

        Target is stored in operand2 (uint32, bytes 4-7) to match the C VM's
        vm_execute_instruction() which reads: vm->pc = operand2.
        For RET, operand2 is set to 0xFFFFFFFF and IS_CALL flag is cleared.
        """
        flags = FLAGS_IS_CALL if is_call else 0
        self.emit_raw(0x1D, flags, 0, target & 0xFFFFFFFF)

    def emit_ret(self) -> None:
        """Emit RET instruction (JUMP with operand2=0xFFFFFFFF)."""
        self.emit_raw(0x1D, 0x00, 0, 0xFFFFFFFF)

    def emit_jump_if_false(self, target: int) -> None:
        """Emit JUMP_IF_FALSE instruction.

        Target is stored in operand2 (uint32) to match the C VM.
        """
        self.emit_raw(0x1E, 0x00, 0, target & 0xFFFFFFFF)

    def emit_jump_if_true(self, target: int) -> None:
        """Emit JUMP_IF_TRUE instruction.

        Target is stored in operand2 (uint32) to match the C VM.
        """
        self.emit_raw(0x1F, 0x00, 0, target & 0xFFFFFFFF)

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
