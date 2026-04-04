"""NEXUS Reflex Compiler - Binary bytecode emitter.

Emits 8-byte fixed-length instructions from an intermediate
representation into raw binary bytecode.
"""

from __future__ import annotations

import struct


class BytecodeEmitter:
    """Binary bytecode emitter (stub)."""

    def __init__(self) -> None:
        self._instructions: list[bytes] = []

    def emit_nop(self) -> None:
        """Emit a NOP instruction."""
        self._emit(0x00, 0x00, 0, 0)

    def emit_push_f32(self, value: float) -> None:
        """Emit a PUSH_F32 instruction.

        Args:
            value: Float32 value to push.
        """
        self._emit(0x03, 0x00, 0, 0)
        # TODO: Encode float value in operand2

    def emit_halt(self) -> None:
        """Emit a HALT syscall instruction."""
        self._emit(0x00, 0x80, 0, 1)  # NOP with SYSCALL flag, syscall_id=1

    def _emit(self, opcode: int, flags: int, operand1: int, operand2: int) -> None:
        """Emit a raw 8-byte instruction."""
        instr = struct.pack("<BBHI", opcode, flags, operand1, operand2)
        self._instructions.append(instr)

    def get_bytecode(self) -> bytes:
        """Return the compiled bytecode."""
        return b"".join(self._instructions)

    def instruction_count(self) -> int:
        """Return the number of instructions emitted."""
        return len(self._instructions)

    def reset(self) -> None:
        """Reset the emitter state."""
        self._instructions.clear()
