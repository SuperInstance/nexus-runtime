"""
NEXUS VM Executor — 32-opcode bytecode virtual machine for marine robotics.

Instruction format (8 bytes, little-endian):
    [opcode:u8][rd:u8][rs1:u8][rs2:u8][imm32:i32]

Register file:
    R0-R15  — general-purpose registers
    R16-R31 — IO-mapped registers (sensor/actuator channels)

Architecture features:
    - 32 registers (16 GP + 16 IO-mapped)
    - 64 KB addressable memory
    - Hardware stack (1024 entries)
    - Deterministic cycle-count execution
"""

from __future__ import annotations

import enum
import struct
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_REGISTERS: int = 32
NUM_GP_REGS: int = 16
NUM_IO_REGS: int = 16
MEMORY_SIZE: int = 65536  # 64 KB
STACK_DEPTH: int = 1024
INSTRUCTION_SIZE: int = 8  # bytes


# ---------------------------------------------------------------------------
# Opcodes
# ---------------------------------------------------------------------------

class Opcodes(enum.IntEnum):
    """32 VM opcodes for marine-robotics control."""

    NOP = 0x00
    LOAD_CONST = 0x01
    LOAD_REG = 0x02
    STORE_REG = 0x03
    ADD = 0x04
    SUB = 0x05
    MUL = 0x06
    DIV = 0x07
    AND = 0x08
    OR = 0x09
    XOR = 0x0A
    NOT = 0x0B
    SHL = 0x0C
    SHR = 0x0D
    CMP = 0x0E
    JMP = 0x0F
    JZ = 0x10
    JNZ = 0x11
    CALL = 0x12
    RET = 0x13
    PUSH = 0x14
    POP = 0x15
    READ_IO = 0x16
    WRITE_IO = 0x17
    HALT = 0x18
    SLEEP = 0x19
    SEND = 0x1A
    RECV = 0x1B
    ALLOC = 0x1C
    FREE = 0x1D
    DMA_COPY = 0x1E
    INTERRUPT = 0x1F
    CUSTOM_0 = 0x20
    CUSTOM_1 = 0x21


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Instruction:
    """Single decoded VM instruction (8 bytes)."""

    opcode: int
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    imm32: int = 0

    # --- helpers ---

    def encode(self) -> bytes:
        """Encode instruction to 8-byte little-endian representation."""
        return struct.pack("<BBBBi", self.opcode, self.rd, self.rs1, self.rs2, self.imm32)

    @classmethod
    def decode(cls, data: bytes, offset: int = 0) -> "Instruction":
        """Decode an instruction from *data* starting at *offset*."""
        opcode, rd, rs1, rs2, imm32 = struct.unpack_from("<BBBBi", data, offset)
        return cls(opcode=opcode, rd=rd, rs1=rs1, rs2=rs2, imm32=imm32)

    def __repr__(self) -> str:
        name = Opcodes(self.opcode).name if self.opcode in _OPCODE_VALUES else f"OP_{self.opcode:#04x}"
        return f"Instruction({name}, rd={self.rd}, rs1={self.rs1}, rs2={self.rs2}, imm={self.imm32})"


_OPCODE_VALUES = {op.value for op in Opcodes}


@dataclass
class VMState:
    """Snapshot of the full VM state."""

    pc: int = 0
    registers: List[int] = field(default_factory=lambda: [0] * NUM_REGISTERS)
    stack: List[int] = field(default_factory=list)
    flags_zero: bool = False
    flags_negative: bool = False
    halted: bool = False
    cycle_count: int = 0
    memory: bytes = b"\x00" * MEMORY_SIZE


# ---------------------------------------------------------------------------
# IO Callbacks
# ---------------------------------------------------------------------------

IOReadCallback = Callable[[int], int]
IOWriteCallback = Callable[[int, int], None]


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:
    """NEXUS bytecode VM executor.

    Parameters
    ----------
    program : bytes
        Raw bytecode to execute.
    io_read_cb : optional
        Callback invoked on READ_IO; receives IO register index, returns value.
    io_write_cb : optional
        Callback invoked on WRITE_IO; receives (register index, value).
    memory_size : int
        Total addressable memory in bytes (default 64 KB).
    stack_depth : int
        Maximum hardware stack depth (default 1024).
    """

    def __init__(
        self,
        program: bytes = b"",
        io_read_cb: Optional[IOReadCallback] = None,
        io_write_cb: Optional[IOWriteCallback] = None,
        memory_size: int = MEMORY_SIZE,
        stack_depth: int = STACK_DEPTH,
    ) -> None:
        self.program = bytearray(program)
        self.pc: int = 0
        self.registers: List[int] = [0] * NUM_REGISTERS
        self.stack: List[int] = []
        self.memory: bytearray = bytearray(memory_size)
        self.flags_zero: bool = False
        self.flags_negative: bool = False
        self.halted: bool = False
        self.cycle_count: int = 0
        self._stack_depth = stack_depth
        self._memory_size = memory_size
        self._io_read_cb = io_read_cb
        self._io_write_cb = io_write_cb
        self._send_buffer: List[tuple] = []
        self._recv_buffer: List[int] = []
        self._interrupts: List[int] = []

        # Load program into memory starting at address 0
        self.memory[: len(program)] = program

    # ----- public API -----

    def step(self) -> Optional[Instruction]:
        """Execute one instruction and return it (or ``None`` if halted)."""
        if self.halted:
            return None

        if self.pc + INSTRUCTION_SIZE > len(self.program):
            self.halted = True
            return None

        insn = Instruction.decode(self.program, self.pc)
        self._execute(insn)
        if not self.halted and insn.opcode not in (Opcodes.JMP, Opcodes.JZ, Opcodes.JNZ, Opcodes.CALL, Opcodes.RET):
            self.pc += INSTRUCTION_SIZE

        self.cycle_count += 1
        return insn

    def run(self, max_cycles: int = 100_000) -> int:
        """Run until halted or *max_cycles* reached. Returns cycles executed."""
        executed = 0
        while not self.halted and executed < max_cycles:
            self.step()
            executed += 1
        return executed

    def reset(self) -> None:
        """Reset the VM to its initial state (program preserved)."""
        self.pc = 0
        self.registers = [0] * NUM_REGISTERS
        self.stack.clear()
        self.memory = bytearray(self._memory_size)
        self.memory[: len(self.program)] = self.program
        self.flags_zero = False
        self.flags_negative = False
        self.halted = False
        self.cycle_count = 0
        self._send_buffer.clear()
        self._recv_buffer.clear()
        self._interrupts.clear()

    def get_state(self) -> VMState:
        """Return a snapshot of the current VM state."""
        return VMState(
            pc=self.pc,
            registers=list(self.registers),
            stack=list(self.stack),
            flags_zero=self.flags_zero,
            flags_negative=self.flags_negative,
            halted=self.halted,
            cycle_count=self.cycle_count,
            memory=bytes(self.memory),
        )

    def load_program(self, program: bytes) -> None:
        """Replace the current program and reset the VM."""
        self.program = bytearray(program)
        self.reset()

    def push_recv(self, value: int) -> None:
        """Push a value into the receive buffer for RECV opcode."""
        self._recv_buffer.append(value)

    def pop_send(self) -> Optional[tuple]:
        """Pop a (dest, value) tuple from the send buffer (from SEND opcode)."""
        if self._send_buffer:
            return self._send_buffer.pop(0)
        return None

    def push_interrupt(self, irq: int) -> None:
        """Queue an external interrupt."""
        self._interrupts.append(irq)

    # ----- internal execution -----

    def _execute(self, insn: Instruction) -> None:
        """Dispatch and execute a single instruction."""

        def _reg(idx: int) -> int:
            if not 0 <= idx < NUM_REGISTERS:
                raise RuntimeError(f"Register index out of range: {idx}")
            return self.registers[idx]

        def _set_reg(idx: int, val: int) -> None:
            if not 0 <= idx < NUM_REGISTERS:
                raise RuntimeError(f"Register index out of range: {idx}")
            self.registers[idx] = val & 0xFFFFFFFF  # 32-bit mask

        def _signed(val: int) -> int:
            """Interpret a 32-bit unsigned value as signed."""
            val &= 0xFFFFFFFF
            return val if val < 0x80000000 else val - 0x100000000

        def _update_flags(result: int) -> None:
            result &= 0xFFFFFFFF
            self.flags_zero = result == 0
            self.flags_negative = bool(result & 0x80000000)

        op = insn.opcode

        if op == Opcodes.NOP:
            pass

        elif op == Opcodes.LOAD_CONST:
            _set_reg(insn.rd, insn.imm32 & 0xFFFFFFFF)

        elif op == Opcodes.LOAD_REG:
            addr = (_reg(insn.rs1) + insn.imm32) & 0xFFFFFFFF
            if addr + 4 > self._memory_size:
                raise RuntimeError(f"Memory read out of bounds at {addr:#x}")
            val = struct.unpack_from("<I", self.memory, addr)[0]
            _set_reg(insn.rd, val)

        elif op == Opcodes.STORE_REG:
            addr = (_reg(insn.rs1) + insn.imm32) & 0xFFFFFFFF
            if addr + 4 > self._memory_size:
                raise RuntimeError(f"Memory write out of bounds at {addr:#x}")
            struct.pack_into("<I", self.memory, addr, _reg(insn.rd) & 0xFFFFFFFF)

        elif op == Opcodes.ADD:
            result = (_signed(_reg(insn.rs1)) + _signed(_reg(insn.rs2))) & 0xFFFFFFFF
            _set_reg(insn.rd, result)
            _update_flags(result)

        elif op == Opcodes.SUB:
            result = (_signed(_reg(insn.rs1)) - _signed(_reg(insn.rs2))) & 0xFFFFFFFF
            _set_reg(insn.rd, result)
            _update_flags(result)

        elif op == Opcodes.MUL:
            result = (_signed(_reg(insn.rs1)) * _signed(_reg(insn.rs2))) & 0xFFFFFFFF
            _set_reg(insn.rd, result)
            _update_flags(result)

        elif op == Opcodes.DIV:
            divisor = _signed(_reg(insn.rs2))
            if divisor == 0:
                raise RuntimeError("Division by zero")
            result = int(_signed(_reg(insn.rs1)) / divisor) & 0xFFFFFFFF
            _set_reg(insn.rd, result)
            _update_flags(result)

        elif op == Opcodes.AND:
            result = _reg(insn.rs1) & _reg(insn.rs2)
            _set_reg(insn.rd, result)
            _update_flags(result)

        elif op == Opcodes.OR:
            result = _reg(insn.rs1) | _reg(insn.rs2)
            _set_reg(insn.rd, result)
            _update_flags(result)

        elif op == Opcodes.XOR:
            result = _reg(insn.rs1) ^ _reg(insn.rs2)
            _set_reg(insn.rd, result)
            _update_flags(result)

        elif op == Opcodes.NOT:
            result = (~_reg(insn.rs1)) & 0xFFFFFFFF
            _set_reg(insn.rd, result)
            _update_flags(result)

        elif op == Opcodes.SHL:
            result = (_reg(insn.rs1) << (_reg(insn.rs2) & 0x1F)) & 0xFFFFFFFF
            _set_reg(insn.rd, result)
            _update_flags(result)

        elif op == Opcodes.SHR:
            result = (_reg(insn.rs1) >> (_reg(insn.rs2) & 0x1F)) & 0xFFFFFFFF
            _set_reg(insn.rd, result)
            _update_flags(result)

        elif op == Opcodes.CMP:
            result = (_signed(_reg(insn.rs1)) - _signed(_reg(insn.rs2))) & 0xFFFFFFFF
            _update_flags(result)

        elif op == Opcodes.JMP:
            self.pc = insn.imm32

        elif op == Opcodes.JZ:
            if self.flags_zero:
                self.pc = insn.imm32
            else:
                self.pc += INSTRUCTION_SIZE

        elif op == Opcodes.JNZ:
            if not self.flags_zero:
                self.pc = insn.imm32
            else:
                self.pc += INSTRUCTION_SIZE

        elif op == Opcodes.CALL:
            if len(self.stack) >= self._stack_depth:
                raise RuntimeError("Stack overflow on CALL")
            self.stack.append(self.pc + INSTRUCTION_SIZE)
            self.pc = insn.imm32

        elif op == Opcodes.RET:
            if not self.stack:
                raise RuntimeError("Stack underflow on RET")
            self.pc = self.stack.pop()

        elif op == Opcodes.PUSH:
            if len(self.stack) >= self._stack_depth:
                raise RuntimeError("Stack overflow on PUSH")
            self.stack.append(_reg(insn.rd))

        elif op == Opcodes.POP:
            if not self.stack:
                raise RuntimeError("Stack underflow on POP")
            _set_reg(insn.rd, self.stack.pop())

        elif op == Opcodes.READ_IO:
            io_idx = insn.rs1
            if io_idx < NUM_GP_REGS or io_idx >= NUM_REGISTERS:
                raise RuntimeError(f"IO register index out of range: {io_idx}")
            if self._io_read_cb:
                val = self._io_read_cb(io_idx)
            else:
                val = 0
            _set_reg(insn.rd, val)

        elif op == Opcodes.WRITE_IO:
            io_idx = insn.rd
            if io_idx < NUM_GP_REGS or io_idx >= NUM_REGISTERS:
                raise RuntimeError(f"IO register index out of range: {io_idx}")
            if self._io_write_cb:
                self._io_write_cb(io_idx, _reg(insn.rs1))

        elif op == Opcodes.HALT:
            self.halted = True

        elif op == Opcodes.SLEEP:
            # In real firmware this would yield to a scheduler.
            # In simulation we just count it as a cycle.
            pass

        elif op == Opcodes.SEND:
            self._send_buffer.append((insn.rd, _reg(insn.rs1)))

        elif op == Opcodes.RECV:
            if self._recv_buffer:
                _set_reg(insn.rd, self._recv_buffer.pop(0))
            else:
                _set_reg(insn.rd, 0xFFFFFFFF)

        elif op == Opcodes.ALLOC:
            base = insn.imm32
            size = _reg(insn.rs1)
            if base + size > self._memory_size:
                raise RuntimeError("ALLOC out of memory")
            _set_reg(insn.rd, base)

        elif op == Opcodes.FREE:
            pass  # No-op in simple allocator

        elif op == Opcodes.DMA_COPY:
            dst = _reg(insn.rd)
            src = _reg(insn.rs1)
            length = insn.imm32 & 0xFFFFFFFF
            if dst + length > self._memory_size or src + length > self._memory_size:
                raise RuntimeError("DMA_COPY out of bounds")
            self.memory[dst: dst + length] = self.memory[src: src + length]

        elif op == Opcodes.INTERRUPT:
            self._interrupts.append(insn.imm32)

        elif op == Opcodes.CUSTOM_0:
            pass  # Extension slot 0

        elif op == Opcodes.CUSTOM_1:
            pass  # Extension slot 1

        else:
            raise RuntimeError(f"Unknown opcode: {op:#04x}")
