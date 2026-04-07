"""Comprehensive tests for the NEXUS VM Executor (25+ tests)."""

import pytest
import struct

from nexus.vm.executor import (
    Executor, Instruction, Opcodes, VMState,
    INSTRUCTION_SIZE, NUM_REGISTERS, NUM_GP_REGS, MEMORY_SIZE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def executor():
    """Fresh executor with no program."""
    return Executor()


@pytest.fixture
def simple_program():
    """LOAD_CONST R0, 42; HALT"""
    insns = [
        Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=42),
        Instruction(opcode=Opcodes.HALT),
    ]
    return b"".join(i.encode() for i in insns)


@pytest.fixture
def add_program():
    """LOAD_CONST R0, 10; LOAD_CONST R1, 20; ADD R2, R0, R1; HALT"""
    insns = [
        Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=10),
        Instruction(opcode=Opcodes.LOAD_CONST, rd=1, imm32=20),
        Instruction(opcode=Opcodes.ADD, rd=2, rs1=0, rs2=1),
        Instruction(opcode=Opcodes.HALT),
    ]
    return b"".join(i.encode() for i in insns)


# ---------------------------------------------------------------------------
# Instruction encoding/decoding
# ---------------------------------------------------------------------------

class TestInstruction:
    def test_encode_size(self):
        insn = Instruction(opcode=Opcodes.NOP)
        assert len(insn.encode()) == INSTRUCTION_SIZE

    def test_encode_decode_roundtrip(self):
        original = Instruction(opcode=Opcodes.ADD, rd=5, rs1=3, rs2=7, imm32=999)
        encoded = original.encode()
        decoded = Instruction.decode(encoded)
        assert decoded.opcode == original.opcode
        assert decoded.rd == original.rd
        assert decoded.rs1 == original.rs1
        assert decoded.rs2 == original.rs2
        assert decoded.imm32 == original.imm32

    def test_decode_with_offset(self):
        data = Instruction(opcode=Opcodes.NOP).encode() + Instruction(opcode=Opcodes.HALT).encode()
        insn = Instruction.decode(data, offset=INSTRUCTION_SIZE)
        assert insn.opcode == Opcodes.HALT

    def test_repr_contains_opcode_name(self):
        insn = Instruction(opcode=Opcodes.HALT)
        assert "HALT" in repr(insn)


# ---------------------------------------------------------------------------
# Executor initialization
# ---------------------------------------------------------------------------

class TestExecutorInit:
    def test_default_state(self, executor):
        assert executor.pc == 0
        assert len(executor.registers) == NUM_REGISTERS
        assert len(executor.stack) == 0
        assert executor.halted is False
        assert executor.cycle_count == 0

    def test_all_registers_zero(self, executor):
        assert all(r == 0 for r in executor.registers)

    def test_memory_size(self, executor):
        assert len(executor.memory) == MEMORY_SIZE

    def test_with_program(self, simple_program):
        ex = Executor(program=simple_program)
        assert len(ex.program) == INSTRUCTION_SIZE * 2


# ---------------------------------------------------------------------------
# Basic execution
# ---------------------------------------------------------------------------

class TestBasicExecution:
    def test_nop(self, executor):
        prog = Instruction(opcode=Opcodes.NOP).encode() + Instruction(opcode=Opcodes.HALT).encode()
        executor.load_program(prog)
        insn = executor.step()
        assert insn is not None
        assert insn.opcode == Opcodes.NOP
        assert executor.cycle_count == 1

    def test_halt(self, executor, simple_program):
        executor.load_program(simple_program)
        executor.step()  # LOAD_CONST
        executor.step()  # HALT
        assert executor.halted is True

    def test_step_returns_none_when_halted(self, executor, simple_program):
        executor.load_program(simple_program)
        executor.run()
        assert executor.step() is None

    def test_load_const(self, executor, simple_program):
        executor.load_program(simple_program)
        executor.step()
        assert executor.registers[0] == 42

    def test_load_const_negative(self):
        insn = Instruction(opcode=Opcodes.LOAD_CONST, rd=1, imm32=-100)
        prog = insn.encode() + Instruction(opcode=Opcodes.HALT).encode()
        ex = Executor(program=prog)
        ex.step()
        # -100 as unsigned 32-bit
        assert ex.registers[1] == (-100) & 0xFFFFFFFF

    def test_add(self, add_program):
        ex = Executor(program=add_program)
        ex.run()
        assert ex.registers[2] == 30

    def test_sub(self):
        insns = [
            Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=50),
            Instruction(opcode=Opcodes.LOAD_CONST, rd=1, imm32=20),
            Instruction(opcode=Opcodes.SUB, rd=2, rs1=0, rs2=1),
            Instruction(opcode=Opcodes.HALT),
        ]
        prog = b"".join(i.encode() for i in insns)
        ex = Executor(program=prog)
        ex.run()
        assert ex.registers[2] == 30

    def test_mul(self):
        insns = [
            Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=6),
            Instruction(opcode=Opcodes.LOAD_CONST, rd=1, imm32=7),
            Instruction(opcode=Opcodes.MUL, rd=2, rs1=0, rs2=1),
            Instruction(opcode=Opcodes.HALT),
        ]
        prog = b"".join(i.encode() for i in insns)
        ex = Executor(program=prog)
        ex.run()
        assert ex.registers[2] == 42

    def test_div(self):
        insns = [
            Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=100),
            Instruction(opcode=Opcodes.LOAD_CONST, rd=1, imm32=5),
            Instruction(opcode=Opcodes.DIV, rd=2, rs1=0, rs2=1),
            Instruction(opcode=Opcodes.HALT),
        ]
        prog = b"".join(i.encode() for i in insns)
        ex = Executor(program=prog)
        ex.run()
        assert ex.registers[2] == 20

    def test_div_by_zero_raises(self):
        insns = [
            Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=10),
            Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=10),
            Instruction(opcode=Opcodes.DIV, rd=2, rs1=0, rs2=1),  # R1=0 → div by zero
            Instruction(opcode=Opcodes.HALT),
        ]
        prog = b"".join(i.encode() for i in insns)
        ex = Executor(program=prog)
        with pytest.raises(RuntimeError, match="Division by zero"):
            ex.run()


# ---------------------------------------------------------------------------
# Bitwise operations
# ---------------------------------------------------------------------------

class TestBitwiseOps:
    def test_and(self):
        insns = [
            Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=0xFF),
            Instruction(opcode=Opcodes.LOAD_CONST, rd=1, imm32=0x0F),
            Instruction(opcode=Opcodes.AND, rd=2, rs1=0, rs2=1),
            Instruction(opcode=Opcodes.HALT),
        ]
        ex = Executor(program=b"".join(i.encode() for i in insns))
        ex.run()
        assert ex.registers[2] == 0x0F

    def test_or(self):
        insns = [
            Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=0xF0),
            Instruction(opcode=Opcodes.LOAD_CONST, rd=1, imm32=0x0F),
            Instruction(opcode=Opcodes.OR, rd=2, rs1=0, rs2=1),
            Instruction(opcode=Opcodes.HALT),
        ]
        ex = Executor(program=b"".join(i.encode() for i in insns))
        ex.run()
        assert ex.registers[2] == 0xFF

    def test_xor(self):
        insns = [
            Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=0xFF),
            Instruction(opcode=Opcodes.LOAD_CONST, rd=1, imm32=0xFF),
            Instruction(opcode=Opcodes.XOR, rd=2, rs1=0, rs2=1),
            Instruction(opcode=Opcodes.HALT),
        ]
        ex = Executor(program=b"".join(i.encode() for i in insns))
        ex.run()
        assert ex.registers[2] == 0

    def test_not(self):
        insns = [
            Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=0x0F0F0F0F),
            Instruction(opcode=Opcodes.NOT, rd=1, rs1=0),
            Instruction(opcode=Opcodes.HALT),
        ]
        ex = Executor(program=b"".join(i.encode() for i in insns))
        ex.run()
        assert ex.registers[1] == ~0x0F0F0F0F & 0xFFFFFFFF

    def test_shl(self):
        insns = [
            Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=1),
            Instruction(opcode=Opcodes.LOAD_CONST, rd=1, imm32=4),
            Instruction(opcode=Opcodes.SHL, rd=2, rs1=0, rs2=1),
            Instruction(opcode=Opcodes.HALT),
        ]
        ex = Executor(program=b"".join(i.encode() for i in insns))
        ex.run()
        assert ex.registers[2] == 16

    def test_shr(self):
        insns = [
            Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=16),
            Instruction(opcode=Opcodes.LOAD_CONST, rd=1, imm32=2),
            Instruction(opcode=Opcodes.SHR, rd=2, rs1=0, rs2=1),
            Instruction(opcode=Opcodes.HALT),
        ]
        ex = Executor(program=b"".join(i.encode() for i in insns))
        ex.run()
        assert ex.registers[2] == 4


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------

class TestControlFlow:
    def test_jmp(self):
        # JMP to addr 0 (infinite loop, limit cycles)
        insn_jmp = Instruction(opcode=Opcodes.JMP, imm32=0)
        prog = insn_jmp.encode()
        ex = Executor(program=prog)
        cycles = ex.run(max_cycles=10)
        assert cycles == 10
        assert ex.halted is False

    def test_jz_taken(self):
        # CMP R0, R0 (equal → zero flag set) → JZ to HALT
        insns = [
            Instruction(opcode=Opcodes.CMP, rs1=0, rs2=0),  # 0==0, zero flag set
            Instruction(opcode=Opcodes.JZ, imm32=INSTRUCTION_SIZE * 3),  # skip to HALT
            Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=999),  # should be skipped
            Instruction(opcode=Opcodes.HALT),
        ]
        ex = Executor(program=b"".join(i.encode() for i in insns))
        ex.run()
        assert ex.registers[0] == 0  # LOAD_CONST was skipped

    def test_jnz_taken(self):
        # LOAD_CONST R0, 5; CMP R0, R1 (5 != 0) → JNZ to addr
        insns = [
            Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=5),
            Instruction(opcode=Opcodes.CMP, rs1=0, rs2=1),  # R1=0, so not equal
            Instruction(opcode=Opcodes.JNZ, imm32=INSTRUCTION_SIZE * 4),
            Instruction(opcode=Opcodes.LOAD_CONST, rd=2, imm32=111),  # skip
            Instruction(opcode=Opcodes.HALT),
        ]
        ex = Executor(program=b"".join(i.encode() for i in insns))
        ex.run()
        assert ex.registers[2] == 0

    def test_call_ret(self):
        # CALL to subroutine at addr 16 (2 instructions), which does RET
        insns = [
            Instruction(opcode=Opcodes.CALL, imm32=INSTRUCTION_SIZE * 3),  # call subroutine at addr 24
            Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=1),  # return here (addr 8)
            Instruction(opcode=Opcodes.HALT),  # stop after return (addr 16)
            Instruction(opcode=Opcodes.LOAD_CONST, rd=1, imm32=2),  # subroutine: set R1=2 (addr 24)
            Instruction(opcode=Opcodes.RET),  # return (addr 32)
        ]
        ex = Executor(program=b"".join(i.encode() for i in insns))
        ex.run()
        assert ex.registers[0] == 1
        assert ex.registers[1] == 2

    def test_stack_overflow_on_call(self):
        ex = Executor(stack_depth=2)
        # Three nested calls
        insns = [
            Instruction(opcode=Opcodes.CALL, imm32=INSTRUCTION_SIZE),
            Instruction(opcode=Opcodes.CALL, imm32=INSTRUCTION_SIZE),
            Instruction(opcode=Opcodes.CALL, imm32=INSTRUCTION_SIZE),
            Instruction(opcode=Opcodes.HALT),
        ]
        ex.load_program(b"".join(i.encode() for i in insns))
        with pytest.raises(RuntimeError, match="Stack overflow"):
            ex.run()


# ---------------------------------------------------------------------------
# Stack operations
# ---------------------------------------------------------------------------

class TestStackOps:
    def test_push_pop(self):
        insns = [
            Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=42),
            Instruction(opcode=Opcodes.PUSH, rd=0),
            Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=0),  # clear R0
            Instruction(opcode=Opcodes.POP, rd=1),
            Instruction(opcode=Opcodes.HALT),
        ]
        ex = Executor(program=b"".join(i.encode() for i in insns))
        ex.run()
        assert ex.registers[1] == 42

    def test_stack_underflow(self):
        insn = Instruction(opcode=Opcodes.POP, rd=0)
        ex = Executor(program=insn.encode())
        with pytest.raises(RuntimeError, match="Stack underflow"):
            ex.run()


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

class TestFlags:
    def test_cmp_zero_flag(self):
        insns = [
            Instruction(opcode=Opcodes.CMP, rs1=0, rs2=0),
            Instruction(opcode=Opcodes.HALT),
        ]
        ex = Executor(program=b"".join(i.encode() for i in insns))
        ex.run()
        assert ex.flags_zero is True

    def test_cmp_negative_flag(self):
        insns = [
            Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=1),
            Instruction(opcode=Opcodes.LOAD_CONST, rd=1, imm32=2),
            Instruction(opcode=Opcodes.CMP, rs1=0, rs2=1),  # 1-2 = -1
            Instruction(opcode=Opcodes.HALT),
        ]
        ex = Executor(program=b"".join(i.encode() for i in insns))
        ex.run()
        assert ex.flags_negative is True


# ---------------------------------------------------------------------------
# IO operations
# ---------------------------------------------------------------------------

class TestIOOps:
    def test_read_io_callback(self):
        def io_read(reg_idx):
            return 1234 if reg_idx == 16 else 0

        insns = [
            Instruction(opcode=Opcodes.READ_IO, rd=0, rs1=16),
            Instruction(opcode=Opcodes.HALT),
        ]
        ex = Executor(program=b"".join(i.encode() for i in insns), io_read_cb=io_read)
        ex.run()
        assert ex.registers[0] == 1234

    def test_write_io_callback(self):
        written = []
        def io_write(reg_idx, val):
            written.append((reg_idx, val))

        insns = [
            Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=99),
            Instruction(opcode=Opcodes.WRITE_IO, rd=16, rs1=0),
            Instruction(opcode=Opcodes.HALT),
        ]
        ex = Executor(program=b"".join(i.encode() for i in insns), io_write_cb=io_write)
        ex.run()
        assert written == [(16, 99)]

    def test_read_io_invalid_register(self):
        insn = Instruction(opcode=Opcodes.READ_IO, rd=0, rs1=0)  # R0 is GP, not IO
        ex = Executor(program=insn.encode())
        with pytest.raises(RuntimeError, match="IO register index"):
            ex.run()


# ---------------------------------------------------------------------------
# SEND/RECV
# ---------------------------------------------------------------------------

class TestSendRecv:
    def test_send(self):
        insns = [
            Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=42),
            Instruction(opcode=Opcodes.LOAD_CONST, rd=1, imm32=5),
            Instruction(opcode=Opcodes.SEND, rd=0, rs1=1),
            Instruction(opcode=Opcodes.HALT),
        ]
        ex = Executor(program=b"".join(i.encode() for i in insns))
        ex.run()
        msg = ex.pop_send()
        assert msg == (0, 5)

    def test_recv_with_buffer(self):
        insns = [
            Instruction(opcode=Opcodes.RECV, rd=0),
            Instruction(opcode=Opcodes.HALT),
        ]
        ex = Executor(program=b"".join(i.encode() for i in insns))
        ex.push_recv(777)
        ex.run()
        assert ex.registers[0] == 777

    def test_recv_empty_buffer(self):
        insns = [
            Instruction(opcode=Opcodes.RECV, rd=0),
            Instruction(opcode=Opcodes.HALT),
        ]
        ex = Executor(program=b"".join(i.encode() for i in insns))
        ex.run()
        assert ex.registers[0] == 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Memory operations
# ---------------------------------------------------------------------------

class TestMemoryOps:
    def test_store_load(self):
        insns = [
            Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=0x1234),
            Instruction(opcode=Opcodes.STORE_REG, rd=0, rs1=1, imm32=100),  # [R1+100] = R0
            Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=0),  # clear R0
            Instruction(opcode=Opcodes.LOAD_REG, rd=2, rs1=1, imm32=100),  # R2 = [R1+100]
            Instruction(opcode=Opcodes.HALT),
        ]
        ex = Executor(program=b"".join(i.encode() for i in insns))
        ex.run()
        assert ex.registers[2] == 0x1234

    def test_dma_copy(self):
        insns = [
            Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=0x1000),
            Instruction(opcode=Opcodes.LOAD_CONST, rd=1, imm32=0x2000),
            Instruction(opcode=Opcodes.DMA_COPY, rd=0, rs1=1, imm32=4),
            Instruction(opcode=Opcodes.HALT),
        ]
        ex = Executor(program=b"".join(i.encode() for i in insns))
        # Write some data at 0x2000
        struct.pack_into("<I", ex.memory, 0x2000, 0xDEADBEEF)
        ex.run()
        val = struct.unpack_from("<I", ex.memory, 0x1000)[0]
        assert val == 0xDEADBEEF


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

class TestStateManagement:
    def test_get_state(self, simple_program):
        ex = Executor(program=simple_program)
        ex.step()
        state = ex.get_state()
        assert isinstance(state, VMState)
        assert state.pc == INSTRUCTION_SIZE
        assert state.registers[0] == 42

    def test_reset(self, simple_program):
        ex = Executor(program=simple_program)
        ex.run()
        ex.reset()
        assert ex.pc == 0
        assert ex.halted is False
        assert ex.registers[0] == 0
        assert ex.cycle_count == 0

    def test_load_program(self):
        ex = Executor()
        ex.load_program(b"\x00" * 16)
        assert len(ex.program) == 16
        assert ex.pc == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_program(self, executor):
        executor.load_program(b"")
        assert executor.step() is None

    def test_unknown_opcode(self):
        data = struct.pack("<BBBBi", 0xFE, 0, 0, 0, 0)
        ex = Executor(program=data)
        with pytest.raises(RuntimeError, match="Unknown opcode"):
            ex.run()

    def test_sleep_is_nop(self):
        insns = [
            Instruction(opcode=Opcodes.SLEEP, imm32=100),
            Instruction(opcode=Opcodes.HALT),
        ]
        ex = Executor(program=b"".join(i.encode() for i in insns))
        ex.run()
        assert ex.halted is True

    def test_custom_opcodes(self):
        for op in [Opcodes.CUSTOM_0, Opcodes.CUSTOM_1]:
            insns = [Instruction(opcode=op), Instruction(opcode=Opcodes.HALT)]
            ex = Executor(program=b"".join(i.encode() for i in insns))
            ex.run()
            assert ex.halted is True

    def test_alloc(self):
        insns = [
            Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=1024),
            Instruction(opcode=Opcodes.ALLOC, rd=1, rs1=0, imm32=4096),
            Instruction(opcode=Opcodes.HALT),
        ]
        ex = Executor(program=b"".join(i.encode() for i in insns))
        ex.run()
        assert ex.registers[1] == 4096
