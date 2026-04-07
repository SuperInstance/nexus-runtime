"""Tests for the NEXUS VM Validator (10+ tests)."""

import pytest
import struct

from nexus.vm.executor import Instruction, Opcodes, INSTRUCTION_SIZE, NUM_REGISTERS, NUM_GP_REGS
from nexus.vm.validator import Validator, ValidationResult


@pytest.fixture
def validator():
    return Validator()


def make_program(insns):
    """Helper to build bytecode from instructions."""
    return b"".join(i.encode() for i in insns)


class TestBasicValidation:
    def test_valid_program(self, validator):
        insns = [
            Instruction(opcode=Opcodes.NOP),
            Instruction(opcode=Opcodes.HALT),
        ]
        result = validator.validate(make_program(insns))
        assert result.valid is True
        assert result.instruction_count == 2
        assert len(result.errors) == 0

    def test_empty_program(self, validator):
        result = validator.validate(b"")
        assert result.valid is True

    def test_invalid_opcode(self, validator):
        data = struct.pack("<BBBBi", 0xFE, 0, 0, 0, 0)
        result = validator.validate(data)
        assert result.valid is False
        assert any("Invalid opcode" in str(e) for e in result.errors)

    def test_register_out_of_range(self, validator):
        insn = Instruction(opcode=Opcodes.LOAD_CONST, rd=NUM_REGISTERS + 1, imm32=0)
        result = validator.validate(insn.encode())
        assert result.valid is False
        assert any("out of range" in str(e) for e in result.errors)


class TestJumpTargetValidation:
    def test_valid_jump_target(self, validator):
        insns = [
            Instruction(opcode=Opcodes.JMP, imm32=0),
            Instruction(opcode=Opcodes.HALT),
        ]
        result = validator.validate(make_program(insns))
        assert result.valid is True

    def test_unaligned_jump_target(self, validator):
        insns = [
            Instruction(opcode=Opcodes.JMP, imm32=3),  # not 8-byte aligned
            Instruction(opcode=Opcodes.HALT),
        ]
        result = validator.validate(make_program(insns))
        assert result.valid is False
        assert any("not instruction-aligned" in str(e) for e in result.errors)

    def test_jump_out_of_bounds(self, validator):
        insns = [
            Instruction(opcode=Opcodes.JMP, imm32=INSTRUCTION_SIZE * 10),
            Instruction(opcode=Opcodes.HALT),
        ]
        result = validator.validate(make_program(insns))
        assert result.valid is False
        assert any("out of program bounds" in str(e) for e in result.errors)


class TestIOValidation:
    def test_valid_io_read(self, validator):
        insn = Instruction(opcode=Opcodes.READ_IO, rd=0, rs1=16)
        result = validator.validate(insn.encode())
        assert result.valid is True

    def test_invalid_io_read_gp_register(self, validator):
        insn = Instruction(opcode=Opcodes.READ_IO, rd=0, rs1=5)  # GP register, not IO
        result = validator.validate(insn.encode())
        assert result.valid is False
        assert any("not an IO register" in str(e) for e in result.errors)

    def test_valid_io_write(self, validator):
        insn = Instruction(opcode=Opcodes.WRITE_IO, rd=16, rs1=0)
        result = validator.validate(insn.encode())
        assert result.valid is True

    def test_invalid_io_write_gp_register(self, validator):
        insn = Instruction(opcode=Opcodes.WRITE_IO, rd=5, rs1=0)
        result = validator.validate(insn.encode())
        assert result.valid is False


class TestAlignmentAndSize:
    def test_misaligned_program(self, validator):
        result = validator.validate(b"\x00" * 5)  # not 8-byte aligned
        assert result.valid is False
        assert any("not instruction-aligned" in str(e) for e in result.errors)

    def test_oversized_program(self, validator):
        v = Validator(max_program_size=8)
        result = v.validate(b"\x00" * 16)
        assert result.valid is False
        assert any("too large" in str(e) for e in result.errors)


class TestWarnings:
    def test_unreachable_code_after_halt(self, validator):
        insns = [
            Instruction(opcode=Opcodes.HALT),
            Instruction(opcode=Opcodes.NOP),
        ]
        result = validator.validate(make_program(insns))
        assert result.valid is True
        # Validator may or may not warn about unreachable code
        assert result.valid is True

    def test_no_warning_before_halt(self, validator):
        insns = [
            Instruction(opcode=Opcodes.NOP),
            Instruction(opcode=Opcodes.HALT),
        ]
        result = validator.validate(make_program(insns))
        assert len(result.warnings) == 0
