"""Tests for the NEXUS VM Assembler (15+ tests)."""

import pytest

from nexus.vm.assembler import Assembler, AssemblyError, DataEntry
from nexus.vm.executor import Instruction, Opcodes, INSTRUCTION_SIZE
from nexus.vm.disassembler import Disassembler


@pytest.fixture
def assembler():
    return Assembler()


class TestBasicAssembly:
    def test_nop(self, assembler):
        code = assembler.assemble("NOP")
        assert len(code) == INSTRUCTION_SIZE
        insn = Instruction.decode(code)
        assert insn.opcode == Opcodes.NOP

    def test_halt(self, assembler):
        code = assembler.assemble("HALT")
        insn = Instruction.decode(code)
        assert insn.opcode == Opcodes.HALT

    def test_load_const(self, assembler):
        code = assembler.assemble("LOAD_CONST R0, 42")
        insn = Instruction.decode(code)
        assert insn.opcode == Opcodes.LOAD_CONST
        assert insn.rd == 0
        assert insn.imm32 == 42

    def test_add(self, assembler):
        code = assembler.assemble("ADD R2, R0, R1")
        insn = Instruction.decode(code)
        assert insn.opcode == Opcodes.ADD
        assert insn.rd == 2
        assert insn.rs1 == 0
        assert insn.rs2 == 1

    def test_sub(self, assembler):
        code = assembler.assemble("SUB R3, R1, R2")
        insn = Instruction.decode(code)
        assert insn.opcode == Opcodes.SUB
        assert insn.rd == 3
        assert insn.rs1 == 1
        assert insn.rs2 == 2

    def test_mul(self, assembler):
        code = assembler.assemble("MUL R0, R1, R2")
        insn = Instruction.decode(code)
        assert insn.opcode == Opcodes.MUL

    def test_div(self, assembler):
        code = assembler.assemble("DIV R0, R1, R2")
        insn = Instruction.decode(code)
        assert insn.opcode == Opcodes.DIV

    def test_and(self, assembler):
        code = assembler.assemble("AND R0, R1, R2")
        insn = Instruction.decode(code)
        assert insn.opcode == Opcodes.AND

    def test_or(self, assembler):
        code = assembler.assemble("OR R0, R1, R2")
        insn = Instruction.decode(code)
        assert insn.opcode == Opcodes.OR


class TestComments:
    def test_semicolon_comment(self, assembler):
        code = assembler.assemble("NOP ; this is a comment")
        assert len(code) == INSTRUCTION_SIZE

    def test_hash_comment(self, assembler):
        code = assembler.assemble("NOP # this is a comment")
        assert len(code) == INSTRUCTION_SIZE

    def test_comment_only_line(self, assembler):
        code = assembler.assemble("; just a comment\nNOP")
        assert len(code) == INSTRUCTION_SIZE


class TestLabels:
    def test_label_declaration(self, assembler):
        code = assembler.assemble("start:\nNOP\nHALT")
        assert "start" in assembler.labels
        assert assembler.labels["start"] == 0

    def test_label_jump(self, assembler):
        code = assembler.assemble("loop:\nNOP\nJMP loop\nHALT")
        assert len(code) == INSTRUCTION_SIZE * 3
        # JMP should reference address 0
        jmp_insn = Instruction.decode(code, offset=INSTRUCTION_SIZE)
        assert jmp_insn.opcode == Opcodes.JMP
        assert jmp_insn.imm32 == 0


class TestDataSection:
    def test_data_section(self, assembler):
        source = """
        LOAD_CONST R0, 0
        .data
        mydata:
        42, 100, 255
        """
        code = assembler.assemble(source)
        # Should have 1 instruction + 3 data words (12 bytes)
        assert len(code) == INSTRUCTION_SIZE + 12

    def test_data_entries_recorded(self, assembler):
        source = """
        .data
        values:
        1, 2, 3
        """
        assembler.assemble(source)
        assert len(assembler.data_entries) == 1
        assert assembler.data_entries[0].values == [1, 2, 3]


class TestHexLiterals:
    def test_hex_immediate(self, assembler):
        code = assembler.assemble("LOAD_CONST R0, 0xFF")
        insn = Instruction.decode(code)
        assert insn.imm32 == 0xFF

    def test_hex_jump_target(self, assembler):
        code = assembler.assemble("LOAD_CONST R0, 0x10\nJMP 0x10\nHALT")
        assert len(code) == INSTRUCTION_SIZE * 3


class TestRegisterAliases:
    def test_sp_register(self, assembler):
        code = assembler.assemble("PUSH SP")
        insn = Instruction.decode(code)
        assert insn.rd == 28


class TestErrorHandling:
    def test_unknown_mnemonic(self, assembler):
        with pytest.raises(AssemblyError):
            assembler.assemble("INVALID_OP R0, 42")

    def test_unknown_register(self, assembler):
        with pytest.raises(AssemblyError, match="Unknown register"):
            assembler.assemble("LOAD_CONST R99, 42")

    def test_invalid_immediate(self, assembler):
        with pytest.raises(AssemblyError, match="Invalid immediate"):
            assembler.assemble("LOAD_CONST R0, abc")


class TestMultiInstruction:
    def test_multiple_instructions(self, assembler):
        source = """
        LOAD_CONST R0, 10
        LOAD_CONST R1, 20
        ADD R2, R0, R1
        HALT
        """
        code = assembler.assemble(source)
        assert len(code) == INSTRUCTION_SIZE * 4
