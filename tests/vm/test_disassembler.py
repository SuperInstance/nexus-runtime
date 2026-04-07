"""Tests for the NEXUS VM Disassembler (10+ tests)."""

import pytest

from nexus.vm.executor import Instruction, Opcodes, INSTRUCTION_SIZE
from nexus.vm.disassembler import Disassembler


@pytest.fixture
def dis():
    return Disassembler()


class TestBasicDisassembly:
    def test_nop(self, dis):
        code = Instruction(opcode=Opcodes.NOP).encode()
        text = dis.disassemble(code)
        assert "NOP" in text

    def test_halt(self, dis):
        code = Instruction(opcode=Opcodes.HALT).encode()
        text = dis.disassemble(code)
        assert "HALT" in text

    def test_load_const(self, dis):
        code = Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=42).encode()
        text = dis.disassemble(code)
        assert "LOAD_CONST" in text
        assert "42" in text

    def test_add(self, dis):
        code = Instruction(opcode=Opcodes.ADD, rd=2, rs1=0, rs2=1).encode()
        text = dis.disassemble(code)
        assert "ADD" in text
        assert "R2" in text

    def test_sub(self, dis):
        code = Instruction(opcode=Opcodes.SUB, rd=0, rs1=1, rs2=2).encode()
        text = dis.disassemble(code)
        assert "SUB" in text

    def test_jmp(self, dis):
        code = Instruction(opcode=Opcodes.JMP, imm32=16).encode()
        text = dis.disassemble(code)
        assert "JMP" in text

    def test_multiple_instructions(self, dis):
        insns = [
            Instruction(opcode=Opcodes.LOAD_CONST, rd=0, imm32=1),
            Instruction(opcode=Opcodes.LOAD_CONST, rd=1, imm32=2),
            Instruction(opcode=Opcodes.ADD, rd=2, rs1=0, rs2=1),
            Instruction(opcode=Opcodes.HALT),
        ]
        code = b"".join(i.encode() for i in insns)
        lines = dis.disassemble_lines(code)
        assert len(lines) == 4

    def test_addresses_shown(self, dis):
        code = Instruction(opcode=Opcodes.NOP).encode()
        lines = dis.disassemble_lines(code)
        assert "000000" in lines[0]

    def test_disassemble_single_instruction(self, dis):
        code = Instruction(opcode=Opcodes.HALT).encode()
        text = dis.disassemble_instruction(code)
        assert "HALT" in text

    def test_origin_offset(self):
        dis = Disassembler(origin=0x1000)
        code = Instruction(opcode=Opcodes.NOP).encode()
        lines = dis.disassemble_lines(code)
        assert "001000" in lines[0]

    def test_trailing_data(self, dis):
        code = Instruction(opcode=Opcodes.NOP).encode() + b"\xDE\xAD"
        lines = dis.disassemble_lines(code)
        assert len(lines) == 2
        assert ".data" in lines[1]
