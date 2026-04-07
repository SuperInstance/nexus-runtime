"""
NEXUS VM Disassembler — converts bytecode back to human-readable mnemonics.

Usage::

    dis = Disassembler()
    text = dis.disassemble(bytecode)
    for line in dis.disassemble_lines(bytecode):
        print(line)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from nexus.vm.executor import INSTRUCTION_SIZE, Instruction, Opcodes

# Register name mapping (index -> name)
_REG_NAMES: Dict[int, str] = {i: f"R{i}" for i in range(32)}
_REG_NAMES[28] = "SP"
_REG_NAMES[29] = "LR"
_REG_NAMES[30] = "FP"
_REG_NAMES[31] = "PC"


class Disassembler:
    """Disassembles NEXUS VM bytecode into mnemonic assembly text.

    Parameters
    ----------
    origin : int
        Base address to subtract from displayed addresses (default 0).
    """

    def __init__(self, origin: int = 0) -> None:
        self.origin = origin

    def disassemble(self, code: bytes) -> str:
        """Disassemble *code* and return formatted multi-line string."""
        lines = self.disassemble_lines(code)
        return "\n".join(lines)

    def disassemble_lines(self, code: bytes) -> List[str]:
        """Disassemble *code* and return list of formatted lines."""
        lines: List[str] = []
        offset = 0
        while offset + INSTRUCTION_SIZE <= len(code):
            insn = Instruction.decode(code, offset)
            addr = offset + self.origin
            mnemonic = self._format_instruction(insn)
            lines.append(f"  {addr:06x}:  {mnemonic}")
            offset += INSTRUCTION_SIZE
        if offset < len(code):
            lines.append(f"  {offset + self.origin:06x}:  .data {code[offset:].hex()}")
        return lines

    def disassemble_instruction(self, code: bytes, offset: int = 0) -> str:
        """Disassemble a single instruction at *offset*."""
        if offset + INSTRUCTION_SIZE > len(code):
            return "  ???"
        insn = Instruction.decode(code, offset)
        addr = offset + self.origin
        return f"  {addr:06x}:  {self._format_instruction(insn)}"

    # ----- internal -----

    def _reg(self, idx: int) -> str:
        return _REG_NAMES.get(idx, f"R{idx}")

    def _imm(self, val: int) -> str:
        # Display as signed if it looks like a small negative offset
        if val > 0x7FFFFFFF:
            return str(val - 0x100000000)
        return str(val)

    def _hex(self, val: int) -> str:
        return f"0x{val:x}"

    def _format_instruction(self, insn: Instruction) -> str:
        """Format an instruction as a mnemonic string."""
        op = insn.opcode
        try:
            name = Opcodes(op).name
        except ValueError:
            name = f"UNKNOWN_{op:#04x}"

        rd = self._reg(insn.rd)
        rs1 = self._reg(insn.rs1)
        rs2 = self._reg(insn.rs2)
        imm = self._hex(insn.imm32) if insn.imm32 >= 0 else self._imm(insn.imm32)

        if op == Opcodes.NOP:
            return name
        if op == Opcodes.HALT:
            return name
        if op == Opcodes.LOAD_CONST:
            return f"{name} {rd}, {insn.imm32}"
        if op == Opcodes.ADD or op == Opcodes.SUB or op == Opcodes.MUL or op == Opcodes.DIV:
            return f"{name} {rd}, {rs1}, {rs2}"
        if op == Opcodes.AND or op == Opcodes.OR or op == Opcodes.XOR:
            return f"{name} {rd}, {rs1}, {rs2}"
        if op == Opcodes.CMP:
            return f"{name} {rs1}, {rs2}"
        if op == Opcodes.SHL or op == Opcodes.SHR:
            return f"{name} {rd}, {rs1}, {rs2}"
        if op == Opcodes.NOT:
            return f"{name} {rd}, {rs1}"
        if op == Opcodes.LOAD_REG:
            extra = f", {imm}" if insn.imm32 else ""
            return f"{name} {rd}, [{rs1}{extra}]"
        if op == Opcodes.STORE_REG:
            extra = f", {imm}" if insn.imm32 else ""
            return f"{name} {rd}, [{rs1}{extra}]"
        if op in (Opcodes.JMP, Opcodes.CALL):
            return f"{name} {imm}"
        if op in (Opcodes.JZ, Opcodes.JNZ):
            return f"{name} {imm}"
        if op == Opcodes.PUSH:
            return f"{name} {rd}"
        if op == Opcodes.POP:
            return f"{name} {rd}"
        if op == Opcodes.READ_IO:
            return f"{name} {rd}, {rs1}"
        if op == Opcodes.WRITE_IO:
            return f"{name} {rd}, {rs1}"
        if op == Opcodes.SLEEP:
            return f"{name} {insn.imm32}"
        if op == Opcodes.SEND:
            return f"{name} {rd}, {rs1}"
        if op == Opcodes.RECV:
            return f"{name} {rd}"
        if op == Opcodes.ALLOC:
            return f"{name} {rd}, {imm}"
        if op == Opcodes.FREE:
            return f"{name} {rd}"
        if op == Opcodes.DMA_COPY:
            return f"{name} {rd}, {rs1}, {imm}"
        if op == Opcodes.INTERRUPT:
            return f"{name} {imm}"
        if op in (Opcodes.CUSTOM_0, Opcodes.CUSTOM_1):
            return f"{name} {rd}, {rs1}, {imm}"
        return f"{name} {rd}, {rs1}, {rs2}, {imm}"
