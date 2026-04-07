"""
NEXUS VM Assembler — text-based assembler for the NEXUS bytecode VM.

Supports:
    - Mnemonic-based assembly with named registers (R0-R31)
    - Labels for jump/call targets
    - .data section for embedded constants
    - Comments (; and #)
    - Hexadecimal (0x), decimal, and binary (0b) literals

Example::

    LOAD_CONST R0, 42
    ADD R1, R0, R1
    loop:
    CMP R0, R2
    JNZ loop
    HALT
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from nexus.vm.executor import INSTRUCTION_SIZE, Instruction, Opcodes

# ---------------------------------------------------------------------------
# Register name mapping
# ---------------------------------------------------------------------------

_REG_NAMES: Dict[str, int] = {f"R{i}": i for i in range(32)}
_REG_NAMES["SP"] = 28  # stack pointer alias
_REG_NAMES["LR"] = 29  # link register alias
_REG_NAMES["FP"] = 30  # frame pointer alias
_REG_NAMES["PC"] = 31  # program counter alias

# Reverse mapping for disassembly
_REG_BY_IDX: Dict[int, str] = {v: k for k, v in _REG_NAMES.items()}

# Opcode mnemonic -> value
_OPCODE_NAMES: Dict[str, int] = {op.name: op.value for op in Opcodes}

# Data section directive prefix
_DATA_DIRECTIVE = ".data"


@dataclass
class AssemblyError(Exception):
    """Error during assembly with optional location info."""

    line: int = 0
    message: str = ""

    def __str__(self) -> str:
        if self.line:
            return f"Line {self.line}: {self.message}"
        return self.message


@dataclass
class DataEntry:
    """A single .data section entry."""

    label: str
    offset: int
    values: List[int]


class Assembler:
    """Two-pass text-based assembler for the NEXUS VM.

    Usage::

        asm = Assembler()
        bytecode = asm.assemble(source_text)
    """

    def __init__(self) -> None:
        self.labels: Dict[str, int] = {}
        self.data_entries: List[DataEntry] = []
        self.errors: List[AssemblyError] = []
        self._instructions: List[Instruction] = []
        self._data_bytes: bytearray = bytearray()
        self._source_lines: List[str] = []

    def assemble(self, source: str) -> bytes:
        """Assemble source text into bytecode.

        Raises ``AssemblyError`` on first fatal error.
        """
        self.labels.clear()
        self.data_entries.clear()
        self.errors.clear()
        self._instructions.clear()
        self._data_bytes = bytearray()
        self._source_lines = source.splitlines()

        self._pass1()  # collect labels and data
        self._pass2()  # emit bytecode
        return self._emit()

    def assemble_file(self, path: str | Path) -> bytes:
        """Read a file and assemble it."""
        text = Path(path).read_text()
        return self.assemble(text)

    # ----- internal -----

    def _parse_register(self, token: str, line: int) -> int:
        """Parse a register token like R0 or SP."""
        upper = token.upper()
        if upper not in _REG_NAMES:
            raise AssemblyError(line, f"Unknown register: {token}")
        return _REG_NAMES[upper]

    def _parse_immediate(self, token: str, line: int) -> int:
        """Parse an integer literal (decimal, hex, or binary)."""
        token = token.strip()
        try:
            if token.startswith("0x") or token.startswith("0X"):
                return int(token, 16)
            elif token.startswith("0b") or token.startswith("0B"):
                return int(token, 2)
            return int(token)
        except ValueError:
            raise AssemblyError(line, f"Invalid immediate value: {token}")

    def _strip_comment(self, line: str) -> str:
        """Remove comments (``;`` and ``#``)."""
        # Find first comment character not inside a string
        result = []
        in_string = False
        for ch in line:
            if ch == '"':
                in_string = not in_string
            if ch in (';', '#') and not in_string:
                break
            result.append(ch)
        return "".join(result).strip()

    def _parse_instruction_line(
        self, tokens: List[str], line_num: int
    ) -> Optional[Instruction]:
        """Parse a list of tokens into an Instruction."""
        if not tokens:
            return None

        mnemonic = tokens[0].upper()
        if mnemonic not in _OPCODE_NAMES:
            raise AssemblyError(line_num, f"Unknown mnemonic: {mnemonic}")

        opcode = _OPCODE_NAMES[mnemonic]

        # Parse operands (tokens are already comma-split during tokenization)
        operands: List[str] = tokens[1:]

        rd = 0
        rs1 = 0
        rs2 = 0
        imm32 = 0

        if opcode in (
            Opcodes.LOAD_CONST, Opcodes.JMP, Opcodes.JZ, Opcodes.JNZ,
            Opcodes.CALL, Opcodes.SLEEP, Opcodes.INTERRUPT, Opcodes.ALLOC,
        ):
            # One explicit operand: reg or label/immediate
            if opcode == Opcodes.LOAD_CONST:
                rd = self._parse_register(operands[0], line_num)
                imm32 = self._parse_immediate(operands[1], line_num) if len(operands) > 1 else 0
            elif opcode in (Opcodes.JMP, Opcodes.JZ, Opcodes.JNZ, Opcodes.CALL):
                if operands:
                    try:
                        imm32 = self._parse_immediate(operands[0], line_num)
                    except AssemblyError:
                        imm32 = 0  # placeholder for label resolution in pass2
                else:
                    imm32 = 0
            elif opcode == Opcodes.SLEEP:
                imm32 = self._parse_immediate(operands[0], line_num) if operands else 0
            elif opcode == Opcodes.INTERRUPT:
                imm32 = self._parse_immediate(operands[0], line_num) if operands else 0
            elif opcode == Opcodes.ALLOC:
                rd = self._parse_register(operands[0], line_num) if operands else 0
                imm32 = self._parse_immediate(operands[1], line_num) if len(operands) > 1 else 0

        elif opcode == Opcodes.HALT or opcode == Opcodes.NOP:
            pass

        elif opcode in (Opcodes.NOT,):
            rd = self._parse_register(operands[0], line_num)
            rs1 = self._parse_register(operands[1], line_num)

        elif opcode in (
            Opcodes.LOAD_REG, Opcodes.STORE_REG, Opcodes.PUSH, Opcodes.POP,
            Opcodes.READ_IO, Opcodes.RECV, Opcodes.SEND, Opcodes.WRITE_IO, Opcodes.FREE,
        ):
            rd = self._parse_register(operands[0], line_num)
            rs1 = self._parse_register(operands[1], line_num) if len(operands) > 1 else 0
            imm32 = self._parse_immediate(operands[2], line_num) if len(operands) > 2 else 0

        elif opcode == Opcodes.DMA_COPY:
            rd = self._parse_register(operands[0], line_num)
            rs1 = self._parse_register(operands[1], line_num) if len(operands) > 1 else 0
            imm32 = self._parse_immediate(operands[2], line_num) if len(operands) > 2 else 0

        elif opcode in (
            Opcodes.ADD, Opcodes.SUB, Opcodes.MUL, Opcodes.DIV,
            Opcodes.AND, Opcodes.OR, Opcodes.XOR, Opcodes.CMP,
            Opcodes.SHL, Opcodes.SHR,
        ):
            rd = self._parse_register(operands[0], line_num)
            rs1 = self._parse_register(operands[1], line_num)
            rs2 = self._parse_register(operands[2], line_num) if len(operands) > 2 else 0

        else:
            # CUSTOM_0, CUSTOM_1, etc.
            rd = self._parse_register(operands[0], line_num) if operands else 0
            rs1 = self._parse_register(operands[1], line_num) if len(operands) > 1 else 0
            imm32 = self._parse_immediate(operands[2], line_num) if len(operands) > 2 else 0

        return Instruction(opcode=opcode, rd=rd, rs1=rs1, rs2=rs2, imm32=imm32)

    def _pass1(self) -> None:
        """First pass: collect labels and data section, compute offsets."""
        in_data = False
        current_data_label: Optional[str] = None
        instr_index = 0

        for line_idx, raw_line in enumerate(self._source_lines, 1):
            line = self._strip_comment(raw_line).strip()
            if not line:
                continue

            # Data section toggle
            if line.upper().startswith(_DATA_DIRECTIVE.upper()):
                in_data = True
                continue

            # Handle labels
            if line.endswith(":"):
                label = line[:-1].strip()
                if in_data:
                    self.labels[label] = len(self._data_bytes)
                    current_data_label = label
                else:
                    self.labels[label] = instr_index * INSTRUCTION_SIZE
                continue

            # Data entries
            if in_data:
                values = [self._parse_immediate(v.strip(), line_idx) for v in line.split(",")]
                entry = DataEntry(
                    label=current_data_label or f"data_{len(self.data_entries)}",
                    offset=len(self._data_bytes),
                    values=values,
                )
                self.data_entries.append(entry)
                for v in values:
                    self._data_bytes.extend(struct.pack("<I", v & 0xFFFFFFFF))
                current_data_label = None
                continue

            # Regular instruction — just count
            instr_index += 1

    def _pass2(self) -> None:
        """Second pass: parse instructions and resolve labels."""
        in_data = False

        for line_idx, raw_line in enumerate(self._source_lines, 1):
            line = self._strip_comment(raw_line).strip()
            if not line:
                continue

            if line.upper().startswith(_DATA_DIRECTIVE.upper()):
                in_data = True
                continue

            # Skip labels
            if line.endswith(":"):
                continue

            # Skip data entries
            if in_data:
                continue

            # Tokenize
            tokens = line.replace(",", " ").split()
            insn = self._parse_instruction_line(tokens, line_idx)
            if insn is not None:
                # Resolve label references in immediate field for jumps/calls
                if insn.opcode in (Opcodes.JMP, Opcodes.JZ, Opcodes.JNZ, Opcodes.CALL):
                    operand_str = tokens[1] if len(tokens) > 1 else "0"
                    if operand_str in self.labels:
                        insn.imm32 = self.labels[operand_str]
                self._instructions.append(insn)

    def _emit(self) -> bytes:
        """Emit final bytecode: instructions followed by data."""
        out = bytearray()
        for insn in self._instructions:
            out.extend(insn.encode())
        out.extend(self._data_bytes)
        return bytes(out)
