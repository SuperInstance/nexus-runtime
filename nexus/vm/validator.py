"""
NEXUS VM Bytecode Validator — validates bytecode programs before execution.

Checks performed:
    - Instruction boundary alignment (8-byte instructions)
    - Register index ranges (0-31)
    - Jump/call target alignment
    - Known opcode values
    - Program length sanity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from nexus.vm.executor import INSTRUCTION_SIZE, Instruction, Opcodes, NUM_REGISTERS, NUM_GP_REGS

# All valid opcode values
_VALID_OPCODES = {op.value for op in Opcodes}


@dataclass
class ValidationError:
    """A single validation error with location info."""

    offset: int
    message: str
    severity: str = "error"  # "error" or "warning"

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] @ {self.offset:#06x}: {self.message}"


@dataclass
class ValidationResult:
    """Result of bytecode validation."""

    valid: bool = True
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    instruction_count: int = 0

    @property
    def all_issues(self) -> List[ValidationError]:
        return self.errors + self.warnings


class Validator:
    """Validates NEXUS VM bytecode programs.

    Usage::

        v = Validator()
        result = v.validate(bytecode)
        if not result.valid:
            for err in result.errors:
                print(err)
    """

    def __init__(
        self,
        max_program_size: int = 65536,
        allow_custom_opcodes: bool = True,
    ) -> None:
        self.max_program_size = max_program_size
        self.allow_custom_opcodes = allow_custom_opcodes

    def validate(self, code: bytes) -> ValidationResult:
        """Validate bytecode and return a :class:`ValidationResult`."""
        result = ValidationResult()

        # Check total size
        if len(code) > self.max_program_size:
            result.errors.append(
                ValidationError(
                    offset=0,
                    message=f"Program too large: {len(code)} bytes (max {self.max_program_size})",
                )
            )

        # Check alignment
        if len(code) % INSTRUCTION_SIZE != 0:
            result.errors.append(
                ValidationError(
                    offset=len(code),
                    message=f"Program not instruction-aligned: {len(code)} % {INSTRUCTION_SIZE} != 0",
                )
            )

        # Walk instructions
        num_instructions = len(code) // INSTRUCTION_SIZE
        result.instruction_count = num_instructions

        jump_targets_seen: set = set()

        for i in range(num_instructions):
            offset = i * INSTRUCTION_SIZE
            insn = Instruction.decode(code, offset)
            self._validate_instruction(insn, offset, num_instructions, result, jump_targets_seen)

        # Check for unreachable code after HALT (warning)
        halt_found = False
        for i in range(num_instructions):
            insn = Instruction.decode(code, i * INSTRUCTION_SIZE)
            if insn.opcode == Opcodes.HALT:
                halt_found = True
            elif halt_found and insn.opcode != Opcodes.NOP:
                result.warnings.append(
                    ValidationError(
                        offset=i * INSTRUCTION_SIZE,
                        message="Unreachable code after HALT",
                        severity="warning",
                    )
                )
                break

        # Check that jump targets are within bounds
        for target in jump_targets_seen:
            if target < 0 or target >= len(code):
                result.errors.append(
                    ValidationError(
                        offset=0,
                        message=f"Jump target {target:#06x} out of program bounds",
                    )
                )
            elif target % INSTRUCTION_SIZE != 0:
                result.errors.append(
                    ValidationError(
                        offset=0,
                        message=f"Jump target {target:#06x} not instruction-aligned",
                    )
                )

        result.valid = len(result.errors) == 0
        return result

    def _validate_instruction(
        self,
        insn: Instruction,
        offset: int,
        total_insns: int,
        result: ValidationResult,
        jump_targets: set,
    ) -> None:
        """Validate a single instruction."""
        # Check opcode
        if insn.opcode not in _VALID_OPCODES:
            result.errors.append(
                ValidationError(offset, f"Invalid opcode: {insn.opcode:#04x}")
            )
            return

        # Check register ranges
        for reg_val, reg_name in [(insn.rd, "rd"), (insn.rs1, "rs1"), (insn.rs2, "rs2")]:
            if reg_val >= NUM_REGISTERS:
                result.errors.append(
                    ValidationError(offset, f"Register {reg_name}={reg_val} out of range (0-{NUM_REGISTERS - 1})")
                )

        # Check IO register access
        if insn.opcode == Opcodes.READ_IO:
            if insn.rs1 < NUM_GP_REGS:
                result.errors.append(
                    ValidationError(offset, f"READ_IO rs1={insn.rs1} is not an IO register (need R16-R31)")
                )
        if insn.opcode == Opcodes.WRITE_IO:
            if insn.rd < NUM_GP_REGS:
                result.errors.append(
                    ValidationError(offset, f"WRITE_IO rd={insn.rd} is not an IO register (need R16-R31)")
                )

        # Collect jump targets
        if insn.opcode in (Opcodes.JMP, Opcodes.JZ, Opcodes.JNZ, Opcodes.CALL):
            target = insn.imm32
            if target < 0:
                # Convert negative values
                target = target & 0xFFFFFFFF
            jump_targets.add(target)
