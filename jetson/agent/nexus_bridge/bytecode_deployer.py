"""NEXUS git-agent bridge — Bytecode deployment pipeline.

Validates compiled bytecode for safety, stores it in git with metadata,
and deploys to ESP32 via NEXUS Wire Protocol.

Safety checks (mirrors jetson/reflex/safety_validator.py):
  - Instruction alignment (must be multiple of 8 bytes)
  - Opcode range validation (0x00-0x56)
  - Jump target bounds checking
  - Stack depth analysis (max 16)
  - Cycle budget verification (max 1000 instructions)
  - CLAMP_F before WRITE_PIN
  - NaN/Infinity guard in PUSH_F32
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

try:
    from git import Repo
except ImportError:
    Repo = Any  # type: ignore[assignment,misc]

# ── Constants ──────────────────────────────────────────────────────

INSTR_SIZE = 8  # Each instruction is 8 bytes

# Valid opcode ranges
CORE_OPCODE_MAX = 0x1F
A2A_OPCODE_MAX = 0x56

# Stack effects for core opcodes (positive = pushes, negative = pops)
STACK_EFFECTS: dict[int, int] = {
    0x00: 0,   # NOP
    0x01: 1,   # PUSH_I8
    0x02: 1,   # PUSH_I16
    0x03: 1,   # PUSH_F32
    0x04: -1,  # POP
    0x05: 1,   # DUP
    0x06: 0,   # SWAP
    0x07: 0,   # ROT
    0x08: -1,  # ADD_F
    0x09: -1,  # SUB_F
    0x0A: -1,  # MUL_F
    0x0B: -1,  # DIV_F
    0x0C: 0,   # NEG_F
    0x0D: 0,   # ABS_F
    0x0E: -1,  # MIN_F
    0x0F: -1,  # MAX_F
    0x10: 0,   # CLAMP_F
    0x11: -1,  # EQ_F
    0x12: -1,  # LT_F
    0x13: -1,  # GT_F
    0x14: -1,  # LTE_F
    0x15: -1,  # GTE_F
    0x16: -1,  # AND_B
    0x17: -1,  # OR_B
    0x18: -1,  # XOR_B
    0x19: 0,   # NOT_B
    0x1A: 1,   # READ_PIN
    0x1B: -1,  # WRITE_PIN
    0x1C: 1,   # READ_TIMER_MS
    0x1D: 0,   # JUMP
    0x1E: -1,  # JUMP_IF_FALSE
    0x1F: -1,  # JUMP_IF_TRUE
}
# A2A opcodes (0x20-0x56) have no stack effect
for _op in range(0x20, 0x57):
    STACK_EFFECTS[_op] = 0

DEFAULT_MAX_STACK = 16
DEFAULT_MAX_CYCLES = 1000
CLAMP_F_OPCODE = 0x10
WRITE_PIN_OPCODE = 0x1B
JUMP_OPCODES = {0x1D, 0x1E, 0x1F}


# ── Result types ───────────────────────────────────────────────────

@dataclass
class ValidationReport:
    """Detailed bytecode validation report."""
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    instruction_count: int = 0
    max_stack_depth: int = 0
    hash_sha256: str = ""
    timestamp: str = ""


@dataclass
class ValidationResult:
    """Result of bytecode validation."""
    passed: bool
    report: ValidationReport

    @classmethod
    def ok(cls, report: ValidationReport) -> ValidationResult:
        report.is_valid = True
        return cls(passed=True, report=report)

    @classmethod
    def fail(cls, report: ValidationReport) -> ValidationResult:
        report.is_valid = False
        return cls(passed=False, report=report)


# ── Instruction unpacking ──────────────────────────────────────────

def unpack_instruction(bytecode: bytes, offset: int) -> tuple[int, int, int, int]:
    """Unpack an 8-byte instruction from bytecode.

    Returns (opcode, flags, operand1, operand2).
    """
    if offset + INSTR_SIZE > len(bytecode):
        return (0xFF, 0, 0, 0)
    opcode = bytecode[offset]
    flags = bytecode[offset + 1]
    op1 = struct.unpack_from("<H", bytecode, offset + 2)[0]
    op2 = struct.unpack_from("<I", bytecode, offset + 4)[0]
    return (opcode, flags, op1, op2)


# ── Bytecode Deployer ──────────────────────────────────────────────

class BytecodeDeployer:
    """Handles bytecode validation, git storage, and deployment.

    Pipeline:
      1. validate_bytecode() — safety checks
      2. commit_bytecode()   — store in git with metadata
      3. deploy_to_device()  — send to ESP32 via Wire Protocol (stub)
    """

    def __init__(
        self,
        max_stack: int = DEFAULT_MAX_STACK,
        max_cycles: int = DEFAULT_MAX_CYCLES,
    ) -> None:
        self.max_stack = max_stack
        self.max_cycles = max_cycles

    # ── Step 1: Validate ───────────────────────────────────────────

    def validate_bytecode(self, bytecode: bytes) -> ValidationResult:
        """Run safety validation on bytecode before deployment.

        Checks:
          - Alignment (must be multiple of 8 bytes)
          - Opcode range (0x00-0x56)
          - Jump target bounds
          - Stack depth analysis
          - Cycle budget
          - CLAMP_F before WRITE_PIN
          - NaN/Infinity guard
        """
        errors: list[str] = []
        warnings: list[str] = []

        sha256 = hashlib.sha256(bytecode).hexdigest()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Alignment
        if len(bytecode) == 0:
            return ValidationResult.fail(ValidationReport(
                is_valid=False,
                errors=["Bytecode is empty"],
                hash_sha256=sha256,
                timestamp=now_iso,
            ))

        if len(bytecode) % INSTR_SIZE != 0:
            errors.append(
                f"Bytecode length {len(bytecode)} is not a multiple of {INSTR_SIZE}"
            )

        n_instr = len(bytecode) // INSTR_SIZE

        # Cycle budget
        if n_instr > self.max_cycles:
            errors.append(
                f"Instruction count {n_instr} exceeds cycle budget {self.max_cycles}"
            )

        # Per-instruction checks
        max_stack = 0
        current_stack = 0
        for i in range(n_instr):
            opcode, flags, op1, op2 = unpack_instruction(bytecode, i * INSTR_SIZE)

            # Opcode range
            if opcode > A2A_OPCODE_MAX:
                errors.append(
                    f"Invalid opcode 0x{opcode:02X} at instruction {i} "
                    f"(max allowed: 0x{A2A_OPCODE_MAX:02X})"
                )

            # Stack effect
            effect = STACK_EFFECTS.get(opcode, 0)
            current_stack += effect
            if current_stack < 0:
                errors.append(f"Stack underflow at instruction {i}")
            elif current_stack > max_stack:
                max_stack = current_stack

            # Jump bounds
            if opcode in JUMP_OPCODES:
                if op1 >= n_instr:
                    errors.append(
                        f"Jump at instruction {i} targets {op1}, "
                        f"but only {n_instr} instructions exist"
                    )

            # WRITE_PIN must be preceded by CLAMP_F
            if opcode == WRITE_PIN_OPCODE:
                found_clamp = False
                for j in range(i - 1, -1, -1):
                    prev_op, _, _, _ = unpack_instruction(bytecode, j * INSTR_SIZE)
                    if prev_op == WRITE_PIN_OPCODE:
                        break
                    if prev_op == CLAMP_F_OPCODE:
                        found_clamp = True
                        break
                if not found_clamp:
                    errors.append(
                        f"WRITE_PIN at instruction {i} not preceded by CLAMP_F"
                    )

            # NaN/Infinity guard for PUSH_F32
            if opcode == 0x03 and (flags & 0x02):
                value = struct.unpack("<f", struct.pack("<I", op2))[0]
                if math.isnan(value):
                    errors.append(f"PUSH_F32 at instruction {i} has NaN value")
                elif math.isinf(value):
                    errors.append(f"PUSH_F32 at instruction {i} has Infinity value")

        if max_stack > self.max_stack:
            errors.append(
                f"Max stack depth {max_stack} exceeds limit {self.max_stack}"
            )

        report = ValidationReport(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            instruction_count=n_instr,
            max_stack_depth=max_stack,
            hash_sha256=sha256,
            timestamp=now_iso,
        )

        if errors:
            return ValidationResult.fail(report)
        return ValidationResult.ok(report)

    # ── Step 2: Commit to git ──────────────────────────────────────

    def commit_bytecode(
        self,
        bytecode: bytes,
        metadata: dict[str, Any],
        repo: Any,
    ) -> str:
        """Store bytecode in git repo with metadata JSON.

        Directory structure:
          .agent/bytecode/<source_reflex>/
            <timestamp>.bin        — raw bytecode
            <timestamp>.meta.json  — validation report + provenance

        Args:
            bytecode: Validated bytecode bytes.
            metadata: Dict with keys like source_reflex, provenance,
                      vessel_id, etc.
            repo: gitpython Repo object (or path string).

        Returns:
            Commit hash (str).
        """
        if isinstance(repo, str):
            repo = Repo(repo)

        source_reflex = metadata.get("source_reflex", "unnamed")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%z")
        base_dir = os.path.join(".agent", "bytecode", source_reflex)
        abs_base = os.path.join(repo.working_dir, base_dir)
        os.makedirs(abs_base, exist_ok=True)

        # Write raw bytecode
        bin_path = os.path.join(abs_base, f"{timestamp}.bin")
        with open(bin_path, "wb") as f:
            f.write(bytecode)

        # Write metadata JSON
        meta_path = os.path.join(abs_base, f"{timestamp}.meta.json")
        meta = {
            "source_reflex": source_reflex,
            "timestamp": timestamp,
            "bytecode_size": len(bytecode),
            "bytecode_sha256": hashlib.sha256(bytecode).hexdigest(),
            "provenance": metadata.get("provenance", {}),
            "vessel_id": metadata.get("vessel_id", "unknown"),
            "validation": metadata.get("validation", {}),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        # Stage and commit
        repo.index.add([bin_path, meta_path])
        commit_msg = (
            f"DEPLOY: {source_reflex} | sha256={hashlib.sha256(bytecode).hexdigest()[:12]} "
            f"| size={len(bytecode)} | vessel={metadata.get('vessel_id', 'unknown')}"
        )
        commit = repo.index.commit(commit_msg)
        return commit.hexsha

    # ── Step 3: Deploy to device ───────────────────────────────────

    def deploy_to_device(
        self,
        bytecode: bytes,
        device_serial: str,
    ) -> bool:
        """Deploy bytecode to ESP32 via NEXUS Wire Protocol.

        This is a stub — actual deployment requires serial connection
        to the ESP32 via the Wire Protocol frame layer.

        Args:
            bytecode: Validated bytecode bytes.
            device_serial: Serial port path (e.g. /dev/ttyUSB0).

        Returns:
            True if deployment succeeded, False otherwise.
        """
        # TODO: Implement actual serial deployment via Wire Protocol
        # 1. Open serial connection
        # 2. Send DEVICE_ERASE message (clear existing reflex)
        # 3. Send BYTECODE_UPLOAD message with bytecode
        # 4. Wait for ACK
        # 5. Send DEPLOY message
        # For now, this is a stub that returns True
        return True

    # ── Utility: Generate safe minimal bytecode ────────────────────

    @staticmethod
    def make_nop_bytecode(count: int = 1) -> bytes:
        """Generate a bytecode blob of NOP instructions for testing."""
        return b"\x00" * (count * INSTR_SIZE)
