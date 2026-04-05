"""NEXUS CI Bytecode Gate — standalone validator for CI pipelines.

Validates bytecode files against safety policies and trust levels.
Produces JSON reports suitable for PR comments and CI gates.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field, asdict
from typing import Any

from core.safety_validator.pipeline import BytecodeSafetyPipeline
from core.safety_validator.models import SafetyReport
from reflex.bytecode_emitter import unpack_instruction
from shared.opcodes import is_valid_opcode, opcode_name


@dataclass
class FileValidationResult:
    """Result for a single bytecode file."""
    filepath: str
    passed: bool
    error: str = ""
    safety_report: dict[str, Any] = field(default_factory=dict)
    instruction_count: int = 0
    file_size_bytes: int = 0
    unique_opcodes: list[str] = field(default_factory=list)


@dataclass
class GateReport:
    """Overall gate report for CI."""
    total_files: int = 0
    passed_files: int = 0
    failed_files: int = 0
    results: list[FileValidationResult] = field(default_factory=list)
    trust_level: int = 0
    policy_version: str = "1.0"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @property
    def exit_code(self) -> int:
        return 0 if self.failed_files == 0 else 1

    def summary(self) -> str:
        status = "✅ ALL PASSED" if self.exit_code == 0 else "❌ FAILURES FOUND"
        lines = [
            f"## NEXUS Bytecode Safety Gate — {status}",
            f"",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Total files | {self.total_files} |",
            f"| Passed | {self.passed_files} |",
            f"| Failed | {self.failed_files} |",
            f"| Trust Level | L{self.trust_level} |",
            f"",
        ]
        for r in self.results:
            icon = "✅" if r.passed else "❌"
            lines.append(f"- {icon} `{os.path.basename(r.filepath)}` — {r.instruction_count} instructions, {len(r.unique_opcodes)} unique opcodes")
            if r.error:
                lines.append(f"  - Error: {r.error}")
        return "\n".join(lines)


class BytecodeGate:
    """CI bytecode validation gate."""

    def __init__(self, trust_level: int = 0, policy_version: str = "1.0") -> None:
        self.trust_level = trust_level
        self.policy_version = policy_version
        self.pipeline = BytecodeSafetyPipeline(trust_level=trust_level)

    def validate_bytes(self, bytecode: bytes, filepath: str = "<memory>") -> FileValidationResult:
        """Validate a bytecode blob."""
        file_size = len(bytecode)
        result = FileValidationResult(filepath=filepath, passed=False, file_size_bytes=file_size)

        # Basic checks
        if file_size == 0:
            result.error = "Empty bytecode file"
            return result
        if file_size % 8 != 0:
            result.error = f"Misaligned: {file_size} bytes (not multiple of 8)"
            return result

        result.instruction_count = file_size // 8

        # Extract unique opcodes
        opcodes_seen = set()
        for i in range(0, file_size, 8):
            opcode, _, _, _ = unpack_instruction(bytecode, i)
            opcodes_seen.add(opcode)
        result.unique_opcodes = [opcode_name(op) for op in sorted(opcodes_seen) if is_valid_opcode(op)]

        # Run safety pipeline
        try:
            report = self.pipeline.validate(bytecode)
            result.safety_report = {
                "passed": report.overall_passed,
                "stages_passed": report.passed_stages_count if hasattr(report, 'passed_stages_count') else len(report.passed_stages),
                "stages_total": len(report.passed_stages) + len(report.failed_stages),
                "violations": [str(v) for v in report.violations],
                "total_warnings": report.total_warnings,
            }
            result.passed = report.overall_passed
            if not result.passed:
                result.error = "; ".join(str(v) for v in report.violations)
        except Exception as exc:
            result.error = f"Pipeline error: {exc}"

        return result

    def validate_file(self, filepath: str) -> FileValidationResult:
        """Validate a bytecode file from disk."""
        if not os.path.exists(filepath):
            return FileValidationResult(filepath=filepath, passed=False, error="File not found")
        try:
            with open(filepath, 'rb') as f:
                bytecode = f.read()
        except IOError as exc:
            return FileValidationResult(filepath=filepath, passed=False, error=str(exc))
        return self.validate_bytes(bytecode, filepath)

    def validate_directory(self, dirpath: str, extensions: tuple[str, ...] = (".bin", ".bytecode", ".nxb")) -> GateReport:
        """Validate all bytecode files in a directory."""
        report = GateReport(trust_level=self.trust_level, policy_version=self.policy_version)
        if not os.path.isdir(dirpath):
            report.total_files = 0
            return report

        for root, _dirs, files in os.walk(dirpath):
            for fname in files:
                if any(fname.endswith(ext) for ext in extensions):
                    fpath = os.path.join(root, fname)
                    result = self.validate_file(fpath)
                    report.results.append(result)
                    report.total_files += 1
                    if result.passed:
                        report.passed_files += 1
                    else:
                        report.failed_files += 1

        return report
