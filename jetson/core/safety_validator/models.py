"""NEXUS Safety Validator — Data structures for the 6-stage validation pipeline.

Defines SafetyReport, StageResult, SafetyViolation, and helper types
used throughout the bytecode safety validation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class SafetyViolation:
    """A single safety violation found during validation."""

    stage: str
    severity: str  # "error" | "warning" | "info"
    instruction_index: int
    opcode: int
    description: str
    remediation: str

    def __str__(self) -> str:
        sev = self.severity.upper()
        return (
            f"[{sev}] Stage:{self.stage} "
            f"Instr#{self.instruction_index} Op=0x{self.opcode:02X}: "
            f"{self.description} — Fix: {self.remediation}"
        )


@dataclass
class StageResult:
    """Result of a single validation stage."""

    stage_name: str
    passed: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    violations: list[SafetyViolation] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


@dataclass
class SafetyReport:
    """Complete result of running the full 6-stage safety validation pipeline."""

    overall_passed: bool
    stages: list[StageResult] = field(default_factory=list)
    bytecode_hash: str = ""
    bytecode_size: int = 0
    instruction_count: int = 0
    trust_level: int = 0
    timestamp: str = ""
    violations: list[SafetyViolation] = field(default_factory=list)

    @property
    def passed_stages(self) -> list[str]:
        return [s.stage_name for s in self.stages if s.passed]

    @property
    def failed_stages(self) -> list[str]:
        return [s.stage_name for s in self.stages if not s.passed]

    @property
    def total_errors(self) -> int:
        return sum(len(s.errors) for s in self.stages)

    @property
    def total_warnings(self) -> int:
        return sum(len(s.warnings) for s in self.stages)

    def summary(self) -> str:
        status = "PASS" if self.overall_passed else "FAIL"
        lines = [
            f"=== Safety Validation {status} ===",
            f"Bytecode: {self.bytecode_size} bytes, "
            f"{self.instruction_count} instructions, "
            f"hash={self.bytecode_hash[:16]}...",
            f"Trust Level: L{self.trust_level}",
            f"Timestamp: {self.timestamp}",
            f"Stages: {len(self.passed_stages)}/{len(self.stages)} passed",
        ]
        for s in self.stages:
            marker = "OK" if s.passed else "FAIL"
            lines.append(
                f"  [{marker}] {s.stage_name} "
                f"({s.duration_ms:.2f}ms, "
                f"{len(s.errors)} err, {len(s.warnings)} warn)"
            )
        if self.violations:
            lines.append(f"\nViolations ({len(self.violations)}):")
            for v in self.violations:
                lines.append(f"  {v}")
        return "\n".join(lines)


def make_timestamp() -> str:
    """Return ISO-8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()
