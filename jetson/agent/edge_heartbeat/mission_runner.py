"""NEXUS Edge Heartbeat — Mission Runner.

Executes missions from .agent/next queue. Supports mission types:
deploy_reflex, survey, calibrate, diagnostic, report.
Each mission is parsed from a text file and produces a MissionResult.
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class MissionType(str, Enum):
    """Supported mission types for edge vessel execution."""

    DEPLOY_REFLEX = "deploy_reflex"
    SURVEY = "survey"
    CALIBRATE = "calibrate"
    DIAGNOSTIC = "diagnostic"
    REPORT = "report"


class MissionStatus(str, Enum):
    """Possible outcomes of a mission execution."""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class Mission:
    """A mission parsed from the .agent/next queue.

    Format (one per line in .agent/next):
        mission_type:param1=value1,param2=value2,...,description=...

    Example:
        deploy_reflex:reflex=heading_hold,target=esp32,description=Deploy heading hold reflex
    """

    raw_line: str = ""
    mission_type: MissionType = MissionType.REPORT
    params: dict[str, str] = field(default_factory=dict)
    description: str = ""

    @property
    def name(self) -> str:
        """Short mission name for logging."""
        desc = self.description[:40] if self.description else self.mission_type.value
        return desc

    def __str__(self) -> str:
        return self.raw_line or f"{self.mission_type.value}:{self.description}"


@dataclass
class MissionResult:
    """Result of a mission execution."""

    mission: Mission
    status: MissionStatus = MissionStatus.SUCCESS
    output: str = ""
    duration_seconds: float = 0.0
    bytecode: bytes | None = None
    telemetry: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class MissionParseError(Exception):
    """Raised when a mission line cannot be parsed."""


class MissionRunner:
    """Executes missions from .agent/next queue.

    Each mission type maps to a specific execution method:
    - deploy_reflex: Compile and deploy a reflex behavior
    - survey: Execute sensor survey pattern
    - calibrate: Run calibration sequence
    - diagnostic: Run system diagnostics
    - report: Generate and submit status report
    """

    # Mission type dispatch table
    MISSION_EXECUTORS: dict[MissionType, str] = {
        MissionType.DEPLOY_REFLEX: "_execute_deploy_reflex",
        MissionType.SURVEY: "_execute_survey",
        MissionType.CALIBRATE: "_execute_calibrate",
        MissionType.DIAGNOSTIC: "_execute_diagnostic",
        MissionType.REPORT: "_execute_report",
    }

    def __init__(self, repo_path: str = "/opt/nexus-runtime") -> None:
        self.repo_path = Path(repo_path)
        self._executed_count = 0

    def load_mission(self, mission_line: str) -> Mission:
        """Parse a mission from a .agent/next file line.

        Expected format:
            mission_type:key1=value1,key2=value2,description=Free text here

        Args:
            mission_line: Raw text line from .agent/next.

        Returns:
            Parsed Mission object.

        Raises:
            MissionParseError: If line format is invalid or mission type unknown.
        """
        line = mission_line.strip()
        if not line or line.startswith("#"):
            raise MissionParseError(f"Empty or comment line: {mission_line!r}")

        # Split type from params
        if ":" not in line:
            raise MissionParseError(f"Missing colon separator: {mission_line!r}")

        type_str, param_str = line.split(":", 1)
        type_str = type_str.strip()

        try:
            mission_type = MissionType(type_str)
        except ValueError:
            valid = [mt.value for mt in MissionType]
            raise MissionParseError(
                f"Unknown mission type: {type_str!r}. Valid: {valid}"
            ) from None

        # Parse params
        params: dict[str, str] = {}
        description = ""

        if param_str.strip():
            # Split by comma, but handle description specially
            parts = param_str.split(",")
            for part in parts:
                part = part.strip()
                if "=" in part:
                    key, value = part.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if key == "description":
                        description = value
                    else:
                        params[key] = value
                else:
                    # Treat bare text as part of description
                    description = (description + " " + part).strip()

        return Mission(
            raw_line=line,
            mission_type=mission_type,
            params=params,
            description=description,
        )

    def execute_mission(self, mission: Mission) -> MissionResult:
        """Execute a parsed mission.

        Dispatches to the appropriate executor based on mission type.

        Args:
            mission: Parsed Mission to execute.

        Returns:
            MissionResult with execution outcome.
        """
        start = time.time()
        self._executed_count += 1

        executor_name = self.MISSION_EXECUTORS.get(mission.mission_type)
        if executor_name is None:
            result = MissionResult(
                mission=mission,
                status=MissionStatus.SKIPPED,
                error=f"No executor for mission type: {mission.mission_type.value}",
            )
            result.duration_seconds = time.time() - start
            return result

        try:
            executor = getattr(self, executor_name)
            result = executor(mission)
        except Exception as exc:
            result = MissionResult(
                mission=mission,
                status=MissionStatus.FAILED,
                error=str(exc),
            )

        result.duration_seconds = time.time() - start
        return result

    def generate_bytecode_for_mission(self, mission: Mission) -> bytes:
        """Convert mission parameters to AAB bytecode stub.

        Generates a minimal AAB-encoded bytecode payload for deploy_reflex
        missions. Other mission types return an empty bytecode stub.

        Args:
            mission: Mission to generate bytecode for.

        Returns:
            AAB-formatted bytecode bytes.
        """
        if mission.mission_type != MissionType.DEPLOY_REFLEX:
            return b"\x00\x00\x00\x00\x00\x00\x00\x00"

        # AAB header: NXAB magic + version + 1 instruction
        header = b"NXAB"
        header += struct.pack("<H", 0x0001)  # version
        header += struct.pack("<H", 1)  # instruction count

        # Core instruction stub: NOP (opcode 0x00) + padding
        core = b"\x00\x00\x00\x00\x00\x00\x00\x00"

        # TLV metadata for the reflex
        metadata = b""
        reflex_name = mission.params.get("reflex", "unknown").encode("utf-8")
        target = mission.params.get("target", "esp32").encode("utf-8")

        # TLV: Type description (tag 0x01)
        metadata += bytes([0x01]) + struct.pack("<H", len(b"reflex")) + b"reflex"
        # TLV: Narrative (tag 0x05) with description
        narrative = mission.description.encode("utf-8") if mission.description else b""
        if narrative:
            metadata += bytes([0x05]) + struct.pack("<H", len(narrative)) + narrative
        # TLV: Domain tag (tag 0x0B)
        metadata += bytes([0x0B]) + struct.pack("<H", len(target)) + target

        # TLV end marker
        metadata += bytes([0x00])

        return header + core + metadata

    # ===================================================================
    # Mission Executors
    # ===================================================================

    def _execute_deploy_reflex(self, mission: Mission) -> MissionResult:
        """Compile and deploy a reflex behavior.

        Stub implementation — in production, this would:
        1. Parse reflex parameters
        2. Compile to AAB bytecode
        3. Deploy via wire protocol to ESP32
        4. Verify deployment acknowledgment
        """
        reflex_name = mission.params.get("reflex", "unknown")
        target = mission.params.get("target", "esp32")

        bytecode = self.generate_bytecode_for_mission(mission)

        return MissionResult(
            mission=mission,
            status=MissionStatus.SUCCESS,
            output=f"Deployed reflex '{reflex_name}' to {target}",
            bytecode=bytecode,
            telemetry={
                "reflex_name": reflex_name,
                "target": target,
                "bytecode_size": len(bytecode),
            },
        )

    def _execute_survey(self, mission: Mission) -> MissionResult:
        """Execute sensor survey pattern.

        Stub — in production, this would trigger sensor reads
        across all configured sensors and collect telemetry.
        """
        pattern = mission.params.get("pattern", "standard")
        duration = mission.params.get("duration", "60")

        return MissionResult(
            mission=mission,
            status=MissionStatus.SUCCESS,
            output=f"Survey completed: pattern={pattern}, duration={duration}s",
            telemetry={
                "pattern": pattern,
                "duration_seconds": int(duration),
                "readings_collected": 0,
            },
        )

    def _execute_calibrate(self, mission: Mission) -> MissionResult:
        """Run calibration sequence.

        Stub — in production, this would run calibration routines
        for specified sensors or subsystems.
        """
        subsystem = mission.params.get("subsystem", "all")
        mode = mission.params.get("mode", "auto")

        return MissionResult(
            mission=mission,
            status=MissionStatus.SUCCESS,
            output=f"Calibration complete: {subsystem} ({mode})",
            telemetry={
                "subsystem": subsystem,
                "mode": mode,
            },
        )

    def _execute_diagnostic(self, mission: Mission) -> MissionResult:
        """Run system diagnostics.

        Stub — in production, this would check all subsystems,
        verify wire protocol connectivity, and report health.
        """
        scope = mission.params.get("scope", "full")
        checks = [
            "wire_protocol",
            "sensors",
            "trust_engine",
            "bytecode_vm",
            "safety_system",
        ] if scope == "full" else [scope]

        results = {check: "ok" for check in checks}

        return MissionResult(
            mission=mission,
            status=MissionStatus.SUCCESS,
            output=f"Diagnostics passed: {len(checks)} checks",
            telemetry={
                "scope": scope,
                "checks": len(checks),
                "results": results,
            },
        )

    def _execute_report(self, mission: Mission) -> MissionResult:
        """Generate and submit status report.

        Returns a status report suitable for .agent/done logging.
        """
        report_type = mission.params.get("type", "status")

        report = {
            "report_type": report_type,
            "executed_missions": self._executed_count,
            "timestamp": time.time(),
        }

        return MissionResult(
            mission=mission,
            status=MissionStatus.SUCCESS,
            output=f"Report generated: {report_type}",
            telemetry=report,
        )


# ===================================================================
# Mission Queue Helpers
# ===================================================================

def read_next_queue(agent_dir: Path) -> list[str]:
    """Read all lines from .agent/next queue file.

    Args:
        agent_dir: Path to the .agent directory.

    Returns:
        List of non-empty, non-comment lines.
    """
    next_file = agent_dir / "next"
    if not next_file.exists():
        return []
    lines = next_file.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def pop_next_mission(agent_dir: Path) -> str | None:
    """Pop the top mission from .agent/next.

    Removes the first non-empty, non-comment line from .agent/next
    and returns it. Returns None if queue is empty.

    Args:
        agent_dir: Path to the .agent directory.

    Returns:
        The mission line, or None if queue is empty.
    """
    lines = read_next_queue(agent_dir)
    if not lines:
        return None

    mission_line = lines[0]
    remaining = lines[1:]

    next_file = agent_dir / "next"
    if remaining:
        next_file.write_text("\n".join(remaining) + "\n", encoding="utf-8")
    else:
        next_file.write_text("", encoding="utf-8")

    return mission_line


def append_done(agent_dir: Path, mission_line: str, result: MissionResult) -> None:
    """Append a completed mission to .agent/done.

    Format: ISO_TIMESTAMP mission_line -> status (duration)

    Args:
        agent_dir: Path to the .agent directory.
        mission_line: Original mission line.
        result: Mission execution result.
    """
    done_file = agent_dir / "done"
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    entry = f"{timestamp} {mission_line} -> {result.status.value}"
    if result.error:
        entry += f" ({result.error})"

    with open(done_file, "a", encoding="utf-8") as f:
        f.write(entry + "\n")
