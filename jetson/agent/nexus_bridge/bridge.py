"""NEXUS git-agent bridge — Main bridge coordinator.

Coordinates between NEXUS edge runtime and git-agent fleet.
This is the primary interface for vessel agents to interact with
the git-native coordination layer.

The bridge provides:
  1. Bytecode deployment pipeline (validate → commit → deploy)
  2. Telemetry ingestion (sensor data → git commits)
  3. Trust event recording (INCREMENTS ↔ git audit trail)
  4. Safety event reporting (git commit + GitHub Issue)
  5. Mission queue management (.agent/next and .agent/done)
  6. Fleet coordination (GitHub PR/Issue when token provided)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

try:
    from git import Repo
except ImportError:
    Repo = Any  # type: ignore[assignment,misc]

from .bytecode_deployer import (
    BytecodeDeployer,
    ValidationResult,
)
from .telemetry_ingester import (
    TelemetryIngester,
    TelemetryResult,
)
from .trust_sync import (
    TrustSync,
    TrustResult,
)
from .equipment_manifest import EquipmentManifest


# ── Result types ───────────────────────────────────────────────────

@dataclass
class DeployResult:
    """Result of a bytecode deployment."""
    success: bool
    commit_hash: str = ""
    deployment_status: str = "pending"
    safety_report: dict = field(default_factory=dict)
    error: str = ""


@dataclass
class SafetyResult:
    """Result of a safety event report."""
    committed: bool
    commit_hash: str = ""
    issue_created: bool = False
    issue_number: int = 0
    event_details: dict = field(default_factory=dict)


@dataclass
class MissionResult:
    """Result of a mission completion."""
    success: bool
    mission_id: str = ""
    commit_hash: str = ""
    error: str = ""


@dataclass
class BridgeStatus:
    """Current bridge status."""
    vessel_id: str = ""
    connected: bool = False
    repo_path: str = ""
    branch: str = ""
    pending_deploys: int = 0
    pending_missions: int = 0
    trust_snapshot: dict = field(default_factory=dict)
    last_activity: str = ""


# ── Main Bridge ────────────────────────────────────────────────────

class NexusBridge:
    """Coordinates between NEXUS edge runtime and git-agent fleet.

    Usage:
        bridge = NexusBridge(
            vessel_id="vessel-001",
            repo_path="/path/to/vessel/repo",
            github_token="ghp_xxxx",  # optional
        )
        result = bridge.deploy_bytecode(
            bytecode=compiled_bytecode,
            source_reflex="heading-hold-v2",
            provenance={"author": "agent-flux", "model": "deepseek-v3"},
        )
    """

    def __init__(
        self,
        vessel_id: str,
        repo_path: str,
        github_token: str | None = None,
    ) -> None:
        self.vessel_id = vessel_id
        self.repo_path = repo_path
        self.github_token = github_token

        # Initialize repo
        self._repo: Any = None
        self._init_repo()

        # Initialize sub-modules
        self.deployer = BytecodeDeployer()
        self.ingester = TelemetryIngester(vessel_id=vessel_id)
        self.trust = TrustSync(vessel_id=vessel_id)
        self.manifest = EquipmentManifest()

        # .agent directory structure
        self._init_agent_dirs()

    def _init_repo(self) -> None:
        """Initialize or open the git repository."""
        if os.path.isdir(os.path.join(self.repo_path, ".git")):
            self._repo = Repo(self.repo_path)
        else:
            self._repo = Repo.init(self.repo_path)

    def _init_agent_dirs(self) -> None:
        """Create .agent directory structure."""
        dirs = [
            ".agent/bytecode",
            ".agent/telemetry",
            ".agent/trust",
            ".agent/safety",
            ".agent/next",
            ".agent/done",
            ".agent/manifest",
        ]
        for d in dirs:
            path = os.path.join(self.repo_path, d)
            os.makedirs(path, exist_ok=True)

        # Create .gitkeep files to preserve empty directories
        for d in dirs:
            gitkeep = os.path.join(self.repo_path, d, ".gitkeep")
            if not os.path.exists(gitkeep):
                with open(gitkeep, "w") as f:
                    f.write("")

    @property
    def repo(self) -> Any:
        """Get the git repo object."""
        if self._repo is None:
            self._init_repo()
        return self._repo

    # ── Bytecode Deployment ────────────────────────────────────────

    def deploy_bytecode(
        self,
        bytecode: bytes,
        source_reflex: str,
        provenance: dict,
    ) -> DeployResult:
        """Deploy compiled bytecode to the vessel.

        Pipeline:
          1. Validate bytecode safety
          2. Create git commit with bytecode + metadata
          3. (If GitHub token) Create PR for review
          4. On merge, deploy via Wire Protocol

        Args:
            bytecode: Compiled NEXUS bytecode.
            source_reflex: Name of the source reflex.
            provenance: Dict with author, model, timestamp, etc.

        Returns:
            DeployResult with commit hash, status, safety report.
        """
        # Step 1: Validate
        validation = self.deployer.validate_bytecode(bytecode)
        safety_report = {
            "is_valid": validation.passed,
            "errors": validation.report.errors,
            "warnings": validation.report.warnings,
            "instruction_count": validation.report.instruction_count,
            "max_stack_depth": validation.report.max_stack_depth,
            "hash_sha256": validation.report.hash_sha256,
        }

        if not validation.passed:
            return DeployResult(
                success=False,
                deployment_status="rejected",
                safety_report=safety_report,
                error=f"Safety validation failed: {'; '.join(validation.report.errors)}",
            )

        # Step 2: Commit to git
        metadata = {
            "source_reflex": source_reflex,
            "provenance": provenance,
            "vessel_id": self.vessel_id,
            "validation": {
                "is_valid": True,
                "instruction_count": validation.report.instruction_count,
                "max_stack_depth": validation.report.max_stack_depth,
                "hash_sha256": validation.report.hash_sha256,
            },
        }

        try:
            commit_hash = self.deployer.commit_bytecode(
                bytecode, metadata, self.repo
            )
        except Exception as e:
            return DeployResult(
                success=False,
                deployment_status="commit_failed",
                safety_report=safety_report,
                error=f"Git commit failed: {e}",
            )

        # Step 3: GitHub PR (if token provided)
        deployment_status = "deployed"
        if self.github_token:
            # TODO: Create GitHub PR for review
            # For now, mark as deployed (auto-merge for local-only use)
            deployment_status = "pending_review"

        return DeployResult(
            success=True,
            commit_hash=commit_hash,
            deployment_status=deployment_status,
            safety_report=safety_report,
        )

    # ── Telemetry Ingestion ────────────────────────────────────────

    def ingest_telemetry(self, telemetry_data: dict) -> TelemetryResult:
        """Ingest sensor/actuator telemetry as git commits.

        Batches telemetry into 5-minute windows.
        Commits as structured JSON in .agent/telemetry/ directory.

        Args:
            telemetry_data: Dict with keys:
                - readings: list of {sensor_id, value, timestamp}
                - (optional) sensor_type, unit

        Returns:
            TelemetryResult with commit hash and stats.
        """
        readings = telemetry_data.get("readings", [])
        sensor_type = telemetry_data.get("sensor_type", "analog")
        unit = telemetry_data.get("unit", "")

        for reading in readings:
            self.ingester.add_reading(
                sensor_id=reading.get("sensor_id", 0),
                value=reading.get("value", 0.0),
                timestamp=reading.get("timestamp", time.time()),
                sensor_type=sensor_type,
                unit=unit,
            )

        # Flush if batch is full or time window elapsed
        if (self.ingester.pending_count >= self.ingester.batch_size
                or self.ingester.should_flush(time.time())):
            return self.ingester.flush(self.repo)

        return TelemetryResult(
            committed=False,
            readings_count=self.ingester.pending_count,
        )

    # ── Trust Events ───────────────────────────────────────────────

    def record_trust_event(
        self,
        subsystem: str,
        event_type: str,
        severity: int,
        details: str,
    ) -> TrustResult:
        """Record a trust event as a git commit.

        Format: TRUST: <subsystem> <event_type> +<severity> | <details>
        Enables git log based trust audit trail.

        Args:
            subsystem: Subsystem name.
            event_type: Event type name.
            severity: Severity value (0-10, will be normalized to 0.0-1.0).
            details: Human-readable description.

        Returns:
            TrustResult with commit hash and updated score.
        """
        # Normalize severity from 0-10 scale to 0.0-1.0
        normalized_severity = min(severity / 10.0, 1.0)

        return self.trust.record_event(
            subsystem=subsystem,
            event_type=event_type,
            severity=normalized_severity,
            details=details,
            repo=self.repo,
        )

    # ── Safety Events ──────────────────────────────────────────────

    def report_safety_event(self, event: dict) -> SafetyResult:
        """Report safety-critical events.

        Creates both a git commit AND (if GitHub token) a GitHub Issue
        with RED label.

        Args:
            event: Dict with keys:
                - level: severity level (1-4)
                - subsystem: affected subsystem
                - event_type: type of safety event
                - details: human-readable description
                - timestamp: optional unix timestamp

        Returns:
            SafetyResult with commit hash and optional issue number.
        """
        level = event.get("level", 1)
        subsystem = event.get("subsystem", "unknown")
        event_type = event.get("event_type", "unknown")
        details = event.get("details", "")
        timestamp = event.get("timestamp", time.time())

        # Create safety event directory
        safety_dir = os.path.join(self.repo_path, ".agent", "safety")
        os.makedirs(safety_dir, exist_ok=True)

        # Write event JSON
        ts = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(
            "%Y%m%dT%H%M%S%z"
        )
        event_data = {
            "vessel_id": self.vessel_id,
            "level": level,
            "subsystem": subsystem,
            "event_type": event_type,
            "details": details,
            "timestamp": timestamp,
            "timestamp_iso": datetime.fromtimestamp(
                timestamp, tz=timezone.utc
            ).isoformat(),
        }

        filename = f"{ts}_L{level}_{subsystem}_{event_type}.json"
        filepath = os.path.join(safety_dir, filename)
        with open(filepath, "w") as f:
            json.dump(event_data, f, indent=2)

        # Git commit
        self.repo.index.add([filepath])
        severity_label = ["INFO", "WARNING", "CRITICAL", "EMERGENCY"][min(level - 1, 3)]
        commit_msg = (
            f"SAFETY[{severity_label}]: {subsystem} {event_type} | {details}"
        )
        commit = self.repo.index.commit(commit_msg)

        # GitHub Issue (if token provided)
        issue_created = False
        issue_number = 0
        if self.github_token and level >= 3:
            # TODO: Create GitHub Issue with RED label
            # issue_number = self._create_github_issue(event, severity_label)
            pass

        return SafetyResult(
            committed=True,
            commit_hash=commit.hexsha,
            issue_created=issue_created,
            issue_number=issue_number,
            event_details=event_data,
        )

    # ── Mission Queue ──────────────────────────────────────────────

    def get_mission_queue(self) -> list[dict]:
        """Read .agent/next for pending missions from fleet.

        Returns:
            List of mission dicts with id, description, priority, etc.
        """
        next_dir = os.path.join(self.repo_path, ".agent", "next")
        missions: list[dict] = []

        if not os.path.isdir(next_dir):
            return missions

        for filename in sorted(os.listdir(next_dir)):
            if filename.startswith(".") or not filename.endswith(".json"):
                continue
            filepath = os.path.join(next_dir, filename)
            try:
                with open(filepath) as f:
                    mission = json.load(f)
                mission["_filename"] = filename
                missions.append(mission)
            except (json.JSONDecodeError, OSError):
                continue

        return missions

    def complete_mission(
        self,
        mission_id: str,
        results: dict,
    ) -> MissionResult:
        """Mark mission complete, move from .agent/next to .agent/done.

        Args:
            mission_id: Mission identifier.
            results: Mission results dict.

        Returns:
            MissionResult with commit hash.
        """
        # Find mission in .agent/next
        next_dir = os.path.join(self.repo_path, ".agent", "next")
        done_dir = os.path.join(self.repo_path, ".agent", "done")
        os.makedirs(done_dir, exist_ok=True)

        source_file = None
        for filename in os.listdir(next_dir):
            if filename.startswith("."):
                continue
            filepath = os.path.join(next_dir, filename)
            try:
                with open(filepath) as f:
                    data = json.load(f)
                if data.get("id") == mission_id or filename.startswith(mission_id):
                    source_file = filepath
                    break
            except (json.JSONDecodeError, OSError):
                continue

        if source_file is None:
            return MissionResult(
                success=False,
                mission_id=mission_id,
                error=f"Mission {mission_id} not found in .agent/next",
            )

        # Add completion data
        try:
            with open(source_file) as f:
                mission_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            mission_data = {"id": mission_id}

        mission_data["completed_at"] = datetime.now(timezone.utc).isoformat()
        mission_data["results"] = results
        mission_data["status"] = "done"

        # Move to .agent/done
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%z")
        done_filename = f"{ts}_{mission_id}.json"
        done_filepath = os.path.join(done_dir, done_filename)
        with open(done_filepath, "w") as f:
            json.dump(mission_data, f, indent=2)

        # Remove from .agent/next and commit
        try:
            self.repo.index.remove([source_file], working_tree=True)
        except Exception:
            pass  # File may not be tracked yet
        if os.path.exists(source_file):
            os.remove(source_file)
        self.repo.index.add([done_filepath])
        commit = self.repo.index.commit(
            f"MISSION DONE: {mission_id} | {results.get('summary', 'completed')}"
        )

        return MissionResult(
            success=True,
            mission_id=mission_id,
            commit_hash=commit.hexsha,
        )

    # ── Status ─────────────────────────────────────────────────────

    def get_status(self) -> BridgeStatus:
        """Get bridge status: connected repos, pending deploys, trust snapshot."""
        try:
            branch = self.repo.active_branch.name
        except Exception:
            branch = "unknown"

        missions = self.get_mission_queue()

        return BridgeStatus(
            vessel_id=self.vessel_id,
            connected=self._repo is not None,
            repo_path=self.repo_path,
            branch=branch,
            pending_deploys=0,  # Would need PR API for accurate count
            pending_missions=len(missions),
            trust_snapshot=self.trust.get_trust_snapshot(),
            last_activity=datetime.now(timezone.utc).isoformat(),
        )
