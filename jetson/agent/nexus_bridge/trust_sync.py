"""NEXUS git-agent bridge — Trust synchronization.

Synchronizes INCREMENTS trust events between the NEXUS trust engine
and git log. Every trust event is recorded as a git commit, creating
an immutable audit trail that can be parsed and recomputed.

Trust commit format:
    TRUST: <subsystem> <event_type> +<severity> | <details>

Example:
    TRUST: steering sensor_invalid +0.5 | IMU reading out of range
    TRUST: navigation heartbeat_ok +0.0 | Normal heartbeat received
    TRUST: engine actuator_overrange +0.7 | PWM exceeded 100%

This enables:
    1. Git log based trust audit trail
    2. Trust recomputation from history
    3. Cross-vessel trust comparison
    4. Regulatory compliance audit
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

try:
    from git import Repo
except ImportError:
    Repo = Any  # type: ignore[assignment,misc]

try:
    from trust.increments import (
        IncrementTrustEngine,
        TrustEvent,
        TrustParams,
    )
except ImportError:
    # Standalone fallback when not running from jetson/
    IncrementTrustEngine = None  # type: ignore[assignment,misc]
    TrustEvent = None  # type: ignore[assignment,misc]
    TrustParams = None  # type: ignore[assignment,misc]


# ── Constants ──────────────────────────────────────────────────────

TRUST_DIR = ".agent/trust"
TRUST_COMMIT_PREFIX = "TRUST:"
TRUST_COMMIT_PATTERN = re.compile(
    r"^TRUST:\s+(\S+)\s+(\S+)\s+([+-]\d+\.?\d*)\s*\|\s*(.*)$"
)

# INCREMENTS algorithm parameters (matching trust/increments.py)
DEFAULT_PARAMS = {
    "alpha_gain": 0.002,
    "alpha_loss": 0.05,
    "alpha_decay": 0.0001,
    "t_floor": 0.2,
}


# ── Data types ─────────────────────────────────────────────────────

@dataclass
class TrustRecord:
    """A parsed trust event from git log."""
    commit_hash: str
    subsystem: str
    event_type: str
    severity: float
    details: str
    timestamp: str
    is_bad: bool = False


@dataclass
class TrustResult:
    """Result of a trust event recording."""
    committed: bool
    commit_hash: str = ""
    subsystem: str = ""
    event_type: str = ""
    new_trust_score: float = 0.0


# ── Trust Sync ─────────────────────────────────────────────────────

class TrustSync:
    """Synchronizes trust events between INCREMENTS engine and git log.

    Maintains a dual record:
      1. In-memory INCREMENTS trust engine (real-time)
      2. Git commit log (persistent audit trail)

    Trust scores can be recomputed from git log at any time,
    providing verifiable consistency between the two records.
    """

    def __init__(
        self,
        vessel_id: str = "unknown",
        params: dict[str, float] | None = None,
    ) -> None:
        self.vessel_id = vessel_id
        self._params = params or DEFAULT_PARAMS

        # Initialize INCREMENTS engine if available
        if IncrementTrustEngine is not None:
            self._engine = IncrementTrustEngine(
                params=TrustParams(**self._params)  # type: ignore[arg-type]
            )
            self._engine.register_all_subsystems()
        else:
            self._engine = _StandaloneTrustEngine(self._params)

    def record_event(
        self,
        subsystem: str,
        event_type: str,
        severity: float,
        details: str = "",
        repo: Any = None,
    ) -> TrustResult:
        """Create git commit for a trust event.

        Format: TRUST: <subsystem> <event_type> +<severity> | <details>

        Also updates the in-memory INCREMENTS engine.

        Args:
            subsystem: Subsystem name (e.g. "steering", "navigation").
            event_type: Event type (e.g. "sensor_invalid", "heartbeat_ok").
            severity: Severity value (0.0-1.0, higher = more severe).
            details: Human-readable description.
            repo: gitpython Repo object (or path string). Required for commit.

        Returns:
            TrustResult with commit hash and updated trust score.
        """
        # Update in-memory engine
        is_bad = severity > 0
        if TrustEvent is not None:
            if is_bad:
                event = TrustEvent.bad(
                    event_type=event_type,
                    severity=severity,
                    timestamp=time.time(),
                    subsystem=subsystem,
                )
            else:
                event = TrustEvent.good(
                    event_type=event_type,
                    quality=0.7,
                    timestamp=time.time(),
                    subsystem=subsystem,
                )
            self._engine.record_event(event)
        else:
            self._engine.record_event(subsystem, event_type, severity)

        trust_score = self._engine.get_trust_score(subsystem)

        # Write git commit if repo provided
        commit_hash = ""
        if repo is not None:
            if isinstance(repo, str):
                repo = Repo(repo)
            commit_hash = self._commit_trust_event(
                repo, subsystem, event_type, severity, details
            )

        return TrustResult(
            committed=bool(commit_hash),
            commit_hash=commit_hash,
            subsystem=subsystem,
            event_type=event_type,
            new_trust_score=trust_score,
        )

    def get_trust_history(
        self,
        subsystem: str,
        since: str | None = None,
        repo: Any = None,
    ) -> list[dict]:
        """Parse git log for trust events.

        Args:
            subsystem: Filter by subsystem name.
            since: Git log since expression (e.g. "2 weeks ago", commit hash).
            repo: gitpython Repo object (or path string).

        Returns:
            List of trust event dicts with commit_hash, subsystem,
            event_type, severity, details, timestamp.
        """
        if repo is None:
            return []

        if isinstance(repo, str):
            repo = Repo(repo)

        records: list[dict] = []
        log_kwargs: dict[str, Any] = {"paths": TRUST_DIR}
        if since:
            log_kwargs["since"] = since

        for commit in repo.iter_commits(**log_kwargs):
            msg = commit.message.strip()
            match = TRUST_COMMIT_PATTERN.match(msg)
            if match:
                rec_subsystem = match.group(1)
                if subsystem and rec_subsystem != subsystem:
                    continue
                records.append({
                    "commit_hash": commit.hexsha[:12],
                    "subsystem": rec_subsystem,
                    "event_type": match.group(2),
                    "severity": float(match.group(3).lstrip("+")),
                    "details": match.group(4),
                    "timestamp": datetime.fromtimestamp(
                        commit.committed_date, tz=timezone.utc
                    ).isoformat(),
                    "is_bad": float(match.group(3).lstrip("+")) > 0,
                })

        return records

    def compute_trust_from_log(
        self,
        subsystem: str,
        repo: Any,
    ) -> float:
        """Recompute trust score from git history.

        Uses simplified INCREMENTS algorithm applied to git log events.
        Should match the in-memory engine's trust score for the same
        subsystem if all events were properly recorded.

        Args:
            subsystem: Subsystem to recompute trust for.
            repo: gitpython Repo object (or path string).

        Returns:
            Computed trust score (0.0 to 1.0).
        """
        if isinstance(repo, str):
            repo = Repo(repo)

        events = self.get_trust_history(subsystem, repo=repo)
        if not events:
            return 0.0

        # Reverse to process in chronological order (oldest first)
        events = list(reversed(events))

        # Simplified INCREMENTS recomputation
        trust = 0.0
        consecutive_clean = 0
        alpha_gain = self._params["alpha_gain"]
        alpha_loss = self._params["alpha_loss"]
        alpha_decay = self._params["alpha_decay"]
        t_floor = self._params["t_floor"]

        for event in events:
            severity = event["severity"]
            if severity > 0:
                # Bad event — penalty branch
                delta = -alpha_loss * trust * severity
                trust = max(trust + delta, t_floor)
                consecutive_clean = 0
            else:
                # Good event — gain branch
                avg_quality = 0.7  # default quality for good events
                delta = alpha_gain * (1.0 - trust) * avg_quality
                trust = min(trust + delta, 1.0)
                consecutive_clean += 1

        return round(trust, 6)

    def get_trust_snapshot(self) -> dict[str, float]:
        """Get current trust scores for all subsystems."""
        return self._engine.get_all_scores()

    def _commit_trust_event(
        self,
        repo: Any,
        subsystem: str,
        event_type: str,
        severity: float,
        details: str,
    ) -> str:
        """Write trust event to git as a commit."""
        trust_dir = os.path.join(repo.working_dir, TRUST_DIR)
        os.makedirs(trust_dir, exist_ok=True)

        # Write event JSON
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%z")
        event_data = {
            "vessel_id": self.vessel_id,
            "subsystem": subsystem,
            "event_type": event_type,
            "severity": severity,
            "details": details,
            "timestamp": timestamp,
            "is_bad": severity > 0,
        }
        filepath = os.path.join(trust_dir, f"{timestamp}_{subsystem}_{event_type}.json")
        with open(filepath, "w") as f:
            json.dump(event_data, f, indent=2)

        # Stage and commit
        repo.index.add([filepath])
        sign = "+" if severity >= 0 else ""
        commit_msg = f"TRUST: {subsystem} {event_type} {sign}{severity} | {details}"
        commit = repo.index.commit(commit_msg)
        return commit.hexsha


# ── Standalone trust engine (when trust module not available) ──────

class _StandaloneTrustEngine:
    """Minimal standalone trust engine for environments without
    the full NEXUS trust module."""

    def __init__(self, params: dict[str, float]) -> None:
        self._params = params
        self._scores: dict[str, float] = {}

    def register_subsystem(self, name: str) -> None:
        self._scores.setdefault(name, 0.0)

    def register_all_subsystems(self) -> None:
        for name in ["steering", "engine", "navigation", "payload", "communication"]:
            self.register_subsystem(name)

    def record_event(self, subsystem: str, event_type: str, severity: float) -> None:
        """Record a trust event. If severity > 0, it's a bad event."""
        trust = self._scores.get(subsystem, 0.0)
        if severity > 0:
            delta = -self._params["alpha_loss"] * trust * severity
            trust = max(trust + delta, self._params["t_floor"])
        else:
            delta = self._params["alpha_gain"] * (1.0 - trust) * 0.7
            trust = min(trust + delta, 1.0)
        self._scores[subsystem] = trust

    def get_trust_score(self, subsystem: str) -> float:
        return self._scores.get(subsystem, 0.0)

    def get_all_scores(self) -> dict[str, float]:
        return dict(self._scores)
