"""
NEXUS Conflict Resolution Policies

When CRDTs can't auto-resolve, these policies determine the winner:
- Last-Writer-Wins (LWW) with vector clocks
- Trust-weighted resolution (higher trust vessel's state wins)
- Domain-specific rules (safety alerts always win over routine state)
- Configurable policy per state type
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .types import FleetState, VectorClock


class ResolutionPolicy(Enum):
    """Available conflict resolution strategies."""
    LWW = "last_writer_wins"
    TRUST_WEIGHTED = "trust_weighted"
    DOMAIN_PRIORITY = "domain_priority"
    HIGHEST_WINS = "highest_wins"
    LOWEST_WINS = "lowest_wins"
    UNION = "union"
    FIRST_WRITER_WINS = "first_writer_wins"
    CUSTOM = "custom"


class StateDomain(Enum):
    """Fleet state domains for policy assignment."""
    TRUST_SCORES = "trust_scores"
    VESSEL_POSITIONS = "vessel_positions"
    TASK_ASSIGNMENTS = "task_assignments"
    SAFETY_ALERTS = "safety_alerts"
    RESOURCE_LEVELS = "resource_levels"
    VESSEL_STATUSES = "vessel_statuses"
    SKILL_VERSIONS = "skill_versions"


# Default policy per domain
DEFAULT_DOMAIN_POLICIES: Dict[StateDomain, ResolutionPolicy] = {
    StateDomain.TRUST_SCORES: ResolutionPolicy.UNION,
    StateDomain.VESSEL_POSITIONS: ResolutionPolicy.LWW,
    StateDomain.TASK_ASSIGNMENTS: ResolutionPolicy.LWW,
    StateDomain.SAFETY_ALERTS: ResolutionPolicy.DOMAIN_PRIORITY,
    StateDomain.RESOURCE_LEVELS: ResolutionPolicy.LOWEST_WINS,
    StateDomain.VESSEL_STATUSES: ResolutionPolicy.LWW,
    StateDomain.SKILL_VERSIONS: ResolutionPolicy.HIGHEST_WINS,
}


@dataclass
class ConflictRecord:
    """Record of a resolved conflict."""
    key: str
    domain: StateDomain
    local_value: Any
    remote_value: Any
    resolved_value: Any
    winner_vessel: str
    loser_vessel: str
    resolution_policy: ResolutionPolicy
    timestamp: float = 0.0
    reason: str = ""


@dataclass
class LWWEntry:
    """A value with LWW metadata."""
    value: Any
    timestamp: float
    vessel_id: str
    vector_clock: Dict[str, int] = field(default_factory=dict)


class ConflictResolver:
    """
    Configurable conflict resolution engine for fleet state sync.

    Each state domain can have its own resolution policy. The resolver
    determines which value wins when two vessels have conflicting state.

    Usage:
        resolver = ConflictResolver()
        resolver.set_policy(StateDomain.SAFETY_ALERTS, ResolutionPolicy.DOMAIN_PRIORITY)
        resolver.set_trust_scores({"vessel-0": 0.8, "vessel-1": 0.6})

        entry = resolver.resolve(
            key="alert-123",
            domain=StateDomain.SAFETY_ALERTS,
            local=LWWEntry(...),
            remote=LWWEntry(...),
        )
    """

    def __init__(self):
        self._policies: Dict[StateDomain, ResolutionPolicy] = dict(DEFAULT_DOMAIN_POLICIES)
        self._trust_scores: Dict[str, float] = {}
        self._conflict_log: List[ConflictRecord] = []
        self._custom_resolvers: Dict[StateDomain, Callable] = {}
        self._domain_priority_map: Dict[str, int] = {
            "safety_alerts": 100,
            "trust_scores": 50,
            "resource_levels": 40,
            "vessel_positions": 30,
            "task_assignments": 20,
            "vessel_statuses": 10,
            "skill_versions": 5,
        }

    def set_policy(self, domain: StateDomain, policy: ResolutionPolicy):
        """Set the resolution policy for a state domain."""
        self._policies[domain] = policy

    def get_policy(self, domain: StateDomain) -> ResolutionPolicy:
        """Get the resolution policy for a state domain."""
        return self._policies.get(domain, ResolutionPolicy.LWW)

    def set_trust_scores(self, scores: Dict[str, float]):
        """Set vessel trust scores for trust-weighted resolution."""
        self._trust_scores = dict(scores)

    def update_trust_score(self, vessel_id: str, score: float):
        """Update a single vessel's trust score."""
        self._trust_scores[vessel_id] = max(0.0, min(1.0, score))

    def get_trust_score(self, vessel_id: str) -> float:
        """Get vessel trust score, default 0.5."""
        return self._trust_scores.get(vessel_id, 0.5)

    def set_custom_resolver(self, domain: StateDomain, resolver: Callable):
        """Set a custom resolution function for a domain."""
        self._custom_resolvers[domain] = resolver
        self._policies[domain] = ResolutionPolicy.CUSTOM

    def resolve(self, key: str, domain: StateDomain,
                local: LWWEntry, remote: LWWEntry) -> LWWEntry:
        """
        Resolve a conflict between local and remote state.
        Returns the winning entry.
        """
        policy = self._policies.get(domain, ResolutionPolicy.LWW)
        winner = None

        if policy == ResolutionPolicy.LWW:
            winner = self._resolve_lww(local, remote)
        elif policy == ResolutionPolicy.TRUST_WEIGHTED:
            winner = self._resolve_trust_weighted(local, remote)
        elif policy == ResolutionPolicy.DOMAIN_PRIORITY:
            winner = self._resolve_domain_priority(key, local, remote)
        elif policy == ResolutionPolicy.HIGHEST_WINS:
            winner = self._resolve_highest(local, remote)
        elif policy == ResolutionPolicy.LOWEST_WINS:
            winner = self._resolve_lowest(local, remote)
        elif policy == ResolutionPolicy.UNION:
            winner = self._resolve_union(local, remote)
        elif policy == ResolutionPolicy.FIRST_WRITER_WINS:
            winner = self._resolve_first_writer(local, remote)
        elif policy == ResolutionPolicy.CUSTOM:
            winner = self._resolve_custom(domain, local, remote)
        else:
            winner = self._resolve_lww(local, remote)

        # Log the conflict
        if winner.vessel_id != local.vessel_id and winner.vessel_id != remote.vessel_id:
            pass  # Shouldn't happen
        loser_id = remote.vessel_id if winner.vessel_id == local.vessel_id else local.vessel_id
        if local.vessel_id != remote.vessel_id and (
            winner.value != local.value or winner.value != remote.value
        ):
            self._conflict_log.append(ConflictRecord(
                key=key,
                domain=domain,
                local_value=local.value,
                remote_value=remote.value,
                resolved_value=winner.value,
                winner_vessel=winner.vessel_id,
                loser_vessel=loser_id,
                resolution_policy=policy,
                timestamp=time.time(),
            ))

        return winner

    def _resolve_lww(self, local: LWWEntry, remote: LWWEntry) -> LWWEntry:
        """Last-Writer-Wins: highest timestamp wins, vessel_id tiebreaker."""
        if remote.timestamp > local.timestamp:
            return remote
        elif remote.timestamp == local.timestamp:
            if remote.vessel_id > local.vessel_id:
                return remote
        return local

    def _resolve_trust_weighted(self, local: LWWEntry, remote: LWWEntry) -> LWWEntry:
        """Higher trust vessel's state wins. Ties broken by LWW."""
        local_trust = self._trust_scores.get(local.vessel_id, 0.5)
        remote_trust = self._trust_scores.get(remote.vessel_id, 0.5)

        if abs(remote_trust - local_trust) > 0.01:
            if remote_trust > local_trust:
                return remote
            return local

        # Trust scores are close — fall back to LWW
        return self._resolve_lww(local, remote)

    def _resolve_domain_priority(self, key: str, local: LWWEntry,
                                  remote: LWWEntry) -> LWWEntry:
        """Domain-specific priority rules."""
        # For safety alerts: unresolved emergencies always win
        if isinstance(local.value, dict) and isinstance(remote.value, dict):
            local_resolved = local.value.get("resolved", False)
            remote_resolved = remote.value.get("resolved", False)

            if local_resolved and not remote_resolved:
                return remote
            if remote_resolved and not local_resolved:
                return local

            # Higher severity wins
            severity_order = {"emergency": 100, "critical": 75, "warning": 50, "info": 25}
            local_sev = severity_order.get(local.value.get("severity", ""), 0)
            remote_sev = severity_order.get(remote.value.get("severity", ""), 0)
            if remote_sev > local_sev:
                return remote
            elif remote_sev < local_sev:
                return local

        # Fall back to LWW
        return self._resolve_lww(local, remote)

    def _resolve_highest(self, local: LWWEntry, remote: LWWEntry) -> LWWEntry:
        """Higher numeric value wins."""
        try:
            if float(remote.value) > float(local.value):
                return remote
            elif float(remote.value) == float(local.value):
                return self._resolve_lww(local, remote)
        except (ValueError, TypeError):
            pass
        return local

    def _resolve_lowest(self, local: LWWEntry, remote: LWWEntry) -> LWWEntry:
        """Lower numeric value wins (conservative estimate)."""
        try:
            if float(remote.value) < float(local.value):
                return remote
            elif float(remote.value) == float(local.value):
                return self._resolve_lww(local, remote)
        except (ValueError, TypeError):
            pass
        return local

    def _resolve_union(self, local: LWWEntry, remote: LWWEntry) -> LWWEntry:
        """Merge both values (union). Returns newer entry with merged value."""
        # For trust scores: additive merge
        if isinstance(local.value, (int, float)) and isinstance(remote.value, (int, float)):
            merged = local.value + remote.value
            winner = local if local.timestamp >= remote.timestamp else remote
            return LWWEntry(
                value=merged,
                timestamp=max(local.timestamp, remote.timestamp),
                vessel_id=winner.vessel_id,
                vector_clock=winner.vector_clock,
            )

        # For dicts: merge keys
        if isinstance(local.value, dict) and isinstance(remote.value, dict):
            merged = {**local.value, **remote.value}
            winner = local if local.timestamp >= remote.timestamp else remote
            return LWWEntry(
                value=merged,
                timestamp=max(local.timestamp, remote.timestamp),
                vessel_id=winner.vessel_id,
                vector_clock=winner.vector_clock,
            )

        # For lists: concat and deduplicate
        if isinstance(local.value, list) and isinstance(remote.value, list):
            seen = set()
            merged = []
            for item in local.value + remote.value:
                key = str(item) if not isinstance(item, dict) else str(sorted(item.items()))
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
            winner = local if local.timestamp >= remote.timestamp else remote
            return LWWEntry(
                value=merged,
                timestamp=max(local.timestamp, remote.timestamp),
                vessel_id=winner.vessel_id,
                vector_clock=winner.vector_clock,
            )

        return self._resolve_lww(local, remote)

    def _resolve_first_writer(self, local: LWWEntry, remote: LWWEntry) -> LWWEntry:
        """First writer wins — earliest timestamp."""
        if remote.timestamp < local.timestamp:
            return remote
        elif remote.timestamp == local.timestamp:
            if remote.vessel_id < local.vessel_id:
                return remote
        return local

    def _resolve_custom(self, domain: StateDomain, local: LWWEntry,
                        remote: LWWEntry) -> LWWEntry:
        """Use custom resolver function."""
        resolver = self._custom_resolvers.get(domain)
        if resolver:
            return resolver(local, remote)
        return self._resolve_lww(local, remote)

    def get_conflict_log(self) -> List[ConflictRecord]:
        """Return all conflict records."""
        return list(self._conflict_log)

    def get_conflict_count(self) -> int:
        return len(self._conflict_log)

    def clear_conflict_log(self):
        self._conflict_log.clear()

    def get_domain_conflict_counts(self) -> Dict[str, int]:
        """Count conflicts per domain."""
        counts: Dict[str, int] = {}
        for record in self._conflict_log:
            domain = record.domain.value
            counts[domain] = counts.get(domain, 0) + 1
        return counts
