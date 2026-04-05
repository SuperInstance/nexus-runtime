"""
NEXUS Fleet State Types
Common type definitions for all fleet sync solutions.

Designed for Jetson (Python, aarch64) with zero heavy dependencies.
Only uses Python stdlib.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import hashlib
import json


@dataclass
class TrustScore:
    """Represents a trust score for a vessel (float, 0.0 to 1.0)."""
    vessel_id: str
    score: float = 0.5

    def clamp(self) -> "TrustScore":
        self.score = max(0.0, min(1.0, self.score))
        return self


@dataclass
class TaskItem:
    """A task in the fleet task queue."""
    task_id: str
    description: str
    priority: int = 5  # 1=highest, 10=lowest
    assigned_to: str = ""
    status: str = "pending"  # pending, in_progress, completed

    def key(self) -> str:
        return self.task_id


@dataclass
class SkillVersion:
    """A skill with semantic versioning."""
    skill_name: str
    major: int = 0
    minor: int = 0
    patch: int = 0

    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.major, self.minor, self.patch)

    def as_string(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __gt__(self, other: "SkillVersion") -> bool:
        return self.as_tuple() > other.as_tuple()

    def __lt__(self, other: "SkillVersion") -> bool:
        return self.as_tuple() < other.as_tuple()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SkillVersion):
            return False
        return self.as_tuple() == other.as_tuple()

    def __hash__(self) -> int:
        return hash(self.as_tuple())


@dataclass
class FleetState:
    """Complete fleet state held by a vessel."""
    vessel_id: str
    trust_scores: Dict[str, float] = field(default_factory=dict)
    task_queue: List[TaskItem] = field(default_factory=list)
    vessel_statuses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    skill_versions: Dict[str, SkillVersion] = field(default_factory=dict)

    def state_hash(self) -> str:
        """Generate a hash of the current state for comparison."""
        state_dict = {
            "trust_scores": {k: round(v, 6) for k, v in sorted(self.trust_scores.items())},
            "task_queue": sorted([
                {"id": t.task_id, "desc": t.description, "priority": t.priority,
                 "assigned_to": t.assigned_to, "status": t.status}
                for t in self.task_queue
            ], key=lambda x: (x["priority"], x["id"])),
            "vessel_statuses": {k: v for k, v in sorted(self.vessel_statuses.items())},
            "skill_versions": {k: v.as_string() for k, v in sorted(self.skill_versions.items())},
        }
        raw = json.dumps(state_dict, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def is_equivalent(self, other: "FleetState") -> bool:
        """Check if two fleet states are logically equivalent."""
        # Compare trust scores
        if set(self.trust_scores.keys()) != set(other.trust_scores.keys()):
            return False
        for k in self.trust_scores:
            if abs(self.trust_scores[k] - other.trust_scores[k]) > 0.0001:
                return False

        # Compare task queues (by task_id set and content)
        self_tasks = {t.task_id: t for t in self.task_queue}
        other_tasks = {t.task_id: t for t in other.task_queue}
        if set(self_tasks.keys()) != set(other_tasks.keys()):
            return False
        for tid in self_tasks:
            st, ot = self_tasks[tid], other_tasks[tid]
            if (st.description != ot.description or st.priority != ot.priority
                    or st.assigned_to != ot.assigned_to or st.status != ot.status):
                return False

        # Compare vessel statuses
        if self.vessel_statuses != other.vessel_statuses:
            return False

        # Compare skill versions
        if set(self.skill_versions.keys()) != set(other.skill_versions.keys()):
            return False
        for k in self.skill_versions:
            if self.skill_versions[k] != other.skill_versions[k]:
                return False

        return True


@dataclass
class TimestampedValue:
    """A value with a timestamp for LWW resolution."""
    value: Any
    timestamp: float
    vessel_id: str

    def is_newer_than(self, other: "TimestampedValue") -> bool:
        if self.timestamp != other.timestamp:
            return self.timestamp > other.timestamp
        return self.vessel_id > other.vessel_id


class VectorClock:
    """Simple vector clock for causal ordering."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.clock: Dict[str, int] = {}

    def increment(self):
        self.clock[self.node_id] = self.clock.get(self.node_id, 0) + 1

    def merge(self, other: "VectorClock"):
        all_keys = set(self.clock.keys()) | set(other.clock.keys())
        for key in all_keys:
            self.clock[key] = max(self.clock.get(key, 0), other.clock.get(key, 0))

    def happens_before(self, other: "VectorClock") -> Optional[bool]:
        self_keys = set(self.clock.keys())
        other_keys = set(other.clock.keys())
        all_keys = self_keys | other_keys

        at_least_one_less = False
        at_least_one_greater = False

        for key in all_keys:
            sv = self.clock.get(key, 0)
            ov = other.clock.get(key, 0)
            if sv < ov:
                at_least_one_less = True
            elif sv > ov:
                at_least_one_greater = True

        if at_least_one_less and not at_least_one_greater:
            return True
        elif at_least_one_greater and not at_least_one_less:
            return False
        else:
            return None

    def copy(self) -> "VectorClock":
        vc = VectorClock(self.node_id)
        vc.clock = dict(self.clock)
        return vc

    def as_dict(self) -> Dict[str, int]:
        return dict(self.clock)

    def from_dict(self, data: Dict[str, int]):
        self.clock = dict(data)
        return self


@dataclass
class SyncMetrics:
    """Metrics for evaluating a sync solution."""
    convergence_correct: bool = False
    data_loss_count: int = 0
    conflict_count: int = 0
    conflict_quality_score: float = 0.0
    memory_bytes: int = 0
    lines_of_code: int = 0
    edge_cases: int = 0
    sync_rounds_to_converge: int = 0
    total_operations_processed: int = 0
