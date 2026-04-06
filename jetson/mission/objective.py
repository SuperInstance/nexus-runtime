"""Mission objective management for NEXUS marine robotics platform.

Manages objective lifecycle, priority resolution, conflict detection,
and value scoring for mission objectives.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional


class ObjectiveStatus(Enum):
    """Status of a mission objective."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class ObjectivePriority(Enum):
    """Priority levels for objectives."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class MissionObjective:
    """Represents a single mission objective."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    objective_type: str = "general"
    target: Any = None
    priority: ObjectivePriority = ObjectivePriority.MEDIUM
    constraints: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[float] = None
    weight: float = 1.0
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class ObjectiveResult:
    """Result of an objective attempt."""
    objective: MissionObjective = field(default_factory=MissionObjective)
    status: ObjectiveStatus = ObjectiveStatus.PENDING
    actual_value: Any = None
    target_value: Any = None
    achieved_at: Optional[float] = None
    deviation: float = 0.0
    notes: str = ""


@dataclass
class PriorityConflict:
    """Represents a conflict between objectives."""
    objective_a: str
    objective_b: str
    conflict_type: str
    severity: float
    description: str


@dataclass
class ObjectiveReport:
    """Comprehensive objective status report."""
    total_objectives: int = 0
    completed: int = 0
    in_progress: int = 0
    pending: int = 0
    failed: int = 0
    blocked: int = 0
    overall_score: float = 0.0
    details: List[Dict[str, Any]] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)


class ObjectiveManager:
    """Manages mission objectives lifecycle, priorities, and status tracking."""

    def __init__(self):
        self._objectives: Dict[str, MissionObjective] = {}
        self._results: Dict[str, ObjectiveResult] = {}
        self._status_history: Dict[str, List[ObjectiveStatus]] = {}

    def add_objective(self, objective: MissionObjective) -> str:
        """Add a new objective to the manager. Returns objective id."""
        if objective.id in self._objectives:
            raise ValueError(f"Objective {objective.id} already exists")
        self._objectives[objective.id] = objective
        self._status_history[objective.id] = [ObjectiveStatus.PENDING]
        result = ObjectiveResult(objective=objective, target_value=objective.target)
        self._results[objective.id] = result
        return objective.id

    def remove_objective(self, objective_id: str) -> bool:
        """Remove an objective. Returns True if removed."""
        if objective_id in self._objectives:
            del self._objectives[objective_id]
            del self._results[objective_id]
            del self._status_history[objective_id]
            return True
        return False

    def get_objective(self, objective_id: str) -> Optional[MissionObjective]:
        """Retrieve an objective by id."""
        return self._objectives.get(objective_id)

    def update_status(self, objective_id: str, status: ObjectiveStatus) -> bool:
        """Update the status of an objective. Returns True if updated."""
        if objective_id not in self._objectives:
            return False
        old_status = self._results[objective_id].status
        self._results[objective_id].status = status
        self._status_history[objective_id].append(status)
        if status == ObjectiveStatus.COMPLETED:
            self._results[objective_id].achieved_at = time.time()
        return True

    def get_status(self, objective_id: str) -> Optional[ObjectiveStatus]:
        """Get the current status of an objective."""
        if objective_id in self._results:
            return self._results[objective_id].status
        return None

    def get_result(self, objective_id: str) -> Optional[ObjectiveResult]:
        """Get the result for an objective."""
        return self._results.get(objective_id)

    def set_actual_value(self, objective_id: str, value: Any) -> bool:
        """Set the actual achieved value for an objective."""
        if objective_id not in self._results:
            return False
        result = self._results[objective_id]
        result.actual_value = value
        # Compute deviation if target is numeric
        obj = self._objectives[objective_id]
        if (isinstance(obj.target, (int, float)) and
                isinstance(value, (int, float)) and obj.target != 0):
            result.deviation = abs(value - obj.target) / abs(obj.target)
        return True

    def check_completion(self, objectives: Optional[List[str]] = None) -> List[str]:
        """Return list of completed objective ids. If objectives is None, checks all."""
        if objectives is None:
            objectives = list(self._objectives.keys())
        return [
            oid for oid in objectives
            if oid in self._results and self._results[oid].status == ObjectiveStatus.COMPLETED
        ]

    def check_pending(self) -> List[str]:
        """Return list of pending objective ids."""
        return [
            oid for oid, r in self._results.items()
            if r.status == ObjectiveStatus.PENDING
        ]

    def check_blocked(self) -> List[str]:
        """Return list of blocked objective ids."""
        return [
            oid for oid, r in self._results.items()
            if r.status == ObjectiveStatus.BLOCKED
        ]

    def compute_priority_conflicts(self, objectives: Optional[List[str]] = None) -> List[PriorityConflict]:
        """Detect priority conflicts between objectives based on constraints."""
        if objectives is None:
            objectives = list(self._objectives.keys())
        conflicts = []
        objs = [(oid, self._objectives[oid]) for oid in objectives if oid in self._objectives]
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                oid_a, obj_a = objs[i]
                oid_b, obj_b = objs[j]
                # Check resource constraint overlap
                res_a = set(obj_a.constraints.get("required_resources", []))
                res_b = set(obj_b.constraints.get("required_resources", []))
                if res_a and res_b and res_a & res_b:
                    severity = len(res_a & res_b) / max(len(res_a | res_b), 1)
                    conflicts.append(PriorityConflict(
                        objective_a=oid_a,
                        objective_b=oid_b,
                        conflict_type="resource_overlap",
                        severity=severity,
                        description=f"Resource overlap: {res_a & res_b}"
                    ))
                # Check time constraint conflict
                if (obj_a.deadline and obj_b.deadline and
                        obj_a.priority != obj_b.priority):
                    if (obj_a.priority.value < obj_b.priority.value and
                            obj_a.deadline > obj_b.deadline):
                        conflicts.append(PriorityConflict(
                            objective_a=oid_a,
                            objective_b=oid_b,
                            conflict_type="priority_deadline_mismatch",
                            severity=0.7,
                            description="Higher priority objective has earlier deadline"
                        ))
        return conflicts

    def reallocate_priorities(self, objectives: List[str],
                              new_constraint: Dict[str, Any]) -> List[str]:
        """Reallocate priorities given a new constraint. Returns affected ids."""
        affected = []
        for oid in objectives:
            if oid not in self._objectives:
                continue
            obj = self._objectives[oid]
            # Check if new constraint affects this objective
            constraint_type = new_constraint.get("type", "")
            if constraint_type == "deadline":
                new_deadline = new_constraint.get("deadline", 0)
                if obj.deadline and obj.deadline > new_deadline:
                    # Boost priority
                    current = obj.priority.value
                    new_pri = ObjectivePriority(max(1, current - 1))
                    obj.priority = new_pri
                    affected.append(oid)
            elif constraint_type == "resource_limit":
                needed = set(obj.constraints.get("required_resources", []))
                limited = set(new_constraint.get("resources", []))
                if needed & limited:
                    current = obj.priority.value
                    new_pri = ObjectivePriority(max(1, current - 1))
                    obj.priority = new_pri
                    affected.append(oid)
        return affected

    def compute_objective_value(self, result: ObjectiveResult) -> float:
        """Compute a value score for an objective result (0.0 to 1.0+)."""
        obj = result.objective
        if result.status == ObjectiveStatus.FAILED:
            return 0.0
        if result.status == ObjectiveStatus.BLOCKED:
            return 0.0
        if result.status == ObjectiveStatus.PENDING:
            return 0.0
        if result.status == ObjectiveStatus.IN_PROGRESS:
            # Partial value based on deviation
            base = 0.3 * obj.weight * (4 - obj.priority.value + 1) / 4
            if result.deviation > 0:
                base *= max(0, 1.0 - result.deviation)
            return base
        # COMPLETED
        base = obj.weight * (4 - obj.priority.value + 1) / 4
        if result.deviation > 0:
            base *= max(0, 1.0 - result.deviation)
        return base

    def generate_objective_report(self, objective_ids: Optional[List[str]] = None) -> ObjectiveReport:
        """Generate a comprehensive status report for objectives."""
        if objective_ids is None:
            objective_ids = list(self._objectives.keys())

        report = ObjectiveReport(total_objectives=len(objective_ids))
        details = []
        total_score = 0.0

        for oid in objective_ids:
            result = self._results.get(oid)
            if result is None:
                report.pending += 1
                continue
            status = result.status
            if status == ObjectiveStatus.COMPLETED:
                report.completed += 1
            elif status == ObjectiveStatus.IN_PROGRESS:
                report.in_progress += 1
            elif status == ObjectiveStatus.PENDING:
                report.pending += 1
            elif status == ObjectiveStatus.FAILED:
                report.failed += 1
            elif status == ObjectiveStatus.BLOCKED:
                report.blocked += 1

            score = self.compute_objective_value(result)
            total_score += score
            details.append({
                "id": oid,
                "name": self._objectives[oid].name if oid in self._objectives else "",
                "status": status.value,
                "score": round(score, 3),
                "deviation": round(result.deviation, 3),
            })

        report.details = details
        report.overall_score = round(total_score, 3) if objective_ids else 0.0
        return report

    def get_all_objectives(self) -> List[MissionObjective]:
        """Return all managed objectives."""
        return list(self._objectives.values())

    def get_status_history(self, objective_id: str) -> List[ObjectiveStatus]:
        """Return status history for an objective."""
        return self._status_history.get(objective_id, [])

    def clear(self):
        """Clear all objectives and results."""
        self._objectives.clear()
        self._results.clear()
        self._status_history.clear()
