"""Task orchestration across the fleet — scheduling, assignment, deadlock detection."""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class TaskType(Enum):
    PATROL = "patrol"
    SURVEY = "survey"
    DELIVERY = "delivery"
    SEARCH = "search"
    RESCUE = "rescue"
    INSPECTION = "inspection"
    MAINTENANCE = "maintenance"


class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class TaskRequirement:
    """A single requirement for a task."""
    skill: str                    # e.g., "sonar", "manipulator"
    min_count: int = 1
    min_trust: float = 0.0
    min_health: float = 0.0


@dataclass
class FleetTask:
    """A task that may be assigned to one or more vessels."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    type: TaskType = TaskType.PATROL
    priority: float = 0.5         # 0.0 (low) to 1.0 (critical)
    requirements: List[TaskRequirement] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_vessels: List[str] = field(default_factory=list)
    progress: float = 0.0         # 0.0 to 1.0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkloadAssignment:
    """Mapping of task_id -> list of vessel_ids."""
    task_id: str
    vessel_ids: List[str]
    priority: float = 0.5


class TaskOrchestrator:
    """Orchestrates task submission, assignment, reassignment, and scheduling."""

    def __init__(self) -> None:
        self._tasks: Dict[str, FleetTask] = {}
        self._vessel_tasks: Dict[str, Set[str]] = {}  # vessel -> task ids
        self._history: List[Dict[str, Any]] = []

    # ----------------------------------------------------------- CRUD
    def submit_task(self, task: FleetTask) -> str:
        """Submit a task. Returns the task id."""
        if task.id in self._tasks:
            raise ValueError(f"Task {task.id} already exists")
        task.created_at = time.time()
        task.status = TaskStatus.PENDING
        self._tasks[task.id] = task
        return task.id

    def cancel_task(self, task_id: str) -> bool:
        if task_id not in self._tasks:
            return False
        task = self._tasks[task_id]
        task.status = TaskStatus.CANCELLED
        # Free vessels
        for vid in task.assigned_vessels:
            if vid in self._vessel_tasks:
                self._vessel_tasks[vid].discard(task_id)
        task.assigned_vessels.clear()
        return True

    def get_task_status(self, task_id: str) -> Optional[FleetTask]:
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> List[FleetTask]:
        return list(self._tasks.values())

    # ------------------------------------------------------- Assignment
    def assign_vessels(self, task_id: str, vessel_ids: List[str]) -> bool:
        if task_id not in self._tasks:
            return False
        task = self._tasks[task_id]
        if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED):
            return False
        # Remove old assignments
        for vid in task.assigned_vessels:
            if vid in self._vessel_tasks:
                self._vessel_tasks[vid].discard(task_id)
        task.assigned_vessels = list(vessel_ids)
        task.status = TaskStatus.ASSIGNED
        for vid in vessel_ids:
            if vid not in self._vessel_tasks:
                self._vessel_tasks[vid] = set()
            self._vessel_tasks[vid].add(task_id)
        return True

    def reassign_task(self, task_id: str, new_vessel_ids: List[str]) -> bool:
        return self.assign_vessels(task_id, new_vessel_ids)

    # ------------------------------------------------------- Priority
    def compute_task_priority(self, task: FleetTask, fleet_state: Any) -> float:
        """Compute dynamic priority based on task properties and fleet context."""
        score = task.priority

        # Age factor: older tasks get slightly higher priority
        age = time.time() - task.created_at
        score += min(0.2, age / 3600.0 * 0.05)

        # Fleet load factor: if fleet is busy, prioritize high-value tasks
        if hasattr(fleet_state, "vessels"):
            total = len(fleet_state.vessels)
            available = sum(1 for v in fleet_state.vessels if getattr(v, "available", True))
            if total > 0:
                load_ratio = 1.0 - (available / total)
                score += load_ratio * 0.1

        # Emergency types get a boost
        if task.type in (TaskType.RESCUE, TaskType.SEARCH):
            score += 0.3

        return min(1.0, max(0.0, score))

    # ------------------------------------------------- Workload balance
    def balance_workload(self, tasks: List[FleetTask],
                         vessels: List[Any]) -> List[WorkloadAssignment]:
        """Greedy workload balancer: assign available vessels to pending tasks."""
        assignments: List[WorkloadAssignment] = []

        # Compute current load per vessel
        load: Dict[str, int] = {v.vessel_id: len(self._vessel_tasks.get(v.vessel_id, set()))
                                for v in vessels}

        available = sorted(
            [v for v in vessels if v.available],
            key=lambda v: load.get(v.vessel_id, 0),
        )
        pending = sorted(tasks, key=lambda t: t.priority, reverse=True)

        for task in pending:
            if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
                continue
            needed = 1
            if task.requirements:
                needed = min(task.requirements[0].min_count, len(available))
            chosen = []
            for v in available:
                if v.vessel_id not in [a.vessel_id for a in assignments
                                        for tid in [a.task_id]
                                        if tid == task.id]:
                    if v.health >= 0.3 and v.fuel > 5.0:
                        chosen.append(v.vessel_id)
                        if len(chosen) >= needed:
                            break
            if chosen:
                assignments.append(WorkloadAssignment(
                    task_id=task.id,
                    vessel_ids=chosen,
                    priority=task.priority,
                ))
        return assignments

    # --------------------------------------------------- Deadlock detect
    def detect_deadlocks(self, tasks: List[FleetTask]) -> List[Dict[str, Any]]:
        """Detect circular task dependencies / vessel contention deadlocks."""
        deadlocks: List[Dict[str, Any]] = []

        # Build a resource contention graph: vessel -> tasks needing it
        vessel_demand: Dict[str, List[str]] = {}
        for t in tasks:
            if t.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
                for vid in t.assigned_vessels:
                    vessel_demand.setdefault(vid, []).append(t.id)

        # Check for vessels assigned to multiple tasks
        for vid, tids in vessel_demand.items():
            if len(tids) > 1:
                deadlocks.append({
                    "type": "vessel_contention",
                    "vessel_id": vid,
                    "competing_tasks": tids,
                    "description": f"Vessel {vid} assigned to {len(tids)} tasks",
                })

        # Check for tasks with no vessels assigned that are in_progress
        for t in tasks:
            if t.status == TaskStatus.IN_PROGRESS and not t.assigned_vessels:
                deadlocks.append({
                    "type": "orphaned_task",
                    "task_id": t.id,
                    "description": f"Task {t.id} in_progress but no vessels assigned",
                })

        # Check for zero-progress long-running tasks
        for t in tasks:
            if (t.status == TaskStatus.IN_PROGRESS
                    and t.progress < 0.01
                    and (time.time() - t.created_at) > 1800):
                deadlocks.append({
                    "type": "stalled_task",
                    "task_id": t.id,
                    "description": f"Task {t.id} stalled — no progress in 30 min",
                })

        return deadlocks

    # ------------------------------------------------- ETA estimation
    def estimate_completion(self, task: FleetTask,
                            assigned_vessels: List[Any]) -> Optional[float]:
        """Estimate time to completion in seconds. Returns None if incalculable."""
        if task.progress >= 1.0:
            return 0.0
        if not assigned_vessels:
            return None

        # Base rate per vessel (units/s)
        avg_speed = sum(getattr(v, "speed", 5.0) for v in assigned_vessels) / len(assigned_vessels)
        avg_health = sum(getattr(v, "health", 1.0) for v in assigned_vessels) / len(assigned_vessels)

        # Effective work rate
        work_rate = len(assigned_vessels) * avg_speed * avg_health * 0.01  # progress per second

        if work_rate <= 0:
            return None

        remaining = 1.0 - task.progress
        eta_seconds = remaining / work_rate
        return eta_seconds

    # ------------------------------------------------ Vessel task query
    def get_tasks_for_vessel(self, vessel_id: str) -> List[FleetTask]:
        tid_set = self._vessel_tasks.get(vessel_id, set())
        return [self._tasks[tid] for tid in tid_set if tid in self._tasks]

    def get_active_task_count(self, vessel_id: str) -> int:
        return len(self._vessel_tasks.get(vessel_id, set()))
