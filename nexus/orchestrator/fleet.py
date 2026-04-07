"""
NEXUS Fleet Orchestrator — fleet-level task distribution for marine robotics.

Manages a fleet of autonomous vessels:
    - Task submission and prioritized scheduling
    - Resource-aware task assignment
    - Fleet state tracking
    - Workload balancing across vessels
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class TaskStatus(enum.Enum):
    """Lifecycle states for fleet tasks."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(enum.IntEnum):
    """Task priority levels."""

    LOW = 0
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class Task:
    """A task to be distributed across the fleet."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    required_capabilities: Dict[str, float] = field(default_factory=dict)
    assigned_vessel: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    estimated_duration: float = 0.0  # seconds
    max_retries: int = 3
    retry_count: int = 0

    def __repr__(self) -> str:
        return f"Task({self.task_id} '{self.name}', status={self.status.value}, vessel={self.assigned_vessel})"


# ---------------------------------------------------------------------------
# Vessel
# ---------------------------------------------------------------------------

@dataclass
class VesselInfo:
    """Information about a vessel in the fleet."""

    vessel_id: str
    name: str = ""
    capabilities: Dict[str, float] = field(default_factory=dict)
    available: bool = True
    current_load: float = 0.0  # 0.0 - 1.0
    max_load: float = 1.0
    position: Tuple[float, float] = (0.0, 0.0)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_seen: float = field(default_factory=time.time)

    @property
    def load_capacity(self) -> float:
        return max(0.0, self.max_load - self.current_load)

    @property
    def is_overloaded(self) -> bool:
        return self.current_load >= self.max_load


# ---------------------------------------------------------------------------
# Assignment result
# ---------------------------------------------------------------------------

@dataclass
class AssignmentResult:
    """Result of a task assignment attempt."""

    success: bool
    task_id: str = ""
    vessel_id: str = ""
    match_score: float = 0.0
    reason: str = ""


# ---------------------------------------------------------------------------
# Fleet Orchestrator
# ---------------------------------------------------------------------------

class FleetOrchestrator:
    """Fleet-level orchestrator for task distribution and resource allocation.

    Usage::

        fleet = FleetOrchestrator()
        fleet.register_vessel(VesselInfo(vessel_id="AUV-001", capabilities={"navigation": 0.9}))
        fleet.register_vessel(VesselInfo(vessel_id="AUV-002", capabilities={"sensing": 0.8}))
        task = fleet.submit_task("Survey Area A", required_capabilities={"navigation": 0.7})
        result = fleet.assign_task(task.task_id)
    """

    def __init__(self) -> None:
        self._vessels: Dict[str, VesselInfo] = {}
        self._tasks: Dict[str, Task] = {}
        self._vessel_tasks: Dict[str, List[str]] = {}  # vessel_id -> task_ids
        self._assignment_history: List[AssignmentResult] = []

    @property
    def vessel_count(self) -> int:
        return len(self._vessels)

    @property
    def task_count(self) -> int:
        return len(self._tasks)

    # ----- vessel management -----

    def register_vessel(self, vessel: VesselInfo) -> VesselInfo:
        """Register a vessel in the fleet."""
        self._vessels[vessel.vessel_id] = vessel
        if vessel.vessel_id not in self._vessel_tasks:
            self._vessel_tasks[vessel.vessel_id] = []
        return vessel

    def unregister_vessel(self, vessel_id: str) -> bool:
        """Remove a vessel from the fleet. Returns True if it existed."""
        if vessel_id in self._vessels:
            del self._vessels[vessel_id]
            self._vessel_tasks.pop(vessel_id, None)
            return True
        return False

    def get_vessel(self, vessel_id: str) -> Optional[VesselInfo]:
        return self._vessels.get(vessel_id)

    def list_vessels(self) -> List[VesselInfo]:
        return list(self._vessels.values())

    def update_vessel_load(self, vessel_id: str, load: float) -> None:
        """Update a vessel's current load (0.0 - 1.0)."""
        if vessel_id in self._vessels:
            self._vessels[vessel_id].current_load = max(0.0, min(1.0, load))

    # ----- task management -----

    def submit_task(
        self,
        name: str,
        description: str = "",
        priority: TaskPriority = TaskPriority.NORMAL,
        required_capabilities: Optional[Dict[str, float]] = None,
        payload: Optional[Dict[str, Any]] = None,
        estimated_duration: float = 0.0,
    ) -> Task:
        """Submit a new task to the fleet."""
        task = Task(
            name=name,
            description=description,
            priority=priority,
            required_capabilities=required_capabilities or {},
            payload=payload or {},
            estimated_duration=estimated_duration,
        )
        self._tasks[task.task_id] = task
        return task

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task. Returns True if it was cancellable."""
        task = self._tasks.get(task_id)
        if task is None:
            return False
        if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
            return False
        task.status = TaskStatus.CANCELLED
        return True

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Task]:
        """List tasks, optionally filtered by status."""
        tasks = list(self._tasks.values())
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        return tasks

    # ----- assignment -----

    def assign_task(self, task_id: str, vessel_id: Optional[str] = None) -> AssignmentResult:
        """Assign a task to a vessel.

        If *vessel_id* is None, automatically selects the best vessel.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return AssignmentResult(False, task_id, reason="Task not found")

        if task.status not in (TaskStatus.PENDING, TaskStatus.FAILED):
            return AssignmentResult(False, task_id, reason=f"Task in {task.status.value} state")

        # Select vessel
        if vessel_id is not None:
            vessel = self._vessels.get(vessel_id)
            if vessel is None:
                return AssignmentResult(False, task_id, reason="Vessel not found")
            if not vessel.available or vessel.is_overloaded:
                return AssignmentResult(False, task_id, reason="Vessel unavailable or overloaded")
            match_score = self._compute_match(vessel, task)
        else:
            result = self._find_best_vessel(task)
            if result is None:
                return AssignmentResult(False, task_id, reason="No suitable vessel available")
            vessel, match_score = result
            vessel_id = vessel.vessel_id

        # Assign
        task.status = TaskStatus.ASSIGNED
        task.assigned_vessel = vessel_id
        vessel.current_load += self._task_load_weight(task)
        self._vessel_tasks.setdefault(vessel_id, []).append(task_id)

        assignment = AssignmentResult(
            success=True,
            task_id=task_id,
            vessel_id=vessel_id,
            match_score=match_score,
        )
        self._assignment_history.append(assignment)
        return assignment

    def start_task(self, task_id: str) -> bool:
        """Mark a task as in-progress."""
        task = self._tasks.get(task_id)
        if task and task.status == TaskStatus.ASSIGNED:
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = time.time()
            return True
        return False

    def complete_task(self, task_id: str, success: bool = True) -> bool:
        """Mark a task as completed or failed."""
        task = self._tasks.get(task_id)
        if task is None:
            return False

        if success:
            task.status = TaskStatus.COMPLETED
        else:
            task.status = TaskStatus.FAILED
            task.retry_count += 1
            if task.retry_count < task.max_retries:
                task.status = TaskStatus.PENDING
                task.assigned_vessel = ""
                return True

        task.completed_at = time.time()

        # Reduce vessel load
        if task.assigned_vessel and task.assigned_vessel in self._vessels:
            vessel = self._vessels[task.assigned_vessel]
            vessel.current_load = max(0.0, vessel.current_load - self._task_load_weight(task))

        return True

    # ----- auto-assignment -----

    def auto_assign_pending(self) -> List[AssignmentResult]:
        """Assign all pending tasks to the best available vessels."""
        results: List[AssignmentResult] = []
        pending = self.list_tasks(TaskStatus.PENDING)
        pending.sort(key=lambda t: t.priority.value, reverse=True)

        for task in pending:
            result = self.assign_task(task.task_id)
            results.append(result)

        return results

    def get_fleet_status(self) -> Dict[str, Any]:
        """Get overall fleet status summary."""
        tasks_by_status = {}
        for status in TaskStatus:
            tasks_by_status[status.value] = len(self.list_tasks(status))

        return {
            "vessel_count": self.vessel_count,
            "task_count": self.task_count,
            "tasks_by_status": tasks_by_status,
            "vessels": {
                vid: {
                    "available": v.available,
                    "load": v.current_load,
                    "tasks": len(self._vessel_tasks.get(vid, [])),
                }
                for vid, v in self._vessels.items()
            },
        }

    # ----- internal -----

    def _compute_match(self, vessel: VesselInfo, task: Task) -> float:
        """Compute match score between vessel and task."""
        if not task.required_capabilities:
            return 1.0

        scores: List[float] = []
        for domain, required in task.required_capabilities.items():
            if required <= 0:
                scores.append(1.0)
                continue
            vessel_cap = vessel.capabilities.get(domain, 0.0)
            scores.append(min(vessel_cap / required, 1.0))

        base_score = sum(scores) / len(scores) if scores else 0.0

        # Penalize overloaded vessels
        load_penalty = 1.0 - (vessel.current_load / vessel.max_load) * 0.5

        return base_score * load_penalty

    def _find_best_vessel(self, task: Task) -> Optional[Tuple[VesselInfo, float]]:
        """Find the best vessel for a task."""
        best: Optional[Tuple[VesselInfo, float]] = None
        for vessel in self._vessels.values():
            if not vessel.available or vessel.is_overloaded:
                continue
            score = self._compute_match(vessel, task)
            if best is None or score > best[1]:
                best = (vessel, score)
        return best

    @staticmethod
    def _task_load_weight(task: Task) -> float:
        """Compute the load weight of a task."""
        priority_factor = task.priority.value / 10.0
        return 0.1 * priority_factor
