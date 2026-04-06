"""
Maintenance scheduling optimisation.

Priority ranking, cost-vs-risk trade-offs, downtime computation,
and failure-triggered rescheduling. Pure Python – no external deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class MaintenanceTask:
    """A single maintenance work item."""
    id: str
    equipment_id: str
    priority: int = 0  # higher = more urgent
    duration: float = 1.0  # hours
    deadline: Optional[float] = None  # timestamp
    cost: float = 0.0
    scheduled_start: Optional[float] = None


@dataclass
class ScheduleResult:
    """Output of a scheduling pass."""
    tasks: List[MaintenanceTask] = field(default_factory=list)
    total_cost: float = 0.0
    total_downtime: float = 0.0  # hours
    risk_score: float = 0.0


class MaintenanceScheduler:
    """Build and optimise maintenance schedules."""

    # ── public API ──────────────────────────────────────────────

    def schedule(
        self,
        tasks: List[MaintenanceTask],
        available_windows: List[Tuple[float, float]],
        resources: int = 1,
    ) -> ScheduleResult:
        """Assign tasks to the earliest feasible maintenance window.

        Parameters
        ----------
        tasks : list[MaintenanceTask]
        available_windows : list[(start, end)]
        resources : int
            Number of parallel maintenance bays.
        """
        # Sort by priority descending, then deadline ascending
        sorted_tasks = sorted(
            tasks,
            key=lambda t: (-t.priority, t.deadline or float("inf")),
        )

        assigned: List[MaintenanceTask] = []
        used_ends = [w[0] for w in available_windows]  # track per-resource

        for task in sorted_tasks:
            placed = False
            for idx, (win_start, win_end) in enumerate(available_windows):
                earliest = max(win_start, used_ends[idx]) if idx < len(used_ends) else win_start
                if earliest + task.duration <= win_end:
                    task.scheduled_start = earliest
                    assigned.append(task)
                    if idx < len(used_ends):
                        used_ends[idx] = earliest + task.duration
                    else:
                        used_ends.append(earliest + task.duration)
                    placed = True
                    break
            if not placed:
                # Could not schedule – still include for risk accounting
                task.scheduled_start = None
                assigned.append(task)

        total_cost = sum(t.cost for t in assigned)
        downtime = sum(t.duration for t in assigned if t.scheduled_start is not None)
        risk = self._compute_risk(assigned, tasks)

        return ScheduleResult(
            tasks=assigned,
            total_cost=total_cost,
            total_downtime=downtime,
            risk_score=risk,
        )

    def prioritize_by_risk(
        self,
        tasks: List[MaintenanceTask],
        risk_scores: Dict[str, float],
    ) -> List[MaintenanceTask]:
        """Return tasks sorted by risk score (descending)."""
        return sorted(
            tasks,
            key=lambda t: risk_scores.get(t.equipment_id, 0.0),
            reverse=True,
        )

    def optimize_cost_vs_risk(
        self,
        tasks: List[MaintenanceTask],
        budget: float,
    ) -> ScheduleResult:
        """Greedy knapsack: maximise risk-reduction within budget."""
        # Each task's "value" is priority as a proxy for risk reduction
        scored = sorted(tasks, key=lambda t: t.priority / max(t.cost, 1e-9), reverse=True)

        selected: List[MaintenanceTask] = []
        spent = 0.0
        for t in scored:
            if spent + t.cost <= budget:
                t.scheduled_start = spent  # continuous scheduling
                selected.append(t)
                spent += t.cost

        downtime = sum(t.duration for t in selected)
        risk = self._compute_risk(selected, tasks)

        return ScheduleResult(
            tasks=selected,
            total_cost=spent,
            total_downtime=downtime,
            risk_score=risk,
        )

    def compute_downtime(self, schedule: ScheduleResult) -> float:
        """Compute total downtime including task overlaps."""
        starts_ends = [
            (t.scheduled_start, t.scheduled_start + t.duration)
            for t in schedule.tasks
            if t.scheduled_start is not None
        ]
        if not starts_ends:
            return 0.0

        # Merge overlapping intervals
        starts_ends.sort()
        merged: List[Tuple[float, float]] = [starts_ends[0]]
        for start, end in starts_ends[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

        return sum(end - s for s, end in merged)

    def reschedule_after_failure(
        self,
        current_schedule: ScheduleResult,
        failed_task: MaintenanceTask,
    ) -> ScheduleResult:
        """Re-plan after an unexpected equipment failure.

        The failed task is promoted to highest priority and inserted
        at the earliest position.
        """
        failed_task.priority = max(t.priority for t in current_schedule.tasks) + 1 if current_schedule.tasks else 99
        failed_task.scheduled_start = None

        remaining = [
            t for t in current_schedule.tasks if t.id != failed_task.id
        ]
        remaining.append(failed_task)

        # Simple reschedule: just re-sort by priority
        remaining.sort(key=lambda t: -t.priority)
        cursor = 0.0
        for t in remaining:
            if t.scheduled_start is None or t.id == failed_task.id:
                t.scheduled_start = cursor
            cursor = t.scheduled_start + t.duration

        total_cost = sum(t.cost for t in remaining)
        downtime = sum(t.duration for t in remaining)
        risk = self._compute_risk(remaining, remaining)

        return ScheduleResult(
            tasks=remaining,
            total_cost=total_cost,
            total_downtime=downtime,
            risk_score=risk,
        )

    # ── private helpers ─────────────────────────────────────────

    @staticmethod
    def _compute_risk(scheduled: List[MaintenanceTask], all_tasks: List[MaintenanceTask]) -> float:
        """Risk = fraction of high-priority tasks left unscheduled."""
        if not all_tasks:
            return 0.0
        scheduled_ids = {t.id for t in scheduled if t.scheduled_start is not None}
        high_priority = [t for t in all_tasks if t.priority >= 5]
        if not high_priority:
            return 0.0
        unscheduled_high = sum(1 for t in high_priority if t.id not in scheduled_ids)
        return unscheduled_high / len(high_priority)
