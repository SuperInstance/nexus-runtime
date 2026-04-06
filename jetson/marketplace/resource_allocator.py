"""Resource allocation and vessel capability matching."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class VesselCapability:
    """Describes a vessel's capabilities."""
    vessel_id: str = ""
    max_speed: float = 0.0  # knots
    endurance_hours: float = 0.0
    sensor_types: List[str] = field(default_factory=list)
    actuator_types: List[str] = field(default_factory=list)
    trust_score: float = 1.0
    hourly_cost: float = 0.0
    location: Dict[str, float] = field(default_factory=lambda: {"lat": 0.0, "lon": 0.0})


@dataclass
class ResourceRequest:
    """A resource request for a task."""
    task_id: str = ""
    requirements: Dict[str, Any] = field(default_factory=dict)
    location: Dict[str, float] = field(default_factory=lambda: {"lat": 0.0, "lon": 0.0})
    time_window: Tuple[datetime, datetime] = field(
        default_factory=lambda: (datetime.utcnow(), datetime.utcnow() + timedelta(hours=24))
    )


@dataclass
class AllocationResult:
    """Result of matching a vessel to a task."""
    vessel_id: str = ""
    task_id: str = ""
    match_score: float = 0.0
    estimated_cost: float = 0.0
    estimated_duration: float = 0.0  # hours


class ResourceAllocator:
    """Allocates vessels to tasks based on capability matching."""

    def match_vessels_to_task(
        self, request: ResourceRequest, available_vessels: List[VesselCapability]
    ) -> List[AllocationResult]:
        """Match available vessels to a task request, sorted by match score."""
        results: List[AllocationResult] = []
        for vessel in available_vessels:
            score = self.compute_match_score(request.requirements, vessel)
            if score > 0:
                est_duration = self._estimate_duration(request, vessel)
                est_cost = est_duration * vessel.hourly_cost
                results.append(AllocationResult(
                    vessel_id=vessel.vessel_id,
                    task_id=request.task_id,
                    match_score=score,
                    estimated_cost=est_cost,
                    estimated_duration=est_duration,
                ))
        results.sort(key=lambda r: r.match_score, reverse=True)
        return results

    def compute_match_score(self, requirements: Dict[str, Any], capabilities: VesselCapability) -> float:
        """Compute a match score between 0 and 1."""
        if not requirements:
            return 0.0

        scores: List[float] = []
        total_weight = 0.0

        # Speed requirement
        if "min_speed" in requirements:
            required = requirements["min_speed"]
            if capabilities.max_speed >= required:
                scores.append(1.0)
            else:
                scores.append(capabilities.max_speed / required if required > 0 else 0.0)
            total_weight += 1.0

        # Endurance requirement
        if "min_endurance" in requirements:
            required = requirements["min_endurance"]
            if capabilities.endurance_hours >= required:
                scores.append(1.0)
            else:
                scores.append(capabilities.endurance_hours / required if required > 0 else 0.0)
            total_weight += 1.0

        # Sensor requirement
        if "required_sensors" in requirements:
            required_sensors = set(requirements["required_sensors"])
            available = set(capabilities.sensor_types)
            overlap = required_sensors & available
            if required_sensors:
                scores.append(len(overlap) / len(required_sensors))
                total_weight += 1.0

        # Actuator requirement
        if "required_actuators" in requirements:
            required_actuators = set(requirements["required_actuators"])
            available = set(capabilities.actuator_types)
            overlap = required_actuators & available
            if required_actuators:
                scores.append(len(overlap) / len(required_actuators))
                total_weight += 1.0

        # Trust score
        if "min_trust" in requirements:
            required = requirements["min_trust"]
            if capabilities.trust_score >= required:
                scores.append(1.0)
            else:
                scores.append(capabilities.trust_score / required if required > 0 else 0.0)
            total_weight += 1.0

        # Max hourly cost
        if "max_hourly_cost" in requirements:
            max_cost = requirements["max_hourly_cost"]
            if capabilities.hourly_cost <= max_cost:
                scores.append(1.0)
            else:
                scores.append(max_cost / capabilities.hourly_cost if capabilities.hourly_cost > 0 else 0.0)
            total_weight += 1.0

        if not scores or total_weight == 0:
            return 0.0

        return sum(scores) / len(scores)

    def optimize_fleet_allocation(
        self,
        tasks: List[ResourceRequest],
        vessels: List[VesselCapability],
    ) -> List[AllocationResult]:
        """Find an optimal task-vessel mapping using greedy assignment."""
        # Build all possible task-vessel match scores
        all_matches: List[Tuple[float, int, int]] = []  # (score, task_idx, vessel_idx)
        for ti, task in enumerate(tasks):
            for vi, vessel in enumerate(vessels):
                score = self.compute_match_score(task.requirements, vessel)
                all_matches.append((score, ti, vi))

        # Sort by score descending
        all_matches.sort(key=lambda x: x[0], reverse=True)

        assigned_tasks: set = set()
        assigned_vessels: set = set()
        results: List[AllocationResult] = []

        for score, ti, vi in all_matches:
            if ti in assigned_tasks or vi in assigned_vessels:
                continue
            if score <= 0:
                continue
            task = tasks[ti]
            vessel = vessels[vi]
            est_duration = self._estimate_duration(task, vessel)
            est_cost = est_duration * vessel.hourly_cost
            results.append(AllocationResult(
                vessel_id=vessel.vessel_id,
                task_id=task.task_id,
                match_score=score,
                estimated_cost=est_cost,
                estimated_duration=est_duration,
            ))
            assigned_tasks.add(ti)
            assigned_vessels.add(vi)

        return results

    def check_availability(
        self,
        vessel_id: str,
        time_window: Tuple[datetime, datetime],
        existing_assignments: List[Dict[str, Any]],
    ) -> bool:
        """Check if a vessel is available during the given time window."""
        start, end = time_window
        for assignment in existing_assignments:
            if assignment.get("vessel_id") != vessel_id:
                continue
            a_start = assignment.get("start")
            a_end = assignment.get("end")
            if a_start is None or a_end is None:
                continue
            # Check overlap
            if start < a_end and end > a_start:
                return False
        return True

    def compute_fleet_utilization(
        self,
        assignments: List[Dict[str, Any]],
        vessels: List[VesselCapability],
    ) -> float:
        """Compute fleet utilization as a percentage (0-100)."""
        if not vessels:
            return 0.0
        assigned_vessel_ids = set()
        for a in assignments:
            vid = a.get("vessel_id")
            if vid:
                assigned_vessel_ids.add(vid)
        return (len(assigned_vessel_ids) / len(vessels)) * 100.0

    def _estimate_duration(self, request: ResourceRequest, vessel: VesselCapability) -> float:
        """Estimate task duration in hours."""
        req = request.requirements
        if "estimated_duration" in req:
            return float(req["estimated_duration"])
        # Fallback: assume 4 hours
        return 4.0
