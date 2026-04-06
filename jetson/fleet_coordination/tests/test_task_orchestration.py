"""Tests for task_orchestration module — TaskOrchestrator, FleetTask."""

import time

import pytest

from jetson.fleet_coordination.fleet_manager import FleetState, VesselStatus
from jetson.fleet_coordination.task_orchestration import (
    FleetTask,
    TaskOrchestrator,
    TaskRequirement,
    TaskStatus,
    TaskType,
    WorkloadAssignment,
)


# ────────────────────────────────────────────────────── fixtures

@pytest.fixture
def orchestrator():
    return TaskOrchestrator()


@pytest.fixture
def sample_task():
    return FleetTask(type=TaskType.PATROL, priority=0.7)


@pytest.fixture
def sample_vessels():
    return [
        VesselStatus(vessel_id=f"V{i}", health=0.9, fuel=80.0, available=True, speed=5.0)
        for i in range(5)
    ]


def _mock_fleet_state(vessels):
    return FleetState(vessels=vessels)


# ────────────────────────────────────────────────────── FleetTask

class TestFleetTask:
    def test_default_values(self):
        t = FleetTask()
        assert t.status == TaskStatus.PENDING
        assert t.progress == 0.0
        assert t.assigned_vessels == []
        assert t.priority == 0.5

    def test_task_type_enum(self):
        assert TaskType.PATROL.value == "patrol"
        assert TaskType.RESCUE.value == "rescue"
        assert TaskType.SURVEY.value == "survey"

    def test_task_status_enum(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.COMPLETED.value == "completed"

    def test_custom_task(self):
        req = TaskRequirement(skill="sonar", min_count=2, min_trust=0.5)
        t = FleetTask(type=TaskType.SURVEY, priority=0.9, requirements=[req])
        assert t.type == TaskType.SURVEY
        assert t.priority == 0.9
        assert len(t.requirements) == 1


# ────────────────────────────────────────────────────── Submit

class TestSubmitTask:
    def test_submit_task_success(self, orchestrator, sample_task):
        tid = orchestrator.submit_task(sample_task)
        assert tid == sample_task.id
        assert orchestrator.get_task_status(tid).status == TaskStatus.PENDING

    def test_submit_duplicate_raises(self, orchestrator, sample_task):
        orchestrator.submit_task(sample_task)
        with pytest.raises(ValueError, match="already exists"):
            orchestrator.submit_task(sample_task)

    def test_submit_multiple_tasks(self, orchestrator):
        for i in range(10):
            t = FleetTask(type=TaskType.PATROL)
            orchestrator.submit_task(t)
        assert len(orchestrator.get_all_tasks()) == 10

    def test_submit_sets_timestamp(self, orchestrator):
        before = time.time()
        t = FleetTask(type=TaskType.DELIVERY)
        orchestrator.submit_task(t)
        after = time.time()
        assert before <= t.created_at <= after


# ────────────────────────────────────────────────────── Cancel

class TestCancelTask:
    def test_cancel_success(self, orchestrator, sample_task):
        orchestrator.submit_task(sample_task)
        assert orchestrator.cancel_task(sample_task.id) is True
        assert orchestrator.get_task_status(sample_task.id).status == TaskStatus.CANCELLED

    def test_cancel_nonexistent(self, orchestrator):
        assert orchestrator.cancel_task("NOPE") is False

    def test_cancel_frees_vessels(self, orchestrator):
        t = FleetTask(type=TaskType.PATROL)
        orchestrator.submit_task(t)
        orchestrator.assign_vessels(t.id, ["V1", "V2"])
        orchestrator.cancel_task(t.id)
        assert t.assigned_vessels == []


# ────────────────────────────────────────────────────── Get

class TestGetTask:
    def test_get_task_status_found(self, orchestrator, sample_task):
        orchestrator.submit_task(sample_task)
        result = orchestrator.get_task_status(sample_task.id)
        assert result is not None
        assert result.id == sample_task.id

    def test_get_task_status_not_found(self, orchestrator):
        assert orchestrator.get_task_status("NOPE") is None

    def test_get_all_tasks_empty(self, orchestrator):
        assert orchestrator.get_all_tasks() == []


# ────────────────────────────────────────────────────── Assign

class TestAssignVessels:
    def test_assign_success(self, orchestrator, sample_task):
        orchestrator.submit_task(sample_task)
        assert orchestrator.assign_vessels(sample_task.id, ["V1", "V2"]) is True
        t = orchestrator.get_task_status(sample_task.id)
        assert t.assigned_vessels == ["V1", "V2"]
        assert t.status == TaskStatus.ASSIGNED

    def test_assign_nonexistent_task(self, orchestrator):
        assert orchestrator.assign_vessels("NOPE", ["V1"]) is False

    def test_assign_completed_task_fails(self, orchestrator):
        t = FleetTask(type=TaskType.PATROL)
        orchestrator.submit_task(t)
        t.status = TaskStatus.COMPLETED
        assert orchestrator.assign_vessels(t.id, ["V1"]) is False

    def test_assign_cancelled_task_fails(self, orchestrator):
        t = FleetTask(type=TaskType.PATROL)
        orchestrator.submit_task(t)
        orchestrator.cancel_task(t.id)
        assert orchestrator.assign_vessels(t.id, ["V1"]) is False

    def test_reassign(self, orchestrator, sample_task):
        orchestrator.submit_task(sample_task)
        orchestrator.assign_vessels(sample_task.id, ["V1"])
        orchestrator.assign_vessels(sample_task.id, ["V2", "V3"])
        t = orchestrator.get_task_status(sample_task.id)
        assert t.assigned_vessels == ["V2", "V3"]

    def test_assign_vessel_task_tracking(self, orchestrator):
        t = FleetTask(type=TaskType.PATROL)
        orchestrator.submit_task(t)
        orchestrator.assign_vessels(t.id, ["V1"])
        tasks = orchestrator.get_tasks_for_vessel("V1")
        assert len(tasks) == 1

    def test_assign_active_task_count(self, orchestrator):
        t1 = FleetTask(type=TaskType.PATROL)
        t2 = FleetTask(type=TaskType.SURVEY)
        orchestrator.submit_task(t1)
        orchestrator.submit_task(t2)
        orchestrator.assign_vessels(t1.id, ["V1"])
        orchestrator.assign_vessels(t2.id, ["V1"])
        assert orchestrator.get_active_task_count("V1") == 2


# ────────────────────────────────────────────────────── Priority

class TestComputePriority:
    def test_base_priority(self, orchestrator, sample_vessels):
        t = FleetTask(type=TaskType.PATROL, priority=0.5)
        fleet = _mock_fleet_state(sample_vessels)
        score = orchestrator.compute_task_priority(t, fleet)
        assert 0.0 <= score <= 1.0

    def test_rescue_boost(self, orchestrator, sample_vessels):
        t_patrol = FleetTask(type=TaskType.PATROL, priority=0.5)
        t_rescue = FleetTask(type=TaskType.RESCUE, priority=0.5)
        fleet = _mock_fleet_state(sample_vessels)
        s1 = orchestrator.compute_task_priority(t_patrol, fleet)
        s2 = orchestrator.compute_task_priority(t_rescue, fleet)
        assert s2 > s1

    def test_search_boost(self, orchestrator, sample_vessels):
        t_patrol = FleetTask(type=TaskType.PATROL, priority=0.5)
        t_search = FleetTask(type=TaskType.SEARCH, priority=0.5)
        fleet = _mock_fleet_state(sample_vessels)
        s1 = orchestrator.compute_task_priority(t_patrol, fleet)
        s2 = orchestrator.compute_task_priority(t_search, fleet)
        assert s2 > s1

    def test_clamped_output(self, orchestrator):
        t = FleetTask(type=TaskType.RESCUE, priority=1.0)
        fleet = FleetState()
        score = orchestrator.compute_task_priority(t, fleet)
        assert 0.0 <= score <= 1.0


# ────────────────────────────────────────────────────── Workload

class TestBalanceWorkload:
    def test_basic_balance(self, orchestrator, sample_vessels):
        tasks = [
            FleetTask(type=TaskType.PATROL, priority=0.8),
            FleetTask(type=TaskType.SURVEY, priority=0.6),
        ]
        for t in tasks:
            orchestrator.submit_task(t)
        assignments = orchestrator.balance_workload(tasks, sample_vessels)
        assert len(assignments) >= 1

    def test_balance_uses_available_vessels(self, orchestrator):
        vessels = [
            VesselStatus(vessel_id="V1", available=True, health=0.9, fuel=80, speed=5),
            VesselStatus(vessel_id="V2", available=False, health=0.9, fuel=80, speed=5),
        ]
        t = FleetTask(type=TaskType.PATROL, priority=0.7)
        orchestrator.submit_task(t)
        assignments = orchestrator.balance_workload([t], vessels)
        for a in assignments:
            assert all(vid != "V2" for vid in a.vessel_ids)

    def test_balance_skips_completed(self, orchestrator, sample_vessels):
        t = FleetTask(type=TaskType.PATROL, status=TaskStatus.COMPLETED)
        assignments = orchestrator.balance_workload([t], sample_vessels)
        assert len(assignments) == 0

    def test_balance_empty(self, orchestrator):
        assignments = orchestrator.balance_workload([], [])
        assert assignments == []


# ────────────────────────────────────────────────────── Deadlock

class TestDetectDeadlocks:
    def test_vessel_contention(self, orchestrator):
        t1 = FleetTask(type=TaskType.PATROL, assigned_vessels=["V1"], status=TaskStatus.ASSIGNED)
        t2 = FleetTask(type=TaskType.SURVEY, assigned_vessels=["V1"], status=TaskStatus.ASSIGNED)
        deadlocks = orchestrator.detect_deadlocks([t1, t2])
        types = [d["type"] for d in deadlocks]
        assert "vessel_contention" in types

    def test_orphaned_task(self, orchestrator):
        t = FleetTask(type=TaskType.PATROL, status=TaskStatus.IN_PROGRESS, assigned_vessels=[])
        deadlocks = orchestrator.detect_deadlocks([t])
        types = [d["type"] for d in deadlocks]
        assert "orphaned_task" in types

    def test_stalled_task(self, orchestrator):
        t = FleetTask(
            type=TaskType.PATROL,
            status=TaskStatus.IN_PROGRESS,
            progress=0.0,
            created_at=time.time() - 2000,
        )
        deadlocks = orchestrator.detect_deadlocks([t])
        types = [d["type"] for d in deadlocks]
        assert "stalled_task" in types

    def test_no_deadlocks(self, orchestrator):
        t = FleetTask(type=TaskType.PATROL, status=TaskStatus.COMPLETED)
        deadlocks = orchestrator.detect_deadlocks([t])
        assert len(deadlocks) == 0

    def test_empty_tasks(self, orchestrator):
        assert orchestrator.detect_deadlocks([]) == []


# ────────────────────────────────────────────────────── ETA

class TestEstimateCompletion:
    def test_completed_task_eta_zero(self, orchestrator):
        t = FleetTask(progress=1.0)
        vessels = [VesselStatus(vessel_id="V1", speed=5.0, health=1.0)]
        assert orchestrator.estimate_completion(t, vessels) == 0.0

    def test_no_vessels_returns_none(self, orchestrator):
        t = FleetTask(progress=0.5)
        assert orchestrator.estimate_completion(t, []) is None

    def test_eta_positive(self, orchestrator):
        t = FleetTask(progress=0.5)
        vessels = [VesselStatus(vessel_id="V1", speed=5.0, health=1.0)]
        eta = orchestrator.estimate_completion(t, vessels)
        assert eta is not None and eta > 0

    def test_eta_multiple_vessels_faster(self, orchestrator):
        t = FleetTask(progress=0.5)
        one = [VesselStatus(vessel_id="V1", speed=5.0, health=1.0)]
        two = [VesselStatus(vessel_id="V1", speed=5.0, health=1.0),
               VesselStatus(vessel_id="V2", speed=5.0, health=1.0)]
        eta_one = orchestrator.estimate_completion(t, one)
        eta_two = orchestrator.estimate_completion(t, two)
        assert eta_two < eta_one

    def test_zero_speed_returns_none(self, orchestrator):
        t = FleetTask(progress=0.5)
        vessels = [VesselStatus(vessel_id="V1", speed=0.0, health=1.0)]
        assert orchestrator.estimate_completion(t, vessels) is None


# ────────────────────────────────────────────────────── Vessel query

class TestVesselTasks:
    def test_get_tasks_for_vessel_empty(self, orchestrator):
        assert orchestrator.get_tasks_for_vessel("V1") == []

    def test_active_task_count_empty(self, orchestrator):
        assert orchestrator.get_active_task_count("V1") == 0
