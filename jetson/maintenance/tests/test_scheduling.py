"""Tests for scheduling module — 30+ tests."""

import pytest
from jetson.maintenance.scheduling import (
    MaintenanceTask,
    ScheduleResult,
    MaintenanceScheduler,
)


@pytest.fixture
def scheduler():
    return MaintenanceScheduler()


@pytest.fixture
def sample_tasks():
    return [
        MaintenanceTask(id="t1", equipment_id="EQ1", priority=10, duration=2.0, cost=500.0),
        MaintenanceTask(id="t2", equipment_id="EQ2", priority=5, duration=1.0, cost=200.0),
        MaintenanceTask(id="t3", equipment_id="EQ3", priority=8, duration=3.0, cost=800.0),
    ]


# ── MaintenanceTask ──────────────────────────────────────────

class TestMaintenanceTask:
    def test_defaults(self):
        t = MaintenanceTask(id="t1", equipment_id="EQ1")
        assert t.priority == 0
        assert t.duration == 1.0
        assert t.deadline is None
        assert t.cost == 0.0
        assert t.scheduled_start is None

    def test_custom(self):
        t = MaintenanceTask(id="t1", equipment_id="EQ1", priority=9,
                            duration=4.0, cost=1000.0, deadline=100.0)
        assert t.deadline == 100.0


# ── ScheduleResult ───────────────────────────────────────────

class TestScheduleResult:
    def test_defaults(self):
        sr = ScheduleResult()
        assert sr.tasks == []
        assert sr.total_cost == 0.0
        assert sr.total_downtime == 0.0
        assert sr.risk_score == 0.0


# ── schedule ─────────────────────────────────────────────────

class TestSchedule:
    def test_basic_scheduling(self, scheduler, sample_tasks):
        windows = [(0.0, 100.0)]
        result = scheduler.schedule(sample_tasks, windows)
        assert len(result.tasks) == 3
        assert result.total_cost == 1500.0

    def test_priority_ordering(self, scheduler):
        tasks = [
            MaintenanceTask(id="low", equipment_id="EQ", priority=1, duration=1.0),
            MaintenanceTask(id="high", equipment_id="EQ", priority=10, duration=1.0),
        ]
        windows = [(0.0, 100.0)]
        result = scheduler.schedule(tasks, windows)
        assert result.tasks[0].id == "high"

    def test_window_too_small(self, scheduler):
        tasks = [MaintenanceTask(id="t1", equipment_id="EQ", priority=5, duration=50.0)]
        windows = [(0.0, 10.0)]
        result = scheduler.schedule(tasks, windows)
        assert result.tasks[0].scheduled_start is None

    def test_multiple_windows(self, scheduler):
        tasks = [
            MaintenanceTask(id="t1", equipment_id="EQ1", priority=5, duration=2.0),
            MaintenanceTask(id="t2", equipment_id="EQ2", priority=3, duration=2.0),
        ]
        windows = [(0.0, 3.0), (5.0, 10.0)]
        result = scheduler.schedule(tasks, windows)
        scheduled = [t for t in result.tasks if t.scheduled_start is not None]
        assert len(scheduled) >= 1

    def test_multiple_resources(self, scheduler):
        tasks = [
            MaintenanceTask(id="t1", equipment_id="EQ1", priority=5, duration=5.0),
            MaintenanceTask(id="t2", equipment_id="EQ2", priority=5, duration=5.0),
        ]
        windows = [(0.0, 100.0)]
        result = scheduler.schedule(tasks, windows, resources=2)
        # Both should be scheduled
        scheduled = [t for t in result.tasks if t.scheduled_start is not None]
        assert len(scheduled) == 2

    def test_empty_tasks(self, scheduler):
        result = scheduler.schedule([], [(0.0, 100.0)])
        assert result.tasks == []
        assert result.total_cost == 0.0

    def test_total_downtime(self, scheduler, sample_tasks):
        windows = [(0.0, 100.0)]
        result = scheduler.schedule(sample_tasks, windows)
        assert result.total_downtime == 6.0

    def test_risk_score(self, scheduler):
        # One high-priority task that can't be scheduled
        tasks = [
            MaintenanceTask(id="t1", equipment_id="EQ1", priority=10, duration=50.0),
        ]
        windows = [(0.0, 10.0)]
        result = scheduler.schedule(tasks, windows)
        assert result.risk_score == 1.0

    def test_no_risk_when_all_scheduled(self, scheduler, sample_tasks):
        windows = [(0.0, 100.0)]
        result = scheduler.schedule(sample_tasks, windows)
        # No unscheduled high-priority tasks
        assert result.risk_score == 0.0


# ── prioritize_by_risk ───────────────────────────────────────

class TestPrioritizeByRisk:
    def test_sort_by_risk(self, scheduler, sample_tasks):
        risk = {"EQ1": 0.9, "EQ2": 0.3, "EQ3": 0.6}
        result = scheduler.prioritize_by_risk(sample_tasks, risk)
        assert result[0].equipment_id == "EQ1"
        assert result[2].equipment_id == "EQ2"

    def test_missing_risk_key(self, scheduler):
        tasks = [MaintenanceTask(id="t1", equipment_id="unknown")]
        result = scheduler.prioritize_by_risk(tasks, {"other": 0.5})
        assert len(result) == 1

    def test_empty(self, scheduler):
        assert scheduler.prioritize_by_risk([], {}) == []


# ── optimize_cost_vs_risk ────────────────────────────────────

class TestOptimizeCostVsRisk:
    def test_within_budget(self, scheduler, sample_tasks):
        result = scheduler.optimize_cost_vs_risk(sample_tasks, budget=2000.0)
        assert len(result.tasks) == 3
        assert result.total_cost <= 2000.0

    def test_limited_budget(self, scheduler):
        tasks = [
            MaintenanceTask(id="t1", equipment_id="EQ1", priority=10, duration=1.0, cost=500.0),
            MaintenanceTask(id="t2", equipment_id="EQ2", priority=1, duration=1.0, cost=100.0),
        ]
        result = scheduler.optimize_cost_vs_risk(tasks, budget=600.0)
        # t1 has higher priority/cost ratio so selected first
        ids = [t.id for t in result.tasks]
        assert "t1" in ids

    def test_zero_budget(self, scheduler, sample_tasks):
        result = scheduler.optimize_cost_vs_risk(sample_tasks, budget=0.0)
        assert len(result.tasks) == 0

    def test_empty_tasks(self, scheduler):
        result = scheduler.optimize_cost_vs_risk([], budget=1000.0)
        assert result.tasks == []

    def test_cost_tracking(self, scheduler):
        tasks = [
            MaintenanceTask(id="t1", equipment_id="EQ1", priority=5, duration=1.0, cost=300.0),
        ]
        result = scheduler.optimize_cost_vs_risk(tasks, budget=500.0)
        assert result.total_cost == 300.0

    def test_priority_ordering_in_result(self, scheduler, sample_tasks):
        result = scheduler.optimize_cost_vs_risk(sample_tasks, budget=10000.0)
        # All should be selected; check they have scheduled_start
        for t in result.tasks:
            assert t.scheduled_start is not None


# ── compute_downtime ─────────────────────────────────────────

class TestComputeDowntime:
    def test_basic(self, scheduler, sample_tasks):
        windows = [(0.0, 100.0)]
        sched = scheduler.schedule(sample_tasks, windows)
        downtime = scheduler.compute_downtime(sched)
        assert downtime == 6.0

    def test_overlapping_tasks(self, scheduler):
        # Create overlapping tasks manually
        t1 = MaintenanceTask(id="t1", equipment_id="EQ1", duration=5.0)
        t1.scheduled_start = 0.0
        t2 = MaintenanceTask(id="t2", equipment_id="EQ2", duration=5.0)
        t2.scheduled_start = 3.0
        sr = ScheduleResult(tasks=[t1, t2])
        downtime = scheduler.compute_downtime(sr)
        # Overlap: max end = 8, start = 0 -> 8 hours
        assert downtime == 8.0

    def test_non_overlapping(self, scheduler):
        t1 = MaintenanceTask(id="t1", equipment_id="EQ1", duration=2.0)
        t1.scheduled_start = 0.0
        t2 = MaintenanceTask(id="t2", equipment_id="EQ2", duration=3.0)
        t2.scheduled_start = 5.0
        sr = ScheduleResult(tasks=[t1, t2])
        downtime = scheduler.compute_downtime(sr)
        assert downtime == 5.0

    def test_unscheduled_ignored(self, scheduler):
        t1 = MaintenanceTask(id="t1", equipment_id="EQ1", duration=5.0)
        t1.scheduled_start = None
        sr = ScheduleResult(tasks=[t1])
        assert scheduler.compute_downtime(sr) == 0.0

    def test_empty_schedule(self, scheduler):
        sr = ScheduleResult()
        assert scheduler.compute_downtime(sr) == 0.0


# ── reschedule_after_failure ─────────────────────────────────

class TestRescheduleAfterFailure:
    def test_failed_task_promoted(self, scheduler, sample_tasks):
        windows = [(0.0, 100.0)]
        sched = scheduler.schedule(sample_tasks, windows)
        failed = MaintenanceTask(id="t_fail", equipment_id="EQ_F", priority=1, duration=2.0)
        new_sched = scheduler.reschedule_after_failure(sched, failed)
        # Failed task should be first (highest priority)
        assert new_sched.tasks[0].id == "t_fail"
        assert new_sched.tasks[0].priority > 10

    def test_original_tasks_preserved(self, scheduler, sample_tasks):
        windows = [(0.0, 100.0)]
        sched = scheduler.schedule(sample_tasks, windows)
        failed = MaintenanceTask(id="t_fail", equipment_id="EQ_F", priority=1, duration=1.0)
        new_sched = scheduler.reschedule_after_failure(sched, failed)
        original_ids = {t.id for t in sample_tasks}
        new_ids = {t.id for t in new_sched.tasks}
        assert original_ids.issubset(new_ids)

    def test_empty_schedule(self, scheduler):
        failed = MaintenanceTask(id="t_fail", equipment_id="EQ", priority=1)
        new_sched = scheduler.reschedule_after_failure(
            ScheduleResult(), failed
        )
        assert len(new_sched.tasks) == 1

    def test_total_cost_updated(self, scheduler, sample_tasks):
        windows = [(0.0, 100.0)]
        sched = scheduler.schedule(sample_tasks, windows)
        failed = MaintenanceTask(id="t_fail", equipment_id="EQ_F", priority=1,
                                  duration=1.0, cost=100.0)
        new_sched = scheduler.reschedule_after_failure(sched, failed)
        assert new_sched.total_cost == sched.total_cost + 100.0
