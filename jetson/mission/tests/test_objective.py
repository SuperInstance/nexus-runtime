"""Tests for objective management module."""

import pytest
from jetson.mission.objective import (
    ObjectiveStatus,
    ObjectivePriority,
    MissionObjective,
    ObjectiveResult,
    PriorityConflict,
    ObjectiveReport,
    ObjectiveManager,
)


class TestObjectiveStatus:
    def test_pending(self):
        assert ObjectiveStatus.PENDING.value == "pending"

    def test_in_progress(self):
        assert ObjectiveStatus.IN_PROGRESS.value == "in_progress"

    def test_completed(self):
        assert ObjectiveStatus.COMPLETED.value == "completed"

    def test_failed(self):
        assert ObjectiveStatus.FAILED.value == "failed"

    def test_blocked(self):
        assert ObjectiveStatus.BLOCKED.value == "blocked"


class TestObjectivePriority:
    def test_critical(self):
        assert ObjectivePriority.CRITICAL.value == 1

    def test_high(self):
        assert ObjectivePriority.HIGH.value == 2

    def test_medium(self):
        assert ObjectivePriority.MEDIUM.value == 3

    def test_low(self):
        assert ObjectivePriority.LOW.value == 4


class TestMissionObjective:
    def test_default_creation(self):
        obj = MissionObjective()
        assert obj.name == ""
        assert obj.objective_type == "general"
        assert obj.priority == ObjectivePriority.MEDIUM
        assert obj.constraints == {}
        assert obj.deadline is None

    def test_custom_creation(self):
        obj = MissionObjective(
            id="custom",
            name="survey_area",
            objective_type="survey",
            target=100.0,
            priority=ObjectivePriority.CRITICAL,
            constraints={"max_depth": 50},
            deadline=3600.0,
            weight=2.0,
            description="Survey area A",
            tags=["survey", "area_a"],
        )
        assert obj.id == "custom"
        assert obj.name == "survey_area"
        assert obj.objective_type == "survey"
        assert obj.target == 100.0
        assert obj.priority == ObjectivePriority.CRITICAL
        assert obj.weight == 2.0
        assert "survey" in obj.tags

    def test_auto_id(self):
        obj1 = MissionObjective()
        obj2 = MissionObjective()
        assert obj1.id != obj2.id

    def test_default_weight(self):
        obj = MissionObjective()
        assert obj.weight == 1.0

    def test_default_tags(self):
        obj = MissionObjective()
        assert obj.tags == []


class TestObjectiveResult:
    def test_default_creation(self):
        r = ObjectiveResult()
        assert r.status == ObjectiveStatus.PENDING
        assert r.actual_value is None
        assert r.deviation == 0.0

    def test_custom_creation(self):
        obj = MissionObjective(id="o1")
        r = ObjectiveResult(
            objective=obj,
            status=ObjectiveStatus.COMPLETED,
            actual_value=95.0,
            target_value=100.0,
            deviation=0.05,
            notes="Close to target",
        )
        assert r.objective.id == "o1"
        assert r.status == ObjectiveStatus.COMPLETED
        assert r.deviation == 0.05
        assert r.notes == "Close to target"


class TestPriorityConflict:
    def test_default_creation(self):
        pc = PriorityConflict(
            objective_a="o1", objective_b="o2",
            conflict_type="resource", severity=0.5,
            description="Overlap",
        )
        assert pc.objective_a == "o1"
        assert pc.conflict_type == "resource"
        assert pc.severity == 0.5


class TestObjectiveReport:
    def test_default_creation(self):
        r = ObjectiveReport()
        assert r.total_objectives == 0
        assert r.overall_score == 0.0
        assert r.details == []

    def test_custom_creation(self):
        r = ObjectiveReport(
            total_objectives=5,
            completed=3,
            in_progress=1,
            pending=1,
            failed=0,
            blocked=0,
            overall_score=2.5,
        )
        assert r.completed == 3
        assert r.overall_score == 2.5


class TestObjectiveManager:
    def setup_method(self):
        self.mgr = ObjectiveManager()

    def test_add_objective(self):
        obj = MissionObjective(name="survey", objective_type="survey")
        oid = self.mgr.add_objective(obj)
        assert oid == obj.id
        retrieved = self.mgr.get_objective(oid)
        assert retrieved is not None

    def test_add_objective_duplicate(self):
        obj = MissionObjective(id="dup", name="test")
        self.mgr.add_objective(obj)
        with pytest.raises(ValueError, match="already exists"):
            self.mgr.add_objective(obj)

    def test_remove_objective(self):
        obj = MissionObjective(name="test")
        self.mgr.add_objective(obj)
        assert self.mgr.remove_objective(obj.id) is True
        assert self.mgr.get_objective(obj.id) is None

    def test_remove_objective_nonexistent(self):
        assert self.mgr.remove_objective("bad") is False

    def test_get_objective(self):
        obj = MissionObjective(name="find_me")
        self.mgr.add_objective(obj)
        retrieved = self.mgr.get_objective(obj.id)
        assert retrieved.name == "find_me"

    def test_get_objective_nonexistent(self):
        assert self.mgr.get_objective("bad") is None

    def test_update_status(self):
        obj = MissionObjective(name="test")
        self.mgr.add_objective(obj)
        assert self.mgr.update_status(obj.id, ObjectiveStatus.IN_PROGRESS) is True
        assert self.mgr.get_status(obj.id) == ObjectiveStatus.IN_PROGRESS

    def test_update_status_nonexistent(self):
        assert self.mgr.update_status("bad", ObjectiveStatus.COMPLETED) is False

    def test_update_status_completed_sets_achieved_at(self):
        obj = MissionObjective(name="test")
        self.mgr.add_objective(obj)
        self.mgr.update_status(obj.id, ObjectiveStatus.COMPLETED)
        result = self.mgr.get_result(obj.id)
        assert result.achieved_at is not None

    def test_get_status(self):
        obj = MissionObjective(name="test")
        self.mgr.add_objective(obj)
        assert self.mgr.get_status(obj.id) == ObjectiveStatus.PENDING

    def test_get_status_nonexistent(self):
        assert self.mgr.get_status("bad") is None

    def test_get_result(self):
        obj = MissionObjective(name="test")
        self.mgr.add_objective(obj)
        result = self.mgr.get_result(obj.id)
        assert result is not None
        assert result.objective.id == obj.id

    def test_get_result_nonexistent(self):
        assert self.mgr.get_result("bad") is None

    def test_set_actual_value(self):
        obj = MissionObjective(name="test", target=100.0)
        self.mgr.add_objective(obj)
        assert self.mgr.set_actual_value(obj.id, 95.0) is True
        result = self.mgr.get_result(obj.id)
        assert result.actual_value == 95.0
        assert abs(result.deviation - 0.05) < 0.001

    def test_set_actual_value_nonexistent(self):
        assert self.mgr.set_actual_value("bad", 50.0) is False

    def test_set_actual_value_zero_target(self):
        obj = MissionObjective(name="test", target=0.0)
        self.mgr.add_objective(obj)
        self.mgr.set_actual_value(obj.id, 50.0)
        result = self.mgr.get_result(obj.id)
        assert result.deviation == 0.0

    def test_set_actual_value_non_numeric(self):
        obj = MissionObjective(name="test", target="complete")
        self.mgr.add_objective(obj)
        self.mgr.set_actual_value(obj.id, "done")
        result = self.mgr.get_result(obj.id)
        assert result.actual_value == "done"

    def test_check_completion_empty(self):
        assert self.mgr.check_completion() == []

    def test_check_completion_none_completed(self):
        obj = MissionObjective(name="test")
        self.mgr.add_objective(obj)
        assert self.mgr.check_completion() == []

    def test_check_completion_with_completed(self):
        obj1 = MissionObjective(name="test1")
        obj2 = MissionObjective(name="test2")
        self.mgr.add_objective(obj1)
        self.mgr.add_objective(obj2)
        self.mgr.update_status(obj1.id, ObjectiveStatus.COMPLETED)
        completed = self.mgr.check_completion()
        assert obj1.id in completed
        assert obj2.id not in completed

    def test_check_completion_specific_ids(self):
        obj1 = MissionObjective(name="test1")
        obj2 = MissionObjective(name="test2")
        self.mgr.add_objective(obj1)
        self.mgr.add_objective(obj2)
        self.mgr.update_status(obj1.id, ObjectiveStatus.COMPLETED)
        completed = self.mgr.check_completion([obj1.id])
        assert len(completed) == 1

    def test_check_pending(self):
        obj = MissionObjective(name="test")
        self.mgr.add_objective(obj)
        pending = self.mgr.check_pending()
        assert obj.id in pending

    def test_check_blocked(self):
        obj = MissionObjective(name="test")
        self.mgr.add_objective(obj)
        self.mgr.update_status(obj.id, ObjectiveStatus.BLOCKED)
        blocked = self.mgr.check_blocked()
        assert obj.id in blocked

    def test_compute_priority_conflicts_no_conflicts(self):
        obj1 = MissionObjective(name="a", constraints={"required_resources": ["sonar"]})
        obj2 = MissionObjective(name="b", constraints={"required_resources": ["camera"]})
        self.mgr.add_objective(obj1)
        self.mgr.add_objective(obj2)
        conflicts = self.mgr.compute_priority_conflicts()
        assert conflicts == []

    def test_compute_priority_conflicts_resource_overlap(self):
        obj1 = MissionObjective(name="a", constraints={"required_resources": ["sonar", "camera"]})
        obj2 = MissionObjective(name="b", constraints={"required_resources": ["camera", "gps"]})
        self.mgr.add_objective(obj1)
        self.mgr.add_objective(obj2)
        conflicts = self.mgr.compute_priority_conflicts()
        assert len(conflicts) >= 1
        assert conflicts[0].conflict_type == "resource_overlap"

    def test_compute_priority_conflicts_deadline_mismatch(self):
        obj1 = MissionObjective(
            name="a",
            priority=ObjectivePriority.CRITICAL,
            deadline=7200.0,
        )
        obj2 = MissionObjective(
            name="b",
            priority=ObjectivePriority.LOW,
            deadline=3600.0,
        )
        self.mgr.add_objective(obj1)
        self.mgr.add_objective(obj2)
        conflicts = self.mgr.compute_priority_conflicts()
        assert any(c.conflict_type == "priority_deadline_mismatch" for c in conflicts)

    def test_reallocate_priorities_deadline(self):
        obj = MissionObjective(
            name="test",
            priority=ObjectivePriority.LOW,
            deadline=7200.0,
        )
        self.mgr.add_objective(obj)
        affected = self.mgr.reallocate_priorities(
            [obj.id], {"type": "deadline", "deadline": 1800.0}
        )
        assert obj.id in affected
        assert obj.priority.value < ObjectivePriority.LOW.value

    def test_reallocate_priorities_resource_limit(self):
        obj = MissionObjective(
            name="test",
            priority=ObjectivePriority.MEDIUM,
            constraints={"required_resources": ["sonar"]},
        )
        self.mgr.add_objective(obj)
        affected = self.mgr.reallocate_priorities(
            [obj.id], {"type": "resource_limit", "resources": ["sonar"]}
        )
        assert obj.id in affected
        assert obj.priority.value < ObjectivePriority.MEDIUM.value

    def test_reallocate_priorities_no_match(self):
        obj = MissionObjective(
            name="test",
            priority=ObjectivePriority.LOW,
            deadline=3600.0,
        )
        self.mgr.add_objective(obj)
        affected = self.mgr.reallocate_priorities(
            [obj.id], {"type": "deadline", "deadline": 7200.0}
        )
        assert obj.id not in affected

    def test_compute_objective_value_completed(self):
        obj = MissionObjective(name="test", priority=ObjectivePriority.MEDIUM)
        result = ObjectiveResult(
            objective=obj,
            status=ObjectiveStatus.COMPLETED,
            deviation=0.0,
        )
        value = self.mgr.compute_objective_value(result)
        assert value > 0

    def test_compute_objective_value_failed(self):
        obj = MissionObjective(name="test", priority=ObjectivePriority.CRITICAL)
        result = ObjectiveResult(
            objective=obj,
            status=ObjectiveStatus.FAILED,
        )
        value = self.mgr.compute_objective_value(result)
        assert value == 0.0

    def test_compute_objective_value_blocked(self):
        obj = MissionObjective(name="test")
        result = ObjectiveResult(objective=obj, status=ObjectiveStatus.BLOCKED)
        assert self.mgr.compute_objective_value(result) == 0.0

    def test_compute_objective_value_pending(self):
        obj = MissionObjective(name="test")
        result = ObjectiveResult(objective=obj, status=ObjectiveStatus.PENDING)
        assert self.mgr.compute_objective_value(result) == 0.0

    def test_compute_objective_value_in_progress(self):
        obj = MissionObjective(name="test", weight=1.0, priority=ObjectivePriority.MEDIUM)
        result = ObjectiveResult(
            objective=obj,
            status=ObjectiveStatus.IN_PROGRESS,
            deviation=0.0,
        )
        value = self.mgr.compute_objective_value(result)
        assert value > 0
        # In progress value should be less than completed
        completed_result = ObjectiveResult(
            objective=obj, status=ObjectiveStatus.COMPLETED, deviation=0.0,
        )
        assert value < self.mgr.compute_objective_value(completed_result)

    def test_compute_objective_value_with_deviation(self):
        obj = MissionObjective(name="test", priority=ObjectivePriority.MEDIUM)
        result_no_dev = ObjectiveResult(
            objective=obj, status=ObjectiveStatus.COMPLETED, deviation=0.0,
        )
        result_dev = ObjectiveResult(
            objective=obj, status=ObjectiveStatus.COMPLETED, deviation=0.5,
        )
        assert self.mgr.compute_objective_value(result_no_dev) > \
               self.mgr.compute_objective_value(result_dev)

    def test_compute_objective_value_critical_priority(self):
        obj = MissionObjective(name="test", priority=ObjectivePriority.CRITICAL)
        result = ObjectiveResult(
            objective=obj, status=ObjectiveStatus.COMPLETED, deviation=0.0,
        )
        assert self.mgr.compute_objective_value(result) > 0

    def test_generate_objective_report_empty(self):
        report = self.mgr.generate_objective_report()
        assert report.total_objectives == 0
        assert report.overall_score == 0.0

    def test_generate_objective_report_multiple(self):
        obj1 = MissionObjective(name="a")
        obj2 = MissionObjective(name="b")
        obj3 = MissionObjective(name="c")
        self.mgr.add_objective(obj1)
        self.mgr.add_objective(obj2)
        self.mgr.add_objective(obj3)
        self.mgr.update_status(obj1.id, ObjectiveStatus.COMPLETED)
        self.mgr.update_status(obj2.id, ObjectiveStatus.IN_PROGRESS)
        self.mgr.update_status(obj3.id, ObjectiveStatus.FAILED)

        report = self.mgr.generate_objective_report()
        assert report.total_objectives == 3
        assert report.completed == 1
        assert report.in_progress == 1
        assert report.failed == 1
        assert len(report.details) == 3

    def test_generate_objective_report_score(self):
        obj = MissionObjective(name="test", priority=ObjectivePriority.MEDIUM)
        self.mgr.add_objective(obj)
        self.mgr.update_status(obj.id, ObjectiveStatus.COMPLETED)
        report = self.mgr.generate_objective_report()
        assert report.overall_score > 0

    def test_generate_objective_report_with_values(self):
        obj = MissionObjective(name="test", target=100.0)
        self.mgr.add_objective(obj)
        self.mgr.update_status(obj.id, ObjectiveStatus.COMPLETED)
        self.mgr.set_actual_value(obj.id, 90.0)
        report = self.mgr.generate_objective_report()
        detail = report.details[0]
        assert detail["score"] > 0

    def test_get_all_objectives(self):
        obj1 = MissionObjective(name="a")
        obj2 = MissionObjective(name="b")
        self.mgr.add_objective(obj1)
        self.mgr.add_objective(obj2)
        all_objs = self.mgr.get_all_objectives()
        assert len(all_objs) == 2

    def test_get_status_history(self):
        obj = MissionObjective(name="test")
        self.mgr.add_objective(obj)
        self.mgr.update_status(obj.id, ObjectiveStatus.IN_PROGRESS)
        self.mgr.update_status(obj.id, ObjectiveStatus.COMPLETED)
        history = self.mgr.get_status_history(obj.id)
        assert len(history) == 3  # PENDING, IN_PROGRESS, COMPLETED

    def test_get_status_history_empty(self):
        history = self.mgr.get_status_history("bad")
        assert history == []

    def test_clear(self):
        obj = MissionObjective(name="test")
        self.mgr.add_objective(obj)
        self.mgr.clear()
        assert self.mgr.get_all_objectives() == []
        assert self.mgr.check_pending() == []

    def test_multiple_objectives_independent(self):
        obj1 = MissionObjective(name="a", target=100.0)
        obj2 = MissionObjective(name="b", target=50.0)
        self.mgr.add_objective(obj1)
        self.mgr.add_objective(obj2)
        self.mgr.set_actual_value(obj1.id, 100.0)
        self.mgr.set_actual_value(obj2.id, 25.0)
        r1 = self.mgr.get_result(obj1.id)
        r2 = self.mgr.get_result(obj2.id)
        assert r1.actual_value == 100.0
        assert r2.actual_value == 25.0
