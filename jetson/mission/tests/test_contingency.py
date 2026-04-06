"""Tests for contingency and abort management."""

import pytest
from jetson.mission.planner import (
    MissionPlan, MissionPhase, MissionObjective, RiskLevel,
)
from jetson.mission.contingency import (
    ContingencyPriority,
    ContingencyStatus,
    AbortSeverity,
    ContingencyAction,
    ContingencyPlan,
    AbortCriteria,
    TriggerEvaluation,
    ContingencyResult,
    AbortRecommendation,
    FallbackPlan,
    ContingencyManager,
)


class TestContingencyPriority:
    def test_critical(self):
        assert ContingencyPriority.CRITICAL.value == 1

    def test_high(self):
        assert ContingencyPriority.HIGH.value == 2

    def test_medium(self):
        assert ContingencyPriority.MEDIUM.value == 3

    def test_low(self):
        assert ContingencyPriority.LOW.value == 4


class TestContingencyStatus:
    def test_idle(self):
        assert ContingencyStatus.IDLE.value == "idle"

    def test_active(self):
        assert ContingencyStatus.ACTIVE.value == "active"

    def test_executing(self):
        assert ContingencyStatus.EXECUTING.value == "executing"

    def test_completed(self):
        assert ContingencyStatus.COMPLETED.value == "completed"

    def test_failed(self):
        assert ContingencyStatus.FAILED.value == "failed"


class TestAbortSeverity:
    def test_all_levels(self):
        levels = {s.value for s in AbortSeverity}
        assert levels == {"low", "medium", "high", "critical"}


class TestContingencyAction:
    def test_default(self):
        a = ContingencyAction()
        assert a.name == ""
        assert a.timeout == 30.0

    def test_custom(self):
        a = ContingencyAction(name="reroute", action_type="navigate",
                              parameters={"waypoint": "safe"}, timeout=60.0)
        assert a.action_type == "navigate"
        assert a.parameters["waypoint"] == "safe"


class TestContingencyPlan:
    def test_default_creation(self):
        cp = ContingencyPlan()
        assert cp.trigger_condition == ""
        assert cp.priority == ContingencyPriority.MEDIUM
        assert cp.status == ContingencyStatus.IDLE

    def test_custom_creation(self):
        actions = [ContingencyAction(name="retreat")]
        cp = ContingencyPlan(
            name="collision_avoidance",
            trigger_condition="obstacle_distance < 5",
            trigger_type="threshold",
            response_actions=actions,
            priority=ContingencyPriority.CRITICAL,
            preconditions=["sensor_active"],
            estimated_recovery_time=120.0,
        )
        assert cp.name == "collision_avoidance"
        assert len(cp.response_actions) == 1
        assert cp.priority == ContingencyPriority.CRITICAL


class TestAbortCriteria:
    def test_default_creation(self):
        ac = AbortCriteria()
        assert ac.severity == AbortSeverity.MEDIUM
        assert ac.auto_abort is False

    def test_custom_creation(self):
        ac = AbortCriteria(
            condition="hull_integrity < 50",
            severity=AbortSeverity.CRITICAL,
            auto_abort=True,
            notification="OPERATOR",
            description="Hull breach",
        )
        assert ac.auto_abort is True
        assert ac.notification == "OPERATOR"


class TestTriggerEvaluation:
    def test_default(self):
        te = TriggerEvaluation(trigger_id="t1")
        assert te.triggered is False
        assert te.severity == 0.0


class TestContingencyResult:
    def test_default(self):
        cr = ContingencyResult()
        assert cr.success is False
        assert cr.actions_completed == 0


class TestAbortRecommendation:
    def test_default(self):
        ar = AbortRecommendation()
        assert ar.should_abort is False
        assert ar.severity == AbortSeverity.LOW


class TestFallbackPlan:
    def test_default(self):
        fp = FallbackPlan()
        assert fp.original_plan_id == ""
        assert fp.fallback_phases == []


class TestContingencyManager:
    def setup_method(self):
        self.manager = ContingencyManager()

    def test_register_contingency(self):
        cp = ContingencyPlan(name="test")
        cid = self.manager.register_contingency(cp)
        assert cid == cp.id

    def test_unregister_contingency(self):
        cp = ContingencyPlan(name="test")
        self.manager.register_contingency(cp)
        assert self.manager.unregister_contingency(cp.id) is True

    def test_unregister_nonexistent(self):
        assert self.manager.unregister_contingency("bad") is False

    def test_get_contingency(self):
        cp = ContingencyPlan(name="test")
        self.manager.register_contingency(cp)
        retrieved = self.manager.get_contingency(cp.id)
        assert retrieved is not None
        assert retrieved.name == "test"

    def test_get_contingency_nonexistent(self):
        assert self.manager.get_contingency("bad") is None

    def test_evaluate_triggers_no_match(self):
        cp = ContingencyPlan(
            name="high_temp",
            trigger_condition="temperature > 100",
            trigger_type="threshold",
        )
        self.manager.register_contingency(cp)
        results = self.manager.evaluate_triggers({"temperature": 50.0})
        assert len(results) == 1
        assert results[0].triggered is False

    def test_evaluate_triggers_match(self):
        cp = ContingencyPlan(
            name="high_temp",
            trigger_condition="temperature > 40",
            trigger_type="threshold",
        )
        self.manager.register_contingency(cp)
        results = self.manager.evaluate_triggers({"temperature": 50.0})
        assert results[0].triggered is True

    def test_evaluate_triggers_less_than(self):
        cp = ContingencyPlan(
            name="low_battery",
            trigger_condition="battery < 20",
            trigger_type="threshold",
        )
        self.manager.register_contingency(cp)
        results = self.manager.evaluate_triggers({"battery": 10.0})
        assert results[0].triggered is True

    def test_evaluate_triggers_greater_equal(self):
        cp = ContingencyPlan(
            name="pressure",
            trigger_condition="pressure >= 100",
            trigger_type="threshold",
        )
        self.manager.register_contingency(cp)
        results = self.manager.evaluate_triggers({"pressure": 100.0})
        assert results[0].triggered is True

    def test_evaluate_triggers_less_equal(self):
        cp = ContingencyPlan(
            name="depth",
            trigger_condition="depth <= 50",
            trigger_type="threshold",
        )
        self.manager.register_contingency(cp)
        results = self.manager.evaluate_triggers({"depth": 30.0})
        assert results[0].triggered is True

    def test_evaluate_triggers_equals(self):
        cp = ContingencyPlan(
            name="mode",
            trigger_condition="mode == emergency",
            trigger_type="threshold",
        )
        self.manager.register_contingency(cp)
        results = self.manager.evaluate_triggers({"mode": "emergency"})
        assert results[0].triggered is True

    def test_evaluate_triggers_preconditions_met(self):
        cp = ContingencyPlan(
            name="test",
            trigger_condition="temperature > 80",
            trigger_type="threshold",
            preconditions=["sensor_active"],
        )
        self.manager.register_contingency(cp)
        results = self.manager.evaluate_triggers({
            "temperature": 90.0,
            "sensor_active": True,
        })
        assert results[0].triggered is True
        assert results[0].preconditions_met is True

    def test_evaluate_triggers_preconditions_not_met(self):
        cp = ContingencyPlan(
            name="test",
            trigger_condition="temperature > 80",
            trigger_type="threshold",
            preconditions=["sensor_active"],
        )
        self.manager.register_contingency(cp)
        results = self.manager.evaluate_triggers({
            "temperature": 90.0,
            "sensor_active": False,
        })
        assert results[0].triggered is False
        assert results[0].preconditions_met is False

    def test_evaluate_triggers_specific_contingencies(self):
        cp1 = ContingencyPlan(name="a", trigger_condition="x > 5", trigger_type="threshold")
        cp2 = ContingencyPlan(name="b", trigger_condition="y > 5", trigger_type="threshold")
        self.manager.register_contingency(cp1)
        self.manager.register_contingency(cp2)
        results = self.manager.evaluate_triggers({"x": 10, "y": 1}, [cp1.id])
        assert len(results) == 1

    def test_evaluate_triggers_sets_active(self):
        cp = ContingencyPlan(
            name="test",
            trigger_condition="v > 0",
            trigger_type="threshold",
        )
        self.manager.register_contingency(cp)
        self.manager.evaluate_triggers({"v": 1.0})
        assert cp.status == ContingencyStatus.ACTIVE

    def test_execute_contingency_success(self):
        actions = [
            ContingencyAction(name="a1"),
            ContingencyAction(name="a2"),
        ]
        cp = ContingencyPlan(
            name="test",
            response_actions=actions,
        )
        self.manager.register_contingency(cp)
        result = self.manager.execute_contingency(cp.id)
        assert result.success is True
        assert result.actions_completed == 2
        assert result.actions_total == 2

    def test_execute_contingency_not_found(self):
        result = self.manager.execute_contingency("nonexistent")
        assert result.success is False
        assert len(result.errors) > 0

    def test_execute_contingency_empty_actions(self):
        cp = ContingencyPlan(name="empty")
        self.manager.register_contingency(cp)
        result = self.manager.execute_contingency(cp.id)
        assert result.success is True
        assert result.actions_completed == 0

    def test_execute_contingency_updates_status(self):
        cp = ContingencyPlan(
            name="test",
            response_actions=[ContingencyAction(name="a1")],
        )
        self.manager.register_contingency(cp)
        self.manager.execute_contingency(cp.id)
        assert cp.status == ContingencyStatus.COMPLETED

    def test_register_abort_criteria(self):
        ac = AbortCriteria(condition="hull_breach", severity=AbortSeverity.CRITICAL)
        cid = self.manager.register_abort_criteria(ac)
        assert cid == ac.id
        assert len(self.manager.get_abort_criteria()) == 1

    def test_unregister_abort_criteria(self):
        ac = AbortCriteria(condition="test")
        self.manager.register_abort_criteria(ac)
        assert self.manager.unregister_abort_criteria(ac.id) is True
        assert len(self.manager.get_abort_criteria()) == 0

    def test_unregister_abort_criteria_nonexistent(self):
        assert self.manager.unregister_abort_criteria("bad") is False

    def test_evaluate_abort_no_criteria(self):
        rec = self.manager.evaluate_abort({})
        assert rec.should_abort is False

    def test_evaluate_abort_not_triggered(self):
        self.manager.register_abort_criteria(
            AbortCriteria(condition="hull_integrity", severity=AbortSeverity.HIGH)
        )
        rec = self.manager.evaluate_abort({"hull_integrity": False})
        assert rec.should_abort is False

    def test_evaluate_abort_triggered_high(self):
        self.manager.register_abort_criteria(
            AbortCriteria(condition="hull_breach", severity=AbortSeverity.HIGH)
        )
        rec = self.manager.evaluate_abort({"hull_breach": True})
        assert rec.should_abort is True
        assert rec.severity == AbortSeverity.HIGH

    def test_evaluate_abort_auto_abort(self):
        self.manager.register_abort_criteria(
            AbortCriteria(
                condition="critical_failure",
                severity=AbortSeverity.CRITICAL,
                auto_abort=True,
            )
        )
        rec = self.manager.evaluate_abort({"critical_failure": True})
        assert rec.should_abort is True
        assert rec.auto_abort is True

    def test_evaluate_abort_with_check_fn(self):
        def check(state):
            return state.get("value", 0) > 100
        self.manager.register_abort_criteria(
            AbortCriteria(
                condition="custom",
                severity=AbortSeverity.HIGH,
                check_fn=check,
            )
        )
        rec = self.manager.evaluate_abort({"value": 150})
        assert rec.should_abort is True

    def test_evaluate_abort_check_fn_false(self):
        def check(state):
            return state.get("value", 0) > 100
        self.manager.register_abort_criteria(
            AbortCriteria(condition="custom", severity=AbortSeverity.HIGH, check_fn=check)
        )
        rec = self.manager.evaluate_abort({"value": 50})
        assert rec.should_abort is False

    def test_generate_fallback_plan(self):
        plan = MissionPlan(
            id="orig",
            objectives=[
                MissionObjective(id="o1"),
                MissionObjective(id="o2"),
                MissionObjective(id="o3"),
            ],
            phases=[
                MissionPhase(name="p1", duration=60.0),
                MissionPhase(name="p2", duration=60.0),
                MissionPhase(name="p3", duration=60.0),
            ],
        )
        fb = self.manager.generate_fallback_plan(plan, "p2")
        assert fb.original_plan_id == "orig"
        assert fb.failure_point == "p2"
        assert len(fb.objectives_preserved) == 1
        assert len(fb.objectives_dropped) >= 1
        assert fb.risk_increase > 0

    def test_generate_fallback_plan_unknown_phase(self):
        plan = MissionPlan(
            id="orig",
            objectives=[MissionObjective()],
            phases=[MissionPhase(name="p1", duration=60.0)],
        )
        fb = self.manager.generate_fallback_plan(plan, "unknown")
        assert fb.failure_point == "unknown"

    def test_generate_fallback_plan_no_remaining(self):
        plan = MissionPlan(
            id="orig",
            objectives=[MissionObjective()],
            phases=[MissionPhase(name="p1", duration=60.0)],
        )
        fb = self.manager.generate_fallback_plan(plan, "p1")
        assert any(p.name == "safe_return" for p in fb.fallback_phases)

    def test_compute_recovery_time(self):
        cp = ContingencyPlan(
            estimated_recovery_time=60.0,
            response_actions=[
                ContingencyAction(name="a1", timeout=10.0),
                ContingencyAction(name="a2", timeout=20.0),
            ],
            priority=ContingencyPriority.HIGH,
        )
        rt = self.manager.compute_recovery_time(cp)
        assert rt > 60.0

    def test_compute_recovery_time_critical_priority(self):
        cp = ContingencyPlan(
            estimated_recovery_time=30.0,
            response_actions=[ContingencyAction(name="a1", timeout=10.0)],
            priority=ContingencyPriority.CRITICAL,
        )
        rt = self.manager.compute_recovery_time(cp)
        assert rt > 30.0

    def test_get_execution_log(self):
        cp = ContingencyPlan(response_actions=[ContingencyAction(name="a1")])
        self.manager.register_contingency(cp)
        self.manager.execute_contingency(cp.id)
        log = self.manager.get_execution_log()
        assert len(log) == 1

    def test_get_trigger_history(self):
        cp = ContingencyPlan(
            trigger_condition="x > 0", trigger_type="threshold",
        )
        self.manager.register_contingency(cp)
        self.manager.evaluate_triggers({"x": 1.0})
        history = self.manager.get_trigger_history()
        assert len(history) == 1

    def test_get_active_contingencies(self):
        cp = ContingencyPlan(
            trigger_condition="x > 0", trigger_type="threshold",
        )
        self.manager.register_contingency(cp)
        self.manager.evaluate_triggers({"x": 1.0})
        active = self.manager.get_active_contingencies()
        assert len(active) == 1

    def test_reset_contingency(self):
        cp = ContingencyPlan(trigger_condition="x > 0", trigger_type="threshold")
        self.manager.register_contingency(cp)
        self.manager.evaluate_triggers({"x": 1.0})
        assert self.manager.reset_contingency(cp.id) is True
        assert cp.status == ContingencyStatus.IDLE

    def test_reset_nonexistent(self):
        assert self.manager.reset_contingency("bad") is False

    def test_clear(self):
        self.manager.register_contingency(ContingencyPlan(name="a"))
        self.manager.register_abort_criteria(AbortCriteria(condition="b"))
        self.manager.clear()
        assert len(self.manager.get_abort_criteria()) == 0
        assert self.manager.get_contingency("a") is None

    def test_trigger_evaluation_message(self):
        cp = ContingencyPlan(name="test", trigger_condition="v > 5", trigger_type="threshold")
        self.manager.register_contingency(cp)
        results = self.manager.evaluate_triggers({"v": 10})
        assert "test" in results[0].message
        assert "TRIGGERED" in results[0].message

    def test_evaluate_triggers_nonexistent_contingency(self):
        results = self.manager.evaluate_triggers({}, ["nonexistent"])
        assert results == []

    def test_abort_recommendation_triggered_criteria(self):
        self.manager.register_abort_criteria(
            AbortCriteria(condition="a", severity=AbortSeverity.HIGH)
        )
        rec = self.manager.evaluate_abort({"a": True})
        assert len(rec.triggered_criteria) == 1
