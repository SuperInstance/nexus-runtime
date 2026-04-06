"""Cross-module integration tests: Mission lifecycle (planner, executor, monitor, contingency, autonomy).

Each test calls 2+ mission/autonomy modules together.
"""

import time
import pytest

from jetson.mission.planner import (
    MissionPlanner, MissionPlan, MissionPhase, MissionObjective,
    MissionAction, RiskLevel, ResourceRequirements, RiskAssessment,
)
from jetson.mission.execution import (
    MissionExecutor, ExecutionState, PhaseResult, MissionResult, TransitionResult,
)
from jetson.mission.monitoring import (
    MissionMonitor, ProgressMetric, MissionAlert, AlertLevel, TrendDirection,
    DeviationReport, StatusReport, ResourceWarning,
)
from jetson.mission.contingency import (
    ContingencyManager, ContingencyPlan, ContingencyAction, ContingencyPriority,
    ContingencyStatus, AbortCriteria, AbortSeverity, FallbackPlan,
)
from jetson.adaptive_autonomy.transition import (
    TransitionManager, TransitionRequest, TransitionPolicy,
)
from jetson.adaptive_autonomy.override import (
    OverrideManager, OverrideRequest,
)
from jetson.adaptive_autonomy.levels import AutonomyLevel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_objective(name="survey", obj_type="survey", priority=3):
    return MissionObjective(name=name, type=obj_type, priority=priority)


def _make_plan(planner=None, n_objectives=2):
    if planner is None:
        planner = MissionPlanner()
    objectives = [_make_objective(f"obj_{i}", f"survey") for i in range(n_objectives)]
    return planner.create_plan(objectives)


# ===========================================================================
# 1. Planner + Executor cross-module tests
# ===========================================================================

class TestPlannerExecutorIntegration:
    """Tests that exercise MissionPlanner together with MissionExecutor."""

    def test_create_and_execute_plan(self):
        planner = MissionPlanner()
        plan = _make_plan(planner)
        executor = MissionExecutor()
        result = executor.execute_plan(plan)
        assert result.state == ExecutionState.COMPLETED
        assert result.plan_id == plan.id

    def test_execute_plan_with_multiple_phases(self):
        planner = MissionPlanner()
        objectives = [_make_objective(f"obj_{i}") for i in range(5)]
        plan = planner.create_plan(objectives)
        executor = MissionExecutor()
        result = executor.execute_plan(plan)
        assert len(result.completed_phases) == len(plan.phases)
        assert result.total_progress == 100.0

    def test_plan_history_reflects_created_plans(self):
        planner = MissionPlanner()
        planner.create_plan([_make_objective()])
        planner.create_plan([_make_objective("other")])
        assert len(planner.get_plan_history()) == 2
        executor = MissionExecutor()
        plan = planner.get_plan_history()[0]
        result = executor.execute_plan(plan)
        assert result.state == ExecutionState.COMPLETED

    def test_optimize_plan_then_execute(self):
        planner = MissionPlanner()
        plan = _make_plan(planner)
        optimized = planner.optimize_plan(plan, {"objective": "duration"})
        executor = MissionExecutor()
        result = executor.execute_plan(optimized)
        assert result.state == ExecutionState.COMPLETED
        assert optimized.estimated_duration < plan.estimated_duration

    def test_validate_plan_before_execution(self):
        planner = MissionPlanner()
        plan = _make_plan(planner)
        is_valid, issues = planner.validate_plan(plan)
        assert is_valid
        executor = MissionExecutor()
        result = executor.execute_plan(plan)
        assert result.state == ExecutionState.COMPLETED

    def test_execute_phase_directly(self):
        planner = MissionPlanner()
        plan = _make_plan(planner)
        executor = MissionExecutor()
        phase_result = executor.execute_phase(plan.phases[0])
        assert phase_result.state == ExecutionState.COMPLETED
        assert phase_result.phase_name == plan.phases[0].name

    def test_phase_transition_during_execution(self):
        planner = MissionPlanner()
        plan = _make_plan(planner)
        executor = MissionExecutor()
        tr = executor.handle_phase_transition(plan.phases[0].name, plan.phases[1].name)
        assert tr.success
        assert tr.from_phase == plan.phases[0].name

    def test_executor_hooks_fire_during_plan(self):
        planner = MissionPlanner()
        plan = _make_plan(planner)
        executor = MissionExecutor()
        log = []
        executor.register_hook("mission_start", lambda p: log.append("start"))
        executor.register_hook("mission_end", lambda s: log.append("end"))
        executor.register_hook("phase_start", lambda p: log.append("phase_start"))
        executor.register_hook("phase_end", lambda r: log.append("phase_end"))
        executor.execute_plan(plan)
        assert "start" in log
        assert "end" in log
        assert "phase_start" in log
        assert "phase_end" in log


# ===========================================================================
# 2. Planner + Monitor cross-module tests
# ===========================================================================

class TestPlannerMonitorIntegration:
    """Tests that exercise MissionPlanner together with MissionMonitor."""

    def test_register_and_monitor_plan(self):
        planner = MissionPlanner()
        plan = _make_plan(planner)
        monitor = MissionMonitor()
        status = monitor.register_mission(plan.id, plan)
        assert status.mission_id == plan.id

    def test_update_progress_from_plan_execution(self):
        planner = MissionPlanner()
        plan = _make_plan(planner)
        executor = MissionExecutor()
        result = executor.execute_plan(plan)
        monitor = MissionMonitor()
        monitor.register_mission(plan.id, plan)
        metric = ProgressMetric(name="overall", value=result.total_progress)
        status = monitor.update_progress(plan.id, metric)
        assert status.progress > 0

    def test_monitor_detects_deviations(self):
        planner = MissionPlanner()
        plan = _make_plan(planner)
        monitor = MissionMonitor()
        actual_durations = {phase.name: phase.duration * 1.5 for phase in plan.phases}
        deviations = monitor.detect_deviations(plan, actual_durations)
        assert len(deviations) > 0
        assert deviations[0].deviation_type == "overrun"

    def test_check_objectives_with_progress(self):
        planner = MissionPlanner()
        plan = _make_plan(planner, n_objectives=3)
        monitor = MissionMonitor()
        obj_statuses = monitor.check_objectives(plan, 50.0)
        assert len(obj_statuses) == 3
        for status in obj_statuses:
            assert "objective_id" in status
            assert "progress" in status

    def test_generate_status_report(self):
        planner = MissionPlanner()
        plan = _make_plan(planner)
        monitor = MissionMonitor()
        monitor.register_mission(plan.id, plan)
        metric = ProgressMetric(name="overall", value=50.0)
        monitor.update_progress(plan.id, metric)
        report = monitor.generate_status_report(plan.id)
        assert report.mission_id == plan.id
        assert report.progress > 0

    def test_resource_warnings_from_plan_resources(self):
        planner = MissionPlanner()
        plan = _make_plan(planner)
        monitor = MissionMonitor()
        monitor.set_resource_limit("energy_wh", 100.0)
        warnings = monitor.check_resource_status(
            plan.id, {"energy_wh": plan.resource_requirements.energy_wh}
        )
        assert len(warnings) > 0

    def test_planner_history_and_monitor_awareness(self):
        planner = MissionPlanner()
        plans = []
        for i in range(3):
            plans.append(_make_plan(planner))
        monitor = MissionMonitor()
        for plan in plans:
            monitor.register_mission(plan.id, plan)
        all_missions = monitor.get_all_missions()
        assert len(all_missions) == 3


# ===========================================================================
# 3. Executor + Monitor cross-module tests
# ===========================================================================

class TestExecutorMonitorIntegration:
    """Tests that exercise MissionExecutor together with MissionMonitor."""

    def test_execution_progress_fed_to_monitor(self):
        planner = MissionPlanner()
        plan = _make_plan(planner, n_objectives=3)
        executor = MissionExecutor()
        monitor = MissionMonitor()
        monitor.register_mission(plan.id, plan)
        result = executor.execute_plan(plan)
        progress = ProgressMetric(name="mission_progress", value=result.total_progress)
        monitor.update_progress(plan.id, progress)
        status = monitor.get_status(plan.id)
        assert status is not None
        assert status.progress > 0

    def test_monitor_alerts_during_execution(self):
        planner = MissionPlanner()
        plan = _make_plan(planner)
        monitor = MissionMonitor()
        monitor.register_mission(plan.id, plan)
        monitor.add_alert(plan.id, MissionAlert(level=AlertLevel.WARNING, message="Test alert"))
        alerts = monitor.get_alerts(plan.id)
        assert len(alerts) == 1
        assert alerts[0].level == AlertLevel.WARNING

    def test_executor_abort_monitored_mission(self):
        planner = MissionPlanner()
        plan = _make_plan(planner, n_objectives=5)
        executor = MissionExecutor()
        monitor = MissionMonitor()
        monitor.register_mission(plan.id, plan)
        def abort_on_phase_start(phase):
            executor.abort_mission("automatic abort")
        executor.register_hook("phase_start", abort_on_phase_start)
        executor.reset()
        result = executor.execute_plan(plan)
        assert result.aborted

    def test_compute_efficiency_after_execution(self):
        planner = MissionPlanner()
        plan = _make_plan(planner)
        executor = MissionExecutor()
        result = executor.execute_plan(plan)
        monitor = MissionMonitor()
        monitor.register_mission(plan.id, plan)
        metric = ProgressMetric(name="eff", value=100.0)
        monitor.update_progress(plan.id, metric)
        eff = monitor.compute_mission_efficiency(plan.id)
        assert 0.0 <= eff <= 1.0

    def test_phase_results_and_monitor_deviations(self):
        planner = MissionPlanner()
        plan = _make_plan(planner)
        executor = MissionExecutor()
        result = executor.execute_plan(plan)
        monitor = MissionMonitor()
        actual_durs = {}
        for pr in result.phase_results:
            actual_durs[pr.phase_name] = pr.duration + 1.0
        deviations = monitor.detect_deviations(plan, actual_durs)
        assert isinstance(deviations, list)

    def test_monitor_acknowledge_alert_after_execution(self):
        planner = MissionPlanner()
        plan = _make_plan(planner)
        executor = MissionExecutor()
        executor.execute_plan(plan)
        monitor = MissionMonitor()
        monitor.register_mission(plan.id, plan)
        monitor.add_alert(plan.id, MissionAlert(level=AlertLevel.INFO, message="info"))
        ack = monitor.acknowledge_alert(plan.id, 0)
        assert ack is True


# ===========================================================================
# 4. Contingency + Executor cross-module tests
# ===========================================================================

class TestContingencyExecutorIntegration:
    """Tests that exercise ContingencyManager together with MissionExecutor."""

    def test_contingency_trigger_during_execution(self):
        planner = MissionPlanner()
        plan = _make_plan(planner)
        executor = MissionExecutor()
        contingency = ContingencyManager()
        cplan = ContingencyPlan(
            name="energy_low",
            trigger_condition="energy > 90",
            trigger_type="threshold",
            response_actions=[ContingencyAction(name="reduce_speed", action_type="adjust")],
        )
        cid = contingency.register_contingency(cplan)
        evaluations = contingency.evaluate_triggers({"energy": 95.0})
        triggered = [e for e in evaluations if e.triggered]
        assert len(triggered) > 0

    def test_execute_contingency_after_plan_failure(self):
        planner = MissionPlanner()
        plan = _make_plan(planner)
        contingency = ContingencyManager()
        cplan = ContingencyPlan(
            name="sensor_failure",
            trigger_condition="sensor_failed",
            trigger_type="simple",
            response_actions=[
                ContingencyAction(name="switch_backup", action_type="adjust"),
                ContingencyAction(name="notify_operator", action_type="adjust"),
            ],
        )
        cid = contingency.register_contingency(cplan)
        contingency.evaluate_triggers({"sensor_failed": True})
        result = contingency.execute_contingency(cid)
        assert result.success
        assert result.actions_completed == 2

    def test_fallback_plan_generation(self):
        planner = MissionPlanner()
        plan = _make_plan(planner, n_objectives=4)
        contingency = ContingencyManager()
        fallback = contingency.generate_fallback_plan(plan, plan.phases[1].name)
        assert fallback.original_plan_id == plan.id
        assert fallback.failure_point == plan.phases[1].name
        assert len(fallback.fallback_phases) > 0

    def test_abort_criteria_evaluation(self):
        planner = MissionPlanner()
        plan = _make_plan(planner)
        contingency = ContingencyManager()
        contingency.register_abort_criteria(AbortCriteria(
            condition="engine_overheating",
            severity=AbortSeverity.CRITICAL,
            auto_abort=True,
        ))
        rec = contingency.evaluate_abort({"engine_overheating": True})
        assert rec.should_abort is True
        assert rec.auto_abort is True

    def test_contingency_recovery_time(self):
        contingency = ContingencyManager()
        cplan = ContingencyPlan(
            name="test",
            trigger_condition="x > 0",
            estimated_recovery_time=10.0,
            response_actions=[ContingencyAction(name="act", timeout=5.0)],
        )
        contingency.register_contingency(cplan)
        recovery = contingency.compute_recovery_time(cplan)
        assert recovery > 10.0

    def test_executor_and_contingency_cooperation(self):
        planner = MissionPlanner()
        plan = _make_plan(planner)
        executor = MissionExecutor()
        contingency = ContingencyManager()
        cplan = ContingencyPlan(
            name="comms_loss",
            trigger_condition="comm_status == lost",
            trigger_type="threshold",
            response_actions=[ContingencyAction(name="switch_radio", action_type="adjust")],
        )
        cid = contingency.register_contingency(cplan)
        executor.execute_plan(plan)
        evals = contingency.evaluate_triggers({"comm_status": "lost"})
        assert isinstance(evals, list)


# ===========================================================================
# 5. Transition + Override (Autonomy) cross-module tests
# ===========================================================================

class TestTransitionOverrideIntegration:
    """Tests that exercise TransitionManager together with OverrideManager."""

    def test_override_forces_manual_then_transition_back(self):
        override_mgr = OverrideManager()
        transition_mgr = TransitionManager()
        result = override_mgr.emergency_override("operator_1")
        assert result.new_level == AutonomyLevel.MANUAL
        assert override_mgr._current_level == AutonomyLevel.MANUAL
        req = TransitionRequest(
            from_level=AutonomyLevel.MANUAL,
            to_level=AutonomyLevel.ASSISTED,
            reason="recovery",
        )
        approval = transition_mgr.request_transition(req)
        assert approval["approved"] is True

    def test_transition_safety_with_override_context(self):
        override_mgr = OverrideManager()
        transition_mgr = TransitionManager()
        override_mgr.request_override(OverrideRequest(
            operator_id="op1",
            target_level=AutonomyLevel.MANUAL,
            reason="storm",
        ))
        safety = transition_mgr.compute_transition_safety(
            AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED, {"risk": 0.2}
        )
        assert 0.0 <= safety <= 1.0

    def test_available_transitions_after_override(self):
        override_mgr = OverrideManager()
        transition_mgr = TransitionManager()
        override_mgr.request_override(OverrideRequest(
            operator_id="op1",
            target_level=AutonomyLevel.ASSISTED,
            reason="test",
        ))
        available = transition_mgr.get_available_transitions(AutonomyLevel.ASSISTED)
        assert AutonomyLevel.MANUAL in available
        assert AutonomyLevel.ASSISTED not in available

    def test_override_priority_between_two_operators(self):
        override_mgr = OverrideManager()
        req_a = OverrideRequest(
            operator_id="op_a",
            target_level=AutonomyLevel.SEMI_AUTO,
            timestamp=time.time(),
        )
        req_b = OverrideRequest(
            operator_id="op_b",
            target_level=AutonomyLevel.MANUAL,
            timestamp=time.time() + 1,
        )
        winner = override_mgr.compute_override_priority(req_a, req_b)
        assert winner == "b"

    def test_override_validation_with_permissions(self):
        override_mgr = OverrideManager()
        req = OverrideRequest(
            operator_id="op1",
            target_level=AutonomyLevel.AUTONOMOUS,
            reason="test",
        )
        valid = override_mgr.validate_override(req, {"max_level": AutonomyLevel.SEMI_AUTO})
        assert valid is False

    def test_recovery_plan_after_override(self):
        override_mgr = OverrideManager()
        override_mgr.request_override(OverrideRequest(
            operator_id="op1",
            target_level=AutonomyLevel.MANUAL,
            reason="test",
        ))
        plan = override_mgr.compute_recovery_plan(AutonomyLevel.MANUAL)
        assert len(plan) > 0
        assert plan[0]["step"] == "verify_system_status"

    def test_transition_cooldown_after_override(self):
        transition_mgr = TransitionManager()
        transition_mgr.execute_transition(AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED)
        remaining = transition_mgr.check_cooldown(transition_mgr._last_transition_time)
        assert remaining > 0
        req = TransitionRequest(
            from_level=AutonomyLevel.ASSISTED,
            to_level=AutonomyLevel.MANUAL,
            urgency="critical",
        )
        approval = transition_mgr.request_transition(req)
        assert approval["approved"] is True

    def test_override_history_tracks_all_changes(self):
        override_mgr = OverrideManager()
        override_mgr.request_override(OverrideRequest(
            operator_id="op1",
            target_level=AutonomyLevel.ASSISTED,
        ))
        override_mgr.emergency_override("op2")
        history = override_mgr.get_active_overrides()
        assert isinstance(history, list)

    def test_acknowledge_override(self):
        override_mgr = OverrideManager()
        result = override_mgr.request_override(OverrideRequest(
            operator_id="op1",
            target_level=AutonomyLevel.SEMI_AUTO,
        ))
        active = override_mgr.get_active_overrides()
        if active:
            ack = override_mgr.acknowledge_override(active[0]["override_id"])
            assert ack is True


# ===========================================================================
# 6. Planner + Autonomy (Transition + Override) cross-module tests
# ===========================================================================

class TestPlannerAutonomyIntegration:
    """Tests that exercise MissionPlanner with autonomy modules."""

    def test_plan_risk_influences_transition_safety(self):
        planner = MissionPlanner()
        objectives = [MissionObjective(name="risky", type="risky", priority=1)]
        plan = planner.create_plan(objectives)
        transition_mgr = TransitionManager()
        risk_score = plan.risk_assessment.overall_score
        safety = transition_mgr.compute_transition_safety(
            AutonomyLevel.MANUAL, AutonomyLevel.SEMI_AUTO, {"risk": risk_score}
        )
        assert 0.0 <= safety <= 1.0

    def test_override_during_high_risk_plan(self):
        planner = MissionPlanner()
        objectives = [MissionObjective(name="dangerous", type="navigate", priority=1)]
        plan = planner.create_plan(objectives)
        override_mgr = OverrideManager()
        result = override_mgr.emergency_override("operator")
        assert result.new_level == AutonomyLevel.MANUAL
        assert result.accepted is True

    def test_generate_alternatives_and_autonomy_consideration(self):
        planner = MissionPlanner()
        plan = _make_plan(planner, n_objectives=3)
        alternatives = planner.generate_alternatives(plan)
        assert len(alternatives) >= 2
        transition_mgr = TransitionManager()
        for alt in alternatives:
            risk = alt.risk_assessment.overall_score
            safety = transition_mgr.compute_transition_safety(
                AutonomyLevel.ASSISTED, AutonomyLevel.SEMI_AUTO, {"risk": risk}
            )
            assert isinstance(safety, float)

    def test_phase_decomposition_and_autonomy_level(self):
        planner = MissionPlanner()
        objectives = [MissionObjective(name="complex", type="navigate")]
        plan = planner.create_plan(objectives)
        if plan.phases:
            phase = plan.phases[0]
            phase.actions = [
                MissionAction(name="scan", action_type="observe"),
                MissionAction(name="navigate", action_type="navigate"),
            ]
            sub_phases = planner.decompose_phase(phase)
            assert len(sub_phases) == 2


# ===========================================================================
# 7. Monitor + Contingency cross-module tests
# ===========================================================================

class TestMonitorContingencyIntegration:
    """Tests that exercise MissionMonitor together with ContingencyManager."""

    def test_monitor_resource_alert_triggers_contingency(self):
        planner = MissionPlanner()
        plan = _make_plan(planner)
        monitor = MissionMonitor()
        contingency = ContingencyManager()
        monitor.register_mission(plan.id, plan)
        contingency.register_contingency(ContingencyPlan(
            name="low_energy",
            trigger_condition="energy > 80",
            trigger_type="threshold",
            response_actions=[ContingencyAction(name="conserve", action_type="adjust")],
        ))
        warnings = monitor.check_resource_status(plan.id, {"energy": 90.0})
        if warnings:
            contingency.evaluate_triggers({"energy": 90.0})

    def test_monitor_deviations_trigger_contingency(self):
        planner = MissionPlanner()
        plan = _make_plan(planner)
        monitor = MissionMonitor()
        contingency = ContingencyManager()
        actual_durations = {phase.name: phase.duration * 2.0 for phase in plan.phases}
        deviations = monitor.detect_deviations(plan, actual_durations)
        if deviations:
            contingency.register_contingency(ContingencyPlan(
                name="schedule_slip",
                trigger_condition="deviation_count > 0",
                trigger_type="simple",
                response_actions=[ContingencyAction(name="replan", action_type="adjust")],
            ))

    def test_contingency_fallback_plan_validated_by_monitor(self):
        planner = MissionPlanner()
        plan = _make_plan(planner, n_objectives=4)
        contingency = ContingencyManager()
        monitor = MissionMonitor()
        fallback = contingency.generate_fallback_plan(plan, plan.phases[2].name)
        monitor.register_mission(plan.id, plan)
        assert fallback.risk_increase > 0

    def test_abort_recommendation_and_monitor_alerts(self):
        planner = MissionPlanner()
        plan = _make_plan(planner)
        monitor = MissionMonitor()
        contingency = ContingencyManager()
        monitor.register_mission(plan.id, plan)
        contingency.register_abort_criteria(AbortCriteria(
            condition="propulsion_failure",
            severity=AbortSeverity.HIGH,
            check_fn=lambda s: s.get("propulsion_failure", False),
        ))
        rec = contingency.evaluate_abort({"propulsion_failure": True})
        assert rec.should_abort is True
        monitor.add_alert(plan.id, MissionAlert(
            level=AlertLevel.CRITICAL, message=f"Abort recommended: {rec.reason}"
        ))


# ===========================================================================
# 8. Full lifecycle integration tests (3+ modules)
# ===========================================================================

class TestFullLifecycleIntegration:
    """End-to-end tests exercising planner + executor + monitor + contingency + autonomy."""

    def test_full_plan_execute_monitor_cycle(self):
        planner = MissionPlanner()
        plan = _make_plan(planner, n_objectives=3)
        is_valid, issues = planner.validate_plan(plan)
        assert is_valid
        executor = MissionExecutor()
        result = executor.execute_plan(plan)
        assert result.state == ExecutionState.COMPLETED
        monitor = MissionMonitor()
        monitor.register_mission(plan.id, plan)
        metric = ProgressMetric(name="progress", value=100.0)
        monitor.update_progress(plan.id, metric)
        report = monitor.generate_status_report(plan.id)
        assert report.progress == 100.0

    def test_plan_with_contingency_and_monitor(self):
        planner = MissionPlanner()
        plan = _make_plan(planner, n_objectives=2)
        contingency = ContingencyManager()
        contingency.register_contingency(ContingencyPlan(
            name="sensor_fail",
            trigger_condition="sensor_ok == false",
            trigger_type="threshold",
            response_actions=[ContingencyAction(name="reset_sensor", action_type="adjust")],
        ))
        contingency.register_abort_criteria(AbortCriteria(
            condition="critical_failure",
            severity=AbortSeverity.CRITICAL,
        ))
        monitor = MissionMonitor()
        monitor.register_mission(plan.id, plan)
        monitor.add_alert(plan.id, MissionAlert(level=AlertLevel.INFO, message="Mission started"))
        executor = MissionExecutor()
        result = executor.execute_plan(plan)
        assert result.state == ExecutionState.COMPLETED
        alerts = monitor.get_alerts(plan.id)
        assert len(alerts) >= 1

    def test_lifecycle_with_autonomy_override(self):
        planner = MissionPlanner()
        plan = _make_plan(planner)
        override_mgr = OverrideManager()
        transition_mgr = TransitionManager()
        executor = MissionExecutor()
        monitor = MissionMonitor()
        monitor.register_mission(plan.id, plan)
        override_mgr.emergency_override("operator_1")
        safety = transition_mgr.compute_transition_safety(
            AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED, {"risk": 0.1}
        )
        assert safety > 0
        result = executor.execute_plan(plan)
        assert result.state == ExecutionState.COMPLETED

    def test_optimize_then_execute_with_monitoring(self):
        planner = MissionPlanner()
        plan = _make_plan(planner, n_objectives=3)
        optimized = planner.optimize_plan(plan, {"objective": "balanced"})
        executor = MissionExecutor()
        monitor = MissionMonitor()
        monitor.register_mission(optimized.id, optimized)
        result = executor.execute_plan(optimized)
        assert result.state == ExecutionState.COMPLETED
        metric = ProgressMetric(name="done", value=result.total_progress)
        monitor.update_progress(optimized.id, metric)
        eff = monitor.compute_mission_efficiency(optimized.id)
        assert eff >= 0

    def test_deviations_trigger_contingency_fallback(self):
        planner = MissionPlanner()
        plan = _make_plan(planner, n_objectives=4)
        monitor = MissionMonitor()
        contingency = ContingencyManager()
        actual_durs = {p.name: p.duration * 3.0 for p in plan.phases}
        deviations = monitor.detect_deviations(plan, actual_durs)
        assert len(deviations) > 0
        fallback = contingency.generate_fallback_plan(plan, plan.phases[0].name)
        assert len(fallback.fallback_phases) > 0

    def test_multi_plan_comparison_with_monitoring(self):
        planner = MissionPlanner()
        base = _make_plan(planner)
        alt_duration = planner.optimize_plan(base, {"objective": "duration"})
        alt_resources = planner.optimize_plan(base, {"objective": "resources"})
        alt_risk = planner.optimize_plan(base, {"objective": "risk"})
        monitor = MissionMonitor()
        for p in [base, alt_duration, alt_resources, alt_risk]:
            monitor.register_mission(p.id, p)
        assert len(monitor.get_all_missions()) == 4

    def test_transition_history_after_multiple_changes(self):
        transition_mgr = TransitionManager()
        transition_mgr.execute_transition(AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED)
        transition_mgr.execute_transition(AutonomyLevel.ASSISTED, AutonomyLevel.SEMI_AUTO)
        history = transition_mgr.get_transition_history()
        assert len(history) == 2
        assert history[0]["to_level"] == AutonomyLevel.ASSISTED
        assert history[1]["to_level"] == AutonomyLevel.SEMI_AUTO

    def test_executor_state_transitions(self):
        planner = MissionPlanner()
        plan = _make_plan(planner)
        executor = MissionExecutor()
        assert executor.get_state() == ExecutionState.IDLE
        executor.execute_plan(plan)
        assert executor.get_state() == ExecutionState.COMPLETED
        executor.reset()
        assert executor.get_state() == ExecutionState.IDLE

    def test_estimate_completion_with_monitor(self):
        planner = MissionPlanner()
        plan = _make_plan(planner)
        monitor = MissionMonitor()
        monitor.register_mission(plan.id, plan)
        metric = ProgressMetric(name="p", value=50.0)
        monitor.update_progress(plan.id, metric)
        eta = monitor.estimate_completion(plan.id)
        assert eta is not None
