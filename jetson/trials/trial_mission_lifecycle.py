"""Trial 3: Mission Lifecycle — Mission planning + Adaptive autonomy + Contingency.

Tests cross-module integration between mission planning, execution, monitoring,
contingency management, adaptive autonomy transitions, and human overrides.
"""

from jetson.mission.planner import (
    MissionPlanner, MissionPlan, MissionObjective, MissionPhase,
    MissionAction, ResourceRequirements, RiskAssessment, RiskLevel,
)
from jetson.mission.execution import (
    MissionExecutor, MissionResult, PhaseResult, ExecutionState,
    TransitionResult,
)
from jetson.mission.monitoring import (
    MissionMonitor, ProgressMetric, MissionAlert, MissionStatus,
    DeviationReport, StatusReport, AlertLevel, TrendDirection,
    ResourceWarning,
)
from jetson.mission.contingency import (
    ContingencyManager, ContingencyPlan, ContingencyAction, ContingencyPriority,
    ContingencyStatus, AbortCriteria, AbortSeverity, TriggerEvaluation,
    ContingencyResult, AbortRecommendation, FallbackPlan,
)
from jetson.adaptive_autonomy.transition import (
    TransitionManager, TransitionRequest, TransitionPolicy,
)
from jetson.adaptive_autonomy.override import (
    OverrideManager, OverrideRequest, OverrideResult,
)
from jetson.adaptive_autonomy.levels import (
    AutonomyLevel, AutonomyLevelManager,
)


def run_trial():
    """Run all mission lifecycle integration tests. Returns True if all pass."""
    passed = 0
    failed = 0
    total = 0

    def check(name, condition):
        nonlocal passed, failed, total
        total += 1
        if condition:
            passed += 1
        else:
            failed += 1

    # === Mission Planner ===

    mp = MissionPlanner()

    # 1. Create plan with objectives
    obj1 = MissionObjective(name="survey", type="survey", priority=3)
    obj2 = MissionObjective(name="patrol", type="patrol", priority=4)
    plan = mp.create_plan([obj1, obj2])
    check("plan_created", plan.id is not None and len(plan.id) > 0)
    check("plan_objectives", len(plan.objectives) == 2)
    check("plan_phases", len(plan.phases) == 2)

    # 2. Plan duration estimated
    check("duration_positive", plan.estimated_duration > 0)

    # 3. Resource requirements
    rr = plan.resource_requirements
    check("energy_positive", rr.energy_wh > 0)
    check("compute_positive", rr.compute_percent >= 0)

    # 4. Risk assessment
    ra = plan.risk_assessment
    check("risk_score_range", 0 <= ra.overall_score <= 1.0)
    check("risk_level_valid", ra.risk_level in RiskLevel)

    # 5. Optimize plan
    opt_plan = mp.optimize_plan(plan, {"objective": "duration"})
    check("optimized_plan", opt_plan.id != plan.id)
    check("optimized_duration_shorter", opt_plan.estimated_duration <= plan.estimated_duration * 1.1)

    # 6. Generate alternatives
    alts = mp.generate_alternatives(plan)
    check("alternatives_count", len(alts) >= 2)
    alt_names = [a.name for a in alts]
    check("unique_alternatives", len(alt_names) == len(set(alt_names)))

    # 7. Validate plan
    valid, issues = mp.validate_plan(plan)
    check("valid_plan", valid is True)
    check("no_issues", len(issues) == 0)

    # 8. Validate empty plan
    empty_plan = MissionPlan()
    valid2, issues2 = mp.validate_plan(empty_plan)
    check("empty_plan_invalid", valid2 is False)
    check("empty_plan_issues", len(issues2) > 0)

    # 9. Decompose phase
    phase = MissionPhase(
        name="complex",
        actions=[
            MissionAction(name="step1", action_type="navigate", duration=10),
            MissionAction(name="step2", action_type="scan", duration=20),
        ],
        success_criteria=["Complete"],
    )
    sub_phases = mp.decompose_phase(phase)
    check("sub_phase_count", len(sub_phases) == 2)

    # 10. Plan history
    history = mp.get_plan_history()
    check("history_count", len(history) >= 2)

    # === Mission Executor ===

    me = MissionExecutor()

    # 11. Execute plan
    result = me.execute_plan(plan)
    check("execution_completed", result.state == ExecutionState.COMPLETED)
    check("execution_phases", len(result.phase_results) == 2)
    check("execution_progress", result.total_progress > 0)

    # 12. Phase results
    for pr in result.phase_results:
        check("phase_result_name", pr.phase_name != "")
        check("phase_result_state", pr.state == ExecutionState.COMPLETED)

    # 13. Completed phases list
    check("completed_phases", len(result.completed_phases) == 2)

    # 14. Execution with actions
    plan_actions = MissionPlan(
        name="action_plan",
        phases=[MissionPhase(
            name="action_phase",
            actions=[MissionAction(name="move", action_type="navigate", duration=5)],
            success_criteria=["moved"],
        )],
    )
    result2 = me.execute_plan(plan_actions)
    check("action_plan_completed", result2.state == ExecutionState.COMPLETED)

    # 15. Pause and resume
    me2 = MissionExecutor()
    plan_long = MissionPlan(phases=[
        MissionPhase(name="p1", actions=[MissionAction(name="a", duration=1)] * 3,
                     success_criteria=["done"]),
        MissionPhase(name="p2", actions=[MissionAction(name="b", duration=1)] * 3,
                     success_criteria=["done"]),
    ])
    # Register hook to pause after first phase
    pause_after_first = {"paused": False}
    def on_phase_end(data):
        if isinstance(data, PhaseResult) and not pause_after_first["paused"]:
            pause_after_first["paused"] = True
            me2.pause_mission()
    me2.register_hook("phase_end", on_phase_end)
    result3 = me2.execute_plan(plan_long)
    check("paused_state", pause_after_first["paused"])

    # 16. Abort
    me3 = MissionExecutor()
    aborted_plan = MissionPlan(phases=[
        MissionPhase(name="abort_test", actions=[
            MissionAction(name="long_action", duration=10),
        ] * 100, success_criteria=["done"]),
    ])
    abort_called = {"done": False}
    def on_start(data):
        if not abort_called["done"]:
            abort_called["done"] = True
            me3.abort_mission("test abort")
    me3.register_hook("mission_start", on_start)
    result4 = me3.execute_plan(aborted_plan)
    check("aborted", result4.state == ExecutionState.ABORTED)
    check("abort_reason", result4.abort_reason == "test abort")

    # 17. Reset executor
    me3.reset()
    check("reset_idle", me3.get_state() == ExecutionState.IDLE)

    # 18. Phase transition
    tr = me2.handle_phase_transition("p1", "p2")
    check("transition_success", tr.success is True)

    # === Mission Monitor ===

    mm = MissionMonitor()

    # 19. Register mission
    status = mm.register_mission("m1", plan)
    check("mission_registered", status.mission_id == "m1")

    # 20. Update progress
    metric = ProgressMetric(name="coverage", value=50.0, target=100.0)
    status2 = mm.update_progress("m1", metric)
    check("progress_updated", status2.progress > 0)

    # 21. Check objectives
    obj_statuses = mm.check_objectives(plan, 50.0)
    check("obj_statuses_count", len(obj_statuses) == 2)

    # 22. Detect deviations
    actual_durs = {p.name: p.duration * 1.5 for p in plan.phases}
    devs = mm.detect_deviations(plan, actual_durs)
    check("deviations_detected", len(devs) > 0)
    check("deviation_type", devs[0].deviation_type == "overrun")

    # 23. Generate status report
    report = mm.generate_status_report("m1")
    check("report_type", isinstance(report, StatusReport))
    check("report_has_recommendations", len(report.recommendations) > 0)

    # 24. Resource limits and warnings
    mm.set_resource_limit("battery", 100.0)
    warnings = mm.check_resource_status("m1", {"battery": 95.0})
    check("resource_warnings_list", isinstance(warnings, list))

    # 25. Mission efficiency
    eff = mm.compute_mission_efficiency("m1")
    check("efficiency_range", 0 <= eff <= 1.0)

    # 26. Alerts
    mm.add_alert("m1", MissionAlert(level=AlertLevel.WARNING, message="test", source="trial"))
    alerts = mm.get_alerts("m1")
    check("alerts_count", len(alerts) >= 1)
    mm.acknowledge_alert("m1", 0)
    check("alert_acknowledged", mm.get_alerts("m1")[0].acknowledged)

    # 27. Metric history
    history = mm.get_metric_history("m1", "coverage")
    check("metric_history", len(history) >= 1)

    # 28. Unregister mission
    ok = mm.unregister_mission("m1")
    check("unregister", ok is True)

    # === Contingency Manager ===

    cm = ContingencyManager()

    # 29. Register contingency
    cp = ContingencyPlan(
        name="engine_failure",
        trigger_condition="engine_temp > 90",
        trigger_type="threshold",
        response_actions=[ContingencyAction(name="reduce_speed")],
        priority=ContingencyPriority.HIGH,
    )
    cid = cm.register_contingency(cp)
    check("contingency_registered", cid == cp.id)

    # 30. Evaluate triggers
    evals = cm.evaluate_triggers({"engine_temp": 95.0})
    check("trigger_evaluated", len(evals) >= 1)
    check("trigger_fired", evals[0].triggered is True)

    # 31. Execute contingency
    cres = cm.execute_contingency(cid)
    check("contingency_executed", cres.success is True)
    check("contingency_actions_completed", cres.actions_completed == 1)

    # 32. Contingency with false trigger
    evals2 = cm.evaluate_triggers({"engine_temp": 50.0})
    check("no_trigger_normal_temp", evals2[0].triggered is False)

    # 33. Abort criteria
    ac = AbortCriteria(
        condition="fuel < 5",
        severity=AbortSeverity.CRITICAL,
        auto_abort=True,
    )
    cm.register_abort_criteria(ac)
    rec = cm.evaluate_abort({"fuel": 3.0})
    check("abort_recommended", rec.should_abort is True)
    check("abort_severity", rec.severity == AbortSeverity.CRITICAL)

    # 34. No abort
    rec2 = cm.evaluate_abort({"fuel": 50.0})
    check("no_abort", rec2.should_abort is False)

    # 35. Generate fallback plan
    fb = cm.generate_fallback_plan(plan, "phase_survey_0")
    check("fallback_generated", fb.original_plan_id == plan.id)
    check("fallback_phases", len(fb.fallback_phases) >= 1)

    # 36. Recovery time
    cp2 = ContingencyPlan(
        name="comms_loss",
        response_actions=[ContingencyAction(name="reconnect", timeout=10)],
        estimated_recovery_time=30.0,
    )
    rt = cm.compute_recovery_time(cp2)
    check("recovery_time_positive", rt >= 30.0)

    # 37. Active contingencies
    active = cm.get_active_contingencies()
    check("active_contingencies_list", isinstance(active, list))

    # 38. Trigger history
    thist = cm.get_trigger_history()
    check("trigger_history_list", isinstance(thist, list))

    # 39. Reset contingency
    ok = cm.reset_contingency(cid)
    check("reset_contingency", ok is True)

    # === Adaptive Autonomy ===

    tm = TransitionManager()

    # 40. Transition request
    req = TransitionRequest(
        from_level=AutonomyLevel.MANUAL,
        to_level=AutonomyLevel.ASSISTED,
        reason="Operator request",
    )
    result = tm.request_transition(req)
    check("transition_approved", result["approved"] is True)

    # 41. Execute transition
    new_level = tm.execute_transition(AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED)
    check("transition_executed", new_level == AutonomyLevel.ASSISTED)
    check("current_level_updated", tm.current_level == AutonomyLevel.ASSISTED)

    # 42. Available transitions
    avail = tm.get_available_transitions(AutonomyLevel.ASSISTED)
    check("available_transitions", len(avail) >= 1)

    # 43. Transition safety
    safety = tm.compute_transition_safety(
        AutonomyLevel.ASSISTED, AutonomyLevel.SEMI_AUTO, {"risk": 0.2},
    )
    check("safety_in_range", 0 <= safety <= 1.0)

    # 44. Transition history
    hist = tm.get_transition_history()
    check("transition_history", len(hist) >= 1)

    # === Override Manager ===

    om = OverrideManager()

    # 45. Request override
    oreq = OverrideRequest(operator_id="captain", target_level=AutonomyLevel.MANUAL)
    oresult = om.request_override(oreq)
    check("override_accepted", oresult.accepted is True)
    check("override_level", oresult.new_level == AutonomyLevel.MANUAL)

    # 46. Emergency override
    eresult = om.emergency_override("safety_officer")
    check("emergency_accepted", eresult.accepted is True)
    check("emergency_manual", eresult.new_level == AutonomyLevel.MANUAL)
    check("emergency_no_ack_needed", eresult.acknowledgment_required is False)

    # 47. Validate override
    valid = om.validate_override(oreq, {"max_level": AutonomyLevel.AUTONOMOUS})
    check("override_valid", valid is True)
    invalid = om.validate_override(
        OverrideRequest(operator_id="cadet", target_level=AutonomyLevel.AUTONOMOUS),
        {"max_level": AutonomyLevel.SEMI_AUTO},
    )
    check("override_invalid", invalid is False)

    # 48. Override priority
    prio_a = om.compute_override_priority(
        OverrideRequest(operator_id="a", target_level=AutonomyLevel.ASSISTED),
        OverrideRequest(operator_id="b", target_level=AutonomyLevel.FULL_AUTO),
    )
    check("override_priority_a", prio_a == "a")  # lower = more cautious = higher priority

    # 49. Active overrides
    active = om.get_active_overrides()
    check("active_overrides_list", isinstance(active, list))

    # 50. Recovery plan
    recovery = om.compute_recovery_plan(AutonomyLevel.MANUAL)
    check("recovery_plan_steps", len(recovery) >= 3)
    step_names = [s["step"] for s in recovery]
    check("recovery_has_verify", "verify_system_status" in step_names)

    return failed == 0
