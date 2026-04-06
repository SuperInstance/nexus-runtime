"""Tests for mission planner module."""

import time
import pytest
from jetson.mission.planner import (
    RiskLevel,
    MissionObjective,
    MissionAction,
    MissionPhase,
    ResourceRequirements,
    RiskAssessment,
    MissionPlan,
    MissionPlanner,
)


class TestRiskLevel:
    def test_low_value(self):
        assert RiskLevel.LOW.value == "low"

    def test_medium_value(self):
        assert RiskLevel.MEDIUM.value == "medium"

    def test_high_value(self):
        assert RiskLevel.HIGH.value == "high"

    def test_critical_value(self):
        assert RiskLevel.CRITICAL.value == "critical"

    def test_all_values(self):
        values = {r.value for r in RiskLevel}
        assert values == {"low", "medium", "high", "critical"}


class TestMissionObjective:
    def test_default_creation(self):
        obj = MissionObjective()
        assert obj.type == "general"
        assert obj.priority == 3
        assert obj.constraints == {}

    def test_custom_creation(self):
        obj = MissionObjective(
            name="survey", type="survey", target="area_a",
            priority=1, deadline=3600.0
        )
        assert obj.name == "survey"
        assert obj.type == "survey"
        assert obj.target == "area_a"
        assert obj.priority == 1
        assert obj.deadline == 3600.0

    def test_auto_id(self):
        obj1 = MissionObjective()
        obj2 = MissionObjective()
        assert obj1.id != obj2.id

    def test_weight_default(self):
        obj = MissionObjective()
        assert obj.weight == 1.0


class TestMissionAction:
    def test_default_creation(self):
        action = MissionAction()
        assert action.name == ""
        assert action.action_type == "navigate"
        assert action.duration == 0.0

    def test_custom_creation(self):
        action = MissionAction(
            name="dive", action_type="maneuver",
            parameters={"depth": 10}, duration=30.0,
            resource_cost={"energy": 5.0},
        )
        assert action.name == "dive"
        assert action.parameters["depth"] == 10
        assert action.resource_cost["energy"] == 5.0


class TestMissionPhase:
    def test_default_creation(self):
        phase = MissionPhase()
        assert phase.name == ""
        assert phase.actions == []
        assert phase.dependencies == []
        assert phase.risk_level == RiskLevel.MEDIUM

    def test_with_actions(self):
        actions = [MissionAction(name="a1", duration=10.0)]
        phase = MissionPhase(
            name="deploy", actions=actions,
            duration=10.0, success_criteria=["sensor_active"],
        )
        assert len(phase.actions) == 1
        assert phase.duration == 10.0
        assert len(phase.success_criteria) == 1


class TestResourceRequirements:
    def test_default_values(self):
        rr = ResourceRequirements()
        assert rr.energy_wh == 0.0
        assert rr.compute_percent == 0.0
        assert rr.actuators == []

    def test_custom_values(self):
        rr = ResourceRequirements(
            energy_wh=500.0, compute_percent=75.0,
            actuators=["thruster"],
        )
        assert rr.energy_wh == 500.0
        assert "thruster" in rr.actuators


class TestRiskAssessment:
    def test_default_values(self):
        ra = RiskAssessment()
        assert ra.overall_score == 0.0
        assert ra.risk_level == RiskLevel.MEDIUM
        assert ra.factors == []

    def test_with_factors(self):
        ra = RiskAssessment(
            overall_score=0.8,
            risk_level=RiskLevel.HIGH,
            factors=[{"factor": "weather"}],
            mitigation=["wait"],
        )
        assert ra.overall_score == 0.8
        assert len(ra.mitigation) == 1


class TestMissionPlan:
    def test_default_creation(self):
        plan = MissionPlan()
        assert plan.objectives == []
        assert plan.phases == []
        assert plan.estimated_duration == 0.0

    def test_auto_id(self):
        p1 = MissionPlan()
        p2 = MissionPlan()
        assert p1.id != p2.id

    def test_created_at(self):
        before = time.time()
        plan = MissionPlan()
        after = time.time()
        assert before <= plan.created_at <= after


class TestMissionPlanner:
    def setup_method(self):
        self.planner = MissionPlanner()

    def test_create_plan_basic(self):
        obj = MissionObjective(name="survey", type="survey")
        plan = self.planner.create_plan([obj])
        assert isinstance(plan, MissionPlan)
        assert len(plan.objectives) == 1
        assert len(plan.phases) == 1

    def test_create_plan_with_constraints(self):
        obj = MissionObjective(name="nav", type="navigation")
        plan = self.planner.create_plan([obj], {"phase_duration": 120.0})
        assert plan.constraints == {"phase_duration": 120.0}
        assert plan.phases[0].duration == 120.0

    def test_create_plan_multiple_objectives(self):
        objs = [
            MissionObjective(name=f"obj_{i}", type="survey")
            for i in range(3)
        ]
        plan = self.planner.create_plan(objs)
        assert len(plan.phases) == 3

    def test_create_plan_phase_dependencies(self):
        objs = [
            MissionObjective(name="a", type="type_a"),
            MissionObjective(name="b", type="type_b"),
        ]
        plan = self.planner.create_plan(objs)
        assert len(plan.phases[1].dependencies) == 1

    def test_create_plan_empty_objectives(self):
        plan = self.planner.create_plan([])
        assert len(plan.phases) == 0

    def test_estimate_duration_single_phase(self):
        phase = MissionPhase(name="p1", duration=60.0)
        dur = self.planner.estimate_duration([phase])
        assert dur == 60.0

    def test_estimate_duration_multiple_phases(self):
        phases = [
            MissionPhase(name="p1", duration=60.0),
            MissionPhase(name="p2", duration=30.0),
        ]
        dur = self.planner.estimate_duration(phases)
        assert dur == 45.0  # average

    def test_estimate_duration_with_dependencies(self):
        phases = [
            MissionPhase(name="p1", duration=60.0),
            MissionPhase(name="p2", duration=30.0, dependencies=["p1"]),
        ]
        dur = self.planner.estimate_duration(phases)
        assert dur > 30.0

    def test_estimate_duration_empty(self):
        dur = self.planner.estimate_duration([])
        assert dur == 0.0

    def test_estimate_resources_basic(self):
        phase = MissionPhase(
            name="p1", duration=60.0,
            resource_requirements={
                "energy_wh": 100.0,
                "compute_percent": 50.0,
                "storage_mb": 20.0,
                "bandwidth_kbps": 100.0,
                "actuators": ["thruster"],
                "sensors": ["gps"],
            },
        )
        rr = self.planner.estimate_resources([phase])
        assert rr.energy_wh == 100.0
        assert rr.compute_percent == 50.0
        assert "thruster" in rr.actuators
        assert "gps" in rr.sensors

    def test_estimate_resources_multiple_phases(self):
        phases = [
            MissionPhase(name="p1", resource_requirements={
                "energy_wh": 50.0, "compute_percent": 30.0,
            }),
            MissionPhase(name="p2", resource_requirements={
                "energy_wh": 70.0, "compute_percent": 80.0,
            }),
        ]
        rr = self.planner.estimate_resources(phases)
        assert rr.energy_wh == 120.0
        assert rr.compute_percent == 80.0  # max

    def test_estimate_resources_empty(self):
        rr = self.planner.estimate_resources([])
        assert rr.energy_wh == 0.0
        assert rr.actuators == []

    def test_compute_risk_assessment_low_risk(self):
        plan = MissionPlan(
            phases=[MissionPhase(name="p1", duration=30.0)],
            estimated_duration=30.0,
            resource_requirements=ResourceRequirements(),
        )
        risk = self.planner.compute_risk_assessment(plan)
        assert risk.overall_score < 0.25
        assert risk.risk_level == RiskLevel.LOW

    def test_compute_risk_assessment_high_risk(self):
        plan = MissionPlan(
            phases=[
                MissionPhase(name="p1", risk_level=RiskLevel.CRITICAL),
                MissionPhase(name="p2", risk_level=RiskLevel.HIGH),
            ],
            estimated_duration=7200.0,
            objectives=[
                MissionObjective() for _ in range(6)
            ],
            resource_requirements=ResourceRequirements(energy_wh=2000.0),
        )
        risk = self.planner.compute_risk_assessment(plan)
        assert risk.overall_score > 0.5

    def test_compute_risk_deadline_miss(self):
        plan = MissionPlan(
            phases=[MissionPhase(name="p1", duration=100.0)],
            estimated_duration=5000.0,
            objectives=[MissionObjective(deadline=100.0)],
        )
        risk = self.planner.compute_risk_assessment(plan)
        assert any(f["factor"] == "deadline_miss_risk" for f in risk.factors)

    def test_generate_alternatives(self):
        obj = MissionObjective(name="survey", type="survey")
        plan = self.planner.create_plan([obj])
        alts = self.planner.generate_alternatives(plan)
        assert len(alts) >= 3

    def test_generate_alternatives_with_constraint(self):
        obj = MissionObjective(name="nav", type="navigation")
        plan = self.planner.create_plan([obj])
        alts = self.planner.generate_alternatives(plan, {"max_duration": 10000.0})
        for alt in alts:
            assert alt.estimated_duration <= 10000.0

    def test_optimize_plan_duration(self):
        obj = MissionObjective(name="survey", type="survey")
        plan = self.planner.create_plan([obj])
        optimized = self.planner.optimize_plan(plan, {"objective": "duration"})
        assert optimized.estimated_duration < plan.estimated_duration

    def test_optimize_plan_resources(self):
        obj = MissionObjective(name="survey", type="survey")
        plan = self.planner.create_plan([obj])
        optimized = self.planner.optimize_plan(plan, {"objective": "resources"})
        assert optimized.resource_requirements.energy_wh < plan.resource_requirements.energy_wh

    def test_optimize_plan_risk(self):
        obj = MissionObjective(name="survey", type="survey")
        plan = self.planner.create_plan([obj])
        plan.phases[0].risk_level = RiskLevel.CRITICAL
        optimized = self.planner.optimize_plan(plan, {"objective": "risk"})
        assert optimized.phases[0].risk_level == RiskLevel.MEDIUM

    def test_optimize_plan_balanced(self):
        obj = MissionObjective(name="survey", type="survey")
        plan = self.planner.create_plan([obj])
        optimized = self.planner.optimize_plan(plan, {"objective": "balanced"})
        assert optimized.estimated_duration < plan.estimated_duration

    def test_decompose_phase_with_actions(self):
        actions = [
            MissionAction(name="start", duration=5.0),
            MissionAction(name="move", duration=10.0),
        ]
        phase = MissionPhase(name="op", actions=actions, duration=15.0)
        sub_phases = self.planner.decompose_phase(phase)
        assert len(sub_phases) == 2
        assert sub_phases[0].name.startswith("op_sub0")
        assert sub_phases[1].dependencies == [sub_phases[0].name]

    def test_decompose_phase_empty(self):
        phase = MissionPhase(name="empty")
        sub_phases = self.planner.decompose_phase(phase)
        assert len(sub_phases) == 1
        assert "empty" in sub_phases[0].name

    def test_validate_plan_valid(self):
        obj = MissionObjective(name="s", type="survey")
        plan = self.planner.create_plan([obj])
        plan.phases[0].success_criteria = ["done"]
        valid, issues = self.planner.validate_plan(plan)
        assert valid is True
        assert issues == []

    def test_validate_plan_no_objectives(self):
        plan = MissionPlan(phases=[MissionPhase(name="p1")])
        valid, issues = self.planner.validate_plan(plan)
        assert valid is False
        assert any("no objectives" in i.lower() for i in issues)

    def test_validate_plan_no_phases(self):
        plan = MissionPlan(objectives=[MissionObjective()])
        valid, issues = self.planner.validate_plan(plan)
        assert valid is False
        assert any("no phases" in i.lower() for i in issues)

    def test_validate_plan_unknown_dependency(self):
        plan = MissionPlan(
            objectives=[MissionObjective()],
            phases=[MissionPhase(name="p1", dependencies=["unknown"])],
        )
        valid, issues = self.planner.validate_plan(plan)
        assert any("unknown" in i for i in issues)

    def test_validate_plan_no_success_criteria(self):
        plan = MissionPlan(
            objectives=[MissionObjective()],
            phases=[MissionPhase(name="p1", success_criteria=[])],
        )
        valid, issues = self.planner.validate_plan(plan)
        assert any("success criteria" in i for i in issues)

    def test_validate_plan_duplicate_names(self):
        plan = MissionPlan(
            objectives=[MissionObjective()],
            phases=[
                MissionPhase(name="dup"),
                MissionPhase(name="dup"),
            ],
        )
        valid, issues = self.planner.validate_plan(plan)
        assert any("duplicate" in i.lower() for i in issues)

    def test_validate_plan_circular_deps(self):
        plan = MissionPlan(
            objectives=[MissionObjective()],
            phases=[
                MissionPhase(name="a", dependencies=["b"]),
                MissionPhase(name="b", dependencies=["a"]),
            ],
        )
        valid, issues = self.planner.validate_plan(plan)
        assert any("ircular" in i for i in issues)

    def test_get_plan_history(self):
        obj = MissionObjective(name="s", type="survey")
        self.planner.create_plan([obj])
        self.planner.create_plan([obj])
        history = self.planner.get_plan_history()
        assert len(history) == 2

    def test_clear_history(self):
        obj = MissionObjective(name="s", type="survey")
        self.planner.create_plan([obj])
        self.planner.clear_history()
        assert len(self.planner.get_plan_history()) == 0

    def test_risk_mitigations_long_duration(self):
        plan = MissionPlan(
            phases=[MissionPhase(name="p1")],
            estimated_duration=5000.0,
        )
        risk = self.planner.compute_risk_assessment(plan)
        assert any("rest" in m.lower() for m in risk.mitigation)

    def test_risk_mitigations_high_risk_phases(self):
        plan = MissionPlan(
            phases=[MissionPhase(name="p1", risk_level=RiskLevel.CRITICAL)],
            estimated_duration=100.0,
        )
        risk = self.planner.compute_risk_assessment(plan)
        assert any("pre-flight" in m.lower() or "checks" in m.lower() for m in risk.mitigation)
