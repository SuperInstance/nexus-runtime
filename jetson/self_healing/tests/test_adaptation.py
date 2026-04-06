"""Tests for adaptation module — 38 tests."""

import pytest

from jetson.self_healing.adaptation import (
    AdaptationPlan,
    AdaptationResult,
    AdaptationRiskLevel,
    SystemAdapter,
)
from jetson.self_healing.diagnosis import Diagnosis


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def adapter():
    return SystemAdapter()


@pytest.fixture
def sample_diagnosis():
    return Diagnosis(
        fault_id="f-001",
        root_cause="gps_antenna_obstruction",
        confidence=0.7,
        contributing_factors=["component=gps_module", "ctx.snr=-5", "ctx.satellites=0"],
        recommended_fix="Clear antenna line of sight",
    )


@pytest.fixture
def sample_plan():
    return AdaptationPlan(
        trigger="fault_x",
        changes=[
            {"parameter": "x.retry", "old_value": 3, "new_value": 5, "component": "x"},
        ],
        expected_improvement=0.6,
        risk_of_change=0.3,
    )


# ── AdaptationPlan ───────────────────────────────────────────────────────

class TestAdaptationPlan:
    def test_fields(self):
        plan = AdaptationPlan(
            trigger="test_trigger",
            changes=[{"parameter": "p1", "old_value": 1, "new_value": 2}],
            expected_improvement=0.5,
            risk_of_change=0.2,
        )
        assert plan.trigger == "test_trigger"
        assert len(plan.changes) == 1
        assert plan.expected_improvement == 0.5
        assert plan.risk_of_change == 0.2
        assert len(plan.id) == 12

    def test_defaults(self):
        plan = AdaptationPlan(trigger="t", changes=[], expected_improvement=0.0, risk_of_change=0.0)
        assert plan.description == ""

    def test_timestamp_auto(self):
        import time
        before = time.time()
        plan = AdaptationPlan(trigger="t", changes=[], expected_improvement=0.0, risk_of_change=0.0)
        after = time.time()
        assert before <= plan.timestamp <= after


# ── AdaptationResult ─────────────────────────────────────────────────────

class TestAdaptationResult:
    def test_success(self):
        result = AdaptationResult(
            success=True,
            plan_id="p-1",
            applied_changes=[{"parameter": "x", "old_value": 1, "new_value": 2}],
            measured_improvement=0.5,
        )
        assert result.success
        assert result.plan_id == "p-1"
        assert len(result.applied_changes) == 1

    def test_failure(self):
        result = AdaptationResult(
            success=False, plan_id="p-1", applied_changes=[], measured_improvement=0.0
        )
        assert not result.success

    def test_side_effects(self):
        result = AdaptationResult(
            success=True, plan_id="p", applied_changes=[], measured_improvement=0.5,
            side_effects=["warning 1", "warning 2"],
        )
        assert len(result.side_effects) == 2


# ── SystemAdapter ────────────────────────────────────────────────────────

class TestSystemAdapter:
    def test_create_adaptation_plan(self, adapter, sample_diagnosis):
        plan = adapter.create_adaptation_plan(sample_diagnosis)
        assert plan.trigger != ""
        assert len(plan.changes) >= 2
        assert plan.expected_improvement > 0
        assert plan.risk_of_change >= 0

    def test_create_plan_with_config(self, adapter, sample_diagnosis):
        config = {"gps_module.retry_count": 5, "gps_module.timeout": 60}
        plan = adapter.create_adaptation_plan(sample_diagnosis, config)
        assert len(plan.changes) >= 2

    def test_create_plan_low_confidence(self, adapter):
        diag = Diagnosis(
            fault_id="f", root_cause="unknown", confidence=0.2,
            contributing_factors=["component=sensor"],
        )
        plan = adapter.create_adaptation_plan(diag)
        # Low confidence should add health_check_interval change
        has_health = any("health_check" in c["parameter"] for c in plan.changes)
        assert has_health

    def test_apply_adaptation(self, adapter, sample_plan):
        result = adapter.apply_adaptation(sample_plan)
        assert result.success
        assert len(result.applied_changes) == 1
        assert result.plan_id == sample_plan.id

    def test_apply_adaptation_updates_config(self, adapter, sample_plan):
        adapter.apply_adaptation(sample_plan)
        config = adapter.current_config
        assert config["x.retry"] == 5

    def test_apply_empty_plan(self, adapter):
        plan = AdaptationPlan(trigger="t", changes=[], expected_improvement=0.0, risk_of_change=0.0)
        result = adapter.apply_adaptation(plan)
        assert not result.success

    def test_apply_multiple_changes(self, adapter):
        plan = AdaptationPlan(
            trigger="t",
            changes=[
                {"parameter": "a", "old_value": 1, "new_value": 2},
                {"parameter": "b", "old_value": 3, "new_value": 4},
            ],
            expected_improvement=0.5,
            risk_of_change=0.3,
        )
        result = adapter.apply_adaptation(plan)
        assert result.success
        assert len(result.applied_changes) == 2
        assert adapter.current_config["a"] == 2
        assert adapter.current_config["b"] == 4

    def test_apply_side_effects_high_timeout(self, adapter):
        plan = AdaptationPlan(
            trigger="t",
            changes=[{"parameter": "x.timeout", "old_value": 30, "new_value": 200}],
            expected_improvement=0.3,
            risk_of_change=0.5,
        )
        result = adapter.apply_adaptation(plan)
        assert len(result.side_effects) > 0

    def test_apply_side_effects_high_retry(self, adapter):
        plan = AdaptationPlan(
            trigger="t",
            changes=[{"parameter": "x.retry", "old_value": 3, "new_value": 20}],
            expected_improvement=0.3,
            risk_of_change=0.5,
        )
        result = adapter.apply_adaptation(plan)
        assert len(result.side_effects) > 0

    def test_apply_records_history(self, adapter, sample_plan):
        adapter.apply_adaptation(sample_plan)
        history = adapter.track_adaptation_history()
        assert len(history) == 1

    def test_evaluate_adaptation_result_improvement(self, adapter, sample_plan):
        before = {"error_rate": 10.0, "latency": 200.0}
        after = {"error_rate": 5.0, "latency": 100.0}
        effectiveness = adapter.evaluate_adaptation_result(sample_plan, before, after)
        assert effectiveness > 0.5

    def test_evaluate_adaptation_result_no_change(self, adapter, sample_plan):
        before = {"throughput": 100.0}
        after = {"throughput": 100.0}
        effectiveness = adapter.evaluate_adaptation_result(sample_plan, before, after)
        assert abs(effectiveness - 0.5) < 0.01

    def test_evaluate_adaptation_result_degradation(self, adapter, sample_plan):
        before = {"throughput": 100.0}
        after = {"throughput": 50.0}
        effectiveness = adapter.evaluate_adaptation_result(sample_plan, before, after)
        assert effectiveness < 0.5

    def test_evaluate_empty_metrics(self, adapter, sample_plan):
        eff = adapter.evaluate_adaptation_result(sample_plan, {}, {})
        assert eff == 0.5

    def test_evaluate_no_common_keys(self, adapter, sample_plan):
        eff = adapter.evaluate_adaptation_result(sample_plan, {"a": 1}, {"b": 2})
        assert eff == 0.5

    def test_compute_risk_from_changes_empty(self, adapter):
        assert adapter.compute_adaptation_risk_from_changes([]) == 0.0

    def test_compute_risk_small_change(self, adapter):
        changes = [{"parameter": "x", "old_value": 100, "new_value": 105}]
        risk = adapter.compute_adaptation_risk_from_changes(changes)
        assert risk < 0.5

    def test_compute_risk_large_change(self, adapter):
        changes = [{"parameter": "x", "old_value": 10, "new_value": 1000}]
        risk = adapter.compute_adaptation_risk_from_changes(changes)
        assert risk > 0.5

    def test_compute_risk_multiple_changes(self, adapter):
        changes = [
            {"parameter": "a.timeout", "old_value": 30, "new_value": 60},
            {"parameter": "b.retry", "old_value": 3, "new_value": 10},
            {"parameter": "c.interval", "old_value": 60, "new_value": 30},
        ]
        risk = adapter.compute_adaptation_risk_from_changes(changes)
        assert risk > 0

    def test_compute_adaptation_risk_plan(self, adapter, sample_plan):
        risk = adapter.compute_adaptation_risk(sample_plan)
        assert 0.0 <= risk <= 1.0

    def test_select_safest_adaptation(self, adapter):
        plans = [
            AdaptationPlan("t1", [], 0.8, 0.8),  # improvement/risk = 1.0
            AdaptationPlan("t2", [], 0.3, 0.1),  # improvement/risk = 3.0
            AdaptationPlan("t3", [], 0.5, 0.2),  # improvement/risk = 2.5
        ]
        best = adapter.select_safest_adaptation(plans)
        assert best is not None
        assert best.trigger == "t2"

    def test_select_safest_empty(self, adapter):
        assert adapter.select_safest_adaptation([]) is None

    def test_select_safest_zero_risk(self, adapter):
        plans = [
            AdaptationPlan("t1", [], 0.5, 0.0),
            AdaptationPlan("t2", [], 0.3, 0.5),
        ]
        best = adapter.select_safest_adaptation(plans)
        assert best.trigger == "t1"

    def test_revert_adaptation(self, adapter, sample_plan):
        adapter.current_config = {"x.retry": 3}
        adapter.apply_adaptation(sample_plan)
        assert adapter.current_config["x.retry"] == 5
        success = adapter.revert_adaptation(sample_plan)
        assert success
        assert adapter.current_config == {"x.retry": 3}

    def test_revert_empty_stack(self, adapter):
        assert adapter.revert_adaptation(sample_plan) is False

    def test_revert_multiple(self, adapter):
        adapter.current_config = {}
        plan1 = AdaptationPlan("t1", [{"parameter": "a", "old_value": 1, "new_value": 2}], 0.5, 0.3)
        plan2 = AdaptationPlan("t2", [{"parameter": "b", "old_value": 3, "new_value": 4}], 0.5, 0.3)
        adapter.apply_adaptation(plan1)
        adapter.apply_adaptation(plan2)
        adapter.revert_adaptation(plan2)
        assert "b" not in adapter.current_config
        assert adapter.current_config["a"] == 2

    def test_current_config_property(self, adapter):
        adapter.current_config = {"key": "value"}
        assert adapter.current_config == {"key": "value"}

    def test_current_config_is_copy(self, adapter):
        adapter.current_config = {"key": "value"}
        cfg = adapter.current_config
        cfg["key"] = "modified"
        assert adapter.current_config["key"] == "value"

    def test_clear_history(self, adapter, sample_plan):
        adapter.apply_adaptation(sample_plan)
        adapter.clear_history()
        assert adapter.track_adaptation_history() == []

    def test_track_history_is_copy(self, adapter, sample_plan):
        adapter.apply_adaptation(sample_plan)
        h = adapter.track_adaptation_history()
        h.append({})
        assert len(adapter.track_adaptation_history()) == 1
