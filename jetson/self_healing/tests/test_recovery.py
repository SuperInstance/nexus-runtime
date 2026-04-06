"""Tests for recovery module — 36 tests."""

import pytest

from jetson.self_healing.diagnosis import Diagnosis
from jetson.self_healing.recovery import (
    RecoveryAction,
    RecoveryManager,
    RecoveryResult,
    RecoveryStrategy,
    RecoveryType,
    Urgency,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def manager():
    return RecoveryManager()


@pytest.fixture
def sample_diagnosis():
    return Diagnosis(
        fault_id="f-001",
        root_cause="overheating",
        confidence=0.8,
        contributing_factors=["high_temp", "fan_failure", "dust_buildup"],
        recommended_fix="Clean dust and replace fan",
    )


@pytest.fixture
def sample_action():
    return RecoveryAction(
        type=RecoveryType.RESTART,
        target="nav_engine",
        steps=["shutdown", "restart", "verify"],
        expected_outcome="Engine running",
        rollback_steps=["restore_snapshot"],
        estimated_time_seconds=10.0,
        risk_level=0.3,
    )


# ── RecoveryAction ────────────────────────────────────────────────────────

class TestRecoveryAction:
    def test_default_fields(self):
        action = RecoveryAction()
        assert action.type == RecoveryType.RESTART
        assert action.target == ""
        assert action.steps == []
        assert action.rollback_steps == []
        assert action.estimated_time_seconds == 5.0
        assert action.risk_level == 0.5
        assert len(action.id) == 12

    def test_custom_fields(self, sample_action):
        assert sample_action.type == RecoveryType.RESTART
        assert sample_action.target == "nav_engine"
        assert len(sample_action.steps) == 3


# ── RecoveryResult ───────────────────────────────────────────────────────

class TestRecoveryResult:
    def test_success(self):
        result = RecoveryResult(success=True, action_taken="restart", time_to_recover=5.0, residual_impact=0.1)
        assert result.success is True

    def test_failure(self):
        result = RecoveryResult(success=False, action_taken="failover", time_to_recover=0.0, residual_impact=1.0)
        assert result.success is False

    def test_new_state(self):
        result = RecoveryResult(success=True, action_taken="r", time_to_recover=1.0, residual_impact=0.0,
                                new_state={"key": "value"})
        assert result.new_state["key"] == "value"


# ── RecoveryManager ──────────────────────────────────────────────────────

class TestRecoveryManager:
    def test_register_handler(self, manager):
        def custom_handler(action):
            return RecoveryResult(True, "custom", 1.0, 0.0)
        manager.register_handler(RecoveryType.CUSTOM, custom_handler)
        action = RecoveryAction(type=RecoveryType.CUSTOM, target="x")
        result = manager.execute_recovery(action)
        assert result.success
        assert result.action_taken == "custom"

    def test_generate_recovery_plan(self, manager, sample_diagnosis):
        plan = manager.generate_recovery_plan(sample_diagnosis)
        assert len(plan) >= 2
        types = [a.type for a in plan]
        assert RecoveryType.RECONFIGURE in types
        assert RecoveryType.RESTART in types

    def test_generate_recovery_plan_low_confidence(self, manager):
        diag = Diagnosis(fault_id="f-1", root_cause="unknown", confidence=0.2,
                         contributing_factors=["a", "b", "c", "d"])
        plan = manager.generate_recovery_plan(diag)
        types = [a.type for a in plan]
        assert RecoveryType.FAILOVER in types
        assert RecoveryType.ISOLATE in types

    def test_generate_recovery_plan_no_fix(self, manager):
        diag = Diagnosis(fault_id="f-1", root_cause="cause", confidence=0.6, recommended_fix="")
        plan = manager.generate_recovery_plan(diag)
        # Should still have restart
        types = [a.type for a in plan]
        assert RecoveryType.RESTART in types

    def test_execute_recovery_restart(self, manager, sample_action):
        result = manager.execute_recovery(sample_action)
        assert result.success
        assert result.action_id == sample_action.id

    def test_execute_recovery_records_history(self, manager, sample_action):
        manager.execute_recovery(sample_action)
        assert len(manager.history) == 1

    def test_execute_recovery_no_handler(self, manager):
        action = RecoveryAction(type=RecoveryType.CUSTOM)
        result = manager.execute_recovery(action)
        assert result.success is False
        assert "No handler" in result.message

    def test_execute_recovery_reconfigure(self, manager):
        action = RecoveryAction(type=RecoveryType.RECONFIGURE, target="x")
        result = manager.execute_recovery(action)
        assert result.success
        assert result.new_state["status"] == "reconfigured"

    def test_execute_recovery_failover(self, manager):
        action = RecoveryAction(type=RecoveryType.FAILOVER, target="x")
        result = manager.execute_recovery(action)
        assert result.success
        assert result.new_state["status"] == "failed_over"

    def test_execute_recovery_isolate(self, manager):
        action = RecoveryAction(type=RecoveryType.ISOLATE, target="x")
        result = manager.execute_recovery(action)
        assert result.success
        assert result.new_state["status"] == "isolated"

    def test_execute_recovery_patch(self, manager):
        action = RecoveryAction(type=RecoveryType.PATCH, target="x")
        result = manager.execute_recovery(action)
        assert result.success

    def test_execute_recovery_reset(self, manager):
        action = RecoveryAction(type=RecoveryType.RESET, target="x")
        result = manager.execute_recovery(action)
        assert result.success

    def test_rollback(self, manager, sample_action):
        result = manager.rollback(sample_action)
        assert result.success
        assert "rollback_restart" in result.action_taken

    def test_rollback_records_history(self, manager, sample_action):
        manager.rollback(sample_action)
        assert len(manager.history) == 1

    def test_rollback_empty_steps(self, manager):
        action = RecoveryAction(type=RecoveryType.RESTART, target="x", rollback_steps=[])
        result = manager.rollback(action)
        assert result.success
        assert result.new_state["steps_executed"] == 0

    def test_select_strategy_critical(self, manager):
        diag = Diagnosis(fault_id="f", root_cause="c", confidence=0.9)
        strategy = manager.select_recovery_strategy(diag, Urgency.CRITICAL)
        assert strategy == RecoveryStrategy.AGGRESSIVE

    def test_select_strategy_high(self, manager):
        diag = Diagnosis(fault_id="f", root_cause="c", confidence=0.8)
        strategy = manager.select_recovery_strategy(diag, Urgency.HIGH)
        assert strategy == RecoveryStrategy.AGGRESSIVE

    def test_select_strategy_high_low_conf(self, manager):
        diag = Diagnosis(fault_id="f", root_cause="c", confidence=0.3)
        strategy = manager.select_recovery_strategy(diag, Urgency.HIGH)
        assert strategy == RecoveryStrategy.MODERATE

    def test_select_strategy_medium(self, manager):
        diag = Diagnosis(fault_id="f", root_cause="c", confidence=0.5)
        strategy = manager.select_recovery_strategy(diag, Urgency.MEDIUM)
        assert strategy == RecoveryStrategy.MODERATE

    def test_select_strategy_medium_low_conf(self, manager):
        diag = Diagnosis(fault_id="f", root_cause="c", confidence=0.2)
        strategy = manager.select_recovery_strategy(diag, Urgency.MEDIUM)
        assert strategy == RecoveryStrategy.CONSERVATIVE

    def test_select_strategy_low(self, manager):
        diag = Diagnosis(fault_id="f", root_cause="c", confidence=0.9)
        strategy = manager.select_recovery_strategy(diag, Urgency.LOW)
        assert strategy == RecoveryStrategy.CONSERVATIVE

    def test_estimate_recovery_time(self, manager, sample_action):
        est = manager.estimate_recovery_time(sample_action)
        assert est > 0
        # Base 10 * (1 + 0.3*0.5) * (1 + 3*0.1)
        base = 10.0
        expected = base * (1.0 + 0.3 * 0.5) * (1.0 + 3 * 0.1)
        assert abs(est - expected) < 0.01

    def test_estimate_recovery_time_no_steps(self, manager):
        action = RecoveryAction(estimated_time_seconds=5.0, risk_level=0.5)
        est = manager.estimate_recovery_time(action)
        expected = 5.0 * (1.0 + 0.25) * 1.0
        assert abs(est - expected) < 0.01

    def test_compute_recovery_success_rate_empty(self, manager):
        assert manager.compute_recovery_success_rate() == 0.0

    def test_compute_recovery_success_rate_all_success(self, manager):
        history = [
            {"success": True, "time_to_recover": 1.0},
            {"success": True, "time_to_recover": 2.0},
        ]
        rate = manager.compute_recovery_success_rate(history)
        assert rate == 100.0

    def test_compute_recovery_success_rate_mixed(self, manager):
        history = [
            {"success": True, "time_to_recover": 1.0},
            {"success": False, "time_to_recover": 5.0},
            {"success": True, "time_to_recover": 1.0},
            {"success": False, "time_to_recover": 5.0},
        ]
        rate = manager.compute_recovery_success_rate(history)
        assert rate == 50.0

    def test_compute_recovery_success_rate_from_internal_history(self, manager, sample_action):
        manager.execute_recovery(sample_action)
        rate = manager.compute_recovery_success_rate()
        assert rate == 100.0

    def test_history_property(self, manager, sample_action):
        manager.execute_recovery(sample_action)
        h = manager.history
        assert len(h) == 1
        h.append({})
        assert len(manager.history) == 1

    def test_urgency_ordering(self):
        assert Urgency.LOW.value < Urgency.MEDIUM.value < Urgency.HIGH.value < Urgency.CRITICAL.value

    def test_recovery_type_values(self):
        assert RecoveryType.RESTART.value == "restart"
        assert RecoveryType.RECONFIGURE.value == "reconfigure"
        assert RecoveryType.FAILOVER.value == "failover"
        assert RecoveryType.ISOLATE.value == "isolate"
        assert RecoveryType.PATCH.value == "patch"
        assert RecoveryType.RESET.value == "reset"
        assert RecoveryType.CUSTOM.value == "custom"

    def test_strategy_values(self):
        assert RecoveryStrategy.CONSERVATIVE.value == "conservative"
        assert RecoveryStrategy.MODERATE.value == "moderate"
        assert RecoveryStrategy.AGGRESSIVE.value == "aggressive"

    def test_multiple_recovery_actions_plan(self, manager):
        diag = Diagnosis(fault_id="f", root_cause="complex_failure", confidence=0.1,
                         contributing_factors=["a", "b", "c", "d", "e"],
                         recommended_fix="Apply patch XYZ")
        plan = manager.generate_recovery_plan(diag)
        assert len(plan) >= 4
