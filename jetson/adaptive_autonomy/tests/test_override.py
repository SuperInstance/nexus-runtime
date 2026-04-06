"""Tests for jetson.adaptive_autonomy.override."""

import time

import pytest

from jetson.adaptive_autonomy.levels import AutonomyLevel
from jetson.adaptive_autonomy.override import (
    OverrideManager,
    OverrideRequest,
    OverrideResult,
)


# ── OverrideRequest ───────────────────────────────────────────────

class TestOverrideRequest:
    def test_defaults(self):
        req = OverrideRequest(operator_id="op1", target_level=AutonomyLevel.MANUAL)
        assert req.reason == ""
        assert isinstance(req.timestamp, float)

    def test_custom(self):
        ts = 1700000000.0
        req = OverrideRequest(
            operator_id="op2",
            target_level=AutonomyLevel.FULL_AUTO,
            reason="testing",
            timestamp=ts,
        )
        assert req.operator_id == "op2"
        assert req.timestamp == ts


# ── OverrideResult ────────────────────────────────────────────────

class TestOverrideResult:
    def test_defaults(self):
        r = OverrideResult(accepted=True, new_level=AutonomyLevel.MANUAL)
        assert r.transition_time == 0.0
        assert r.acknowledgment_required is False

    def test_accepted(self):
        r = OverrideResult(accepted=True, new_level=AutonomyLevel.ASSISTED)
        assert r.accepted is True


# ── OverrideManager ───────────────────────────────────────────────

class TestOverrideManager:

    @pytest.fixture()
    def om(self):
        return OverrideManager()

    # request_override
    def test_request_returns_result(self, om):
        req = OverrideRequest(operator_id="op1", target_level=AutonomyLevel.ASSISTED)
        result = om.request_override(req)
        assert isinstance(result, OverrideResult)

    def test_request_accepted(self, om):
        req = OverrideRequest(operator_id="op1", target_level=AutonomyLevel.ASSISTED)
        result = om.request_override(req)
        assert result.accepted is True

    def test_request_sets_level(self, om):
        req = OverrideRequest(operator_id="op1", target_level=AutonomyLevel.FULL_AUTO)
        om.request_override(req)
        assert om._current_level == AutonomyLevel.FULL_AUTO

    def test_request_downgrade_ack_required(self, om):
        om._current_level = AutonomyLevel.AUTONOMOUS
        req = OverrideRequest(operator_id="op1", target_level=AutonomyLevel.MANUAL)
        result = om.request_override(req)
        assert result.acknowledgment_required is True

    def test_request_upgrade_no_ack(self, om):
        om._current_level = AutonomyLevel.MANUAL
        req = OverrideRequest(operator_id="op1", target_level=AutonomyLevel.ASSISTED)
        result = om.request_override(req)
        assert result.acknowledgment_required is False

    def test_request_same_level_no_ack(self, om):
        om._current_level = AutonomyLevel.SEMI_AUTO
        req = OverrideRequest(operator_id="op1", target_level=AutonomyLevel.SEMI_AUTO)
        result = om.request_override(req)
        assert result.acknowledgment_required is False

    # emergency_override
    def test_emergency_returns_result(self, om):
        result = om.emergency_override("op1")
        assert isinstance(result, OverrideResult)

    def test_emergency_sets_manual(self, om):
        om._current_level = AutonomyLevel.AUTONOMOUS
        result = om.emergency_override("op1")
        assert result.new_level == AutonomyLevel.MANUAL
        assert om._current_level == AutonomyLevel.MANUAL

    def test_emergency_no_acknowledgment(self, om):
        result = om.emergency_override("op1")
        assert result.acknowledgment_required is False

    def test_emergency_accepted(self, om):
        result = om.emergency_override("op1")
        assert result.accepted is True

    # validate_override
    def test_validate_no_restrictions(self, om):
        req = OverrideRequest(operator_id="op1", target_level=AutonomyLevel.AUTONOMOUS)
        assert om.validate_override(req, {}) is True

    def test_validate_max_level_ok(self, om):
        req = OverrideRequest(operator_id="op1", target_level=AutonomyLevel.SEMI_AUTO)
        perms = {"max_level": AutonomyLevel.FULL_AUTO}
        assert om.validate_override(req, perms) is True

    def test_validate_max_level_denied(self, om):
        req = OverrideRequest(operator_id="op1", target_level=AutonomyLevel.AUTONOMOUS)
        perms = {"max_level": AutonomyLevel.SEMI_AUTO}
        assert om.validate_override(req, perms) is False

    def test_validate_max_level_int(self, om):
        req = OverrideRequest(operator_id="op1", target_level=AutonomyLevel.AUTONOMOUS)
        perms = {"max_level": 2}
        assert om.validate_override(req, perms) is False

    def test_validate_allowed_targets_ok(self, om):
        req = OverrideRequest(operator_id="op1", target_level=AutonomyLevel.SEMI_AUTO)
        perms = {"allowed_targets": [AutonomyLevel.SEMI_AUTO, AutonomyLevel.MANUAL]}
        assert om.validate_override(req, perms) is True

    def test_validate_allowed_targets_denied(self, om):
        req = OverrideRequest(operator_id="op1", target_level=AutonomyLevel.AUTONOMOUS)
        perms = {"allowed_targets": [AutonomyLevel.MANUAL]}
        assert om.validate_override(req, perms) is False

    # compute_override_priority
    def test_priority_a_emergency(self, om):
        a = OverrideRequest(operator_id="op1", target_level=AutonomyLevel.MANUAL)
        b = OverrideRequest(operator_id="op2", target_level=AutonomyLevel.ASSISTED)
        assert om.compute_override_priority(a, b) == "a"

    def test_priority_b_emergency(self, om):
        a = OverrideRequest(operator_id="op1", target_level=AutonomyLevel.ASSISTED)
        b = OverrideRequest(operator_id="op2", target_level=AutonomyLevel.MANUAL)
        assert om.compute_override_priority(a, b) == "b"

    def test_priority_lower_level_wins(self, om):
        a = OverrideRequest(operator_id="op1", target_level=AutonomyLevel.ASSISTED)
        b = OverrideRequest(operator_id="op2", target_level=AutonomyLevel.SEMI_AUTO)
        assert om.compute_override_priority(a, b) == "a"

    def test_priority_same_level_recent_wins(self, om):
        a = OverrideRequest(operator_id="op1", target_level=AutonomyLevel.SEMI_AUTO, timestamp=10.0)
        b = OverrideRequest(operator_id="op2", target_level=AutonomyLevel.SEMI_AUTO, timestamp=5.0)
        assert om.compute_override_priority(a, b) == "a"

    def test_priority_same_level_older_b(self, om):
        a = OverrideRequest(operator_id="op1", target_level=AutonomyLevel.SEMI_AUTO, timestamp=5.0)
        b = OverrideRequest(operator_id="op2", target_level=AutonomyLevel.SEMI_AUTO, timestamp=10.0)
        assert om.compute_override_priority(a, b) == "b"

    # get_active_overrides
    def test_active_empty(self, om):
        assert om.get_active_overrides() == []

    def test_active_after_request(self, om):
        req = OverrideRequest(operator_id="op1", target_level=AutonomyLevel.ASSISTED)
        om.request_override(req)
        active = om.get_active_overrides()
        assert len(active) == 1
        assert active[0]["operator_id"] == "op1"

    def test_active_has_keys(self, om):
        req = OverrideRequest(operator_id="op1", target_level=AutonomyLevel.ASSISTED)
        om.request_override(req)
        rec = om.get_active_overrides()[0]
        for key in ("override_id", "operator_id", "target_level", "reason", "timestamp", "acknowledged"):
            assert key in rec

    # acknowledge_override
    def test_acknowledge_existing(self, om):
        req = OverrideRequest(operator_id="op1", target_level=AutonomyLevel.ASSISTED)
        om.request_override(req)
        oid = om.get_active_overrides()[0]["override_id"]
        assert om.acknowledge_override(oid) is True
        assert om.get_active_overrides() == []

    def test_acknowledge_nonexistent(self, om):
        assert om.acknowledge_override("nonexistent") is False

    # compute_recovery_plan
    def test_recovery_plan_returns_list(self, om):
        plan = om.compute_recovery_plan(AutonomyLevel.MANUAL)
        assert isinstance(plan, list)

    def test_recovery_plan_has_steps(self, om):
        plan = om.compute_recovery_plan(AutonomyLevel.MANUAL)
        assert len(plan) >= 3
        step_names = [s["step"] for s in plan]
        assert "verify_system_status" in step_names
        assert "reassess_environment" in step_names
        assert "confirm_recovery" in step_names

    def test_recovery_plan_increment_when_not_max(self, om):
        plan = om.compute_recovery_plan(AutonomyLevel.MANUAL)
        step_names = [s["step"] for s in plan]
        assert "increment_autonomy" in step_names

    def test_recovery_plan_maintain_at_max(self, om):
        plan = om.compute_recovery_plan(AutonomyLevel.AUTONOMOUS)
        step_names = [s["step"] for s in plan]
        assert "maintain_autonomy" in step_names

    def test_recovery_plan_dicts_have_description(self, om):
        plan = om.compute_recovery_plan(AutonomyLevel.SEMI_AUTO)
        for step in plan:
            assert "description" in step
            assert "step" in step
