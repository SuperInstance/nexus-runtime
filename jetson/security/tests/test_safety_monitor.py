"""Tests for safety_monitor module."""

import time
import pytest
from jetson.security.safety_monitor import (
    ActionResult,
    Criticality,
    SafetyInvariant,
    SafetyInvariantMonitor,
    SafetyStatus,
    SafetyViolation,
)


# ── SafetyInvariant ────────────────────────────────────────────────

class TestSafetyInvariant:
    def test_create(self):
        inv = SafetyInvariant(
            name="depth_limit",
            expression="depth < 100",
            criticality=Criticality.HIGH,
        )
        assert inv.name == "depth_limit"
        assert inv.status is True
        assert inv.last_check == 0.0

    def test_with_check_fn(self):
        def check(state):
            return (state["x"] < 10, None)
        inv = SafetyInvariant(
            name="x_limit", expression="x<10",
            criticality=Criticality.MEDIUM, check_fn=check,
        )
        assert inv.check_fn is not None


# ── SafetyViolation ────────────────────────────────────────────────

class TestSafetyViolation:
    def test_create(self):
        v = SafetyViolation(
            invariant="depth",
            value=150.0,
            limit=100.0,
            timestamp=1.0,
            action_taken="none",
        )
        assert v.invariant == "depth"
        assert v.value == 150.0
        assert v.action_taken == "none"


# ── ActionResult ────────────────────────────────────────────────────

class TestActionResult:
    def test_success(self):
        r = ActionResult(action="shutdown", success=True, message="ok")
        assert r.success is True
        assert r.action == "shutdown"

    def test_failure(self):
        r = ActionResult(action="retry", success=False, message="failed")
        assert r.success is False


# ── SafetyInvariantMonitor ─────────────────────────────────────────

class TestMonitorConstruction:
    def test_construct(self):
        m = SafetyInvariantMonitor()
        assert m.get_invariants() == {}
        assert m.get_violations() == []


class TestRegisterInvariant:
    def test_register_returns_id(self):
        m = SafetyInvariantMonitor()
        inv_id = m.register_invariant("depth", lambda s: (True, None))
        assert inv_id.startswith("inv_")

    def test_register_multiple(self):
        m = SafetyInvariantMonitor()
        id1 = m.register_invariant("a", lambda s: (True, None))
        id2 = m.register_invariant("b", lambda s: (True, None))
        assert id1 != id2
        assert len(m.get_invariants()) == 2

    def test_register_with_criticality(self):
        m = SafetyInvariantMonitor()
        inv_id = m.register_invariant(
            "crit", lambda s: (True, None), criticality=Criticality.CRITICAL
        )
        inv = m.get_invariants()[inv_id]
        assert inv.criticality == Criticality.CRITICAL

    def test_register_with_expression(self):
        m = SafetyInvariantMonitor()
        inv_id = m.register_invariant(
            "temp", lambda s: (True, None), expression="temp < 80"
        )
        inv = m.get_invariants()[inv_id]
        assert inv.expression == "temp < 80"


class TestCheckAll:
    def test_no_violations(self):
        m = SafetyInvariantMonitor()
        m.register_invariant("depth", lambda s: (s["depth"] < 100, None))
        violations = m.check_all({"depth": 50})
        assert len(violations) == 0

    def test_violations_detected(self):
        m = SafetyInvariantMonitor()
        m.register_invariant("depth", lambda s: (s["depth"] < 100, "too deep"))
        violations = m.check_all({"depth": 150})
        assert len(violations) == 1
        assert violations[0].invariant == "depth"

    def test_multiple_violations(self):
        m = SafetyInvariantMonitor()
        m.register_invariant("depth", lambda s: (s["depth"] < 100, "deep"))
        m.register_invariant("temp", lambda s: (s["temp"] < 80, "hot"))
        violations = m.check_all({"depth": 200, "temp": 100})
        assert len(violations) == 2

    def test_partial_violations(self):
        m = SafetyInvariantMonitor()
        m.register_invariant("a", lambda s: (s["a"] < 10, None))
        m.register_invariant("b", lambda s: (s["b"] < 10, None))
        violations = m.check_all({"a": 5, "b": 20})
        assert len(violations) == 1

    def test_violation_details(self):
        m = SafetyInvariantMonitor()
        m.register_invariant("x", lambda s: (False, "x is bad"))
        violations = m.check_all({"x": 1})
        assert violations[0].details == "x is bad"

    def test_violations_accumulate(self):
        m = SafetyInvariantMonitor()
        m.register_invariant("x", lambda s: (False, "err"))
        m.check_all({"x": 1})
        m.check_all({"x": 1})
        assert len(m.get_violations()) == 2

    def test_updates_last_check(self):
        m = SafetyInvariantMonitor()
        inv_id = m.register_invariant("x", lambda s: (True, None))
        m.check_all({"x": 1})
        inv = m.get_invariants()[inv_id]
        assert inv.last_check > 0

    def test_updates_status(self):
        m = SafetyInvariantMonitor()
        inv_id = m.register_invariant("x", lambda s: (s["x"] < 5, None))
        m.check_all({"x": 10})
        assert m.get_invariants()[inv_id].status is False
        m.check_all({"x": 2})
        assert m.get_invariants()[inv_id].status is True


class TestCheckInvariant:
    def test_valid(self):
        m = SafetyInvariantMonitor()
        inv_id = m.register_invariant("x", lambda s: (s["x"] < 10, None))
        assert m.check_invariant(inv_id, {"x": 5}) is None

    def test_violation(self):
        m = SafetyInvariantMonitor()
        inv_id = m.register_invariant("x", lambda s: (s["x"] < 10, "over"))
        v = m.check_invariant(inv_id, {"x": 20})
        assert v is not None
        assert v.invariant == "x"

    def test_nonexistent_id(self):
        m = SafetyInvariantMonitor()
        assert m.check_invariant("missing", {}) is None

    def test_no_check_fn(self):
        m = SafetyInvariantMonitor()
        inv_id = m.register_invariant("x", lambda s: (True, None))
        inv = m.get_invariants()[inv_id]
        inv.check_fn = None
        assert m.check_invariant(inv_id, {"x": 999}) is None


class TestHandleViolation:
    def test_default_handler(self):
        m = SafetyInvariantMonitor()
        v = SafetyViolation(invariant="x", value=10, limit=5, timestamp=1.0)
        result = m.handle_violation(v)
        assert result.success is True
        assert result.action == "logged"
        assert v.action_taken == "logged"

    def test_custom_handler(self):
        m = SafetyInvariantMonitor()
        v = SafetyViolation(invariant="x", value=10, limit=5, timestamp=1.0)
        handler = lambda viol: ActionResult(action="shutdown", success=True, message="shutdown")
        result = m.handle_violation(v, handler)
        assert result.action == "shutdown"

    def test_violation_action_updated(self):
        m = SafetyInvariantMonitor()
        v = SafetyViolation(invariant="x", value=10, limit=5, timestamp=1.0)
        handler = lambda viol: ActionResult(action="abort", success=True)
        m.handle_violation(v, handler)
        assert v.action_taken == "abort"


class TestRegisterHandler:
    def test_handler_triggered_on_violation(self):
        m = SafetyInvariantMonitor()
        inv_id = m.register_invariant("x", lambda s: (False, "err"))
        actions = []
        def handler(viol):
            actions.append("handled")
            return ActionResult(action="handle", success=True)
        m.register_handler(inv_id, handler)
        m.check_all({"x": 1})
        assert len(actions) == 1


class TestGetSafetyStatus:
    def test_all_safe(self):
        m = SafetyInvariantMonitor()
        m.register_invariant("a", lambda s: (True, None), Criticality.HIGH)
        m.check_all({})
        assert m.get_safety_status() == SafetyStatus.SAFE

    def test_medium_violation_warning(self):
        m = SafetyInvariantMonitor()
        m.register_invariant("a", lambda s: (False, None), Criticality.MEDIUM)
        m.check_all({})
        assert m.get_safety_status() == SafetyStatus.WARNING

    def test_high_violation_critical(self):
        m = SafetyInvariantMonitor()
        m.register_invariant("a", lambda s: (False, None), Criticality.HIGH)
        m.check_all({})
        assert m.get_safety_status() == SafetyStatus.CRITICAL

    def test_critical_violation_emergency(self):
        m = SafetyInvariantMonitor()
        m.register_invariant("a", lambda s: (False, None), Criticality.CRITICAL)
        m.check_all({})
        assert m.get_safety_status() == SafetyStatus.EMERGENCY

    def test_critical_takes_precedence(self):
        m = SafetyInvariantMonitor()
        m.register_invariant("a", lambda s: (False, None), Criticality.MEDIUM)
        m.register_invariant("b", lambda s: (False, None), Criticality.CRITICAL)
        m.check_all({})
        assert m.get_safety_status() == SafetyStatus.EMERGENCY

    def test_no_invariants_safe(self):
        m = SafetyInvariantMonitor()
        assert m.get_safety_status() == SafetyStatus.SAFE


class TestComputeSafetyScore:
    def test_perfect_score(self):
        m = SafetyInvariantMonitor()
        m.register_invariant("a", lambda s: (True, None), Criticality.HIGH)
        m.check_all({})
        score = m.compute_safety_score()
        assert score == 1.0

    def test_zero_score(self):
        m = SafetyInvariantMonitor()
        m.register_invariant("a", lambda s: (False, None), Criticality.HIGH)
        m.check_all({})
        score = m.compute_safety_score()
        assert score < 1.0

    def test_custom_inputs(self):
        m = SafetyInvariantMonitor()
        invs = [
            SafetyInvariant(name="a", expression="", criticality=Criticality.HIGH, status=True),
            SafetyInvariant(name="b", expression="", criticality=Criticality.HIGH, status=False),
        ]
        score = m.compute_safety_score([], invs)
        assert 0.0 <= score <= 1.0

    def test_empty_invariants(self):
        m = SafetyInvariantMonitor()
        assert m.compute_safety_score() == 1.0

    def test_weighted_criticality(self):
        # LOW violation has less weight than CRITICAL in the base score
        from jetson.security.safety_monitor import SafetyInvariant
        invs_low = [SafetyInvariant(name="a", expression="", criticality=Criticality.LOW, status=False)]
        invs_crit = [SafetyInvariant(name="a", expression="", criticality=Criticality.CRITICAL, status=False)]
        m = SafetyInvariantMonitor()
        # With no violations, base scores differ by criticality weight
        score_low = m.compute_safety_score([], invs_low)
        score_crit = m.compute_safety_score([], invs_crit)
        # Both are 0.0 base (all violated). Weight affects the score when mixed.
        # Add a passing invariant to see the difference
        invs_mixed_low = [
            SafetyInvariant(name="a", expression="", criticality=Criticality.LOW, status=True),
            SafetyInvariant(name="b", expression="", criticality=Criticality.LOW, status=False),
        ]
        invs_mixed_crit = [
            SafetyInvariant(name="a", expression="", criticality=Criticality.CRITICAL, status=True),
            SafetyInvariant(name="b", expression="", criticality=Criticality.CRITICAL, status=False),
        ]
        score_mixed_low = m.compute_safety_score([], invs_mixed_low)
        score_mixed_crit = m.compute_safety_score([], invs_mixed_crit)
        # Both should be 0.5 (one pass, one fail, same weight)
        assert abs(score_mixed_low - 0.5) < 1e-9
        assert abs(score_mixed_crit - 0.5) < 1e-9
        # Now with different weights: LOW=1, CRITICAL=8
        invs_weighted = [
            SafetyInvariant(name="low_ok", expression="", criticality=Criticality.LOW, status=True),
            SafetyInvariant(name="crit_fail", expression="", criticality=Criticality.CRITICAL, status=False),
        ]
        score_w = m.compute_safety_score([], invs_weighted)
        # total_weight = 1 + 8 = 9, satisfied = 1, score = 1/9 ≈ 0.111
        assert abs(score_w - 1.0/9.0) < 1e-9


class TestClearViolations:
    def test_clear(self):
        m = SafetyInvariantMonitor()
        m.register_invariant("a", lambda s: (False, "err"))
        m.check_all({})
        assert len(m.get_violations()) == 1
        m.clear_violations()
        assert m.get_violations() == []


class TestRemoveInvariant:
    def test_remove_existing(self):
        m = SafetyInvariantMonitor()
        inv_id = m.register_invariant("a", lambda s: (True, None))
        assert m.remove_invariant(inv_id) is True
        assert inv_id not in m.get_invariants()

    def test_remove_nonexistent(self):
        m = SafetyInvariantMonitor()
        assert m.remove_invariant("missing") is False

    def test_remove_clears_handler(self):
        m = SafetyInvariantMonitor()
        inv_id = m.register_invariant("a", lambda s: (False, "err"))
        m.register_handler(inv_id, lambda v: ActionResult("ok", True))
        m.remove_invariant(inv_id)
        # Should not crash
        m.check_all({})


# ── SafetyStatus Enum ──────────────────────────────────────────────

class TestSafetyStatusEnum:
    def test_values(self):
        statuses = list(SafetyStatus)
        assert SafetyStatus.SAFE in statuses
        assert SafetyStatus.EMERGENCY in statuses
        assert len(statuses) == 4


# ── Criticality Enum ───────────────────────────────────────────────

class TestCriticalityEnum:
    def test_values(self):
        crits = list(Criticality)
        assert Criticality.LOW in crits
        assert Criticality.CRITICAL in crits
        assert len(crits) == 4
