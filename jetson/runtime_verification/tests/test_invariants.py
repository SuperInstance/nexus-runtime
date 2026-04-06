"""Tests for invariants module — 40+ tests."""

import time
import pytest
from jetson.runtime_verification.invariants import (
    Invariant,
    Violation,
    InvariantChecker,
)


# ---------- Invariant dataclass tests ----------

class TestInvariantDataclass:
    def test_create_minimal(self):
        inv = Invariant(name="test", check_fn=lambda s: True)
        assert inv.name == "test"
        assert inv.severity == "warning"
        assert inv.description == ""
        assert inv.category == "general"

    def test_create_full(self):
        inv = Invariant(
            name="temp_limit",
            check_fn=lambda s: s.get("temp", 0) < 100,
            severity="critical",
            description="Temperature must not exceed 100C",
            category="thermal",
        )
        assert inv.name == "temp_limit"
        assert inv.severity == "critical"
        assert inv.description == "Temperature must not exceed 100C"
        assert inv.category == "thermal"

    def test_check_fn_returns_true(self):
        inv = Invariant(name="pos", check_fn=lambda s: s > 0)
        assert inv.check_fn(5) is True

    def test_check_fn_returns_false(self):
        inv = Invariant(name="pos", check_fn=lambda s: s > 0)
        assert inv.check_fn(-1) is False

    def test_severity_levels(self):
        for sev in ("info", "warning", "error", "critical"):
            inv = Invariant(name=f"inv_{sev}", check_fn=lambda s: True, severity=sev)
            assert inv.severity == sev


# ---------- Violation dataclass tests ----------

class TestViolationDataclass:
    def test_create_minimal(self):
        v = Violation(invariant="test")
        assert v.invariant == "test"
        assert v.value is None
        assert v.limit is None
        assert v.context is None
        assert isinstance(v.timestamp, float)

    def test_create_full(self):
        ctx = {"severity": "critical", "extra": 42}
        v = Violation(
            invariant="temp_exceeded",
            value=105.3,
            limit=100.0,
            timestamp=1700000000.0,
            context=ctx,
        )
        assert v.invariant == "temp_exceeded"
        assert v.value == 105.3
        assert v.limit == 100.0
        assert v.timestamp == 1700000000.0
        assert v.context == ctx

    def test_default_timestamp_is_recent(self):
        before = time.time()
        v = Violation(invariant="x")
        after = time.time()
        assert before <= v.timestamp <= after


# ---------- InvariantChecker tests ----------

class TestInvariantChecker:
    def setup_method(self):
        self.checker = InvariantChecker()

    def test_register_single(self):
        inv = Invariant(name="a", check_fn=lambda s: True)
        self.checker.register(inv)
        assert "a" in self.checker._invariants

    def test_register_multiple(self):
        for i in range(5):
            self.checker.register(Invariant(name=f"inv_{i}", check_fn=lambda s: True))
        assert len(self.checker._invariants) == 5

    def test_register_duplicate_overwrites(self):
        inv1 = Invariant(name="x", check_fn=lambda s: True, severity="info")
        inv2 = Invariant(name="x", check_fn=lambda s: False, severity="critical")
        self.checker.register(inv1)
        self.checker.register(inv2)
        assert self.checker._invariants["x"].severity == "critical"

    def test_check_all_no_violations(self):
        self.checker.register(Invariant(name="positive", check_fn=lambda s: s["val"] > 0))
        self.checker.register(Invariant(name="under_100", check_fn=lambda s: s["val"] < 100))
        result = self.checker.check_all({"val": 50})
        assert result == []

    def test_check_all_with_violations(self):
        self.checker.register(Invariant(name="positive", check_fn=lambda s: s["val"] > 0))
        self.checker.register(Invariant(name="under_100", check_fn=lambda s: s["val"] < 100))
        result = self.checker.check_all({"val": 150})
        assert len(result) == 1
        assert result[0].invariant == "under_100"

    def test_check_all_all_violate(self):
        self.checker.register(Invariant(name="positive", check_fn=lambda s: s["val"] > 0))
        self.checker.register(Invariant(name="string", check_fn=lambda s: isinstance(s["val"], str)))
        result = self.checker.check_all({"val": -5})
        assert len(result) == 2

    def test_check_all_empty_checker(self):
        result = self.checker.check_all({"val": 42})
        assert result == []

    def test_check_passing(self):
        self.checker.register(Invariant(name="pos", check_fn=lambda s: s > 0))
        result = self.checker.check("pos", 5)
        assert result is None

    def test_check_failing(self):
        self.checker.register(Invariant(name="pos", check_fn=lambda s: s > 0))
        result = self.checker.check("pos", -1)
        assert result is not None
        assert isinstance(result, Violation)
        assert result.invariant == "pos"

    def test_check_nonexistent(self):
        result = self.checker.check("nonexistent", {})
        assert result is None

    def test_check_records_violation_history(self):
        self.checker.register(Invariant(name="pos", check_fn=lambda s: s > 0))
        self.checker.check("pos", -1)
        self.checker.check("pos", -2)
        history = self.checker.get_violation_history("pos")
        assert len(history) == 2

    def test_check_violation_context_includes_severity(self):
        self.checker.register(
            Invariant(name="crit", check_fn=lambda s: False, severity="critical")
        )
        v = self.checker.check("crit", {})
        assert v is not None
        assert v.context["severity"] == "critical"

    def test_check_violation_context_includes_category(self):
        self.checker.register(
            Invariant(name="thermal", check_fn=lambda s: False, category="thermal")
        )
        v = self.checker.check("thermal", {})
        assert v is not None
        assert v.context["category"] == "thermal"

    def test_check_exception_treated_as_violation(self):
        def bad_check(s):
            raise ValueError("bad")

        self.checker.register(Invariant(name="bad", check_fn=bad_check))
        v = self.checker.check("bad", {})
        assert v is not None
        assert v.invariant == "bad"

    def test_get_violation_history_respects_limit(self):
        self.checker.register(Invariant(name="x", check_fn=lambda s: False))
        for _ in range(20):
            self.checker.check("x", {})
        hist = self.checker.get_violation_history("x", limit=5)
        assert len(hist) == 5

    def test_get_violation_history_no_invariant(self):
        hist = self.checker.get_violation_history("nonexistent")
        assert hist == []

    def test_get_violation_history_default_limit(self):
        self.checker.register(Invariant(name="x", check_fn=lambda s: False))
        for _ in range(5):
            self.checker.check("x", {})
        hist = self.checker.get_violation_history("x")
        assert len(hist) == 5

    def test_reset_violations(self):
        self.checker.register(Invariant(name="x", check_fn=lambda s: False))
        self.checker.check("x", {})
        self.checker.check("x", {})
        self.checker.reset_violations("x")
        assert self.checker.get_violation_history("x") == []

    def test_reset_violations_nonexistent(self):
        # Should not raise
        self.checker.reset_violations("nope")

    def test_compute_invariant_coverage_full(self):
        cov = InvariantChecker.compute_invariant_coverage(10, 10)
        assert cov == 100.0

    def test_compute_invariant_coverage_half(self):
        cov = InvariantChecker.compute_invariant_coverage(5, 10)
        assert cov == 50.0

    def test_compute_invariant_coverage_zero(self):
        cov = InvariantChecker.compute_invariant_coverage(0, 10)
        assert cov == 0.0

    def test_compute_invariant_coverage_zero_total(self):
        cov = InvariantChecker.compute_invariant_coverage(5, 0)
        assert cov == 0.0

    def test_compute_invariant_coverage_clamped_high(self):
        cov = InvariantChecker.compute_invariant_coverage(15, 10)
        assert cov == 100.0

    def test_compute_invariant_coverage_clamped_negative(self):
        cov = InvariantChecker.compute_invariant_coverage(-1, 10)
        assert cov == 0.0

    def test_get_summary_empty(self):
        summary = self.checker.get_summary()
        assert summary["total_invariants"] == 0
        assert summary["total_violations"] == 0
        assert summary["invariants"] == {}

    def test_get_summary_with_data(self):
        self.checker.register(Invariant(name="a", check_fn=lambda s: True, severity="error", category="cat1", description="desc1"))
        self.checker.register(Invariant(name="b", check_fn=lambda s: False, severity="warning", category="cat2", description="desc2"))
        self.checker.check_all({})
        summary = self.checker.get_summary()
        assert summary["total_invariants"] == 2
        assert summary["total_violations"] == 1
        assert "a" in summary["invariants"]
        assert summary["invariants"]["b"]["violation_count"] == 1

    def test_get_summary_includes_all_fields(self):
        self.checker.register(
            Invariant(name="x", check_fn=lambda s: True, severity="critical", category="safety", description="Safety check")
        )
        s = self.checker.get_summary()
        info = s["invariants"]["x"]
        assert info["severity"] == "critical"
        assert info["category"] == "safety"
        assert info["description"] == "Safety check"

    def test_dict_state_value_extraction(self):
        self.checker.register(Invariant(name="temperature", check_fn=lambda s: s["temperature"] < 100))
        v = self.checker.check("temperature", {"temperature": 110})
        assert v is not None
        assert v.value == 110

    def test_non_dict_state_value_extraction(self):
        self.checker.register(Invariant(name="positive", check_fn=lambda s: s > 0))
        v = self.checker.check("positive", -5)
        assert v is not None
        assert v.value == -5
