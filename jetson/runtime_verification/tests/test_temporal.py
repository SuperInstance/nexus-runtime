"""Tests for temporal module — 38+ tests."""

import pytest
from jetson.runtime_verification.temporal import (
    TemporalFormula,
    TemporalLogicChecker,
)


class TestTemporalLogicChecker:
    def setup_method(self):
        self.chk = TemporalLogicChecker()

    # ---------- always (G) ----------
    def test_always_true_all_satisfy(self):
        trace = [{"safe": True}, {"safe": True}, {"safe": True}]
        assert self.chk.always(lambda s: s["safe"], trace) is True

    def test_always_false_one_fails(self):
        trace = [{"safe": True}, {"safe": False}, {"safe": True}]
        assert self.chk.always(lambda s: s["safe"], trace) is False

    def test_always_empty_trace(self):
        assert self.chk.always(lambda s: True, []) is True

    def test_always_single_true(self):
        assert self.chk.always(lambda s: s > 0, [5]) is True

    def test_always_single_false(self):
        assert self.chk.always(lambda s: s > 0, [-1]) is False

    def test_always_numeric_range(self):
        trace = [1, 2, 3, 4, 5]
        assert self.chk.always(lambda s: s < 10, trace) is True

    # ---------- eventually (F) ----------
    def test_eventually_true_present(self):
        trace = [{"found": False}, {"found": False}, {"found": True}]
        assert self.chk.eventually(lambda s: s["found"], trace) is True

    def test_eventually_false_never(self):
        trace = [{"found": False}, {"found": False}]
        assert self.chk.eventually(lambda s: s["found"], trace) is False

    def test_eventually_empty_trace(self):
        assert self.chk.eventually(lambda s: True, []) is False

    def test_eventually_first_step(self):
        assert self.chk.eventually(lambda s: s == 42, [42, 0, 0]) is True

    def test_eventually_last_step(self):
        assert self.chk.eventually(lambda s: s == 99, [0, 0, 99]) is True

    # ---------- until (U) ----------
    def test_until_satisfied(self):
        trace = [{"a": True, "b": False}, {"a": True, "b": False}, {"a": True, "b": True}]
        assert self.chk.until(lambda s: s["a"], lambda s: s["b"], trace) is True

    def test_until_a_fails_before_b(self):
        trace = [{"a": True, "b": False}, {"a": False, "b": False}, {"a": True, "b": True}]
        assert self.chk.until(lambda s: s["a"], lambda s: s["b"], trace) is False

    def test_until_b_never_holds(self):
        trace = [{"a": True, "b": False}, {"a": True, "b": False}]
        assert self.chk.until(lambda s: s["a"], lambda s: s["b"], trace) is False

    def test_until_b_at_first_step(self):
        trace = [{"a": True, "b": True}]
        assert self.chk.until(lambda s: s["a"], lambda s: s["b"], trace) is True

    def test_until_empty_trace(self):
        assert self.chk.until(lambda s: True, lambda s: True, []) is False

    def test_until_numeric(self):
        trace = [5, 4, 3, 2, 1, 0]
        assert self.chk.until(lambda s: s > 0, lambda s: s == 0, trace) is True

    # ---------- next_step (X) ----------
    def test_next_step_true(self):
        trace = [1, 2, 3]
        assert self.chk.next_step(lambda s: s == 2, trace, 0) is True

    def test_next_step_false(self):
        trace = [1, 3, 3]
        assert self.chk.next_step(lambda s: s == 2, trace, 0) is False

    def test_next_step_negative_index(self):
        trace = [1, 2, 3]
        assert self.chk.next_step(lambda s: True, trace, -1) is False

    def test_next_step_at_last_step(self):
        trace = [1, 2]
        assert self.chk.next_step(lambda s: True, trace, 1) is False

    def test_next_step_beyond_trace(self):
        trace = [1]
        assert self.chk.next_step(lambda s: True, trace, 0) is False

    def test_next_step_middle(self):
        trace = [10, 20, 30]
        assert self.chk.next_step(lambda s: s == 30, trace, 1) is True

    # ---------- never ----------
    def test_never_true(self):
        trace = [{"bad": False}, {"bad": False}]
        assert self.chk.never(lambda s: s["bad"], trace) is True

    def test_never_false(self):
        trace = [{"bad": False}, {"bad": True}]
        assert self.chk.never(lambda s: s["bad"], trace) is False

    def test_never_empty_trace(self):
        assert self.chk.never(lambda s: True, []) is True

    # ---------- imply ----------
    def test_imply_always_holds(self):
        trace = [{"a": True, "b": True}, {"a": True, "b": True}]
        assert self.chk.imply(lambda s: s["a"], lambda s: s["b"], trace) is True

    def test_imply_antecedent_never(self):
        trace = [{"a": False, "b": False}, {"a": False, "b": True}]
        assert self.chk.imply(lambda s: s["a"], lambda s: s["b"], trace) is True

    def test_imply_violation(self):
        trace = [{"a": True, "b": True}, {"a": True, "b": False}]
        assert self.chk.imply(lambda s: s["a"], lambda s: s["b"], trace) is False

    def test_imply_empty_trace(self):
        assert self.chk.imply(lambda s: True, lambda s: False, []) is True

    def test_imply_numeric(self):
        trace = [5, 10, 15]
        # if val > 0 then val > 0 is trivially true
        assert self.chk.imply(lambda s: s > 0, lambda s: s > 0, trace) is True

    # ---------- parse_formula ----------
    def test_parse_always(self):
        f = self.chk.parse_formula("G(safe)")
        assert f.formula_type == "always"
        assert f.atomic_props == ["safe"]

    def test_parse_eventually(self):
        f = self.chk.parse_formula("F(found)")
        assert f.formula_type == "eventually"
        assert f.atomic_props == ["found"]

    def test_parse_next(self):
        f = self.chk.parse_formula("X(next_prop)")
        assert f.formula_type == "next"

    def test_parse_never(self):
        f = self.chk.parse_formula("!danger")
        assert f.formula_type == "never"
        assert f.atomic_props == ["danger"]

    def test_parse_until(self):
        f = self.chk.parse_formula("A U B")
        assert f.formula_type == "until"
        assert f.atomic_props == ["A", "B"]

    def test_parse_imply(self):
        f = self.chk.parse_formula("locked => safe")
        assert f.formula_type == "imply"
        assert f.atomic_props == ["locked", "safe"]

    def test_parse_atomic(self):
        f = self.chk.parse_formula("safe")
        assert f.formula_type == "atomic"
        assert f.atomic_props == ["safe"]

    def test_parse_case_insensitive_g(self):
        f = self.chk.parse_formula("g(x)")
        assert f.formula_type == "always"

    def test_parse_whitespace_stripped(self):
        f = self.chk.parse_formula("  G( prop )  ")
        assert f.formula_str.strip() == "G( prop )"
        assert f.formula_type == "always"

    # ---------- check_trace ----------
    def test_check_trace_always_pass(self):
        formula = TemporalFormula(
            formula_str="G(safe)", formula_type="always", atomic_props=["safe"]
        )
        trace = [{"safe": True}, {"safe": True}]
        passed, _ = self.chk.check_trace(trace, formula)
        assert passed is True

    def test_check_trace_always_fail(self):
        formula = TemporalFormula(
            formula_str="G(safe)", formula_type="always", atomic_props=["safe"]
        )
        trace = [{"safe": True}, {"safe": False}]
        passed, counter = self.chk.check_trace(trace, formula)
        assert passed is False
        assert counter is not None

    def test_check_trace_eventually_pass(self):
        formula = TemporalFormula(
            formula_str="F(found)", formula_type="eventually", atomic_props=["found"]
        )
        trace = [{"found": False}, {"found": True}]
        passed, _ = self.chk.check_trace(trace, formula)
        assert passed is True

    def test_check_trace_eventually_fail(self):
        formula = TemporalFormula(
            formula_str="F(found)", formula_type="eventually", atomic_props=["found"]
        )
        trace = [{"found": False}]
        passed, counter = self.chk.check_trace(trace, formula)
        assert passed is False
        assert counter is not None

    def test_check_trace_until_pass(self):
        formula = TemporalFormula(
            formula_str="wait U ready",
            formula_type="until",
            atomic_props=["wait", "ready"],
        )
        trace = [{"wait": True, "ready": False}, {"wait": True, "ready": True}]
        passed, _ = self.chk.check_trace(trace, formula)
        assert passed is True

    def test_check_trace_never_pass(self):
        formula = TemporalFormula(
            formula_str="!danger", formula_type="never", atomic_props=["danger"]
        )
        trace = [{"danger": False}, {"danger": False}]
        passed, _ = self.chk.check_trace(trace, formula)
        assert passed is True

    def test_check_trace_imply_pass(self):
        formula = TemporalFormula(
            formula_str="hot => alarm",
            formula_type="imply",
            atomic_props=["hot", " alarm"],
        )
        trace = [{"hot": True, " alarm": True}, {"hot": False, " alarm": False}]
        passed, _ = self.chk.check_trace(trace, formula)
        assert passed is True

    def test_check_trace_imply_fail(self):
        formula = TemporalFormula(
            formula_str="hot => alarm",
            formula_type="imply",
            atomic_props=["hot", " alarm"],
        )
        trace = [{"hot": True, " alarm": False}]
        passed, counter = self.chk.check_trace(trace, formula)
        assert passed is False

    def test_check_trace_next_pass(self):
        formula = TemporalFormula(
            formula_str="X(safe)", formula_type="next", atomic_props=["safe"]
        )
        trace = [{"x": 1}, {"safe": True}]
        passed, _ = self.chk.check_trace(trace, formula)
        assert passed is True

    def test_check_trace_next_fail(self):
        formula = TemporalFormula(
            formula_str="X(safe)", formula_type="next", atomic_props=["safe"]
        )
        trace = [{"x": 1}, {"safe": False}]
        passed, _ = self.chk.check_trace(trace, formula)
        assert passed is False

    def test_check_trace_atomic_pass(self):
        formula = TemporalFormula(
            formula_str="safe", formula_type="atomic", atomic_props=["safe"]
        )
        trace = [{"safe": True}]
        passed, _ = self.chk.check_trace(trace, formula)
        assert passed is True

    def test_check_trace_empty_trace(self):
        formula = TemporalFormula(
            formula_str="G(safe)", formula_type="always", atomic_props=["safe"]
        )
        passed, counter = self.chk.check_trace([], formula)
        assert passed is True
        assert counter is None

    def test_check_trace_unknown_type(self):
        formula = TemporalFormula(
            formula_str="?", formula_type="compound", atomic_props=[]
        )
        passed, counter = self.chk.check_trace([{"x": 1}], formula)
        assert passed is False
        assert counter is not None

    def test_check_trace_until_insufficient_props(self):
        formula = TemporalFormula(
            formula_str="A U", formula_type="until", atomic_props=["A"]
        )
        passed, counter = self.chk.check_trace([{"A": True}], formula)
        assert passed is False

    def test_check_trace_imply_insufficient_props(self):
        formula = TemporalFormula(
            formula_str="A =>", formula_type="imply", atomic_props=["A"]
        )
        passed, counter = self.chk.check_trace([{"A": True}], formula)
        assert passed is False
