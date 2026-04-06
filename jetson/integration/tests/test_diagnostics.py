"""Tests for DiagnosticSuite — Phase 5 Round 10."""

import time
import pytest
from jetson.integration.diagnostics import (
    DiagnosticResult,
    DiagnosticSuite,
)


def _passing_test():
    return DiagnosticResult(test_name="pass1", passed=True, message="OK")


def _failing_test():
    return DiagnosticResult(test_name="fail1", passed=False, message="not OK")


def _error_test():
    raise RuntimeError("boom")


def _slow_test():
    time.sleep(0.05)
    return DiagnosticResult(test_name="slow1", passed=True, message="done")


def _skipped_test():
    return DiagnosticResult(test_name="skip1", passed=True, message="SKIP: not applicable")


@pytest.fixture
def suite():
    return DiagnosticSuite()


# === DiagnosticResult ===

class TestDiagnosticResult:
    def test_default(self):
        r = DiagnosticResult(test_name="x", passed=True)
        assert r.test_name == "x"
        assert r.passed is True
        assert r.message == ""
        assert r.details == ""
        assert r.category == "general"
        assert r.duration_s == 0.0
        assert r.error is None

    def test_timestamp_populated(self):
        before = time.time()
        r = DiagnosticResult(test_name="x", passed=True)
        after = time.time()
        assert before <= r.timestamp <= after

    def test_full(self):
        r = DiagnosticResult(
            test_name="y", passed=False,
            message="err", details="stack trace",
            category="network", duration_s=0.1, error="E")
        assert r.error == "E"
        assert r.category == "network"


# === Registration ===

class TestRegistration:
    def test_add_test(self, suite):
        suite.add_test("t1", _passing_test, "unit")
        assert "t1" in suite.get_test_names()

    def test_add_multiple(self, suite):
        for i in range(5):
            suite.add_test(f"t{i}", _passing_test)
        assert len(suite.get_test_names()) == 5

    def test_remove_test(self, suite):
        suite.add_test("t1", _passing_test)
        assert suite.remove_test("t1") is True
        assert "t1" not in suite.get_test_names()

    def test_remove_missing(self, suite):
        assert suite.remove_test("ghost") is False

    def test_categories(self, suite):
        suite.add_test("t1", _passing_test, "unit")
        suite.add_test("t2", _passing_test, "network")
        suite.add_test("t3", _passing_test, "unit")
        cats = suite.get_categories()
        assert "unit" in cats
        assert "network" in cats

    def test_empty_categories(self, suite):
        assert suite.get_categories() == []

    def test_empty_test_names(self, suite):
        assert suite.get_test_names() == []


# === Execution ===

class TestExecution:
    def test_run_all(self, suite):
        suite.add_test("t1", _passing_test)
        suite.add_test("t2", _failing_test)
        results = suite.run_all()
        assert len(results) == 2

    def test_run_all_passing(self, suite):
        suite.add_test("t1", _passing_test)
        suite.add_test("t2", _passing_test)
        results = suite.run_all()
        assert all(r.passed for r in results)

    def test_run_single(self, suite):
        suite.add_test("t1", _passing_test)
        r = suite.run_test("t1")
        assert r.passed is True

    def test_run_single_missing(self, suite):
        r = suite.run_test("ghost")
        assert r.passed is False
        assert "not found" in r.message

    def test_run_category(self, suite):
        suite.add_test("t1", _passing_test, "unit")
        suite.add_test("t2", _failing_test, "network")
        results = suite.run_category("unit")
        assert len(results) == 1

    def test_run_category_empty(self, suite):
        suite.add_test("t1", _passing_test, "unit")
        results = suite.run_category("network")
        assert results == []

    def test_error_test(self, suite):
        suite.add_test("err", _error_test)
        r = suite.run_test("err")
        assert r.passed is False
        assert r.error is not None

    def test_duration_recorded(self, suite):
        suite.add_test("slow", _slow_test)
        r = suite.run_test("slow")
        assert r.duration_s >= 0.04

    def test_category_inherited(self, suite):
        suite.add_test("t1", _passing_test, "custom")
        r = suite.run_test("t1")
        assert r.category == "custom"


# === Summary ===

class TestSummary:
    def test_empty_summary(self, suite):
        s = suite.get_results_summary()
        assert s["total"] == 0
        assert s["passed"] == 0
        assert s["failed"] == 0

    def test_pass_fail_summary(self, suite):
        suite.add_test("p", _passing_test)
        suite.add_test("f", _failing_test)
        suite.run_all()
        s = suite.get_results_summary()
        assert s["passed"] == 1
        assert s["failed"] == 1

    def test_error_counted(self, suite):
        suite.add_test("err", _error_test)
        suite.run_all()
        s = suite.get_results_summary()
        assert s["errored"] == 1

    def test_skip_counted(self, suite):
        suite.add_test("sk", _skipped_test)
        suite.run_all()
        s = suite.get_results_summary()
        assert s["skipped"] == 1

    def test_all_pass_summary(self, suite):
        for i in range(3):
            suite.add_test(f"p{i}", _passing_test)
        suite.run_all()
        s = suite.get_results_summary()
        assert s["passed"] == 3
        assert s["failed"] == 0


# === Report Generation ===

class TestReport:
    def test_report_not_empty(self, suite):
        suite.add_test("t1", _passing_test)
        suite.run_all()
        report = suite.generate_report()
        assert len(report) > 0

    def test_report_contains_test_name(self, suite):
        suite.add_test("t1", _passing_test)
        suite.run_all()
        report = suite.generate_report()
        assert "t1" in report

    def test_report_contains_pass(self, suite):
        suite.add_test("t1", _passing_test)
        suite.run_all()
        report = suite.generate_report()
        assert "PASS" in report

    def test_report_contains_fail(self, suite):
        suite.add_test("f1", _failing_test)
        suite.run_all()
        report = suite.generate_report()
        assert "FAIL" in report

    def test_report_custom_results(self, suite):
        results = [DiagnosticResult(test_name="custom", passed=True)]
        report = suite.generate_report(results)
        assert "custom" in report

    def test_report_header(self, suite):
        suite.add_test("t1", _passing_test)
        suite.run_all()
        report = suite.generate_report()
        assert "DIAGNOSTIC REPORT" in report


# === Clear / Last Results ===

class TestClearResults:
    def test_clear_results(self, suite):
        suite.add_test("t1", _passing_test)
        suite.run_all()
        suite.clear_results()
        s = suite.get_results_summary()
        assert s["total"] == 0

    def test_get_last_results(self, suite):
        suite.add_test("t1", _passing_test)
        suite.add_test("t2", _failing_test)
        suite.run_all()
        last = suite.get_last_results()
        assert len(last) == 2

    def test_last_results_after_clear(self, suite):
        suite.add_test("t1", _passing_test)
        suite.run_all()
        suite.clear_results()
        assert suite.get_last_results() == []

    def test_run_all_replaces_results(self, suite):
        suite.add_test("t1", _passing_test)
        suite.run_all()
        suite.add_test("t2", _passing_test)
        suite.run_all()
        s = suite.get_results_summary()
        assert s["total"] == 2
