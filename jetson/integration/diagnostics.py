"""
Diagnostic tools — register test functions, run them individually or
by category, and produce structured reports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class DiagnosticResult:
    test_name: str
    passed: bool
    message: str = ""
    details: str = ""
    timestamp: float = field(default_factory=time.time)
    category: str = "general"
    duration_s: float = 0.0
    error: Optional[str] = None


@dataclass
class _TestEntry:
    name: str
    test_fn: Callable[[], DiagnosticResult]
    category: str = "general"


class DiagnosticSuite:
    """Collect, categorize, and run diagnostic tests."""

    def __init__(self) -> None:
        self._tests: Dict[str, _TestEntry] = {}
        self._results: List[DiagnosticResult] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_test(self, name: str, test_fn: Callable[[], DiagnosticResult],
                 category: str = "general") -> None:
        self._tests[name] = _TestEntry(name=name, test_fn=test_fn, category=category)

    def remove_test(self, name: str) -> bool:
        return self._tests.pop(name, None) is not None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_all(self) -> List[DiagnosticResult]:
        self._results.clear()
        for entry in self._tests.values():
            result = self._run_single(entry)
            self._results.append(result)
        return list(self._results)

    def run_test(self, name: str) -> DiagnosticResult:
        entry = self._tests.get(name)
        if entry is None:
            return DiagnosticResult(
                test_name=name, passed=False,
                message=f"test '{name}' not found", category="error")
        return self._run_single(entry)

    def run_category(self, category: str) -> List[DiagnosticResult]:
        results: List[DiagnosticResult] = []
        for entry in self._tests.values():
            if entry.category == category:
                results.append(self._run_single(entry))
        return results

    def _run_single(self, entry: _TestEntry) -> DiagnosticResult:
        start = time.time()
        try:
            result = entry.test_fn()
            result.duration_s = time.time() - start
            result.category = entry.category
            result.test_name = entry.name
            return result
        except Exception as exc:
            return DiagnosticResult(
                test_name=entry.name, passed=False,
                message="exception during test",
                details=str(exc), category=entry.category,
                duration_s=time.time() - start,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def get_results_summary(self) -> Dict[str, Any]:
        passed = sum(1 for r in self._results if r.passed)
        failed = sum(1 for r in self._results if not r.passed and r.error is None)
        errored = sum(1 for r in self._results if r.error is not None)
        skipped = sum(1 for r in self._results if r.message and "skip" in r.message.lower())
        return {
            "total": len(self._results),
            "passed": passed,
            "failed": failed,
            "errored": errored,
            "skipped": skipped,
        }

    def generate_report(self, results: Optional[List[DiagnosticResult]] = None) -> str:
        if results is None:
            results = self._results
        summary = self.get_results_summary()
        lines: List[str] = [
            "=" * 60,
            "DIAGNOSTIC REPORT",
            "=" * 60,
            f"Total: {summary['total']}  "
            f"Passed: {summary['passed']}  "
            f"Failed: {summary['failed']}  "
            f"Errored: {summary['errored']}",
            "-" * 60,
        ]
        for r in results:
            icon = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{icon}] {r.test_name} ({r.category})")
            if r.message:
                lines.append(f"         {r.message}")
            if r.details:
                lines.append(f"         {r.details}")
            lines.append(f"         Duration: {r.duration_s:.4f}s")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_categories(self) -> List[str]:
        cats: List[str] = []
        seen: set = set()
        for entry in self._tests.values():
            if entry.category not in seen:
                seen.add(entry.category)
                cats.append(entry.category)
        return cats

    def get_test_names(self) -> List[str]:
        return list(self._tests.keys())

    def clear_results(self) -> None:
        self._results.clear()

    def get_last_results(self) -> List[DiagnosticResult]:
        return list(self._results)
