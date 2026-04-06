"""
Safety invariant checking for NEXUS runtime verification.

Provides data classes for invariant definitions and violations,
plus an InvariantChecker that manages registration, evaluation,
violation tracking, and coverage computation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Invariant:
    """Represents a safety invariant to be checked at runtime."""

    name: str
    check_fn: Callable[[Any], bool]
    severity: str = "warning"  # "info", "warning", "error", "critical"
    description: str = ""
    category: str = "general"


@dataclass
class Violation:
    """Represents a single invariant violation event."""

    invariant: str
    value: Any = None
    limit: Any = None
    timestamp: float = field(default_factory=time.time)
    context: Optional[Dict[str, Any]] = None


class InvariantChecker:
    """Manages safety invariants: registration, checking, and violation history."""

    def __init__(self) -> None:
        self._invariants: Dict[str, Invariant] = {}
        self._violation_history: Dict[str, List[Violation]] = {}

    def register(self, invariant: Invariant) -> None:
        """Register an invariant for checking."""
        self._invariants[invariant.name] = invariant
        if invariant.name not in self._violation_history:
            self._violation_history[invariant.name] = []

    def check_all(self, state: Any) -> List[Violation]:
        """Check all registered invariants against the given state."""
        violations: List[Violation] = []
        for name, inv in self._invariants.items():
            result = self.check(name, state)
            if result is not None:
                violations.append(result)
        return violations

    def check(self, name: str, state: Any) -> Optional[Violation]:
        """Check a single invariant by name. Returns Violation or None."""
        inv = self._invariants.get(name)
        if inv is None:
            return None
        try:
            passed = inv.check_fn(state)
        except Exception:
            # Treat exceptions as violations
            passed = False
        if not passed:
            violation = Violation(
                invariant=name,
                value=self._extract_value(state, name),
                limit=None,
                timestamp=time.time(),
                context={"severity": inv.severity, "category": inv.category},
            )
            self._violation_history[name].append(violation)
            return violation
        return None

    def get_violation_history(self, name: str, limit: int = 100) -> List[Violation]:
        """Get violation history for a named invariant, up to `limit` entries."""
        history = self._violation_history.get(name, [])
        return history[-limit:]

    @staticmethod
    def compute_invariant_coverage(checks_performed: int, total_invariants: int) -> float:
        """Compute the percentage of invariants that have been checked.

        Returns a float between 0.0 and 100.0.
        """
        if total_invariants <= 0:
            return 0.0
        coverage = (checks_performed / total_invariants) * 100.0
        return min(max(coverage, 0.0), 100.0)

    def reset_violations(self, name: str) -> None:
        """Reset violation history for a named invariant."""
        if name in self._violation_history:
            self._violation_history[name] = []

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all invariants and their violation counts."""
        summary: Dict[str, Any] = {
            "total_invariants": len(self._invariants),
            "invariants": {},
            "total_violations": 0,
        }
        for name, inv in self._invariants.items():
            count = len(self._violation_history.get(name, []))
            summary["invariants"][name] = {
                "severity": inv.severity,
                "category": inv.category,
                "description": inv.description,
                "violation_count": count,
            }
            summary["total_violations"] += count
        return summary

    def _extract_value(self, state: Any, name: str) -> Any:
        """Try to extract a relevant value from state for the violation."""
        if isinstance(state, dict):
            return state.get(name, None)
        return state
