"""Safety invariant monitoring for marine robotics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class Criticality(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SafetyStatus(Enum):
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SafetyInvariant:
    name: str
    expression: str
    criticality: Criticality
    check_fn: Optional[Callable[[Dict[str, Any]], Tuple[bool, Optional[str]]]] = None
    invariant_id: str = ""
    last_check: float = 0.0
    status: bool = True  # True = satisfied, False = violated


@dataclass
class SafetyViolation:
    invariant: str
    value: float
    limit: float
    timestamp: float
    action_taken: str = "none"
    details: str = ""


@dataclass
class ActionResult:
    action: str
    success: bool
    message: str = ""


class SafetyInvariantMonitor:
    """Monitor safety invariants and handle violations."""

    def __init__(self) -> None:
        self._invariants: Dict[str, SafetyInvariant] = {}
        self._violations: List[SafetyViolation] = []
        self._handlers: Dict[str, Callable[[SafetyViolation], ActionResult]] = {}
        self._next_id: int = 0

    def register_invariant(
        self,
        name: str,
        check_fn: Callable[[Dict[str, Any]], Tuple[bool, Optional[str]]],
        criticality: Criticality = Criticality.HIGH,
        expression: str = "",
    ) -> str:
        """Register a safety invariant. Returns invariant_id."""
        inv_id = f"inv_{self._next_id}"
        self._next_id += 1
        inv = SafetyInvariant(
            name=name,
            expression=expression or name,
            criticality=criticality,
            check_fn=check_fn,
            invariant_id=inv_id,
            last_check=0.0,
            status=True,
        )
        self._invariants[inv_id] = inv
        return inv_id

    def check_all(self, state: Dict[str, Any]) -> List[SafetyViolation]:
        """Check all registered invariants against the current state."""
        violations: List[SafetyViolation] = []
        now = time.time()
        for inv_id, inv in self._invariants.items():
            if inv.check_fn is not None:
                passed, detail = inv.check_fn(state)
                inv.last_check = now
                inv.status = passed
                if not passed:
                    v = SafetyViolation(
                        invariant=inv.name,
                        value=state.get(inv.name, 0.0),
                        limit=0.0,
                        timestamp=now,
                        action_taken="none",
                        details=detail or "",
                    )
                    violations.append(v)
                    self._violations.append(v)
                    # Auto-trigger handler if registered
                    if inv_id in self._handlers:
                        self._handlers[inv_id](v)
        return violations

    def check_invariant(
        self, invariant_id: str, state: Dict[str, Any]
    ) -> Optional[SafetyViolation]:
        """Check a specific invariant. Returns violation or None."""
        inv = self._invariants.get(invariant_id)
        if inv is None or inv.check_fn is None:
            return None
        now = time.time()
        passed, detail = inv.check_fn(state)
        inv.last_check = now
        inv.status = passed
        if not passed:
            v = SafetyViolation(
                invariant=inv.name,
                value=state.get(inv.name, 0.0),
                limit=0.0,
                timestamp=now,
                action_taken="none",
                details=detail or "",
            )
            self._violations.append(v)
            return v
        return None

    def handle_violation(
        self,
        violation: SafetyViolation,
        handler: Optional[Callable[[SafetyViolation], ActionResult]] = None,
    ) -> ActionResult:
        """Handle a safety violation with a custom handler or default."""
        if handler is not None:
            result = handler(violation)
            violation.action_taken = result.action
            return result
        # Default handler: log and mark
        violation.action_taken = "logged"
        return ActionResult(
            action="logged",
            success=True,
            message=f"Violation logged for {violation.invariant}",
        )

    def register_handler(
        self, invariant_id: str, handler: Callable[[SafetyViolation], ActionResult]
    ) -> None:
        """Register a violation handler for a specific invariant."""
        self._handlers[invariant_id] = handler

    def get_safety_status(self) -> SafetyStatus:
        """Get overall safety status based on current invariant states."""
        has_critical = False
        has_high = False
        has_medium = False
        for inv in self._invariants.values():
            if not inv.status:
                if inv.criticality == Criticality.CRITICAL:
                    has_critical = True
                elif inv.criticality == Criticality.HIGH:
                    has_high = True
                elif inv.criticality == Criticality.MEDIUM:
                    has_medium = True
        if has_critical:
            return SafetyStatus.EMERGENCY
        if has_high:
            return SafetyStatus.CRITICAL
        if has_medium:
            return SafetyStatus.WARNING
        return SafetyStatus.SAFE

    def compute_safety_score(
        self,
        violations: Optional[List[SafetyViolation]] = None,
        invariants: Optional[List[SafetyInvariant]] = None,
    ) -> float:
        """Compute a safety score from 0.0 (unsafe) to 1.0 (fully safe)."""
        if violations is None:
            violations = self._violations
        if invariants is None:
            invariants = list(self._invariants.values())
        if not invariants:
            return 1.0

        total_weight = 0.0
        satisfied_weight = 0.0
        weight_map = {
            Criticality.LOW: 1.0,
            Criticality.MEDIUM: 2.0,
            Criticality.HIGH: 4.0,
            Criticality.CRITICAL: 8.0,
        }
        for inv in invariants:
            w = weight_map.get(inv.criticality, 1.0)
            total_weight += w
            if inv.status:
                satisfied_weight += w

        if total_weight == 0:
            return 1.0

        base_score = satisfied_weight / total_weight

        # Penalize for recent violations
        now = time.time()
        violation_penalty = 0.0
        for v in violations:
            age = now - v.timestamp
            if age < 60:
                violation_penalty += 0.1
            elif age < 300:
                violation_penalty += 0.05

        score = base_score - min(violation_penalty, base_score)
        return max(0.0, min(1.0, score))

    def get_invariants(self) -> Dict[str, SafetyInvariant]:
        return dict(self._invariants)

    def get_violations(self) -> List[SafetyViolation]:
        return list(self._violations)

    def clear_violations(self) -> None:
        self._violations.clear()

    def remove_invariant(self, invariant_id: str) -> bool:
        if invariant_id in self._invariants:
            del self._invariants[invariant_id]
            self._handlers.pop(invariant_id, None)
            return True
        return False
