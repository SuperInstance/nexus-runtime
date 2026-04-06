"""Recovery strategies module — plan and execute automatic recovery actions.

Pure Python, zero external dependencies.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# ── Enums ─────────────────────────────────────────────────────────────────

class RecoveryType(Enum):
    RESTART = "restart"
    RECONFIGURE = "reconfigure"
    FAILOVER = "failover"
    ISOLATE = "isolate"
    PATCH = "patch"
    RESET = "reset"
    SCALING = "scaling"
    CUSTOM = "custom"


class Urgency(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class RecoveryStrategy(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class RecoveryAction:
    """A single step in a recovery plan."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    type: RecoveryType = RecoveryType.RESTART
    target: str = ""
    steps: List[str] = field(default_factory=list)
    expected_outcome: str = ""
    rollback_steps: List[str] = field(default_factory=list)
    estimated_time_seconds: float = 5.0
    risk_level: float = 0.5  # 0.0 – 1.0


@dataclass
class RecoveryResult:
    """Outcome of executing a recovery action."""
    success: bool
    action_taken: str
    time_to_recover: float
    residual_impact: float  # 0.0 – 1.0
    new_state: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    action_id: str = ""


# ── RecoveryManager ──────────────────────────────────────────────────────

class RecoveryManager:
    """Orchestrates fault recovery: planning, execution, and rollback."""

    def __init__(self) -> None:
        self._history: List[Dict[str, Any]] = []
        self._handlers: Dict[RecoveryType, Callable[[RecoveryAction], RecoveryResult]] = {}
        self._register_default_handlers()

    # ── handler management ──

    def register_handler(self, rtype: RecoveryType,
                         handler: Callable[[RecoveryAction], RecoveryResult]) -> None:
        self._handlers[rtype] = handler

    def _register_default_handlers(self) -> None:
        self._handlers[RecoveryType.RESTART] = self._default_restart
        self._handlers[RecoveryType.RECONFIGURE] = self._default_reconfigure
        self._handlers[RecoveryType.FAILOVER] = self._default_failover
        self._handlers[RecoveryType.ISOLATE] = self._default_isolate
        self._handlers[RecoveryType.PATCH] = self._default_patch
        self._handlers[RecoveryType.RESET] = self._default_reset

    # ── recovery planning ──

    def generate_recovery_plan(self, diagnosis: Any) -> List[RecoveryAction]:
        """Generate a prioritised list of recovery actions from a Diagnosis."""
        root_cause = getattr(diagnosis, "root_cause", "")
        fix = getattr(diagnosis, "recommended_fix", "")
        conf = getattr(diagnosis, "confidence", 0.5)
        fault_id = getattr(diagnosis, "fault_id", "")
        contributing = getattr(diagnosis, "contributing_factors", [])

        actions: List[RecoveryAction] = []

        # Action 1: Apply recommended fix (reconfigure)
        if fix:
            actions.append(RecoveryAction(
                type=RecoveryType.RECONFIGURE,
                target=root_cause,
                steps=["Analyse recommended fix", "Apply configuration change", "Verify system state"],
                expected_outcome="System reconfigured to address root cause",
                rollback_steps=["Revert configuration change"],
                estimated_time_seconds=10.0,
                risk_level=0.3,
            ))

        # Action 2: Restart the affected component
        actions.append(RecoveryAction(
            type=RecoveryType.RESTART,
            target=root_cause,
            steps=["Graceful shutdown", "Wait for cleanup", "Restart component", "Health check"],
            expected_outcome="Component restarted in clean state",
            rollback_steps=["Restore previous state snapshot"],
            estimated_time_seconds=15.0,
            risk_level=0.4,
        ))

        # Action 3: Failover if available
        if conf < 0.5 or len(contributing) > 3:
            actions.append(RecoveryAction(
                type=RecoveryType.FAILOVER,
                target=root_cause,
                steps=["Identify backup", "Activate backup", "Redirect traffic", "Verify backup health"],
                expected_outcome="Traffic redirected to healthy backup",
                rollback_steps=["Deactivate backup", "Restore original routing"],
                estimated_time_seconds=20.0,
                risk_level=0.5,
            ))

        # Action 4: Isolate if uncertain
        if conf < 0.3:
            actions.append(RecoveryAction(
                type=RecoveryType.ISOLATE,
                target=root_cause,
                steps=["Quarantine component", "Redirect dependencies", "Alert operators"],
                expected_outcome="Faulty component isolated from system",
                rollback_steps=["Remove quarantine", "Restore dependencies"],
                estimated_time_seconds=5.0,
                risk_level=0.2,
            ))

        return actions

    # ── execution ──

    def execute_recovery(self, action: RecoveryAction) -> RecoveryResult:
        """Execute a single recovery action and return the result."""
        start = time.time()
        handler = self._handlers.get(action.type)

        if handler is None:
            result = RecoveryResult(
                success=False,
                action_taken=action.type.value,
                time_to_recover=0.0,
                residual_impact=1.0,
                message=f"No handler for recovery type: {action.type.value}",
                action_id=action.id,
            )
        else:
            result = handler(action)

        result.time_to_recover = time.time() - start
        result.action_id = action.id

        self._history.append({
            "action_id": action.id,
            "type": action.type.value,
            "target": action.target,
            "success": result.success,
            "time_to_recover": result.time_to_recover,
            "residual_impact": result.residual_impact,
        })
        return result

    def rollback(self, action: RecoveryAction) -> RecoveryResult:
        """Roll back a previously executed action."""
        start = time.time()
        steps_executed = 0
        for step in action.rollback_steps:
            steps_executed += 1

        result = RecoveryResult(
            success=True,
            action_taken=f"rollback_{action.type.value}",
            time_to_recover=time.time() - start,
            residual_impact=0.1,
            new_state={"rollback_complete": True, "steps_executed": steps_executed},
            message=f"Rolled back {action.type.value} on {action.target}",
            action_id=action.id,
        )

        self._history.append({
            "action_id": action.id,
            "type": f"rollback_{action.type.value}",
            "target": action.target,
            "success": result.success,
            "time_to_recover": result.time_to_recover,
        })
        return result

    # ── strategy selection ──

    def select_recovery_strategy(self, diagnosis: Any, urgency: Urgency = Urgency.MEDIUM) -> RecoveryStrategy:
        """Select a recovery strategy based on diagnosis confidence and urgency."""
        conf = getattr(diagnosis, "confidence", 0.5)

        if urgency == Urgency.CRITICAL:
            return RecoveryStrategy.AGGRESSIVE
        elif urgency == Urgency.HIGH:
            return RecoveryStrategy.AGGRESSIVE if conf > 0.7 else RecoveryStrategy.MODERATE
        elif urgency == Urgency.MEDIUM:
            return RecoveryStrategy.MODERATE if conf > 0.4 else RecoveryStrategy.CONSERVATIVE
        else:
            return RecoveryStrategy.CONSERVATIVE

    def estimate_recovery_time(self, action: RecoveryAction) -> float:
        """Estimate time to complete a recovery action in seconds."""
        base = action.estimated_time_seconds
        risk_factor = 1.0 + action.risk_level * 0.5
        step_factor = 1.0 + len(action.steps) * 0.1
        return base * risk_factor * step_factor

    def compute_recovery_success_rate(self, history: Optional[List[Dict[str, Any]]] = None) -> float:
        """Compute the percentage of successful recoveries from history."""
        data = history if history is not None else self._history
        if not data:
            return 0.0
        successes = sum(1 for h in data if h.get("success", False))
        return (successes / len(data)) * 100.0

    # ── default handlers ─────────────────────────────────────────────────

    def _default_restart(self, action: RecoveryAction) -> RecoveryResult:
        return RecoveryResult(
            success=True,
            action_taken=RecoveryType.RESTART.value,
            time_to_recover=action.estimated_time_seconds,
            residual_impact=0.05,
            new_state={"status": "restarted", "target": action.target},
            message=f"Restarted {action.target}",
        )

    def _default_reconfigure(self, action: RecoveryAction) -> RecoveryResult:
        return RecoveryResult(
            success=True,
            action_taken=RecoveryType.RECONFIGURE.value,
            time_to_recover=action.estimated_time_seconds,
            residual_impact=0.02,
            new_state={"status": "reconfigured", "target": action.target},
            message=f"Reconfigured {action.target}",
        )

    def _default_failover(self, action: RecoveryAction) -> RecoveryResult:
        return RecoveryResult(
            success=True,
            action_taken=RecoveryType.FAILOVER.value,
            time_to_recover=action.estimated_time_seconds,
            residual_impact=0.1,
            new_state={"status": "failed_over", "target": action.target},
            message=f"Failed over from {action.target}",
        )

    def _default_isolate(self, action: RecoveryAction) -> RecoveryResult:
        return RecoveryResult(
            success=True,
            action_taken=RecoveryType.ISOLATE.value,
            time_to_recover=action.estimated_time_seconds,
            residual_impact=0.2,
            new_state={"status": "isolated", "target": action.target},
            message=f"Isolated {action.target}",
        )

    def _default_patch(self, action: RecoveryAction) -> RecoveryResult:
        return RecoveryResult(
            success=True,
            action_taken=RecoveryType.PATCH.value,
            time_to_recover=action.estimated_time_seconds,
            residual_impact=0.03,
            new_state={"status": "patched", "target": action.target},
            message=f"Patched {action.target}",
        )

    def _default_reset(self, action: RecoveryAction) -> RecoveryResult:
        return RecoveryResult(
            success=True,
            action_taken=RecoveryType.RESET.value,
            time_to_recover=action.estimated_time_seconds,
            residual_impact=0.15,
            new_state={"status": "reset", "target": action.target},
            message=f"Reset {action.target}",
        )

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)
