"""System adaptation module — adapt system configuration after failures.

Pure Python, zero external dependencies.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Data classes & enums ─────────────────────────────────────────────────

class AdaptationRiskLevel(Enum):
    NEGLIGIBLE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AdaptationPlan:
    """A plan for adapting system configuration."""
    trigger: str  # description of what triggered the adaptation
    changes: List[Dict[str, Any]]  # list of {parameter, old_value, new_value, component}
    expected_improvement: float  # 0.0 – 1.0
    risk_of_change: float  # 0.0 – 1.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class AdaptationResult:
    """Outcome of applying an adaptation plan."""
    success: bool
    plan_id: str
    applied_changes: List[Dict[str, Any]]
    measured_improvement: float
    side_effects: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    message: str = ""


# ── SystemAdapter ────────────────────────────────────────────────────────

class SystemAdapter:
    """Plans and applies system adaptations based on diagnosis and history."""

    def __init__(self) -> None:
        self._adaptation_history: List[Dict[str, Any]] = []
        self._current_config: Dict[str, Any] = {}
        self._rollback_stack: List[Dict[str, Any]] = []

    # ── planning ──

    def create_adaptation_plan(self, diagnosis: Any,
                                current_config: Optional[Dict[str, Any]] = None) -> AdaptationPlan:
        """Create an adaptation plan based on diagnosis and current config."""
        current_config = current_config or self._current_config

        root_cause = getattr(diagnosis, "root_cause", "")
        conf = getattr(diagnosis, "confidence", 0.5)
        fix = getattr(diagnosis, "recommended_fix", "")
        contributing = getattr(diagnosis, "contributing_factors", [])

        changes: List[Dict[str, Any]] = []

        # Generate parameter changes based on diagnosis
        comp = ""
        for factor in contributing:
            if factor.startswith("component="):
                comp = factor.split("=", 1)[1]
                break

        # Change 1: Increase retry count for failing component
        retry_key = f"{comp}.retry_count" if comp else "system.retry_count"
        old_retry = current_config.get(retry_key, 3)
        new_retry = old_retry + 2
        changes.append({
            "parameter": retry_key,
            "old_value": old_retry,
            "new_value": new_retry,
            "component": comp,
            "reason": "Increase retry count to handle transient failures",
        })

        # Change 2: Adjust timeout
        timeout_key = f"{comp}.timeout" if comp else "system.timeout"
        old_timeout = current_config.get(timeout_key, 30)
        new_timeout = int(old_timeout * 1.5)
        changes.append({
            "parameter": timeout_key,
            "old_value": old_timeout,
            "new_value": new_timeout,
            "component": comp,
            "reason": "Increase timeout to accommodate slow responses",
        })

        # Change 3: Enable health check if confidence is low
        if conf < 0.5:
            health_key = f"{comp}.health_check_interval" if comp else "system.health_check_interval"
            old_interval = current_config.get(health_key, 60)
            new_interval = max(5, int(old_interval * 0.5))
            changes.append({
                "parameter": health_key,
                "old_value": old_interval,
                "new_value": new_interval,
                "component": comp,
                "reason": "Increase health check frequency for better monitoring",
            })

        expected_improvement = min(1.0, conf * 0.8 + len(changes) * 0.1)
        risk = self.compute_adaptation_risk_from_changes(changes)

        plan = AdaptationPlan(
            trigger=f"Fault: {root_cause}",
            changes=changes,
            expected_improvement=expected_improvement,
            risk_of_change=risk,
            description=f"Adapt system config after diagnosis of '{root_cause}'",
        )
        return plan

    # ── execution ──

    def apply_adaptation(self, plan: AdaptationPlan) -> AdaptationResult:
        """Apply an adaptation plan to the current configuration."""
        # Save rollback state
        rollback_snapshot = dict(self._current_config)
        self._rollback_stack.append(rollback_snapshot)

        applied: List[Dict[str, Any]] = []
        side_effects: List[str] = []

        for change in plan.changes:
            param = change["parameter"]
            old_val = change["old_value"]
            new_val = change["new_value"]

            # Track what we actually changed (old might differ from expected)
            actual_old = self._current_config.get(param, old_val)
            self._current_config[param] = new_val
            applied.append({
                "parameter": param,
                "old_value": actual_old,
                "new_value": new_val,
            })

            # Detect potential side effects
            if "timeout" in param and new_val > 100:
                side_effects.append(f"High timeout ({new_val}s) on {param} may cause slow failure detection")
            if "retry" in param and new_val > 10:
                side_effects.append(f"High retry count ({new_val}) on {param} may amplify load")

        success = True
        if not applied:
            success = False

        result = AdaptationResult(
            success=success,
            plan_id=plan.id,
            applied_changes=applied,
            measured_improvement=plan.expected_improvement,
            side_effects=side_effects,
            message=f"Applied {len(applied)} changes from plan {plan.id[:8]}",
        )

        self._adaptation_history.append({
            "plan_id": plan.id,
            "success": success,
            "changes_count": len(applied),
            "improvement": plan.expected_improvement,
            "risk": plan.risk_of_change,
            "timestamp": time.time(),
        })
        return result

    # ── evaluation ──

    def evaluate_adaptation_result(self, plan: AdaptationPlan,
                                    before_metrics: Dict[str, float],
                                    after_metrics: Dict[str, float]) -> float:
        """Evaluate effectiveness of an adaptation by comparing before/after metrics.

        Returns effectiveness score 0.0 – 1.0.
        """
        if not before_metrics or not after_metrics:
            return 0.5

        improvements: List[float] = []
        for key in before_metrics:
            if key in after_metrics:
                before_val = before_metrics[key]
                after_val = after_metrics[key]
                # We assume higher values are better for most metrics
                # Exception: lower is better for error/latency metrics
                if "error" in key or "latency" in key or "failure" in key:
                    if before_val > 0:
                        improvement = (before_val - after_val) / before_val
                    else:
                        improvement = 0.0
                else:
                    if before_val > 0:
                        improvement = (after_val - before_val) / before_val
                    else:
                        improvement = 0.0
                improvements.append(max(-1.0, min(1.0, improvement)))

        if not improvements:
            return 0.5

        avg_improvement = sum(improvements) / len(improvements)
        # Map to 0-1 range: 0 = no change, 1 = significant improvement
        effectiveness = max(0.0, min(1.0, 0.5 + avg_improvement * 0.5))
        return effectiveness

    # ── risk assessment ──

    def compute_adaptation_risk(self, plan: AdaptationPlan) -> float:
        """Compute overall risk score (0.0 – 1.0) for an adaptation plan."""
        return self.compute_adaptation_risk_from_changes(plan.changes)

    def compute_adaptation_risk_from_changes(self, changes: List[Dict[str, Any]]) -> float:
        """Compute risk from a list of proposed changes."""
        if not changes:
            return 0.0

        risk_factors: List[float] = []
        for change in changes:
            param = change.get("parameter", "")
            old_val = change.get("old_value", 0)
            new_val = change.get("new_value", 0)

            # Magnitude of change
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                if old_val != 0:
                    magnitude = abs(new_val - old_val) / abs(old_val)
                else:
                    magnitude = 1.0 if new_val != 0 else 0.0
            else:
                magnitude = 0.5 if old_val != new_val else 0.0
            risk_factors.append(min(1.0, magnitude))

            # Risk by parameter type
            if "timeout" in param:
                risk_factors.append(0.3 if new_val <= 120 else 0.6)
            elif "retry" in param:
                risk_factors.append(0.2 if new_val <= 5 else 0.4)
            elif "interval" in param:
                risk_factors.append(0.1 if new_val >= 5 else 0.3)

        # Combine: max risk factor * number of changes scaling
        base_risk = max(risk_factors) if risk_factors else 0.0
        count_factor = min(1.0, len(changes) * 0.2)
        combined = base_risk * 0.6 + count_factor * 0.4
        return min(1.0, combined)

    def select_safest_adaptation(self, alternatives: List[AdaptationPlan]) -> Optional[AdaptationPlan]:
        """Select the adaptation with the best improvement-to-risk ratio."""
        if not alternatives:
            return None
        scored: List[Tuple[float, AdaptationPlan]] = []
        for plan in alternatives:
            risk = plan.risk_of_change
            improvement = plan.expected_improvement
            if risk > 0:
                score = improvement / risk
            elif improvement > 0:
                score = improvement * 10.0  # high benefit, zero risk
            else:
                score = 0.0
            scored.append((score, plan))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    # ── rollback ──

    def revert_adaptation(self, plan: AdaptationPlan) -> bool:
        """Revert the most recent adaptation. Returns True if successful."""
        if not self._rollback_stack:
            return False
        self._current_config = self._rollback_stack.pop()
        return True

    # ── history & config ──

    def track_adaptation_history(self) -> List[Dict[str, Any]]:
        """Return the full adaptation history."""
        return list(self._adaptation_history)

    @property
    def current_config(self) -> Dict[str, Any]:
        return dict(self._current_config)

    @current_config.setter
    def current_config(self, config: Dict[str, Any]) -> None:
        self._current_config = dict(config)

    def clear_history(self) -> None:
        self._adaptation_history.clear()
        self._rollback_stack.clear()
