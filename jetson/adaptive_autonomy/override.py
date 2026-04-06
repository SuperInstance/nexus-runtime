"""Human override and takeover controls for the autonomy system."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from jetson.adaptive_autonomy.levels import AutonomyLevel


@dataclass
class OverrideRequest:
    """An operator's request to override the current autonomy level."""
    operator_id: str
    target_level: AutonomyLevel
    reason: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class OverrideResult:
    """Result of processing an override request."""
    accepted: bool
    new_level: AutonomyLevel
    transition_time: float = 0.0
    acknowledgment_required: bool = False


@dataclass
class _ActiveOverride:
    """Internal bookkeeping for an active override."""
    override_id: str
    request: OverrideRequest
    result: OverrideResult
    acknowledged: bool = False


class OverrideManager:
    """Handles human-initiated autonomy overrides."""

    def __init__(self) -> None:
        self._active_overrides: List[_ActiveOverride] = []
        self._current_level: AutonomyLevel = AutonomyLevel.MANUAL
        self._override_history: List[Dict[str, Any]] = []

    # ---- public API ----

    def request_override(self, request: OverrideRequest) -> OverrideResult:
        """Process an override request and return the result."""
        result = OverrideResult(
            accepted=True,
            new_level=request.target_level,
            transition_time=time.time(),
            acknowledgment_required=request.target_level < self._current_level,
        )
        self._current_level = request.target_level
        oid = str(uuid.uuid4())[:8]
        self._active_overrides.append(
            _ActiveOverride(
                override_id=oid,
                request=request,
                result=result,
            )
        )
        self._override_history.append({
            "operator_id": request.operator_id,
            "target_level": request.target_level,
            "accepted": result.accepted,
            "timestamp": request.timestamp,
            "override_id": oid,
        })
        return result

    def emergency_override(self, operator_id: str) -> OverrideResult:
        """Immediately force MANUAL (L0) — no questions asked."""
        now = time.time()
        request = OverrideRequest(
            operator_id=operator_id,
            target_level=AutonomyLevel.MANUAL,
            reason="EMERGENCY",
            timestamp=now,
        )
        result = OverrideResult(
            accepted=True,
            new_level=AutonomyLevel.MANUAL,
            transition_time=now,
            acknowledgment_required=False,
        )
        self._current_level = AutonomyLevel.MANUAL
        oid = str(uuid.uuid4())[:8]
        self._active_overrides.append(
            _ActiveOverride(
                override_id=oid,
                request=request,
                result=result,
            )
        )
        self._override_history.append({
            "operator_id": operator_id,
            "target_level": AutonomyLevel.MANUAL,
            "accepted": True,
            "timestamp": now,
            "override_id": oid,
            "emergency": True,
        })
        return result

    def validate_override(
        self,
        request: OverrideRequest,
        operator_permissions: Dict[str, Any],
    ) -> bool:
        """Check whether the operator has permission for this override.

        Permissions dict may contain ``"max_level"`` (int or AutonomyLevel)
        and ``"allowed_targets"`` (list of AutonomyLevel values).
        """
        max_level = operator_permissions.get("max_level", AutonomyLevel.AUTONOMOUS)
        if isinstance(max_level, int) and not isinstance(max_level, bool):
            max_level = AutonomyLevel(max_level)
        if isinstance(max_level, AutonomyLevel) and request.target_level > max_level:
            return False

        allowed = operator_permissions.get("allowed_targets")
        if allowed is not None:
            if request.target_level not in allowed:
                return False
        return True

    def compute_override_priority(
        self,
        override_a: OverrideRequest,
        override_b: OverrideRequest,
    ) -> str:
        """Decide which override takes precedence.

        Emergency overrides (target == MANUAL) always win.
        Among equal priorities the more recent one wins.
        Returns ``"a"`` or ``"b"``.
        """
        # Emergency trumps everything
        a_emergency = override_a.target_level == AutonomyLevel.MANUAL
        b_emergency = override_b.target_level == AutonomyLevel.MANUAL
        if a_emergency and not b_emergency:
            return "a"
        if b_emergency and not a_emergency:
            return "b"
        # Lower target level = more cautious = higher priority
        if override_a.target_level < override_b.target_level:
            return "a"
        if override_b.target_level < override_a.target_level:
            return "b"
        # Tie-break by recency
        if override_a.timestamp >= override_b.timestamp:
            return "a"
        return "b"

    def get_active_overrides(self) -> List[Dict[str, Any]]:
        """Return list of dicts for all non-acknowledged overrides."""
        return [
            {
                "override_id": o.override_id,
                "operator_id": o.request.operator_id,
                "target_level": o.request.target_level,
                "reason": o.request.reason,
                "timestamp": o.request.timestamp,
                "acknowledged": o.acknowledged,
            }
            for o in self._active_overrides
            if not o.acknowledged
        ]

    def acknowledge_override(self, override_id: str) -> bool:
        """Mark an override as acknowledged.  Returns True if found."""
        for o in self._active_overrides:
            if o.override_id == override_id:
                o.acknowledged = True
                return True
        return False

    def compute_recovery_plan(
        self, after_override: AutonomyLevel
    ) -> List[Dict[str, str]]:
        """Generate a recovery plan to restore autonomy after a downgrade."""
        steps: List[Dict[str, str]] = []
        level_order = list(AutonomyLevel)
        idx = level_order.index(after_override)

        # Step 1: Verify system status
        steps.append({
            "step": "verify_system_status",
            "description": f"Verify system health after override to {after_override.name}.",
        })

        # Step 2: Re-assess environment
        steps.append({
            "step": "reassess_environment",
            "description": "Run full situation assessment on current environment.",
        })

        # Step 3: Gradual autonomy restoration (one level at a time up to the original)
        if idx < len(level_order) - 1:
            steps.append({
                "step": "increment_autonomy",
                "description": f"Consider raising to {level_order[idx + 1].name} after verification.",
            })
        else:
            steps.append({
                "step": "maintain_autonomy",
                "description": "Already at maximum level; no increment needed.",
            })

        # Step 4: Final confirmation
        steps.append({
            "step": "confirm_recovery",
            "description": "Operator confirms recovery is complete.",
        })

        return steps
