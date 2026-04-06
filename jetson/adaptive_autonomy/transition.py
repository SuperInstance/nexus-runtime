"""Level transition logic — requesting, validating, and executing autonomy changes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from jetson.adaptive_autonomy.levels import AutonomyLevel, AutonomyLevelManager


@dataclass
class TransitionRequest:
    """A request to move from one autonomy level to another."""
    from_level: AutonomyLevel
    to_level: AutonomyLevel
    reason: str = ""
    urgency: str = "normal"          # normal | high | critical
    requires_confirmation: bool = False


@dataclass
class TransitionPolicy:
    """Policy governing how transitions behave."""
    allowed_transitions: Dict[AutonomyLevel, Set[AutonomyLevel]] = field(
        default_factory=dict
    )
    cooldown_seconds: float = 30.0
    confirmation_required: Set[Tuple[AutonomyLevel, AutonomyLevel]] = field(
        default_factory=set
    )
    max_transitions_per_hour: int = 60

    @classmethod
    def default_policy(cls) -> TransitionPolicy:
        """Create a permissive default policy (any level to any neighbour)."""
        all_levels = set(AutonomyLevel)
        allowed: Dict[AutonomyLevel, Set[AutonomyLevel]] = {}
        for lv in all_levels:
            allowed[lv] = set()
            for other in all_levels:
                if other != lv:
                    allowed[lv].add(other)

        confirmation: Set[Tuple[AutonomyLevel, AutonomyLevel]] = set()
        for lv in (AutonomyLevel.FULL_AUTO, AutonomyLevel.AUTONOMOUS):
            for target in (AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED):
                confirmation.add((lv, target))

        return cls(
            allowed_transitions=allowed,
            cooldown_seconds=30.0,
            confirmation_required=confirmation,
            max_transitions_per_hour=60,
        )


@dataclass
class _TransitionRecord:
    """Internal record kept in transition history."""
    from_level: AutonomyLevel
    to_level: AutonomyLevel
    reason: str
    timestamp: float
    approved: bool


class TransitionManager:
    """Orchestrates autonomy-level transitions with policy enforcement."""

    def __init__(self, policy: Optional[TransitionPolicy] = None) -> None:
        self.policy = policy or TransitionPolicy.default_policy()
        self._history: List[_TransitionRecord] = []
        self._last_transition_time: Optional[float] = None
        self._current_level: AutonomyLevel = AutonomyLevel.MANUAL
        self._level_manager = AutonomyLevelManager()

    # ---- public API ----

    @property
    def current_level(self) -> AutonomyLevel:
        return self._current_level

    def request_transition(self, request: TransitionRequest) -> Dict[str, object]:
        """Evaluate a transition request and return approval result.

        Returns dict with keys: ``approved`` (bool), ``reason`` (str),
        ``requires_confirmation`` (bool).
        """
        # Check policy allows this transition
        allowed = self.policy.allowed_transitions.get(request.from_level, set())
        if request.to_level not in allowed:
            return {
                "approved": False,
                "reason": f"Transition from {request.from_level.name} to "
                          f"{request.to_level.name} not allowed by policy.",
                "requires_confirmation": False,
            }

        # Cooldown check
        remaining = self.check_cooldown(self._last_transition_time)
        if remaining > 0 and request.urgency != "critical":
            return {
                "approved": False,
                "reason": f"Cooldown active: {remaining:.1f}s remaining.",
                "requires_confirmation": False,
            }

        # Rate-limit check
        recent = [
            r for r in self._history
            if time.time() - r.timestamp < 3600
        ]
        if len(recent) >= self.policy.max_transitions_per_hour:
            return {
                "approved": False,
                "reason": "Max transitions per hour exceeded.",
                "requires_confirmation": False,
            }

        needs_confirm = (
            request.requires_confirmation
            or (request.from_level, request.to_level)
            in self.policy.confirmation_required
        )
        return {
            "approved": True,
            "reason": "Transition approved.",
            "requires_confirmation": needs_confirm,
        }

    def execute_transition(
        self, from_level: AutonomyLevel, to_level: AutonomyLevel
    ) -> AutonomyLevel:
        """Execute a transition and record it in history."""
        new_level = to_level
        self._current_level = new_level
        now = time.time()
        self._last_transition_time = now
        self._history.append(
            _TransitionRecord(
                from_level=from_level,
                to_level=to_level,
                reason="executed",
                timestamp=now,
                approved=True,
            )
        )
        return new_level

    def get_available_transitions(
        self, current_level: AutonomyLevel
    ) -> List[AutonomyLevel]:
        """Return sorted list of levels reachable from *current_level*."""
        allowed = self.policy.allowed_transitions.get(current_level, set())
        return sorted(allowed, key=lambda lv: lv.value)

    def compute_transition_safety(
        self,
        current: AutonomyLevel,
        target: AutonomyLevel,
        context: Dict[str, float],
    ) -> float:
        """Compute a 0-1 safety score for the proposed transition.

        Factors: risk in context vs risk tolerance at target level,
        decision authority gap, and direction of change.
        """
        caps_target = self._level_manager.get_capabilities(target)
        context_risk = context.get("risk", 0.0)
        risk_ok = 1.0 - max(0.0, context_risk - caps_target.max_risk_tolerance)

        # Authority gap penalty
        authority_diff = abs(
            self._level_manager.compute_decision_authority(current)
            - self._level_manager.compute_decision_authority(target)
        )
        authority_factor = max(0.0, 1.0 - authority_diff / 100.0)

        # Direction bonus — going lower is safer
        direction = 1.0
        if target < current:
            direction = 0.9 + 0.1  # full mark
        elif target > current:
            direction = 0.7

        raw = risk_ok * authority_factor * direction
        return round(max(0.0, min(1.0, raw)), 4)

    def check_cooldown(
        self, last_transition: Optional[float]
    ) -> float:
        """Return seconds remaining in cooldown, or 0 if clear."""
        if last_transition is None:
            return 0.0
        elapsed = time.time() - last_transition
        remaining = self.policy.cooldown_seconds - elapsed
        return max(0.0, remaining)

    def get_transition_history(self) -> List[Dict[str, object]]:
        """Return a plain-dict snapshot of the history."""
        return [
            {
                "from_level": r.from_level,
                "to_level": r.to_level,
                "reason": r.reason,
                "timestamp": r.timestamp,
                "approved": r.approved,
            }
            for r in self._history
        ]
