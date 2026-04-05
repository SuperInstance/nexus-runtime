"""NEXUS Safety Policy Engine — YAML-like policies as code.

Policies define conditions and actions for safety decisions.
Multiple policies merge with priority ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PolicyAction(Enum):
    ALLOW = "allow"
    DENY = "deny"
    WARN = "warn"
    REQUIRE_REVIEW = "require_review"


@dataclass
class PolicyCondition:
    """A single condition in a safety policy."""
    field: str  # e.g. "trust_level", "opcode", "pin"
    operator: str  # eq, ne, gt, gte, lt, lte, in, not_in, contains
    value: Any

    def evaluate(self, context: dict[str, Any]) -> bool:
        actual = context.get(self.field)
        if actual is None:
            return False
        ops = {
            "eq": lambda a, b: a == b,
            "ne": lambda a, b: a != b,
            "gt": lambda a, b: a > b,
            "gte": lambda a, b: a >= b,
            "lt": lambda a, b: a < b,
            "lte": lambda a, b: a <= b,
            "in": lambda a, b: a in b if isinstance(b, (list, tuple, set)) else False,
            "not_in": lambda a, b: a not in b if isinstance(b, (list, tuple, set)) else True,
            "contains": lambda a, b: b in a if isinstance(a, (list, tuple, str)) else False,
        }
        op_fn = ops.get(self.operator)
        if op_fn is None:
            return False
        try:
            return op_fn(actual, self.value)
        except (TypeError, ValueError):
            return False

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "operator": self.operator, "value": self.value}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PolicyCondition:
        return cls(field=d["field"], operator=d.get("operator", "eq"), value=d.get("value"))


@dataclass
class SafetyPolicy:
    """A safety policy with conditions and an action."""
    name: str
    description: str = ""
    conditions: list[PolicyCondition] = field(default_factory=list)
    action: PolicyAction = PolicyAction.DENY
    priority: int = 0  # higher = evaluated first
    enabled: bool = True
    tags: list[str] = field(default_factory=list)

    def evaluate(self, context: dict[str, Any]) -> PolicyAction | None:
        """Evaluate all conditions. Returns action if all match, None otherwise."""
        if not self.enabled:
            return None
        if not self.conditions:
            return self.action
        if all(c.evaluate(context) for c in self.conditions):
            return self.action
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "conditions": [c.to_dict() for c in self.conditions],
            "action": self.action.value, "priority": self.priority,
            "enabled": self.enabled, "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SafetyPolicy:
        return cls(
            name=d["name"], description=d.get("description", ""),
            conditions=[PolicyCondition.from_dict(c) for c in d.get("conditions", [])],
            action=PolicyAction(d.get("action", "deny")),
            priority=d.get("priority", 0),
            enabled=d.get("enabled", True),
            tags=d.get("tags", []),
        )


class PolicyEngine:
    """Manages and evaluates safety policies."""

    def __init__(self) -> None:
        self._policies: list[SafetyPolicy] = []

    def add_policy(self, policy: SafetyPolicy) -> None:
        self._policies.append(policy)
        self._policies.sort(key=lambda p: p.priority, reverse=True)

    def remove_policy(self, name: str) -> bool:
        before = len(self._policies)
        self._policies = [p for p in self._policies if p.name != name]
        return len(self._policies) < before

    def evaluate(self, context: dict[str, Any]) -> PolicyAction:
        """Evaluate all policies against context. Returns first matching action."""
        for policy in self._policies:
            action = policy.evaluate(context)
            if action is not None:
                return action
        return PolicyAction.ALLOW  # default: allow

    def evaluate_all(self, context: dict[str, Any]) -> list[tuple[str, PolicyAction]]:
        """Evaluate all policies, return all matches."""
        results = []
        for policy in self._policies:
            action = policy.evaluate(context)
            if action is not None:
                results.append((policy.name, action))
        return results

    def get_policy(self, name: str) -> SafetyPolicy | None:
        for p in self._policies:
            if p.name == name:
                return p
        return None

    def enable_policy(self, name: str) -> bool:
        p = self.get_policy(name)
        if p:
            p.enabled = True
            return True
        return False

    def disable_policy(self, name: str) -> bool:
        p = self.get_policy(name)
        if p:
            p.enabled = False
            return True
        return False

    @property
    def policy_count(self) -> int:
        return len(self._policies)

    @property
    def enabled_count(self) -> int:
        return sum(1 for p in self._policies if p.enabled)

    def load_defaults(self) -> None:
        """Load default NEXUS safety policies."""
        defaults = [
            SafetyPolicy("trust_minimum", "Require minimum trust for reflex deployment",
                         [PolicyCondition("trust_level", "lt", 2),
                          PolicyCondition("action_type", "eq", "deploy_reflex")],
                         PolicyAction.DENY, priority=100),
            SafetyPolicy("safety_pin_protection", "Block writes to safety-critical pins",
                         [PolicyCondition("target_pin", "in", [0, 1, 2, 3]),
                          PolicyCondition("action_type", "eq", "write_pin")],
                         PolicyAction.DENY, priority=90),
            SafetyPolicy("fault_state_restriction", "Restrict actions in fault state",
                         [PolicyCondition("safety_state", "in", ["fault", "safe_state"]),
                          PolicyCondition("action_type", "not_in", ["emergency_stop", "diagnostic", "reset"])],
                         PolicyAction.DENY, priority=95),
            SafetyPolicy("speed_limit", "Warn on high speed in restricted areas",
                         [PolicyCondition("speed_knots", "gt", 5),
                          PolicyCondition("proximity_m", "lt", 50)],
                         PolicyAction.WARN, priority=50),
            SafetyPolicy("review_required", "Require review for high-trust actions",
                         [PolicyCondition("trust_level", "lt", 4),
                          PolicyCondition("action_type", "eq", "override_safety")],
                         PolicyAction.REQUIRE_REVIEW, priority=80),
        ]
        for p in defaults:
            self.add_policy(p)
