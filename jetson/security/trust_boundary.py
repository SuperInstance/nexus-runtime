"""Trust boundary enforcement for marine robotics fleet."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple


class TrustBoundary(IntEnum):
    EXTERNAL = 0
    EDGE = 1
    AGENT = 2
    FLEET = 3


class AccessResult(Enum):
    ALLOWED = "allowed"
    DENIED = "denied"


class DataClass(IntEnum):
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    SECRET = 3


@dataclass
class TrustPolicy:
    source: TrustBoundary
    target: TrustBoundary
    max_data_class: DataClass = DataClass.INTERNAL
    allowed_operations: Set[str] = field(default_factory=lambda: {"read"})
    rate_limit: int = 100  # max operations per window
    window_seconds: float = 60.0


@dataclass
class AccessRecord:
    timestamp: float
    source: TrustBoundary
    target: TrustBoundary
    operation: str
    result: AccessResult
    details: str = ""


@dataclass
class RateLimitResult:
    allowed: bool
    remaining: int
    reset_time: float


class TrustBoundaryEnforcer:
    """Enforce trust policies across boundaries."""

    # Default trust matrix: higher or equal trust level allowed
    _DEFAULT_TRUST_MATRIX: Dict[Tuple[TrustBoundary, TrustBoundary], bool] = {
        (TrustBoundary.FLEET, TrustBoundary.FLEET): True,
        (TrustBoundary.FLEET, TrustBoundary.AGENT): True,
        (TrustBoundary.FLEET, TrustBoundary.EDGE): True,
        (TrustBoundary.FLEET, TrustBoundary.EXTERNAL): True,
        (TrustBoundary.AGENT, TrustBoundary.AGENT): True,
        (TrustBoundary.AGENT, TrustBoundary.EDGE): True,
        (TrustBoundary.AGENT, TrustBoundary.FLEET): False,
        (TrustBoundary.AGENT, TrustBoundary.EXTERNAL): False,
        (TrustBoundary.EDGE, TrustBoundary.EDGE): True,
        (TrustBoundary.EDGE, TrustBoundary.FLEET): False,
        (TrustBoundary.EDGE, TrustBoundary.AGENT): False,
        (TrustBoundary.EDGE, TrustBoundary.EXTERNAL): False,
        (TrustBoundary.EXTERNAL, TrustBoundary.EXTERNAL): True,
        (TrustBoundary.EXTERNAL, TrustBoundary.EDGE): False,
        (TrustBoundary.EXTERNAL, TrustBoundary.AGENT): False,
        (TrustBoundary.EXTERNAL, TrustBoundary.FLEET): False,
    }

    def __init__(self) -> None:
        self._policies: List[TrustPolicy] = []
        self._access_log: List[AccessRecord] = []
        self._rate_counters: Dict[str, List[float]] = defaultdict(list)
        self._trust_matrix = dict(self._DEFAULT_TRUST_MATRIX)

    def add_policy(self, policy: TrustPolicy) -> None:
        self._policies.append(policy)

    def set_trust_matrix(self, matrix: Dict[Tuple[TrustBoundary, TrustBoundary], bool]) -> None:
        self._trust_matrix = dict(matrix)

    def check_access(
        self,
        source_boundary: TrustBoundary,
        target_boundary: TrustBoundary,
        operation: str,
    ) -> AccessResult:
        allowed = self._trust_matrix.get((source_boundary, target_boundary), False)
        if not allowed:
            record = AccessRecord(
                timestamp=time.time(),
                source=source_boundary,
                target=target_boundary,
                operation=operation,
                result=AccessResult.DENIED,
                details="Trust matrix denied",
            )
            self._access_log.append(record)
            return AccessResult.DENIED
        # Check operation against policy
        for pol in self._policies:
            if pol.source == source_boundary and pol.target == target_boundary:
                if operation not in pol.allowed_operations:
                    record = AccessRecord(
                        timestamp=time.time(),
                        source=source_boundary,
                        target=target_boundary,
                        operation=operation,
                        result=AccessResult.DENIED,
                        details=f"Operation '{operation}' not in allowed set",
                    )
                    self._access_log.append(record)
                    return AccessResult.DENIED
        record = AccessRecord(
            timestamp=time.time(),
            source=source_boundary,
            target=target_boundary,
            operation=operation,
            result=AccessResult.ALLOWED,
        )
        self._access_log.append(record)
        return AccessResult.ALLOWED

    def validate_data(
        self,
        source: TrustBoundary,
        data: Dict[str, Any],
        policy: Optional[TrustPolicy] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {"valid": True, "errors": []}
        if policy is None:
            for pol in self._policies:
                if pol.source == source:
                    policy = pol
                    break
        if policy is not None:
            data_class = data.get("data_class", DataClass.PUBLIC)
            if isinstance(data_class, int) and data_class > policy.max_data_class:
                result["valid"] = False
                result["errors"].append(
                    f"Data class {data_class} exceeds max {policy.max_data_class}"
                )
        return result

    def enforce_rate_limit(
        self,
        source: TrustBoundary,
        operation: str,
        limit: int,
        window: float = 60.0,
    ) -> RateLimitResult:
        key = f"{source.value}:{operation}"
        now = time.time()
        cutoff = now - window
        # Prune old entries
        self._rate_counters[key] = [
            ts for ts in self._rate_counters[key] if ts > cutoff
        ]
        count = len(self._rate_counters[key])
        if count >= limit:
            oldest = self._rate_counters[key][0] if self._rate_counters[key] else now
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=oldest + window,
            )
        self._rate_counters[key].append(now)
        return RateLimitResult(
            allowed=True,
            remaining=limit - count - 1,
            reset_time=now + window,
        )

    def cross_boundary_transfer(
        self,
        source: TrustBoundary,
        target: TrustBoundary,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Transfer data across trust boundary, filtering sensitive fields."""
        access = self.check_access(source, target, "transfer")
        if access == AccessResult.DENIED:
            return {"_error": "access_denied", "_filtered": True}
        # Filter based on data classification
        filtered = {}
        sensitive_fields = {"secret", "password", "token", "key", "credential"}
        for k, v in data.items():
            if k.lower() not in sensitive_fields:
                filtered[k] = v
            elif source == target and source == TrustBoundary.FLEET:
                # Allow secrets within fleet
                filtered[k] = v
        return filtered

    def audit_access_log(self) -> List[AccessRecord]:
        return list(self._access_log)

    def clear_log(self) -> None:
        self._access_log.clear()

    def get_policies(self) -> List[TrustPolicy]:
        return list(self._policies)
