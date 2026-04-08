"""
NEXUS Node — lifecycle management for marine robotics nodes.

Node states:
    INIT → CONNECTING → ACTIVE → DEGRADED → RECOVERY → SHUTDOWN

Each node monitors its own health, manages transitions, and provides
hooks for lifecycle events.
"""

from __future__ import annotations

import enum
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node states
# ---------------------------------------------------------------------------

class NodeState(enum.Enum):
    """Lifecycle states for a NEXUS node."""

    INIT = "init"
    CONNECTING = "connecting"
    ACTIVE = "active"
    DEGRADED = "degraded"
    RECOVERY = "recovery"
    SHUTDOWN = "shutdown"


# Valid transitions
_VALID_TRANSITIONS: Dict[NodeState, Set[NodeState]] = {
    NodeState.INIT: {NodeState.CONNECTING, NodeState.SHUTDOWN},
    NodeState.CONNECTING: {NodeState.ACTIVE, NodeState.DEGRADED, NodeState.SHUTDOWN, NodeState.INIT},
    NodeState.ACTIVE: {NodeState.DEGRADED, NodeState.SHUTDOWN, NodeState.RECOVERY},
    NodeState.DEGRADED: {NodeState.RECOVERY, NodeState.SHUTDOWN, NodeState.ACTIVE},
    NodeState.RECOVERY: {NodeState.ACTIVE, NodeState.DEGRADED, NodeState.SHUTDOWN},
    NodeState.SHUTDOWN: set(),  # terminal
}


# ---------------------------------------------------------------------------
# Health status
# ---------------------------------------------------------------------------

class HealthStatus(str, enum.Enum):
    """Node health status."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthMetric:
    """A single health metric."""

    name: str
    value: float
    unit: str = ""
    min_ok: float = 0.0
    max_ok: float = 100.0
    timestamp: float = field(default_factory=time.time)

    @property
    def is_healthy(self) -> bool:
        return self.min_ok <= self.value <= self.max_ok

    def __repr__(self) -> str:
        return f"HealthMetric({self.name}={self.value}{self.unit})"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

LifecycleHook = Callable[["Node", NodeState, NodeState], None]


class Node:
    """NEXUS node with lifecycle management and health monitoring.

    Usage::

        node = Node(node_id="AUV-001", name="Port Scanner")
        node.on_transition(lambda n, old, new: print(f"{old} → {new}"))
        node.start()
        node.add_health_metric(HealthMetric("battery", 85, "%", 20, 100))
        node.report_degraded()
    """

    def __init__(
        self,
        node_id: str = "",
        name: str = "",
        capabilities: Optional[Dict[str, float]] = None,
    ) -> None:
        self.node_id: str = node_id or str(uuid.uuid4())[:8]
        self.name: str = name or self.node_id
        self.capabilities: Dict[str, float] = capabilities or {}
        self.state: NodeState = NodeState.INIT
        self.health: HealthStatus = HealthStatus.UNKNOWN
        self._health_metrics: Dict[str, HealthMetric] = {}
        self._transition_hooks: List[LifecycleHook] = []
        self._state_history: List[tuple] = [(NodeState.INIT, time.time())]
        self._created_at: float = time.time()
        self._metadata: Dict[str, Any] = {}
        self._error_count: int = 0
        self._last_error: str = ""

    @property
    def uptime(self) -> float:
        """Seconds since node creation."""
        return time.time() - self._created_at

    @property
    def is_active(self) -> bool:
        return self.state == NodeState.ACTIVE

    @property
    def health_metrics(self) -> Dict[str, HealthMetric]:
        return dict(self._health_metrics)

    # ----- lifecycle -----

    def start(self) -> bool:
        """Transition from INIT → CONNECTING → ACTIVE."""
        if self.state != NodeState.INIT:
            return False
        self._transition(NodeState.CONNECTING)
        self._transition(NodeState.ACTIVE)
        self.health = HealthStatus.HEALTHY
        return True

    def shutdown(self) -> bool:
        """Transition to SHUTDOWN (terminal state)."""
        if self.state == NodeState.SHUTDOWN:
            return False
        return self._transition(NodeState.SHUTDOWN)

    def report_degraded(self, reason: str = "") -> bool:
        """Transition to DEGRADED state."""
        if reason:
            self._last_error = reason
        return self._transition(NodeState.DEGRADED)

    def report_recovery(self) -> bool:
        """Transition to RECOVERY state."""
        return self._transition(NodeState.RECOVERY)

    def recover(self) -> bool:
        """Attempt recovery → ACTIVE."""
        if self.state != NodeState.RECOVERY:
            return False
        ok = self._transition(NodeState.ACTIVE)
        if ok:
            self.health = HealthStatus.HEALTHY
            self._error_count = 0
        return ok

    def _transition(self, new_state: NodeState) -> bool:
        """Execute a state transition with validation and hooks."""
        if self.state == new_state:
            return True

        allowed = _VALID_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            return False

        old_state = self.state
        self.state = new_state
        self._state_history.append((new_state, time.time()))

        for hook in self._transition_hooks:
            try:
                hook(self, old_state, new_state)
            except Exception:
                logger.exception("Transition hook failed during %s → %s", old_state.value, new_state.value)

        return True

    def on_transition(self, hook: LifecycleHook) -> None:
        """Register a transition callback."""
        self._transition_hooks.append(hook)

    def get_state_history(self) -> List[tuple]:
        return list(self._state_history)

    # ----- health -----

    def add_health_metric(self, metric: HealthMetric) -> None:
        """Add or update a health metric."""
        self._health_metrics[metric.name] = metric
        self._update_health_status()

    def remove_health_metric(self, name: str) -> bool:
        """Remove a health metric. Returns True if it existed."""
        if name in self._health_metrics:
            del self._health_metrics[name]
            self._update_health_status()
            return True
        return False

    def get_health_metric(self, name: str) -> Optional[HealthMetric]:
        return self._health_metrics.get(name)

    def _update_health_status(self) -> None:
        """Recompute overall health from metrics."""
        if not self._health_metrics:
            return

        any_critical = any(not m.is_healthy for m in self._health_metrics.values())
        if any_critical:
            self.health = HealthStatus.WARNING
            self._error_count += 1
        else:
            self.health = HealthStatus.HEALTHY

    def record_error(self, message: str) -> None:
        """Record an error event."""
        self._error_count += 1
        self._last_error = message

    @property
    def error_count(self) -> int:
        return self._error_count

    @property
    def last_error(self) -> str:
        return self._last_error

    def reset(self) -> None:
        """Reset node to INIT state."""
        self.state = NodeState.INIT
        self.health = HealthStatus.UNKNOWN
        self._health_metrics.clear()
        self._state_history = [(NodeState.INIT, time.time())]
        self._error_count = 0
        self._last_error = ""
