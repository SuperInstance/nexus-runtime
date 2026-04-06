"""
Watchdog timer and timeout management for NEXUS runtime.

Monitors component heartbeats, detects missed beats, manages
escalation actions, and computes uptime metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class WatchdogConfig:
    """Configuration for a component's watchdog."""

    timeout_ms: int = 5000
    heartbeat_interval_ms: int = 1000
    max_missed_beats: int = 3
    escalation_actions: List[str] = field(default_factory=lambda: ["log", "alert"])


@dataclass
class WatchdogState:
    """Current state of a component's watchdog."""

    active: bool = True
    last_heartbeat: float = 0.0
    missed_beats: int = 0
    escalated: bool = False


@dataclass
class WatchdogSummary:
    """Summary of a single component watchdog."""

    component_name: str
    active: bool
    missed_beats: int
    escalated: bool
    last_heartbeat: float
    uptime_percentage: float


class WatchdogManager:
    """Manages watchdog timers for multiple components."""

    def __init__(self) -> None:
        self._configs: Dict[str, WatchdogConfig] = {}
        self._states: Dict[str, WatchdogState] = {}
        self._custom_handlers: Dict[str, Callable[[str, WatchdogState], None]] = {}
        self._heartbeat_log: Dict[str, List[float]] = {}
        self._start_times: Dict[str, float] = {}

    def register(self, component_name: str, config: WatchdogConfig) -> None:
        """Register a component with its watchdog configuration."""
        self._configs[component_name] = config
        self._states[component_name] = WatchdogState(
            active=True,
            last_heartbeat=time.time(),
            missed_beats=0,
            escalated=False,
        )
        self._heartbeat_log[component_name] = []
        self._start_times[component_name] = time.time()

    def feed_heartbeat(self, component_name: str) -> None:
        """Record a heartbeat for a component, resetting its missed-beat counter."""
        now = time.time()
        state = self._states.get(component_name)
        if state is None:
            return
        state.last_heartbeat = now
        state.missed_beats = 0
        if state.escalated:
            state.escalated = False
        self._heartbeat_log[component_name].append(now)

    def check_all(self) -> List[str]:
        """Check all registered components. Returns list of expired component names."""
        now = time.time()
        expired: List[str] = []
        for name, config in self._configs.items():
            state = self._states.get(name)
            if state is None or not state.active:
                continue
            timeout_sec = config.timeout_ms / 1000.0
            elapsed = now - state.last_heartbeat
            if elapsed > timeout_sec:
                if name not in expired:
                    expired.append(name)
                    state.missed_beats += 1
                    if state.missed_beats >= config.max_missed_beats and not state.escalated:
                        state.escalated = True
                        # Fire custom handler if set
                        handler = self._custom_handlers.get(name)
                        if handler:
                            handler(name, state)
        return expired

    def get_state(self, component_name: str) -> Optional[WatchdogState]:
        """Get the current watchdog state for a component."""
        return self._states.get(component_name)

    def reset(self, component_name: str) -> bool:
        """Reset a component's watchdog state. Returns True if component exists."""
        state = self._states.get(component_name)
        if state is None:
            return False
        state.last_heartbeat = time.time()
        state.missed_beats = 0
        state.escalated = False
        state.active = True
        self._start_times[component_name] = time.time()
        return True

    def compute_uptime(self, component_name: str, period: float = 60.0) -> float:
        """Compute uptime percentage for a component over a period.

        Args:
            component_name: Name of the component.
            period: Time window in seconds.

        Returns:
            Percentage between 0.0 and 100.0.
        """
        now = time.time()
        start_time = self._start_times.get(component_name, now)
        elapsed = now - start_time
        if elapsed <= 0:
            return 100.0
        # Count heartbeats within the period
        heartbeats = self._heartbeat_log.get(component_name, [])
        recent = [t for t in heartbeats if (now - t) <= period]
        # Estimate uptime: if there are heartbeats, the component was alive
        # Simple heuristic: ratio of active time
        if not heartbeats:
            return 0.0
        # Check for escalation events as downtime indicators
        state = self._states.get(component_name)
        if state and state.escalated:
            # Reduce uptime estimate based on missed beats
            config = self._configs.get(component_name)
            if config:
                downtime_ratio = min(state.missed_beats / config.max_missed_beats, 1.0)
                uptime = max(100.0 - (downtime_ratio * 100.0), 0.0)
                return uptime
        return 100.0

    def set_custom_handler(
        self, component_name: str, handler_fn: Callable[[str, WatchdogState], None]
    ) -> None:
        """Set a custom escalation handler for a component."""
        self._custom_handlers[component_name] = handler_fn

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all registered watchdogs."""
        summary: Dict[str, Any] = {
            "total_components": len(self._configs),
            "components": {},
        }
        for name, config in self._configs.items():
            state = self._states.get(name)
            uptime = self.compute_uptime(name)
            summary["components"][name] = {
                "active": state.active if state else False,
                "missed_beats": state.missed_beats if state else 0,
                "escalated": state.escalated if state else False,
                "last_heartbeat": state.last_heartbeat if state else 0.0,
                "uptime_percentage": uptime,
                "config": {
                    "timeout_ms": config.timeout_ms,
                    "heartbeat_interval_ms": config.heartbeat_interval_ms,
                    "max_missed_beats": config.max_missed_beats,
                    "escalation_actions": config.escalation_actions,
                },
            }
        return summary
