"""
System-wide orchestrator for managing subsystems, dependencies,
lifecycle, and overall system state.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class SubsystemStatus(str, Enum):
    """Runtime status of a subsystem."""
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class SubsystemInfo:
    """Metadata describing a subsystem."""
    name: str
    version: str = "0.0.0"
    status: SubsystemStatus = SubsystemStatus.REGISTERED
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemState:
    """Snapshot of the entire system state."""
    subsystems: Dict[str, SubsystemInfo] = field(default_factory=dict)
    overall_status: SubsystemStatus = SubsystemStatus.UNKNOWN
    uptime: float = 0.0
    mode: str = "normal"


class SystemOrchestrator:
    """Manages subsystem registration, initialization, dependencies, and lifecycle."""

    def __init__(self) -> None:
        self._subsystems: Dict[str, SubsystemInfo] = {}
        self._start_time: Optional[float] = None
        self._mode: str = "normal"
        self._init_order: List[str] = []
        self._hooks: Dict[str, List[Callable]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_subsystem(self, info: SubsystemInfo) -> None:
        """Register a subsystem by its info descriptor."""
        self._subsystems[info.name] = info

    def unregister_subsystem(self, name: str) -> bool:
        """Remove a subsystem. Returns True if it existed."""
        return self._subsystems.pop(name, None) is not None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _resolve_init_order(self) -> List[str]:
        """Topological sort of subsystems respecting dependencies."""
        order: List[str] = []
        visited: Set[str] = set()
        visiting: Set[str] = set()

        def _visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                return  # cycle guard — skip
            visiting.add(name)
            info = self._subsystems.get(name)
            if info:
                for dep in info.dependencies:
                    _visit(dep)
            visiting.discard(name)
            visited.add(name)
            order.append(name)

        for name in self._subsystems:
            _visit(name)
        return order

    def initialize_all(self) -> Tuple[bool, List[str]]:
        """Initialize all registered subsystems. Returns (success, failed_names)."""
        self._start_time = time.time()
        order = self._resolve_init_order()
        self._init_order = order
        failed: List[str] = []
        for name in order:
            info = self._subsystems.get(name)
            if info is None:
                continue
            info.status = SubsystemStatus.INITIALIZING
            # Check deps satisfied
            for dep in info.dependencies:
                dep_info = self._subsystems.get(dep)
                if dep_info is None or dep_info.status != SubsystemStatus.RUNNING:
                    info.status = SubsystemStatus.FAILED
                    failed.append(name)
                    break
            else:
                info.status = SubsystemStatus.RUNNING
        return (len(failed) == 0, failed)

    def shutdown_all(self) -> List[str]:
        """Stop all running subsystems. Returns list of stopped names."""
        stopped: List[str] = []
        # Reverse init order
        for name in reversed(self._init_order):
            info = self._subsystems.get(name)
            if info and info.status == SubsystemStatus.RUNNING:
                info.status = SubsystemStatus.STOPPED
                stopped.append(name)
        return stopped

    def start_subsystem(self, name: str) -> bool:
        """Start a single subsystem. Returns True on success."""
        info = self._subsystems.get(name)
        if info is None:
            return False
        # Check deps
        for dep in info.dependencies:
            dep_info = self._subsystems.get(dep)
            if dep_info is None or dep_info.status != SubsystemStatus.RUNNING:
                info.status = SubsystemStatus.FAILED
                return False
        info.status = SubsystemStatus.RUNNING
        return True

    def stop_subsystem(self, name: str) -> bool:
        """Stop a single subsystem. Returns True if it was running."""
        info = self._subsystems.get(name)
        if info is None or info.status != SubsystemStatus.RUNNING:
            return False
        info.status = SubsystemStatus.STOPPED
        return True

    def restart_subsystem(self, name: str) -> bool:
        """Restart a subsystem (stop then start). Returns True on success."""
        self.stop_subsystem(name)
        return self.start_subsystem(name)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_system_state(self) -> SystemState:
        """Return a snapshot of the current system state."""
        overall = SubsystemStatus.UNKNOWN
        if self._subsystems:
            statuses = {s.status for s in self._subsystems.values()}
            if statuses == {SubsystemStatus.RUNNING}:
                overall = SubsystemStatus.RUNNING
            elif SubsystemStatus.FAILED in statuses:
                overall = SubsystemStatus.FAILED
            elif statuses - {SubsystemStatus.RUNNING, SubsystemStatus.STOPPED, SubsystemStatus.REGISTERED}:
                overall = SubsystemStatus.UNKNOWN
            else:
                overall = SubsystemStatus.STOPPED
        uptime = 0.0
        if self._start_time:
            uptime = time.time() - self._start_time
        return SystemState(
            subsystems=dict(self._subsystems),
            overall_status=overall,
            uptime=uptime,
            mode=self._mode,
        )

    def check_dependencies(self) -> List[str]:
        """Return list of subsystems with unmet dependencies."""
        missing: List[str] = []
        for name, info in self._subsystems.items():
            for dep in info.dependencies:
                if dep not in self._subsystems:
                    missing.append(f"{name} requires {dep}")
        return missing

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Return adjacency-list dependency graph."""
        return {name: list(info.dependencies) for name, info in self._subsystems.items()}

    # ------------------------------------------------------------------
    # Mode & hooks
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        self._mode = mode

    def add_hook(self, event: str, callback: Callable) -> None:
        self._hooks[event].append(callback)

    def fire_hook(self, event: str, *args: Any, **kwargs: Any) -> List[Any]:
        results: List[Any] = []
        for cb in self._hooks.get(event, []):
            results.append(cb(*args, **kwargs))
        return results
