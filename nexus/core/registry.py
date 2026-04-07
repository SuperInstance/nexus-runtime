"""
NEXUS Service Registry — module discovery, dependency injection, and lifecycle hooks.

Provides a central registry where NEXUS modules can register themselves,
declare dependencies, and be discovered by other components.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Service states
# ---------------------------------------------------------------------------

class ServiceState(enum.Enum):
    """Lifecycle states for registered services."""

    REGISTERED = "registered"
    INITIALIZING = "initializing"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Service info
# ---------------------------------------------------------------------------

@dataclass
class ServiceInfo:
    """Metadata for a registered service."""

    name: str
    version: str = "0.0.0"
    description: str = ""
    service_type: str = ""
    dependencies: List[str] = field(default_factory=list)
    state: ServiceState = ServiceState.REGISTERED
    instance: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)
    ready_at: Optional[float] = None
    provides: List[str] = field(default_factory=list)  # capability names

    def __repr__(self) -> str:
        return f"ServiceInfo({self.name}@{self.version}, state={self.state.value})"


# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------

InitHook = Callable[["ServiceInfo"], Any]
ShutdownHook = Callable[["ServiceInfo"], None]


# ---------------------------------------------------------------------------
# Service Registry
# ---------------------------------------------------------------------------

class ServiceRegistry:
    """Central registry for NEXUS module discovery and dependency injection.

    Usage::

        registry = ServiceRegistry()
        registry.register("vm", executor=Executor(), version="0.1.0")
        registry.register("trust", engine=TrustEngine(), depends_on=["vm"])
        registry.initialize_all()
        vm = registry.get("vm")
    """

    def __init__(self) -> None:
        self._services: Dict[str, ServiceInfo] = {}
        self._init_hooks: Dict[str, List[InitHook]] = {}
        self._shutdown_hooks: Dict[str, List[ShutdownHook]] = {}
        self._init_order: List[str] = []

    @property
    def service_count(self) -> int:
        return len(self._services)

    @property
    def services(self) -> Dict[str, ServiceInfo]:
        return dict(self._services)

    # ----- registration -----

    def register(
        self,
        name: str,
        instance: Any = None,
        version: str = "0.0.0",
        description: str = "",
        service_type: str = "",
        depends_on: Optional[List[str]] = None,
        provides: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ServiceInfo:
        """Register a service."""
        if name in self._services:
            raise ValueError(f"Service '{name}' is already registered")

        info = ServiceInfo(
            name=name,
            version=version,
            description=description,
            service_type=service_type,
            dependencies=depends_on or [],
            instance=instance,
            provides=provides or [],
            metadata=metadata or {},
        )
        self._services[name] = info
        return info

    def unregister(self, name: str) -> bool:
        """Unregister a service. Returns True if it existed."""
        if name in self._services:
            self._init_hooks.pop(name, None)
            self._shutdown_hooks.pop(name, None)
            del self._services[name]
            return True
        return False

    def get(self, name: str) -> Optional[Any]:
        """Get a service instance by name."""
        info = self._services.get(name)
        return info.instance if info else None

    def get_info(self, name: str) -> Optional[ServiceInfo]:
        """Get service metadata by name."""
        return self._services.get(name)

    def list_services(self) -> List[ServiceInfo]:
        """List all registered services."""
        return list(self._services.values())

    def list_ready(self) -> List[ServiceInfo]:
        """List services that are ready."""
        return [s for s in self._services.values() if s.state == ServiceState.READY]

    # ----- dependency injection -----

    def inject(self, name: str, cls: Type[T]) -> Optional[T]:
        """Get a service instance cast to a specific type."""
        instance = self.get(name)
        if instance is None:
            return None
        if isinstance(instance, cls):
            return instance
        return None

    def get_by_capability(self, capability: str) -> List[ServiceInfo]:
        """Find services that provide a given capability."""
        return [s for s in self._services.values() if capability in s.provides]

    # ----- lifecycle -----

    def on_init(self, service_name: str, hook: InitHook) -> None:
        """Register an initialization hook for a service."""
        self._init_hooks.setdefault(service_name, []).append(hook)

    def on_shutdown(self, service_name: str, hook: ShutdownHook) -> None:
        """Register a shutdown hook for a service."""
        self._shutdown_hooks.setdefault(service_name, []).append(hook)

    def initialize(self, name: str) -> bool:
        """Initialize a single service and its dependencies."""
        return self._initialize_recursive(name, set())

    def _initialize_recursive(self, name: str, visited: set) -> bool:
        if name in visited:
            return True  # circular dependency protection
        visited.add(name)

        info = self._services.get(name)
        if info is None:
            return False

        if info.state == ServiceState.READY:
            return True

        info.state = ServiceState.INITIALIZING

        # Initialize dependencies first
        for dep in info.dependencies:
            if not self._initialize_recursive(dep, visited):
                info.state = ServiceState.FAILED
                return False

        # Run init hooks
        for hook in self._init_hooks.get(name, []):
            try:
                result = hook(info)
                if result is not None:
                    info.instance = result
            except Exception:
                info.state = ServiceState.FAILED
                return False

        info.state = ServiceState.READY
        info.ready_at = time.time()
        self._init_order.append(name)
        return True

    def initialize_all(self) -> Tuple[int, int]:
        """Initialize all registered services. Returns (success_count, fail_count)."""
        success = 0
        fail = 0
        visited = set()
        for name in list(self._services.keys()):
            if self._initialize_recursive(name, visited):
                success += 1
            else:
                fail += 1
        return (success, fail)

    def shutdown(self, name: str) -> bool:
        """Shutdown a single service."""
        info = self._services.get(name)
        if info is None or info.state == ServiceState.STOPPED:
            return False

        info.state = ServiceState.STOPPING
        for hook in self._shutdown_hooks.get(name, []):
            try:
                hook(info)
            except Exception:
                pass
        info.state = ServiceState.STOPPED
        return True

    def shutdown_all(self) -> int:
        """Shutdown all services in reverse init order. Returns count shut down."""
        count = 0
        for name in reversed(self._init_order):
            if self.shutdown(name):
                count += 1
        return count

    def get_init_order(self) -> List[str]:
        """Get the initialization order."""
        return list(self._init_order)
