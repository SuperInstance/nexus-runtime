"""
Base class for all fleet sync solutions.
Provides the interface that simulation tests use.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from ..types import FleetState, TaskItem, SkillVersion, SyncMetrics
import time


class FleetSyncBase(ABC):
    """Abstract base class for fleet synchronization solutions."""

    def __init__(self, vessel_id: str):
        self.vessel_id = vessel_id
        self.metrics = SyncMetrics()
        self._operation_count = 0

    @abstractmethod
    def get_state(self) -> FleetState:
        """Return the current fleet state."""
        ...

    @abstractmethod
    def apply_change(self, change_type: str, **kwargs) -> None:
        """Apply a state change while offline."""
        ...

    @abstractmethod
    def get_sync_payload(self) -> Dict[str, Any]:
        """Generate the data to send to another vessel during sync."""
        ...

    @abstractmethod
    def receive_sync(self, payload: Dict[str, Any], from_vessel_id: str) -> int:
        """Receive and merge sync data from another vessel.
        Returns the number of conflicts resolved."""
        ...

    def track_operation(self):
        self._operation_count += 1
        self.metrics.total_operations_processed = self._operation_count

    # Convenience methods for applying specific changes

    def update_trust(self, target_vessel: str, delta: float) -> None:
        """Adjust trust score for a target vessel."""
        self.apply_change("trust_update", target_vessel=target_vessel, delta=delta)

    def add_task(self, task_id: str, description: str, priority: int = 5) -> None:
        """Add a new task to the queue."""
        self.apply_change("task_add", task_id=task_id, description=description, priority=priority)

    def update_task(self, task_id: str, status: str = None, priority: int = None) -> None:
        """Update an existing task."""
        self.apply_change("task_update", task_id=task_id, status=status, priority=priority)

    def update_vessel_status(self, target_vessel: str, key: str, value: Any) -> None:
        """Update a vessel status key-value pair."""
        self.apply_change("status_update", target_vessel=target_vessel, key=key, value=value)

    def update_skill_version(self, skill_name: str, version_str: str) -> None:
        """Update a skill version."""
        self.apply_change("skill_update", skill_name=skill_name, version_str=version_str)

    def get_memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        import sys
        return sys.getsizeof(self.get_state())
