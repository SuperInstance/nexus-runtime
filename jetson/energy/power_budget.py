"""Power budget allocation for marine robotics energy management."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


class ConsumerPriority(IntEnum):
    """Priority levels for power consumers. Higher value = higher priority."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class PowerConsumer:
    """A device or subsystem that consumes power."""
    name: str
    nominal_power_w: float
    priority: int = ConsumerPriority.MEDIUM
    can_throttle: bool = False
    min_power_w: float = 0.0

    def __post_init__(self):
        if self.min_power_w > self.nominal_power_w:
            self.min_power_w = self.nominal_power_w
        if self.min_power_w < 0:
            self.min_power_w = 0.0

    @property
    def throttle_range(self) -> float:
        """Watts that can be saved by throttling to minimum."""
        return max(0.0, self.nominal_power_w - self.min_power_w)


@dataclass
class PowerBudget:
    """Overall power budget for the system."""
    total_available: float
    allocated: float = 0.0
    consumers: List[PowerConsumer] = field(default_factory=list)
    reserve: float = 0.0

    @property
    def remaining(self) -> float:
        """Unallocated power remaining."""
        return max(0.0, self.total_available - self.allocated - self.reserve)

    @property
    def utilization_percent(self) -> float:
        """Percentage of total power that is allocated."""
        if self.total_available <= 0:
            return 0.0
        return min(100.0, (self.allocated / self.total_available) * 100.0)

    @property
    def is_over_budget(self) -> bool:
        """True if allocation exceeds available power."""
        return self.allocated + self.reserve > self.total_available


class PowerAllocator:
    """Manages power allocation across consumers."""

    @staticmethod
    def allocate(
        budget: PowerBudget,
        consumers: List[PowerConsumer],
    ) -> Dict[str, float]:
        """Simple round-robin allocation: each consumer gets its nominal power
        until the budget runs out.

        Returns a mapping of consumer name → allocated watts.
        """
        allocation: Dict[str, float] = {}
        remaining = budget.total_available - budget.reserve

        for consumer in consumers:
            if remaining <= 0:
                allocation[consumer.name] = 0.0
                continue
            allocated = min(consumer.nominal_power_w, remaining)
            allocation[consumer.name] = allocated
            remaining -= allocated

        return allocation

    @staticmethod
    def allocate_priority(
        budget: PowerBudget,
        consumers: List[PowerConsumer],
    ) -> Dict[str, float]:
        """Priority-based allocation. Higher-priority consumers are served first.
        Lower-priority consumers may be throttled or receive zero power.
        """
        sorted_consumers = sorted(consumers, key=lambda c: c.priority, reverse=True)
        allocation: Dict[str, float] = {}
        remaining = budget.total_available - budget.reserve

        for consumer in sorted_consumers:
            if remaining <= 0:
                allocation[consumer.name] = 0.0
                continue
            if consumer.can_throttle:
                allocated = min(consumer.nominal_power_w, remaining)
                allocated = max(consumer.min_power_w, allocated)
            else:
                if remaining < consumer.nominal_power_w:
                    allocation[consumer.name] = 0.0
                    continue
                allocated = consumer.nominal_power_w
            allocation[consumer.name] = allocated
            remaining -= allocated

        return allocation

    @staticmethod
    def reallocate(
        budget: PowerBudget,
        consumers: List[PowerConsumer],
        increased_demand: Dict[str, float],
    ) -> Dict[str, float]:
        """Reallocate power after some consumers request more.

        *increased_demand* maps consumer name → additional watts requested.
        Priority-based: lower-priority consumers are throttled first.
        """
        current = PowerAllocator.allocate_priority(budget, consumers)
        extra_needed = sum(increased_demand.values())
        available_headroom = budget.total_available - budget.reserve - sum(current.values())

        # How much can we actually grant?
        grant = min(extra_needed, available_headroom)

        # Build a demand dict with absolute new targets
        new_allocation = dict(current)
        for name, extra in increased_demand.items():
            if grant <= 0:
                break
            actual_extra = min(extra, grant)
            new_allocation[name] = new_allocation.get(name, 0.0) + actual_extra
            grant -= actual_extra

        # If still over budget, trim lowest priority first
        total = sum(new_allocation.values())
        if total > budget.total_available - budget.reserve:
            overshoot = total - (budget.total_available - budget.reserve)
            sorted_names = sorted(
                new_allocation.keys(),
                key=lambda n: next(
                    (c.priority for c in consumers if c.name == n), 0
                ),
            )
            for name in sorted_names:
                if overshoot <= 0:
                    break
                consumer = next((c for c in consumers if c.name == name), None)
                if consumer is None:
                    continue
                reducible = new_allocation[name] - consumer.min_power_w
                trim = min(reducible, overshoot)
                new_allocation[name] -= trim
                overshoot -= trim

        return new_allocation

    @staticmethod
    def compute_total_consumption(
        consumers: List[PowerConsumer],
        allocations: Dict[str, float],
    ) -> float:
        """Sum of allocated watts for listed consumers."""
        total = 0.0
        for c in consumers:
            total += allocations.get(c.name, 0.0)
        return total

    @staticmethod
    def compute_reserve(
        budget: PowerBudget,
        allocation: Dict[str, float],
    ) -> float:
        """Compute reserve watts after allocation."""
        return max(0.0, budget.total_available - sum(allocation.values()))

    @staticmethod
    def simulate_power_profile(
        consumers: List[PowerConsumer],
        allocations: Dict[str, float],
        time_steps: int,
    ) -> List[float]:
        """Simulate a flat power profile over *time_steps* intervals.

        Returns a list of total watts at each step (all identical for a flat profile).
        """
        total = PowerAllocator.compute_total_consumption(consumers, allocations)
        return [total] * time_steps
