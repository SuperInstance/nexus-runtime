"""Prioritised load shedding for power deficit management."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from .power_budget import PowerConsumer, PowerBudget


class LoadPriority(IntEnum):
    """Load priority levels. Higher value = higher priority (shed last)."""
    SHEDDABLE = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


@dataclass
class SheddingStrategy:
    """Configuration for load shedding behaviour."""
    priority_thresholds: Dict[LoadPriority, float] = field(default_factory=lambda: {
        LoadPriority.CRITICAL: 0.90,   # start shedding below 90% capacity
        LoadPriority.HIGH: 0.70,
        LoadPriority.MEDIUM: 0.50,
        LoadPriority.LOW: 0.30,
        LoadPriority.SHEDDABLE: 0.20,
    })
    shed_sequence: List[LoadPriority] = field(default_factory=lambda: [
        LoadPriority.SHEDDABLE,
        LoadPriority.LOW,
        LoadPriority.MEDIUM,
        LoadPriority.HIGH,
    ])
    recovery_order: List[LoadPriority] = field(default_factory=lambda: [
        LoadPriority.HIGH,
        LoadPriority.MEDIUM,
        LoadPriority.LOW,
        LoadPriority.SHEDDABLE,
    ])


@dataclass
class ShedAction:
    """A single load shedding action."""
    consumer_name: str
    priority: LoadPriority
    power_saved: float
    impact_score: float


@dataclass
class ShedReport:
    """Structured report of a shedding event."""
    deficit_watts: float
    loads_shed: List[ShedAction]
    total_power_saved: float
    operational_impact: Dict[str, str]
    remaining_deficit: float


DEFAULT_STRATEGY = SheddingStrategy()


class LoadShedManager:
    """Manages load shedding and recovery operations."""

    @staticmethod
    def evaluate_power_deficit(
        available: float,
        consumed: float,
    ) -> float:
        """Compute the power deficit in watts (0 if surplus)."""
        return max(0.0, consumed - available)

    @staticmethod
    def select_loads_to_shed(
        consumers: List[PowerConsumer],
        deficit: float,
        strategy: SheddingStrategy | None = None,
    ) -> List[ShedAction]:
        """Select loads to shed to cover the deficit.

        Follows the shed_sequence order in *strategy*, shedding lowest-priority
        loads first.  Stops when enough power is recovered.
        """
        strategy = strategy or DEFAULT_STRATEGY
        actions: List[ShedAction] = []
        remaining_deficit = deficit

        for priority_level in strategy.shed_sequence:
            if remaining_deficit <= 0:
                break
            for consumer in consumers:
                if consumer.priority != priority_level:
                    continue
                if remaining_deficit <= 0:
                    break
                power_saved = consumer.nominal_power_w
                impact = LoadShedManager._impact_score(consumer)
                actions.append(ShedAction(
                    consumer_name=consumer.name,
                    priority=LoadPriority(priority_level),
                    power_saved=power_saved,
                    impact_score=impact,
                ))
                remaining_deficit -= power_saved

        return actions

    @staticmethod
    def _impact_score(consumer: PowerConsumer) -> float:
        """Assign an impact score (0-10) for shedding a consumer."""
        base = {4: 10.0, 3: 7.0, 2: 4.0, 1: 1.0}.get(consumer.priority, 5.0)
        # Throttleable loads have lower impact since they can partially run
        if consumer.can_throttle:
            base *= 0.5
        return round(base, 2)

    @staticmethod
    def compute_shed_sequence(
        consumers: List[PowerConsumer],
        power_budget: PowerBudget,
        strategy: SheddingStrategy | None = None,
    ) -> List[ShedAction]:
        """Compute the full ordered shedding plan for the current budget.

        This is the complete plan that would be executed as power drops.
        """
        strategy = strategy or DEFAULT_STRATEGY
        total_consumed = sum(c.nominal_power_w for c in consumers)
        deficit = LoadShedManager.evaluate_power_deficit(
            power_budget.total_available - power_budget.reserve, total_consumed,
        )
        return LoadShedManager.select_loads_to_shed(consumers, deficit, strategy)

    @staticmethod
    def recover_loads(
        consumers: List[PowerConsumer],
        available_power: float,
        shed_list: List[ShedAction],
        strategy: SheddingStrategy | None = None,
    ) -> List[ShedAction]:
        """Determine which shed loads can be recovered given available power.

        Follows recovery_order from the strategy.
        """
        strategy = strategy or DEFAULT_STRATEGY
        recovered: List[ShedAction] = []
        budget = available_power

        # Sort shed_list by recovery priority (highest priority first)
        priority_order = {p: i for i, p in enumerate(strategy.recovery_order)}
        sorted_shed = sorted(
            shed_list,
            key=lambda a: priority_order.get(a.priority, 99),
        )

        for action in sorted_shed:
            if budget >= action.power_saved:
                recovered.append(action)
                budget -= action.power_saved

        return recovered

    @staticmethod
    def compute_impact(
        shed_consumers: List[ShedAction],
    ) -> Dict[str, str]:
        """Assess operational impact of shedding specific loads.

        Returns a mapping of consumer name → impact description.
        """
        impacts: Dict[str, str] = {}
        for action in shed_consumers:
            if action.impact_score >= 9.0:
                desc = "CRITICAL: Major mission impact"
            elif action.impact_score >= 6.0:
                desc = "HIGH: Significant capability reduction"
            elif action.impact_score >= 3.0:
                desc = "MEDIUM: Partial capability reduction"
            elif action.impact_score >= 1.0:
                desc = "LOW: Minor capability reduction"
            else:
                desc = "NEGLIGIBLE: No meaningful impact"
            impacts[action.consumer_name] = desc
        return impacts

    @staticmethod
    def generate_shed_report(
        consumers: List[PowerConsumer],
        shed_list: List[ShedAction],
    ) -> ShedReport:
        """Generate a structured report of a shedding event."""
        total_saved = sum(a.power_saved for a in shed_list)
        total_consumed = sum(c.nominal_power_w for c in consumers)
        total_nominal = sum(c.nominal_power_w for c in consumers if
                           c.name not in {a.consumer_name for a in shed_list})
        deficit = total_consumed - total_nominal
        remaining = max(0.0, deficit - total_saved)

        impact = LoadShedManager.compute_impact(shed_list)

        return ShedReport(
            deficit_watts=round(total_consumed, 2),
            loads_shed=shed_list,
            total_power_saved=round(total_saved, 2),
            operational_impact=impact,
            remaining_deficit=round(remaining, 2),
        )
