"""Economic model for cost estimation, revenue, and profit sharing."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CostEstimate:
    """Breakdown of estimated costs for a task."""
    fuel: float = 0.0
    labor: float = 0.0
    equipment_depreciation: float = 0.0
    insurance: float = 0.0
    contingency: float = 0.0
    total: float = 0.0


@dataclass
class RevenueModel:
    """Revenue computation model."""
    base_rate: float = 0.0
    time_multiplier: float = 1.0
    complexity_multiplier: float = 1.0
    risk_adjustment: float = 0.0


@dataclass
class ProfitResult:
    """Result of profit computation."""
    profit: float = 0.0
    margin: float = 0.0  # percentage


@dataclass
class ShareAllocation:
    """Per-participant share of profit."""
    participant_id: str = ""
    amount: float = 0.0
    percentage: float = 0.0


class EconomicModel:
    """Economic model for fleet marketplace operations."""

    def estimate_task_cost(
        self,
        task: Any,
        vessel: Any,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> CostEstimate:
        """Estimate the total cost of executing a task."""
        if conditions is None:
            conditions = {}

        # Fuel cost: based on duration and vessel fuel consumption
        duration_hours = getattr(task, "estimated_duration", 4.0)
        if hasattr(task, "requirements") and isinstance(task.requirements, dict):
            duration_hours = task.requirements.get("estimated_duration", duration_hours)

        fuel_rate = getattr(vessel, "fuel_rate", 50.0)  # $/hr default
        if hasattr(vessel, "hourly_cost"):
            fuel_rate = vessel.hourly_cost * 0.4  # fuel ~ 40% of hourly cost
        fuel_cost = fuel_rate * duration_hours

        # Weather factor
        weather_factor = conditions.get("weather_factor", 1.0)
        fuel_cost *= weather_factor

        # Labor cost
        labor_rate = conditions.get("labor_rate", 75.0)  # $/hr
        labor_cost = labor_rate * duration_hours

        # Equipment depreciation
        vessel_value = getattr(vessel, "value", 500000.0)
        depreciation = self.compute_depreciation(vessel_value, useful_life=10.0, age=2.0)
        daily_depreciation = depreciation / 365.0
        equipment_cost = daily_depreciation * duration_hours / 24.0

        # Insurance
        risk_score = getattr(vessel, "risk_score", 0.3)
        risk_score = conditions.get("risk_score", risk_score)
        coverage = vessel_value
        insurance_cost = self.compute_insurance_premium(risk_score, coverage) / 365.0 * (duration_hours / 24.0)

        # Contingency (10% of subtotal)
        subtotal = fuel_cost + labor_cost + equipment_cost + insurance_cost
        contingency = subtotal * 0.10

        total = subtotal + contingency
        return CostEstimate(
            fuel=round(fuel_cost, 2),
            labor=round(labor_cost, 2),
            equipment_depreciation=round(equipment_cost, 2),
            insurance=round(insurance_cost, 2),
            contingency=round(contingency, 2),
            total=round(total, 2),
        )

    def compute_revenue(self, task: Any, model: RevenueModel) -> float:
        """Compute expected revenue for a task."""
        base = model.base_rate
        if base == 0.0:
            base = getattr(task, "reward", 0.0)
        revenue = base * model.time_multiplier * model.complexity_multiplier
        revenue -= model.risk_adjustment
        return max(0.0, round(revenue, 2))

    def compute_profit(self, cost: CostEstimate, revenue: float) -> ProfitResult:
        """Compute profit and margin."""
        total_cost = cost.total
        profit = revenue - total_cost
        margin = (profit / revenue * 100.0) if revenue > 0 else (0.0 if profit <= 0 else 100.0)
        return ProfitResult(profit=round(profit, 2), margin=round(margin, 2))

    def compute_profit_sharing(
        self,
        participants: List[str],
        total_profit: float,
        shares: Optional[List[float]] = None,
    ) -> List[ShareAllocation]:
        """Distribute profit among participants based on shares."""
        n = len(participants)
        if n == 0:
            return []

        if shares is None:
            # Equal distribution
            per_person = total_profit / n
            return [
                ShareAllocation(
                    participant_id=pid,
                    amount=round(per_person, 2),
                    percentage=round(100.0 / n, 2),
                )
                for pid in participants
            ]

        if len(shares) != n:
            raise ValueError("shares list must match participants list length")

        total_shares = sum(shares)
        if total_shares == 0:
            return [
                ShareAllocation(participant_id=pid, amount=0.0, percentage=0.0)
                for pid in participants
            ]

        allocations = []
        for pid, s in zip(participants, shares):
            pct = s / total_shares
            amount = total_profit * pct
            allocations.append(ShareAllocation(
                participant_id=pid,
                amount=round(amount, 2),
                percentage=round(pct * 100.0, 2),
            ))
        return allocations

    def compute_depreciation(self, vessel_value: float, useful_life: float, age: float) -> float:
        """Compute straight-line annual depreciation."""
        if useful_life <= 0:
            return 0.0
        return vessel_value / useful_life

    def compute_insurance_premium(self, risk_score: float, coverage_amount: float) -> float:
        """Compute annual insurance premium based on risk and coverage."""
        # Base premium rate: 2% of coverage
        base_rate = 0.02
        # Risk adjustment: multiply by (1 + risk_score)
        premium = coverage_amount * base_rate * (1.0 + risk_score)
        return round(premium, 2)
