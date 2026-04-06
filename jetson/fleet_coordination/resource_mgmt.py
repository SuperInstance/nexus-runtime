"""Fleet resource management — allocation, deallocation, forecasting, rebalancing."""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class FleetResource:
    """A type of fleet-level resource."""
    type: str                         # e.g., "fuel", "bandwidth", "compute"
    total_capacity: float = 100.0
    used_capacity: float = 0.0
    vessels_sharing: List[str] = field(default_factory=list)

    @property
    def available_capacity(self) -> float:
        return max(0.0, self.total_capacity - self.used_capacity)


@dataclass
class ResourceAllocation:
    """A specific allocation of a resource to a vessel."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    resource_type: str = ""
    vessel_id: str = ""
    amount: float = 0.0
    priority: float = 0.5             # 0.0 (low) to 1.0 (critical)
    expires: Optional[float] = None   # unix timestamp


@dataclass
class ShortageForecast:
    """Forecast of a potential resource shortage."""
    resource_type: str
    current_utilization: float
    projected_utilization: float      # at forecast horizon
    shortage_risk: float              # 0.0 (safe) to 1.0 (critical)
    time_to_shortage: Optional[float]  # seconds until shortage, None if not at risk


@dataclass
class ReallocationPlan:
    """A plan for rebalancing resources."""
    transfers: List[Dict[str, Any]] = field(default_factory=list)
    total_amount: float = 0.0


class FleetResourceManager:
    """Manage fleet-wide resource pools, allocations, and demand forecasting."""

    def __init__(self) -> None:
        self._resources: Dict[str, FleetResource] = {}
        self._allocations: Dict[str, ResourceAllocation] = {}

    # ----------------------------------------------------------- CRUD
    def register_resource(self, resource: FleetResource) -> FleetResource:
        rtype = resource.type
        if rtype in self._resources:
            # Merge capacities
            existing = self._resources[rtype]
            existing.total_capacity += resource.total_capacity
            existing.used_capacity += resource.used_capacity
            for v in resource.vessels_sharing:
                if v not in existing.vessels_sharing:
                    existing.vessels_sharing.append(v)
            return existing
        self._resources[rtype] = FleetResource(
            type=rtype,
            total_capacity=resource.total_capacity,
            used_capacity=resource.used_capacity,
            vessels_sharing=list(resource.vessels_sharing),
        )
        return self._resources[rtype]

    def allocate(self, resource_type: str, amount: float,
                 vessel_id: str, priority: float = 0.5,
                 expires: Optional[float] = None) -> Optional[ResourceAllocation]:
        """Allocate resource to a vessel. Returns allocation or None if insufficient."""
        if resource_type not in self._resources:
            return None
        res = self._resources[resource_type]
        if res.available_capacity < amount:
            return None

        alloc = ResourceAllocation(
            resource_type=resource_type,
            vessel_id=vessel_id,
            amount=amount,
            priority=priority,
            expires=expires,
        )
        self._allocations[alloc.id] = alloc
        res.used_capacity += amount
        if vessel_id not in res.vessels_sharing:
            res.vessels_sharing.append(vessel_id)
        return alloc

    def deallocate(self, allocation_id: str) -> bool:
        if allocation_id not in self._allocations:
            return False
        alloc = self._allocations.pop(allocation_id)
        if alloc.resource_type in self._resources:
            res = self._resources[alloc.resource_type]
            res.used_capacity = max(0.0, res.used_capacity - alloc.amount)
        return True

    # ----------------------------------------------------------- Query
    def get_utilization(self, resource_type: str) -> float:
        if resource_type not in self._resources:
            return 0.0
        res = self._resources[resource_type]
        if res.total_capacity <= 0:
            return 0.0
        return res.used_capacity / res.total_capacity

    def get_resource(self, resource_type: str) -> Optional[FleetResource]:
        return self._resources.get(resource_type)

    def get_all_resources(self) -> List[FleetResource]:
        return list(self._resources.values())

    def get_allocation(self, allocation_id: str) -> Optional[ResourceAllocation]:
        return self._allocations.get(allocation_id)

    def get_vessel_allocations(self, vessel_id: str) -> List[ResourceAllocation]:
        return [a for a in self._allocations.values() if a.vessel_id == vessel_id]

    # ----------------------------------------------------------- Forecast
    def predict_shortage(self, resource_type: str,
                         forecast: List[float]) -> ShortageForecast:
        """Predict shortage based on a list of demand values over time steps."""
        if resource_type not in self._resources:
            return ShortageForecast(
                resource_type=resource_type,
                current_utilization=0.0,
                projected_utilization=0.0,
                shortage_risk=0.0,
                time_to_shortage=None,
            )

        res = self._resources[resource_type]
        current_util = self.get_utilization(resource_type)

        if not forecast:
            return ShortageForecast(
                resource_type=resource_type,
                current_utilization=current_util,
                projected_utilization=current_util,
                shortage_risk=0.0,
                time_to_shortage=None,
            )

        avg_demand = sum(forecast) / len(forecast)
        projected = min(1.0, current_util + avg_demand / max(res.total_capacity, 1.0))

        shortage_risk = 0.0
        time_to_shortage: Optional[float] = None

        # Simulate demand consumption
        cumulative = res.used_capacity
        for i, demand in enumerate(forecast):
            cumulative += demand
            if cumulative >= res.total_capacity and time_to_shortage is None:
                time_to_shortage = float(i)

        if projected > 0.8:
            shortage_risk = (projected - 0.8) / 0.2
        if time_to_shortage is not None:
            shortage_risk = max(shortage_risk, 0.7)

        return ShortageForecast(
            resource_type=resource_type,
            current_utilization=current_util,
            projected_utilization=projected,
            shortage_risk=min(1.0, shortage_risk),
            time_to_shortage=time_to_shortage,
        )

    # --------------------------------------------------------- Rebalance
    def rebalance_resources(self, resources: Dict[str, float],
                            demand: Dict[str, float]) -> ReallocationPlan:
        """Compute a reallocation plan from surplus resources to deficit areas."""
        transfers: List[Dict[str, Any]] = []
        total_amount = 0.0

        surplus = []
        deficit = []

        for rtype, cap in resources.items():
            d = demand.get(rtype, 0.0)
            if cap > d * 1.2:  # >20% surplus
                surplus.append((rtype, cap - d))
            elif d > cap * 1.2:  # >20% deficit
                deficit.append((rtype, d - cap))

        # Match surplus to deficit
        for s_type, s_amount in surplus:
            for d_type, d_amount in deficit:
                if d_amount <= 0:
                    continue
                transfer = min(s_amount, d_amount)
                if transfer > 0:
                    transfers.append({
                        "from": s_type,
                        "to": d_type,
                        "amount": transfer,
                    })
                    total_amount += transfer
                    s_amount -= transfer
                    d_amount -= transfer
                    deficit = [(t, a - transfer if t == d_type else a)
                               for t, a in deficit]
                    deficit = [(t, max(0, a)) for t, a in deficit]
                    if s_amount <= 0:
                        break

        return ReallocationPlan(transfers=transfers, total_amount=total_amount)

    # ---------------------------------------------------------- OPEX
    def compute_opex(self, resources: List[FleetResource]) -> float:
        """Compute operating cost estimate based on resource utilization.

        Cost model:
          - Base cost per resource = capacity * unit_cost
          - Unit costs: fuel=2.5, bandwidth=1.0, compute=3.0, default=1.0
        """
        unit_costs = {
            "fuel": 2.5,
            "bandwidth": 1.0,
            "compute": 3.0,
        }
        total = 0.0
        for res in resources:
            unit = unit_costs.get(res.type, 1.0)
            total += res.used_capacity * unit
            # Idle capacity also costs (maintenance)
            idle = res.available_capacity
            total += idle * unit * 0.1
        return total
