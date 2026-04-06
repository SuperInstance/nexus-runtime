"""Market simulation with supply/demand dynamics."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MarketState:
    """Snapshot of market state at a point in time."""
    time: float = 0.0
    vessels: int = 0
    tasks: int = 0
    prices: List[float] = field(default_factory=list)
    supply_demand_ratio: float = 1.0
    avg_price: float = 0.0
    total_value: float = 0.0
    completed_tasks: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulatedAgent:
    """An agent in the market simulation."""
    agent_id: str = ""
    strategy: str = "competitive"  # competitive, conservative, aggressive
    budget: float = 0.0
    capabilities: Dict[str, Any] = field(default_factory=dict)
    current_tasks: int = 0
    max_tasks: int = 5


@dataclass
class SupplyDemandMetrics:
    """Supply and demand metrics."""
    supply: float = 0.0
    demand: float = 0.0
    ratio: float = 1.0
    surplus: bool = False


@dataclass
class EquilibriumPrice:
    """Equilibrium price result."""
    price: float = 0.0
    supply_at_price: float = 0.0
    demand_at_price: float = 0.0
    converged: bool = False


@dataclass
class EfficiencyMetrics:
    """Market efficiency metrics."""
    allocative_efficiency: float = 0.0
    price_stability: float = 0.0
    task_completion_rate: float = 0.0
    utilization_rate: float = 0.0
    overall_efficiency: float = 0.0


class MarketSimulator:
    """Simulates fleet marketplace dynamics."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._agents: List[SimulatedAgent] = []
        self._task_queue: List[Dict[str, Any]] = []
        self._completed: List[Dict[str, Any]] = []
        self._current_time: float = 0.0
        self._price_history: List[float] = []

    def initialize(
        self,
        agents: List[SimulatedAgent],
        initial_tasks: Optional[List[Dict[str, Any]]] = None,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> MarketState:
        """Initialize the market simulation."""
        self._agents = agents
        self._task_queue = initial_tasks or []
        self._completed = []
        self._current_time = 0.0
        self._price_history = []

        conditions = conditions or {}
        base_price = conditions.get("base_price", 1000.0)

        # Initial prices based on tasks
        initial_prices = [t.get("reward", base_price) for t in self._task_queue]
        self._price_history = list(initial_prices)

        total_vessels = sum(
            a.capabilities.get("vessels", 1) for a in self._agents
        )

        sd = self.compute_supply_demand(self._task_queue, self._agents)
        avg_price = sum(initial_prices) / len(initial_prices) if initial_prices else 0.0
        total_value = sum(t.get("reward", 0.0) for t in self._task_queue)

        state = MarketState(
            time=0.0,
            vessels=total_vessels,
            tasks=len(self._task_queue),
            prices=list(initial_prices),
            supply_demand_ratio=sd.ratio,
            avg_price=round(avg_price, 2),
            total_value=round(total_value, 2),
            completed_tasks=0,
        )
        return state

    def step(self, dt: float) -> MarketState:
        """Advance the simulation by dt time units."""
        self._current_time += dt

        # Agents may complete tasks
        newly_completed = []
        for agent in self._agents:
            # Probability of completing a task based on strategy
            if agent.current_tasks > 0:
                base_prob = 0.1 * dt
                if agent.strategy == "aggressive":
                    base_prob *= 1.3
                elif agent.strategy == "conservative":
                    base_prob *= 0.8
                # Scale with available vessels
                vessels = agent.capabilities.get("vessels", 1)
                base_prob *= min(vessels, agent.current_tasks)
                if self._rng.random() < base_prob:
                    agent.current_tasks = max(0, agent.current_tasks - 1)
                    task = {
                        "completed_by": agent.agent_id,
                        "time": self._current_time,
                    }
                    newly_completed.append(task)
                    self._completed.append(task)

        # New tasks arrive
        arrival_rate = 0.3 * dt
        n_new = int(arrival_rate)
        if self._rng.random() < (arrival_rate - n_new):
            n_new += 1
        base_price = 1000.0
        for _ in range(n_new):
            price = base_price * self._rng.uniform(0.5, 2.0)
            new_task = {
                "reward": round(price, 2),
                "time_posted": self._current_time,
            }
            self._task_queue.append(new_task)
            self._price_history.append(price)

        # Agents bid on tasks
        for agent in self._agents:
            if agent.current_tasks >= agent.max_tasks:
                continue
            if not self._task_queue:
                break
            if self._rng.random() < 0.2 * dt:
                task = self._task_queue.pop(0)
                agent.current_tasks += 1
                task["assigned_to"] = agent.agent_id
                task["assigned_time"] = self._current_time

        # Update state
        total_vessels = sum(a.capabilities.get("vessels", 1) for a in self._agents)
        sd = self.compute_supply_demand(self._task_queue, self._agents)
        current_prices = [t.get("reward", 1000.0) for t in self._task_queue]
        avg_price = sum(current_prices) / len(current_prices) if current_prices else 0.0
        total_value = sum(t.get("reward", 0.0) for t in self._task_queue)

        return MarketState(
            time=self._current_time,
            vessels=total_vessels,
            tasks=len(self._task_queue),
            prices=list(current_prices),
            supply_demand_ratio=round(sd.ratio, 4),
            avg_price=round(avg_price, 2),
            total_value=round(total_value, 2),
            completed_tasks=len(self._completed),
        )

    def run(self, duration: float, dt: float) -> List[MarketState]:
        """Run the simulation for the given duration."""
        states: List[MarketState] = []
        steps = int(duration / dt)
        for _ in range(steps):
            state = self.step(dt)
            states.append(state)
        return states

    def compute_supply_demand(
        self,
        tasks: List[Any],
        agents: List[SimulatedAgent],
    ) -> SupplyDemandMetrics:
        """Compute supply and demand metrics."""
        demand = len(tasks)

        # Supply: total capacity of agents
        total_capacity = 0
        for agent in agents:
            available = agent.max_tasks - agent.current_tasks
            vessels = agent.capabilities.get("vessels", 1)
            total_capacity += available * vessels

        supply = max(0, total_capacity)
        ratio = supply / demand if demand > 0 else float("inf")
        surplus = supply > demand

        return SupplyDemandMetrics(
            supply=float(supply),
            demand=float(demand),
            ratio=round(ratio, 4),
            surplus=surplus,
        )

    def compute_equilibrium_price(
        self,
        tasks: List[Any],
        agents: List[SimulatedAgent],
    ) -> EquilibriumPrice:
        """Compute the equilibrium price using iterative approach."""
        if not tasks:
            return EquilibriumPrice(price=0.0, converged=True)

        rewards = [t.get("reward", 1000.0) for t in tasks]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

        # Iterative price adjustment toward equilibrium
        price = avg_reward
        for _ in range(20):
            sd = self.compute_supply_demand(tasks, agents)
            if sd.ratio > 1.0:
                # Sur supply: lower price
                price *= 0.95
            elif sd.ratio < 1.0:
                # Under supply: raise price
                price *= 1.05
            else:
                break
            price = max(100.0, price)  # Floor price

        return EquilibriumPrice(
            price=round(price, 2),
            supply_at_price=sd.supply,
            demand_at_price=sd.demand,
            converged=True,
        )

    def simulate_shock(
        self,
        market_state: MarketState,
        shock_type: str,
        magnitude: float,
    ) -> MarketState:
        """Apply a shock to the market and return the new state."""
        new_state = MarketState(
            time=market_state.time,
            vessels=market_state.vessels,
            tasks=market_state.tasks,
            prices=list(market_state.prices),
            supply_demand_ratio=market_state.supply_demand_ratio,
            avg_price=market_state.avg_price,
            total_value=market_state.total_value,
            completed_tasks=market_state.completed_tasks,
            metadata=dict(market_state.metadata),
        )

        if shock_type == "demand_surge":
            # More tasks appear
            new_tasks = int(market_state.tasks * magnitude)
            new_state.tasks += new_tasks
            new_prices = new_state.prices + [market_state.avg_price * self._rng.uniform(0.8, 1.5) for _ in range(new_tasks)]
            new_state.prices = new_prices
            new_state.total_value += new_tasks * market_state.avg_price * 1.2
            new_state.supply_demand_ratio *= (1.0 - magnitude * 0.3)

        elif shock_type == "vessel_loss":
            lost = int(market_state.vessels * magnitude)
            new_state.vessels = max(0, market_state.vessels - lost)
            new_state.supply_demand_ratio *= (1.0 - magnitude * 0.5)
            # Prices increase
            new_state.avg_price *= (1.0 + magnitude * 0.2)
            new_state.prices = [p * (1.0 + magnitude * 0.2) for p in new_state.prices]

        elif shock_type == "budget_cut":
            # Lower rewards/prices
            new_state.avg_price *= (1.0 - magnitude * 0.3)
            new_state.prices = [p * (1.0 - magnitude * 0.3) for p in new_state.prices]
            new_state.total_value *= (1.0 - magnitude * 0.3)

        elif shock_type == "technology_upgrade":
            # Vessels become more effective
            new_state.supply_demand_ratio *= (1.0 + magnitude * 0.4)
            new_state.avg_price *= (1.0 - magnitude * 0.1)
            new_state.prices = [p * (1.0 - magnitude * 0.1) for p in new_state.prices]

        new_state.supply_demand_ratio = round(max(0.01, new_state.supply_demand_ratio), 4)
        new_state.avg_price = round(new_state.avg_price, 2)
        new_state.total_value = round(new_state.total_value, 2)
        return new_state

    def compute_market_efficiency(
        self,
        results: List[Dict[str, Any]],
    ) -> EfficiencyMetrics:
        """Compute market efficiency metrics from simulation results."""
        if not results:
            return EfficiencyMetrics()

        n = len(results)

        # Price stability: 1 - coefficient of variation of prices
        prices = [r.get("avg_price", 0.0) for r in results if r.get("avg_price", 0.0) > 0]
        if len(prices) > 1:
            mean_price = sum(prices) / len(prices)
            variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
            std_price = math.sqrt(variance)
            price_stability = max(0.0, 1.0 - std_price / mean_price) if mean_price > 0 else 0.0
        else:
            price_stability = 1.0

        # Task completion rate
        total_tasks_posted = sum(r.get("tasks", 0) for r in results)
        total_completed = results[-1].get("completed_tasks", 0) if results else 0
        completion_rate = total_completed / total_tasks_posted if total_tasks_posted > 0 else 0.0

        # Utilization rate: average supply/demand ratio (capped at 1)
        ratios = [min(1.0, r.get("supply_demand_ratio", 1.0)) for r in results]
        utilization_rate = sum(ratios) / len(ratios) if ratios else 0.0

        # Allocative efficiency: how close supply meets demand on average
        avg_ratio = sum(r.get("supply_demand_ratio", 1.0) for r in results) / n
        allocative_efficiency = 1.0 / (1.0 + abs(avg_ratio - 1.0))

        # Overall efficiency
        overall = (
            0.25 * allocative_efficiency
            + 0.25 * price_stability
            + 0.25 * completion_rate
            + 0.25 * utilization_rate
        )

        return EfficiencyMetrics(
            allocative_efficiency=round(allocative_efficiency, 4),
            price_stability=round(price_stability, 4),
            task_completion_rate=round(completion_rate, 4),
            utilization_rate=round(utilization_rate, 4),
            overall_efficiency=round(overall, 4),
        )
