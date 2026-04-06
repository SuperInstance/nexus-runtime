"""Tests for market_sim module."""

import pytest

from jetson.marketplace.market_sim import (
    MarketState, SimulatedAgent, MarketSimulator,
    SupplyDemandMetrics, EquilibriumPrice, EfficiencyMetrics,
)


class TestMarketState:
    def test_default(self):
        s = MarketState()
        assert s.time == 0.0
        assert s.vessels == 0
        assert s.tasks == 0
        assert s.prices == []
        assert s.supply_demand_ratio == 1.0
        assert s.completed_tasks == 0

    def test_custom(self):
        s = MarketState(
            time=10.0, vessels=5, tasks=3,
            prices=[1000.0, 2000.0, 1500.0],
            supply_demand_ratio=1.5,
        )
        assert s.time == 10.0
        assert len(s.prices) == 3


class TestSimulatedAgent:
    def test_default(self):
        a = SimulatedAgent()
        assert a.agent_id == ""
        assert a.strategy == "competitive"
        assert a.budget == 0.0
        assert a.current_tasks == 0
        assert a.max_tasks == 5

    def test_custom(self):
        a = SimulatedAgent(
            agent_id="a1", strategy="conservative",
            budget=100000.0, capabilities={"vessels": 3},
        )
        assert a.agent_id == "a1"
        assert a.strategy == "conservative"
        assert a.capabilities["vessels"] == 3


class TestMarketSimulator:
    def setup_method(self):
        self.sim = MarketSimulator(seed=42)

    def test_initialize(self):
        agents = [SimulatedAgent(agent_id="a1", capabilities={"vessels": 2})]
        tasks = [{"reward": 1000.0}, {"reward": 2000.0}]
        state = self.sim.initialize(agents, tasks)
        assert state.tasks == 2
        assert state.vessels == 2
        assert state.avg_price == 1500.0

    def test_initialize_empty(self):
        state = self.sim.initialize([], [])
        assert state.tasks == 0
        assert state.vessels == 0
        assert state.avg_price == 0.0

    def test_initialize_no_tasks(self):
        agents = [SimulatedAgent(agent_id="a1", capabilities={"vessels": 3})]
        state = self.sim.initialize(agents)
        assert state.tasks == 0
        assert state.vessels == 3

    def test_initialize_custom_conditions(self):
        agents = [SimulatedAgent(agent_id="a1")]
        tasks = [{"reward": 500.0}]
        state = self.sim.initialize(agents, tasks, {"base_price": 2000.0})
        assert state.avg_price == 500.0

    def test_step_advances_time(self):
        agents = [SimulatedAgent(agent_id="a1", capabilities={"vessels": 2})]
        self.sim.initialize(agents, [])
        state = self.sim.step(1.0)
        assert state.time == pytest.approx(1.0, abs=0.01)

    def test_step_accumulates_time(self):
        agents = [SimulatedAgent(agent_id="a1")]
        self.sim.initialize(agents, [])
        self.sim.step(1.0)
        self.sim.step(2.0)
        state = self.sim.step(1.0)
        assert state.time == pytest.approx(4.0, abs=0.01)

    def test_step_new_tasks(self):
        agents = [SimulatedAgent(agent_id="a1")]
        self.sim.initialize(agents, [])
        state = self.sim.step(10.0)
        assert state.tasks >= 0

    def test_run_basic(self):
        agents = [SimulatedAgent(agent_id="a1", capabilities={"vessels": 2})]
        tasks = [{"reward": 1000.0}] * 5
        self.sim.initialize(agents, tasks)
        states = self.sim.run(duration=10.0, dt=1.0)
        assert len(states) == 10

    def test_run_empty(self):
        self.sim.initialize([], [])
        states = self.sim.run(5.0, 1.0)
        assert len(states) == 5

    def test_compute_supply_demand(self):
        agents = [
            SimulatedAgent(agent_id="a1", max_tasks=5, current_tasks=1, capabilities={"vessels": 2}),
            SimulatedAgent(agent_id="a2", max_tasks=3, current_tasks=0, capabilities={"vessels": 1}),
        ]
        tasks = [{"reward": 1000.0}] * 3
        sd = self.sim.compute_supply_demand(tasks, agents)
        # a1: (5-1)*2=8, a2: (3-0)*1=3, total supply = 11
        assert sd.supply == 11.0
        assert sd.demand == 3.0
        assert sd.surplus is True

    def test_compute_supply_demand_no_demand(self):
        agents = [SimulatedAgent(agent_id="a1", max_tasks=5, capabilities={"vessels": 1})]
        sd = self.sim.compute_supply_demand([], agents)
        assert sd.demand == 0.0
        assert sd.ratio == float("inf")

    def test_compute_supply_demand_deficit(self):
        agents = [SimulatedAgent(agent_id="a1", max_tasks=1, current_tasks=1, capabilities={"vessels": 1})]
        tasks = [{"reward": 1000.0}] * 5
        sd = self.sim.compute_supply_demand(tasks, agents)
        assert sd.surplus is False

    def test_compute_equilibrium_price(self):
        agents = [SimulatedAgent(agent_id="a1", capabilities={"vessels": 2})]
        tasks = [{"reward": 1000.0}, {"reward": 2000.0}]
        eq = self.sim.compute_equilibrium_price(tasks, agents)
        assert eq.price > 0
        assert eq.converged is True

    def test_compute_equilibrium_price_no_tasks(self):
        agents = [SimulatedAgent(agent_id="a1")]
        eq = self.sim.compute_equilibrium_price([], agents)
        assert eq.price == 0.0
        assert eq.converged is True

    def test_compute_equilibrium_price_adjusts_down(self):
        # Large supply relative to demand -> lower price
        agents = [SimulatedAgent(agent_id="a1", max_tasks=10, capabilities={"vessels": 5})]
        tasks = [{"reward": 5000.0}]
        eq = self.sim.compute_equilibrium_price(tasks, agents)
        assert eq.price < 5000.0

    def test_simulate_shock_demand_surge(self):
        state = MarketState(tasks=10, vessels=5, avg_price=1000.0, prices=[1000.0]*10, supply_demand_ratio=0.5, total_value=10000.0)
        new_state = self.sim.simulate_shock(state, "demand_surge", 0.5)
        assert new_state.tasks > 10

    def test_simulate_shock_vessel_loss(self):
        state = MarketState(vessels=10, avg_price=1000.0, prices=[1000.0], supply_demand_ratio=2.0, total_value=5000.0)
        new_state = self.sim.simulate_shock(state, "vessel_loss", 0.3)
        assert new_state.vessels < 10

    def test_simulate_shock_budget_cut(self):
        state = MarketState(avg_price=1000.0, prices=[1000.0, 2000.0], total_value=3000.0, supply_demand_ratio=1.0)
        new_state = self.sim.simulate_shock(state, "budget_cut", 0.2)
        assert new_state.avg_price < 1000.0

    def test_simulate_shock_technology_upgrade(self):
        state = MarketState(avg_price=1000.0, prices=[1000.0], supply_demand_ratio=1.0, total_value=1000.0)
        new_state = self.sim.simulate_shock(state, "technology_upgrade", 0.5)
        assert new_state.supply_demand_ratio > 1.0
        assert new_state.avg_price < 1000.0

    def test_simulate_shock_unknown_type(self):
        state = MarketState(time=1.0, vessels=5, tasks=3, avg_price=1000.0, prices=[1000.0], supply_demand_ratio=1.0, total_value=3000.0)
        new_state = self.sim.simulate_shock(state, "unknown_shock", 0.5)
        assert new_state.time == 1.0  # Unchanged

    def test_simulate_shock_vessel_loss_floor(self):
        state = MarketState(vessels=1, avg_price=1000.0, prices=[1000.0], supply_demand_ratio=0.5, total_value=1000.0)
        new_state = self.sim.simulate_shock(state, "vessel_loss", 0.9)
        # int(0.9) = 0, so 1-0 = 1 (no vessel lost)
        assert new_state.vessels == 1

    def test_compute_market_efficiency_empty(self):
        eff = self.sim.compute_market_efficiency([])
        assert eff.allocative_efficiency == 0.0
        assert eff.overall_efficiency == 0.0

    def test_compute_market_efficiency_stable(self):
        results = [
            {"avg_price": 1000.0, "supply_demand_ratio": 1.0, "tasks": 5},
            {"avg_price": 1000.0, "supply_demand_ratio": 1.0, "tasks": 5},
            {"avg_price": 1000.0, "supply_demand_ratio": 1.0, "tasks": 5},
        ]
        eff = self.sim.compute_market_efficiency(results)
        assert eff.price_stability == 1.0
        assert eff.overall_efficiency > 0

    def test_compute_market_efficiency_volatile(self):
        results = [
            {"avg_price": 500.0, "supply_demand_ratio": 0.5, "tasks": 10, "completed_tasks": 2},
            {"avg_price": 3000.0, "supply_demand_ratio": 2.0, "tasks": 1, "completed_tasks": 5},
        ]
        eff = self.sim.compute_market_efficiency(results)
        assert eff.price_stability < 1.0

    def test_compute_market_efficiency_single_result(self):
        results = [{"avg_price": 1000.0, "supply_demand_ratio": 1.0, "tasks": 5, "completed_tasks": 5}]
        eff = self.sim.compute_market_efficiency(results)
        assert eff.price_stability == 1.0
        assert eff.task_completion_rate == 1.0

    def test_seed_reproducibility(self):
        sim1 = MarketSimulator(seed=123)
        sim2 = MarketSimulator(seed=123)
        agents = [SimulatedAgent(agent_id="a1")]
        sim1.initialize(agents, [{"reward": 1000.0}])
        sim2.initialize(agents, [{"reward": 1000.0}])
        s1 = sim1.step(1.0)
        s2 = sim2.step(1.0)
        assert s1.tasks == s2.tasks

    def test_step_with_tasks(self):
        agents = [SimulatedAgent(agent_id="a1", capabilities={"vessels": 2})]
        tasks = [{"reward": 1000.0}] * 10
        self.sim.initialize(agents, tasks)
        state = self.sim.step(1.0)
        # Some tasks may have been assigned
        assert isinstance(state.completed_tasks, int)

    def test_run_small_dt(self):
        agents = [SimulatedAgent(agent_id="a1")]
        self.sim.initialize(agents, [])
        states = self.sim.run(1.0, 0.1)
        assert len(states) == 10
        for i, s in enumerate(states):
            assert s.time == pytest.approx((i + 1) * 0.1, abs=0.01)

    def test_supply_demand_metrics(self):
        sd = SupplyDemandMetrics(supply=10.0, demand=5.0)
        assert sd.ratio == 1.0  # default, not auto-computed
        assert sd.surplus is False  # not auto-computed

    def test_efficiency_metrics(self):
        e = EfficiencyMetrics(
            allocative_efficiency=0.8,
            price_stability=0.9,
            task_completion_rate=0.7,
            utilization_rate=0.6,
            overall_efficiency=0.75,
        )
        assert e.overall_efficiency == 0.75

    def test_equilibrium_price_fields(self):
        eq = EquilibriumPrice(price=1500.0, supply_at_price=10.0, demand_at_price=8.0, converged=True)
        assert eq.price == 1500.0
        assert eq.converged is True

    def test_agent_strategies(self):
        for strategy in ["competitive", "conservative", "aggressive"]:
            a = SimulatedAgent(agent_id=strategy, strategy=strategy)
            assert a.strategy == strategy

    def test_market_state_metadata(self):
        state = MarketState(metadata={"shock": True})
        assert state.metadata["shock"] is True

    def test_initialize_multiple_agents(self):
        agents = [
            SimulatedAgent(agent_id="a1", capabilities={"vessels": 3}),
            SimulatedAgent(agent_id="a2", capabilities={"vessels": 2}),
            SimulatedAgent(agent_id="a3", capabilities={"vessels": 4}),
        ]
        state = self.sim.initialize(agents, [])
        assert state.vessels == 9

    def test_simulate_shock_preserves_metadata(self):
        state = MarketState(metadata={"key": "value"}, avg_price=1000.0, prices=[1000.0], supply_demand_ratio=1.0, total_value=1000.0)
        new_state = self.sim.simulate_shock(state, "budget_cut", 0.1)
        assert new_state.metadata["key"] == "value"
