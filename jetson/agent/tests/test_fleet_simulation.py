"""Tests for NEXUS End-to-End Fleet Simulation."""

import math
import pytest
from jetson.agent.fleet_simulation import (
    FleetSimulation, ReflexConfig, ReflexStrategy, SimEvent, SimEventType,
    SimulationConfig, VesselState, VesselStatus, Waypoint,
)


class TestVesselState:
    def test_defaults(self):
        v = VesselState("v1")
        assert v.vessel_id == "v1"
        assert v.trust_score == 0.5
        assert v.safety_state == VesselStatus.ACTIVE

    def test_to_dict(self):
        v = VesselState("v1", lat=48.5, lon=-122.3)
        d = v.to_dict()
        assert d["vessel_id"] == "v1"
        assert d["lat"] == 48.5


class TestNavCalc:
    def test_same_point(self):
        b, d = FleetSimulation._nav_calc(0, 0, 0, 0)
        assert d == 0.0

    def test_north(self):
        b, d = FleetSimulation._nav_calc(0, 0, 1, 0)
        assert 58 < d < 62

    def test_east(self):
        b, d = FleetSimulation._nav_calc(0, 0, 0, 1)
        assert 80 < b < 100

    def test_symmetry(self):
        b1, d1 = FleetSimulation._nav_calc(48.0, -123.0, 49.0, -122.0)
        b2, d2 = FleetSimulation._nav_calc(49.0, -122.0, 48.0, -123.0)
        assert abs(d1 - d2) < 0.01  # same distance


class TestSimulationSetup:
    def test_default_setup(self):
        sim = FleetSimulation()
        sim.setup()
        assert len(sim.vessels) == 3
        assert len(sim.waypoints) == 5

    def test_custom_config(self):
        cfg = SimulationConfig(num_vessels=5, num_waypoints=3, max_ticks=100)
        sim = FleetSimulation(config=cfg)
        sim.setup()
        assert len(sim.vessels) == 5

    def test_reflex_configs_assigned(self):
        sim = FleetSimulation()
        sim.setup()
        for vid in sim.vessels:
            assert vid in sim.reflex_configs


class TestSimulationStep:
    def test_single_step(self):
        sim = FleetSimulation(config=SimulationConfig(max_ticks=100))
        sim.setup()
        assert sim.step()
        assert sim.tick == 1

    def test_fuel_decreases(self):
        sim = FleetSimulation(config=SimulationConfig(max_ticks=100, fuel_consumption_per_nm=5.0))
        sim.setup()
        init = {vid: v.fuel_pct for vid, v in sim.vessels.items()}
        for _ in range(50):
            sim.step()
        for vid, v in sim.vessels.items():
            assert v.fuel_pct < init[vid]

    def test_max_ticks(self):
        sim = FleetSimulation(config=SimulationConfig(max_ticks=5))
        sim.setup()
        for _ in range(10):
            sim.step()
        assert sim.tick == 5

    def test_events_recorded(self):
        sim = FleetSimulation(config=SimulationConfig(max_ticks=50, safety_event_probability=0.05))
        sim.setup()
        for _ in range(50):
            sim.step()
        assert len(sim.events) > 0


class TestSimulationRun:
    def test_full_run(self):
        sim = FleetSimulation(config=SimulationConfig(max_ticks=200))
        results = sim.run()
        assert results["winner"] is not None
        assert len(results["ranking"]) == 3

    def test_ranking_sorted(self):
        sim = FleetSimulation()
        results = sim.run()
        fitnesses = [r["fitness"] for r in results["ranking"]]
        assert fitnesses == sorted(fitnesses, reverse=True)

    def test_competition_strategies(self):
        configs = [
            ReflexConfig("slow", ReflexStrategy.CONSERVATIVE, max_speed_knots=2.0, efficiency_weight=0.1),
            ReflexConfig("fast", ReflexStrategy.AGGRESSIVE, max_speed_knots=15.0, efficiency_weight=0.9),
            ReflexConfig("balanced", ReflexStrategy.MODERATE, max_speed_knots=6.0, efficiency_weight=0.5),
        ]
        sim = FleetSimulation(config=SimulationConfig(max_ticks=200), reflex_configs=configs)
        results = sim.run()
        strategies = [r["strategy"] for r in results["ranking"]]
        assert len(set(strategies)) == 3


class TestDeterminism:
    def test_same_seed(self):
        r1 = FleetSimulation(config=SimulationConfig(max_ticks=100)).run()
        r2 = FleetSimulation(config=SimulationConfig(max_ticks=100)).run()
        assert r1["winner"]["vessel_id"] == r2["winner"]["vessel_id"]


class TestEdgeCases:
    def test_single_vessel(self):
        sim = FleetSimulation(config=SimulationConfig(num_vessels=1, num_waypoints=2, max_ticks=50))
        results = sim.run()
        assert len(results["ranking"]) == 1

    def test_large_fleet(self):
        sim = FleetSimulation(config=SimulationConfig(num_vessels=10, max_ticks=100))
        results = sim.run()
        assert len(results["ranking"]) == 10

    def test_results_structure(self):
        results = FleetSimulation().run()
        for key in ["ticks", "vessels", "events", "ranking", "winner"]:
            assert key in results
        for r in results["ranking"]:
            for key in ["fitness", "strategy", "waypoints_reached", "safety_events", "distance_nm"]:
                assert key in r


class TestScoring:
    def test_safety_penalty(self):
        cfg = SimulationConfig(max_ticks=200, safety_event_probability=0.2)
        r = FleetSimulation(config=cfg).run()
        total_events = sum(v["safety_events"] for v in r["vessels"].values())
        assert total_events > 0

    def test_distance_bonus(self):
        results = FleetSimulation(config=SimulationConfig(max_ticks=200)).run()
        for r in results["ranking"]:
            assert r["distance_nm"] > 0
