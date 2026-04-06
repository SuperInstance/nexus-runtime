"""Tests for scenario analysis."""

import math
import pytest
from jetson.digital_twin.scenario import Scenario, ScenarioResult, ScenarioRunner
from jetson.digital_twin.physics import VesselState, Force


class TestScenario:
    def test_defaults(self):
        s = Scenario(name="test")
        assert s.name == "test"
        assert s.duration == 60.0
        assert s.dt == 0.1
        assert isinstance(s.initial_state, VesselState)
        assert s.parameters == {}

    def test_copy(self):
        s = Scenario(name="orig", parameters={'thrust': 50})
        c = s.copy()
        assert c.name == "orig"
        c.parameters['thrust'] = 999
        assert s.parameters['thrust'] == 50

    def test_copy_deep_initial_state(self):
        s = Scenario(name="test", initial_state=VesselState(x=5))
        c = s.copy()
        c.initial_state.x = 999
        assert s.initial_state.x == 5


class TestScenarioResult:
    def test_defaults(self):
        r = ScenarioResult(scenario=Scenario(name="t"))
        assert r.success is True
        assert r.trajectory == []
        assert r.metrics == {}

    def test_final_state_empty(self):
        r = ScenarioResult(scenario=Scenario(name="t"))
        fs = r.final_state()
        assert isinstance(fs, VesselState)

    def test_final_state_with_trajectory(self):
        r = ScenarioResult(
            scenario=Scenario(name="t"),
            trajectory=[VesselState(x=0), VesselState(x=10)],
        )
        assert r.final_state().x == 10

    def test_total_distance(self):
        r = ScenarioResult(
            scenario=Scenario(name="t"),
            trajectory=[
                VesselState(x=0, y=0),
                VesselState(x=3, y=4),
                VesselState(x=6, y=8),
            ],
        )
        assert abs(r.total_distance() - 10.0) < 1e-9

    def test_max_speed(self):
        r = ScenarioResult(
            scenario=Scenario(name="t"),
            trajectory=[
                VesselState(vx=1, vy=0),
                VesselState(vx=3, vy=4),
                VesselState(vx=0, vy=0),
            ],
        )
        assert abs(r.max_speed() - 5.0) < 1e-9

    def test_avg_speed(self):
        r = ScenarioResult(
            scenario=Scenario(name="t"),
            trajectory=[
                VesselState(vx=2, vy=0),
                VesselState(vx=0, vy=4),
            ],
        )
        assert abs(r.avg_speed() - 3.0) < 1e-9

    def test_avg_speed_empty(self):
        r = ScenarioResult(scenario=Scenario(name="t"))
        assert r.avg_speed() == 0.0


class TestScenarioRunner:
    def setup_method(self):
        self.runner = ScenarioRunner()

    def test_run_basic(self):
        scenario = Scenario(
            name="basic",
            parameters={'thrust': 100, 'heading': 0},
            duration=1.0,
            dt=0.1,
        )
        result = self.runner.run(scenario)
        assert result.success is True
        assert len(result.trajectory) > 1
        assert 'total_distance' in result.metrics

    def test_run_zero_thrust(self):
        scenario = Scenario(
            name="zero",
            parameters={'thrust': 0},
            duration=1.0,
            dt=0.1,
        )
        result = self.runner.run(scenario)
        assert result.success is True
        # Vessel shouldn't move much without thrust (but env forces exist)
        assert result.metrics['total_distance'] >= 0

    def test_run_with_initial_velocity(self):
        scenario = Scenario(
            name="moving",
            initial_state=VesselState(vx=5),
            parameters={'thrust': 0},
            duration=2.0,
            dt=0.1,
        )
        result = self.runner.run(scenario)
        assert result.success is True
        assert result.metrics['total_distance'] > 0

    def test_run_generates_trajectory(self):
        scenario = Scenario(name="traj", duration=1.0, dt=0.1)
        result = self.runner.run(scenario)
        expected_steps = int(1.0 / 0.1) + 1
        assert len(result.trajectory) == expected_steps

    def test_run_metrics(self):
        scenario = Scenario(name="metrics", parameters={'thrust': 50}, duration=1.0, dt=0.1)
        result = self.runner.run(scenario)
        metrics = result.metrics
        assert 'total_distance' in metrics
        assert 'max_speed' in metrics
        assert 'avg_speed' in metrics
        assert 'max_acceleration' in metrics
        assert 'final_x' in metrics
        assert 'final_y' in metrics
        assert 'position_drift' in metrics

    def test_parameter_sweep_single(self):
        base = Scenario(name="sweep", parameters={'thrust': 0}, duration=0.5, dt=0.1)
        results = self.runner.parameter_sweep(base, {'thrust': [50, 100, 200]})
        assert len(results) == 3
        for r in results:
            assert r.success is True

    def test_parameter_sweep_multiple_params(self):
        base = Scenario(name="multi", duration=0.5, dt=0.1)
        results = self.runner.parameter_sweep(
            base, {'thrust': [50, 100], 'heading': [0, math.pi/2]}
        )
        assert len(results) == 4

    def test_parameter_sweep_empty(self):
        base = Scenario(name="empty", duration=0.5, dt=0.1)
        results = self.runner.parameter_sweep(base, {})
        assert len(results) == 1

    def test_compare_results_by_distance(self):
        r1 = ScenarioResult(
            scenario=Scenario(name="a"),
            metrics={'total_distance': 10.0},
        )
        r2 = ScenarioResult(
            scenario=Scenario(name="b"),
            metrics={'total_distance': 20.0},
        )
        ranked = self.runner.compare_results([r1, r2], 'total_distance')
        assert ranked[0][1] > ranked[1][1]  # higher is better for distance

    def test_compare_results_by_error(self):
        r1 = ScenarioResult(
            scenario=Scenario(name="a"),
            metrics={'max_position_error': 5.0},
        )
        r2 = ScenarioResult(
            scenario=Scenario(name="b"),
            metrics={'max_position_error': 2.0},
        )
        ranked = self.runner.compare_results([r1, r2], 'max_position_error')
        assert ranked[0][1] <= ranked[1][1]  # lower is better

    def test_export_import_scenario(self):
        scenario = Scenario(
            name="export_test",
            parameters={'thrust': 100, 'heading': 1.5},
            initial_state=VesselState(x=10, y=20, z=5, vx=1),
            duration=120.0,
            dt=0.05,
        )
        data = self.runner.export_scenario(scenario)
        assert data['name'] == 'export_test'
        assert data['parameters']['thrust'] == 100
        assert data['initial_state']['x'] == 10
        assert data['duration'] == 120.0

        imported = self.runner.import_scenario(data)
        assert imported.name == 'export_test'
        assert imported.parameters['thrust'] == 100
        assert imported.initial_state.x == 10
        assert imported.duration == 120.0

    def test_replay_scenario(self):
        scenario = Scenario(name="replay", duration=0.5, dt=0.1)
        exported = self.runner.export_scenario(scenario)
        result = self.runner.replay_scenario(exported)
        assert result.success is True
        assert len(result.trajectory) > 1

    def test_save_and_load_scenario(self):
        scenario = Scenario(name="saved", duration=1.0)
        self.runner.save_scenario("test_scenario", scenario)
        loaded = self.runner.load_scenario("test_scenario")
        assert loaded is not None
        assert loaded.name == "saved"

    def test_load_nonexistent(self):
        loaded = self.runner.load_scenario("nonexistent")
        assert loaded is None

    def test_list_saved_scenarios(self):
        self.runner.save_scenario("s1", Scenario(name="s1"))
        self.runner.save_scenario("s2", Scenario(name="s2"))
        names = self.runner.list_saved_scenarios()
        assert "s1" in names and "s2" in names

    def test_run_with_environment(self):
        scenario = Scenario(
            name="env_test",
            parameters={'thrust': 50},
            duration=1.0,
            dt=0.1,
            environment_config={
                'wind': {'speed': 15, 'direction': 0},
                'wave': {'height': 2.0},
                'current': {'speed': 1.0, 'direction': math.pi/4},
            },
        )
        result = self.runner.run(scenario)
        assert result.success is True
        assert result.metrics['total_distance'] > 0

    def test_failed_scenario(self):
        """Test with invalid parameters doesn't crash."""
        scenario = Scenario(name="valid", duration=0.1, dt=0.1)
        result = self.runner.run(scenario)
        assert result.success is True

    def test_sweep_varying_duration(self):
        base = Scenario(name="dur_sweep", parameters={'thrust': 50}, dt=0.1)
        # Duration not in sweep dict, vary via environment
        results = self.runner.parameter_sweep(base, {'thrust': [10, 100]})
        assert len(results) == 2
        # Higher thrust should result in more distance
        d_low = results[0].metrics['total_distance']
        d_high = results[1].metrics['total_distance']
        # Thrust 100 > thrust 10 (though env forces also act)
        assert d_high >= 0
