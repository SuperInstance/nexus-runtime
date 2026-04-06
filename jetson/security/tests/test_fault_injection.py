"""Tests for fault_injection module."""

import time
import pytest
from jetson.security.fault_injection import (
    FaultConfig,
    FaultInjector,
    FaultScenario,
    FaultSeverity,
    FaultTestRunner,
    FaultType,
    TestResult,
)


# ── FaultConfig ─────────────────────────────────────────────────────

class TestFaultConfig:
    def test_default_creation(self):
        fc = FaultConfig(fault_type=FaultType.STUCK_AT, target="sensor_0")
        assert fc.fault_type == FaultType.STUCK_AT
        assert fc.probability == 1.0
        assert fc.duration == 0.0
        assert fc.severity == FaultSeverity.MEDIUM
        assert fc.params == {}

    def test_custom_params(self):
        fc = FaultConfig(
            fault_type=FaultType.BIAS, target="temp",
            probability=0.5, duration=10.0, severity=FaultSeverity.HIGH,
            params={"bias": 5.0},
        )
        assert fc.probability == 0.5
        assert fc.params["bias"] == 5.0

    def test_all_fault_types(self):
        for ft in FaultType:
            fc = FaultConfig(fault_type=ft, target="x")
            assert fc.fault_type == ft


# ── FaultScenario ───────────────────────────────────────────────────

class TestFaultScenario:
    def test_default(self):
        fs = FaultScenario(name="test_scenario")
        assert fs.name == "test_scenario"
        assert fs.faults == []
        assert fs.expected_behavior == "system_recovers"

    def test_with_faults(self):
        fc = FaultConfig(fault_type=FaultType.STUCK_AT, target="s1")
        fs = FaultScenario(name="s1", faults=[fc], recovery_method="reboot")
        assert len(fs.faults) == 1
        assert fs.recovery_method == "reboot"


# ── FaultInjector ──────────────────────────────────────────────────

class TestFaultInjectorConstruction:
    def test_construct(self):
        fi = FaultInjector()
        assert fi.get_active_faults() == []
        assert fi.get_injection_log() == []

    def test_with_seed(self):
        fi = FaultInjector(seed=42)
        assert fi.get_active_faults() == []


class TestInjectSensorFault:
    def test_inject_stuck_at(self):
        fi = FaultInjector()
        rec = fi.inject_sensor_fault("temp", FaultType.STUCK_AT, {"stuck_value": 37.0})
        assert rec.injected is True
        assert rec.fault_type == FaultType.STUCK_AT
        assert rec.target == "temp"

    def test_inject_bias(self):
        fi = FaultInjector()
        rec = fi.inject_sensor_fault("depth", FaultType.BIAS, {"bias": -10.0})
        assert rec.injected
        assert rec.params["bias"] == -10.0

    def test_inject_default_params(self):
        fi = FaultInjector()
        rec = fi.inject_sensor_fault("x", FaultType.NOISE)
        assert rec.params == {}

    def test_inject_logged(self):
        fi = FaultInjector()
        fi.inject_sensor_fault("s", FaultType.DROP)
        assert len(fi.get_injection_log()) == 1

    def test_inject_activates(self):
        fi = FaultInjector()
        fi.inject_sensor_fault("s", FaultType.STUCK_AT, {"stuck_value": 0})
        active = fi.get_active_faults()
        assert len(active) == 1


class TestInjectCommunicationFault:
    def test_inject_drop(self):
        fi = FaultInjector()
        rec = fi.inject_communication_fault("link_0", FaultType.DROP)
        assert rec.injected
        assert rec.target == "link_0"

    def test_inject_corruption(self):
        fi = FaultInjector()
        rec = fi.inject_communication_fault("uart", FaultType.CORRUPTION)
        assert rec.fault_type == FaultType.CORRUPTION

    def test_comm_fault_logged(self):
        fi = FaultInjector()
        fi.inject_communication_fault("l", FaultType.DELAY)
        assert len(fi.get_injection_log()) == 1


class TestInjectComputationFault:
    def test_inject_crash(self):
        fi = FaultInjector()
        rec = fi.inject_computation_fault("nav", FaultType.CRASH)
        assert rec.injected
        assert rec.fault_type == FaultType.CRASH

    def test_inject_freeze(self):
        fi = FaultInjector()
        rec = fi.inject_computation_fault("pid", FaultType.FREEZE)
        assert rec.fault_type == FaultType.FREEZE


class TestInjectTimingFault:
    def test_inject_delay(self):
        fi = FaultInjector()
        rec = fi.inject_timing_fault("loop", 100.0)
        assert rec.fault_type == FaultType.DELAY
        assert rec.params["delay_ms"] == 100.0

    def test_timing_logged(self):
        fi = FaultInjector()
        fi.inject_timing_fault("x", 50)
        assert len(fi.get_injection_log()) == 1


class TestApplyFault:
    def test_no_fault_returns_value(self):
        fi = FaultInjector()
        assert fi.apply_fault("unknown", 42.0) == 42.0

    def test_stuck_at(self):
        fi = FaultInjector()
        fi.inject_sensor_fault("t", FaultType.STUCK_AT, {"stuck_value": 99.0})
        assert fi.apply_fault("t", 42.0) == 99.0

    def test_bias(self):
        fi = FaultInjector()
        fi.inject_sensor_fault("t", FaultType.BIAS, {"bias": 5.0})
        result = fi.apply_fault("t", 10.0)
        assert abs(result - 15.0) < 1e-9

    def test_out_of_bounds(self):
        fi = FaultInjector()
        fi.inject_sensor_fault("t", FaultType.OUT_OF_BOUNDS, {"invalid_value": 999999})
        assert fi.apply_fault("t", 50.0) == 999999.0

    def test_noise_changes_value(self):
        fi = FaultInjector(seed=42)
        fi.inject_sensor_fault("t", FaultType.NOISE, {"magnitude": 0.0})
        assert fi.apply_fault("t", 10.0) == 10.0  # magnitude 0 = no noise

    def test_noise_nonzero(self):
        fi = FaultInjector(seed=42)
        fi.inject_sensor_fault("t", FaultType.NOISE, {"magnitude": 10.0})
        # Should deviate from original
        vals = [fi.apply_fault("t", 10.0) for _ in range(10)]
        assert any(abs(v - 10.0) > 0.01 for v in vals)

    def test_bit_flip(self):
        fi = FaultInjector(seed=42)
        fi.inject_sensor_fault("t", FaultType.BIT_FLIP, {"bits": 1})
        result = fi.apply_fault("t", 10.0)  # 10 = 0b1010
        assert isinstance(result, float)


class TestClearFaults:
    def test_clear_existing(self):
        fi = FaultInjector()
        fi.inject_sensor_fault("s", FaultType.STUCK_AT)
        assert fi.clear_fault("s") is True
        assert fi.get_active_faults() == []

    def test_clear_nonexistent(self):
        fi = FaultInjector()
        assert fi.clear_fault("nope") is False

    def test_clear_all(self):
        fi = FaultInjector()
        fi.inject_sensor_fault("a", FaultType.STUCK_AT)
        fi.inject_sensor_fault("b", FaultType.BIAS)
        fi.clear_all_faults()
        assert fi.get_active_faults() == []


# ── FaultTestRunner ────────────────────────────────────────────────

class TestFaultTestRunnerConstruction:
    def test_construct(self):
        runner = FaultTestRunner()
        assert runner.get_results() == []

    def test_with_seed(self):
        runner = FaultTestRunner(seed=123)
        assert runner.get_results() == []


class TestRunScenario:
    def test_passing_scenario(self):
        runner = FaultTestRunner()
        scenario = FaultScenario(
            name="pass_test",
            faults=[FaultConfig(fault_type=FaultType.STUCK_AT, target="sensor_0")],
        )
        def healthy_system(injector):
            return {"recovered": True, "behavior": "ok"}
        result = runner.run_scenario(scenario, healthy_system)
        assert result.passed is True
        assert result.scenario_name == "pass_test"

    def test_failing_scenario(self):
        runner = FaultTestRunner()
        scenario = FaultScenario(name="fail_test")
        def broken_system(injector):
            return {"recovered": False, "behavior": "crashed"}
        result = runner.run_scenario(scenario, broken_system)
        assert result.passed is False

    def test_crashing_scenario(self):
        runner = FaultTestRunner()
        scenario = FaultScenario(name="crash_test")
        def crash_system(injector):
            raise RuntimeError("boom")
        result = runner.run_scenario(scenario, crash_system)
        assert result.passed is False
        assert "crashed" in result.observed_behavior

    def test_faults_injected_count(self):
        runner = FaultTestRunner()
        scenario = FaultScenario(
            name="count_test",
            faults=[
                FaultConfig(fault_type=FaultType.STUCK_AT, target="sensor_0"),
                FaultConfig(fault_type=FaultType.BIAS, target="sensor_1"),
            ],
        )
        def sys(inj): return {"recovered": True}
        result = runner.run_scenario(scenario, sys)
        assert result.faults_injected == 2

    def test_recovery_time_recorded(self):
        runner = FaultTestRunner()
        scenario = FaultScenario(name="time_test")
        def sys(inj): return {"recovered": True}
        result = runner.run_scenario(scenario, sys)
        assert result.recovery_time_ms >= 0

    def test_results_stored(self):
        runner = FaultTestRunner()
        scenario = FaultScenario(name="store_test")
        def sys(inj): return {"recovered": True}
        runner.run_scenario(scenario, sys)
        assert len(runner.get_results()) == 1

    def test_sensor_fault_target_routing(self):
        runner = FaultTestRunner()
        scenario = FaultScenario(
            name="sensor_route",
            faults=[FaultConfig(fault_type=FaultType.NOISE, target="sensor_pressure")],
        )
        injected_targets = []
        def check_sys(inj):
            injected_targets.extend([f.target for f in inj.get_active_faults()])
            return {"recovered": True}
        runner.run_scenario(scenario, check_sys)
        assert "sensor_pressure" in injected_targets

    def test_comm_fault_target_routing(self):
        runner = FaultTestRunner()
        scenario = FaultScenario(
            name="comm_route",
            faults=[FaultConfig(fault_type=FaultType.DROP, target="comm_link_1")],
        )
        def check_sys(inj):
            assert len(inj.get_active_faults()) >= 1
            return {"recovered": True}
        result = runner.run_scenario(scenario, check_sys)
        assert result.passed

    def test_computation_fault_routing(self):
        runner = FaultTestRunner()
        scenario = FaultScenario(
            name="comp_route",
            faults=[FaultConfig(fault_type=FaultType.CRASH, target="module_nav")],
        )
        def check_sys(inj):
            return {"recovered": True}
        result = runner.run_scenario(scenario, check_sys)
        assert result.passed

    def test_timing_fault_routing(self):
        runner = FaultTestRunner()
        scenario = FaultScenario(
            name="timing_route",
            faults=[FaultConfig(fault_type=FaultType.DELAY, target="timing_loop", params={"delay_ms": 50})],
        )
        def check_sys(inj):
            return {"recovered": True}
        result = runner.run_scenario(scenario, check_sys)
        assert result.passed


class TestGenerateScenarios:
    def test_default_count(self):
        runner = FaultTestRunner(seed=42)
        scenarios = runner.generate_scenarios({})
        assert len(scenarios) == 3

    def test_custom_count(self):
        runner = FaultTestRunner(seed=42)
        scenarios = runner.generate_scenarios({"count": 5})
        assert len(scenarios) == 5

    def test_faults_per_scenario(self):
        runner = FaultTestRunner(seed=42)
        scenarios = runner.generate_scenarios({"count": 1, "faults_per_scenario": 4})
        assert len(scenarios[0].faults) == 4

    def test_custom_targets(self):
        runner = FaultTestRunner(seed=42)
        scenarios = runner.generate_scenarios({
            "count": 1,
            "targets": ["custom_target"],
        })
        assert scenarios[0].faults[0].target == "custom_target"

    def test_scenario_names_unique(self):
        runner = FaultTestRunner(seed=42)
        scenarios = runner.generate_scenarios({"count": 5})
        names = [s.name for s in scenarios]
        assert len(set(names)) == len(names)


class TestMeasureRecoveryTime:
    def test_recovery_time(self):
        runner = FaultTestRunner()
        scenario = FaultScenario(name="rt_test")
        def sys(inj): return {"recovered": True}
        rt = runner.measure_recovery_time(scenario, sys)
        assert rt >= 0


class TestEvaluateRobustness:
    def test_all_pass(self):
        runner = FaultTestRunner()
        runner._results = [
            TestResult("a", True, 10.0),
            TestResult("b", True, 20.0),
        ]
        assert runner.evaluate_robustness() == 1.0

    def test_none_pass(self):
        runner = FaultTestRunner()
        runner._results = [
            TestResult("a", False, 10.0),
            TestResult("b", False, 20.0),
        ]
        assert runner.evaluate_robustness() == 0.0

    def test_half_pass(self):
        runner = FaultTestRunner()
        runner._results = [
            TestResult("a", True, 10.0),
            TestResult("b", False, 20.0),
        ]
        assert runner.evaluate_robustness() == 0.5

    def test_empty_results(self):
        runner = FaultTestRunner()
        assert runner.evaluate_robustness() == 0.0

    def test_custom_results(self):
        runner = FaultTestRunner()
        results = [TestResult(f"s{i}", True, 1.0) for i in range(3)]
        results.append(TestResult("f", False, 1.0))
        assert runner.evaluate_robustness(results) == 0.75
