"""Tests for fault simulation."""

import math
import pytest
from jetson.digital_twin.fault_sim import (
    FaultType, FailureMode, SimulatedFault, FaultSimulationResult, FaultSimulator
)
from jetson.digital_twin.physics import VesselState, Force


class TestFaultType:
    def test_all_values(self):
        types = list(FaultType)
        assert len(types) >= 10
        assert FaultType.SENSOR_DRIFT in types
        assert FaultType.PROPULSION_LOSS in types
        assert FaultType.LEAK in types

    def test_string_values(self):
        assert FaultType.ACTUATOR_FAILURE.value == "actuator_failure"
        assert FaultType.COMMUNICATION_LOSS.value == "communication_loss"


class TestFailureMode:
    def test_all_values(self):
        modes = list(FailureMode)
        assert len(modes) == 4
        assert FailureMode.STUCK in modes
        assert FailureMode.OSCILLATING in modes
        assert FailureMode.REVERSED in modes
        assert FailureMode.DEAD in modes


class TestSimulatedFault:
    def test_defaults(self):
        f = SimulatedFault(fault_type=FaultType.SENSOR_DRIFT, target_system="gps",
                           start_time=5.0, duration=10.0)
        assert f.severity == 1.0
        assert f.fault_type == FaultType.SENSOR_DRIFT

    def test_is_active_before_start(self):
        f = SimulatedFault(fault_type=FaultType.SENSOR_DRIFT, target_system="gps",
                           start_time=5.0, duration=10.0)
        assert f.is_active(0.0) is False

    def test_is_active_during(self):
        f = SimulatedFault(fault_type=FaultType.SENSOR_DRIFT, target_system="gps",
                           start_time=5.0, duration=10.0)
        assert f.is_active(7.0) is True
        assert f.is_active(5.0) is True
        assert f.is_active(15.0) is True

    def test_is_active_after_end(self):
        f = SimulatedFault(fault_type=FaultType.SENSOR_DRIFT, target_system="gps",
                           start_time=5.0, duration=10.0)
        assert f.is_active(16.0) is False

    def test_is_active_permanent(self):
        f = SimulatedFault(fault_type=FaultType.SENSOR_DRIFT, target_system="gps",
                           start_time=5.0, duration=0.0)
        assert f.is_active(100.0) is True
        assert f.is_active(4.0) is False

    def test_custom_severity(self):
        f = SimulatedFault(fault_type=FaultType.SENSOR_DRIFT, target_system="gps",
                           start_time=0, duration=1, severity=0.5)
        assert f.severity == 0.5

    def test_copy(self):
        f = SimulatedFault(fault_type=FaultType.LEAK, target_system="hull",
                           start_time=1.0, duration=10.0, severity=0.8)
        c = f.copy()
        assert c.fault_type == FaultType.LEAK
        assert c.severity == 0.8
        c.severity = 0.1
        assert f.severity == 0.8


class TestFaultSimulationResult:
    def test_defaults(self):
        r = FaultSimulationResult(fault=SimulatedFault(
            fault_type=FaultType.SENSOR_DRIFT,
            target_system="gps", start_time=0, duration=1))
        assert r.recovery_time == 0.0
        assert r.system_response == {}
        assert r.impact_metrics == {}

    def test_severity_score_zero(self):
        r = FaultSimulationResult(
            fault=SimulatedFault(
                fault_type=FaultType.SENSOR_DRIFT,
                target_system="gps", start_time=0, duration=1),
            impact_metrics={'max_position_deviation': 0, 'max_velocity_deviation': 0, 'energy_impact': 0},
        )
        assert r.severity_score() == 0.0

    def test_severity_score_high(self):
        r = FaultSimulationResult(
            fault=SimulatedFault(
                fault_type=FaultType.SENSOR_DRIFT,
                target_system="gps", start_time=0, duration=1, severity=1.0),
            impact_metrics={'max_position_deviation': 50, 'max_velocity_deviation': 5, 'energy_impact': 500},
        )
        score = r.severity_score()
        assert score > 0


class TestFaultSimulator:
    def setup_method(self):
        self.sim = FaultSimulator()

    def test_inject_sensor_drift(self):
        fault = SimulatedFault(
            fault_type=FaultType.SENSOR_DRIFT, target_system="gps",
            start_time=0.0, duration=5.0, severity=0.5)
        state = VesselState()
        result = self.sim.inject_fault(fault, state)
        assert isinstance(result, FaultSimulationResult)
        assert result.impact_metrics['max_position_deviation'] >= 0

    def test_inject_actuator_failure(self):
        fault = SimulatedFault(
            fault_type=FaultType.ACTUATOR_FAILURE, target_system="steering",
            start_time=0.0, duration=5.0)
        state = VesselState()
        result = self.sim.inject_fault(fault, state)
        assert result.recovery_time > 0

    def test_inject_propulsion_loss(self):
        fault = SimulatedFault(
            fault_type=FaultType.PROPULSION_LOSS, target_system="propulsion",
            start_time=0.0, duration=5.0, severity=1.0)
        state = VesselState()
        result = self.sim.inject_fault(fault, state)
        assert result.recovery_time > 0
        assert result.impact_metrics['max_position_deviation'] >= 0

    def test_inject_navigation_error(self):
        fault = SimulatedFault(
            fault_type=FaultType.NAVIGATION_ERROR, target_system="gps",
            start_time=0.0, duration=5.0)
        state = VesselState()
        result = self.sim.inject_fault(fault, state)
        assert isinstance(result, FaultSimulationResult)

    def test_inject_leak(self):
        fault = SimulatedFault(
            fault_type=FaultType.LEAK, target_system="hull",
            start_time=0.0, duration=5.0, severity=0.5)
        state = VesselState()
        result = self.sim.inject_fault(fault, state)
        assert result.recovery_time > 0

    def test_inject_communication_loss(self):
        fault = SimulatedFault(
            fault_type=FaultType.COMMUNICATION_LOSS, target_system="comms",
            start_time=0.0, duration=5.0)
        state = VesselState()
        result = self.sim.inject_fault(fault, state)
        assert isinstance(result, FaultSimulationResult)

    def test_inject_power_failure(self):
        fault = SimulatedFault(
            fault_type=FaultType.POWER_FAILURE, target_system="main_bus",
            start_time=0.0, duration=5.0)
        state = VesselState()
        result = self.sim.inject_fault(fault, state)
        assert isinstance(result, FaultSimulationResult)

    def test_simulate_sensor_drift(self):
        state = VesselState(x=10.0)
        result = self.sim.simulate_sensor_drift(state, "gps", drift_rate=0.1, duration=5.0)
        assert result['sensor_id'] == "gps"
        assert result['total_drift'] > 0
        assert result['duration'] == 5.0
        assert len(result['readings']) > 0
        assert result['max_error'] > 0

    def test_simulate_sensor_drift_zero_rate(self):
        state = VesselState(x=10.0)
        result = self.sim.simulate_sensor_drift(state, "gps", drift_rate=0.0, duration=5.0)
        assert result['total_drift'] == 0.0

    def test_simulate_sensor_drift_negative_rate(self):
        state = VesselState(x=10.0)
        result = self.sim.simulate_sensor_drift(state, "compass", drift_rate=-0.05, duration=5.0)
        assert result['total_drift'] < 0

    def test_simulate_actuator_stuck(self):
        state = VesselState(vx=2.0)
        result = self.sim.simulate_actuator_failure(state, "propulsion", FailureMode.STUCK)
        assert result['failure_mode'] == 'stuck'
        assert result['trajectory_deviation'] >= 0
        assert 'surge' in result['affected_axes']

    def test_simulate_actuator_oscillating(self):
        state = VesselState()
        result = self.sim.simulate_actuator_failure(state, "steering", FailureMode.OSCILLATING)
        assert result['failure_mode'] == 'oscillating'

    def test_simulate_actuator_reversed(self):
        state = VesselState(vx=3.0)
        result = self.sim.simulate_actuator_failure(state, "propulsion", FailureMode.REVERSED)
        assert result['failure_mode'] == 'reversed'

    def test_simulate_actuator_dead(self):
        state = VesselState(vx=2.0)
        result = self.sim.simulate_actuator_failure(state, "propulsion", FailureMode.DEAD)
        assert result['failure_mode'] == 'dead'

    def test_simulate_actuator_depth(self):
        state = VesselState()
        result = self.sim.simulate_actuator_failure(state, "depth_control", FailureMode.STUCK)
        assert 'heave' in str(result['affected_axes'])

    def test_simulate_communication_loss(self):
        state = VesselState(vx=2.0)
        result = self.sim.simulate_communication_loss(state, duration=10.0)
        assert result['duration'] == 10.0
        assert result['missed_updates'] == 100  # 10s * 10Hz
        assert result['safe_behavior'] is True
        assert result['max_deviation'] >= 0

    def test_simulate_communication_loss_zero_duration(self):
        state = VesselState()
        result = self.sim.simulate_communication_loss(state, duration=0.0)
        assert result['missed_updates'] == 0
        assert result['max_deviation'] == 0

    def test_simulate_power_failure_propulsion(self):
        state = VesselState(vx=2.0)
        result = self.sim.simulate_power_failure(state, ['propulsion'])
        assert result['critical_impact'] is True
        assert 'deploy_surface_buoy' in result['emergency_response']
        assert len(result['trajectory_after_failure']) > 0

    def test_simulate_power_failure_navigation(self):
        state = VesselState()
        result = self.sim.simulate_power_failure(state, ['navigation'])
        assert 'switch_to_dead_reckoning' in result['emergency_response']

    def test_simulate_power_failure_communication(self):
        state = VesselState()
        result = self.sim.simulate_power_failure(state, ['communication'])
        assert 'activate_sos_beacon' in result['emergency_response']

    def test_simulate_power_failure_multiple(self):
        state = VesselState()
        result = self.sim.simulate_power_failure(
            state, ['propulsion', 'navigation', 'communication'])
        assert result['critical_impact'] is True
        assert len(result['emergency_response']) > 3

    def test_simulate_power_failure_no_critical(self):
        state = VesselState()
        result = self.sim.simulate_power_failure(state, ['lighting'])
        assert result['critical_impact'] is False

    def test_batch_simulate(self):
        faults = [
            SimulatedFault(fault_type=FaultType.SENSOR_DRIFT, target_system="gps",
                           start_time=0, duration=2, severity=0.5),
            SimulatedFault(fault_type=FaultType.PROPULSION_LOSS, target_system="propulsion",
                           start_time=0, duration=2, severity=0.5),
        ]
        state = VesselState()
        results = self.sim.batch_simulate(faults, state)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, FaultSimulationResult)

    def test_rank_faults_by_severity(self):
        results = [
            FaultSimulationResult(
                fault=SimulatedFault(fault_type=FaultType.SENSOR_DRIFT, target_system="a",
                                     start_time=0, duration=1, severity=0.1),
                impact_metrics={'max_position_deviation': 1, 'max_velocity_deviation': 0, 'energy_impact': 0}),
            FaultSimulationResult(
                fault=SimulatedFault(fault_type=FaultType.PROPULSION_LOSS, target_system="b",
                                     start_time=0, duration=1, severity=1.0),
                impact_metrics={'max_position_deviation': 30, 'max_velocity_deviation': 3, 'energy_impact': 200}),
        ]
        ranked = self.sim.rank_faults_by_severity(results)
        assert ranked[0].severity_score() >= ranked[1].severity_score()

    def test_rank_faults_empty(self):
        ranked = self.sim.rank_faults_by_severity([])
        assert ranked == []

    def test_rank_faults_single(self):
        r = FaultSimulationResult(
            fault=SimulatedFault(fault_type=FaultType.SENSOR_NOISE, target_system="x",
                                 start_time=0, duration=1),
            impact_metrics={'max_position_deviation': 0.1, 'max_velocity_deviation': 0, 'energy_impact': 0},
        )
        ranked = self.sim.rank_faults_by_severity([r])
        assert len(ranked) == 1

    def test_inject_fault_with_moving_vessel(self):
        fault = SimulatedFault(
            fault_type=FaultType.SENSOR_DRIFT, target_system="imu",
            start_time=0, duration=5, severity=0.8)
        state = VesselState(vx=3, vy=2)
        result = self.sim.inject_fault(fault, state)
        assert result.impact_metrics['trajectory_length'] > 0

    def test_inject_fault_permanent(self):
        fault = SimulatedFault(
            fault_type=FaultType.SENSOR_STUCK, target_system="depth",
            start_time=0, duration=0, severity=1.0)
        state = VesselState()
        result = self.sim.inject_fault(fault, state)
        # Permanent fault should have longer recovery
        assert result.recovery_time > 0

    def test_sensor_drift_readings_structure(self):
        state = VesselState(x=100.0)
        result = self.sim.simulate_sensor_drift(state, "pressure", 0.2, 3.0)
        for reading in result['readings']:
            assert 'timestamp' in reading
            assert 'measured_value' in reading
            assert 'true_value' in reading
            assert 'drift' in reading
