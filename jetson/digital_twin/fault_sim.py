"""Fault simulation: inject failures, observe behavior for NEXUS digital twin."""

import math
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

from .physics import VesselState, Force, VesselPhysics


class FaultType(Enum):
    """Types of simulated faults."""
    SENSOR_DRIFT = "sensor_drift"
    SENSOR_STUCK = "sensor_stuck"
    SENSOR_NOISE = "sensor_noise"
    ACTUATOR_FAILURE = "actuator_failure"
    ACTUATOR_SATURATION = "actuator_saturation"
    COMMUNICATION_LOSS = "communication_loss"
    POWER_FAILURE = "power_failure"
    PROPULSION_LOSS = "propulsion_loss"
    NAVIGATION_ERROR = "navigation_error"
    LEAK = "leak"


class FailureMode(Enum):
    """Actuator failure modes."""
    STUCK = "stuck"
    OSCILLATING = "oscillating"
    REVERSED = "reversed"
    DEAD = "dead"


@dataclass
class SimulatedFault:
    """Defines a fault to inject into the simulation."""
    fault_type: FaultType
    target_system: str           # e.g., 'propulsion', 'steering', 'depth_sensor'
    start_time: float            # seconds into simulation
    duration: float              # seconds (0 = permanent until recovery)
    severity: float = 1.0        # 0.0 (minor) to 1.0 (catastrophic)

    def is_active(self, current_time: float) -> bool:
        if self.duration <= 0:
            return current_time >= self.start_time
        return self.start_time <= current_time <= self.start_time + self.duration

    def copy(self) -> 'SimulatedFault':
        return SimulatedFault(
            fault_type=self.fault_type,
            target_system=self.target_system,
            start_time=self.start_time,
            duration=self.duration,
            severity=self.severity,
        )


@dataclass
class FaultSimulationResult:
    """Results from a fault simulation."""
    fault: SimulatedFault
    system_response: Dict[str, Any] = field(default_factory=dict)
    recovery_time: float = 0.0
    impact_metrics: Dict[str, float] = field(default_factory=dict)

    def severity_score(self) -> float:
        """Compute overall severity score based on impact metrics."""
        pos_dev = self.impact_metrics.get('max_position_deviation', 0.0)
        vel_dev = self.impact_metrics.get('max_velocity_deviation', 0.0)
        energy = self.impact_metrics.get('energy_impact', 0.0)
        return min(1.0, (pos_dev / 50.0 + vel_dev / 5.0 + energy / 500.0) * self.fault.severity)


class FaultSimulator:
    """Simulates various faults and their effects on the digital twin."""

    def __init__(self, physics: VesselPhysics = None):
        self.physics = physics or VesselPhysics()

    def inject_fault(self, fault: SimulatedFault, twin_state: VesselState,
                     physics_model: VesselPhysics = None) -> FaultSimulationResult:
        """Inject a fault and simulate its effect on the twin state.
        Returns the simulation result."""
        phys = physics_model or self.physics
        response = {}
        impact = {}
        sim_duration = max(fault.duration, 5.0)

        original_state = twin_state.copy()
        current = twin_state.copy()
        trajectory = [current.copy()]

        # Simulate with fault
        t = 0.0
        dt = 0.1
        max_pos_dev = 0.0
        max_vel_dev = 0.0
        total_energy = 0.0

        while t < sim_duration:
            forces = []
            if fault.is_active(t):
                fault_force = self._apply_fault_force(fault, current, phys)
                forces.append(fault_force)
                response['fault_active'] = True
            else:
                response.get('fault_active', False)

            # Add drag
            drag = phys.compute_drag(
                (current.vx, current.vy, current.vz),
                (current.wx, state_wy := current.wy, current.wz)
            )
            forces.append(drag)

            current = phys.update_state(current, forces, dt)
            trajectory.append(current.copy())

            # Track deviations
            pos_dev = current.distance_to(original_state)
            vel_dev = abs(current.speed() - original_state.speed())
            max_pos_dev = max(max_pos_dev, pos_dev)
            max_vel_dev = max(max_vel_dev, vel_dev)
            total_energy += abs(forces[0].fx * current.vx * dt) if forces else 0

            t += dt

        recovery_time = self._estimate_recovery_time(fault, current, original_state)

        impact['max_position_deviation'] = max_pos_dev
        impact['max_velocity_deviation'] = max_vel_dev
        impact['energy_impact'] = total_energy
        impact['final_position_deviation'] = current.distance_to(original_state)
        impact['trajectory_length'] = len(trajectory)

        return FaultSimulationResult(
            fault=fault,
            system_response=response,
            recovery_time=recovery_time,
            impact_metrics=impact,
        )

    def simulate_sensor_drift(self, twin_state: VesselState, sensor_id: str,
                              drift_rate: float, duration: float) -> Dict[str, Any]:
        """Simulate sensor drift over time.
        Returns drifted sensor readings."""
        readings = []
        t = 0.0
        dt = 0.1
        current_drift = 0.0

        while t <= duration:
            current_drift += drift_rate * dt
            # Add realistic sensor noise
            noise = math.sin(t * 3.7) * 0.01 * drift_rate
            reading = {
                'timestamp': t,
                'sensor_id': sensor_id,
                'true_value': getattr(twin_state, 'x', 0.0),
                'measured_value': getattr(twin_state, 'x', 0.0) + current_drift + noise,
                'drift': current_drift,
                'noise': noise,
            }
            readings.append(reading)
            t += dt

        return {
            'sensor_id': sensor_id,
            'drift_rate': drift_rate,
            'duration': duration,
            'total_drift': current_drift,
            'readings': readings,
            'max_error': max(abs(r['measured_value'] - r['true_value']) for r in readings),
        }

    def simulate_actuator_failure(self, twin_state: VesselState, actuator_id: str,
                                   failure_mode: FailureMode) -> Dict[str, Any]:
        """Simulate actuator failure and observe response.
        Returns response characteristics."""
        response = {
            'actuator_id': actuator_id,
            'failure_mode': failure_mode.value,
            'time_to_detection': 0.5,  # typical detection time
            'affected_axes': [],
            'response': [],
        }

        if 'propulsion' in actuator_id.lower():
            response['affected_axes'].append('surge')
            response['response'].append('reduced_forward_thrust')
        if 'steering' in actuator_id.lower() or 'rudder' in actuator_id.lower():
            response['affected_axes'].append('yaw')
            response['response'].append('loss_of_directional_control')
        if 'depth' in actuator_id.lower():
            response['affected_axes'].append('heave')
            response['response'].append('loss_of_depth_control')

        # Simulate time response
        dt = 0.1
        t = 0.0
        current = twin_state.copy()
        trajectory = [current.copy()]

        for _ in range(100):  # 10 seconds
            fault_force = Force()
            if failure_mode == FailureMode.STUCK:
                fault_force = Force(fx=10.0)  # constant spurious force
            elif failure_mode == FailureMode.OSCILLATING:
                fault_force = Force(fx=20.0 * math.sin(t * 5.0))
            elif failure_mode == FailureMode.REVERSED:
                fault_force = Force(fx=-twin_state.vx * 2.0)
            elif failure_mode == FailureMode.DEAD:
                fault_force = Force()  # no force (actuator unresponsive)

            forces = [fault_force]
            drag = self.physics.compute_drag(
                (current.vx, current.vy, current.vz),
                (current.wx, current.wy, current.wz)
            )
            forces.append(drag)
            current = self.physics.update_state(current, forces, dt)
            trajectory.append(current.copy())
            t += dt

        response['trajectory_deviation'] = trajectory[-1].distance_to(twin_state)
        response['final_velocity_change'] = abs(trajectory[-1].speed() - twin_state.speed())
        response['simulation_time'] = t

        return response

    def simulate_communication_loss(self, twin_state: VesselState,
                                     duration: float) -> Dict[str, Any]:
        """Simulate communication loss and resulting degraded operations.
        Returns degraded operations result."""
        result = {
            'duration': duration,
            'degraded_systems': ['telemetry', 'remote_control', 'fleet_sync'],
            'autonomous_level': 'full',  # falls back to full autonomy
            'data_buffer_size': 0,
            'missed_updates': 0,
        }

        # Assume 10 Hz communication rate
        comm_rate = 10.0
        result['missed_updates'] = int(duration * comm_rate)
        # Buffer grows during outage
        result['data_buffer_size'] = result['missed_updates'] * 256  # bytes

        # Simulate vessel behavior without external commands
        dt = 0.1
        t = 0.0
        current = twin_state.copy()
        max_deviation = 0.0

        while t < duration:
            # Only drag acts (no new commands)
            drag = self.physics.compute_drag(
                (current.vx, current.vy, current.vz),
                (current.wx, current.wy, current.wz)
            )
            current = self.physics.update_state(current, [drag], dt)
            dev = current.distance_to(twin_state)
            max_deviation = max(max_deviation, dev)
            t += dt

        result['max_deviation'] = max_deviation
        result['vessel_stopped'] = current.speed() < 0.01
        result['safe_behavior'] = True  # vessel coasts safely

        return result

    def simulate_power_failure(self, twin_state: VesselState,
                               affected_systems: List[str]) -> Dict[str, Any]:
        """Simulate power failure in specified systems.
        Returns emergency response details."""
        response = {
            'affected_systems': affected_systems,
            'emergency_response': [],
            'remaining_systems': [],
            'critical_impact': False,
        }

        all_systems = ['propulsion', 'navigation', 'communication',
                       'sensors', 'computing', 'lighting', 'payload']
        response['remaining_systems'] = [s for s in all_systems if s not in affected_systems]

        # Determine emergency response
        if 'propulsion' in affected_systems:
            response['emergency_response'].append('deploy_surface_buoy')
            response['emergency_response'].append('activate_backup_thruster')
            response['critical_impact'] = True
        if 'navigation' in affected_systems:
            response['emergency_response'].append('switch_to_dead_reckoning')
            response['emergency_response'].append('activate_backup_gps')
        if 'communication' in affected_systems:
            response['emergency_response'].append('activate_sos_beacon')
            response['emergency_response'].append('log_to_local_storage')
        if 'computing' in affected_systems:
            response['emergency_response'].append('switch_to_low_power_mode')
            response['critical_impact'] = True
        if 'sensors' in affected_systems:
            response['emergency_response'].append('redundant_sensor_fallback')

        if not response['emergency_response']:
            response['emergency_response'].append('monitor_and_log')

        # Simulate vessel with reduced power
        dt = 0.1
        t = 0.0
        current = twin_state.copy()
        trajectory = [current.copy()]

        for _ in range(100):  # 10 seconds
            forces = []
            # Reduced thrust if propulsion partially available
            if 'propulsion' in affected_systems:
                forces.append(Force(fx=5.0))  # minimal emergency thrust
            else:
                forces.append(self.physics.compute_thrust(0.1, twin_state.yaw, 50.0))

            drag = self.physics.compute_drag(
                (current.vx, current.vy, current.vz),
                (current.wx, current.wy, current.wz)
            )
            forces.append(drag)
            current = self.physics.update_state(current, forces, dt)
            trajectory.append(current.copy())
            t += dt

        response['trajectory_after_failure'] = trajectory
        response['final_speed'] = current.speed()
        response['simulation_duration'] = t

        return response

    def batch_simulate(self, faults: List[SimulatedFault],
                       twin_state: VesselState) -> List[FaultSimulationResult]:
        """Run multiple fault simulations.
        Returns list of results for each fault."""
        results = []
        for fault in faults:
            result = self.inject_fault(fault, twin_state)
            results.append(result)
        return results

    def rank_faults_by_severity(self, results: List[FaultSimulationResult]) -> List[FaultSimulationResult]:
        """Rank fault simulation results by severity (most severe first)."""
        return sorted(results, key=lambda r: r.severity_score(), reverse=True)

    def _apply_fault_force(self, fault: SimulatedFault, state: VesselState,
                            phys: VesselPhysics) -> Force:
        """Generate force perturbation for a given fault."""
        severity = fault.severity

        if fault.fault_type == FaultType.SENSOR_DRIFT:
            # Sensor drift doesn't directly apply force but causes control errors
            return Force(fx=severity * 5.0 * math.sin(state.yaw))
        elif fault.fault_type == FaultType.ACTUATOR_FAILURE:
            # Actuator failure reduces/eliminates intended force
            return Force(fx=-severity * 50.0 * math.cos(state.yaw))
        elif fault.fault_type == FaultType.PROPULSION_LOSS:
            return Force(fx=-severity * 100.0 * math.cos(state.yaw),
                        fy=-severity * 100.0 * math.sin(state.yaw))
        elif fault.fault_type == FaultType.NAVIGATION_ERROR:
            # Wrong heading leads to wrong thrust direction
            error = severity * 0.5  # radians
            return Force(fx=severity * 20.0 * math.cos(state.yaw + error))
        elif fault.fault_type == FaultType.LEAK:
            # Leak causes sinking
            return Force(fz=-severity * 50.0)
        elif fault.fault_type == FaultType.COMMUNICATION_LOSS:
            return Force()  # No direct force, handled separately
        elif fault.fault_type == FaultType.POWER_FAILURE:
            return Force()  # No force, systems shut down
        else:
            return Force()

    def _estimate_recovery_time(self, fault: SimulatedFault,
                                 final_state: VesselState,
                                 original_state: VesselState) -> float:
        """Estimate time to recover from a fault."""
        deviation = final_state.distance_to(original_state)
        base_recovery_speed = 2.0  # m/s recovery speed

        if fault.fault_type == FaultType.SENSOR_DRIFT:
            # Software fix, quick recovery
            return 1.0 + deviation / (base_recovery_speed * 5)
        elif fault.fault_type == FaultType.ACTUATOR_FAILURE:
            # Hardware recovery needed
            return 10.0 + deviation / base_recovery_speed
        elif fault.fault_type == FaultType.PROPULSION_LOSS:
            return 30.0 + deviation / (base_recovery_speed * 0.5)
        elif fault.fault_type == FaultType.NAVIGATION_ERROR:
            return 2.0 + deviation / (base_recovery_speed * 3)
        elif fault.fault_type == FaultType.LEAK:
            return 60.0 + deviation / (base_recovery_speed * 0.3)
        else:
            return 5.0 + deviation / base_recovery_speed
