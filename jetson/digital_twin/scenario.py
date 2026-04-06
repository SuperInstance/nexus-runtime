"""What-if analysis: scenario replay, parameter sweep for NEXUS digital twin."""

import math
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from .physics import VesselState, Force, VesselPhysics, DragCoefficients, VesselProperties
from .environment import EnvironmentModel, WindField, CurrentField, WaveField, EnvironmentConditions


@dataclass
class Scenario:
    """Defines a simulation scenario."""
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    initial_state: VesselState = field(default_factory=VesselState)
    environment_config: Dict[str, Any] = field(default_factory=dict)
    duration: float = 60.0  # seconds
    dt: float = 0.1         # time step

    def copy(self) -> 'Scenario':
        return Scenario(
            name=self.name,
            parameters=dict(self.parameters),
            initial_state=self.initial_state.copy(),
            environment_config=dict(self.environment_config),
            duration=self.duration,
            dt=self.dt,
        )


@dataclass
class ScenarioResult:
    """Results from running a scenario."""
    scenario: Scenario
    trajectory: List[VesselState] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    success: bool = True
    error_message: str = ""

    def final_state(self) -> VesselState:
        return self.trajectory[-1] if self.trajectory else VesselState()

    def total_distance(self) -> float:
        dist = 0.0
        for i in range(1, len(self.trajectory)):
            dist += self.trajectory[i].distance_to(self.trajectory[i-1])
        return dist

    def max_speed(self) -> float:
        return max((s.speed() for s in self.trajectory), default=0.0)

    def avg_speed(self) -> float:
        if not self.trajectory:
            return 0.0
        return sum(s.speed() for s in self.trajectory) / len(self.trajectory)


class ScenarioRunner:
    """Runs simulation scenarios and performs what-if analysis."""

    def __init__(self, physics: VesselPhysics = None):
        self.physics = physics or VesselPhysics()
        self._saved_scenarios: Dict[str, Scenario] = {}

    def run(self, scenario: Scenario) -> ScenarioResult:
        """Run a single scenario and return results."""
        try:
            env = self._build_environment(scenario.environment_config)
            state = scenario.initial_state.copy()
            trajectory = [state.copy()]
            t = 0.0
            steps = int(scenario.duration / scenario.dt)

            thrust = scenario.parameters.get('thrust', 0.0)
            heading = scenario.parameters.get('heading', 0.0)

            for _ in range(steps):
                # Compute forces
                forces = []

                # Thrust
                thrust_force = self.physics.compute_thrust(thrust, heading)
                forces.append(thrust_force)

                # Environmental forces
                env_force = env.total_environmental_force(
                    state, (state.x, state.y, state.z)
                )
                forces.append(env_force)

                # Drag
                drag = self.physics.compute_drag(
                    (state.vx, state.vy, state.vz),
                    (state.wx, state.wy, state.wz)
                )
                forces.append(drag)

                # Update
                state = self.physics.update_state(state, forces, scenario.dt)
                t += scenario.dt
                env.step(scenario.dt)
                trajectory.append(state.copy())

            # Compute metrics
            metrics = self._compute_metrics(scenario, trajectory)

            return ScenarioResult(
                scenario=scenario,
                trajectory=trajectory,
                metrics=metrics,
                success=True,
            )
        except Exception as e:
            return ScenarioResult(
                scenario=scenario,
                trajectory=[scenario.initial_state.copy()],
                metrics={},
                success=False,
                error_message=str(e),
            )

    def parameter_sweep(self, base_scenario: Scenario,
                        parameters_to_vary: Dict[str, List[Any]]) -> List[ScenarioResult]:
        """Run scenario with multiple parameter combinations.
        parameters_to_vary: {param_name: [value1, value2, ...]}"""
        results = []

        if not parameters_to_vary:
            return [self.run(base_scenario)]

        # Get first parameter to sweep
        param_name = list(parameters_to_vary.keys())[0]
        remaining = {k: v for k, v in parameters_to_vary.items() if k != param_name}

        for value in parameters_to_vary[param_name]:
            variant = base_scenario.copy()
            variant.parameters[param_name] = value
            variant.name = f"{base_scenario.name}_{param_name}={value}"

            if remaining:
                sub_results = self.parameter_sweep(variant, remaining)
                results.extend(sub_results)
            else:
                results.append(self.run(variant))

        return results

    def compare_results(self, results: List[ScenarioResult],
                        metric: str) -> List[Tuple[ScenarioResult, float]]:
        """Rank results by a specific metric (lower is better for distance/error,
        higher is better for speed/efficiency)."""
        ranked = []
        for r in results:
            value = r.metrics.get(metric, 0.0)
            ranked.append((r, value))

        # Sort ascending; metrics like efficiency where higher=better use negative
        ascending_metrics = {'max_position_error', 'drift', 'energy_used', 'max_acceleration'}
        if metric in ascending_metrics:
            ranked.sort(key=lambda x: x[1])
        else:
            ranked.sort(key=lambda x: -x[1])

        return ranked

    def replay_scenario(self, saved_state: Dict[str, Any]) -> ScenarioResult:
        """Replay a previously saved scenario state."""
        scenario = self.import_scenario(saved_state)
        return self.run(scenario)

    def export_scenario(self, scenario: Scenario) -> Dict[str, Any]:
        """Export scenario to serializable dict."""
        return {
            'name': scenario.name,
            'parameters': scenario.parameters,
            'initial_state': {
                'x': scenario.initial_state.x,
                'y': scenario.initial_state.y,
                'z': scenario.initial_state.z,
                'vx': scenario.initial_state.vx,
                'vy': scenario.initial_state.vy,
                'vz': scenario.initial_state.vz,
                'roll': scenario.initial_state.roll,
                'pitch': scenario.initial_state.pitch,
                'yaw': scenario.initial_state.yaw,
                'wx': scenario.initial_state.wx,
                'wy': scenario.initial_state.wy,
                'wz': scenario.initial_state.wz,
            },
            'environment_config': scenario.environment_config,
            'duration': scenario.duration,
            'dt': scenario.dt,
        }

    def import_scenario(self, data: Dict[str, Any]) -> Scenario:
        """Import scenario from serializable dict."""
        init = data.get('initial_state', {})
        return Scenario(
            name=data.get('name', 'imported'),
            parameters=data.get('parameters', {}),
            initial_state=VesselState(
                x=init.get('x', 0.0), y=init.get('y', 0.0), z=init.get('z', 0.0),
                vx=init.get('vx', 0.0), vy=init.get('vy', 0.0), vz=init.get('vz', 0.0),
                roll=init.get('roll', 0.0), pitch=init.get('pitch', 0.0),
                yaw=init.get('yaw', 0.0),
                wx=init.get('wx', 0.0), wy=init.get('wy', 0.0), wz=init.get('wz', 0.0),
            ),
            environment_config=data.get('environment_config', {}),
            duration=data.get('duration', 60.0),
            dt=data.get('dt', 0.1),
        )

    def save_scenario(self, name: str, scenario: Scenario) -> None:
        """Save scenario for later replay."""
        self._saved_scenarios[name] = scenario.copy()

    def load_scenario(self, name: str) -> Optional[Scenario]:
        """Load previously saved scenario."""
        return self._saved_scenarios.get(name)

    def list_saved_scenarios(self) -> List[str]:
        """List names of saved scenarios."""
        return list(self._saved_scenarios.keys())

    def _build_environment(self, config: Dict[str, Any]) -> EnvironmentModel:
        """Build environment model from config dict."""
        wind_cfg = config.get('wind', {})
        current_cfg = config.get('current', {})
        wave_cfg = config.get('wave', {})

        wind = WindField(
            speed=wind_cfg.get('speed', 5.0),
            direction=wind_cfg.get('direction', 0.0),
            gust_speed=wind_cfg.get('gust_speed', 15.0),
            gust_probability=wind_cfg.get('gust_probability', 0.1),
        )
        current = CurrentField(
            speed=current_cfg.get('speed', 0.5),
            direction=current_cfg.get('direction', 0.0),
            depth_profile=current_cfg.get('depth_profile', 'uniform'),
        )
        wave = WaveField(
            height=wave_cfg.get('height', 1.0),
            period=wave_cfg.get('period', 8.0),
            direction=wave_cfg.get('direction', 0.0),
            spectrum_type=wave_cfg.get('spectrum_type', 'pierson_moskowitz'),
        )
        return EnvironmentModel(wind=wind, current=current, wave=wave)

    def _compute_metrics(self, scenario: Scenario,
                         trajectory: List[VesselState]) -> Dict[str, float]:
        """Compute performance metrics from trajectory."""
        if len(trajectory) < 2:
            return {}

        total_dist = 0.0
        max_speed = 0.0
        speeds = []
        for i in range(1, len(trajectory)):
            total_dist += trajectory[i].distance_to(trajectory[i-1])
            s = trajectory[i].speed()
            speeds.append(s)
            if s > max_speed:
                max_speed = s

        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

        # Max acceleration
        max_accel = 0.0
        for i in range(1, len(trajectory)):
            ax = (trajectory[i].vx - trajectory[i-1].vx) / scenario.dt
            ay = (trajectory[i].vy - trajectory[i-1].vy) / scenario.dt
            az = (trajectory[i].vz - trajectory[i-1].vz) / scenario.dt
            accel = math.sqrt(ax*ax + ay*ay + az*az)
            if accel > max_accel:
                max_accel = accel

        # Position drift from initial
        final = trajectory[-1]
        initial = trajectory[0]
        pos_drift = final.distance_to(initial)

        # Max orientation change
        max_roll = max(abs(s.roll) for s in trajectory)
        max_pitch = max(abs(s.pitch) for s in trajectory)

        return {
            'total_distance': total_dist,
            'max_speed': max_speed,
            'avg_speed': avg_speed,
            'max_acceleration': max_accel,
            'final_x': final.x,
            'final_y': final.y,
            'position_drift': pos_drift,
            'max_roll': max_roll,
            'max_pitch': max_pitch,
            'trajectory_length': len(trajectory),
        }
