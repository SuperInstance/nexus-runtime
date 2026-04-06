"""Environmental modeling: wind, current, waves for NEXUS digital twin."""

import math
import random
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional


@dataclass
class WindField:
    """Wind field parameters."""
    speed: float = 5.0          # m/s
    direction: float = 0.0      # radians, meteorological convention (from)
    gust_speed: float = 15.0    # m/s
    gust_probability: float = 0.1  # 0..1

    def copy(self) -> 'WindField':
        return WindField(speed=self.speed, direction=self.direction,
                         gust_speed=self.gust_speed, gust_probability=self.gust_probability)


@dataclass
class CurrentField:
    """Ocean current field parameters."""
    speed: float = 0.5          # m/s
    direction: float = 0.0      # radians
    depth_profile: str = 'uniform'  # uniform, surface, logarithmic

    def copy(self) -> 'CurrentField':
        return CurrentField(speed=self.speed, direction=self.direction,
                            depth_profile=self.depth_profile)


@dataclass
class WaveField:
    """Wave field parameters."""
    height: float = 1.0         # significant wave height (m)
    period: float = 8.0         # peak period (s)
    direction: float = 0.0      # radians
    spectrum_type: str = 'pierson_moskowitz'  # pierson_moskowitz, jONSWAP, flat

    def copy(self) -> 'WaveField':
        return WaveField(height=self.height, period=self.period,
                         direction=self.direction, spectrum_type=self.spectrum_type)

    def wave_number(self, depth: float = 10.0) -> float:
        """Compute wave number k from dispersion relation: omega^2 = g*k*tanh(k*d)."""
        omega = 2.0 * math.pi / self.period
        g = 9.81
        # Deep water approximation as initial guess
        k = omega * omega / g
        # Newton iteration for dispersion relation
        for _ in range(20):
            f = omega*omega - g * k * math.tanh(k * depth)
            df = -g * (math.tanh(k * depth) + k * depth * (1.0 - math.tanh(k * depth)**2))
            if abs(df) < 1e-12:
                break
            k -= f / df
        return max(k, 1e-6)

    def wave_length(self, depth: float = 10.0) -> float:
        """Compute wave length L = 2*pi/k."""
        k = self.wave_number(depth)
        return 2.0 * math.pi / k

    def orbital_velocity(self, depth: float, z: float = 0.0) -> Tuple[float, float]:
        """Compute wave orbital velocity at depth z (z=0 at surface, negative down)."""
        omega = 2.0 * math.pi / self.period
        k = self.wave_number(depth)
        amp = self.height / 2.0
        # Exponential decay with depth
        decay = math.exp(k * min(z, 0.0))
        u = amp * omega * decay * math.cos(self.direction)
        v = amp * omega * decay * math.sin(self.direction)
        return u, v


@dataclass
class EnvironmentConditions:
    """Combined environmental conditions at a point."""
    wind_speed: float = 0.0
    wind_direction: float = 0.0
    current_speed: float = 0.0
    current_direction: float = 0.0
    wave_height: float = 0.0
    wave_direction: float = 0.0
    wave_period: float = 0.0
    temperature: float = 15.0  # water temperature (C)
    salinity: float = 35.0     # ppt


class EnvironmentModel:
    """Models wind, current, and wave fields for vessel simulation."""

    def __init__(self, wind: WindField = None, current: CurrentField = None,
                 wave: WaveField = None):
        self.wind = wind or WindField()
        self.current = current or CurrentField()
        self.wave = wave or WaveField()
        self.time = 0.0
        self._rng = random.Random(42)

    def get_wind_at(self, position: Tuple[float, float, float],
                    time: float = None) -> Tuple[float, float]:
        """Get wind speed and direction at a position and time.
        Returns (speed, direction)."""
        t = time if time is not None else self.time
        base_speed = self.wind.speed
        base_dir = self.wind.direction

        # Add slow temporal variation (sinusoidal)
        speed_variation = 1.0 + 0.1 * math.sin(2.0 * math.pi * t / 600.0)
        dir_variation = 0.05 * math.sin(2.0 * math.pi * t / 1200.0)

        speed = base_speed * speed_variation
        direction = base_dir + dir_variation

        # Random gusts
        if self._rng.random() < self.wind.gust_probability:
            gust_factor = 1.0 + (self.wind.gust_speed - base_speed) / max(base_speed, 0.01)
            speed *= gust_factor
            direction += self._rng.uniform(-0.2, 0.2)

        return speed, direction

    def get_current_at(self, position: Tuple[float, float, float],
                       depth: float = 0.0,
                       time: float = None) -> Tuple[float, float]:
        """Get current speed and direction at position/depth/time.
        Returns (speed, direction)."""
        t = time if time is not None else self.time
        base_speed = self.current.speed
        base_dir = self.current.direction

        if self.current.depth_profile == 'uniform':
            speed = base_speed
        elif self.current.depth_profile == 'surface':
            # Exponential decay with depth
            speed = base_speed * math.exp(-abs(depth) / 50.0)
        elif self.current.depth_profile == 'logarithmic':
            # Logarithmic boundary layer
            z0 = 0.01  # roughness length
            speed = base_speed * math.log(max(abs(depth), z0) / z0) / math.log(10.0 / z0)
        else:
            speed = base_speed

        # Temporal variation
        speed *= 1.0 + 0.05 * math.sin(2.0 * math.pi * t / 3600.0)
        direction = base_dir + 0.02 * math.sin(2.0 * math.pi * t / 7200.0)

        return max(speed, 0.0), direction

    def get_wave_at(self, position: Tuple[float, float, float],
                    time: float = None) -> Tuple[float, float, float]:
        """Get wave height, direction, period at a position and time.
        Returns (height, direction, period)."""
        t = time if time is not None else self.time
        height = self.wave.height
        direction = self.wave.direction
        period = self.wave.period

        # Slow swell variation
        height *= 1.0 + 0.15 * math.sin(2.0 * math.pi * t / 1800.0)
        direction += 0.1 * math.sin(2.0 * math.pi * t / 3600.0)

        return max(height, 0.0), direction, period

    def compute_wind_force(self, state, wind: Tuple[float, float]) -> 'Force':
        """Compute wind force on vessel given state and wind (speed, dir).
        Uses drag equation: F = 0.5 * rho * Cd * A * V^2."""
        from .physics import Force
        wind_speed, wind_dir = wind
        rho_air = 1.225  # kg/m^3
        # Simplified windage coefficients for a small vessel
        cd = 0.8
        exposed_area = 2.0  # m^2 above water

        # Relative wind (subtract vessel velocity)
        rel_vx = wind_speed * math.cos(wind_dir) - state.vx
        rel_vy = wind_speed * math.sin(wind_dir) - state.vy
        rel_speed = math.sqrt(rel_vx**2 + rel_vy**2)

        force_mag = 0.5 * rho_air * cd * exposed_area * rel_speed * abs(rel_speed)
        if rel_speed < 1e-8:
            return Force()

        fx = force_mag * rel_vx / rel_speed
        fy = force_mag * rel_vy / rel_speed

        # Wind torque depends on center of pressure offset
        torque_z = 0.5 * force_mag * math.sin(wind_dir - state.yaw)

        return Force(fx=fx, fy=fy, fz=0.0, torque_z=torque_z)

    def compute_current_effect(self, velocity: Tuple[float, float, float],
                               current: Tuple[float, float]) -> Tuple[float, float, float]:
        """Compute effective velocity considering current drift.
        Returns effective (vx, vy, vz)."""
        vx, vy, vz = velocity
        curr_speed, curr_dir = current
        curr_vx = curr_speed * math.cos(curr_dir)
        curr_vy = curr_speed * math.sin(curr_dir)
        return vx + curr_vx, vy + curr_vy, vz

    def compute_wave_force(self, state, waves: Tuple[float, float, float]) -> 'Force':
        """Compute wave excitation force on vessel.
        Simplified Morison-equation inspired model."""
        from .physics import Force
        wave_height, wave_dir, wave_period = waves
        rho_water = 1025.0  # kg/m^3

        # Simplified wave force
        omega = 2.0 * math.pi / max(wave_period, 0.1)
        k = omega * omega / 9.81  # deep water approx

        # Froude-Krylov approximation
        cross_section = 4.0  # m^2
        amplitude = wave_height / 2.0

        force_mag = rho_water * cross_section * amplitude * omega * omega / k
        fx = force_mag * math.cos(wave_dir)
        fy = force_mag * math.sin(wave_dir)

        # Heave (vertical) force
        fz = force_mag * 0.3 * math.sin(omega * self.time)

        # Roll torque from beam seas
        angle_diff = wave_dir - state.yaw
        torque_x = force_mag * 0.2 * math.sin(angle_diff)
        torque_z = force_mag * 0.1 * math.cos(angle_diff)

        return Force(fx=fx, fy=fy, fz=fz, torque_x=torque_x, torque_z=torque_z)

    def get_conditions_at(self, position: Tuple[float, float, float],
                          depth: float = 0.0,
                          time: float = None) -> EnvironmentConditions:
        """Get all environmental conditions at a point."""
        t = time if time is not None else self.time
        ws, wd = self.get_wind_at(position, t)
        cs, cd = self.get_current_at(position, depth, t)
        wh, wdr, wp = self.get_wave_at(position, t)
        return EnvironmentConditions(
            wind_speed=ws, wind_direction=wd,
            current_speed=cs, current_direction=cd,
            wave_height=wh, wave_direction=wdr, wave_period=wp,
        )

    def step(self, time_delta: float) -> 'EnvironmentModel':
        """Advance environment time by time_delta."""
        self.time += time_delta
        return self

    def set_time(self, time: float) -> 'EnvironmentModel':
        """Set environment time directly."""
        self.time = time
        return self

    def get_severity(self) -> float:
        """Get overall environmental severity on 0..1 scale (0=calm, 1=extreme)."""
        wind_sev = min(self.wind.speed / 25.0, 1.0)
        wave_sev = min(self.wave.height / 6.0, 1.0)
        current_sev = min(self.current.speed / 3.0, 1.0)
        return 0.4 * wind_sev + 0.4 * wave_sev + 0.2 * current_sev

    def total_environmental_force(self, state, position: Tuple[float, float, float],
                                   depth: float = 0.0) -> 'Force':
        """Compute total environmental force (wind + current + waves)."""
        from .physics import Force
        wind = self.get_wind_at(position)
        current = self.get_current_at(position, depth)
        waves = self.get_wave_at(position)

        wind_f = self.compute_wind_force(state, wind)
        wave_f = self.compute_wave_force(state, waves)

        # Current adds to velocity (modeled as drag differential)
        curr_vx = current[0] * math.cos(current[1])
        curr_vy = current[0] * math.sin(current[1])
        current_f = Force(fx=curr_vx * 50.0, fy=curr_vy * 50.0)

        return wind_f + wave_f + current_f
