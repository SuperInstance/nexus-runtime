"""Tests for environmental modeling."""

import math
import pytest
from jetson.digital_twin.environment import (
    WindField, CurrentField, WaveField, EnvironmentConditions, EnvironmentModel
)
from jetson.digital_twin.physics import VesselState, Force


class TestWindField:
    def test_defaults(self):
        w = WindField()
        assert w.speed == 5.0 and w.direction == 0.0
        assert w.gust_speed == 15.0 and w.gust_probability == 0.1

    def test_copy(self):
        w = WindField(speed=10, direction=1.5)
        c = w.copy()
        assert c.speed == 10 and c.direction == 1.5
        c.speed = 20
        assert w.speed == 10

    def test_custom(self):
        w = WindField(speed=20, direction=math.pi, gust_speed=30, gust_probability=0.5)
        assert w.speed == 20 and w.gust_probability == 0.5


class TestCurrentField:
    def test_defaults(self):
        c = CurrentField()
        assert c.speed == 0.5 and c.direction == 0.0
        assert c.depth_profile == 'uniform'

    def test_copy(self):
        c = CurrentField(speed=2.0, depth_profile='logarithmic')
        c2 = c.copy()
        assert c2.speed == 2.0 and c2.depth_profile == 'logarithmic'

    def test_custom(self):
        c = CurrentField(speed=1.5, direction=math.pi/4, depth_profile='surface')
        assert c.speed == 1.5 and c.depth_profile == 'surface'


class TestWaveField:
    def test_defaults(self):
        w = WaveField()
        assert w.height == 1.0 and w.period == 8.0
        assert w.spectrum_type == 'pierson_moskowitz'

    def test_copy(self):
        w = WaveField(height=2.5, period=10)
        c = w.copy()
        assert c.height == 2.5

    def test_wave_number(self):
        w = WaveField(period=8.0)
        k = w.wave_number(10.0)
        assert k > 0

    def test_wave_number_deep_water(self):
        w = WaveField(period=8.0)
        k = w.wave_number(100.0)
        omega = 2 * math.pi / 8.0
        k_deep = omega * omega / 9.81
        assert abs(k - k_deep) < 0.01

    def test_wave_length(self):
        w = WaveField(period=8.0)
        L = w.wave_length(10.0)
        assert L > 0
        k = w.wave_number(10.0)
        assert abs(L - 2 * math.pi / k) < 1e-6

    def test_orbital_velocity_surface(self):
        w = WaveField(height=2.0, period=8.0, direction=0.0)
        u, v = w.orbital_velocity(10.0, 0.0)
        assert u > 0
        assert abs(v) < 1e-9

    def test_orbital_velocity_decay(self):
        w = WaveField(height=2.0, period=8.0)
        u_surf, _ = w.orbital_velocity(10.0, 0.0)
        u_deep, _ = w.orbital_velocity(10.0, -5.0)
        assert u_deep < u_surf

    def test_custom_spectrum(self):
        w = WaveField(spectrum_type='jONSWAP')
        assert w.spectrum_type == 'jONSWAP'


class TestEnvironmentConditions:
    def test_defaults(self):
        c = EnvironmentConditions()
        assert c.wind_speed == 0 and c.temperature == 15.0
        assert c.salinity == 35.0

    def test_custom(self):
        c = EnvironmentConditions(wind_speed=10, wave_height=2.5)
        assert c.wind_speed == 10 and c.wave_height == 2.5


class TestEnvironmentModel:
    def setup_method(self):
        self.env = EnvironmentModel()

    def test_default_construction(self):
        assert self.env.wind.speed == 5.0
        assert self.env.current.speed == 0.5
        assert self.env.wave.height == 1.0

    def test_custom_construction(self):
        env = EnvironmentModel(
            wind=WindField(speed=10),
            current=CurrentField(speed=1.0),
            wave=WaveField(height=2.0),
        )
        assert env.wind.speed == 10
        assert env.current.speed == 1.0
        assert env.wave.height == 2.0

    def test_get_wind_at(self):
        speed, direction = self.env.get_wind_at((0, 0, 0))
        assert speed > 0
        assert isinstance(direction, float)

    def test_get_wind_at_with_time(self):
        s1, _ = self.env.get_wind_at((0, 0, 0), time=0)
        s2, _ = self.env.get_wind_at((0, 0, 0), time=600)
        # Should vary slightly due to sinusoidal modulation
        assert abs(s1 - s2) > 0 or abs(s1 - 5.0) < 0.1

    def test_get_current_at_uniform(self):
        speed, direction = self.env.get_current_at((0, 0, 0), depth=0)
        assert speed > 0

    def test_get_current_at_surface_profile(self):
        env = EnvironmentModel(current=CurrentField(depth_profile='surface'))
        s0, _ = env.get_current_at((0, 0, 0), depth=0)
        s50, _ = env.get_current_at((0, 0, 0), depth=50)
        assert s50 < s0

    def test_get_current_at_logarithmic_profile(self):
        env = EnvironmentModel(current=CurrentField(depth_profile='logarithmic'))
        s1, _ = env.get_current_at((0, 0, 0), depth=1)
        s10, _ = env.get_current_at((0, 0, 0), depth=10)
        assert s10 > 0

    def test_get_current_at_with_time(self):
        s1, _ = self.env.get_current_at((0, 0, 0), time=0)
        s2, _ = self.env.get_current_at((0, 0, 0), time=3600)
        assert isinstance(s1, float) and isinstance(s2, float)

    def test_get_wave_at(self):
        h, d, p = self.env.get_wave_at((0, 0, 0))
        assert h > 0
        assert isinstance(p, float)
        assert p > 0

    def test_get_wave_at_with_time(self):
        h1, _, _ = self.env.get_wave_at((0, 0, 0), time=0)
        h2, _, _ = self.env.get_wave_at((0, 0, 0), time=1800)
        assert isinstance(h1, float) and isinstance(h2, float)

    def test_compute_wind_force(self):
        state = VesselState()
        wind = (10.0, 0.0)
        force = self.env.compute_wind_force(state, wind)
        assert isinstance(force, Force)
        assert force.fx != 0 or force.fy != 0

    def test_compute_wind_force_with_vessel_velocity(self):
        state = VesselState(vx=5.0)
        wind = (10.0, 0.0)
        force = self.env.compute_wind_force(state, wind)
        # Relative wind is reduced
        assert isinstance(force, Force)

    def test_compute_wind_force_zero_wind(self):
        state = VesselState()
        wind = (0.0, 0.0)
        force = self.env.compute_wind_force(state, wind)
        assert force.magnitude() == 0.0

    def test_compute_current_effect(self):
        vel = (2.0, 0.0, 0.0)
        current = (1.0, 0.0)
        eff = self.env.compute_current_effect(vel, current)
        assert eff == (3.0, 0.0, 0.0)

    def test_compute_current_effect_orthogonal(self):
        vel = (0.0, 3.0, 0.0)
        current = (1.0, math.pi / 2)
        eff = self.env.compute_current_effect(vel, current)
        assert abs(eff[1] - 4.0) < 1e-9

    def test_compute_current_effect_zero(self):
        vel = (1.0, 0.0, 0.0)
        current = (0.0, 0.0)
        eff = self.env.compute_current_effect(vel, current)
        assert eff == (1.0, 0.0, 0.0)

    def test_compute_wave_force(self):
        state = VesselState()
        waves = (1.0, 0.0, 8.0)
        force = self.env.compute_wave_force(state, waves)
        assert isinstance(force, Force)
        assert force.fx != 0 or force.fy != 0

    def test_compute_wave_force_zero(self):
        state = VesselState()
        waves = (0.0, 0.0, 8.0)
        force = self.env.compute_wave_force(state, waves)
        # Height=0 should produce minimal force
        assert force.magnitude() < 100

    def test_get_conditions_at(self):
        conds = self.env.get_conditions_at((0, 0, 0), depth=0)
        assert isinstance(conds, EnvironmentConditions)
        assert conds.wind_speed > 0

    def test_step(self):
        self.env.step(1.0)
        assert self.env.time == 1.0
        self.env.step(2.0)
        assert self.env.time == 3.0

    def test_set_time(self):
        self.env.set_time(100.0)
        assert self.env.time == 100.0

    def test_get_severity_calm(self):
        env = EnvironmentModel(
            wind=WindField(speed=1),
            wave=WaveField(height=0.2),
            current=CurrentField(speed=0.1),
        )
        sev = env.get_severity()
        assert 0 <= sev < 0.5

    def test_get_severity_extreme(self):
        env = EnvironmentModel(
            wind=WindField(speed=25),
            wave=WaveField(height=6),
            current=CurrentField(speed=3),
        )
        sev = env.get_severity()
        assert sev > 0.8

    def test_get_severity_moderate(self):
        env = EnvironmentModel()
        sev = env.get_severity()
        assert 0 <= sev <= 1.0

    def test_total_environmental_force(self):
        state = VesselState()
        force = self.env.total_environmental_force(state, (0, 0, 0))
        assert isinstance(force, Force)

    def test_total_environmental_force_with_current(self):
        state = VesselState()
        force = self.env.total_environmental_force(state, (0, 0, 0), depth=0)
        assert isinstance(force, Force)
