"""Tests for performance prediction."""

import math
import pytest
from jetson.digital_twin.performance import (
    BatteryModel, PerformancePrediction, MissionProfile, PerformancePredictor
)


class TestBatteryModel:
    def test_defaults(self):
        b = BatteryModel()
        assert b.capacity_wh == 1000.0
        assert b.voltage == 48.0
        assert b.current_draw == 20.0
        assert b.efficiency == 0.90
        assert b.temperature == 25.0

    def test_remaining_capacity_full(self):
        b = BatteryModel(capacity_wh=1000)
        assert b.remaining_capacity(1.0) == 1000.0

    def test_remaining_capacity_half(self):
        b = BatteryModel(capacity_wh=1000)
        assert b.remaining_capacity(0.5) == 500.0

    def test_remaining_capacity_empty(self):
        b = BatteryModel(capacity_wh=1000)
        assert b.remaining_capacity(0.0) == 0.0

    def test_remaining_capacity_clamped(self):
        b = BatteryModel(capacity_wh=1000)
        assert b.remaining_capacity(1.5) == 1000.0  # clamped to 1.0
        assert b.remaining_capacity(-0.5) == 0.0

    def test_state_of_charge_full(self):
        b = BatteryModel(capacity_wh=1000)
        assert b.state_of_charge(0.0) == 1.0

    def test_state_of_charge_half(self):
        b = BatteryModel(capacity_wh=1000)
        assert b.state_of_charge(500.0) == 0.5

    def test_state_of_charge_empty(self):
        b = BatteryModel(capacity_wh=1000)
        assert b.state_of_charge(1000.0) == 0.0

    def test_state_of_charge_overdischarge(self):
        b = BatteryModel(capacity_wh=1000)
        assert b.state_of_charge(2000.0) == 0.0  # clamped

    def test_temperature_effect_optimal(self):
        b = BatteryModel(temperature=25.0)
        assert b.temperature_effect() == 1.0

    def test_temperature_effect_cold(self):
        b = BatteryModel(temperature=0.0)
        effect = b.temperature_effect()
        assert 0.5 <= effect < 1.0

    def test_temperature_effect_hot(self):
        b = BatteryModel(temperature=50.0)
        effect = b.temperature_effect()
        assert 0.5 <= effect < 1.0

    def test_temperature_effect_extreme(self):
        b = BatteryModel(temperature=-20.0)
        assert b.temperature_effect() == 0.5  # minimum

    def test_copy(self):
        b = BatteryModel(capacity_wh=2000)
        c = b.copy()
        assert c.capacity_wh == 2000
        c.capacity_wh = 999
        assert b.capacity_wh == 2000


class TestMissionProfile:
    def test_defaults(self):
        m = MissionProfile()
        assert m.distance == 10.0
        assert m.target_speed == 2.0
        assert m.safety_margin == 0.2

    def test_custom(self):
        m = MissionProfile(distance=50, target_speed=5, safety_margin=0.3)
        assert m.distance == 50 and m.target_speed == 5


class TestPerformancePredictor:
    def setup_method(self):
        self.pred = PerformancePredictor()
        self.battery = BatteryModel(capacity_wh=1000, efficiency=0.9)

    def test_predict_range_basic(self):
        range_km = self.pred.predict_range(self.battery, 2.0)
        assert range_km > 0

    def test_predict_range_higher_speed_less_range(self):
        range_low = self.pred.predict_range(self.battery, 1.0)
        range_high = self.pred.predict_range(self.battery, 5.0)
        assert range_low > range_high

    def test_predict_range_zero_speed(self):
        range_km = self.pred.predict_range(self.battery, 0.0)
        assert range_km == 0.0  # zero speed = zero range regardless of battery

    def test_predict_range_with_conditions(self):
        calm = {'wind_speed': 0, 'wave_height': 0, 'current_speed': 0}
        storm = {'wind_speed': 20, 'wave_height': 4, 'current_speed': 2}
        r_calm = self.pred.predict_range(self.battery, 2.0, calm)
        r_storm = self.pred.predict_range(self.battery, 2.0, storm)
        assert r_calm > r_storm

    def test_predict_battery_life(self):
        hours = self.pred.predict_battery_life(self.battery, 100.0)
        assert hours > 0

    def test_predict_battery_life_zero_power(self):
        hours = self.pred.predict_battery_life(self.battery, 0.0)
        assert hours == float('inf')

    def test_predict_battery_life_math(self):
        hours = self.pred.predict_battery_life(self.battery, 100.0)
        # usable = 1000 * 0.9 * 1.0 = 900 Wh
        expected = 900.0 / 100.0
        assert abs(hours - expected) < 1e-6

    def test_predict_speed_vs_power(self):
        profile = self.pred.predict_speed_vs_power((0.0, 5.0))
        assert len(profile) == 20
        # Power should generally increase with speed
        assert profile[-1][1] > profile[0][1]

    def test_predict_speed_vs_power_custom_steps(self):
        profile = self.pred.predict_speed_vs_power((0, 10), steps=5)
        assert len(profile) == 5

    def test_predict_speed_vs_power_with_conditions(self):
        profile = self.pred.predict_speed_vs_power(
            (0, 5), conditions={'wind_speed': 10}
        )
        assert len(profile) == 20
        # Wind adds power overhead
        idle_power = profile[0][1]
        assert idle_power > PerformancePredictor.BASE_POWER_IDLE

    def test_optimal_cruising_speed(self):
        speed = self.pred.optimal_cruising_speed(self.battery, 5.0)
        assert speed > 0

    def test_optimal_cruising_speed_reasonable(self):
        speed = self.pred.optimal_cruising_speed(self.battery, 1.0)
        # Should be in reasonable range for small USV
        assert 0.25 <= speed <= 20.0

    def test_compute_power_budget(self):
        systems = {'propulsion': 200, 'navigation': 50, 'comms': 30}
        budget = self.pred.compute_power_budget(systems)
        assert budget['total_power'] == 280
        assert 'propulsion_pct' in budget
        assert budget['avionics_power'] == 80  # nav + comms

    def test_compute_power_budget_empty(self):
        budget = self.pred.compute_power_budget({})
        assert budget['total_power'] == 0

    def test_compute_power_budget_single(self):
        budget = self.pred.compute_power_budget({'propulsion': 100})
        assert budget['total_power'] == 100
        assert budget['propulsion_pct'] == 100.0

    def test_estimate_mission_feasibility_feasible(self):
        mission = MissionProfile(distance=1.0, target_speed=2.0)
        feasible, metrics = self.pred.estimate_mission_feasibility(mission, self.battery)
        assert feasible is True
        assert 'transit_power' in metrics
        assert 'total_energy_wh' in metrics

    def test_estimate_mission_feasibility_long_distance(self):
        mission = MissionProfile(distance=1000.0, target_speed=2.0)
        feasible, metrics = self.pred.estimate_mission_feasibility(mission, self.battery)
        # Very long distance likely infeasible
        assert 'feasible' in metrics
        assert 'max_range_km' in metrics

    def test_estimate_mission_feasibility_zero_speed(self):
        mission = MissionProfile(distance=10, target_speed=0.0)
        feasible, metrics = self.pred.estimate_mission_feasibility(mission, self.battery)
        assert feasible is False

    def test_estimate_mission_feasibility_with_hover(self):
        mission = MissionProfile(distance=1.0, target_speed=2.0, hover_time=30)
        feasible, metrics = self.pred.estimate_mission_feasibility(mission, self.battery)
        assert 'hover_energy_wh' in metrics
        assert metrics['hover_energy_wh'] > 0

    def test_predict_endurance(self):
        hours = self.pred.predict_endurance(self.battery, 2.0)
        assert hours > 0

    def test_predict_endurance_math(self):
        hours = self.pred.predict_endurance(self.battery, 2.0)
        power = self.pred._power_at_speed(2.0, {})
        usable = 1000.0 * 0.9 * 1.0
        expected = usable / power
        assert abs(hours - expected) < 1e-6

    def test_predict_soc_over_time(self):
        profile = self.pred.predict_soc_over_time(self.battery, 2.0, 2.0)
        assert len(profile) == 21
        assert profile[0][1] == 1.0  # full at start
        assert profile[-1][1] < 1.0  # depleted at end

    def test_predict_soc_over_time_zero_power(self):
        profile = self.pred.predict_soc_over_time(self.battery, 0.0, 1.0)
        # Still drains at idle
        assert len(profile) == 21

    def test_compute_energy_for_distance(self):
        energy = self.pred.compute_energy_for_distance(self.battery, 10.0, 2.0)
        assert energy > 0

    def test_compute_energy_for_distance_math(self):
        power = self.pred._power_at_speed(2.0, {})
        time_h = (10.0 * 1000.0) / 2.0 / 3600.0
        expected = power * time_h
        energy = self.pred.compute_energy_for_distance(self.battery, 10.0, 2.0)
        assert abs(energy - expected) < 1e-6

    def test_power_at_speed_increases(self):
        p1 = self.pred._power_at_speed(0, {})
        p2 = self.pred._power_at_speed(5, {})
        assert p2 > p1

    def test_power_at_speed_with_conditions(self):
        p_calm = self.pred._power_at_speed(2, {'wind_speed': 0, 'wave_height': 0})
        p_storm = self.pred._power_at_speed(2, {'wind_speed': 15, 'wave_height': 3})
        assert p_storm > p_calm

    def test_power_budget_percentages(self):
        systems = {'a': 30, 'b': 70}
        budget = self.pred.compute_power_budget(systems)
        assert abs(budget['a_pct'] - 30.0) < 1e-6
        assert abs(budget['b_pct'] - 70.0) < 1e-6

    def test_mission_with_safety_margin(self):
        mission = MissionProfile(distance=1, target_speed=2, safety_margin=0.5)
        feasible, metrics = self.pred.estimate_mission_feasibility(mission, self.battery)
        reserve = metrics['reserve_energy_wh']
        total = metrics['available_energy_wh']
        assert abs(reserve - total * 0.5) < 1e-6
