"""Tests for mission_energy module."""

import math
import pytest

from jetson.energy.battery import BatteryState
from jetson.energy.mission_energy import (
    MissionSegment,
    MissionEnergyPlan,
    EnvironmentalConditions,
    MissionEnergyPlanner,
)


# ---------------------------------------------------------------------------
# MissionSegment tests
# ---------------------------------------------------------------------------

class TestMissionSegment:
    def test_energy_wh(self):
        seg = MissionSegment(name="leg1", distance=10.0, speed=5.0,
                             duration=2.0, power_consumption=100.0)
        assert seg.energy_wh == 200.0

    def test_zero_duration(self):
        seg = MissionSegment(name="idle", distance=0, speed=0, duration=0,
                             power_consumption=50.0)
        assert seg.energy_wh == 0.0


# ---------------------------------------------------------------------------
# MissionEnergyPlan tests
# ---------------------------------------------------------------------------

class TestMissionEnergyPlan:
    def _plan(self):
        segs = [
            MissionSegment(name="a", distance=5, speed=5, duration=1, power_consumption=100),
            MissionSegment(name="b", distance=10, speed=5, duration=2, power_consumption=200),
        ]
        return MissionEnergyPlan(segments=segs, total_energy=500.0,
                                 total_distance=15.0, total_time=3.0,
                                 battery_required=550.0)

    def test_avg_power(self):
        plan = self._plan()
        assert plan.avg_power == pytest.approx(500.0 / 3.0)

    def test_avg_power_zero_time(self):
        segs = [MissionSegment(name="x", distance=0, speed=0, duration=0, power_consumption=0)]
        plan = MissionEnergyPlan(segments=segs, total_energy=0, total_distance=0,
                                 total_time=0, battery_required=0)
        assert plan.avg_power == 0.0


# ---------------------------------------------------------------------------
# MissionEnergyPlanner tests
# ---------------------------------------------------------------------------

class TestMissionEnergyPlanner:
    def _battery(self, capacity=1000.0, soc=100.0):
        return BatteryState(
            charge_percent=soc, voltage=4.0, current=0.0,
            temperature=25.0, capacity_wh=capacity, cycles=0,
        )

    def _segments(self):
        return [
            MissionSegment(name="transit", distance=10.0, speed=5.0,
                           duration=2.0, power_consumption=100.0),
            MissionSegment(name="survey", distance=5.0, speed=2.0,
                           duration=2.5, power_consumption=150.0),
        ]

    def test_plan_energy_basic(self):
        segs = self._segments()
        battery = self._battery()
        plan = MissionEnergyPlanner.plan_energy(segs, battery)
        assert plan.total_distance == 15.0
        assert plan.total_time == 4.5
        assert plan.total_energy > 0

    def test_plan_energy_with_conditions(self):
        segs = self._segments()
        battery = self._battery()
        cond = EnvironmentalConditions(current_speed=1.0, wave_height=0.5, wind_speed=5.0)
        plan_adverse = MissionEnergyPlanner.plan_energy(segs, battery, cond)
        plan_calm = MissionEnergyPlanner.plan_energy(segs, battery)
        # Adverse conditions should require more energy
        assert plan_adverse.total_energy >= plan_calm.total_energy

    def test_plan_energy_with_harvest(self):
        segs = self._segments()
        battery = self._battery()
        plan_no_harvest = MissionEnergyPlanner.plan_energy(segs, battery, harvest_wh=0)
        plan_with_harvest = MissionEnergyPlanner.plan_energy(segs, battery, harvest_wh=200)
        assert plan_with_harvest.total_energy < plan_no_harvest.total_energy

    def test_plan_energy_battery_required_has_buffer(self):
        segs = self._segments()
        battery = self._battery()
        plan = MissionEnergyPlanner.plan_energy(segs, battery)
        assert plan.battery_required > plan.total_energy  # 10% buffer

    def test_plan_energy_empty_segments(self):
        battery = self._battery()
        plan = MissionEnergyPlanner.plan_energy([], battery)
        assert plan.total_energy == 0
        assert plan.total_distance == 0
        assert plan.total_time == 0

    def test_optimal_speed_returns_minimum(self):
        battery = self._battery()
        speed = MissionEnergyPlanner.compute_optimal_speed(10.0, battery)
        assert speed == 0.5  # minimum practical speed

    def test_optimal_speed_with_conditions(self):
        battery = self._battery()
        cond = EnvironmentalConditions(current_speed=2.0)
        speed = MissionEnergyPlanner.compute_optimal_speed(10.0, battery, cond)
        assert speed == 0.5

    def test_range_at_speed(self):
        battery = self._battery(capacity=500.0)
        r = MissionEnergyPlanner.compute_range_at_speed(battery, 2.0)
        assert r > 0

    def test_range_decreases_with_speed(self):
        battery = self._battery(capacity=500.0)
        r_low = MissionEnergyPlanner.compute_range_at_speed(battery, 1.0)
        r_high = MissionEnergyPlanner.compute_range_at_speed(battery, 5.0)
        assert r_low > r_high

    def test_range_zero_speed(self):
        battery = self._battery()
        assert MissionEnergyPlanner.compute_range_at_speed(battery, 0.0) == 0.0

    def test_range_with_adverse_conditions(self):
        battery = self._battery(capacity=500.0)
        r_calm = MissionEnergyPlanner.compute_range_at_speed(battery, 2.0)
        cond = EnvironmentalConditions(current_speed=1.0)
        r_adverse = MissionEnergyPlanner.compute_range_at_speed(battery, 2.0, cond)
        assert r_adverse < r_calm

    def test_endurance(self):
        battery = self._battery(capacity=500.0)
        hours = MissionEnergyPlanner.compute_endurance(battery, 100.0)
        assert hours == 5.0

    def test_endurance_zero_power(self):
        battery = self._battery()
        assert MissionEnergyPlanner.compute_endurance(battery, 0.0) == float('inf')

    def test_safety_margin(self):
        segs = self._segments()
        battery = self._battery()
        plan = MissionEnergyPlanner.plan_energy(segs, battery)
        safe = MissionEnergyPlanner.add_safety_margin(plan, 20.0)
        assert safe.total_energy == pytest.approx(plan.total_energy * 1.2, abs=0.1)
        assert safe.battery_required == pytest.approx(plan.battery_required * 1.2, abs=0.1)

    def test_safety_margin_preserves_distance_time(self):
        segs = self._segments()
        battery = self._battery()
        plan = MissionEnergyPlanner.plan_energy(segs, battery)
        safe = MissionEnergyPlanner.add_safety_margin(plan, 10.0)
        assert safe.total_distance == plan.total_distance
        assert safe.total_time == plan.total_time

    def test_feasibility_ok(self):
        segs = self._segments()
        battery = self._battery(capacity=2000.0)
        plan = MissionEnergyPlanner.plan_energy(segs, battery)
        feasible, shortfall = MissionEnergyPlanner.check_feasibility(plan, battery)
        assert feasible is True
        assert shortfall == 0.0

    def test_feasibility_not_ok(self):
        segs = [
            MissionSegment(name="long", distance=100, speed=5, duration=20,
                           power_consumption=200),
        ]
        battery = self._battery(capacity=100.0)
        plan = MissionEnergyPlanner.plan_energy(segs, battery)
        feasible, shortfall = MissionEnergyPlanner.check_feasibility(plan, battery)
        assert feasible is False
        assert shortfall > 0

    def test_plan_energy_conditions_increase_power(self):
        """Adverse conditions should increase adjusted segment power."""
        segs = [
            MissionSegment(name="transit", distance=10, speed=5, duration=2, power_consumption=100),
        ]
        battery = self._battery()
        calm = EnvironmentalConditions()
        rough = EnvironmentalConditions(current_speed=2.0, wave_height=1.0, wind_speed=10.0)
        plan_calm = MissionEnergyPlanner.plan_energy(segs, battery, calm)
        plan_rough = MissionEnergyPlanner.plan_energy(segs, battery, rough)
        assert plan_rough.segments[0].power_consumption > plan_calm.segments[0].power_consumption

    def test_range_zero_capacity(self):
        battery = self._battery(capacity=0.0)
        assert MissionEnergyPlanner.compute_range_at_speed(battery, 5.0) == 0.0

    def test_safety_margin_zero(self):
        segs = self._segments()
        battery = self._battery()
        plan = MissionEnergyPlanner.plan_energy(segs, battery)
        safe = MissionEnergyPlanner.add_safety_margin(plan, 0.0)
        assert safe.total_energy == plan.total_energy
