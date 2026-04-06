"""Energy-aware mission planning for marine robotics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .battery import BatteryState, BatteryModel, BatterySimulator
from .harvesting import HarvestEstimator, SolarPanel, WindTurbine, WeatherCondition


@dataclass
class MissionSegment:
    """One leg of a mission."""
    name: str
    distance: float             # km
    speed: float                # km/h
    duration: float             # hours
    power_consumption: float    # watts

    @property
    def energy_wh(self) -> float:
        """Energy consumed by this segment in Wh."""
        return self.power_consumption * self.duration


@dataclass
class MissionEnergyPlan:
    """Complete energy plan for a mission."""
    segments: List[MissionSegment]
    total_energy: float         # Wh
    total_distance: float       # km
    total_time: float           # hours
    battery_required: float     # Wh minimum battery needed

    @property
    def avg_power(self) -> float:
        """Average power over the mission."""
        if self.total_time <= 0:
            return 0.0
        return self.total_energy / self.total_time


@dataclass
class EnvironmentalConditions:
    """Mission environment parameters affecting power consumption."""
    current_speed: float = 0.0     # m/s water current
    wave_height: float = 0.0       # m
    wind_speed: float = 0.0        # m/s
    water_temperature: float = 15.0  # °C


# Base power consumption at 1 km/h (W) — used for speed-power scaling
_BASE_POWER_AT_1KMH = 50.0


class MissionEnergyPlanner:
    """Plans and evaluates mission energy requirements."""

    @staticmethod
    def plan_energy(
        segments: List[MissionSegment],
        battery: BatteryState,
        conditions: EnvironmentalConditions | None = None,
        harvest_wh: float = 0.0,
    ) -> MissionEnergyPlan:
        """Build an energy plan for the given mission segments."""
        conditions = conditions or EnvironmentalConditions()

        # Adjusted segments (power increases with adverse conditions)
        adjusted_segments: List[MissionSegment] = []
        for seg in segments:
            adj_power = MissionEnergyPlanner._adjust_power_for_conditions(
                seg.power_consumption, conditions,
            )
            adjusted = MissionSegment(
                name=seg.name,
                distance=seg.distance,
                speed=seg.speed,
                duration=seg.duration,
                power_consumption=adj_power,
            )
            adjusted_segments.append(adjusted)

        total_energy = sum(s.energy_wh for s in adjusted_segments) - harvest_wh
        total_energy = max(0.0, total_energy)
        total_distance = sum(s.distance for s in segments)
        total_time = sum(s.duration for s in segments)

        return MissionEnergyPlan(
            segments=adjusted_segments,
            total_energy=round(total_energy, 2),
            total_distance=round(total_distance, 4),
            total_time=round(total_time, 4),
            battery_required=round(total_energy * 1.1, 2),  # 10% buffer
        )

    @staticmethod
    def _adjust_power_for_conditions(
        base_power: float,
        conditions: EnvironmentalConditions,
    ) -> float:
        """Increase base power based on environmental factors."""
        factor = 1.0
        # Current resistance (quadratic)
        factor += 0.1 * (conditions.current_speed ** 2)
        # Wave resistance
        factor += 0.05 * conditions.wave_height
        # Wind drag
        factor += 0.02 * conditions.wind_speed
        return base_power * factor

    @staticmethod
    def compute_optimal_speed(
        distance: float,
        battery: BatteryState,
        conditions: EnvironmentalConditions | None = None,
    ) -> float:
        """Find the speed that maximises range.

        Simplified model: power ∝ speed³ (hull drag).
        Range = energy_available / (power_at_speed / speed)
              = energy * speed / power(s)
        With P(s) = k*s³, range = E / (k*s²)  → maximised at lowest speed.
        Practical minimum speed is 0.5 km/h.
        We return the speed that balances endurance vs. speed.
        """
        conditions = conditions or EnvironmentalConditions()
        energy_wh = battery.capacity_wh

        # Power at speed s: P = BASE * s^2.5 (sub-cubic for marine)
        # Range = energy / (P(s) / s) = energy / (BASE * s^1.5)
        # Maximise range → minimise s → practical lower bound
        optimal = 0.5  # km/h minimum practical speed

        # Verify it's feasible
        power_at_optimal = _BASE_POWER_AT_1KMH * (optimal ** 2.5)
        power_at_optimal = MissionEnergyPlanner._adjust_power_for_conditions(
            power_at_optimal, conditions,
        )
        duration = distance / optimal if optimal > 0 else float('inf')
        energy_needed = power_at_optimal * duration

        if energy_needed > energy_wh:
            # Even at min speed it's not feasible → return min speed anyway
            pass

        return round(optimal, 2)

    @staticmethod
    def compute_range_at_speed(
        battery: BatteryState,
        speed: float,
        conditions: EnvironmentalConditions | None = None,
    ) -> float:
        """Compute maximum range in km at a given speed."""
        conditions = conditions or EnvironmentalConditions()
        if speed <= 0:
            return 0.0

        power = _BASE_POWER_AT_1KMH * (speed ** 2.5)
        power = MissionEnergyPlanner._adjust_power_for_conditions(power, conditions)

        # Range = energy / (power / speed)
        range_km = (battery.capacity_wh * speed) / power if power > 0 else 0.0
        return round(max(0.0, range_km), 4)

    @staticmethod
    def compute_endurance(
        battery: BatteryState,
        average_power: float,
    ) -> float:
        """Compute endurance in hours at a given average power."""
        if average_power <= 0:
            return float('inf')
        return round(battery.capacity_wh / average_power, 4)

    @staticmethod
    def add_safety_margin(
        plan: MissionEnergyPlan,
        margin_percent: float,
    ) -> MissionEnergyPlan:
        """Return a conservative plan with added energy margin."""
        factor = 1.0 + margin_percent / 100.0
        return MissionEnergyPlan(
            segments=list(plan.segments),
            total_energy=round(plan.total_energy * factor, 2),
            total_distance=plan.total_distance,
            total_time=plan.total_time,
            battery_required=round(plan.battery_required * factor, 2),
        )

    @staticmethod
    def check_feasibility(
        plan: MissionEnergyPlan,
        battery: BatteryState,
    ) -> Tuple[bool, float]:
        """Check if the mission is feasible with current battery state.

        Returns (is_feasible, shortfall_wh).
        """
        shortfall = max(0.0, plan.battery_required - battery.capacity_wh)
        return shortfall <= 0, round(shortfall, 2)
