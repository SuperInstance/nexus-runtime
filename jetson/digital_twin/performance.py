"""Performance prediction: battery, speed, range for NEXUS digital twin."""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional


@dataclass
class BatteryModel:
    """Battery model for performance prediction."""
    capacity_wh: float = 1000.0     # watt-hours
    voltage: float = 48.0           # nominal voltage (V)
    current_draw: float = 20.0      # amps at full throttle
    efficiency: float = 0.90        # motor/propulsion efficiency
    temperature: float = 25.0       # ambient temperature (C)

    def remaining_capacity(self, state_of_charge: float) -> float:
        """Get remaining capacity in Wh from state of charge (0..1)."""
        return self.capacity_wh * max(0.0, min(1.0, state_of_charge))

    def state_of_charge(self, energy_used: float) -> float:
        """Get state of charge after using energy_used Wh."""
        return max(0.0, 1.0 - energy_used / self.capacity_wh)

    def temperature_effect(self) -> float:
        """Temperature derating factor. Optimal around 25C."""
        temp_diff = abs(self.temperature - 25.0)
        return max(0.5, 1.0 - 0.02 * temp_diff)

    def copy(self) -> 'BatteryModel':
        return BatteryModel(
            capacity_wh=self.capacity_wh,
            voltage=self.voltage,
            current_draw=self.current_draw,
            efficiency=self.efficiency,
            temperature=self.temperature,
        )


@dataclass
class PerformancePrediction:
    """Results from performance prediction."""
    estimated_range: float = 0.0         # km
    estimated_time: float = 0.0          # hours
    speed_profile: List[Tuple[float, float]] = field(default_factory=list)
    battery_usage: float = 0.0           # Wh used
    feasibility: bool = True
    notes: List[str] = field(default_factory=list)


@dataclass
class MissionProfile:
    """Mission parameters for feasibility estimation."""
    distance: float = 10.0          # km
    target_speed: float = 2.0       # m/s
    payload_weight: float = 0.0     # kg additional
    hover_time: float = 0.0         # minutes spent stationary
    safety_margin: float = 0.2      # reserve battery fraction


class PerformancePredictor:
    """Predicts vessel performance metrics."""

    BASE_POWER_IDLE = 50.0        # W at idle
    SPEED_POWER_COEFF = 80.0      # W per m/s
    QUAD_SPEED_COEFF = 15.0       # W per m/s^2 (quadratic drag)
    WIND_POWER_FACTOR = 20.0      # W per m/s of headwind
    WAVE_POWER_FACTOR = 10.0      # W per m of wave height
    CURRENT_POWER_FACTOR = 5.0    # W per m/s of current

    def predict_range(self, battery: BatteryModel, speed: float,
                      conditions: Dict[str, float] = None) -> float:
        """Predict maximum range in km at given speed under conditions.
        conditions may include: wind_speed, wind_direction, wave_height, current_speed."""
        conditions = conditions or {}
        power = self._power_at_speed(speed, conditions)
        if power <= 0:
            return float('inf') if battery.capacity_wh > 0 else 0.0

        usable_energy = battery.remaining_capacity(1.0) * battery.efficiency * battery.temperature_effect()
        time_hours = usable_energy / power
        range_km = speed * time_hours / 1000.0 * 3600.0
        return range_km

    def predict_battery_life(self, battery: BatteryModel,
                             power_consumption: float) -> float:
        """Predict battery life in hours at given power consumption (W)."""
        if power_consumption <= 0:
            return float('inf')
        usable = battery.remaining_capacity(1.0) * battery.efficiency * battery.temperature_effect()
        return usable / power_consumption

    def predict_speed_vs_power(self, speed_range: Tuple[float, float],
                               conditions: Dict[str, float] = None,
                               steps: int = 20) -> List[Tuple[float, float]]:
        """Compute power consumption across a speed range.
        Returns list of (speed, power) tuples."""
        conditions = conditions or {}
        results = []
        low, high = speed_range
        if high <= low:
            high = low + 1.0
        for i in range(steps):
            speed = low + (high - low) * i / max(steps - 1, 1)
            power = self._power_at_speed(speed, conditions)
            results.append((speed, power))
        return results

    def optimal_cruising_speed(self, battery: BatteryModel, distance: float,
                               conditions: Dict[str, float] = None) -> float:
        """Find optimal cruising speed to maximize range efficiency for a given distance.
        Returns speed in m/s."""
        conditions = conditions or {}
        best_speed = 0.5
        best_efficiency = 0.0

        for speed_val in [i * 0.25 for i in range(1, 81)]:  # 0.25 to 20 m/s
            power = self._power_at_speed(speed_val, conditions)
            if power <= 0:
                continue
            efficiency = speed_val / power  # m/s per W (distance efficiency)
            if efficiency > best_efficiency:
                # Check if this speed can complete the distance
                range_km = self.predict_range(battery, speed_val, conditions)
                if range_km >= distance:
                    best_efficiency = efficiency
                    best_speed = speed_val

        return best_speed

    def compute_power_budget(self, systems_power: Dict[str, float]) -> Dict[str, float]:
        """Compute power budget breakdown.
        systems_power: {system_name: power_W}"""
        total = sum(systems_power.values())
        budget = dict(systems_power)
        budget['total_power'] = total

        # Compute percentages
        if total > 0:
            for key in systems_power:
                budget[f'{key}_pct'] = (systems_power[key] / total) * 100.0

        # Efficiency-adjusted
        budget['propulsion_power'] = systems_power.get('propulsion', 0.0)
        budget['avionics_power'] = sum(
            v for k, v in systems_power.items() if k != 'propulsion'
        )

        return budget

    def estimate_mission_feasibility(self, mission: MissionProfile,
                                      battery: BatteryModel,
                                      conditions: Dict[str, float] = None) -> Tuple[bool, Dict[str, Any]]:
        """Estimate if a mission is feasible with given battery and conditions.
        Returns (feasible, metrics_dict)."""
        conditions = conditions or {}
        metrics = {}

        # Energy needed for transit at target speed
        transit_power = self._power_at_speed(mission.target_speed, conditions)
        transit_time = (mission.distance * 1000.0) / max(mission.target_speed, 0.01)  # seconds
        transit_energy = transit_power * transit_time / 3600.0  # Wh

        # Energy for hover/loiter
        hover_power = self._power_at_speed(0.0, conditions)
        hover_energy = hover_power * mission.hover_time / 60.0  # Wh

        total_energy = transit_energy + hover_energy
        available_energy = battery.remaining_capacity(1.0) * battery.efficiency * battery.temperature_effect()

        # Safety margin
        reserve = available_energy * mission.safety_margin
        usable = available_energy - reserve

        metrics['transit_power'] = transit_power
        metrics['transit_time_hours'] = transit_time / 3600.0
        metrics['transit_energy_wh'] = transit_energy
        metrics['hover_energy_wh'] = hover_energy
        metrics['total_energy_wh'] = total_energy
        metrics['available_energy_wh'] = available_energy
        metrics['reserve_energy_wh'] = reserve
        metrics['usable_energy_wh'] = usable
        metrics['final_soc'] = battery.state_of_charge(total_energy)

        # Range at target speed
        metrics['max_range_km'] = self.predict_range(battery, mission.target_speed, conditions)

        feasible = total_energy <= usable and mission.target_speed > 0
        metrics['feasible'] = feasible

        return feasible, metrics

    def predict_endurance(self, battery: BatteryModel, speed: float,
                          conditions: Dict[str, float] = None) -> float:
        """Predict endurance in hours at given speed."""
        conditions = conditions or {}
        power = self._power_at_speed(speed, conditions)
        if power <= 0:
            return float('inf')
        usable = battery.remaining_capacity(1.0) * battery.efficiency * battery.temperature_effect()
        return usable / power

    def predict_soc_over_time(self, battery: BatteryModel, speed: float,
                               duration_hours: float,
                               conditions: Dict[str, float] = None) -> List[Tuple[float, float]]:
        """Predict state of charge over time. Returns [(time_h, soc), ...]"""
        conditions = conditions or {}
        power = self._power_at_speed(speed, conditions)
        if power <= 0:
            return [(t, 1.0) for t in [i * duration_hours / 20 for i in range(21)]]

        usable = battery.remaining_capacity(1.0) * battery.efficiency * battery.temperature_effect()
        points = []
        n = 20
        for i in range(n + 1):
            t = duration_hours * i / n
            energy_used = power * t
            soc = battery.state_of_charge(energy_used)
            points.append((t, max(0.0, soc)))
        return points

    def compute_energy_for_distance(self, battery: BatteryModel, distance_km: float,
                                     speed: float,
                                     conditions: Dict[str, float] = None) -> float:
        """Compute energy needed to travel a given distance at given speed."""
        conditions = conditions or {}
        power = self._power_at_speed(speed, conditions)
        time_h = (distance_km * 1000.0) / max(speed, 0.01) / 3600.0
        return power * time_h  # Wh

    def _power_at_speed(self, speed: float, conditions: Dict[str, float]) -> float:
        """Compute power consumption at a given speed under conditions."""
        # Base propulsion power: linear + quadratic drag model
        propulsion = (self.BASE_POWER_IDLE +
                      self.SPEED_POWER_COEFF * abs(speed) +
                      self.QUAD_SPEED_COEFF * speed * speed)

        # Environmental factors
        wind = conditions.get('wind_speed', 0.0)
        wave = conditions.get('wave_height', 0.0)
        current = conditions.get('current_speed', 0.0)

        env_power = (self.WIND_POWER_FACTOR * wind +
                     self.WAVE_POWER_FACTOR * wave +
                     self.CURRENT_POWER_FACTOR * current)

        return propulsion + env_power
