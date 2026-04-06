"""Solar and wind energy harvesting with MPPT and prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Air density at sea level (kg/m³)
AIR_DENSITY = 1.225


@dataclass
class SolarPanel:
    """Photovoltaic panel specification."""
    area_m2: float
    efficiency: float         # 0-1
    tilt_angle: float = 0.0   # degrees from horizontal
    degradation: float = 0.005  # annual degradation fraction


@dataclass
class WindTurbine:
    """Micro wind turbine specification."""
    swept_area: float          # m²
    efficiency: float          # 0-1 (Betz limit ≈ 0.593 max)
    cut_in_speed: float = 3.0  # m/s minimum wind to generate
    rated_speed: float = 12.0  # m/s rated wind speed
    cut_out_speed: float = 25.0  # m/s safety shutdown


@dataclass
class WeatherCondition:
    """Hourly weather snapshot."""
    hour: int
    irradiance: float          # W/m²  (solar)
    wind_speed: float          # m/s
    temperature: float         # °C
    cloud_cover: float = 0.0   # 0-1


class HarvestEstimator:
    """Estimates energy harvest from solar panels and wind turbines."""

    # Standard Test Conditions irradiance
    STC_IRRADIANCE = 1000.0  # W/m²

    # Temperature coefficient for power (% per °C above 25 °C)
    TEMP_COEFF = 0.004

    @staticmethod
    def estimate_solar(
        panel: SolarPanel,
        irradiance: float,
        temperature: float,
    ) -> float:
        """Estimate instantaneous solar power output in watts."""
        base_power = panel.area_m2 * panel.efficiency * irradiance

        # Temperature derating: panels lose efficiency at higher temps
        delta_t = max(0.0, temperature - 25.0)
        temp_factor = 1.0 - HarvestEstimator.TEMP_COEFF * delta_t

        # Tilt factor: cos of tilt angle (simple model)
        tilt_rad = math.radians(panel.tilt_angle)
        tilt_factor = max(0.0, math.cos(tilt_rad))

        return max(0.0, base_power * temp_factor * tilt_factor)

    @staticmethod
    def estimate_wind_power(
        turbine: WindTurbine,
        wind_speed: float,
    ) -> float:
        """Estimate wind turbine power output in watts.

        Between cut-in and rated speed: P ∝ v³
        At rated speed and above (up to cut-out): rated power
        Above cut-out: 0 (shutdown)
        """
        if wind_speed < turbine.cut_in_speed:
            return 0.0
        if wind_speed > turbine.cut_out_speed:
            return 0.0

        # Available power in wind: P = 0.5 * ρ * A * v³
        available = 0.5 * AIR_DENSITY * turbine.swept_area * (wind_speed ** 3)

        if wind_speed >= turbine.rated_speed:
            # At or above rated speed → cap at rated power
            rated_power = 0.5 * AIR_DENSITY * turbine.swept_area * (turbine.rated_speed ** 3) * turbine.efficiency
            return rated_power

        return available * turbine.efficiency

    @staticmethod
    def compute_mppt_point(
        panel: SolarPanel,
        irradiance: float,
    ) -> Tuple[float, float]:
        """Compute Maximum Power Point: returns (optimal_voltage, optimal_current).

        Simplified model based on irradiance level.
        """
        # Open-circuit voltage ~ 0.6V per cell, scales with irradiance
        voc = 22.0 * (irradiance / HarvestEstimator.STC_IRRADIANCE)  # rough 22V at STC
        # MPP voltage is ~80% of Voc
        v_mpp = voc * 0.80

        # MPP current proportional to irradiance
        isc = panel.area_m2 * panel.efficiency * irradiance / v_mpp if v_mpp > 0 else 0.0

        return round(v_mpp, 3), round(isc, 3)

    @staticmethod
    def daily_harvest_estimate(
        panel: SolarPanel,
        location: str,
        season: str,
        weather: str,
    ) -> float:
        """Rough daily energy harvest estimate in Wh.

        Uses lookup tables for peak sun hours based on location/season,
        adjusted by weather.
        """
        # Peak sun hours by location and season (simplified)
        psh_table: Dict[str, Dict[str, float]] = {
            "tropical": {"summer": 6.0, "winter": 5.0, "spring": 5.5, "autumn": 5.0},
            "temperate": {"summer": 5.5, "winter": 3.0, "spring": 4.5, "autumn": 3.5},
            "arctic": {"summer": 8.0, "winter": 0.5, "spring": 4.0, "autumn": 2.0},
            "equatorial": {"summer": 5.5, "winter": 5.5, "spring": 5.5, "autumn": 5.5},
        }

        psh = psh_table.get(location, {}).get(season, 4.0)

        # Weather multiplier
        weather_mult = {
            "clear": 1.0,
            "partly_cloudy": 0.7,
            "cloudy": 0.4,
            "rainy": 0.2,
        }
        mult = weather_mult.get(weather, 0.6)

        peak_w = panel.area_m2 * panel.efficiency * HarvestEstimator.STC_IRRADIANCE
        daily_wh = peak_w * psh * mult
        return round(daily_wh, 2)

    @staticmethod
    def compute_harvest_profile(
        panel: SolarPanel,
        wind_turbine: Optional[WindTurbine],
        hourly_conditions: List[WeatherCondition],
    ) -> List[float]:
        """Compute hourly harvest in watts for each condition in the list."""
        profile: List[float] = []
        for cond in hourly_conditions:
            solar_w = HarvestEstimator.estimate_solar(
                panel, cond.irradiance, cond.temperature,
            )
            wind_w = 0.0
            if wind_turbine is not None:
                wind_w = HarvestEstimator.estimate_wind_power(wind_turbine, cond.wind_speed)
            profile.append(round(solar_w + wind_w, 2))
        return profile
