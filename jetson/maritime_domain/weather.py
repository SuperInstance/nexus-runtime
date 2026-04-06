"""
Weather Impact Assessment Module.

Assesses weather impact on maritime operations, computes sea state
(Douglas scale), safe operating limits, weather trends, and route impacts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class RiskLevel(Enum):
    """Weather risk levels."""
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class OperationType(Enum):
    """Types of maritime operations."""
    NAVIGATION = "navigation"
    CARGO_TRANSFER = "cargo_transfer"
    PILOTAGE = "pilotage"
    ANCHORING = "anchoring"
    FISHING = "fishing"
    SEARCH_AND_RESCUE = "search_and_rescue"
    DOCKING = "docking"


class WeatherTrend(Enum):
    """Weather change trend directions."""
    IMPROVING = "improving"
    STABLE = "stable"
    WORSENING = "worsening"
    RAPIDLY_WORSENING = "rapidly_worsening"


@dataclass
class WeatherCondition:
    """Represents current weather conditions."""
    wind_speed: float = 0.0           # knots
    wind_direction: float = 0.0       # degrees true
    wave_height: float = 0.0          # meters
    wave_period: float = 0.0          # seconds
    visibility: float = 20.0          # nautical miles
    current_speed: float = 0.0        # knots
    current_direction: float = 0.0    # degrees true
    temperature: float = 20.0         # celsius
    pressure: float = 1013.25         # hPa
    precipitation: float = 0.0        # mm/hour


@dataclass
class WeatherImpact:
    """Result of weather impact assessment."""
    risk_level: RiskLevel
    affected_operations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    estimated_delay: float = 0.0      # hours


@dataclass
class VesselCapabilities:
    """Vessel-specific weather capabilities."""
    max_wind_speed: float = 50.0      # knots
    max_wave_height: float = 5.0      # meters
    min_visibility: float = 0.5       # nm
    max_current: float = 3.0          # knots
    ice_capable: bool = False
    vessel_size: str = "medium"       # small, medium, large


@dataclass
class SafeOperatingLimits:
    """Safe operating limits for current weather conditions."""
    max_speed: float              # knots
    max_wave_height: float        # meters
    min_visibility: float         # nm
    restricted_operations: List[str]
    caution_notes: List[str]


@dataclass
class WeatherForecast:
    """A weather forecast data point."""
    timestamp: float
    wind_speed: float
    wave_height: float
    visibility: float
    current_speed: float


@dataclass
class ImpactedSegment:
    """A route segment impacted by weather."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    impact_level: RiskLevel
    conditions: WeatherCondition
    recommendation: str


class WeatherAssessor:
    """Assesses weather impact on maritime operations."""

    # Douglas Sea Scale thresholds (wave height in meters)
    DOUGLAS_SCALE = [
        (0.0, 0.1, 0, "Calm (glassy)"),
        (0.1, 0.3, 1, "Calm (rippled)"),
        (0.3, 0.6, 2, "Smooth"),
        (0.6, 1.0, 3, "Slight"),
        (1.0, 2.0, 4, "Moderate"),
        (2.0, 3.0, 5, "Rough"),
        (3.0, 4.0, 6, "Very rough"),
        (4.0, 5.5, 7, "High"),
        (5.5, 7.5, 8, "Very high"),
        (7.5, 99.0, 9, "Phenomenal"),
    ]

    def assess_impact(
        self,
        weather: WeatherCondition,
        operation_type: str,
        vessel_capabilities: Optional[VesselCapabilities] = None,
    ) -> WeatherImpact:
        """
        Assess weather impact on a specific operation type.
        Returns WeatherImpact with risk level and recommendations.
        """
        if vessel_capabilities is None:
            vessel_capabilities = VesselCapabilities()

        risk_score = 0.0
        affected_ops = []
        recommendations = []
        estimated_delay = 0.0

        # Wind assessment
        if weather.wind_speed > vessel_capabilities.max_wind_speed:
            risk_score += 4.0
            affected_ops.append("navigation")
            recommendations.append("Reduce speed or seek shelter due to high winds.")
            estimated_delay += 2.0
        elif weather.wind_speed > vessel_capabilities.max_wind_speed * 0.7:
            risk_score += 2.5
            affected_ops.append("cargo_transfer")
            recommendations.append("Caution: high winds approaching vessel limits.")
            estimated_delay += 1.0

        # Wave height assessment
        if weather.wave_height > vessel_capabilities.max_wave_height:
            risk_score += 4.0
            affected_ops.append("navigation")
            affected_ops.append("anchoring")
            recommendations.append("Sea state exceeds vessel limits. Reduce speed significantly.")
            estimated_delay += 3.0
        elif weather.wave_height > vessel_capabilities.max_wave_height * 0.7:
            risk_score += 2.0
            affected_ops.append("docking")
            recommendations.append("Moderate seas: exercise caution for deck operations.")
            estimated_delay += 1.0

        # Visibility assessment
        if weather.visibility < vessel_capabilities.min_visibility:
            risk_score += 3.5
            affected_ops.append("navigation")
            affected_ops.append("pilotage")
            recommendations.append("Visibility below safe limits. Reduce to safe speed.")
            estimated_delay += 2.0
        elif weather.visibility < 2.0:
            risk_score += 2.0
            affected_ops.append("pilotage")
            recommendations.append("Restricted visibility. Use radar and sound signals.")
            estimated_delay += 0.5

        # Current assessment
        if weather.current_speed > vessel_capabilities.max_current:
            risk_score += 3.0
            affected_ops.append("docking")
            affected_ops.append("anchoring")
            recommendations.append("Strong current: anchoring and docking operations at risk.")
            estimated_delay += 1.5

        # Operation-specific considerations
        op = operation_type.lower() if isinstance(operation_type, str) else str(operation_type)
        if op == "cargo_transfer":
            if weather.wave_height > 1.5 or weather.wind_speed > 25:
                risk_score += 2.0
                recommendations.append("Cargo transfer operations should be suspended.")
        elif op == "docking":
            if weather.visibility < 1.0 or weather.wind_speed > 30:
                risk_score += 3.0
                recommendations.append("Docking operations not recommended.")
        elif op == "fishing":
            if weather.wave_height > 2.0:
                risk_score += 1.5
                recommendations.append("Fishing operations hazardous in current conditions.")

        # Determine risk level
        if risk_score >= 8.0:
            risk_level = RiskLevel.EXTREME
        elif risk_score >= 5.0:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 3.0:
            risk_level = RiskLevel.MODERATE
        elif risk_score >= 1.0:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.NEGLIGIBLE

        if not recommendations:
            recommendations.append("Weather conditions within normal operating parameters.")

        return WeatherImpact(
            risk_level=risk_level,
            affected_operations=affected_ops,
            recommendations=recommendations,
            estimated_delay=estimated_delay,
        )

    def compute_sea_state(self, wind_speed: float, wave_height: float) -> Tuple[int, str]:
        """
        Compute Douglas sea state from wind speed and wave height.
        Returns (sea_state_0_to_9, description).
        Uses wave height as primary criterion.
        """
        for min_h, max_h, state, desc in self.DOUGLAS_SCALE:
            if min_h <= wave_height < max_h:
                return (state, desc)
        return (9, "Phenomenal")

    def compute_safe_operating_limits(
        self,
        weather: WeatherCondition,
        vessel: Optional[VesselCapabilities] = None,
    ) -> SafeOperatingLimits:
        """
        Compute safe operating limits based on weather and vessel capabilities.
        Returns SafeOperatingLimits with speed limits and restrictions.
        """
        if vessel is None:
            vessel = VesselCapabilities()

        max_speed = 25.0  # default max speed
        restricted = []
        cautions = []

        # Speed reduction for wind
        if weather.wind_speed > vessel.max_wind_speed * 0.6:
            reduction_factor = 0.5
            max_speed *= reduction_factor
            cautions.append(f"Speed reduced due to wind {weather.wind_speed:.0f}kts")
        elif weather.wind_speed > vessel.max_wind_speed * 0.4:
            reduction_factor = 0.75
            max_speed *= reduction_factor
            cautions.append(f"Speed slightly reduced for wind {weather.wind_speed:.0f}kts")

        # Speed reduction for waves
        if weather.wave_height > vessel.max_wave_height * 0.7:
            wave_factor = max(0.2, 1.0 - (weather.wave_height / vessel.max_wave_height) * 0.6)
            max_speed = min(max_speed, max_speed * wave_factor)
            cautions.append(f"Speed reduced due to wave height {weather.wave_height:.1f}m")

        # Speed reduction for visibility
        if weather.visibility < 1.0:
            max_speed = min(max_speed, 5.0)
            restricted.append("navigation")
            cautions.append("Visibility severely restricted: dead slow")
        elif weather.visibility < 3.0:
            max_speed = min(max_speed, 10.0)
            cautions.append(f"Speed limited due to reduced visibility {weather.visibility:.1f}nm")

        # Operational restrictions
        if weather.wave_height > 1.5:
            restricted.append("cargo_transfer")
        if weather.visibility < 2.0:
            restricted.append("pilotage")
        if weather.wind_speed > 35:
            restricted.append("docking")
            restricted.append("anchoring")
        if weather.current_speed > 2.5:
            restricted.append("docking")

        max_speed = max(max_speed, 0.0)

        return SafeOperatingLimits(
            max_speed=round(max_speed, 1),
            max_wave_height=vessel.max_wave_height,
            min_visibility=vessel.min_visibility,
            restricted_operations=restricted,
            caution_notes=cautions,
        )

    def predict_weather_change(
        self,
        current: WeatherCondition,
        forecast: List[WeatherForecast],
    ) -> WeatherTrend:
        """
        Predict weather change trend based on current conditions and forecast.
        Returns WeatherTrend enum.
        """
        if not forecast:
            return WeatherTrend.STABLE

        # Compare current with forecast averages
        avg_wind = sum(f.wind_speed for f in forecast) / len(forecast)
        avg_wave = sum(f.wave_height for f in forecast) / len(forecast)
        avg_vis = sum(f.visibility for f in forecast) / len(forecast)

        wind_change = avg_wind - current.wind_speed
        wave_change = avg_wave - current.wave_height
        vis_change = avg_vis - current.visibility

        worsening_score = 0.0
        improving_score = 0.0

        if wind_change > 15:
            worsening_score += 3.0
        elif wind_change > 5:
            worsening_score += 1.5
        elif wind_change < -10:
            improving_score += 2.0
        elif wind_change < -3:
            improving_score += 1.0

        if wave_change > 1.5:
            worsening_score += 3.0
        elif wave_change > 0.5:
            worsening_score += 1.5
        elif wave_change < -1.0:
            improving_score += 2.0
        elif wave_change < -0.3:
            improving_score += 1.0

        if vis_change < -5:
            worsening_score += 2.0
        elif vis_change > 5:
            improving_score += 1.5

        if worsening_score > improving_score + 4.0:
            return WeatherTrend.RAPIDLY_WORSENING
        elif worsening_score > improving_score + 1.5:
            return WeatherTrend.WORSENING
        elif improving_score > worsening_score + 1.5:
            return WeatherTrend.IMPROVING
        else:
            return WeatherTrend.STABLE

    def compute_route_weather_impact(
        self,
        route: List[Tuple[float, float]],
        weather_forecast: Dict[Tuple[float, float], WeatherCondition],
    ) -> List[ImpactedSegment]:
        """
        Compute weather impact on each segment of a route.
        route: list of (lat, lon) waypoints
        weather_forecast: {(lat, lon): WeatherCondition}
        Returns list of ImpactedSegment.
        """
        if len(route) < 2:
            return []

        segments = []
        for i in range(len(route) - 1):
            start = route[i]
            end = route[i + 1]

            # Find closest weather data
            mid_lat = (start[0] + end[0]) / 2
            mid_lon = (start[1] + end[1]) / 2

            closest_weather = self._find_closest_weather(
                (mid_lat, mid_lon), weather_forecast
            )

            if closest_weather:
                impact = self.assess_impact(
                    closest_weather, "navigation", VesselCapabilities()
                )
                rec = "; ".join(impact.recommendations) if impact.recommendations else "No issues."
                segments.append(ImpactedSegment(
                    start=start,
                    end=end,
                    impact_level=impact.risk_level,
                    conditions=closest_weather,
                    recommendation=rec,
                ))
            else:
                segments.append(ImpactedSegment(
                    start=start,
                    end=end,
                    impact_level=RiskLevel.NEGLIGIBLE,
                    conditions=WeatherCondition(),
                    recommendation="No weather data available for this segment.",
                ))

        return segments

    @staticmethod
    def _find_closest_weather(
        position: Tuple[float, float],
        weather_map: Dict[Tuple[float, float], WeatherCondition],
    ) -> Optional[WeatherCondition]:
        """Find the closest weather data point to a position."""
        if not weather_map:
            return None

        min_dist = float('inf')
        closest = None
        for pos, weather in weather_map.items():
            dist = (position[0] - pos[0]) ** 2 + (position[1] - pos[1]) ** 2
            if dist < min_dist:
                min_dist = dist
                closest = weather

        return closest
