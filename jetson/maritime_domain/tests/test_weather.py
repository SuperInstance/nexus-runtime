"""Tests for Weather Impact Assessment module."""

import math

import pytest

from jetson.maritime_domain.weather import (
    ImpactedSegment,
    OperationType,
    RiskLevel,
    SafeOperatingLimits,
    VesselCapabilities,
    WeatherAssessor,
    WeatherCondition,
    WeatherForecast,
    WeatherImpact,
    WeatherTrend,
)


class TestWeatherCondition:
    """Tests for WeatherCondition dataclass."""

    def test_defaults(self):
        w = WeatherCondition()
        assert w.wind_speed == 0.0
        assert w.wave_height == 0.0
        assert w.visibility == 20.0
        assert w.current_speed == 0.0
        assert w.temperature == 20.0
        assert w.pressure == 1013.25

    def test_full_construction(self):
        w = WeatherCondition(
            wind_speed=25.0,
            wind_direction=180.0,
            wave_height=2.5,
            wave_period=8.0,
            visibility=5.0,
            current_speed=1.5,
            current_direction=270.0,
            temperature=15.0,
            pressure=1005.0,
            precipitation=5.0,
        )
        assert w.wind_speed == 25.0
        assert w.pressure == 1005.0
        assert w.precipitation == 5.0


class TestWeatherImpact:
    """Tests for WeatherImpact dataclass."""

    def test_defaults(self):
        impact = WeatherImpact(risk_level=RiskLevel.LOW)
        assert impact.affected_operations == []
        assert impact.recommendations == []
        assert impact.estimated_delay == 0.0

    def test_with_data(self):
        impact = WeatherImpact(
            risk_level=RiskLevel.HIGH,
            affected_operations=["navigation", "docking"],
            recommendations=["Reduce speed"],
            estimated_delay=2.5,
        )
        assert len(impact.affected_operations) == 2
        assert impact.estimated_delay == 2.5


class TestVesselCapabilities:
    """Tests for VesselCapabilities dataclass."""

    def test_defaults(self):
        v = VesselCapabilities()
        assert v.max_wind_speed == 50.0
        assert v.max_wave_height == 5.0
        assert v.min_visibility == 0.5
        assert v.max_current == 3.0
        assert v.ice_capable is False
        assert v.vessel_size == "medium"

    def test_custom(self):
        v = VesselCapabilities(
            max_wind_speed=30.0,
            max_wave_height=3.0,
            ice_capable=True,
        )
        assert v.ice_capable is True
        assert v.max_wind_speed == 30.0


class TestSafeOperatingLimits:
    """Tests for SafeOperatingLimits dataclass."""

    def test_defaults(self):
        limits = SafeOperatingLimits(
            max_speed=15.0,
            max_wave_height=5.0,
            min_visibility=0.5,
            restricted_operations=["cargo_transfer"],
            caution_notes=["High waves"],
        )
        assert limits.max_speed == 15.0
        assert len(limits.caution_notes) == 1


class TestWeatherForecast:
    """Tests for WeatherForecast dataclass."""

    def test_construction(self):
        f = WeatherForecast(
            timestamp=1700000000.0,
            wind_speed=20.0,
            wave_height=2.0,
            visibility=10.0,
            current_speed=1.0,
        )
        assert f.wind_speed == 20.0


class TestImpactedSegment:
    """Tests for ImpactedSegment dataclass."""

    def test_construction(self):
        seg = ImpactedSegment(
            start=(50.0, -5.0),
            end=(51.0, -4.0),
            impact_level=RiskLevel.MODERATE,
            conditions=WeatherCondition(),
            recommendation="Monitor conditions.",
        )
        assert seg.impact_level == RiskLevel.MODERATE


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_all_levels(self):
        assert RiskLevel.NEGLIGIBLE.value == "negligible"
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MODERATE.value == "moderate"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.EXTREME.value == "extreme"


class TestOperationType:
    """Tests for OperationType enum."""

    def test_all_types(self):
        assert OperationType.NAVIGATION.value == "navigation"
        assert OperationType.CARGO_TRANSFER.value == "cargo_transfer"
        assert OperationType.PILOTAGE.value == "pilotage"
        assert OperationType.SEARCH_AND_RESCUE.value == "search_and_rescue"


class TestWeatherTrend:
    """Tests for WeatherTrend enum."""

    def test_all_trends(self):
        assert WeatherTrend.IMPROVING.value == "improving"
        assert WeatherTrend.STABLE.value == "stable"
        assert WeatherTrend.WORSENING.value == "worsening"
        assert WeatherTrend.RAPIDLY_WORSENING.value == "rapidly_worsening"


class TestAssessImpact:
    """Tests for weather impact assessment."""

    def test_clear_weather(self):
        assessor = WeatherAssessor()
        weather = WeatherCondition()
        impact = assessor.assess_impact(weather, "navigation")
        assert impact.risk_level == RiskLevel.NEGLIGIBLE
        assert len(impact.recommendations) > 0

    def test_high_wind(self):
        assessor = WeatherAssessor()
        weather = WeatherCondition(wind_speed=55.0)
        vessel = VesselCapabilities(max_wind_speed=50.0)
        impact = assessor.assess_impact(weather, "navigation", vessel)
        assert impact.risk_level in (RiskLevel.HIGH, RiskLevel.EXTREME, RiskLevel.MODERATE)
        assert "navigation" in impact.affected_operations

    def test_extreme_waves(self):
        assessor = WeatherAssessor()
        weather = WeatherCondition(wave_height=6.0)
        vessel = VesselCapabilities(max_wave_height=5.0)
        impact = assessor.assess_impact(weather, "navigation", vessel)
        assert impact.risk_level in (RiskLevel.HIGH, RiskLevel.EXTREME, RiskLevel.MODERATE)

    def test_low_visibility(self):
        assessor = WeatherAssessor()
        weather = WeatherCondition(visibility=0.3)
        vessel = VesselCapabilities(min_visibility=0.5)
        impact = assessor.assess_impact(weather, "navigation", vessel)
        assert impact.risk_level in (RiskLevel.MODERATE, RiskLevel.HIGH)
        assert "navigation" in impact.affected_operations

    def test_strong_current(self):
        assessor = WeatherAssessor()
        weather = WeatherCondition(current_speed=4.0)
        vessel = VesselCapabilities(max_current=3.0)
        impact = assessor.assess_impact(weather, "docking", vessel)
        assert "docking" in impact.affected_operations
        assert impact.estimated_delay > 0

    def test_cargo_transfer_suspended(self):
        assessor = WeatherAssessor()
        weather = WeatherCondition(wave_height=2.0, wind_speed=30.0)
        impact = assessor.assess_impact(weather, "cargo_transfer")
        # High wind or waves should produce some impact
        assert impact.risk_level != RiskLevel.NEGLIGIBLE or impact.estimated_delay > 0

    def test_docking_restricted(self):
        assessor = WeatherAssessor()
        weather = WeatherCondition(wind_speed=35.0, visibility=0.5)
        impact = assessor.assess_impact(weather, "docking")
        assert impact.risk_level in (RiskLevel.HIGH, RiskLevel.EXTREME)

    def test_fishing_hazardous(self):
        assessor = WeatherAssessor()
        weather = WeatherCondition(wave_height=3.0)
        impact = assessor.assess_impact(weather, "fishing")
        assert impact.risk_level != RiskLevel.NEGLIGIBLE

    def test_multiple_risk_factors(self):
        assessor = WeatherAssessor()
        weather = WeatherCondition(wind_speed=55.0, wave_height=6.0, visibility=0.3)
        vessel = VesselCapabilities(max_wind_speed=50.0, max_wave_height=5.0, min_visibility=0.5)
        impact = assessor.assess_impact(weather, "navigation", vessel)
        assert impact.risk_level == RiskLevel.EXTREME

    def test_no_custom_vessel(self):
        assessor = WeatherAssessor()
        weather = WeatherCondition()
        impact = assessor.assess_impact(weather, "navigation")
        assert impact.risk_level == RiskLevel.NEGLIGIBLE

    def test_operation_type_string_case(self):
        assessor = WeatherAssessor()
        weather = WeatherCondition()
        impact = assessor.assess_impact(weather, "NAVIGATION")
        assert impact.risk_level == RiskLevel.NEGLIGIBLE

    def test_estimated_delay_accumulation(self):
        assessor = WeatherAssessor()
        weather = WeatherCondition(wind_speed=55.0, wave_height=6.0, visibility=0.3)
        vessel = VesselCapabilities(max_wind_speed=50.0, max_wave_height=5.0, min_visibility=0.5)
        impact = assessor.assess_impact(weather, "navigation", vessel)
        assert impact.estimated_delay > 0


class TestComputeSeaState:
    """Tests for Douglas sea state computation."""

    def test_calm_glassy(self):
        assessor = WeatherAssessor()
        state, desc = assessor.compute_sea_state(5.0, 0.0)
        assert state == 0
        assert "glassy" in desc

    def test_calm_rippled(self):
        assessor = WeatherAssessor()
        state, desc = assessor.compute_sea_state(5.0, 0.2)
        assert state == 1
        assert "rippled" in desc

    def test_smooth(self):
        assessor = WeatherAssessor()
        state, desc = assessor.compute_sea_state(10.0, 0.4)
        assert state == 2
        assert "Smooth" in desc

    def test_slight(self):
        assessor = WeatherAssessor()
        state, desc = assessor.compute_sea_state(12.0, 0.8)
        assert state == 3
        assert "Slight" in desc

    def test_moderate_sea(self):
        assessor = WeatherAssessor()
        state, desc = assessor.compute_sea_state(20.0, 1.5)
        assert state == 4
        assert "Moderate" in desc

    def test_rough(self):
        assessor = WeatherAssessor()
        state, desc = assessor.compute_sea_state(25.0, 2.5)
        assert state == 5
        assert "Rough" in desc

    def test_very_rough(self):
        assessor = WeatherAssessor()
        state, desc = assessor.compute_sea_state(30.0, 3.5)
        assert state == 6
        assert "Very rough" in desc

    def test_high(self):
        assessor = WeatherAssessor()
        state, desc = assessor.compute_sea_state(40.0, 5.0)
        assert state == 7
        assert "High" in desc

    def test_very_high(self):
        assessor = WeatherAssessor()
        state, desc = assessor.compute_sea_state(50.0, 6.5)
        assert state == 8
        assert "Very high" in desc

    def test_phenomenal(self):
        assessor = WeatherAssessor()
        state, desc = assessor.compute_sea_state(60.0, 8.0)
        assert state == 9
        assert "Phenomenal" in desc

    def test_all_states_covered(self):
        assessor = WeatherAssessor()
        states = set()
        test_heights = [0.0, 0.2, 0.4, 0.8, 1.5, 2.5, 3.5, 5.0, 6.5, 8.0]
        for h in test_heights:
            state, _ = assessor.compute_sea_state(20.0, h)
            states.add(state)
        assert states == set(range(10))

    def test_returns_tuple(self):
        assessor = WeatherAssessor()
        result = assessor.compute_sea_state(10.0, 1.0)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestComputeSafeOperatingLimits:
    """Tests for safe operating limits computation."""

    def test_clear_weather_full_speed(self):
        assessor = WeatherAssessor()
        weather = WeatherCondition()
        limits = assessor.compute_safe_operating_limits(weather)
        assert limits.max_speed > 20.0
        assert len(limits.restricted_operations) == 0

    def test_high_wind_reduces_speed(self):
        assessor = WeatherAssessor()
        vessel = VesselCapabilities(max_wind_speed=50.0)
        weather = WeatherCondition(wind_speed=40.0)
        limits = assessor.compute_safe_operating_limits(weather, vessel)
        assert limits.max_speed < 25.0
        assert any("wind" in c.lower() for c in limits.caution_notes)

    def test_high_waves_reduces_speed(self):
        assessor = WeatherAssessor()
        vessel = VesselCapabilities(max_wave_height=5.0)
        weather = WeatherCondition(wave_height=4.0)
        limits = assessor.compute_safe_operating_limits(weather, vessel)
        assert limits.max_speed < 25.0

    def test_low_visibility_limits_speed(self):
        assessor = WeatherAssessor()
        weather = WeatherCondition(visibility=0.5)
        limits = assessor.compute_safe_operating_limits(weather)
        assert limits.max_speed <= 5.0
        assert "navigation" in limits.restricted_operations

    def test_moderate_visibility(self):
        assessor = WeatherAssessor()
        weather = WeatherCondition(visibility=2.0)
        limits = assessor.compute_safe_operating_limits(weather)
        assert limits.max_speed <= 10.0

    def test_cargo_transfer_restricted_by_waves(self):
        assessor = WeatherAssessor()
        weather = WeatherCondition(wave_height=2.0)
        limits = assessor.compute_safe_operating_limits(weather)
        assert "cargo_transfer" in limits.restricted_operations

    def test_pilotage_restricted_by_visibility(self):
        assessor = WeatherAssessor()
        weather = WeatherCondition(visibility=1.5)
        limits = assessor.compute_safe_operating_limits(weather)
        assert "pilotage" in limits.restricted_operations

    def test_docking_restricted_by_wind(self):
        assessor = WeatherAssessor()
        weather = WeatherCondition(wind_speed=40.0)
        limits = assessor.compute_safe_operating_limits(weather)
        assert "docking" in limits.restricted_operations

    def test_no_custom_vessel(self):
        assessor = WeatherAssessor()
        weather = WeatherCondition()
        limits = assessor.compute_safe_operating_limits(weather)
        assert limits.max_speed > 0
        assert isinstance(limits.caution_notes, list)

    def test_combined_restrictions(self):
        assessor = WeatherAssessor()
        weather = WeatherCondition(
            wind_speed=40.0, wave_height=4.0, visibility=0.5
        )
        limits = assessor.compute_safe_operating_limits(weather)
        assert len(limits.restricted_operations) >= 2
        assert len(limits.caution_notes) >= 2


class TestPredictWeatherChange:
    """Tests for weather trend prediction."""

    def test_empty_forecast_stable(self):
        assessor = WeatherAssessor()
        current = WeatherCondition()
        trend = assessor.predict_weather_change(current, [])
        assert trend == WeatherTrend.STABLE

    def test_improving_wind(self):
        assessor = WeatherAssessor()
        current = WeatherCondition(wind_speed=30.0)
        forecast = [
            WeatherForecast(0, 5.0, 0.5, 25.0, 0.5),
            WeatherForecast(3600, 3.0, 0.3, 30.0, 0.3),
        ]
        trend = assessor.predict_weather_change(current, forecast)
        assert trend in (WeatherTrend.IMPROVING, WeatherTrend.STABLE)

    def test_worsening_wind(self):
        assessor = WeatherAssessor()
        current = WeatherCondition(wind_speed=10.0)
        forecast = [
            WeatherForecast(0, 30.0, 2.0, 20.0, 1.0),
            WeatherForecast(3600, 40.0, 3.0, 15.0, 1.5),
        ]
        trend = assessor.predict_weather_change(current, forecast)
        assert trend in (WeatherTrend.WORSENING, WeatherTrend.RAPIDLY_WORSENING)

    def test_rapidly_worsening(self):
        assessor = WeatherAssessor()
        current = WeatherCondition(wind_speed=10.0, wave_height=1.0)
        forecast = [
            WeatherForecast(0, 40.0, 4.0, 5.0, 2.0),
            WeatherForecast(3600, 50.0, 5.0, 3.0, 2.5),
        ]
        trend = assessor.predict_weather_change(current, forecast)
        assert trend == WeatherTrend.RAPIDLY_WORSENING

    def test_stable_conditions(self):
        assessor = WeatherAssessor()
        current = WeatherCondition(wind_speed=15.0, wave_height=1.5)
        forecast = [
            WeatherForecast(0, 14.0, 1.3, 18.0, 0.8),
            WeatherForecast(3600, 16.0, 1.7, 22.0, 1.2),
        ]
        trend = assessor.predict_weather_change(current, forecast)
        assert trend == WeatherTrend.STABLE

    def test_improving_visibility(self):
        assessor = WeatherAssessor()
        current = WeatherCondition(visibility=2.0)
        forecast = [
            WeatherForecast(0, 5.0, 0.5, 10.0, 0.5),
            WeatherForecast(3600, 5.0, 0.5, 15.0, 0.5),
        ]
        trend = assessor.predict_weather_change(current, forecast)
        # Improving visibility + stable wind should show improvement or stable
        assert trend in (WeatherTrend.IMPROVING, WeatherTrend.STABLE)


class TestComputeRouteWeatherImpact:
    """Tests for route weather impact computation."""

    def test_empty_route(self):
        assessor = WeatherAssessor()
        segments = assessor.compute_route_weather_impact([], {})
        assert segments == []

    def test_single_point_route(self):
        assessor = WeatherAssessor()
        segments = assessor.compute_route_weather_impact([(50.0, -5.0)], {})
        assert segments == []

    def test_route_with_weather_data(self):
        assessor = WeatherAssessor()
        route = [(50.0, -5.0), (50.5, -4.5), (51.0, -4.0)]
        weather = {
            (50.25, -4.75): WeatherCondition(wind_speed=25.0, wave_height=2.0),
            (50.75, -4.25): WeatherCondition(wind_speed=10.0, wave_height=0.5),
        }
        segments = assessor.compute_route_weather_impact(route, weather)
        assert len(segments) == 2
        for seg in segments:
            assert isinstance(seg, ImpactedSegment)
            assert isinstance(seg.impact_level, RiskLevel)

    def test_route_no_weather_data(self):
        assessor = WeatherAssessor()
        route = [(50.0, -5.0), (51.0, -4.0)]
        segments = assessor.compute_route_weather_impact(route, {})
        assert len(segments) == 1
        assert segments[0].recommendation == "No weather data available for this segment."

    def test_severe_weather_segment(self):
        assessor = WeatherAssessor()
        route = [(50.0, -5.0), (51.0, -4.0)]
        weather = {
            (50.5, -4.5): WeatherCondition(wind_speed=60.0, wave_height=8.0, visibility=0.1),
        }
        segments = assessor.compute_route_weather_impact(route, weather)
        assert segments[0].impact_level in (RiskLevel.HIGH, RiskLevel.EXTREME)
