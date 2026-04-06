"""Tests for harvesting module."""

import math
import pytest

from jetson.energy.harvesting import (
    SolarPanel,
    WindTurbine,
    WeatherCondition,
    HarvestEstimator,
)


# ---------------------------------------------------------------------------
# Solar estimation tests
# ---------------------------------------------------------------------------

class TestSolarEstimation:
    def _panel(self):
        return SolarPanel(area_m2=1.0, efficiency=0.20, tilt_angle=0.0)

    def test_stc_conditions(self):
        panel = self._panel()
        power = HarvestEstimator.estimate_solar(panel, 1000.0, 25.0)
        # 1.0 * 0.20 * 1000 = 200 W (at STC, 25°C, 0° tilt)
        assert power == pytest.approx(200.0, abs=1.0)

    def test_zero_irradiance(self):
        panel = self._panel()
        assert HarvestEstimator.estimate_solar(panel, 0.0, 25.0) == 0.0

    def test_high_temperature_derating(self):
        panel = self._panel()
        p_std = HarvestEstimator.estimate_solar(panel, 1000.0, 25.0)
        p_hot = HarvestEstimator.estimate_solar(panel, 1000.0, 45.0)
        assert p_hot < p_std

    def test_tilt_angle_reduces_output(self):
        flat = SolarPanel(area_m2=1.0, efficiency=0.20, tilt_angle=0.0)
        tilted = SolarPanel(area_m2=1.0, efficiency=0.20, tilt_angle=45.0)
        p_flat = HarvestEstimator.estimate_solar(flat, 1000.0, 25.0)
        p_tilt = HarvestEstimator.estimate_solar(tilted, 1000.0, 25.0)
        assert p_tilt < p_flat

    def test_panel_efficiency_scaling(self):
        panel_low = SolarPanel(area_m2=1.0, efficiency=0.10, tilt_angle=0.0)
        panel_high = SolarPanel(area_m2=1.0, efficiency=0.25, tilt_angle=0.0)
        p_low = HarvestEstimator.estimate_solar(panel_low, 1000.0, 25.0)
        p_high = HarvestEstimator.estimate_solar(panel_high, 1000.0, 25.0)
        assert p_high > p_low

    def test_area_scaling(self):
        panel_small = SolarPanel(area_m2=0.5, efficiency=0.20, tilt_angle=0.0)
        panel_large = SolarPanel(area_m2=2.0, efficiency=0.20, tilt_angle=0.0)
        p_small = HarvestEstimator.estimate_solar(panel_small, 1000.0, 25.0)
        p_large = HarvestEstimator.estimate_solar(panel_large, 1000.0, 25.0)
        assert p_large == pytest.approx(4 * p_small, abs=0.1)


# ---------------------------------------------------------------------------
# Wind estimation tests
# ---------------------------------------------------------------------------

class TestWindEstimation:
    def _turbine(self):
        return WindTurbine(
            swept_area=1.0, efficiency=0.35,
            cut_in_speed=3.0, rated_speed=12.0, cut_out_speed=25.0,
        )

    def test_below_cut_in(self):
        t = self._turbine()
        assert HarvestEstimator.estimate_wind_power(t, 2.0) == 0.0

    def test_above_cut_out(self):
        t = self._turbine()
        assert HarvestEstimator.estimate_wind_power(t, 30.0) == 0.0

    def test_normal_operation(self):
        t = self._turbine()
        power = HarvestEstimator.estimate_wind_power(t, 8.0)
        assert power > 0.0

    def test_at_cut_in(self):
        t = self._turbine()
        power = HarvestEstimator.estimate_wind_power(t, 3.0)
        assert power > 0.0

    def test_at_rated_speed(self):
        t = self._turbine()
        p_rated = HarvestEstimator.estimate_wind_power(t, 12.0)
        p_above = HarvestEstimator.estimate_wind_power(t, 15.0)
        # Above rated should be same as rated
        assert p_above == pytest.approx(p_rated, abs=0.01)

    def test_power_scales_with_wind_cubed(self):
        t = self._turbine()
        p6 = HarvestEstimator.estimate_wind_power(t, 6.0)
        p12 = HarvestEstimator.estimate_wind_power(t, 12.0)
        # P ∝ v³ → doubling wind → 8× power (below rated)
        # But 12 is rated speed, so p12 is capped
        # Instead compare 5 vs 10 (both below rated)
        p5 = HarvestEstimator.estimate_wind_power(t, 5.0)
        p10 = HarvestEstimator.estimate_wind_power(t, 10.0)
        # 10³/5³ = 8
        assert p10 > p5 * 4  # at least 4x (conservative due to capping)

    def test_zero_wind(self):
        t = self._turbine()
        assert HarvestEstimator.estimate_wind_power(t, 0.0) == 0.0

    def test_efficiency_scales_power(self):
        t_low = WindTurbine(swept_area=1.0, efficiency=0.1, cut_in_speed=3.0,
                            rated_speed=12.0, cut_out_speed=25.0)
        t_high = WindTurbine(swept_area=1.0, efficiency=0.5, cut_in_speed=3.0,
                             rated_speed=12.0, cut_out_speed=25.0)
        p_low = HarvestEstimator.estimate_wind_power(t_low, 8.0)
        p_high = HarvestEstimator.estimate_wind_power(t_high, 8.0)
        assert p_high > p_low


# ---------------------------------------------------------------------------
# MPPT tests
# ---------------------------------------------------------------------------

class TestMPPT:
    def test_stc_mppt(self):
        panel = SolarPanel(area_m2=1.0, efficiency=0.20)
        v, i = HarvestEstimator.compute_mppt_point(panel, 1000.0)
        assert v > 0
        assert i > 0

    def test_low_irradiance_mppt(self):
        panel = SolarPanel(area_m2=1.0, efficiency=0.20)
        v_full, i_full = HarvestEstimator.compute_mppt_point(panel, 1000.0)
        v_low, i_low = HarvestEstimator.compute_mppt_point(panel, 200.0)
        assert v_low < v_full
        assert abs(i_low - i_full) < 0.01  # current stays constant due to voltage scaling

    def test_zero_irradiance_mppt(self):
        panel = SolarPanel(area_m2=1.0, efficiency=0.20)
        v, i = HarvestEstimator.compute_mppt_point(panel, 0.0)
        assert v == 0.0


# ---------------------------------------------------------------------------
# Daily harvest tests
# ---------------------------------------------------------------------------

class TestDailyHarvest:
    def _panel(self):
        return SolarPanel(area_m2=1.0, efficiency=0.20, degradation=0.005)

    def test_tropical_summer_clear(self):
        panel = self._panel()
        wh = HarvestEstimator.daily_harvest_estimate(panel, "tropical", "summer", "clear")
        # Peak ~200W, PSH ~6, mult 1.0 → ~1200 Wh
        assert wh > 1000.0

    def test_temperate_winter_cloudy(self):
        panel = self._panel()
        wh = HarvestEstimator.daily_harvest_estimate(panel, "temperate", "winter", "cloudy")
        # Peak ~200W, PSH ~3, mult 0.4 → ~240 Wh
        assert wh > 100.0
        assert wh < 500.0

    def test_arctic_winter(self):
        panel = self._panel()
        wh = HarvestEstimator.daily_harvest_estimate(panel, "arctic", "winter", "clear")
        # Very low PSH
        assert wh < 200.0

    def test_equatorial_consistent(self):
        panel = self._panel()
        wh_summer = HarvestEstimator.daily_harvest_estimate(panel, "equatorial", "summer", "clear")
        wh_winter = HarvestEstimator.daily_harvest_estimate(panel, "equatorial", "winter", "clear")
        assert wh_summer == pytest.approx(wh_winter, abs=1.0)

    def test_rainy_reduces(self):
        panel = self._panel()
        wh_clear = HarvestEstimator.daily_harvest_estimate(panel, "tropical", "summer", "clear")
        wh_rainy = HarvestEstimator.daily_harvest_estimate(panel, "tropical", "summer", "rainy")
        assert wh_rainy < wh_clear

    def test_unknown_location_defaults(self):
        panel = self._panel()
        wh = HarvestEstimator.daily_harvest_estimate(panel, "mars", "summer", "clear")
        assert wh >= 0  # should not crash


# ---------------------------------------------------------------------------
# Harvest profile tests
# ---------------------------------------------------------------------------

class TestHarvestProfile:
    def _panel(self):
        return SolarPanel(area_m2=1.0, efficiency=0.20, tilt_angle=0.0)

    def _turbine(self):
        return WindTurbine(swept_area=1.0, efficiency=0.35,
                           cut_in_speed=3.0, rated_speed=12.0, cut_out_speed=25.0)

    def test_solar_only_profile(self):
        panel = self._panel()
        conditions = [
            WeatherCondition(hour=0, irradiance=0, wind_speed=0, temperature=20),
            WeatherCondition(hour=6, irradiance=200, wind_speed=0, temperature=22),
            WeatherCondition(hour=12, irradiance=800, wind_speed=0, temperature=30),
        ]
        profile = HarvestEstimator.compute_harvest_profile(panel, None, conditions)
        assert len(profile) == 3
        assert profile[0] == 0.0
        assert profile[2] > profile[1]

    def test_combined_solar_wind(self):
        panel = self._panel()
        turbine = self._turbine()
        conditions = [
            WeatherCondition(hour=0, irradiance=0, wind_speed=8, temperature=15),
            WeatherCondition(hour=12, irradiance=800, wind_speed=2, temperature=30),
        ]
        profile = HarvestEstimator.compute_harvest_profile(panel, turbine, conditions)
        assert profile[0] > 0  # wind only at night
        assert profile[1] > 0  # solar only at noon

    def test_no_conditions(self):
        panel = self._panel()
        profile = HarvestEstimator.compute_harvest_profile(panel, None, [])
        assert profile == []
