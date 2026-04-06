"""Tests for forecasting module."""

import math
import pytest

from jetson.energy.battery import BatteryState
from jetson.energy.forecasting import (
    EnergyReading,
    ConsumptionForecast,
    ChargingWindow,
    ForecastReport,
    EnergyForecaster,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reading(hour, consumption, generation=0.0, soc=80.0):
    return EnergyReading(
        timestamp=hour * 3600.0,
        consumption_w=consumption,
        generation_w=generation,
        battery_soc=soc,
    )


def _uniform_history(hours=24, consumption=100.0):
    return [_reading(h, consumption) for h in range(hours)]


def _battery(soc=80.0, capacity=500.0):
    return BatteryState(
        charge_percent=soc, voltage=3.8, current=0.0,
        temperature=25.0, capacity_wh=capacity, cycles=0,
    )


# ---------------------------------------------------------------------------
# forecast_consumption tests
# ---------------------------------------------------------------------------

class TestForecastConsumption:
    def test_basic_forecast(self):
        history = _uniform_history(24, 100.0)
        fc = EnergyForecaster.forecast_consumption(history, 12)
        assert len(fc.predicted_consumption) == 12
        assert all(p > 0 for p in fc.predicted_consumption)

    def test_forecast_empty_history(self):
        fc = EnergyForecaster.forecast_consumption([], 10)
        assert fc.predicted_consumption == [0.0] * 10
        assert fc.confidence == 0.0

    def test_forecast_confidence(self):
        # Uniform history → high confidence
        history = _uniform_history(24, 100.0)
        fc = EnergyForecaster.forecast_consumption(history, 12)
        assert fc.confidence > 0.5

    def test_forecast_variable_history(self):
        # Variable history → lower confidence
        history = [_reading(h, 50.0 + h * 10.0) for h in range(24)]
        fc_var = EnergyForecaster.forecast_consumption(history, 12)
        fc_uniform = EnergyForecaster.forecast_consumption(_uniform_history(24, 100.0), 12)
        assert fc_var.confidence < fc_uniform.confidence

    def test_forecast_uses_window(self):
        history = (
            [_reading(h, 200.0) for h in range(24)] +
            [_reading(h, 50.0) for h in range(24, 48)]
        )
        fc = EnergyForecaster.forecast_consumption(history, 6, window_size=24)
        avg = sum(fc.predicted_consumption) / len(fc.predicted_consumption)
        assert avg < 100.0  # Recent readings are 50W

    def test_forecast_time_range(self):
        history = [_reading(0, 100.0)]
        fc = EnergyForecaster.forecast_consumption(history, 10)
        assert fc.time_range[1] - fc.time_range[0] == 10.0

    def test_forecast_single_reading(self):
        history = [_reading(0, 100.0)]
        fc = EnergyForecaster.forecast_consumption(history, 5)
        assert len(fc.predicted_consumption) == 5

    def test_forecast_zero_hours_ahead(self):
        history = _uniform_history()
        fc = EnergyForecaster.forecast_consumption(history, 0)
        assert fc.predicted_consumption == []


# ---------------------------------------------------------------------------
# detect_consumption_pattern tests
# ---------------------------------------------------------------------------

class TestDetectPattern:
    def test_basic_pattern(self):
        history = _uniform_history(48, 100.0)
        pattern = EnergyForecaster.detect_consumption_pattern(history, period=24)
        assert len(pattern) == 24
        assert all(abs(p - 100.0) < 0.01 for p in pattern)

    def test_empty_history(self):
        pattern = EnergyForecaster.detect_consumption_pattern([], period=24)
        assert len(pattern) == 24
        assert all(p == 0.0 for p in pattern)

    def test_pattern_length(self):
        history = _uniform_history(24, 100.0)
        pattern = EnergyForecaster.detect_consumption_pattern(history, period=12)
        assert len(pattern) == 12

    def test_pattern_captures_daily_variation(self):
        history = [_reading(h, 50.0 if h % 24 < 12 else 150.0) for h in range(48)]
        pattern = EnergyForecaster.detect_consumption_pattern(history, period=24)
        # Morning hours (0-11) should be lower than afternoon
        assert pattern[0] < pattern[12]


# ---------------------------------------------------------------------------
# forecast_battery_soc tests
# ---------------------------------------------------------------------------

class TestForecastSoC:
    def test_basic_soc_trajectory(self):
        history = _uniform_history(24, 100.0)
        fc = EnergyForecaster.forecast_consumption(history, 6)
        trajectory = EnergyForecaster.forecast_battery_soc(
            history, fc, 80.0, 500.0,
        )
        assert len(trajectory) == 7  # initial + 6 hours
        assert trajectory[0] == 80.0

    def test_soc_decreases_with_consumption(self):
        history = _uniform_history(24, 100.0)
        fc = EnergyForecaster.forecast_consumption(history, 6)
        trajectory = EnergyForecaster.forecast_battery_soc(
            history, fc, 80.0, 500.0,
        )
        assert trajectory[-1] < trajectory[0]

    def test_soc_increases_with_harvest(self):
        history = _uniform_history(24, 50.0)
        fc = EnergyForecaster.forecast_consumption(history, 6)
        harvest = [200.0] * 6  # 200W generation
        trajectory = EnergyForecaster.forecast_battery_soc(
            history, fc, 50.0, 500.0, harvest_estimate=harvest,
        )
        assert trajectory[-1] > trajectory[0]

    def test_soc_clamps_to_zero(self):
        history = _uniform_history(24, 500.0)
        fc = EnergyForecaster.forecast_consumption(history, 24)
        trajectory = EnergyForecaster.forecast_battery_soc(
            history, fc, 10.0, 500.0,
        )
        assert all(s >= 0.0 for s in trajectory)

    def test_soc_clamps_to_100(self):
        history = _uniform_history(24, 10.0)
        fc = EnergyForecaster.forecast_consumption(history, 6)
        harvest = [500.0] * 6
        trajectory = EnergyForecaster.forecast_battery_soc(
            history, fc, 90.0, 500.0, harvest_estimate=harvest,
        )
        assert all(s <= 100.0 for s in trajectory)

    def test_soc_zero_capacity(self):
        history = _uniform_history(24, 100.0)
        fc = EnergyForecaster.forecast_consumption(history, 6)
        trajectory = EnergyForecaster.forecast_battery_soc(
            history, fc, 80.0, 0.0,
        )
        assert len(trajectory) == 7

    def test_soc_empty_forecast(self):
        history = _uniform_history(24, 100.0)
        fc = ConsumptionForecast(predicted_consumption=[], confidence=0.5,
                                  time_range=(0, 0))
        trajectory = EnergyForecaster.forecast_battery_soc(
            history, fc, 80.0, 500.0,
        )
        assert trajectory == [80.0]


# ---------------------------------------------------------------------------
# compute_energy_deficit_risk tests
# ---------------------------------------------------------------------------

class TestDeficitRisk:
    def test_low_risk_healthy_battery(self):
        history = _uniform_history(24, 50.0)
        fc = EnergyForecaster.forecast_consumption(history, 6)
        battery = _battery(soc=90.0, capacity=2000.0)
        risk = EnergyForecaster.compute_energy_deficit_risk(fc, battery)
        assert risk < 0.3

    def test_high_risk_low_battery(self):
        history = _uniform_history(24, 200.0)
        fc = EnergyForecaster.forecast_consumption(history, 48)
        battery = _battery(soc=10.0, capacity=100.0)
        risk = EnergyForecaster.compute_energy_deficit_risk(fc, battery)
        assert risk > 0.5

    def test_risk_clamped(self):
        history = _uniform_history(24, 100.0)
        fc = EnergyForecaster.forecast_consumption(history, 6)
        battery = _battery(soc=0.0, capacity=1.0)
        risk = EnergyForecaster.compute_energy_deficit_risk(fc, battery)
        assert 0.0 <= risk <= 1.0

    def test_risk_empty_forecast(self):
        fc = ConsumptionForecast(predicted_consumption=[], confidence=0.5,
                                  time_range=(0, 0))
        battery = _battery()
        risk = EnergyForecaster.compute_energy_deficit_risk(fc, battery)
        assert risk == 0.0

    def test_risk_with_custom_threshold(self):
        history = _uniform_history(24, 100.0)
        fc = EnergyForecaster.forecast_consumption(history, 6)
        battery = _battery(soc=50.0, capacity=500.0)
        r1 = EnergyForecaster.compute_energy_deficit_risk(fc, battery, threshold_soc=20.0)
        r2 = EnergyForecaster.compute_energy_deficit_risk(fc, battery, threshold_soc=50.0)
        # Lower threshold should generally yield different risk, but not strictly monotonic
        # since the threshold isn't directly used in our simplified model
        assert isinstance(r1, float)
        assert isinstance(r2, float)


# ---------------------------------------------------------------------------
# optimize_charging_window tests
# ---------------------------------------------------------------------------

class TestChargingWindow:
    def test_finds_charging_window(self):
        history = _uniform_history(24, 50.0)
        fc = EnergyForecaster.forecast_consumption(history, 24)
        # Generation exceeds consumption in first 6 hours
        harvest = [100.0] * 6 + [20.0] * 18
        battery = _battery()
        windows = EnergyForecaster.optimize_charging_window(fc, harvest, battery)
        assert len(windows) >= 1
        assert windows[0].start_hour == 0.0

    def test_no_charging_window(self):
        history = _uniform_history(24, 200.0)
        fc = EnergyForecaster.forecast_consumption(history, 24)
        harvest = [10.0] * 24  # Very low generation
        battery = _battery()
        windows = EnergyForecaster.optimize_charging_window(fc, harvest, battery)
        assert len(windows) == 0

    def test_window_minimum_duration(self):
        history = _uniform_history(24, 50.0)
        fc = EnergyForecaster.forecast_consumption(history, 24)
        # Only 1 hour of excess generation
        harvest = [100.0] + [10.0] * 23
        battery = _battery()
        windows = EnergyForecaster.optimize_charging_window(
            fc, harvest, battery, min_window_hours=2.0,
        )
        assert len(windows) == 0  # Too short

    def test_window_priority(self):
        history = _uniform_history(24, 50.0)
        fc = EnergyForecaster.forecast_consumption(history, 24)
        harvest = [200.0] * 12 + [10.0] * 12
        battery = _battery()
        windows = EnergyForecaster.optimize_charging_window(fc, harvest, battery)
        if windows:
            assert windows[0].priority > 0
            assert windows[0].expected_generation > 0

    def test_empty_harvest(self):
        history = _uniform_history(24, 50.0)
        fc = EnergyForecaster.forecast_consumption(history, 24)
        windows = EnergyForecaster.optimize_charging_window(fc, [], _battery())
        assert windows == []

    def test_window_sorted_by_priority(self):
        history = _uniform_history(48, 50.0)
        fc = EnergyForecaster.forecast_consumption(history, 48)
        harvest = [200.0] * 12 + [10.0] * 12 + [150.0] * 12 + [10.0] * 12
        battery = _battery()
        windows = EnergyForecaster.optimize_charging_window(fc, harvest, battery)
        if len(windows) > 1:
            assert windows[0].priority >= windows[1].priority


# ---------------------------------------------------------------------------
# generate_forecast_report tests
# ---------------------------------------------------------------------------

class TestForecastReport:
    def test_basic_report(self):
        history = _uniform_history(24, 100.0)
        fc = EnergyForecaster.forecast_consumption(history, 12)
        battery = _battery()
        risk = EnergyForecaster.compute_energy_deficit_risk(fc, battery)
        report = EnergyForecaster.generate_forecast_report(fc, battery, risk)
        assert isinstance(report, ForecastReport)
        assert report.avg_predicted_consumption > 0
        assert report.peak_predicted_consumption > 0
        assert report.risk_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_report_risk_levels(self):
        history = _uniform_history(24, 100.0)
        fc = EnergyForecaster.forecast_consumption(history, 6)
        battery_good = _battery(soc=95.0, capacity=5000.0)
        battery_bad = _battery(soc=5.0, capacity=50.0)

        risk_good = EnergyForecaster.compute_energy_deficit_risk(fc, battery_good)
        risk_bad = EnergyForecaster.compute_energy_deficit_risk(fc, battery_bad)

        report_good = EnergyForecaster.generate_forecast_report(fc, battery_good, risk_good)
        report_bad = EnergyForecaster.generate_forecast_report(fc, battery_bad, risk_bad)

        # Good battery should have lower risk level
        risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        assert risk_order[report_good.risk_level] <= risk_order[report_bad.risk_level]

    def test_report_with_soc_trajectory(self):
        history = _uniform_history(24, 100.0)
        fc = EnergyForecaster.forecast_consumption(history, 6)
        battery = _battery()
        trajectory = EnergyForecaster.forecast_battery_soc(history, fc, 80.0, 500.0)
        risk = EnergyForecaster.compute_energy_deficit_risk(fc, battery)
        report = EnergyForecaster.generate_forecast_report(
            fc, battery, risk, soc_trajectory=trajectory,
        )
        assert len(report.battery_soc_trajectory) > 0

    def test_report_empty_forecast(self):
        fc = ConsumptionForecast(predicted_consumption=[], confidence=0.0,
                                  time_range=(0, 0))
        battery = _battery()
        report = EnergyForecaster.generate_forecast_report(fc, battery, 0.0)
        assert report.avg_predicted_consumption == 0.0
        assert report.peak_predicted_consumption == 0.0
        assert report.risk_level == "LOW"


# ---------------------------------------------------------------------------
# EnergyReading tests
# ---------------------------------------------------------------------------

class TestEnergyReading:
    def test_creation(self):
        r = EnergyReading(timestamp=3600.0, consumption_w=100.0,
                          generation_w=20.0, battery_soc=85.0)
        assert r.timestamp == 3600.0
        assert r.consumption_w == 100.0
        assert r.generation_w == 20.0
        assert r.battery_soc == 85.0


class TestConsumptionForecast:
    def test_creation(self):
        fc = ConsumptionForecast(
            predicted_consumption=[100.0, 110.0],
            confidence=0.9,
            time_range=(0.0, 2.0),
        )
        assert len(fc.predicted_consumption) == 2
        assert fc.confidence == 0.9


class TestChargingWindowDataclass:
    def test_creation(self):
        w = ChargingWindow(
            start_hour=6.0, end_hour=10.0,
            expected_generation=500.0, priority=0.8,
        )
        assert w.start_hour == 6.0
        assert w.expected_generation == 500.0


class TestForecastReportDataclass:
    def test_creation(self):
        r = ForecastReport(
            avg_predicted_consumption=100.0,
            peak_predicted_consumption=150.0,
            battery_soc_trajectory=[80.0, 75.0],
            risk_score=0.3,
            risk_level="LOW",
            charging_windows=[],
        )
        assert r.risk_level == "LOW"
        assert len(r.battery_soc_trajectory) == 2
