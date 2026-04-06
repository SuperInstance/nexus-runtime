"""Energy consumption forecasting and risk assessment."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from .battery import BatteryState, BatteryModel


@dataclass
class EnergyReading:
    """A single energy reading at a point in time."""
    timestamp: float             # seconds (epoch or relative)
    consumption_w: float
    generation_w: float
    battery_soc: float           # percent


@dataclass
class ConsumptionForecast:
    """Forecast of future energy consumption."""
    predicted_consumption: List[float]  # watts at each future hour
    confidence: float                  # 0-1 overall confidence
    time_range: Tuple[float, float]    # (start_hour, end_hour)


@dataclass
class ForecastReport:
    """Comprehensive forecast report."""
    avg_predicted_consumption: float
    peak_predicted_consumption: float
    battery_soc_trajectory: List[float]
    risk_score: float             # 0-1
    risk_level: str               # LOW / MEDIUM / HIGH / CRITICAL
    charging_windows: List[Tuple[float, float]]  # (start_hour, end_hour)


@dataclass
class ChargingWindow:
    """A recommended charging time window."""
    start_hour: float
    end_hour: float
    expected_generation: float    # Wh during window
    priority: float               # 0-1


class EnergyForecaster:
    """Forecasts energy consumption, battery SoC, and risk."""

    # Maximum history length for moving average
    MAX_HISTORY = 168  # one week of hourly readings

    @staticmethod
    def forecast_consumption(
        history: List[EnergyReading],
        hours_ahead: int,
        window_size: int = 24,
    ) -> ConsumptionForecast:
        """Forecast consumption using a simple moving average.

        Uses the last *window_size* readings to compute the average,
        then projects that forward for *hours_ahead* hours.
        Confidence decreases as we forecast further out.
        """
        if not history:
            return ConsumptionForecast(
                predicted_consumption=[0.0] * hours_ahead,
                confidence=0.0,
                time_range=(0.0, float(hours_ahead)),
            )

        # Use most recent readings up to window_size
        recent = history[-window_size:] if len(history) > window_size else history
        avg = sum(r.consumption_w for r in recent) / len(recent)

        # Compute standard deviation for confidence
        if len(recent) > 1:
            variance = sum((r.consumption_w - avg) ** 2 for r in recent) / len(recent)
            std_dev = math.sqrt(variance)
            cv = std_dev / avg if avg > 0 else 1.0
            confidence = max(0.1, min(1.0, 1.0 - cv))
        else:
            confidence = 0.5

        # Predicted values: slight decay of confidence over time
        predictions = []
        for i in range(hours_ahead):
            # Add slight random-ish variation using deterministic spread
            variation = 1.0 + 0.05 * math.sin(i * 0.5)
            predictions.append(round(avg * variation, 2))

        start = history[-1].timestamp / 3600.0 if history else 0.0
        end = start + hours_ahead

        return ConsumptionForecast(
            predicted_consumption=predictions,
            confidence=round(confidence, 3),
            time_range=(round(start, 2), round(end, 2)),
        )

    @staticmethod
    def detect_consumption_pattern(
        history: List[EnergyReading],
        period: int = 24,
    ) -> List[float]:
        """Detect periodic consumption pattern.

        Groups readings by hour modulo *period* and returns the
        average consumption for each time slot.
        """
        if not history:
            return [0.0] * period

        buckets: List[List[float]] = [[] for _ in range(period)]
        for reading in history:
            hour_idx = int(reading.timestamp / 3600.0) % period
            buckets[hour_idx].append(reading.consumption_w)

        pattern = []
        for bucket in buckets:
            if bucket:
                pattern.append(round(sum(bucket) / len(bucket), 2))
            else:
                pattern.append(0.0)

        return pattern

    @staticmethod
    def forecast_battery_soc(
        history: List[EnergyReading],
        forecast: ConsumptionForecast,
        initial_soc: float,
        battery_capacity_wh: float,
        harvest_estimate: Optional[List[float]] = None,
    ) -> List[float]:
        """Forecast battery SoC trajectory over the forecast horizon.

        Each step represents one hour.
        """
        trajectory = [initial_soc]

        if battery_capacity_wh <= 0:
            return [initial_soc] + [0.0] * len(forecast.predicted_consumption)

        current_soc = initial_soc

        for i, consumption_w in enumerate(forecast.predicted_consumption):
            # Energy consumed this hour
            consumed_wh = consumption_w * 1.0  # 1 hour

            # Energy generated this hour
            generated_wh = 0.0
            if harvest_estimate and i < len(harvest_estimate):
                generated_wh = harvest_estimate[i] * 1.0  # watts * 1 hour = Wh

            # Net energy change
            net_wh = generated_wh - consumed_wh
            soc_change = (net_wh / battery_capacity_wh) * 100.0

            current_soc = max(0.0, min(100.0, current_soc + soc_change))
            trajectory.append(round(current_soc, 2))

        return trajectory

    @staticmethod
    def compute_energy_deficit_risk(
        forecast: ConsumptionForecast,
        battery: BatteryState,
        threshold_soc: float = 20.0,
    ) -> float:
        """Compute a risk score (0-1) for energy deficit.

        Risk is higher when:
        - Predicted consumption is high relative to battery capacity
        - Forecast confidence is low
        """
        if not forecast.predicted_consumption:
            return 0.0

        avg_consumption = sum(forecast.predicted_consumption) / len(forecast.predicted_consumption)
        peak_consumption = max(forecast.predicted_consumption)

        # Hours until battery depletion at average consumption
        if avg_consumption > 0:
            hours_until_empty = (battery.capacity_wh / avg_consumption)
        else:
            hours_until_empty = 9999.0

        forecast_hours = len(forecast.predicted_consumption)

        # Risk factors
        depletion_risk = max(0.0, 1.0 - hours_until_empty / forecast_hours) if forecast_hours > 0 else 0.0
        low_soc_risk = max(0.0, 1.0 - battery.charge_percent / 100.0)
        uncertainty_risk = 1.0 - forecast.confidence
        peak_risk = min(1.0, peak_consumption / (battery.capacity_wh + 1)) * 0.3

        risk = (
            depletion_risk * 0.4 +
            low_soc_risk * 0.3 +
            uncertainty_risk * 0.2 +
            peak_risk * 0.1
        )
        return round(max(0.0, min(1.0, risk)), 4)

    @staticmethod
    def optimize_charging_window(
        forecast: ConsumptionForecast,
        harvest_estimate: List[float],
        battery: BatteryState,
        min_window_hours: float = 2.0,
    ) -> List[ChargingWindow]:
        """Find optimal charging windows based on harvest vs consumption.

        A charging window is a contiguous period where generation > consumption.
        """
        windows: List[ChargingWindow] = []
        if not harvest_estimate or not forecast.predicted_consumption:
            return windows

        in_window = False
        window_start = 0.0
        window_gen = 0.0

        hours = min(len(harvest_estimate), len(forecast.predicted_consumption))

        for i in range(hours):
            gen = harvest_estimate[i]
            con = forecast.predicted_consumption[i]
            net = gen - con

            if net > 0 and not in_window:
                # Start a new window
                in_window = True
                window_start = float(i)
                window_gen = gen
            elif net > 0 and in_window:
                window_gen += gen
            elif net <= 0 and in_window:
                # Close the window
                window_end = float(i)
                duration = window_end - window_start
                if duration >= min_window_hours:
                    priority = min(1.0, window_gen / (battery.capacity_wh + 1))
                    windows.append(ChargingWindow(
                        start_hour=window_start,
                        end_hour=window_end,
                        expected_generation=round(window_gen, 2),
                        priority=round(priority, 3),
                    ))
                in_window = False
                window_gen = 0.0

        # Close trailing window
        if in_window:
            window_end = float(hours)
            duration = window_end - window_start
            if duration >= min_window_hours:
                priority = min(1.0, window_gen / (battery.capacity_wh + 1))
                windows.append(ChargingWindow(
                    start_hour=window_start,
                    end_hour=window_end,
                    expected_generation=round(window_gen, 2),
                    priority=round(priority, 3),
                ))

        # Sort by priority (best windows first)
        windows.sort(key=lambda w: w.priority, reverse=True)
        return windows

    @staticmethod
    def generate_forecast_report(
        forecast: ConsumptionForecast,
        battery: BatteryState,
        risk: float,
        soc_trajectory: Optional[List[float]] = None,
    ) -> ForecastReport:
        """Generate a comprehensive forecast report."""
        if not forecast.predicted_consumption:
            avg_c = 0.0
            peak_c = 0.0
        else:
            avg_c = sum(forecast.predicted_consumption) / len(forecast.predicted_consumption)
            peak_c = max(forecast.predicted_consumption)

        if risk >= 0.75:
            level = "CRITICAL"
        elif risk >= 0.5:
            level = "HIGH"
        elif risk >= 0.25:
            level = "MEDIUM"
        else:
            level = "LOW"

        return ForecastReport(
            avg_predicted_consumption=round(avg_c, 2),
            peak_predicted_consumption=round(peak_c, 2),
            battery_soc_trajectory=soc_trajectory or [],
            risk_score=risk,
            risk_level=level,
            charging_windows=[],
        )
