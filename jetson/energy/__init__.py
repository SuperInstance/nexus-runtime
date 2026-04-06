"""NEXUS Phase 4 Round 8 — Energy Management.

Provides power budget allocation, Li-ion battery modelling, solar/wind
harvesting estimation, energy-aware mission planning, load shedding, and
energy consumption forecasting for marine robotics.
"""

from .power_budget import (
    ConsumerPriority,
    PowerConsumer,
    PowerBudget,
    PowerAllocator,
)

from .battery import (
    BatteryState,
    BatteryModel,
    BatterySimulator,
    LoadPoint,
    DEFAULT_PEUKERT_EXPONENT,
)

from .harvesting import (
    SolarPanel,
    WindTurbine,
    WeatherCondition,
    HarvestEstimator,
)

from .mission_energy import (
    MissionSegment,
    MissionEnergyPlan,
    EnvironmentalConditions,
    MissionEnergyPlanner,
)

from .load_shedding import (
    LoadPriority,
    SheddingStrategy,
    ShedAction,
    ShedReport,
    LoadShedManager,
)

from .forecasting import (
    EnergyReading,
    ConsumptionForecast,
    ChargingWindow,
    ForecastReport,
    EnergyForecaster,
)

__all__ = [
    # power_budget
    "ConsumerPriority",
    "PowerConsumer",
    "PowerBudget",
    "PowerAllocator",
    # battery
    "BatteryState",
    "BatteryModel",
    "BatterySimulator",
    "LoadPoint",
    "DEFAULT_PEUKERT_EXPONENT",
    # harvesting
    "SolarPanel",
    "WindTurbine",
    "WeatherCondition",
    "HarvestEstimator",
    # mission_energy
    "MissionSegment",
    "MissionEnergyPlan",
    "EnvironmentalConditions",
    "MissionEnergyPlanner",
    # load_shedding
    "LoadPriority",
    "SheddingStrategy",
    "ShedAction",
    "ShedReport",
    "LoadShedManager",
    # forecasting
    "EnergyReading",
    "ConsumptionForecast",
    "ChargingWindow",
    "ForecastReport",
    "EnergyForecaster",
]
