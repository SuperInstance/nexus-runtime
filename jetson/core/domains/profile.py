"""NEXUS Domain Portability — Domain profiles for cross-domain operation.

Supports marine, agriculture, factory, HVAC, and generic domains.
Each profile defines safety limits, trust defaults, sensor/actuator maps, and reflex templates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DomainProfile:
    """Complete domain configuration for NEXUS operation."""
    domain_id: str = "generic"
    name: str = "Generic"
    version: str = "1.0.0"
    description: str = ""

    # Safety limits
    max_speed: float = 10.0
    max_acceleration: float = 2.0
    max_deceleration: float = 5.0
    max_rudder_angle: float = 45.0
    max_throttle_pct: float = 100.0
    proximity_limit_m: float = 1.0
    max_heading_rate: float = 30.0  # deg/sec
    max_operation_hours: float = 24.0

    # Trust defaults
    initial_trust: float = 0.5
    trust_decay_rate: float = 0.01
    trust_recovery_rate: float = 0.05
    event_weights: dict[str, float] = field(default_factory=lambda: {
        "good": 0.1, "bad": -0.25, "neutral": 0.0, "critical": -0.5
    })

    # Safety pins (0-3 are safety-critical in NEXUS)
    safety_pins: list[int] = field(default_factory=lambda: [0, 1, 2, 3])

    # Sensor map: physical sensor -> VM pin
    sensor_map: dict[str, int] = field(default_factory=dict)
    # Actuator map: VM pin -> physical actuator
    actuator_map: dict[int, str] = field(default_factory=dict)
    # Actuator ranges: actuator_name -> (min, max)
    actuator_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)

    # No-go zones
    no_go_zones: list[dict[str, Any]] = field(default_factory=list)

    # Environmental limits
    min_temperature_c: float = -20.0
    max_temperature_c: float = 60.0
    max_humidity_pct: float = 100.0
    max_vibration_g: float = 10.0
    max_wind_knots: float = 50.0
    max_wave_height_m: float = 10.0

    # Reflex templates available for this domain
    available_reflexes: list[str] = field(default_factory=list)

    # Custom domain parameters
    custom_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DomainProfile:
        safe = {}
        for k, v in d.items():
            if k in cls.__dataclass_fields__:
                safe[k] = v
        return cls(**safe)


# ===================================================================
# Built-in Profiles
# ===================================================================

def marine_profile() -> DomainProfile:
    return DomainProfile(
        domain_id="marine",
        name="Marine",
        version="1.0.0",
        description="Marine autonomous surface vessel operations",
        max_speed=10.0, max_throttle_pct=80.0, max_rudder_angle=45.0,
        proximity_limit_m=50.0, max_heading_rate=10.0,
        initial_trust=0.5, trust_decay_rate=0.005, trust_recovery_rate=0.03,
        event_weights={"good": 0.1, "bad": -0.25, "neutral": 0.0, "critical": -0.5},
        sensor_map={
            "gps_lat": 16, "gps_lon": 17, "compass_heading": 18,
            "speed_sog": 19, "sonar_depth": 20, "water_temp": 21,
            "wind_speed": 22, "wind_dir": 23, "battery_voltage": 24,
            "rudder_feedback": 25, "throttle_feedback": 26,
        },
        actuator_map={8: "rudder_servo", 9: "throttle_motor", 10: "anchor_winch"},
        actuator_ranges={"rudder_servo": (-45.0, 45.0), "throttle_motor": (0.0, 80.0), "anchor_winch": (0.0, 100.0)},
        no_go_zones=[
            {"name": "default_shallow", "bounds": {"south": -90, "north": 90, "west": -180, "east": 180}, "min_depth_m": 2.0}
        ],
        min_temperature_c=-10.0, max_temperature_c=55.0,
        max_wind_knots=25.0, max_wave_height_m=2.0,
        available_reflexes=["heading_hold", "collision_avoidance", "waypoint_follow", "station_keeping", "emergency_stop"],
        custom_params={"colregs_enabled": True, "ais_enabled": True},
    )


def agriculture_profile() -> DomainProfile:
    return DomainProfile(
        domain_id="agriculture",
        name="Agriculture",
        version="1.0.0",
        description="Agricultural autonomous vehicle operations",
        max_speed=4.0, max_throttle_pct=60.0, max_rudder_angle=30.0,
        proximity_limit_m=2.0, max_heading_rate=15.0,
        initial_trust=0.6, trust_decay_rate=0.008, trust_recovery_rate=0.04,
        sensor_map={
            "gps_lat": 16, "gps_lon": 17, "soil_moisture": 18,
            "crop_health": 19, "air_temp": 20, "humidity": 21,
        },
        actuator_map={8: "steering", 9: "drive_motor", 10: "sprayer", 11: "plow"},
        actuator_ranges={"steering": (-30.0, 30.0), "drive_motor": (0.0, 60.0), "sprayer": (0.0, 100.0)},
        min_temperature_c=0.0, max_temperature_c=45.0,
        available_reflexes=["row_follow", "obstacle_avoidance", "spray_control", "boundary_patrol"],
        custom_params={"row_spacing_m": 0.75, "max_field_slope_deg": 15.0},
    )


def factory_profile() -> DomainProfile:
    return DomainProfile(
        domain_id="factory",
        name="Factory",
        version="1.0.0",
        description="Factory floor autonomous robot operations",
        max_speed=2.0, max_throttle_pct=40.0, max_rudder_angle=0.0,
        proximity_limit_m=0.3, max_heading_rate=45.0,
        initial_trust=0.7, trust_decay_rate=0.01, trust_recovery_rate=0.05,
        sensor_map={
            "lidar_front": 16, "lidar_rear": 17,
            "force_torque": 18, "vision": 19, "battery": 20,
        },
        actuator_map={8: "left_wheel", 9: "right_wheel", 10: "gripper", 11: "conveyor"},
        actuator_ranges={"left_wheel": (-2.0, 2.0), "right_wheel": (-2.0, 2.0), "gripper": (0.0, 50.0)},
        min_temperature_c=5.0, max_temperature_c=45.0, max_humidity_pct=85.0,
        available_reflexes=["obstacle_avoidance", "line_follow", "pick_and_place", "conveyor_sync"],
        custom_params={"human_detection_required": True, "safety_zone_m": 0.3},
    )


def hvac_profile() -> DomainProfile:
    return DomainProfile(
        domain_id="hvac",
        name="HVAC",
        version="1.0.0",
        description="HVAC system control operations",
        max_speed=0.0, max_throttle_pct=100.0, max_rudder_angle=0.0,
        proximity_limit_m=0.0,
        initial_trust=0.8, trust_decay_rate=0.003, trust_recovery_rate=0.02,
        sensor_map={
            "temp_supply": 16, "temp_return": 17, "temp_zone": 18,
            "humidity": 19, "co2_ppm": 20, "airflow_cfm": 21,
        },
        actuator_map={8: "damper_1", 9: "damper_2", 10: "valve_hot", 11: "valve_cold", 12: "fan_speed"},
        actuator_ranges={
            "damper_1": (0.0, 100.0), "damper_2": (0.0, 100.0),
            "valve_hot": (0.0, 100.0), "valve_cold": (0.0, 100.0),
            "fan_speed": (0.0, 100.0),
        },
        min_temperature_c=5.0, max_temperature_c=40.0, max_humidity_pct=70.0,
        available_reflexes=["temp_control", "humidity_control", "ventilation", "schedule_control"],
        custom_params={"setpoint_temp_c": 22.0, "setpoint_humidity_pct": 50.0, "deadband_c": 1.0},
    )


def generic_profile() -> DomainProfile:
    return DomainProfile(
        domain_id="generic",
        name="Generic",
        version="1.0.0",
        description="Minimal default profile for custom applications",
        available_reflexes=["emergency_stop"],
    )


# Registry
BUILT_IN_PROFILES: dict[str, callable] = {
    "marine": marine_profile,
    "agriculture": agriculture_profile,
    "factory": factory_profile,
    "hvac": hvac_profile,
    "generic": generic_profile,
}
