"""NEXUS Marine Safety Rules Database.

COLREGs rules, no-go zones, equipment limits, and environmental rules
for maritime autonomous operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class VesselSituation(Enum):
    POWER_DRIVEN = "power_driven"
    SAILING = "sailing"
    FISHING = "fishing"
    NOT_UNDER_COMMAND = "not_under_command"
    RESTRICTED_MANEUVERABILITY = "restricted_maneuverability"
    CONSTRAINED_BY_DRAFT = "constrained_by_draft"
    ANCHORED = "anchored"
    AGROUND = "aground"


@dataclass(frozen=True)
class COLREGsRule:
    """A single COLREGs rule."""
    rule_id: str
    name: str
    description: str
    situation: VesselSituation
    action: str  # stand_on, give_way, avoid
    priority: int  # higher = more important


@dataclass
class NoGoZone:
    """A geographic no-go zone."""
    name: str
    bounds: dict[str, float]  # south, north, west, east
    zone_type: str = "restricted"  # restricted, protected, shallow, military
    active: bool = True
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "bounds": self.bounds,
                "zone_type": self.zone_type, "active": self.active, "reason": self.reason}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NoGoZone:
        return cls(name=d.get("name", ""), bounds=d.get("bounds", {}),
                   zone_type=d.get("zone_type", "restricted"),
                   active=d.get("active", True), reason=d.get("reason", ""))


@dataclass
class EquipmentLimits:
    """Equipment operational limits."""
    max_speed_knots: float = 10.0
    max_rudder_angle_deg: float = 45.0
    max_throttle_pct: float = 80.0
    max_heading_rate_deg_per_sec: float = 10.0
    min_turn_radius_m: float = 5.0
    max_wind_knots: float = 25.0
    max_wave_height_m: float = 2.0
    max_current_knots: float = 3.0

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EquipmentLimits:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EnvironmentalRules:
    """Environmental protection rules."""
    min_distance_to_marine_life_m: float = 100.0
    min_distance_to_whales_m: float = 300.0
    max_noise_db: float = 120.0
    no_discharge_zones: bool = True
    prohibited_times: list[str] = field(default_factory=lambda: ["22:00-06:00"])  # nighttime in protected areas
    min_depth_m: float = 2.0  # minimum operating depth


class MarineRulesDatabase:
    """Database of marine safety rules, no-go zones, and equipment limits."""

    def __init__(self) -> None:
        self.colregs: list[COLREGsRule] = self._default_colregs()
        self.no_go_zones: list[NoGoZone] = []
        self.equipment_limits = EquipmentLimits()
        self.environmental_rules = EnvironmentalRules()

    def add_colregs(self, rule: COLREGsRule) -> None:
        self.colregs.append(rule)
        self.colregs.sort(key=lambda r: r.priority, reverse=True)

    def add_no_go_zone(self, zone: NoGoZone) -> None:
        self.no_go_zones.append(zone)

    def get_active_no_go_zones(self) -> list[dict[str, Any]]:
        return [z.to_dict() for z in self.no_go_zones if z.active]

    def check_colregs(self, situation: VesselSituation,
                      nearby_vessels: int = 0) -> list[COLREGsRule]:
        """Get applicable COLREGs rules for the current situation."""
        rules = [r for r in self.colregs if r.situation == situation]
        if nearby_vessels > 0:
            rules += [r for r in self.colregs if r.situation != situation and "avoid" in r.action]
        return rules

    def check_no_go_zone(self, lat: float, lon: float) -> NoGoZone | None:
        """Check if a position is in any no-go zone."""
        for zone in self.no_go_zones:
            if not zone.active:
                continue
            b = zone.bounds
            if b.get("south", -90) <= lat <= b.get("north", 90) and \
               b.get("west", -180) <= lon <= b.get("east", 180):
                return zone
        return None

    def check_equipment_limits(self, speed: float = 0.0, rudder: float = 0.0,
                                throttle: float = 0.0) -> list[str]:
        """Check if proposed values exceed equipment limits."""
        violations = []
        if speed > self.equipment_limits.max_speed_knots:
            violations.append(f"Speed {speed}kn exceeds max {self.equipment_limits.max_speed_knots}kn")
        if abs(rudder) > self.equipment_limits.max_rudder_angle_deg:
            violations.append(f"Rudder {rudder}° exceeds max {self.equipment_limits.max_rudder_angle_deg}°")
        if throttle > self.equipment_limits.max_throttle_pct:
            violations.append(f"Throttle {throttle}% exceeds max {self.equipment_limits.max_throttle_pct}%")
        return violations

    @staticmethod
    def _default_colregs() -> list[COLREGsRule]:
        return [
            COLREGsRule("COLREG-5", "Lookout", "Maintain proper lookout at all times",
                        VesselSituation.POWER_DRIVEN, "always", priority=100),
            COLREGsRule("COLREG-6", "Safe Speed", "Proceed at safe speed for conditions",
                        VesselSituation.POWER_DRIVEN, "always", priority=99),
            COLREGsRule("COLREG-7", "Risk Assessment", "Use all available means to assess collision risk",
                        VesselSituation.POWER_DRIVEN, "always", priority=98),
            COLREGsRule("COLREG-13", "Overtaking", "Vessel overtaking must keep clear",
                        VesselSituation.POWER_DRIVEN, "give_way", priority=90),
            COLREGsRule("COLREG-14", "Head-On", "Each vessel alters course to starboard",
                        VesselSituation.POWER_DRIVEN, "give_way", priority=91),
            COLREGsRule("COLREG-15", "Crossing", "Give way to vessel on starboard side",
                        VesselSituation.POWER_DRIVEN, "give_way", priority=89),
            COLREGsRule("COLREG-16", "Give Way Action", "Take early and substantial action",
                        VesselSituation.POWER_DRIVEN, "give_way", priority=88),
            COLREGsRule("COLREG-17", "Stand On", "Maintain course and speed",
                        VesselSituation.POWER_DRIVEN, "stand_on", priority=87),
            COLREGsRule("COLREG-18", "Responsibilities", "Fishing > Sailing > Power",
                        VesselSituation.FISHING, "stand_on", priority=85),
            COLREGsRule("COLREG-19", "Restricted Visibility", "Proceed at safe speed, radar",
                        VesselSituation.POWER_DRIVEN, "avoid", priority=95),
        ]
