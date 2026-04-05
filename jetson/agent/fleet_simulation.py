"""NEXUS End-to-End Fleet Simulation.

Simulates multi-vessel fleet operations with:
  - Multiple vessels with independent state
  - Competing reflex strategies (survival of the fittest)
  - Dead reckoning position updates
  - Trust score evolution
  - Safety event injection
  - CRDT-like state sync between vessels
  - Tripartite consensus for critical decisions
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ===================================================================
# Vessel State
# ===================================================================

class VesselStatus(Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    SAFE_STATE = "safe_state"
    FAULT = "fault"
    OFFLINE = "offline"


@dataclass
class VesselState:
    """Complete state of a single vessel."""
    vessel_id: str
    lat: float = 0.0
    lon: float = 0.0
    heading: float = 0.0   # degrees, 0=N, 90=E
    speed: float = 0.0     # knots
    fuel_pct: float = 100.0
    battery_pct: float = 100.0
    trust_score: float = 0.5
    trust_level: int = 0
    safety_state: VesselStatus = VesselStatus.ACTIVE
    uptime_ticks: int = 0
    total_distance_nm: float = 0.0
    reflex_deployments: int = 0
    safety_events: int = 0
    mission_complete: bool = False
    score: float = 0.0  # fitness score for competition

    def to_dict(self) -> dict[str, Any]:
        return {
            "vessel_id": self.vessel_id, "lat": round(self.lat, 6), "lon": round(self.lon, 6),
            "heading": round(self.heading, 1), "speed": round(self.speed, 2),
            "fuel_pct": round(self.fuel_pct, 1), "battery_pct": round(self.battery_pct, 1),
            "trust_score": round(self.trust_score, 4), "trust_level": self.trust_level,
            "safety_state": self.safety_state.value, "uptime_ticks": self.uptime_ticks,
            "total_distance_nm": round(self.total_distance_nm, 3),
            "reflex_deployments": self.reflex_deployments,
            "safety_events": self.safety_events, "score": round(self.score, 4),
        }


# ===================================================================
# Reflex Strategy (for competition)
# ===================================================================

class ReflexStrategy(Enum):
    CONSERVATIVE = "conservative"    # slow, safe, high trust threshold
    MODERATE = "moderate"            # balanced speed and safety
    AGGRESSIVE = "aggressive"        # fast, efficient, lower safety margin
    ADAPTIVE = "adaptive"            # adjusts based on conditions


@dataclass
class ReflexConfig:
    """Configuration for a reflex strategy."""
    name: str
    strategy: ReflexStrategy
    max_speed_knots: float = 8.0
    safety_margin_pct: float = 20.0  # extra margin on limits
    trust_threshold: int = 1
    reaction_distance_m: float = 100.0  # distance at which collision avoidance triggers
    efficiency_weight: float = 0.5  # balance between speed and safety (0=safe, 1=fast)


# ===================================================================
# Simulation Events
# ===================================================================

class SimEventType(Enum):
    TICK = "tick"
    WAYPOINT_REACHED = "waypoint_reached"
    SAFETY_EVENT = "safety_event"
    TRUST_CHANGE = "trust_change"
    REFLEX_DEPLOY = "reflex_deploy"
    COLLISION_WARNING = "collision_warning"
    FUEL_LOW = "fuel_low"
    MISSION_COMPLETE = "mission_complete"


@dataclass
class SimEvent:
    tick: int
    vessel_id: str
    event_type: SimEventType
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ===================================================================
# Fleet Simulation Engine
# ===================================================================

@dataclass
class Waypoint:
    lat: float
    lon: float
    arrival_radius_nm: float = 0.1


@dataclass
class SimulationConfig:
    num_vessels: int = 3
    num_waypoints: int = 5
    max_ticks: int = 1000
    tick_interval_sec: float = 1.0
    initial_fuel_pct: float = 100.0
    fuel_consumption_per_nm: float = 0.5
    collision_radius_nm: float = 0.05  # vessels collide if closer than this
    safety_event_probability: float = 0.01  # per tick per vessel
    waypoint_area_lat: tuple[float, float] = (48.0, 49.0)
    waypoint_area_lon: tuple[float, float] = (-123.0, -122.0)
    trust_decay_rate: float = 0.001
    trust_recovery_rate: float = 0.01
    competition_mode: bool = True  # if True, vessels compete on fitness score


class FleetSimulation:
    """End-to-end fleet simulation engine."""

    def __init__(self, config: SimulationConfig | None = None,
                 reflex_configs: list[ReflexConfig] | None = None) -> None:
        self.config = config or SimulationConfig()
        self.vessels: dict[str, VesselState] = {}
        self.events: list[SimEvent] = []
        self.tick: int = 0
        self.running = False
        self.waypoints: list[Waypoint] = []
        self.vessel_waypoints: dict[str, list[Waypoint]] = {}
        self.vessel_current_wp: dict[str, int] = {}
        self.reflex_configs: dict[str, ReflexConfig] = {}
        self._rng = random.Random(42)

        if reflex_configs is None:
            reflex_configs = [
                ReflexConfig("conservative", ReflexStrategy.CONSERVATIVE, max_speed_knots=4.0,
                           safety_margin_pct=40.0, trust_threshold=2, reaction_distance_m=200.0,
                           efficiency_weight=0.2),
                ReflexConfig("moderate", ReflexStrategy.MODERATE, max_speed_knots=6.0,
                           safety_margin_pct=20.0, trust_threshold=1, reaction_distance_m=100.0,
                           efficiency_weight=0.5),
                ReflexConfig("aggressive", ReflexStrategy.AGGRESSIVE, max_speed_knots=9.0,
                           safety_margin_pct=5.0, trust_threshold=1, reaction_distance_m=50.0,
                           efficiency_weight=0.8),
            ]
        self._default_reflex_configs = reflex_configs

    def setup(self) -> None:
        """Initialize vessels and waypoints."""
        cfg = self.config
        self.vessels.clear()
        self.events.clear()
        self.tick = 0

        # Create waypoints
        self.waypoints = []
        for _ in range(cfg.num_waypoints):
            lat = self._rng.uniform(*cfg.waypoint_area_lat)
            lon = self._rng.uniform(*cfg.waypoint_area_lon)
            self.waypoints.append(Waypoint(lat, lon))

        # Create vessels with different reflex strategies
        for i in range(cfg.num_vessels):
            vid = f"vessel_{i}"
            default_cfgs = getattr(self, '_default_reflex_configs', None) or []
            cfgs = default_cfgs if default_cfgs else []
            reflex_cfg = cfgs[i % len(cfgs)] if cfgs else ReflexConfig("default", ReflexStrategy.MODERATE)
            start_wp = self.waypoints[i % len(self.waypoints)]

            self.vessels[vid] = VesselState(
                vessel_id=vid, lat=start_wp.lat + self._rng.uniform(-0.01, 0.01),
                lon=start_wp.lon + self._rng.uniform(-0.01, 0.01),
                heading=0.0, speed=0.0,
                fuel_pct=cfg.initial_fuel_pct,
                battery_pct=100.0,
                trust_score=0.5, trust_level=1,
            )
            # Each vessel visits all waypoints in random order
            wp_order = list(range(len(self.waypoints)))
            self._rng.shuffle(wp_order)
            self.vessel_waypoints[vid] = [self.waypoints[j] for j in wp_order]
            self.vessel_current_wp[vid] = 0
            self.reflex_configs[vid] = reflex_cfg

    def step(self) -> bool:
        """Execute one simulation tick. Returns False if simulation should stop."""
        if self.tick >= self.config.max_ticks:
            return False

        self.tick += 1
        any_active = False

        for vid, vessel in self.vessels.items():
            if vessel.safety_state == VesselStatus.OFFLINE:
                continue
            if vessel.fuel_pct <= 0:
                vessel.safety_state = VesselStatus.OFFLINE
                self._add_event(vid, SimEventType.FUEL_LOW, "Out of fuel")
                continue
            if vessel.mission_complete:
                continue

            any_active = True
            reflex = self.reflex_configs[vid]

            # Navigation: head toward current waypoint
            wp_idx = self.vessel_current_wp[vid]
            if wp_idx >= len(self.vessel_waypoints[vid]):
                vessel.mission_complete = True
                self._add_event(vid, SimEventType.MISSION_COMPLETE, "All waypoints reached")
                vessel.score += 100.0  # big bonus for completion
                continue

            wp = self.vessel_waypoints[vid][wp_idx]
            bearing, distance = self._nav_calc(vessel.lat, vessel.lon, wp.lat, wp.lon)

            # Speed based on strategy and conditions
            target_speed = reflex.max_speed_knots * reflex.efficiency_weight
            # Slow down near waypoint
            if distance < 0.5:
                target_speed *= distance / 0.5
            # Slow down if low fuel
            if vessel.fuel_pct < 20:
                target_speed *= vessel.fuel_pct / 20.0
            target_speed = max(0.5, target_speed)

            vessel.speed = target_speed
            vessel.heading = bearing

            # Move vessel
            dist_per_tick = target_speed * self.config.tick_interval_sec / 3600.0  # nm per tick
            lat_rad = math.radians(vessel.lat)
            dlat = dist_per_tick * math.cos(math.radians(bearing)) / 60.0
            dlon = dist_per_tick * math.sin(math.radians(bearing)) / (60.0 * math.cos(lat_rad))
            vessel.lat += dlat
            vessel.lon += dlon
            vessel.total_distance_nm += dist_per_tick

            # Fuel consumption
            vessel.fuel_pct -= dist_per_tick * self.config.fuel_consumption_per_nm
            vessel.fuel_pct = max(0.0, vessel.fuel_pct)

            # Check waypoint arrival
            _, new_dist = self._nav_calc(vessel.lat, vessel.lon, wp.lat, wp.lon)
            if new_dist < wp.arrival_radius_nm:
                self.vessel_current_wp[vid] = wp_idx + 1
                self._add_event(vid, SimEventType.WAYPOINT_REACHED,
                              f"Reached waypoint {wp_idx} at ({wp.lat:.4f}, {wp.lon:.4f})")
                vessel.score += 10.0  # bonus for waypoint

            # Collision avoidance
            for other_id, other in self.vessels.items():
                if other_id == vid or other.safety_state == VesselStatus.OFFLINE:
                    continue
                _, sep = self._nav_calc(vessel.lat, vessel.lon, other.lat, other.lon)
                if sep < self.config.collision_radius_nm * 3:
                    self._add_event(vid, SimEventType.COLLISION_WARNING,
                                  f"Near {other_id}: {sep:.4f}nm")
                    vessel.speed *= 0.5  # slow down
                    vessel.score -= 2.0  # penalty

            # Trust evolution
            if vessel.safety_state == VesselStatus.ACTIVE:
                vessel.trust_score = min(1.0, vessel.trust_score + self.config.trust_recovery_rate)
            else:
                vessel.trust_score = max(0.0, vessel.trust_score - self.config.trust_decay_rate)
            vessel.trust_level = max(0, min(5, int(vessel.trust_score * 5)))

            # Safety event injection
            if self._rng.random() < self.config.safety_event_probability:
                vessel.safety_events += 1
                vessel.trust_score -= 0.05
                vessel.score -= 5.0
                self._add_event(vid, SimEventType.SAFETY_EVENT, "Random safety event injected")
                if vessel.trust_score < 0.2:
                    vessel.safety_state = VesselStatus.DEGRADED

            vessel.uptime_ticks = self.tick

            # Fuel low warning
            if vessel.fuel_pct < 10 and vessel.fuel_pct + dist_per_tick * self.config.fuel_consumption_per_nm >= 10:
                self._add_event(vid, SimEventType.FUEL_LOW,
                              f"Fuel low: {vessel.fuel_pct:.1f}%")

        return any_active

    def run(self) -> dict[str, Any]:
        """Run the full simulation. Returns results dict."""
        self.setup()
        self.running = True
        while self.running:
            if not self.step():
                self.running = False

        # Calculate final scores
        results = {
            "ticks": self.tick,
            "vessels": {},
            "events": len(self.events),
            "ranking": [],
        }
        for vid, vessel in self.vessels.items():
            # Fitness score: completion + efficiency + safety
            fitness = vessel.score
            fitness += vessel.total_distance_nm * 0.1  # distance bonus
            fitness -= vessel.safety_events * 3.0       # safety penalty
            if vessel.mission_complete:
                fitness += 50.0
            if vessel.fuel_pct > 0:
                fitness += vessel.fuel_pct * 0.5
            vessel.score = round(fitness, 2)
            results["vessels"][vid] = vessel.to_dict()
            results["vessels"][vid]["reflex_strategy"] = self.reflex_configs[vid].name
            results["vessels"][vid]["waypoints_reached"] = self.vessel_current_wp.get(vid, 0)
            results["vessels"][vid]["fitness"] = vessel.score

        # Ranking
        ranked = sorted(results["vessels"].items(), key=lambda x: x[1]["fitness"], reverse=True)
        results["ranking"] = [{"vessel_id": vid, "fitness": data["fitness"],
                               "strategy": data["reflex_strategy"],
                               "waypoints_reached": data["waypoints_reached"],
                               "safety_events": data["safety_events"],
                               "distance_nm": data["total_distance_nm"]}
                              for vid, data in ranked]
        results["winner"] = results["ranking"][0] if results["ranking"] else None
        return results

    def _add_event(self, vessel_id: str, event_type: SimEventType, description: str,
                   metadata: dict[str, Any] | None = None) -> None:
        self.events.append(SimEvent(self.tick, vessel_id, event_type, description, metadata or {}))

    @staticmethod
    def _nav_calc(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
        """Calculate (bearing_deg, distance_nm) between two points."""
        lat1r, lon1r = math.radians(lat1), math.radians(lon1)
        lat2r, lon2r = math.radians(lat2), math.radians(lon2)
        dlat = lat2r - lat1r
        dlon = lon2r - lon1r
        x = math.sin(dlon) * math.cos(lat2r)
        y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
        bearing = math.degrees(math.atan2(x, y)) % 360
        a = math.sin(dlat/2)**2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon/2)**2
        distance = 3440.065 * 2 * math.asin(math.sqrt(min(a, 1.0)))
        return bearing, distance
