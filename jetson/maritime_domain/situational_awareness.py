"""
Maritime Situational Awareness Module.

Builds comprehensive maritime pictures, assesses threats, computes spatial
risk maps, generates navigation warnings, updates pictures with new data,
and computes safe passage routes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .ais import AISMessage, AISDecoder
from .traffic import (
    CongestionForecast, DensityLevel, HotspotArea, TrafficAnalyzer, TrafficPattern,
)
from .weather import (
    RiskLevel, WeatherAssessor, WeatherCondition, WeatherImpact, VesselCapabilities,
)
from .zoning import MaritimeZone, ZoneType, ZoneManager


class ThreatType(Enum):
    """Types of maritime threats."""
    COLLISION = "collision"
    ZONE_VIOLATION = "zone_violation"
    WEATHER = "weather"
    GROUNDING = "grounding"
    PIRACY = "piracy"
    SAR = "search_and_rescue"
    MECHANICAL = "mechanical"
    UNKNOWN = "unknown"


class ThreatSeverity(Enum):
    """Threat severity levels."""
    INFO = "info"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"


@dataclass
class MaritimePicture:
    """Comprehensive maritime situational awareness picture."""
    own_vessel: Dict[str, Any] = field(default_factory=dict)
    contacts: List[AISMessage] = field(default_factory=list)
    zones: List[MaritimeZone] = field(default_factory=list)
    weather: Optional[WeatherCondition] = None
    traffic: List[TrafficPattern] = field(default_factory=list)
    threats: List[Dict[str, Any]] = field(default_factory=list)
    risk_areas: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class ThreatAssessment:
    """Assessment of a specific maritime threat."""
    threat_type: str
    severity: str
    probability: float          # 0.0 - 1.0
    recommended_action: str
    time_to_impact: float       # minutes, 0 = immediate


@dataclass
class SpatialRiskCell:
    """A cell in the spatial risk map."""
    position: Tuple[float, float]
    risk_score: float           # 0.0 - 10.0
    risk_factors: List[str]


@dataclass
class SafePassageRoute:
    """Recommended safe passage route."""
    waypoints: List[Tuple[float, float]]
    total_risk_score: float
    estimated_time: float       # minutes
    warnings: List[str]


class MaritimeSituationalAwareness:
    """Maintains and analyzes the complete maritime situational picture."""

    def __init__(self) -> None:
        self._ais_decoder = AISDecoder()
        self._traffic_analyzer = TrafficAnalyzer()
        self._zone_manager = ZoneManager()
        self._weather_assessor = WeatherAssessor()
        self._current_picture: Optional[MaritimePicture] = None

    def build_maritime_picture(
        self,
        own_vessel: Dict[str, Any],
        contacts: List[AISMessage],
        zones: List[MaritimeZone],
        weather: Optional[WeatherCondition],
    ) -> MaritimePicture:
        """
        Build a comprehensive maritime picture from all available data.
        Analyzes traffic patterns and integrates all information sources.
        """
        import time

        # Register zones
        for zone in zones:
            self._zone_manager.add_zone(zone)

        # Analyze traffic patterns from contacts
        traffic_patterns: List[TrafficPattern] = []
        if contacts:
            from .traffic import VesselTrack
            tracks = []
            for contact in contacts:
                if contact.position:
                    track = VesselTrack(
                        mmsi=contact.mmsi,
                        positions=[(contact.position[0], contact.position[1], contact.timestamp)],
                        speeds=[contact.speed],
                        headings=[contact.heading],
                    )
                    tracks.append(track)
            traffic_patterns = self._traffic_analyzer.detect_patterns(tracks)

        # Compute risk areas from weather
        risk_areas: List[Dict[str, Any]] = []
        if weather:
            weather_impact = self._weather_assessor.assess_impact(
                weather, "navigation", VesselCapabilities()
            )
            if weather_impact.risk_level in (RiskLevel.HIGH, RiskLevel.EXTREME):
                own_pos = own_vessel.get("position", (0.0, 0.0))
                risk_areas.append({
                    "type": "weather",
                    "center": own_pos,
                    "radius": 10.0,
                    "risk_level": weather_impact.risk_level.value,
                    "details": weather_impact.recommendations,
                })

        picture = MaritimePicture(
            own_vessel=own_vessel,
            contacts=contacts,
            zones=zones,
            weather=weather,
            traffic=traffic_patterns,
            threats=[],
            risk_areas=risk_areas,
            timestamp=time.time(),
        )

        self._current_picture = picture
        return picture

    def assess_threats(self, picture: MaritimePicture) -> List[ThreatAssessment]:
        """
        Assess all threats from the maritime picture.
        Evaluates collision risks, zone violations, weather threats, and grounding risks.
        """
        threats: List[ThreatAssessment] = []
        own_pos = picture.own_vessel.get("position", (0.0, 0.0))
        own_speed = picture.own_vessel.get("speed", 0.0)
        own_heading = picture.own_vessel.get("heading", 0.0)

        # Compute own velocity components
        own_vn = own_speed * math.cos(math.radians(own_heading))
        own_ve = own_speed * math.sin(math.radians(own_heading))

        # Collision threat assessment
        for contact in picture.contacts:
            if contact.position is None:
                continue

            cpa_dist, cpa_time = self._ais_decoder.compute_cpa(
                own_pos, (own_vn, own_ve),
                contact.position, (contact.speed * math.cos(math.radians(contact.course)),
                                   contact.speed * math.sin(math.radians(contact.course)))
            )

            if cpa_dist < 0.5:  # within 0.5 nm
                severity = ThreatSeverity.CRITICAL.value
                probability = 0.9
                action = "IMMEDIATE: Take evasive action. Vessel on collision course."
            elif cpa_dist < 1.0:
                severity = ThreatSeverity.DANGER.value
                probability = 0.7
                action = "Reduce speed and monitor closely. Prepare for evasive maneuvers."
            elif cpa_dist < 2.0:
                severity = ThreatSeverity.WARNING.value
                probability = 0.4
                action = "Monitor vessel. Maintain current course with vigilance."
            else:
                continue

            threats.append(ThreatAssessment(
                threat_type=ThreatType.COLLISION.value,
                severity=severity,
                probability=probability,
                recommended_action=action,
                time_to_impact=cpa_time,
            ))

        # Zone violation threats
        if own_pos != (0.0, 0.0):
            overlapping_zones = self._zone_manager.check_position(own_pos)
            for zone in overlapping_zones:
                if zone.zone_type == ZoneType.EXCLUSION:
                    threats.append(ThreatAssessment(
                        threat_type=ThreatType.ZONE_VIOLATION.value,
                        severity=ThreatSeverity.CRITICAL.value,
                        probability=1.0,
                        recommended_action=f"IMMEDIATE: Exit exclusion zone {zone.name}.",
                        time_to_impact=0.0,
                    ))
                elif zone.zone_type == ZoneType.CAUTION:
                    threats.append(ThreatAssessment(
                        threat_type=ThreatType.ZONE_VIOLATION.value,
                        severity=ThreatSeverity.WARNING.value,
                        probability=0.5,
                        recommended_action=f"Exercise caution in {zone.name}.",
                        time_to_impact=0.0,
                    ))

        # Weather threats
        if picture.weather:
            weather_impact = self._weather_assessor.assess_impact(
                picture.weather, "navigation", VesselCapabilities()
            )
            if weather_impact.risk_level == RiskLevel.EXTREME:
                threats.append(ThreatAssessment(
                    threat_type=ThreatType.WEATHER.value,
                    severity=ThreatSeverity.CRITICAL.value,
                    probability=0.8,
                    recommended_action="IMMEDIATE: Seek shelter. Weather conditions extreme.",
                    time_to_impact=0.0,
                ))
            elif weather_impact.risk_level == RiskLevel.HIGH:
                threats.append(ThreatAssessment(
                    threat_type=ThreatType.WEATHER.value,
                    severity=ThreatSeverity.DANGER.value,
                    probability=0.6,
                    recommended_action="Reduce speed. Consider course change for better conditions.",
                    time_to_impact=30.0,
                ))

        # Grounding risk (simplified: check if near coast/shallow water)
        own_vessel_type = picture.own_vessel.get("vessel_type", "cargo")
        own_draft = picture.own_vessel.get("draft", 5.0)
        if own_draft > 0 and picture.weather:
            wave_height = picture.weather.wave_height
            if wave_height > 3.0:
                threats.append(ThreatAssessment(
                    threat_type=ThreatType.GROUNDING.value,
                    severity=ThreatSeverity.WARNING.value,
                    probability=0.3,
                    recommended_action="High waves: maintain safe distance from shoals and coastlines.",
                    time_to_impact=60.0,
                ))

        # Sort by severity
        severity_order = {
            ThreatSeverity.CRITICAL.value: 0,
            ThreatSeverity.DANGER.value: 1,
            ThreatSeverity.WARNING.value: 2,
            ThreatSeverity.INFO.value: 3,
        }
        threats.sort(key=lambda t: severity_order.get(t.severity, 4))
        return threats

    def compute_spatial_risk(
        self,
        position: Tuple[float, float],
        threats: List[ThreatAssessment],
        zones: List[MaritimeZone],
        radius_nm: float = 5.0,
    ) -> List[SpatialRiskCell]:
        """
        Compute a spatial risk map around a position.
        Returns grid of risk cells covering the area.
        """
        import time

        grid_size = 0.5  # nm per cell
        cells: List[SpatialRiskCell] = []

        # Convert radius to approximate degrees
        lat_cells = int(radius_nm / grid_size) * 2 + 1
        lon_cells = int(radius_nm / grid_size) * 2 + 1
        lat_step = (radius_nm / 60.0) / (lat_cells / 2) if lat_cells > 0 else 0
        lon_step = (radius_nm / (60.0 * math.cos(math.radians(position[0])))) / (lon_cells / 2) if lon_cells > 0 else 0

        for i in range(-lat_cells // 2, lat_cells // 2 + 1):
            for j in range(-lon_cells // 2, lon_cells // 2 + 1):
                cell_lat = position[0] + i * lat_step
                cell_lon = position[1] + j * lon_step
                cell_pos = (cell_lat, cell_lon)

                risk_score = 0.0
                risk_factors = []

                # Threat proximity risk
                for threat in threats:
                    if threat.threat_type == ThreatType.COLLISION.value:
                        dist_factor = max(0, 1.0 - threat.time_to_impact / 60.0)
                        risk_score += threat.probability * 5.0 * dist_factor
                        risk_factors.append(f"collision_risk:{threat.probability:.1f}")

                # Zone risk
                for zone in zones:
                    if self._zone_manager._point_in_polygon(cell_pos, zone.boundary):
                        if zone.zone_type == ZoneType.EXCLUSION:
                            risk_score += 8.0
                            risk_factors.append(f"exclusion_zone:{zone.name}")
                        elif zone.zone_type == ZoneType.CAUTION:
                            risk_score += 3.0
                            risk_factors.append(f"caution_zone:{zone.name}")
                        elif zone.zone_type == ZoneType.TRAFFIC_SEPARATION:
                            risk_score += 1.0
                            risk_factors.append(f"traffic_separation:{zone.name}")

                risk_score = min(risk_score, 10.0)
                cells.append(SpatialRiskCell(
                    position=(round(cell_lat, 4), round(cell_lon, 4)),
                    risk_score=round(risk_score, 2),
                    risk_factors=risk_factors,
                ))

        cells.sort(key=lambda c: c.risk_score, reverse=True)
        return cells

    def generate_nav_warning(self, threats: List[ThreatAssessment]) -> str:
        """
        Generate a navigational warning text from current threats.
        Returns formatted warning string.
        """
        if not threats:
            return "No active navigational warnings. All clear."

        warnings = ["*** NAVIGATION WARNING ***", ""]

        critical_threats = [t for t in threats if t.severity == ThreatSeverity.CRITICAL.value]
        danger_threats = [t for t in threats if t.severity == ThreatSeverity.DANGER.value]
        warning_threats = [t for t in threats if t.severity == ThreatSeverity.WARNING.value]

        if critical_threats:
            warnings.append("CRITICAL ALERTS:")
            for t in critical_threats:
                warnings.append(
                    f"  - [{t.threat_type.upper()}] {t.recommended_action} "
                    f"(ETA: {t.time_to_impact:.0f} min, Prob: {t.probability:.0%})"
                )
            warnings.append("")

        if danger_threats:
            warnings.append("DANGER ALERTS:")
            for t in danger_threats:
                warnings.append(
                    f"  - [{t.threat_type.upper()}] {t.recommended_action} "
                    f"(ETA: {t.time_to_impact:.0f} min, Prob: {t.probability:.0%})"
                )
            warnings.append("")

        if warning_threats:
            warnings.append("WARNINGS:")
            for t in warning_threats:
                warnings.append(
                    f"  - [{t.threat_type.upper()}] {t.recommended_action} "
                    f"(ETA: {t.time_to_impact:.0f} min)"
                )

        warnings.append("")
        warnings.append(f"Total active threats: {len(threats)}")
        return "\n".join(warnings)

    def update_picture(
        self,
        picture: MaritimePicture,
        new_data: Dict[str, Any],
    ) -> MaritimePicture:
        """
        Update an existing maritime picture with new data.
        new_data can contain: 'contacts', 'weather', 'own_vessel', 'zones'
        """
        import copy
        updated = copy.deepcopy(picture)

        if "contacts" in new_data:
            new_contacts = new_data["contacts"]
            if isinstance(new_contacts, list):
                # Merge with existing, update by MMSI
                existing_by_mmsi = {c.mmsi: c for c in updated.contacts}
                for nc in new_contacts:
                    existing_by_mmsi[nc.mmsi] = nc
                updated.contacts = list(existing_by_mmsi.values())

        if "weather" in new_data:
            updated.weather = new_data["weather"]

        if "own_vessel" in new_data:
            for key, value in new_data["own_vessel"].items():
                updated.own_vessel[key] = value

        if "zones" in new_data:
            new_zones = new_data["zones"]
            if isinstance(new_zones, list):
                zone_names = {z.name: z for z in updated.zones}
                for nz in new_zones:
                    zone_names[nz.name] = nz
                updated.zones = list(zone_names.values())

        import time
        updated.timestamp = time.time()

        self._current_picture = updated
        return updated

    def compute_safe_passage(
        self,
        current_pos: Tuple[float, float],
        destination: Tuple[float, float],
        picture: MaritimePicture,
    ) -> SafePassageRoute:
        """
        Compute a recommended safe passage route considering all factors.
        Takes into account zones, traffic patterns, weather, and threats.
        """
        import time

        # Get optimal route from zone manager
        route_result = self._zone_manager.compute_optimal_route(
            current_pos, destination, picture.zones
        )

        waypoints = [(wp.lat, wp.lon) for wp in route_result.waypoints]

        # Compute risk score along route
        total_risk = 0.0
        all_warnings = []

        # Check threats for this route
        threats = self.assess_threats(picture)
        for threat in threats:
            if threat.threat_type == ThreatType.COLLISION.value:
                total_risk += threat.probability * 3.0
            elif threat.threat_type == ThreatType.WEATHER.value:
                total_risk += threat.probability * 2.0
            elif threat.threat_type == ThreatType.ZONE_VIOLATION.value:
                total_risk += threat.probability * 5.0

        # Weather impact on route
        if picture.weather:
            weather_impact = self._weather_assessor.assess_impact(
                picture.weather, "navigation", VesselCapabilities()
            )
            if weather_impact.risk_level in (RiskLevel.HIGH, RiskLevel.EXTREME):
                total_risk += 3.0
                all_warnings.extend(weather_impact.recommendations)

        # Zone avoidance warnings
        if route_result.avoids_zones:
            for zone_name in route_result.avoids_zones:
                all_warnings.append(f"Route avoids restricted zone: {zone_name}")

        # Traffic warnings
        if picture.traffic:
            for pattern in picture.traffic:
                if pattern.pattern_type == "fishing_ground":
                    all_warnings.append(f"Fishing activity detected in corridor (confidence: {pattern.confidence:.0%})")

        total_risk = min(total_risk, 10.0)
        own_speed = picture.own_vessel.get("speed", 12.0)
        est_time = route_result.estimated_time * 60 if route_result.estimated_time > 0 else 0  # to minutes

        return SafePassageRoute(
            waypoints=waypoints,
            total_risk_score=round(total_risk, 2),
            estimated_time=round(est_time, 1),
            warnings=all_warnings,
        )
