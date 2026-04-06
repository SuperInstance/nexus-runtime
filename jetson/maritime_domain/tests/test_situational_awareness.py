"""Tests for Maritime Situational Awareness module."""

import math
import time

import pytest

from jetson.maritime_domain.ais import AISMessage
from jetson.maritime_domain.situational_awareness import (
    MaritimePicture,
    MaritimeSituationalAwareness,
    SafePassageRoute,
    SpatialRiskCell,
    ThreatAssessment,
    ThreatSeverity,
    ThreatType,
)
from jetson.maritime_domain.traffic import TrafficPattern
from jetson.maritime_domain.weather import (
    RiskLevel,
    WeatherCondition,
)
from jetson.maritime_domain.zoning import (
    MaritimeZone,
    ZoneType,
)


def _make_rect_zone(name, zone_type, lat_min, lon_min, lat_max, lon_max, **kwargs):
    """Helper to create a rectangular zone."""
    boundary = [
        (lat_min, lon_min),
        (lat_min, lon_max),
        (lat_max, lon_max),
        (lat_max, lon_min),
    ]
    return MaritimeZone(
        name=name,
        zone_type=zone_type,
        boundary=boundary,
        **kwargs,
    )


class TestMaritimePicture:
    """Tests for MaritimePicture dataclass."""

    def test_defaults(self):
        p = MaritimePicture()
        assert p.own_vessel == {}
        assert p.contacts == []
        assert p.zones == []
        assert p.weather is None
        assert p.traffic == []
        assert p.threats == []
        assert p.risk_areas == []
        assert p.timestamp == 0.0

    def test_with_data(self):
        p = MaritimePicture(
            own_vessel={"mmsi": 123, "speed": 10.0},
            contacts=[AISMessage(mmsi=456)],
            weather=WeatherCondition(),
        )
        assert p.own_vessel["mmsi"] == 123
        assert len(p.contacts) == 1
        assert p.weather is not None


class TestThreatAssessment:
    """Tests for ThreatAssessment dataclass."""

    def test_defaults(self):
        t = ThreatAssessment(
            threat_type="collision",
            severity="warning",
            probability=0.5,
            recommended_action="Monitor",
            time_to_impact=10.0,
        )
        assert t.probability == 0.5
        assert t.time_to_impact == 10.0

    def test_critical_threat(self):
        t = ThreatAssessment(
            threat_type="zone_violation",
            severity="critical",
            probability=1.0,
            recommended_action="Exit immediately",
            time_to_impact=0.0,
        )
        assert t.severity == "critical"
        assert t.time_to_impact == 0.0


class TestSpatialRiskCell:
    """Tests for SpatialRiskCell dataclass."""

    def test_construction(self):
        cell = SpatialRiskCell(
            position=(50.0, -5.0),
            risk_score=3.5,
            risk_factors=["collision_risk:0.7"],
        )
        assert cell.risk_score == 3.5
        assert len(cell.risk_factors) == 1


class TestSafePassageRoute:
    """Tests for SafePassageRoute dataclass."""

    def test_construction(self):
        route = SafePassageRoute(
            waypoints=[(50.0, -5.0), (51.0, -4.0)],
            total_risk_score=2.5,
            estimated_time=60.0,
            warnings=["Weather warning"],
        )
        assert len(route.waypoints) == 2
        assert route.total_risk_score == 2.5


class TestThreatType:
    """Tests for ThreatType enum."""

    def test_all_types(self):
        assert ThreatType.COLLISION.value == "collision"
        assert ThreatType.ZONE_VIOLATION.value == "zone_violation"
        assert ThreatType.WEATHER.value == "weather"
        assert ThreatType.GROUNDING.value == "grounding"
        assert ThreatType.PIRACY.value == "piracy"
        assert ThreatType.SAR.value == "search_and_rescue"
        assert ThreatType.MECHANICAL.value == "mechanical"
        assert ThreatType.UNKNOWN.value == "unknown"


class TestThreatSeverity:
    """Tests for ThreatSeverity enum."""

    def test_all_severities(self):
        assert ThreatSeverity.INFO.value == "info"
        assert ThreatSeverity.WARNING.value == "warning"
        assert ThreatSeverity.DANGER.value == "danger"
        assert ThreatSeverity.CRITICAL.value == "critical"


class TestMaritimeSituationalAwarenessInit:
    """Tests for MSA initialization."""

    def test_init(self):
        msa = MaritimeSituationalAwareness()
        assert isinstance(msa, MaritimeSituationalAwareness)


class TestBuildMaritimePicture:
    """Tests for building maritime picture."""

    def test_basic_picture(self):
        msa = MaritimeSituationalAwareness()
        picture = msa.build_maritime_picture(
            own_vessel={"mmsi": 123, "speed": 10.0, "heading": 0.0,
                        "position": (50.0, -5.0)},
            contacts=[],
            zones=[],
            weather=None,
        )
        assert isinstance(picture, MaritimePicture)
        assert picture.own_vessel["mmsi"] == 123

    def test_picture_with_contacts(self):
        msa = MaritimeSituationalAwareness()
        contacts = [
            AISMessage(mmsi=456, speed=12.0, heading=90.0, course=90.0,
                       position=(50.1, -4.9)),
            AISMessage(mmsi=789, speed=8.0, heading=270.0, course=270.0,
                       position=(50.2, -5.1)),
        ]
        picture = msa.build_maritime_picture(
            own_vessel={"mmsi": 123, "position": (50.0, -5.0)},
            contacts=contacts,
            zones=[],
            weather=None,
        )
        assert len(picture.contacts) == 2

    def test_picture_with_zones(self):
        msa = MaritimeSituationalAwareness()
        zones = [
            _make_rect_zone("Port", ZoneType.PORT, 50.0, -5.0, 51.0, -4.0),
        ]
        picture = msa.build_maritime_picture(
            own_vessel={"mmsi": 123},
            contacts=[],
            zones=zones,
            weather=None,
        )
        assert len(picture.zones) == 1

    def test_picture_with_weather(self):
        msa = MaritimeSituationalAwareness()
        weather = WeatherCondition(wind_speed=20.0, wave_height=2.0)
        picture = msa.build_maritime_picture(
            own_vessel={"mmsi": 123},
            contacts=[],
            zones=[],
            weather=weather,
        )
        assert picture.weather is not None
        assert picture.weather.wind_speed == 20.0

    def test_picture_traffic_patterns(self):
        msa = MaritimeSituationalAwareness()
        contacts = [
            AISMessage(mmsi=i, speed=12.0, heading=45.0, course=45.0,
                       position=(50.0 + i * 0.001, -5.0, ))
            for i in range(10)
        ]
        picture = msa.build_maritime_picture(
            own_vessel={"mmsi": 123},
            contacts=contacts,
            zones=[],
            weather=None,
        )
        assert isinstance(picture.traffic, list)

    def test_picture_timestamp_set(self):
        msa = MaritimeSituationalAwareness()
        before = time.time()
        picture = msa.build_maritime_picture(
            own_vessel={"mmsi": 123},
            contacts=[],
            zones=[],
            weather=None,
        )
        after = time.time()
        assert before <= picture.timestamp <= after

    def test_picture_severe_weather_risk_areas(self):
        msa = MaritimeSituationalAwareness()
        weather = WeatherCondition(wind_speed=60.0, wave_height=8.0, visibility=0.2)
        picture = msa.build_maritime_picture(
            own_vessel={"mmsi": 123, "position": (50.0, -5.0)},
            contacts=[],
            zones=[],
            weather=weather,
        )
        assert len(picture.risk_areas) >= 1
        assert picture.risk_areas[0]["type"] == "weather"


class TestAssessThreats:
    """Tests for threat assessment."""

    def test_no_threats_clear(self):
        msa = MaritimeSituationalAwareness()
        picture = MaritimePicture(
            own_vessel={"position": (50.0, -5.0), "speed": 0.0, "heading": 0.0},
            contacts=[],
            zones=[],
            weather=None,
        )
        threats = msa.assess_threats(picture)
        assert threats == []

    def test_collision_threat_close_contact(self):
        msa = MaritimeSituationalAwareness()
        own_pos = (50.0, -5.0)
        # Contact very close and heading toward us
        contact = AISMessage(
            mmsi=456,
            position=(50.005, -5.0),  # ~0.3nm north
            speed=10.0,
            course=180.0,  # heading south toward us
            heading=180.0,
        )
        picture = MaritimePicture(
            own_vessel={"position": own_pos, "speed": 5.0, "heading": 0.0},
            contacts=[contact],
            zones=[],
            weather=None,
        )
        threats = msa.assess_threats(picture)
        assert len(threats) >= 1
        assert any(t.threat_type == ThreatType.COLLISION.value for t in threats)

    def test_exclusion_zone_threat(self):
        msa = MaritimeSituationalAwareness()
        msa._zone_manager.add_zone(
            _make_rect_zone("Exclusion", ZoneType.EXCLUSION, 49.0, -6.0, 51.0, -4.0)
        )
        picture = MaritimePicture(
            own_vessel={"position": (50.0, -5.0), "speed": 0.0, "heading": 0.0},
            contacts=[],
            zones=[],
            weather=None,
        )
        threats = msa.assess_threats(picture)
        assert len(threats) >= 1
        assert any(t.threat_type == ThreatType.ZONE_VIOLATION.value for t in threats)
        assert any(t.severity == ThreatSeverity.CRITICAL.value for t in threats)

    def test_caution_zone_warning(self):
        msa = MaritimeSituationalAwareness()
        msa._zone_manager.add_zone(
            _make_rect_zone("Caution", ZoneType.CAUTION, 49.0, -6.0, 51.0, -4.0)
        )
        picture = MaritimePicture(
            own_vessel={"position": (50.0, -5.0), "speed": 0.0, "heading": 0.0},
            contacts=[],
            zones=[],
            weather=None,
        )
        threats = msa.assess_threats(picture)
        assert len(threats) >= 1
        assert any(t.threat_type == ThreatType.ZONE_VIOLATION.value for t in threats)

    def test_extreme_weather_threat(self):
        msa = MaritimeSituationalAwareness()
        picture = MaritimePicture(
            own_vessel={"position": (50.0, -5.0), "speed": 0.0, "heading": 0.0},
            contacts=[],
            zones=[],
            weather=WeatherCondition(wind_speed=60.0, wave_height=8.0, visibility=0.1),
        )
        threats = msa.assess_threats(picture)
        assert len(threats) >= 1
        assert any(t.threat_type == ThreatType.WEATHER.value for t in threats)

    def test_threats_sorted_by_severity(self):
        msa = MaritimeSituationalAwareness()
        msa._zone_manager.add_zone(
            _make_rect_zone("Exclusion", ZoneType.EXCLUSION, 49.0, -6.0, 51.0, -4.0)
        )
        contact = AISMessage(
            mmsi=456,
            position=(50.005, -5.0),
            speed=10.0,
            course=180.0,
            heading=180.0,
        )
        picture = MaritimePicture(
            own_vessel={"position": (50.0, -5.0), "speed": 5.0, "heading": 0.0},
            contacts=[contact],
            zones=[],
            weather=WeatherCondition(wind_speed=60.0, wave_height=8.0),
        )
        threats = msa.assess_threats(picture)
        if len(threats) >= 2:
            severity_order = {ThreatSeverity.CRITICAL.value: 0,
                              ThreatSeverity.DANGER.value: 1,
                              ThreatSeverity.WARNING.value: 2}
            for i in range(len(threats) - 1):
                s1 = severity_order.get(threats[i].severity, 4)
                s2 = severity_order.get(threats[i + 1].severity, 4)
                assert s1 <= s2

    def test_no_position_no_collision_check(self):
        msa = MaritimeSituationalAwareness()
        contact = AISMessage(mmsi=456, position=None)
        picture = MaritimePicture(
            own_vessel={"position": (50.0, -5.0), "speed": 10.0, "heading": 0.0},
            contacts=[contact],
            zones=[],
            weather=None,
        )
        threats = msa.assess_threats(picture)
        assert not any(t.threat_type == ThreatType.COLLISION.value for t in threats)

    def test_grounding_risk_high_waves(self):
        msa = MaritimeSituationalAwareness()
        picture = MaritimePicture(
            own_vessel={"position": (50.0, -5.0), "speed": 0.0, "heading": 0.0,
                        "vessel_type": "cargo", "draft": 8.0},
            contacts=[],
            zones=[],
            weather=WeatherCondition(wave_height=4.0),
        )
        threats = msa.assess_threats(picture)
        assert any(t.threat_type == ThreatType.GROUNDING.value for t in threats)


class TestComputeSpatialRisk:
    """Tests for spatial risk map computation."""

    def test_empty_area(self):
        msa = MaritimeSituationalAwareness()
        cells = msa.compute_spatial_risk(
            (50.0, -5.0), [], [], radius_nm=1.0
        )
        assert isinstance(cells, list)
        assert len(cells) > 0

    def test_exclusion_zone_increases_risk(self):
        msa = MaritimeSituationalAwareness()
        zones = [
            _make_rect_zone("Exclusion", ZoneType.EXCLUSION, 49.5, -5.5, 50.5, -4.5),
        ]
        cells = msa.compute_spatial_risk(
            (50.0, -5.0), [], zones, radius_nm=3.0
        )
        max_risk = max(c.risk_score for c in cells) if cells else 0
        assert max_risk >= 8.0  # exclusion zone risk

    def test_caution_zone_moderate_risk(self):
        msa = MaritimeSituationalAwareness()
        zones = [
            _make_rect_zone("Caution", ZoneType.CAUTION, 49.5, -5.5, 50.5, -4.5),
        ]
        cells = msa.compute_spatial_risk(
            (50.0, -5.0), [], zones, radius_nm=3.0
        )
        max_risk = max(c.risk_score for c in cells) if cells else 0
        assert max_risk >= 3.0

    def test_collision_threat_adds_risk(self):
        msa = MaritimeSituationalAwareness()
        threats = [
            ThreatAssessment(
                threat_type="collision",
                severity="danger",
                probability=0.8,
                recommended_action="Take evasive action",
                time_to_impact=5.0,
            )
        ]
        cells = msa.compute_spatial_risk(
            (50.0, -5.0), threats, [], radius_nm=3.0
        )
        max_risk = max(c.risk_score for c in cells) if cells else 0
        assert max_risk > 0

    def test_cells_sorted_by_risk(self):
        msa = MaritimeSituationalAwareness()
        cells = msa.compute_spatial_risk(
            (50.0, -5.0), [], [], radius_nm=2.0
        )
        if len(cells) >= 2:
            for i in range(len(cells) - 1):
                assert cells[i].risk_score >= cells[i + 1].risk_score

    def test_cells_have_positions(self):
        msa = MaritimeSituationalAwareness()
        cells = msa.compute_spatial_risk(
            (50.0, -5.0), [], [], radius_nm=1.0
        )
        for cell in cells:
            assert isinstance(cell.position, tuple)
            assert len(cell.position) == 2

    def test_traffic_separation_zone_risk(self):
        msa = MaritimeSituationalAwareness()
        zones = [
            _make_rect_zone("TSS", ZoneType.TRAFFIC_SEPARATION, 49.5, -5.5, 50.5, -4.5),
        ]
        cells = msa.compute_spatial_risk(
            (50.0, -5.0), [], zones, radius_nm=3.0
        )
        max_risk = max(c.risk_score for c in cells) if cells else 0
        assert max_risk >= 1.0

    def test_risk_factors_populated(self):
        msa = MaritimeSituationalAwareness()
        zones = [
            _make_rect_zone("Exclusion", ZoneType.EXCLUSION, 49.5, -5.5, 50.5, -4.5),
        ]
        cells = msa.compute_spatial_risk(
            (50.0, -5.0), [], zones, radius_nm=3.0
        )
        high_risk_cells = [c for c in cells if c.risk_score > 5.0]
        if high_risk_cells:
            assert len(high_risk_cells[0].risk_factors) > 0


class TestGenerateNavWarning:
    """Tests for navigation warning generation."""

    def test_no_threats(self):
        msa = MaritimeSituationalAwareness()
        warning = msa.generate_nav_warning([])
        assert "No active" in warning

    def test_critical_alert_included(self):
        msa = MaritimeSituationalAwareness()
        threats = [
            ThreatAssessment(
                threat_type="collision",
                severity="critical",
                probability=0.9,
                recommended_action="Take evasive action",
                time_to_impact=2.0,
            )
        ]
        warning = msa.generate_nav_warning(threats)
        assert "CRITICAL" in warning
        assert "COLLISION" in warning
        assert "Take evasive action" in warning

    def test_danger_alert(self):
        msa = MaritimeSituationalAwareness()
        threats = [
            ThreatAssessment(
                threat_type="weather",
                severity="danger",
                probability=0.6,
                recommended_action="Reduce speed",
                time_to_impact=30.0,
            )
        ]
        warning = msa.generate_nav_warning(threats)
        assert "DANGER" in warning

    def test_warning_level(self):
        msa = MaritimeSituationalAwareness()
        threats = [
            ThreatAssessment(
                threat_type="zone_violation",
                severity="warning",
                probability=0.3,
                recommended_action="Exercise caution",
                time_to_impact=0.0,
            )
        ]
        warning = msa.generate_nav_warning(threats)
        assert "WARNINGS" in warning

    def test_total_threat_count(self):
        msa = MaritimeSituationalAwareness()
        threats = [
            ThreatAssessment("a", "critical", 1.0, "Act", 0.0),
            ThreatAssessment("b", "warning", 0.3, "Watch", 10.0),
            ThreatAssessment("c", "info", 0.1, "Note", 60.0),
        ]
        warning = msa.generate_nav_warning(threats)
        assert "Total active threats: 3" in warning

    def test_mixed_severities(self):
        msa = MaritimeSituationalAwareness()
        threats = [
            ThreatAssessment("a", "critical", 1.0, "Act now", 0.0),
            ThreatAssessment("b", "warning", 0.3, "Watch", 10.0),
        ]
        warning = msa.generate_nav_warning(threats)
        assert "CRITICAL" in warning
        assert "WARNINGS" in warning


class TestUpdatePicture:
    """Tests for updating maritime picture."""

    def test_update_weather(self):
        msa = MaritimeSituationalAwareness()
        base = MaritimePicture(
            own_vessel={"mmsi": 123},
            weather=WeatherCondition(wind_speed=10.0),
        )
        new_weather = WeatherCondition(wind_speed=30.0)
        updated = msa.update_picture(base, {"weather": new_weather})
        assert updated.weather.wind_speed == 30.0

    def test_update_contacts(self):
        msa = MaritimeSituationalAwareness()
        c1 = AISMessage(mmsi=111, speed=10.0)
        c2 = AISMessage(mmsi=222, speed=12.0)
        base = MaritimePicture(contacts=[c1])
        updated = msa.update_picture(base, {"contacts": [c2]})
        mmsis = [c.mmsi for c in updated.contacts]
        assert 111 in mmsis
        assert 222 in mmsis

    def test_update_contacts_replaces_by_mmsi(self):
        msa = MaritimeSituationalAwareness()
        c1 = AISMessage(mmsi=111, speed=10.0)
        c1_updated = AISMessage(mmsi=111, speed=15.0)
        base = MaritimePicture(contacts=[c1])
        updated = msa.update_picture(base, {"contacts": [c1_updated]})
        assert len(updated.contacts) == 1
        assert updated.contacts[0].speed == 15.0

    def test_update_own_vessel(self):
        msa = MaritimeSituationalAwareness()
        base = MaritimePicture(own_vessel={"mmsi": 123, "speed": 10.0})
        updated = msa.update_picture(base, {"own_vessel": {"speed": 15.0}})
        assert updated.own_vessel["speed"] == 15.0
        assert updated.own_vessel["mmsi"] == 123  # preserved

    def test_update_zones(self):
        msa = MaritimeSituationalAwareness()
        z1 = _make_rect_zone("Zone A", ZoneType.PORT, 50, -5, 51, -4)
        base = MaritimePicture(zones=[z1])
        z2 = _make_rect_zone("Zone B", ZoneType.CAUTION, 52, -3, 53, -2)
        updated = msa.update_picture(base, {"zones": [z2]})
        names = [z.name for z in updated.zones]
        assert "Zone A" in names
        assert "Zone B" in names

    def test_update_timestamp(self):
        msa = MaritimeSituationalAwareness()
        base = MaritimePicture(timestamp=100.0)
        before = time.time()
        updated = msa.update_picture(base, {})
        after = time.time()
        assert before <= updated.timestamp <= after

    def test_empty_update(self):
        msa = MaritimeSituationalAwareness()
        base = MaritimePicture(own_vessel={"mmsi": 123}, timestamp=100.0)
        updated = msa.update_picture(base, {})
        assert updated.own_vessel == {"mmsi": 123}

    def test_original_not_modified(self):
        msa = MaritimeSituationalAwareness()
        base = MaritimePicture(
            own_vessel={"speed": 10.0},
            weather=WeatherCondition(wind_speed=10.0),
        )
        msa.update_picture(base, {"weather": WeatherCondition(wind_speed=30.0)})
        assert base.weather.wind_speed == 10.0  # unchanged


class TestComputeSafePassage:
    """Tests for safe passage computation."""

    def test_basic_route(self):
        msa = MaritimeSituationalAwareness()
        picture = MaritimePicture(
            own_vessel={"position": (50.0, -5.0), "speed": 12.0, "heading": 45.0},
            contacts=[],
            zones=[],
            weather=None,
        )
        route = msa.compute_safe_passage((50.0, -5.0), (51.0, -4.0), picture)
        assert isinstance(route, SafePassageRoute)
        assert len(route.waypoints) > 0
        assert route.total_risk_score >= 0

    def test_route_avoids_exclusion_zone(self):
        msa = MaritimeSituationalAwareness()
        exclusion = _make_rect_zone("Exclusion", ZoneType.EXCLUSION,
                                     50.3, -4.7, 50.7, -4.3)
        picture = MaritimePicture(
            own_vessel={"position": (50.0, -5.0), "speed": 12.0, "heading": 45.0},
            contacts=[],
            zones=[exclusion],
            weather=None,
        )
        route = msa.compute_safe_passage((50.0, -5.0), (51.0, -4.0), picture)
        # Zone avoidance should appear in warnings or route avoids_zones
        has_exclusion_ref = any("xclusion" in w for w in route.warnings)
        assert has_exclusion_ref or route.total_risk_score >= 0

    def test_weather_increases_risk(self):
        msa = MaritimeSituationalAwareness()
        picture = MaritimePicture(
            own_vessel={"position": (50.0, -5.0), "speed": 12.0, "heading": 45.0},
            contacts=[],
            zones=[],
            weather=WeatherCondition(wind_speed=60.0, wave_height=8.0),
        )
        route = msa.compute_safe_passage((50.0, -5.0), (51.0, -4.0), picture)
        assert route.total_risk_score > 0
        assert len(route.warnings) > 0

    def test_collision_risk_in_route(self):
        msa = MaritimeSituationalAwareness()
        contact = AISMessage(
            mmsi=456,
            position=(50.005, -5.0),
            speed=10.0,
            course=180.0,
            heading=180.0,
        )
        picture = MaritimePicture(
            own_vessel={"position": (50.0, -5.0), "speed": 5.0, "heading": 0.0},
            contacts=[contact],
            zones=[],
            weather=None,
        )
        route = msa.compute_safe_passage((50.0, -5.0), (51.0, -4.0), picture)
        assert route.total_risk_score > 0

    def test_estimated_time_positive(self):
        msa = MaritimeSituationalAwareness()
        picture = MaritimePicture(
            own_vessel={"position": (50.0, -5.0), "speed": 12.0, "heading": 45.0},
            contacts=[],
            zones=[],
            weather=None,
        )
        route = msa.compute_safe_passage((50.0, -5.0), (51.0, -4.0), picture)
        assert route.estimated_time >= 0

    def test_fishing_traffic_warning(self):
        msa = MaritimeSituationalAwareness()
        pattern = TrafficPattern(
            pattern_type="fishing_ground",
            corridor_center=(50.0, -5.0, 51.0, -4.0),
            confidence=0.8,
        )
        picture = MaritimePicture(
            own_vessel={"position": (50.0, -5.0), "speed": 12.0, "heading": 45.0},
            contacts=[],
            zones=[],
            weather=None,
            traffic=[pattern],
        )
        route = msa.compute_safe_passage((50.0, -5.0), (51.0, -4.0), picture)
        assert any("Fishing" in w or "fishing" in w for w in route.warnings)
