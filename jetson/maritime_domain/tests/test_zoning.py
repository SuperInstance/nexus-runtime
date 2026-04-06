"""Tests for Maritime Zone Management module."""

import math

import pytest

from jetson.maritime_domain.zoning import (
    MaritimeZone,
    Route,
    RoutePoint,
    VesselInfo,
    ZoneManager,
    ZoneType,
    ZoneViolation,
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


class TestMaritimeZone:
    """Tests for MaritimeZone dataclass."""

    def test_defaults(self):
        zone = MaritimeZone(
            name="Test Zone",
            zone_type=ZoneType.PORT,
            boundary=[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)],
        )
        assert zone.name == "Test Zone"
        assert zone.zone_type == ZoneType.PORT
        assert zone.restrictions == []
        assert zone.max_speed == float('inf')
        assert zone.entry_requirements == []
        assert zone.active is True

    def test_with_restrictions(self):
        zone = MaritimeZone(
            name="Restricted Port",
            zone_type=ZoneType.PORT,
            boundary=[(0, 0), (0, 1), (1, 1), (1, 0)],
            restrictions=["clearance_required", "allowed_types:small"],
            max_speed=8.0,
            entry_requirements=["pilot", "customs_clearance"],
        )
        assert zone.max_speed == 8.0
        assert len(zone.restrictions) == 2
        assert len(zone.entry_requirements) == 2


class TestZoneType:
    """Tests for ZoneType enum."""

    def test_all_types(self):
        assert ZoneType.TERRITORIAL.value == "territorial"
        assert ZoneType.PORT.value == "port"
        assert ZoneType.ANCHORAGE.value == "anchorage"
        assert ZoneType.TRAFFIC_SEPARATION.value == "traffic_separation"
        assert ZoneType.EXCLUSION.value == "exclusion"
        assert ZoneType.CAUTION.value == "caution"
        assert ZoneType.PILOTAGE.value == "pilotage"


class TestVesselInfo:
    """Tests for VesselInfo dataclass."""

    def test_defaults(self):
        v = VesselInfo(mmsi=123)
        assert v.vessel_type == "cargo"
        assert v.speed == 0.0
        assert v.has_pilot is False
        assert v.has_clearance is False

    def test_full_construction(self):
        v = VesselInfo(
            mmsi=456,
            vessel_type="tanker",
            speed=12.0,
            has_pilot=True,
            has_clearance=True,
        )
        assert v.vessel_type == "tanker"
        assert v.has_pilot is True


class TestZoneManagerInit:
    """Tests for ZoneManager initialization."""

    def test_init(self):
        zm = ZoneManager()
        assert zm.list_zones() == []


class TestAddZone:
    """Tests for zone registration."""

    def test_add_single_zone(self):
        zm = ZoneManager()
        zone = _make_rect_zone("Port A", ZoneType.PORT, 50.0, -5.0, 50.5, -4.5)
        zm.add_zone(zone)
        assert len(zm.list_zones()) == 1

    def test_add_multiple_zones(self):
        zm = ZoneManager()
        zm.add_zone(_make_rect_zone("Zone A", ZoneType.PORT, 50.0, -5.0, 50.5, -4.5))
        zm.add_zone(_make_rect_zone("Zone B", ZoneType.ANCHORAGE, 51.0, -5.0, 51.5, -4.5))
        assert len(zm.list_zones()) == 2

    def test_add_duplicate_zone(self):
        zm = ZoneManager()
        zone = _make_rect_zone("Zone X", ZoneType.PORT, 50.0, -5.0, 50.5, -4.5)
        zm.add_zone(zone)
        zm.add_zone(zone)  # overwrite
        assert len(zm.list_zones()) == 1


class TestRemoveZone:
    """Tests for zone removal."""

    def test_remove_existing(self):
        zm = ZoneManager()
        zm.add_zone(_make_rect_zone("Zone A", ZoneType.PORT, 50.0, -5.0, 50.5, -4.5))
        result = zm.remove_zone("Zone A")
        assert result is True
        assert len(zm.list_zones()) == 0

    def test_remove_nonexistent(self):
        zm = ZoneManager()
        result = zm.remove_zone("No Such Zone")
        assert result is False


class TestGetZone:
    """Tests for zone retrieval."""

    def test_get_existing(self):
        zm = ZoneManager()
        zone = _make_rect_zone("My Zone", ZoneType.CAUTION, 50.0, -5.0, 50.5, -4.5)
        zm.add_zone(zone)
        retrieved = zm.get_zone("My Zone")
        assert retrieved is not None
        assert retrieved.name == "My Zone"

    def test_get_nonexistent(self):
        zm = ZoneManager()
        assert zm.get_zone("Missing") is None


class TestCheckPosition:
    """Tests for position overlap checking."""

    def test_position_inside_zone(self):
        zm = ZoneManager()
        zm.add_zone(_make_rect_zone("Port", ZoneType.PORT, 50.0, -5.0, 51.0, -4.0))
        zones = zm.check_position((50.5, -4.5))
        assert len(zones) == 1
        assert zones[0].name == "Port"

    def test_position_outside_zone(self):
        zm = ZoneManager()
        zm.add_zone(_make_rect_zone("Port", ZoneType.PORT, 50.0, -5.0, 51.0, -4.0))
        zones = zm.check_position((0.0, 0.0))
        assert zones == []

    def test_position_in_multiple_zones(self):
        zm = ZoneManager()
        zm.add_zone(_make_rect_zone("Zone A", ZoneType.PORT, 50.0, -5.0, 51.0, -4.0))
        zm.add_zone(_make_rect_zone("Zone B", ZoneType.CAUTION, 50.0, -4.5, 51.5, -3.5))
        zones = zm.check_position((50.5, -4.2))
        assert len(zones) == 2

    def test_inactive_zone_excluded(self):
        zm = ZoneManager()
        zone = _make_rect_zone("Inactive", ZoneType.PORT, 50.0, -5.0, 51.0, -4.0)
        zone.active = False
        zm.add_zone(zone)
        zones = zm.check_position((50.5, -4.5))
        assert zones == []

    def test_empty_boundary(self):
        zm = ZoneManager()
        zm.add_zone(MaritimeZone(name="Bad", zone_type=ZoneType.PORT, boundary=[]))
        zones = zm.check_position((50.0, -5.0))
        assert zones == []

    def test_triangle_boundary(self):
        zm = ZoneManager()
        zm.add_zone(MaritimeZone(
            name="Triangle",
            zone_type=ZoneType.CAUTION,
            boundary=[(50.0, -5.0), (51.0, -5.0), (50.5, -4.0)],
        ))
        # Point inside triangle
        zones = zm.check_position((50.4, -4.7))
        assert len(zones) == 1


class TestCheckEntryPermission:
    """Tests for entry permission checking."""

    def test_exclusion_zone_denied(self):
        zm = ZoneManager()
        zone = _make_rect_zone("Exclusion", ZoneType.EXCLUSION, 50.0, -5.0, 51.0, -4.0)
        vessel = VesselInfo(mmsi=123)
        allowed, reason = zm.check_entry_permission(vessel, zone)
        assert allowed is False
        assert "prohibited" in reason.lower()

    def test_pilotage_without_pilot_denied(self):
        zm = ZoneManager()
        zone = _make_rect_zone("Pilotage", ZoneType.PILOTAGE, 50.0, -5.0, 51.0, -4.0)
        vessel = VesselInfo(mmsi=123, has_pilot=False)
        allowed, reason = zm.check_entry_permission(vessel, zone)
        assert allowed is False

    def test_pilotage_with_pilot_allowed(self):
        zm = ZoneManager()
        zone = _make_rect_zone("Pilotage", ZoneType.PILOTAGE, 50.0, -5.0, 51.0, -4.0)
        vessel = VesselInfo(mmsi=123, has_pilot=True)
        allowed, reason = zm.check_entry_permission(vessel, zone)
        assert allowed is True

    def test_clearance_required_denied(self):
        zm = ZoneManager()
        zone = _make_rect_zone("Port", ZoneType.PORT, 50.0, -5.0, 51.0, -4.0,
                               restrictions=["clearance_required"])
        vessel = VesselInfo(mmsi=123, has_clearance=False)
        allowed, reason = zm.check_entry_permission(vessel, zone)
        assert allowed is False

    def test_clearance_present_allowed(self):
        zm = ZoneManager()
        zone = _make_rect_zone("Port", ZoneType.PORT, 50.0, -5.0, 51.0, -4.0,
                               restrictions=["clearance_required"])
        vessel = VesselInfo(mmsi=123, has_clearance=True)
        allowed, reason = zm.check_entry_permission(vessel, zone)
        assert allowed is True

    def test_vessel_type_restriction(self):
        zm = ZoneManager()
        zone = _make_rect_zone("Port", ZoneType.PORT, 50.0, -5.0, 51.0, -4.0,
                               restrictions=["allowed_types:small"])
        vessel = VesselInfo(mmsi=123, vessel_type="large")
        allowed, reason = zm.check_entry_permission(vessel, zone)
        assert allowed is False

    def test_vessel_type_allowed(self):
        zm = ZoneManager()
        zone = _make_rect_zone("Port", ZoneType.PORT, 50.0, -5.0, 51.0, -4.0,
                               restrictions=["allowed_types:small"])
        vessel = VesselInfo(mmsi=123, vessel_type="small")
        allowed, reason = zm.check_entry_permission(vessel, zone)
        assert allowed is True

    def test_speed_warning(self):
        zm = ZoneManager()
        zone = _make_rect_zone("Port", ZoneType.PORT, 50.0, -5.0, 51.0, -4.0, max_speed=8.0)
        vessel = VesselInfo(mmsi=123, speed=15.0)
        allowed, reason = zm.check_entry_permission(vessel, zone)
        assert allowed is True  # allowed but with warning
        assert "speed" in reason.lower()

    def test_inactive_zone_allowed(self):
        zm = ZoneManager()
        zone = _make_rect_zone("Exclusion", ZoneType.EXCLUSION, 50.0, -5.0, 51.0, -4.0)
        zone.active = False
        vessel = VesselInfo(mmsi=123)
        allowed, reason = zm.check_entry_permission(vessel, zone)
        assert allowed is True

    def test_normal_port_allowed(self):
        zm = ZoneManager()
        zone = _make_rect_zone("Port", ZoneType.PORT, 50.0, -5.0, 51.0, -4.0)
        vessel = VesselInfo(mmsi=123, speed=5.0)
        allowed, reason = zm.check_entry_permission(vessel, zone)
        assert allowed is True


class TestComputeEntryRequirements:
    """Tests for entry requirements computation."""

    def test_no_zones_no_requirements(self):
        zm = ZoneManager()
        reqs = zm.compute_entry_requirements((50.0, -5.0), "cargo")
        assert reqs == []

    def test_pilotage_zone_requirement(self):
        zm = ZoneManager()
        zm.add_zone(_make_rect_zone("Pilot", ZoneType.PILOTAGE, 50.0, -5.0, 51.0, -4.0))
        reqs = zm.compute_entry_requirements((50.5, -4.5), "cargo")
        assert any("Pilot required" in r for r in reqs)

    def test_clearance_zone_requirement(self):
        zm = ZoneManager()
        zm.add_zone(_make_rect_zone("Port", ZoneType.PORT, 50.0, -5.0, 51.0, -4.0,
                                    restrictions=["clearance_required"]))
        reqs = zm.compute_entry_requirements((50.5, -4.5), "cargo")
        assert any("clearance" in r.lower() for r in reqs)

    def test_speed_limit_requirement(self):
        zm = ZoneManager()
        zm.add_zone(_make_rect_zone("Port", ZoneType.PORT, 50.0, -5.0, 51.0, -4.0, max_speed=6.0))
        reqs = zm.compute_entry_requirements((50.5, -4.5), "cargo")
        assert any("6.0kts" in r for r in reqs)

    def test_custom_entry_requirements(self):
        zm = ZoneManager()
        zm.add_zone(_make_rect_zone("Port", ZoneType.PORT, 50.0, -5.0, 51.0, -4.0,
                                    entry_requirements=["customs", "immigration"]))
        reqs = zm.compute_entry_requirements((50.5, -4.5), "cargo")
        assert any("customs" in r for r in reqs)
        assert any("immigration" in r for r in reqs)

    def test_multiple_zones_multiple_requirements(self):
        zm = ZoneManager()
        zm.add_zone(_make_rect_zone("Pilot", ZoneType.PILOTAGE, 50.0, -5.0, 51.0, -4.0))
        zm.add_zone(_make_rect_zone("Port", ZoneType.PORT, 50.0, -4.5, 51.5, -3.5,
                                    restrictions=["clearance_required"]))
        reqs = zm.compute_entry_requirements((50.5, -4.2), "cargo")
        assert len(reqs) >= 2


class TestComputeOptimalRoute:
    """Tests for optimal route computation."""

    def test_direct_route_no_zones(self):
        zm = ZoneManager()
        route = zm.compute_optimal_route((50.0, -5.0), (51.0, -4.0))
        assert route.total_distance > 0
        assert len(route.waypoints) > 0

    def test_route_avoids_exclusion_zone(self):
        zm = ZoneManager()
        zm.add_zone(_make_rect_zone("Exclusion", ZoneType.EXCLUSION, 50.3, -4.7, 50.7, -4.3))
        route = zm.compute_optimal_route((50.0, -5.0), (51.0, -4.0))
        assert "Exclusion" in route.avoids_zones

    def test_route_with_heading(self):
        zm = ZoneManager()
        route = zm.compute_optimal_route((50.0, -5.0), (51.0, -4.0))
        for wp in route.waypoints[:-1]:  # last waypoint has no next
            assert 0.0 <= wp.heading <= 360.0

    def test_route_estimated_time(self):
        zm = ZoneManager()
        route = zm.compute_optimal_route((50.0, -5.0), (51.0, -4.0))
        assert route.estimated_time > 0

    def test_start_equals_end(self):
        zm = ZoneManager()
        route = zm.compute_optimal_route((50.0, -5.0), (50.0, -5.0))
        assert route.total_distance == 0.0

    def test_non_exclusion_zone_not_avoided(self):
        zm = ZoneManager()
        zm.add_zone(_make_rect_zone("Port", ZoneType.PORT, 50.3, -4.7, 50.7, -4.3))
        route = zm.compute_optimal_route((50.0, -5.0), (51.0, -4.0))
        assert "Port" not in route.avoids_zones


class TestDetectZoneViolation:
    """Tests for zone violation detection."""

    def test_no_violation_outside_zones(self):
        zm = ZoneManager()
        violations = zm.detect_zone_violation((0.0, 0.0))
        assert violations == []

    def test_exclusion_zone_violation(self):
        zm = ZoneManager()
        zm.add_zone(_make_rect_zone("Exclusion", ZoneType.EXCLUSION, 50.0, -5.0, 51.0, -4.0))
        violations = zm.detect_zone_violation((50.5, -4.5))
        assert len(violations) == 1
        assert violations[0].severity == "critical"
        assert violations[0].violation_type == "entry"

    def test_type_restriction_violation(self):
        zm = ZoneManager()
        zm.add_zone(_make_rect_zone("Port", ZoneType.PORT, 50.0, -5.0, 51.0, -4.0,
                                    restrictions=["allowed_types:small"]))
        violations = zm.detect_zone_violation((50.5, -4.5), "large")
        assert len(violations) == 1
        assert violations[0].violation_type == "requirements"

    def test_no_violation_allowed_type(self):
        zm = ZoneManager()
        zm.add_zone(_make_rect_zone("Port", ZoneType.PORT, 50.0, -5.0, 51.0, -4.0,
                                    restrictions=["allowed_types:small"]))
        violations = zm.detect_zone_violation((50.5, -4.5), "small")
        assert violations == []

    def test_multiple_violations(self):
        zm = ZoneManager()
        zm.add_zone(_make_rect_zone("Exclusion", ZoneType.EXCLUSION, 50.0, -5.0, 51.0, -4.0))
        zm.add_zone(_make_rect_zone("Port", ZoneType.PORT, 50.0, -4.5, 51.5, -3.5,
                                    restrictions=["allowed_types:small"]))
        violations = zm.detect_zone_violation((50.5, -4.2), "large")
        assert len(violations) == 2


class TestPointInPolygon:
    """Tests for point-in-polygon algorithm."""

    def test_point_inside_square(self):
        assert ZoneManager._point_in_polygon(
            (0.5, 0.5),
            [(0, 0), (0, 1), (1, 1), (1, 0)]
        ) is True

    def test_point_outside_square(self):
        assert ZoneManager._point_in_polygon(
            (2.0, 2.0),
            [(0, 0), (0, 1), (1, 1), (1, 0)]
        ) is False

    def test_point_on_edge(self):
        # Edge case: point exactly on boundary
        result = ZoneManager._point_in_polygon(
            (0.0, 0.5),
            [(0, 0), (0, 1), (1, 1), (1, 0)]
        )
        assert isinstance(result, bool)


class TestHaversineDistance:
    """Tests for haversine distance computation."""

    def test_zero_distance(self):
        d = ZoneManager._haversine_distance((50.0, -5.0), (50.0, -5.0))
        assert d == 0.0

    def test_known_distance(self):
        # London to Paris is roughly 140-160 nm
        d = ZoneManager._haversine_distance((51.5, -0.12), (48.85, 2.35))
        assert 100 < d < 200  # approximate range

    def test_positive_distance(self):
        d = ZoneManager._haversine_distance((0.0, 0.0), (1.0, 1.0))
        assert d > 0
