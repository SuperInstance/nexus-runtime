"""
Tests for Fleet State Types module.
"""

import math
import time
import pytest
from jetson.agent.fleet_sync.fleet_state_types import (
    VesselPosition, SafetyAlert, ResourceLevel, TaskAssignment,
    SubsystemTrustScore, AlertSeverity, AlertType, ResourceType, TaskStatus,
)


# ==============================================================================
# VesselPosition Tests
# ==============================================================================

class TestVesselPosition:
    """Test vessel position type."""

    def test_valid_position(self):
        pos = VesselPosition("v0", 37.7749, -122.4194, 45.0, 5.5, gps_fix="3d")
        assert pos.is_valid()

    def test_latitude_bounds(self):
        pos = VesselPosition("v0", 91.0, 0.0)
        assert not pos.is_valid()
        pos2 = VesselPosition("v0", -91.0, 0.0)
        assert not pos2.is_valid()

    def test_longitude_bounds(self):
        pos = VesselPosition("v0", 0.0, 181.0)
        assert not pos.is_valid()
        pos2 = VesselPosition("v0", 0.0, -181.0)
        assert not pos2.is_valid()

    def test_negative_speed(self):
        pos = VesselPosition("v0", 0.0, 0.0, speed_knots=-1.0)
        assert not pos.is_valid()

    def test_heading_bounds(self):
        pos = VesselPosition("v0", 0.0, 0.0, heading=361.0)
        assert not pos.is_valid()

    def test_default_position(self):
        pos = VesselPosition("v0")
        assert pos.latitude == 0.0
        assert pos.longitude == 0.0
        assert pos.gps_fix == "none"

    def test_distance_to_same_point(self):
        pos = VesselPosition("v0", 37.7749, -122.4194)
        assert pos.distance_to(VesselPosition("v1", 37.7749, -122.4194)) < 0.01

    def test_distance_to_known(self):
        sf = VesselPosition("v0", 37.7749, -122.4194)
        ny = VesselPosition("v1", 40.7128, -74.0060)
        dist = sf.distance_to(ny)
        # SF to NY is approximately 2572 nautical miles
        assert 2200 < dist < 2300

    def test_bearing_to_east(self):
        pos = VesselPosition("v0", 0.0, 0.0)
        east = VesselPosition("v1", 0.0, 1.0)
        bearing = pos.bearing_to(east)
        assert 85 < bearing < 95  # approximately east (90)

    def test_bearing_to_north(self):
        pos = VesselPosition("v0", 0.0, 0.0)
        north = VesselPosition("v1", 1.0, 0.0)
        bearing = pos.bearing_to(north)
        assert 0 <= bearing < 10  # approximately north (0)

    def test_to_dict(self):
        pos = VesselPosition("v0", 37.7749, -122.4194, 45.0, 5.5, 10.0, 0.0, "3d")
        d = pos.to_dict()
        assert d["vessel_id"] == "v0"
        assert d["latitude"] == 37.7749
        assert d["gps_fix"] == "3d"

    def test_from_dict(self):
        d = {"vessel_id": "v0", "latitude": 37.7749, "longitude": -122.4194,
             "heading": 45.0, "speed_knots": 5.5, "altitude": 10.0,
             "timestamp": 1234.5, "gps_fix": "3d"}
        pos = VesselPosition.from_dict(d)
        assert pos.vessel_id == "v0"
        assert pos.latitude == 37.7749
        assert pos.gps_fix == "3d"

    def test_roundtrip_dict(self):
        pos = VesselPosition("v0", 37.7749, -122.4194, 45.0, 5.5, 10.0, 0.0, "3d")
        pos2 = VesselPosition.from_dict(pos.to_dict())
        assert pos.vessel_id == pos2.vessel_id
        assert pos.latitude == pos2.latitude
        assert pos.longitude == pos2.longitude

    def test_from_dict_defaults(self):
        pos = VesselPosition.from_dict({"vessel_id": "v0"})
        assert pos.latitude == 0.0
        assert pos.gps_fix == "none"

    def test_distance_equator(self):
        pos1 = VesselPosition("v0", 0.0, 0.0)
        pos2 = VesselPosition("v1", 0.0, 1.0)
        dist = pos1.distance_to(pos2)
        # 1 degree longitude at equator = 60 nautical miles
        assert 55 < dist < 65


# ==============================================================================
# SafetyAlert Tests
# ==============================================================================

class TestSafetyAlert:
    """Test safety alert type."""

    def test_create_alert(self):
        alert = SafetyAlert("a1", "v0", AlertType.COLLISION_RISK, AlertSeverity.EMERGENCY)
        assert alert.alert_id == "a1"
        assert not alert.resolved

    def test_priority_emergency(self):
        alert = SafetyAlert("a1", "v0", AlertType.FIRE, AlertSeverity.EMERGENCY)
        assert alert.priority == 100

    def test_priority_critical(self):
        alert = SafetyAlert("a1", "v0", AlertType.ENGINE_FAILURE, AlertSeverity.CRITICAL)
        assert alert.priority == 75

    def test_priority_warning(self):
        alert = SafetyAlert("a1", "v0", AlertType.WEATHER_WARNING, AlertSeverity.WARNING)
        assert alert.priority == 50

    def test_priority_info(self):
        alert = SafetyAlert("a1", "v0", AlertType.COMMUNICATIONS_LOSS, AlertSeverity.INFO)
        assert alert.priority == 25

    def test_severity_ordering(self):
        e = SafetyAlert("a1", "v0", AlertType.FIRE, AlertSeverity.EMERGENCY)
        c = SafetyAlert("a2", "v0", AlertType.FIRE, AlertSeverity.CRITICAL)
        w = SafetyAlert("a3", "v0", AlertType.FIRE, AlertSeverity.WARNING)
        i = SafetyAlert("a4", "v0", AlertType.FIRE, AlertSeverity.INFO)
        assert e.priority > c.priority > w.priority > i.priority

    def test_to_dict(self):
        alert = SafetyAlert("a1", "v0", AlertType.COLLISION_RISK, AlertSeverity.EMERGENCY,
                           "Collision imminent", affected_vessels=["v0", "v1"])
        d = alert.to_dict()
        assert d["alert_id"] == "a1"
        assert d["alert_type"] == "collision_risk"
        assert d["severity"] == "emergency"
        assert d["affected_vessels"] == ["v0", "v1"]

    def test_from_dict(self):
        d = {"alert_id": "a1", "vessel_id": "v0", "alert_type": "fire",
             "severity": "critical", "message": "Engine on fire"}
        alert = SafetyAlert.from_dict(d)
        assert alert.alert_type == AlertType.FIRE
        assert alert.severity == AlertSeverity.CRITICAL

    def test_roundtrip_dict(self):
        alert = SafetyAlert("a1", "v0", AlertType.FIRE, AlertSeverity.CRITICAL, "Fire!", 123.0)
        alert2 = SafetyAlert.from_dict(alert.to_dict())
        assert alert.alert_id == alert2.alert_id
        assert alert.severity == alert2.severity
        assert alert.message == alert2.message

    def test_resolved_state(self):
        alert = SafetyAlert("a1", "v0", AlertType.FIRE, AlertSeverity.CRITICAL)
        alert.resolved = True
        alert.resolved_by = "v1"
        alert.resolved_at = 1234.0
        assert alert.resolved

    def test_all_alert_types(self):
        for at in AlertType:
            alert = SafetyAlert("a1", "v0", at, AlertSeverity.WARNING)
            assert alert.alert_type.value in [e.value for e in AlertType]

    def test_all_severities(self):
        for sev in AlertSeverity:
            alert = SafetyAlert("a1", "v0", AlertType.FIRE, sev)
            assert 0 < alert.priority <= 100


# ==============================================================================
# ResourceLevel Tests
# ==============================================================================

class TestResourceLevel:
    """Test resource level type."""

    def test_create_resource(self):
        res = ResourceLevel("v0", ResourceType.FUEL, 75.0)
        assert res.current_level == 75.0
        assert res.status == "normal"

    def test_remaining_hours(self):
        res = ResourceLevel("v0", ResourceType.FUEL, 50.0, 100.0, 10.0)
        hours = res.remaining_hours
        assert 4.9 < hours < 5.1

    def test_remaining_hours_zero_consumption(self):
        res = ResourceLevel("v0", ResourceType.WATER, 80.0, consumption_rate=0.0)
        assert res.remaining_hours == float("inf")

    def test_to_dict(self):
        res = ResourceLevel("v0", ResourceType.BATTERY, 30.0, 100.0, 5.0)
        d = res.to_dict()
        assert d["resource_type"] == "battery"
        assert d["current_level"] == 30.0

    def test_from_dict(self):
        d = {"vessel_id": "v0", "resource_type": "fuel",
             "current_level": 60.0, "capacity": 200.0}
        res = ResourceLevel.from_dict(d)
        assert res.resource_type == ResourceType.FUEL
        assert res.current_level == 60.0

    def test_roundtrip_dict(self):
        res = ResourceLevel("v0", ResourceType.FOOD, 40.0, 100.0, 2.0, status="low")
        res2 = ResourceLevel.from_dict(res.to_dict())
        assert res.vessel_id == res2.vessel_id
        assert res.resource_type == res2.resource_type
        assert res.current_level == res2.current_level

    def test_from_dict_defaults(self):
        res = ResourceLevel.from_dict({"vessel_id": "v0", "resource_type": "fuel"})
        assert res.current_level == 100.0
        assert res.status == "normal"

    def test_all_resource_types(self):
        for rt in ResourceType:
            res = ResourceLevel("v0", rt, 50.0)
            assert isinstance(res.resource_type, ResourceType)


# ==============================================================================
# TaskAssignment Tests
# ==============================================================================

class TestTaskAssignment:
    """Test task assignment type."""

    def test_create_task(self):
        task = TaskAssignment("t1", "Survey reef", "v0", 3)
        assert task.task_id == "t1"
        assert task.assigned_to == "v0"
        assert task.priority == 3

    def test_default_status(self):
        task = TaskAssignment("t1")
        assert task.status == TaskStatus.PENDING

    def test_all_statuses(self):
        for status in TaskStatus:
            task = TaskAssignment("t1", status=status)
            assert isinstance(task.status, TaskStatus)

    def test_to_dict(self):
        task = TaskAssignment("t1", "Survey reef", "v0", 3,
                             dependencies=["t0"], due_at=1234.5)
        d = task.to_dict()
        assert d["task_id"] == "t1"
        assert d["status"] == "pending"
        assert d["dependencies"] == ["t0"]

    def test_from_dict(self):
        d = {"task_id": "t1", "description": "Patrol", "assigned_to": "v1",
             "priority": 2, "status": "in_progress"}
        task = TaskAssignment.from_dict(d)
        assert task.status == TaskStatus.IN_PROGRESS

    def test_roundtrip_dict(self):
        task = TaskAssignment("t1", "Patrol", "v0", 1, TaskStatus.IN_PROGRESS,
                             due_at=1234.5, estimated_duration_hours=2.5)
        task2 = TaskAssignment.from_dict(task.to_dict())
        assert task.task_id == task2.task_id
        assert task.status == task2.status
        assert task.estimated_duration_hours == task2.estimated_duration_hours


# ==============================================================================
# SubsystemTrustScore Tests
# ==============================================================================

class TestSubsystemTrustScore:
    """Test subsystem trust score type."""

    def test_create(self):
        ts = SubsystemTrustScore("v0", "navigation", 0.75, 3)
        assert ts.score == 0.75
        assert ts.autonomy_level == 3

    def test_defaults(self):
        ts = SubsystemTrustScore("v0", "engine")
        assert ts.score == 0.0
        assert ts.autonomy_level == 0

    def test_to_dict(self):
        ts = SubsystemTrustScore("v0", "steering", 0.8, 4, 1234.0, 100)
        d = ts.to_dict()
        assert d["vessel_id"] == "v0"
        assert d["subsystem"] == "steering"
        assert d["score"] == 0.8

    def test_from_dict(self):
        d = {"vessel_id": "v0", "subsystem": "navigation", "score": 0.6}
        ts = SubsystemTrustScore.from_dict(d)
        assert ts.subsystem == "navigation"

    def test_roundtrip_dict(self):
        ts = SubsystemTrustScore("v0", "payload", 0.9, 5, 1234.0, 500)
        ts2 = SubsystemTrustScore.from_dict(ts.to_dict())
        assert ts.vessel_id == ts2.vessel_id
        assert ts.score == ts2.score
        assert ts.autonomy_level == ts2.autonomy_level
