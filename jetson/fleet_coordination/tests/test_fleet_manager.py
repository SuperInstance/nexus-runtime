"""Tests for fleet_manager module — FleetManager, VesselStatus, FleetState."""

import math
import time

import pytest

from jetson.fleet_coordination.fleet_manager import (
    AnomalyRecord,
    FleetManager,
    FleetState,
    VesselHealth,
    VesselStatus,
)


# ────────────────────────────────────────────────────────────── fixtures

@pytest.fixture
def manager():
    return FleetManager()


def _make_vessel(vid="V1", **overrides):
    defaults = dict(
        vessel_id=vid,
        position=(10.0, 20.0),
        heading=90.0,
        speed=5.0,
        fuel=80.0,
        health=0.95,
        trust_score=0.9,
        available=True,
    )
    defaults.update(overrides)
    return defaults


# ────────────────────────────────────────────────────── VesselStatus

class TestVesselStatus:
    def test_default_fields(self):
        v = VesselStatus(vessel_id="X")
        assert v.vessel_id == "X"
        assert v.position == (0.0, 0.0)
        assert v.heading == 0.0
        assert v.speed == 0.0
        assert v.fuel == 100.0
        assert v.health == 1.0
        assert v.trust_score == 1.0
        assert v.available is True
        assert v.current_task is None
        assert v.metadata == {}

    def test_distance_to_self_is_zero(self):
        v = VesselStatus(vessel_id="A", position=(3.0, 4.0))
        assert v.distance_to(v) == pytest.approx(0.0)

    def test_distance_to_other(self):
        a = VesselStatus(vessel_id="A", position=(0.0, 0.0))
        b = VesselStatus(vessel_id="B", position=(3.0, 4.0))
        assert a.distance_to(b) == pytest.approx(5.0)

    def test_bearing_to_north(self):
        a = VesselStatus(vessel_id="A", position=(0.0, 0.0))
        b = VesselStatus(vessel_id="B", position=(0.0, 10.0))
        assert a.bearing_to(b) == pytest.approx(0.0)

    def test_bearing_to_east(self):
        a = VesselStatus(vessel_id="A", position=(0.0, 0.0))
        b = VesselStatus(vessel_id="B", position=(10.0, 0.0))
        assert a.bearing_to(b) == pytest.approx(90.0)

    def test_bearing_to_west(self):
        a = VesselStatus(vessel_id="A", position=(0.0, 0.0))
        b = VesselStatus(vessel_id="B", position=(-10.0, 0.0))
        assert a.bearing_to(b) == pytest.approx(270.0)


# ────────────────────────────────────────────────────── FleetState

class TestFleetState:
    def test_default_fleet_state(self):
        fs = FleetState()
        assert fs.vessels == []
        assert fs.tasks == []
        assert fs.connectivity_graph == {}
        assert fs.last_updated > 0

    def test_fleet_state_with_data(self):
        v = VesselStatus(vessel_id="X")
        fs = FleetState(vessels=[v], tasks=["t1"], connectivity_graph={"X": []})
        assert len(fs.vessels) == 1
        assert fs.tasks == ["t1"]
        assert "X" in fs.connectivity_graph


# ────────────────────────────────────────────────────── FleetManager

class TestFleetManagerRegister:
    def test_register_vessel_success(self, manager):
        info = _make_vessel("V1")
        v = manager.register_vessel(info)
        assert v.vessel_id == "V1"
        assert v.speed == 5.0

    def test_register_duplicate_raises(self, manager):
        manager.register_vessel(_make_vessel("V1"))
        with pytest.raises(ValueError, match="already registered"):
            manager.register_vessel(_make_vessel("V1"))

    def test_register_missing_id_raises(self, manager):
        with pytest.raises(ValueError, match="vessel_id is required"):
            manager.register_vessel({})

    def test_register_multiple_vessels(self, manager):
        for i in range(5):
            manager.register_vessel(_make_vessel(f"V{i}"))
        assert len(manager.get_all_vessels()) == 5

    def test_register_with_metadata(self, manager):
        info = _make_vessel("V1", metadata={"model": "ASV-200"})
        v = manager.register_vessel(info)
        assert v.metadata["model"] == "ASV-200"


class TestFleetManagerDeregister:
    def test_deregister_existing(self, manager):
        manager.register_vessel(_make_vessel("V1"))
        assert manager.deregister_vessel("V1") is True
        assert manager.get_vessel("V1") is None

    def test_deregister_nonexistent(self, manager):
        assert manager.deregister_vessel("NOPE") is False

    def test_deregister_cleans_connectivity(self, manager):
        manager.register_vessel(_make_vessel("A"))
        manager.register_vessel(_make_vessel("B"))
        manager.add_connection("A", "B")
        manager.deregister_vessel("B")
        snap = manager.get_fleet_snapshot()
        assert "B" not in snap.connectivity_graph
        assert "B" not in snap.connectivity_graph.get("A", [])


class TestFleetManagerUpdate:
    def test_update_position(self, manager):
        manager.register_vessel(_make_vessel("V1", position=(0, 0)))
        assert manager.update_vessel_status("V1", {"position": (5.0, 10.0)}) is True
        v = manager.get_vessel("V1")
        assert v.position == (5.0, 10.0)

    def test_update_fuel_clamps(self, manager):
        manager.register_vessel(_make_vessel("V1"))
        manager.update_vessel_status("V1", {"fuel": 150.0})
        assert manager.get_vessel("V1").fuel == 100.0
        manager.update_vessel_status("V1", {"fuel": -10.0})
        assert manager.get_vessel("V1").fuel == 0.0

    def test_update_health_clamps(self, manager):
        manager.register_vessel(_make_vessel("V1"))
        manager.update_vessel_status("V1", {"health": 2.0})
        assert manager.get_vessel("V1").health == 1.0
        manager.update_vessel_status("V1", {"health": -0.5})
        assert manager.get_vessel("V1").health == 0.0

    def test_update_nonexistent(self, manager):
        assert manager.update_vessel_status("NOPE", {"speed": 10}) is False

    def test_update_multiple_fields(self, manager):
        manager.register_vessel(_make_vessel("V1"))
        manager.update_vessel_status("V1", {
            "speed": 12.0, "heading": 180.0, "fuel": 50.0
        })
        v = manager.get_vessel("V1")
        assert v.speed == 12.0
        assert v.heading == 180.0
        assert v.fuel == 50.0

    def test_update_trust_clamps(self, manager):
        manager.register_vessel(_make_vessel("V1"))
        manager.update_vessel_status("V1", {"trust_score": 5.0})
        assert manager.get_vessel("V1").trust_score == 1.0

    def test_update_metadata_merges(self, manager):
        manager.register_vessel(_make_vessel("V1", metadata={"a": 1}))
        manager.update_vessel_status("V1", {"metadata": {"b": 2}})
        v = manager.get_vessel("V1")
        assert v.metadata == {"a": 1, "b": 2}


class TestFleetManagerQuery:
    def test_get_vessel_found(self, manager):
        manager.register_vessel(_make_vessel("V1"))
        assert manager.get_vessel("V1").vessel_id == "V1"

    def test_get_vessel_not_found(self, manager):
        assert manager.get_vessel("X") is None

    def test_get_all_vessels_empty(self, manager):
        assert manager.get_all_vessels() == []

    def test_get_all_vessels(self, manager):
        for i in range(3):
            manager.register_vessel(_make_vessel(f"V{i}"))
        assert len(manager.get_all_vessels()) == 3

    def test_get_available_vessels(self, manager):
        manager.register_vessel(_make_vessel("V1", available=True))
        manager.register_vessel(_make_vessel("V2", available=False))
        avail = manager.get_available_vessels()
        assert len(avail) == 1
        assert avail[0].vessel_id == "V1"

    def test_get_available_when_none(self, manager):
        manager.register_vessel(_make_vessel("V1", available=False))
        assert manager.get_available_vessels() == []


class TestFleetManagerHealth:
    def test_compute_health_empty(self, manager):
        assert manager.compute_fleet_health() == 0.0

    def test_compute_health_perfect(self, manager):
        manager.register_vessel(_make_vessel("V1", health=1.0))
        manager.register_vessel(_make_vessel("V2", health=1.0))
        assert manager.compute_fleet_health() == pytest.approx(1.0)

    def test_compute_health_degraded(self, manager):
        manager.register_vessel(_make_vessel("V1", health=0.5))
        manager.register_vessel(_make_vessel("V2", health=0.7))
        assert manager.compute_fleet_health() == pytest.approx(0.6)

    def test_compute_health_zero(self, manager):
        manager.register_vessel(_make_vessel("V1", health=0.0))
        assert manager.compute_fleet_health() == pytest.approx(0.0)


class TestFleetManagerSnapshot:
    def test_snapshot_matches_state(self, manager):
        manager.register_vessel(_make_vessel("V1"))
        snap = manager.get_fleet_snapshot()
        assert len(snap.vessels) == 1
        assert "V1" in snap.connectivity_graph
        assert snap.last_updated > 0


class TestFleetManagerAnomalies:
    def test_no_anomalies_healthy_fleet(self, manager):
        manager.register_vessel(_make_vessel("V1"))
        anomalies = manager.detect_anomalies()
        assert len(anomalies) == 0

    def test_low_fuel_anomaly(self, manager):
        manager.register_vessel(_make_vessel("V1", fuel=5.0))
        anomalies = manager.detect_anomalies()
        types = [a.anomaly_type for a in anomalies]
        assert "low_fuel" in types

    def test_health_degradation_anomaly(self, manager):
        manager.register_vessel(_make_vessel("V1", health=0.3))
        anomalies = manager.detect_anomalies()
        types = [a.anomaly_type for a in anomalies]
        assert "health_degradation" in types

    def test_low_trust_anomaly(self, manager):
        manager.register_vessel(_make_vessel("V1", trust_score=0.1))
        anomalies = manager.detect_anomalies()
        types = [a.anomaly_type for a in anomalies]
        assert "low_trust" in types

    def test_stale_heartbeat_anomaly(self, manager):
        manager.register_vessel(_make_vessel("V1"))
        v = manager.get_vessel("V1")
        v.last_heartbeat = time.time() - 400  # 400 seconds ago
        anomalies = manager.detect_anomalies()
        types = [a.anomaly_type for a in anomalies]
        assert "stale_heartbeat" in types

    def test_proximity_anomaly(self, manager):
        manager.register_vessel(_make_vessel("V1", position=(0.0, 0.0)))
        manager.register_vessel(_make_vessel("V2", position=(10.0, 10.0)))
        anomalies = manager.detect_anomalies()
        types = [a.anomaly_type for a in anomalies]
        assert "proximity_warning" in types

    def test_anomaly_severity(self, manager):
        manager.register_vessel(_make_vessel("V1", fuel=2.0))
        anomalies = manager.detect_anomalies()
        fuel_anom = [a for a in anomalies if a.anomaly_type == "low_fuel"][0]
        assert fuel_anom.severity > 0.5

    def test_anomaly_history(self, manager):
        manager.register_vessel(_make_vessel("V1", fuel=5.0))
        manager.detect_anomalies()
        assert len(manager.get_anomaly_history()) > 0

    def test_multiple_anomalies(self, manager):
        manager.register_vessel(_make_vessel("V1", fuel=5.0, health=0.2, trust_score=0.1))
        anomalies = manager.detect_anomalies()
        assert len(anomalies) >= 3


class TestFleetManagerConnectivity:
    def test_add_connection(self, manager):
        manager.register_vessel(_make_vessel("A"))
        manager.register_vessel(_make_vessel("B"))
        assert manager.add_connection("A", "B") is True
        snap = manager.get_fleet_snapshot()
        assert "B" in snap.connectivity_graph["A"]

    def test_add_connection_nonexistent(self, manager):
        assert manager.add_connection("A", "B") is False

    def test_remove_connection(self, manager):
        manager.register_vessel(_make_vessel("A"))
        manager.register_vessel(_make_vessel("B"))
        manager.add_connection("A", "B")
        assert manager.remove_connection("A", "B") is True
        snap = manager.get_fleet_snapshot()
        assert "B" not in snap.connectivity_graph["A"]

    def test_bidirectional_connection(self, manager):
        manager.register_vessel(_make_vessel("A"))
        manager.register_vessel(_make_vessel("B"))
        manager.add_connection("A", "B")
        snap = manager.get_fleet_snapshot()
        assert "B" in snap.connectivity_graph["A"]
        assert "A" in snap.connectivity_graph["B"]
