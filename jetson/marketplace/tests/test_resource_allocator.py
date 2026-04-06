"""Tests for resource_allocator module."""

import pytest
from datetime import datetime, timedelta

from jetson.marketplace.resource_allocator import (
    VesselCapability, ResourceRequest, AllocationResult, ResourceAllocator,
)


class TestVesselCapability:
    def test_default(self):
        v = VesselCapability()
        assert v.vessel_id == ""
        assert v.max_speed == 0.0
        assert v.sensor_types == []

    def test_custom(self):
        v = VesselCapability(
            vessel_id="v1", max_speed=15.0, endurance_hours=48.0,
            sensor_types=["sonar", "camera"],
            trust_score=0.9,
            hourly_cost=150.0,
        )
        assert v.vessel_id == "v1"
        assert v.max_speed == 15.0
        assert v.endurance_hours == 48.0
        assert "sonar" in v.sensor_types
        assert v.trust_score == 0.9
        assert v.hourly_cost == 150.0

    def test_location_default(self):
        v = VesselCapability()
        assert v.location == {"lat": 0.0, "lon": 0.0}


class TestResourceRequest:
    def test_default(self):
        r = ResourceRequest()
        assert r.task_id == ""
        assert r.requirements == {}
        assert r.time_window[0] < r.time_window[1]

    def test_custom(self):
        start = datetime.utcnow()
        end = start + timedelta(hours=12)
        r = ResourceRequest(
            task_id="t1",
            requirements={"min_speed": 10},
            location={"lat": 45.0, "lon": -70.0},
            time_window=(start, end),
        )
        assert r.task_id == "t1"
        assert r.requirements["min_speed"] == 10
        assert r.location["lat"] == 45.0


class TestAllocationResult:
    def test_default(self):
        a = AllocationResult()
        assert a.vessel_id == ""
        assert a.match_score == 0.0
        assert a.estimated_cost == 0.0

    def test_custom(self):
        a = AllocationResult(
            vessel_id="v1", task_id="t1",
            match_score=0.85, estimated_cost=600.0, estimated_duration=4.0,
        )
        assert a.vessel_id == "v1"
        assert a.match_score == 0.85


class TestResourceAllocator:
    def setup_method(self):
        self.allocator = ResourceAllocator()

    def test_match_vessels_to_task_basic(self):
        v = VesselCapability(
            vessel_id="v1", max_speed=15.0, endurance_hours=48.0,
            sensor_types=["sonar", "camera"], hourly_cost=100.0,
        )
        req = ResourceRequest(
            task_id="t1",
            requirements={"min_speed": 10, "required_sensors": ["sonar"]},
        )
        results = self.allocator.match_vessels_to_task(req, [v])
        assert len(results) == 1
        assert results[0].vessel_id == "v1"
        assert results[0].match_score > 0

    def test_match_vessels_sorted_by_score(self):
        v1 = VesselCapability(vessel_id="v1", max_speed=5.0, hourly_cost=100.0)
        v2 = VesselCapability(vessel_id="v2", max_speed=20.0, hourly_cost=100.0)
        req = ResourceRequest(requirements={"min_speed": 10})
        results = self.allocator.match_vessels_to_task(req, [v1, v2])
        assert results[0].vessel_id == "v2"

    def test_match_no_requirements(self):
        v = VesselCapability(vessel_id="v1", max_speed=15.0, hourly_cost=100.0)
        req = ResourceRequest(task_id="t1", requirements={})
        results = self.allocator.match_vessels_to_task(req, [v])
        assert results == []

    def test_match_empty_vessels(self):
        req = ResourceRequest(requirements={"min_speed": 10})
        results = self.allocator.match_vessels_to_task(req, [])
        assert results == []

    def test_match_speed_requirement_exact(self):
        v = VesselCapability(vessel_id="v1", max_speed=10.0, hourly_cost=100.0)
        req = ResourceRequest(requirements={"min_speed": 10})
        score = self.allocator.compute_match_score(req.requirements, v)
        assert score == 1.0

    def test_match_speed_requirement_exceeds(self):
        v = VesselCapability(vessel_id="v1", max_speed=15.0, hourly_cost=100.0)
        req = ResourceRequest(requirements={"min_speed": 10})
        score = self.allocator.compute_match_score(req.requirements, v)
        assert score == 1.0

    def test_match_speed_requirement_below(self):
        v = VesselCapability(vessel_id="v1", max_speed=5.0, hourly_cost=100.0)
        req = ResourceRequest(requirements={"min_speed": 10})
        score = self.allocator.compute_match_score(req.requirements, v)
        assert 0.0 < score < 1.0
        assert abs(score - 0.5) < 0.01

    def test_match_endurance(self):
        v = VesselCapability(vessel_id="v1", endurance_hours=24.0, hourly_cost=100.0)
        req = ResourceRequest(requirements={"min_endurance": 48.0})
        score = self.allocator.compute_match_score(req.requirements, v)
        assert score == 0.5

    def test_match_sensors_partial(self):
        v = VesselCapability(vessel_id="v1", sensor_types=["sonar"], hourly_cost=100.0)
        req = ResourceRequest(requirements={"required_sensors": ["sonar", "camera", "lidar"]})
        score = self.allocator.compute_match_score(req.requirements, v)
        assert abs(score - (1.0 / 3.0)) < 0.01

    def test_match_sensors_full(self):
        v = VesselCapability(vessel_id="v1", sensor_types=["sonar", "camera"], hourly_cost=100.0)
        req = ResourceRequest(requirements={"required_sensors": ["sonar", "camera"]})
        score = self.allocator.compute_match_score(req.requirements, v)
        assert score == 1.0

    def test_match_actuators(self):
        v = VesselCapability(vessel_id="v1", actuator_types=["arm"], hourly_cost=100.0)
        req = ResourceRequest(requirements={"required_actuators": ["arm", "thruster"]})
        score = self.allocator.compute_match_score(req.requirements, v)
        assert score == 0.5

    def test_match_trust_score(self):
        v = VesselCapability(vessel_id="v1", trust_score=0.7, hourly_cost=100.0)
        req = ResourceRequest(requirements={"min_trust": 0.8})
        score = self.allocator.compute_match_score(req.requirements, v)
        assert abs(score - (0.7 / 0.8)) < 0.01

    def test_match_hourly_cost_within(self):
        v = VesselCapability(vessel_id="v1", hourly_cost=80.0)
        req = ResourceRequest(requirements={"max_hourly_cost": 100.0})
        score = self.allocator.compute_match_score(req.requirements, v)
        assert score == 1.0

    def test_match_hourly_cost_exceeds(self):
        v = VesselCapability(vessel_id="v1", hourly_cost=200.0)
        req = ResourceRequest(requirements={"max_hourly_cost": 100.0})
        score = self.allocator.compute_match_score(req.requirements, v)
        assert score == 0.5

    def test_match_combined_requirements(self):
        v = VesselCapability(
            vessel_id="v1", max_speed=15.0, endurance_hours=48.0,
            sensor_types=["sonar"], trust_score=0.9, hourly_cost=80.0,
        )
        req = ResourceRequest(requirements={
            "min_speed": 10, "min_endurance": 24,
            "required_sensors": ["sonar", "camera"],
        })
        score = self.allocator.compute_match_score(req.requirements, v)
        assert 0.5 < score < 1.0  # Speed and endurance met, sensors partial

    def test_optimize_fleet_allocation(self):
        v1 = VesselCapability(vessel_id="v1", max_speed=15.0, hourly_cost=100.0, sensor_types=["sonar"])
        v2 = VesselCapability(vessel_id="v2", max_speed=10.0, hourly_cost=80.0, sensor_types=["camera"])
        t1 = ResourceRequest(task_id="t1", requirements={"min_speed": 12, "required_sensors": ["sonar"]})
        t2 = ResourceRequest(task_id="t2", requirements={"required_sensors": ["camera"]})
        results = self.allocator.optimize_fleet_allocation([t1, t2], [v1, v2])
        assert len(results) == 2
        vessel_ids = {r.vessel_id for r in results}
        assert len(vessel_ids) == 2

    def test_optimize_empty(self):
        results = self.allocator.optimize_fleet_allocation([], [])
        assert results == []

    def test_optimize_more_tasks_than_vessels(self):
        v1 = VesselCapability(vessel_id="v1", max_speed=15.0, hourly_cost=100.0)
        t1 = ResourceRequest(task_id="t1", requirements={"min_speed": 5})
        t2 = ResourceRequest(task_id="t2", requirements={"min_speed": 5})
        t3 = ResourceRequest(task_id="t3", requirements={"min_speed": 5})
        results = self.allocator.optimize_fleet_allocation([t1, t2, t3], [v1])
        assert len(results) == 1

    def test_optimize_no_matching(self):
        v1 = VesselCapability(vessel_id="v1", max_speed=5.0, hourly_cost=100.0)
        t1 = ResourceRequest(task_id="t1", requirements={"min_speed": 20})
        results = self.allocator.optimize_fleet_allocation([t1], [v1])
        assert len(results) == 1  # partial match (score 0.25) > 0

    def test_check_availability_free(self):
        start = datetime.utcnow()
        end = start + timedelta(hours=4)
        assert self.allocator.check_availability("v1", (start, end), []) is True

    def test_check_availability_no_overlap(self):
        now = datetime.utcnow()
        assignment = {"vessel_id": "v1", "start": now + timedelta(hours=8), "end": now + timedelta(hours=12)}
        window = (now, now + timedelta(hours=4))
        assert self.allocator.check_availability("v1", window, [assignment]) is True

    def test_check_availability_overlap(self):
        now = datetime.utcnow()
        assignment = {"vessel_id": "v1", "start": now + timedelta(hours=2), "end": now + timedelta(hours=6)}
        window = (now, now + timedelta(hours=4))
        assert self.allocator.check_availability("v1", window, [assignment]) is False

    def test_check_availability_different_vessel(self):
        now = datetime.utcnow()
        assignment = {"vessel_id": "v2", "start": now, "end": now + timedelta(hours=10)}
        window = (now, now + timedelta(hours=4))
        assert self.allocator.check_availability("v1", window, [assignment]) is True

    def test_check_availability_missing_dates(self):
        assignment = {"vessel_id": "v1"}
        window = (datetime.utcnow(), datetime.utcnow() + timedelta(hours=4))
        assert self.allocator.check_availability("v1", window, [assignment]) is True

    def test_compute_fleet_utilization(self):
        v1 = VesselCapability(vessel_id="v1")
        v2 = VesselCapability(vessel_id="v2")
        v3 = VesselCapability(vessel_id="v3")
        assignments = [
            {"vessel_id": "v1"},
            {"vessel_id": "v2"},
        ]
        util = self.allocator.compute_fleet_utilization(assignments, [v1, v2, v3])
        assert abs(util - (2.0 / 3.0 * 100.0)) < 0.01

    def test_compute_fleet_utilization_empty(self):
        assert self.allocator.compute_fleet_utilization([], []) == 0.0

    def test_compute_fleet_utilization_none_assigned(self):
        v1 = VesselCapability(vessel_id="v1")
        assert self.allocator.compute_fleet_utilization([], [v1]) == 0.0

    def test_compute_fleet_utilization_full(self):
        v1 = VesselCapability(vessel_id="v1")
        v2 = VesselCapability(vessel_id="v2")
        assignments = [{"vessel_id": "v1"}, {"vessel_id": "v2"}]
        assert self.allocator.compute_fleet_utilization(assignments, [v1, v2]) == 100.0

    def test_estimated_duration_from_requirements(self):
        v = VesselCapability(vessel_id="v1", max_speed=15.0, hourly_cost=100.0)
        req = ResourceRequest(requirements={"min_speed": 10, "estimated_duration": 8.0})
        results = self.allocator.match_vessels_to_task(req, [v])
        assert results[0].estimated_duration == 8.0

    def test_estimated_cost(self):
        v = VesselCapability(vessel_id="v1", max_speed=15.0, hourly_cost=100.0)
        req = ResourceRequest(requirements={"min_speed": 10, "estimated_duration": 5.0})
        results = self.allocator.match_vessels_to_task(req, [v])
        assert results[0].estimated_cost == 500.0

    def test_match_zero_speed_requirement(self):
        v = VesselCapability(vessel_id="v1", max_speed=15.0, hourly_cost=100.0)
        req = ResourceRequest(requirements={"min_speed": 0})
        score = self.allocator.compute_match_score(req.requirements, v)
        assert score == 1.0
