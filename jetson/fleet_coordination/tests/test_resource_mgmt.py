"""Tests for resource_mgmt module — FleetResourceManager, FleetResource, ResourceAllocation."""

import time

import pytest

from jetson.fleet_coordination.resource_mgmt import (
    FleetResource,
    FleetResourceManager,
    ReallocationPlan,
    ResourceAllocation,
    ShortageForecast,
)


# ────────────────────────────────────────────────────── fixtures

@pytest.fixture
def manager():
    return FleetResourceManager()


@pytest.fixture
def fuel_resource():
    return FleetResource(type="fuel", total_capacity=1000.0, used_capacity=200.0)


# ────────────────────────────────────────────────────── FleetResource

class TestFleetResource:
    def test_default_values(self):
        r = FleetResource(type="test")
        assert r.total_capacity == 100.0
        assert r.used_capacity == 0.0
        assert r.available_capacity == 100.0

    def test_available_capacity(self):
        r = FleetResource(type="fuel", total_capacity=500, used_capacity=200)
        assert r.available_capacity == 300.0

    def test_available_capacity_zero_when_full(self):
        r = FleetResource(type="fuel", total_capacity=100, used_capacity=100)
        assert r.available_capacity == 0.0

    def test_available_capacity_never_negative(self):
        r = FleetResource(type="fuel", total_capacity=50, used_capacity=200)
        assert r.available_capacity == 0.0


# ────────────────────────────────────────────────────── ResourceAllocation

class TestResourceAllocation:
    def test_default_values(self):
        a = ResourceAllocation()
        assert a.resource_type == ""
        assert a.vessel_id == ""
        assert a.amount == 0.0
        assert a.priority == 0.5
        assert a.expires is None

    def test_custom_allocation(self):
        a = ResourceAllocation(
            resource_type="fuel", vessel_id="V1",
            amount=50.0, priority=0.9, expires=time.time() + 3600,
        )
        assert a.resource_type == "fuel"
        assert a.amount == 50.0

    def test_auto_id(self):
        a1 = ResourceAllocation()
        a2 = ResourceAllocation()
        assert a1.id != a2.id


# ────────────────────────────────────────────────────── Register

class TestRegisterResource:
    def test_register_new(self, manager, fuel_resource):
        r = manager.register_resource(fuel_resource)
        assert r.type == "fuel"
        assert r.total_capacity == 1000.0

    def test_register_merge(self, manager):
        r1 = FleetResource(type="fuel", total_capacity=500, used_capacity=100, vessels_sharing=["V1"])
        r2 = FleetResource(type="fuel", total_capacity=500, used_capacity=50, vessels_sharing=["V2"])
        manager.register_resource(r1)
        merged = manager.register_resource(r2)
        assert merged.total_capacity == 1000.0
        assert merged.used_capacity == 150.0
        assert "V1" in merged.vessels_sharing
        assert "V2" in merged.vessels_sharing

    def test_register_different_types(self, manager):
        manager.register_resource(FleetResource(type="fuel", total_capacity=100))
        manager.register_resource(FleetResource(type="bandwidth", total_capacity=50))
        assert len(manager.get_all_resources()) == 2


# ────────────────────────────────────────────────────── Allocate

class TestAllocate:
    def test_allocate_success(self, manager, fuel_resource):
        manager.register_resource(fuel_resource)
        alloc = manager.allocate("fuel", 100.0, "V1")
        assert alloc is not None
        assert alloc.amount == 100.0

    def test_allocate_insufficient(self, manager, fuel_resource):
        manager.register_resource(fuel_resource)
        # Only 800 available, requesting 900
        alloc = manager.allocate("fuel", 900.0, "V1")
        assert alloc is None

    def test_allocate_unknown_resource(self, manager):
        assert manager.allocate("nonexistent", 10, "V1") is None

    def test_allocate_updates_used(self, manager, fuel_resource):
        manager.register_resource(fuel_resource)
        manager.allocate("fuel", 100.0, "V1")
        res = manager.get_resource("fuel")
        assert res.used_capacity == 300.0

    def test_allocate_adds_vessel_sharing(self, manager, fuel_resource):
        manager.register_resource(fuel_resource)
        manager.allocate("fuel", 50, "V1")
        res = manager.get_resource("fuel")
        assert "V1" in res.vessels_sharing

    def test_allocate_with_priority_and_expiry(self, manager, fuel_resource):
        manager.register_resource(fuel_resource)
        exp = time.time() + 3600
        alloc = manager.allocate("fuel", 50, "V1", priority=0.9, expires=exp)
        assert alloc.priority == 0.9
        assert alloc.expires == exp


# ────────────────────────────────────────────────────── Deallocate

class TestDeallocate:
    def test_deallocate_success(self, manager, fuel_resource):
        manager.register_resource(fuel_resource)
        alloc = manager.allocate("fuel", 100, "V1")
        assert manager.deallocate(alloc.id) is True
        res = manager.get_resource("fuel")
        assert res.used_capacity == 200.0

    def test_deallocate_nonexistent(self, manager):
        assert manager.deallocate("NOPE") is False

    def test_deallocate_twice(self, manager, fuel_resource):
        manager.register_resource(fuel_resource)
        alloc = manager.allocate("fuel", 100, "V1")
        manager.deallocate(alloc.id)
        assert manager.deallocate(alloc.id) is False

    def test_deallocate_used_floor_zero(self, manager):
        r = FleetResource(type="x", total_capacity=100, used_capacity=0)
        manager.register_resource(r)
        alloc = manager.allocate("x", 50, "V1")
        assert manager.get_resource("x").used_capacity == 50
        manager.deallocate(alloc.id)
        assert manager.get_resource("x").used_capacity == 0.0


# ────────────────────────────────────────────────────── Utilization

class TestUtilization:
    def test_utilization(self, manager, fuel_resource):
        manager.register_resource(fuel_resource)
        util = manager.get_utilization("fuel")
        assert util == pytest.approx(200.0 / 1000.0)

    def test_utilization_unknown(self, manager):
        assert manager.get_utilization("nonexistent") == 0.0

    def test_utilization_zero_capacity(self, manager):
        manager.register_resource(FleetResource(type="x", total_capacity=0, used_capacity=0))
        assert manager.get_utilization("x") == 0.0


# ────────────────────────────────────────────────────── Query

class TestResourceQuery:
    def test_get_resource(self, manager, fuel_resource):
        manager.register_resource(fuel_resource)
        assert manager.get_resource("fuel").type == "fuel"

    def test_get_resource_not_found(self, manager):
        assert manager.get_resource("nope") is None

    def test_get_all_resources_empty(self, manager):
        assert manager.get_all_resources() == []

    def test_get_allocation(self, manager, fuel_resource):
        manager.register_resource(fuel_resource)
        alloc = manager.allocate("fuel", 50, "V1")
        assert manager.get_allocation(alloc.id).amount == 50

    def test_get_allocation_not_found(self, manager):
        assert manager.get_allocation("nope") is None

    def test_get_vessel_allocations(self, manager, fuel_resource):
        manager.register_resource(fuel_resource)
        a1 = manager.allocate("fuel", 50, "V1")
        a2 = manager.allocate("fuel", 30, "V1")
        allocs = manager.get_vessel_allocations("V1")
        assert len(allocs) == 2

    def test_get_vessel_allocations_empty(self, manager):
        assert manager.get_vessel_allocations("V1") == []


# ────────────────────────────────────────────────────── Forecast

class TestPredictShortage:
    def test_safe_forecast(self, manager, fuel_resource):
        manager.register_resource(fuel_resource)
        forecast = manager.predict_shortage("fuel", [10, 10, 10])
        assert forecast.shortage_risk < 0.5

    def test_unknown_resource_forecast(self, manager):
        f = manager.predict_shortage("nope", [100])
        assert f.shortage_risk == 0.0

    def test_empty_forecast(self, manager, fuel_resource):
        manager.register_resource(fuel_resource)
        f = manager.predict_shortage("fuel", [])
        assert f.shortage_risk == 0.0
        assert f.time_to_shortage is None

    def test_high_demand_forecast(self, manager):
        r = FleetResource(type="fuel", total_capacity=100, used_capacity=80)
        manager.register_resource(r)
        f = manager.predict_shortage("fuel", [30, 30, 30])
        assert f.shortage_risk > 0.5

    def test_forecast_fields(self, manager, fuel_resource):
        manager.register_resource(fuel_resource)
        f = manager.predict_shortage("fuel", [100, 100])
        assert f.resource_type == "fuel"
        assert 0.0 <= f.current_utilization <= 1.0
        assert 0.0 <= f.projected_utilization <= 1.0


# ────────────────────────────────────────────────────── Rebalance

class TestRebalance:
    def test_basic_rebalance(self, manager):
        resources = {"fuel": 200, "bandwidth": 50}
        demand = {"fuel": 100, "bandwidth": 100}
        plan = manager.rebalance_resources(resources, demand)
        assert isinstance(plan, ReallocationPlan)
        assert plan.total_amount > 0

    def test_no_surplus(self, manager):
        resources = {"fuel": 100}
        demand = {"fuel": 200}
        plan = manager.rebalance_resources(resources, demand)
        # No surplus to transfer from fuel, no deficit for other
        assert len(plan.transfers) == 0

    def test_no_demand(self, manager):
        plan = manager.rebalance_resources({"fuel": 200}, {"fuel": 150})
        assert len(plan.transfers) == 0

    def test_empty_inputs(self, manager):
        plan = manager.rebalance_resources({}, {})
        assert plan.total_amount == 0


# ────────────────────────────────────────────────────── OPEX

class TestComputeOpex:
    def test_opex_fuel(self, manager):
        r = FleetResource(type="fuel", total_capacity=500, used_capacity=200)
        cost = manager.compute_opex([r])
        # fuel unit cost = 2.5; 200*2.5 + 300*2.5*0.1 = 500 + 75 = 575
        assert cost == pytest.approx(575.0)

    def test_opex_bandwidth(self, manager):
        r = FleetResource(type="bandwidth", total_capacity=100, used_capacity=50)
        cost = manager.compute_opex([r])
        # 50*1.0 + 50*1.0*0.1 = 50 + 5 = 55
        assert cost == pytest.approx(55.0)

    def test_opex_compute(self, manager):
        r = FleetResource(type="compute", total_capacity=200, used_capacity=100)
        cost = manager.compute_opex([r])
        # 100*3.0 + 100*3.0*0.1 = 300 + 30 = 330
        assert cost == pytest.approx(330.0)

    def test_opex_unknown_type(self, manager):
        r = FleetResource(type="unknown", total_capacity=100, used_capacity=50)
        cost = manager.compute_opex([r])
        # default cost = 1.0; 50*1 + 50*0.1 = 55
        assert cost == pytest.approx(55.0)

    def test_opex_multiple_resources(self, manager):
        r1 = FleetResource(type="fuel", total_capacity=100, used_capacity=50)
        r2 = FleetResource(type="compute", total_capacity=100, used_capacity=50)
        cost = manager.compute_opex([r1, r2])
        expected = (50 * 2.5 + 50 * 2.5 * 0.1) + (50 * 3.0 + 50 * 3.0 * 0.1)
        assert cost == pytest.approx(expected)

    def test_opex_empty(self, manager):
        assert manager.compute_opex([]) == 0.0
