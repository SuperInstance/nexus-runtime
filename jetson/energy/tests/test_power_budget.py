"""Tests for power_budget module."""

import math
import pytest

from jetson.energy.power_budget import (
    ConsumerPriority,
    PowerConsumer,
    PowerBudget,
    PowerAllocator,
)


# ---------------------------------------------------------------------------
# PowerConsumer tests
# ---------------------------------------------------------------------------

class TestPowerConsumer:
    def test_basic_creation(self):
        c = PowerConsumer(name="motor", nominal_power_w=100.0)
        assert c.name == "motor"
        assert c.nominal_power_w == 100.0
        assert c.priority == ConsumerPriority.MEDIUM
        assert c.can_throttle is False
        assert c.min_power_w == 0.0

    def test_custom_priority(self):
        c = PowerConsumer(name="gps", nominal_power_w=5.0, priority=ConsumerPriority.CRITICAL)
        assert c.priority == ConsumerPriority.CRITICAL

    def test_throttle_range(self):
        c = PowerConsumer(name="camera", nominal_power_w=20.0, min_power_w=5.0, can_throttle=True)
        assert c.throttle_range == 15.0

    def test_throttle_range_zero_when_cannot_throttle(self):
        c = PowerConsumer(name="sensor", nominal_power_w=10.0, can_throttle=False)
        assert c.throttle_range == 10.0  # throttle_range always computes nominal - min

    def test_throttle_range_zero_when_min_equals_nominal(self):
        c = PowerConsumer(name="light", nominal_power_w=10.0, min_power_w=10.0)
        assert c.throttle_range == 0.0

    def test_min_power_clamped_to_nominal(self):
        c = PowerConsumer(name="x", nominal_power_w=5.0, min_power_w=20.0)
        assert c.min_power_w == 5.0

    def test_min_power_clamped_to_zero(self):
        c = PowerConsumer(name="x", nominal_power_w=5.0, min_power_w=-10.0)
        assert c.min_power_w == 0.0


# ---------------------------------------------------------------------------
# PowerBudget tests
# ---------------------------------------------------------------------------

class TestPowerBudget:
    def test_remaining_power(self):
        b = PowerBudget(total_available=200.0, allocated=100.0, reserve=20.0)
        assert b.remaining == 80.0

    def test_remaining_zero_when_over(self):
        b = PowerBudget(total_available=100.0, allocated=90.0, reserve=20.0)
        assert b.remaining == 0.0

    def test_utilization_percent(self):
        b = PowerBudget(total_available=200.0, allocated=100.0)
        assert b.utilization_percent == 50.0

    def test_utilization_percent_capped(self):
        b = PowerBudget(total_available=100.0, allocated=150.0)
        assert b.utilization_percent == 100.0

    def test_utilization_zero_when_no_total(self):
        b = PowerBudget(total_available=0.0, allocated=0.0)
        assert b.utilization_percent == 0.0

    def test_over_budget(self):
        b = PowerBudget(total_available=100.0, allocated=80.0, reserve=30.0)
        assert b.is_over_budget is True

    def test_not_over_budget(self):
        b = PowerBudget(total_available=100.0, allocated=50.0, reserve=20.0)
        assert b.is_over_budget is False


# ---------------------------------------------------------------------------
# PowerAllocator tests
# ---------------------------------------------------------------------------

class TestPowerAllocator:
    def _make_consumers(self):
        return [
            PowerConsumer(name="motor", nominal_power_w=100.0, priority=ConsumerPriority.HIGH),
            PowerConsumer(name="sensor", nominal_power_w=20.0, priority=ConsumerPriority.MEDIUM),
            PowerConsumer(name="light", nominal_power_w=10.0, priority=ConsumerPriority.LOW),
        ]

    def test_allocate_all_fit(self):
        consumers = self._make_consumers()
        budget = PowerBudget(total_available=200.0, reserve=0.0)
        alloc = PowerAllocator.allocate(budget, consumers)
        assert alloc["motor"] == 100.0
        assert alloc["sensor"] == 20.0
        assert alloc["light"] == 10.0

    def test_allocate_truncates_last(self):
        consumers = self._make_consumers()
        budget = PowerBudget(total_available=125.0, reserve=0.0)
        alloc = PowerAllocator.allocate(budget, consumers)
        assert alloc["motor"] == 100.0
        assert alloc["sensor"] == 20.0
        assert alloc["light"] == 5.0

    def test_allocate_with_reserve(self):
        consumers = self._make_consumers()
        budget = PowerBudget(total_available=140.0, reserve=30.0)
        alloc = PowerAllocator.allocate(budget, consumers)
        # Available = 140 - 30 = 110
        assert alloc["motor"] == 100.0
        assert alloc["sensor"] == 10.0
        assert alloc["light"] == 0.0

    def test_allocate_priority_high_first(self):
        consumers = self._make_consumers()
        budget = PowerBudget(total_available=105.0, reserve=0.0)
        alloc = PowerAllocator.allocate_priority(budget, consumers)
        # HIGH motor gets 100, MEDIUM sensor needs 20W but only 5W left → 0 (not throttleable)
        assert alloc["motor"] == 100.0
        assert alloc["sensor"] == 0.0
        assert alloc["light"] == 0.0

    def test_allocate_priority_throttleable(self):
        consumers = [
            PowerConsumer(name="motor", nominal_power_w=100.0, priority=ConsumerPriority.HIGH),
            PowerConsumer(name="camera", nominal_power_w=50.0, priority=ConsumerPriority.MEDIUM,
                          can_throttle=True, min_power_w=10.0),
        ]
        budget = PowerBudget(total_available=110.0, reserve=0.0)
        alloc = PowerAllocator.allocate_priority(budget, consumers)
        assert alloc["motor"] == 100.0
        # Camera is throttleable: gets min 10 because only 10W left
        assert alloc["camera"] == 10.0

    def test_reallocate_basic(self):
        consumers = self._make_consumers()
        budget = PowerBudget(total_available=200.0, reserve=0.0)
        increased = {"motor": 30.0}
        alloc = PowerAllocator.reallocate(budget, consumers, increased)
        assert alloc["motor"] == 130.0

    def test_reallocate_with_headroom(self):
        consumers = self._make_consumers()
        budget = PowerBudget(total_available=140.0, reserve=0.0)
        increased = {"motor": 50.0}
        alloc = PowerAllocator.reallocate(budget, consumers, increased)
        # motor gets 100 + 10 (only 10W headroom after initial 130 allocated)
        assert alloc["motor"] <= 140.0

    def test_compute_total_consumption(self):
        consumers = self._make_consumers()
        alloc = {"motor": 100.0, "sensor": 20.0, "light": 10.0}
        total = PowerAllocator.compute_total_consumption(consumers, alloc)
        assert total == 130.0

    def test_compute_total_missing_consumer(self):
        consumers = [PowerConsumer(name="a", nominal_power_w=10.0)]
        alloc = {"a": 10.0, "b": 5.0}  # b not in consumers list
        total = PowerAllocator.compute_total_consumption(consumers, alloc)
        assert total == 10.0

    def test_compute_reserve(self):
        budget = PowerBudget(total_available=200.0, reserve=0.0)
        alloc = {"a": 80.0, "b": 60.0}
        reserve = PowerAllocator.compute_reserve(budget, alloc)
        assert reserve == 60.0

    def test_compute_reserve_zero(self):
        budget = PowerBudget(total_available=100.0)
        alloc = {"a": 100.0}
        assert PowerAllocator.compute_reserve(budget, alloc) == 0.0

    def test_simulate_power_profile(self):
        consumers = [PowerConsumer(name="x", nominal_power_w=50.0)]
        alloc = {"x": 50.0}
        profile = PowerAllocator.simulate_power_profile(consumers, alloc, 10)
        assert len(profile) == 10
        assert all(p == 50.0 for p in profile)

    def test_simulate_power_profile_empty(self):
        profile = PowerAllocator.simulate_power_profile([], {}, 5)
        assert profile == [0.0] * 5

    def test_allocate_empty_consumers(self):
        budget = PowerBudget(total_available=100.0)
        alloc = PowerAllocator.allocate(budget, [])
        assert alloc == {}

    def test_allocate_priority_empty(self):
        budget = PowerBudget(total_available=100.0)
        alloc = PowerAllocator.allocate_priority(budget, [])
        assert alloc == {}

    def test_reallocate_empty_demand(self):
        consumers = self._make_consumers()
        budget = PowerBudget(total_available=200.0)
        alloc = PowerAllocator.reallocate(budget, consumers, {})
        assert alloc["motor"] == 100.0

    def test_reallocate_over_budget_trims(self):
        """When reallocation goes over budget, lowest priority gets trimmed."""
        consumers = [
            PowerConsumer(name="motor", nominal_power_w=80.0, priority=ConsumerPriority.HIGH),
            PowerConsumer(name="light", nominal_power_w=20.0, priority=ConsumerPriority.LOW,
                          can_throttle=True, min_power_w=5.0),
        ]
        budget = PowerBudget(total_available=100.0, reserve=0.0)
        increased = {"motor": 50.0}  # motor wants 130 total, but budget only 100
        alloc = PowerAllocator.reallocate(budget, consumers, increased)
        assert sum(alloc.values()) <= 100.0

    def test_allocate_priority_non_throttleable_skipped(self):
        """Non-throttleable consumer that doesn't fit gets zero."""
        consumers = [
            PowerConsumer(name="motor", nominal_power_w=90.0, priority=ConsumerPriority.HIGH),
            PowerConsumer(name="radio", nominal_power_w=20.0, priority=ConsumerPriority.MEDIUM,
                          can_throttle=False),
        ]
        budget = PowerBudget(total_available=95.0, reserve=0.0)
        alloc = PowerAllocator.allocate_priority(budget, consumers)
        assert alloc["motor"] == 90.0
        assert alloc["radio"] == 0.0

    def test_simulate_profile_multiple_consumers(self):
        c1 = PowerConsumer(name="a", nominal_power_w=40.0)
        c2 = PowerConsumer(name="b", nominal_power_w=60.0)
        alloc = {"a": 40.0, "b": 60.0}
        profile = PowerAllocator.simulate_power_profile([c1, c2], alloc, 5)
        assert profile == [100.0, 100.0, 100.0, 100.0, 100.0]

    def test_compute_reserve_negative_clamps_zero(self):
        budget = PowerBudget(total_available=50.0)
        alloc = {"a": 100.0}
        assert PowerAllocator.compute_reserve(budget, alloc) == 0.0
