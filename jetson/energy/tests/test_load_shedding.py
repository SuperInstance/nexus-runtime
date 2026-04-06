"""Tests for load_shedding module."""

import pytest

from jetson.energy.power_budget import PowerConsumer, PowerBudget, ConsumerPriority
from jetson.energy.load_shedding import (
    LoadPriority,
    SheddingStrategy,
    ShedAction,
    ShedReport,
    LoadShedManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _consumer(name, power, priority, can_throttle=False, min_power=0.0):
    return PowerConsumer(
        name=name, nominal_power_w=power,
        priority=priority, can_throttle=can_throttle, min_power_w=min_power,
    )


# ---------------------------------------------------------------------------
# evaluate_power_deficit tests
# ---------------------------------------------------------------------------

class TestEvaluateDeficit:
    def test_surplus(self):
        assert LoadShedManager.evaluate_power_deficit(200.0, 150.0) == 0.0

    def test_exact_balance(self):
        assert LoadShedManager.evaluate_power_deficit(100.0, 100.0) == 0.0

    def test_deficit(self):
        deficit = LoadShedManager.evaluate_power_deficit(100.0, 150.0)
        assert deficit == 50.0

    def test_zero_available(self):
        assert LoadShedManager.evaluate_power_deficit(0.0, 50.0) == 50.0


# ---------------------------------------------------------------------------
# select_loads_to_shed tests
# ---------------------------------------------------------------------------

class TestSelectLoads:
    def _consumers(self):
        return [
            _consumer("propulsion", 100, ConsumerPriority.HIGH),
            _consumer("sonar", 50, ConsumerPriority.HIGH),
            _consumer("camera", 30, ConsumerPriority.MEDIUM),
            _consumer("lights", 20, ConsumerPriority.LOW),
            _consumer("satcom", 10, ConsumerPriority.LOW),
            _consumer("extra_sensor", 5, ConsumerPriority.LOW),
        ]

    def test_shed_lowest_priority_first(self):
        consumers = self._consumers()
        # Deficit of 30 W → shed LOW loads first
        actions = LoadShedManager.select_loads_to_shed(consumers, 30.0)
        names = [a.consumer_name for a in actions]
        # LOW loads: lights (20W) + satcom (10W) = 30W
        assert "propulsion" not in names
        assert "sonar" not in names

    def test_shed_covers_deficit(self):
        consumers = self._consumers()
        actions = LoadShedManager.select_loads_to_shed(consumers, 25.0)
        total_saved = sum(a.power_saved for a in actions)
        assert total_saved >= 25.0

    def test_shed_empty_deficit(self):
        consumers = self._consumers()
        actions = LoadShedManager.select_loads_to_shed(consumers, 0.0)
        assert actions == []

    def test_shed_large_deficit(self):
        consumers = self._consumers()
        actions = LoadShedManager.select_loads_to_shed(consumers, 500.0)
        # Should shed everything
        assert len(actions) > 0

    def test_shed_uses_custom_strategy(self):
        consumers = self._consumers()
        strategy = SheddingStrategy(
            shed_sequence=[LoadPriority.MEDIUM, LoadPriority.LOW, LoadPriority.HIGH],
        )
        actions = LoadShedManager.select_loads_to_shed(consumers, 200.0, strategy)
        priorities = [a.priority for a in actions]
        # First shed should be MEDIUM
        assert priorities[0] == LoadPriority.MEDIUM


# ---------------------------------------------------------------------------
# compute_shed_sequence tests
# ---------------------------------------------------------------------------

class TestShedSequence:
    def test_basic_sequence(self):
        consumers = [
            _consumer("propulsion", 100, ConsumerPriority.HIGH),
            _consumer("camera", 30, ConsumerPriority.LOW),
        ]
        budget = PowerBudget(total_available=80.0, reserve=0.0)
        actions = LoadShedManager.compute_shed_sequence(consumers, budget)
        total_saved = sum(a.power_saved for a in actions)
        assert total_saved > 0

    def test_no_shed_when_sufficient(self):
        consumers = [
            _consumer("light", 10, ConsumerPriority.LOW),
        ]
        budget = PowerBudget(total_available=100.0, reserve=0.0)
        actions = LoadShedManager.compute_shed_sequence(consumers, budget)
        assert actions == []


# ---------------------------------------------------------------------------
# recover_loads tests
# ---------------------------------------------------------------------------

class TestRecoverLoads:
    def test_recover_when_power_available(self):
        consumers = [
            _consumer("camera", 30, ConsumerPriority.MEDIUM),
            _consumer("lights", 20, ConsumerPriority.LOW),
        ]
        shed_list = [
            ShedAction(consumer_name="camera", priority=LoadPriority.MEDIUM,
                       power_saved=30.0, impact_score=5.0),
            ShedAction(consumer_name="lights", priority=LoadPriority.LOW,
                       power_saved=20.0, impact_score=2.0),
        ]
        # 50W available → recover all
        recovered = LoadShedManager.recover_loads(consumers, 50.0, shed_list)
        assert len(recovered) == 2

    def test_recover_partial(self):
        shed_list = [
            ShedAction(consumer_name="camera", priority=LoadPriority.MEDIUM,
                       power_saved=30.0, impact_score=5.0),
            ShedAction(consumer_name="lights", priority=LoadPriority.LOW,
                       power_saved=20.0, impact_score=2.0),
        ]
        # Only 25W available → camera needs 30W (too much), lights needs 20W (fits)
        recovered = LoadShedManager.recover_loads([], 25.0, shed_list)
        assert len(recovered) == 1
        assert recovered[0].consumer_name == "lights"

    def test_recover_follows_strategy_order(self):
        shed_list = [
            ShedAction(consumer_name="low1", priority=LoadPriority.LOW,
                       power_saved=10.0, impact_score=2.0),
            ShedAction(consumer_name="high1", priority=LoadPriority.HIGH,
                       power_saved=50.0, impact_score=9.0),
        ]
        strategy = SheddingStrategy(
            recovery_order=[LoadPriority.HIGH, LoadPriority.LOW],
        )
        recovered = LoadShedManager.recover_loads([], 60.0, shed_list, strategy)
        assert recovered[0].consumer_name == "high1"

    def test_recover_empty(self):
        recovered = LoadShedManager.recover_loads([], 100.0, [])
        assert recovered == []

    def test_recover_no_power(self):
        shed_list = [
            ShedAction(consumer_name="x", priority=LoadPriority.LOW,
                       power_saved=10.0, impact_score=2.0),
        ]
        recovered = LoadShedManager.recover_loads([], 0.0, shed_list)
        assert recovered == []


# ---------------------------------------------------------------------------
# compute_impact tests
# ---------------------------------------------------------------------------

class TestComputeImpact:
    def test_critical_impact(self):
        actions = [
            ShedAction(consumer_name="propulsion", priority=LoadPriority.CRITICAL,
                       power_saved=100.0, impact_score=9.5),
        ]
        impact = LoadShedManager.compute_impact(actions)
        assert "CRITICAL" in impact["propulsion"]

    def test_high_impact(self):
        actions = [
            ShedAction(consumer_name="sonar", priority=LoadPriority.HIGH,
                       power_saved=50.0, impact_score=7.0),
        ]
        impact = LoadShedManager.compute_impact(actions)
        assert "HIGH" in impact["sonar"]

    def test_medium_impact(self):
        actions = [
            ShedAction(consumer_name="camera", priority=LoadPriority.MEDIUM,
                       power_saved=30.0, impact_score=4.0),
        ]
        impact = LoadShedManager.compute_impact(actions)
        assert "MEDIUM" in impact["camera"]

    def test_low_impact(self):
        actions = [
            ShedAction(consumer_name="lights", priority=LoadPriority.LOW,
                       power_saved=20.0, impact_score=2.0),
        ]
        impact = LoadShedManager.compute_impact(actions)
        assert "LOW" in impact["lights"]

    def test_throttleable_reduced_impact(self):
        consumers = [
            _consumer("camera", 30, ConsumerPriority.MEDIUM, can_throttle=True),
        ]
        deficit = LoadShedManager.evaluate_power_deficit(50.0, 80.0)
        actions = LoadShedManager.select_loads_to_shed(consumers, deficit)
        if actions:
            impact = LoadShedManager.compute_impact(actions)
            # Throttleable should have lower impact
            assert actions[0].impact_score < 5.0

    def test_empty_impact(self):
        assert LoadShedManager.compute_impact([]) == {}


# ---------------------------------------------------------------------------
# generate_shed_report tests
# ---------------------------------------------------------------------------

class TestShedReport:
    def test_report_structure(self):
        consumers = [
            _consumer("motor", 100, ConsumerPriority.HIGH),
            _consumer("lights", 20, ConsumerPriority.LOW),
        ]
        shed_list = [
            ShedAction(consumer_name="lights", priority=LoadPriority.LOW,
                       power_saved=20.0, impact_score=2.0),
        ]
        report = LoadShedManager.generate_shed_report(consumers, shed_list)
        assert isinstance(report, ShedReport)
        assert report.total_power_saved == 20.0
        assert len(report.loads_shed) == 1
        assert "lights" in report.operational_impact

    def test_report_total_consumed(self):
        consumers = [
            _consumer("a", 50, ConsumerPriority.HIGH),
            _consumer("b", 30, ConsumerPriority.LOW),
        ]
        shed_list = [
            ShedAction(consumer_name="b", priority=LoadPriority.LOW,
                       power_saved=30.0, impact_score=2.0),
        ]
        report = LoadShedManager.generate_shed_report(consumers, shed_list)
        assert report.deficit_watts == 80.0

    def test_report_no_shed(self):
        consumers = [_consumer("a", 10, ConsumerPriority.LOW)]
        report = LoadShedManager.generate_shed_report(consumers, [])
        assert report.total_power_saved == 0.0
        assert report.loads_shed == []


# ---------------------------------------------------------------------------
# SheddingStrategy tests
# ---------------------------------------------------------------------------

class TestSheddingStrategy:
    def test_default_strategy(self):
        s = SheddingStrategy()
        assert LoadPriority.SHEDDABLE in s.priority_thresholds
        assert len(s.shed_sequence) == 4
        assert len(s.recovery_order) == 4

    def test_custom_strategy(self):
        s = SheddingStrategy(
            shed_sequence=[LoadPriority.LOW],
            recovery_order=[LoadPriority.LOW],
        )
        assert len(s.shed_sequence) == 1
        assert s.shed_sequence[0] == LoadPriority.LOW


# ---------------------------------------------------------------------------
# LoadPriority tests
# ---------------------------------------------------------------------------

class TestLoadPriority:
    def test_ordering(self):
        assert LoadPriority.CRITICAL > LoadPriority.HIGH
        assert LoadPriority.HIGH > LoadPriority.MEDIUM
        assert LoadPriority.MEDIUM > LoadPriority.LOW
        assert LoadPriority.LOW > LoadPriority.SHEDDABLE
