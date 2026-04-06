"""Tests for economics module."""

import pytest

from jetson.marketplace.economics import (
    CostEstimate, RevenueModel, EconomicModel, ProfitResult, ShareAllocation,
)


class TestCostEstimate:
    def test_default(self):
        c = CostEstimate()
        assert c.fuel == 0.0
        assert c.total == 0.0

    def test_custom(self):
        c = CostEstimate(fuel=200.0, labor=300.0, total=620.0)
        assert c.fuel == 200.0
        assert c.labor == 300.0
        assert c.total == 620.0


class TestRevenueModel:
    def test_default(self):
        r = RevenueModel()
        assert r.base_rate == 0.0
        assert r.time_multiplier == 1.0
        assert r.complexity_multiplier == 1.0
        assert r.risk_adjustment == 0.0

    def test_custom(self):
        r = RevenueModel(base_rate=5000.0, time_multiplier=1.2, complexity_multiplier=1.5, risk_adjustment=500.0)
        assert r.base_rate == 5000.0
        assert r.time_multiplier == 1.2


class TestEconomicModel:
    def setup_method(self):
        self.model = EconomicModel()

    def test_estimate_task_cost_basic(self):
        class FakeTask:
            estimated_duration = 4.0
        class FakeVessel:
            hourly_cost = 100.0
            value = 500000.0
            risk_score = 0.3
        cost = self.model.estimate_task_cost(FakeTask(), FakeVessel())
        assert cost.total > 0
        assert cost.fuel > 0
        assert cost.labor > 0
        assert cost.equipment_depreciation > 0
        assert cost.insurance > 0
        assert cost.contingency > 0

    def test_estimate_task_cost_with_weather(self):
        class FakeTask:
            estimated_duration = 4.0
        class FakeVessel:
            hourly_cost = 100.0
            value = 500000.0
            risk_score = 0.3
        cost_normal = self.model.estimate_task_cost(FakeTask(), FakeVessel())
        cost_storm = self.model.estimate_task_cost(FakeTask(), FakeVessel(), {"weather_factor": 2.0})
        assert cost_storm.fuel > cost_normal.fuel

    def test_estimate_task_cost_custom_labor(self):
        class FakeTask:
            estimated_duration = 2.0
        class FakeVessel:
            hourly_cost = 100.0
            value = 500000.0
            risk_score = 0.3
        cost = self.model.estimate_task_cost(FakeTask(), FakeVessel(), {"labor_rate": 200.0})
        assert cost.labor == 400.0

    def test_estimate_task_contingency_is_10_percent(self):
        class FakeTask:
            estimated_duration = 4.0
        class FakeVessel:
            hourly_cost = 100.0
            value = 500000.0
            risk_score = 0.3
        cost = self.model.estimate_task_cost(FakeTask(), FakeVessel())
        subtotal = cost.fuel + cost.labor + cost.equipment_depreciation + cost.insurance
        assert abs(cost.contingency - subtotal * 0.10) < 0.01

    def test_estimate_task_cost_total_is_subtotal_plus_contingency(self):
        class FakeTask:
            estimated_duration = 4.0
        class FakeVessel:
            hourly_cost = 100.0
            value = 500000.0
            risk_score = 0.3
        cost = self.model.estimate_task_cost(FakeTask(), FakeVessel())
        subtotal = cost.fuel + cost.labor + cost.equipment_depreciation + cost.insurance
        assert abs(cost.total - (subtotal + cost.contingency)) < 0.01

    def test_estimate_with_requirements(self):
        class FakeTask:
            requirements = {"estimated_duration": 10.0}
        class FakeVessel:
            hourly_cost = 200.0
            value = 1000000.0
            risk_score = 0.5
        cost = self.model.estimate_task_cost(FakeTask(), FakeVessel())
        assert cost.labor == 750.0  # 75.0 * 10

    def test_compute_revenue_basic(self):
        rm = RevenueModel(base_rate=5000.0, time_multiplier=1.0, complexity_multiplier=1.0)
        class FakeTask:
            reward = 3000.0
        rev = self.model.compute_revenue(FakeTask(), rm)
        assert rev == 5000.0

    def test_compute_revenue_with_multipliers(self):
        rm = RevenueModel(base_rate=5000.0, time_multiplier=1.2, complexity_multiplier=1.5)
        class FakeTask:
            reward = 3000.0
        rev = self.model.compute_revenue(FakeTask(), rm)
        assert rev == 9000.0

    def test_compute_revenue_with_risk(self):
        rm = RevenueModel(base_rate=5000.0, risk_adjustment=1000.0)
        class FakeTask:
            reward = 3000.0
        rev = self.model.compute_revenue(FakeTask(), rm)
        assert rev == 4000.0

    def test_compute_revenue_no_negative(self):
        rm = RevenueModel(base_rate=100.0, risk_adjustment=500.0)
        class FakeTask:
            reward = 0.0
        rev = self.model.compute_revenue(FakeTask(), rm)
        assert rev == 0.0

    def test_compute_revenue_zero_base_uses_task_reward(self):
        rm = RevenueModel(base_rate=0.0)
        class FakeTask:
            reward = 3000.0
        rev = self.model.compute_revenue(FakeTask(), rm)
        assert rev == 3000.0

    def test_compute_profit_positive(self):
        cost = CostEstimate(total=3000.0)
        profit = self.model.compute_profit(cost, 5000.0)
        assert profit.profit == 2000.0
        assert profit.margin == 40.0

    def test_compute_profit_negative(self):
        cost = CostEstimate(total=6000.0)
        profit = self.model.compute_profit(cost, 5000.0)
        assert profit.profit == -1000.0

    def test_compute_profit_zero_revenue(self):
        cost = CostEstimate(total=3000.0)
        profit = self.model.compute_profit(cost, 0.0)
        assert profit.profit == -3000.0
        assert profit.margin == 0.0

    def test_compute_profit_zero_cost(self):
        cost = CostEstimate(total=0.0)
        profit = self.model.compute_profit(cost, 5000.0)
        assert profit.profit == 5000.0
        assert profit.margin == 100.0

    def test_compute_profit_sharing_equal(self):
        shares = self.model.compute_profit_sharing(["a", "b", "c"], 3000.0)
        assert len(shares) == 3
        assert all(s.amount == 1000.0 for s in shares)
        assert all(abs(s.percentage - 33.33) < 0.01 for s in shares)

    def test_compute_profit_sharing_custom(self):
        shares = self.model.compute_profit_sharing(["a", "b"], 3000.0, [2.0, 1.0])
        assert shares[0].amount == 2000.0
        assert shares[1].amount == 1000.0
        assert shares[0].percentage == 66.67

    def test_compute_profit_sharing_empty(self):
        shares = self.model.compute_profit_sharing([], 1000.0)
        assert shares == []

    def test_compute_profit_sharing_zero_total(self):
        shares = self.model.compute_profit_sharing(["a", "b"], 0.0)
        assert all(s.amount == 0.0 for s in shares)

    def test_compute_profit_sharing_zero_shares(self):
        shares = self.model.compute_profit_sharing(["a", "b"], 1000.0, [0.0, 0.0])
        assert all(s.amount == 0.0 for s in shares)

    def test_compute_profit_sharing_mismatch_raises(self):
        with pytest.raises(ValueError):
            self.model.compute_profit_sharing(["a"], 1000.0, [1.0, 2.0])

    def test_compute_profit_sharing_single(self):
        shares = self.model.compute_profit_sharing(["a"], 500.0)
        assert len(shares) == 1
        assert shares[0].amount == 500.0
        assert shares[0].percentage == 100.0

    def test_compute_depreciation(self):
        dep = self.model.compute_depreciation(100000.0, 10.0, 2.0)
        assert dep == 10000.0

    def test_compute_depreciation_zero_life(self):
        dep = self.model.compute_depreciation(100000.0, 0.0, 2.0)
        assert dep == 0.0

    def test_compute_depreciation_various(self):
        dep = self.model.compute_depreciation(500000.0, 20.0, 5.0)
        assert dep == 25000.0

    def test_compute_insurance_premium(self):
        premium = self.model.compute_insurance_premium(0.3, 500000.0)
        assert premium == 500000.0 * 0.02 * 1.3

    def test_compute_insurance_zero_risk(self):
        premium = self.model.compute_insurance_premium(0.0, 500000.0)
        assert premium == 10000.0

    def test_compute_insurance_high_risk(self):
        low = self.model.compute_insurance_premium(0.1, 100000.0)
        high = self.model.compute_insurance_premium(0.9, 100000.0)
        assert high > low

    def test_compute_insurance_zero_coverage(self):
        premium = self.model.compute_insurance_premium(0.5, 0.0)
        assert premium == 0.0

    def test_cost_estimate_rounding(self):
        class FakeTask:
            estimated_duration = 3.14
        class FakeVessel:
            hourly_cost = 100.0
            value = 500000.0
            risk_score = 0.3
        cost = self.model.estimate_task_cost(FakeTask(), FakeVessel())
        assert cost.fuel == round(cost.fuel, 2)
        assert cost.total == round(cost.total, 2)
