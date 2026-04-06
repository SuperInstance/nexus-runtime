"""Tests for contracts module."""

import pytest
from datetime import datetime, timedelta

from jetson.marketplace.contracts import (
    Contract, SLATerm, ContractStatus, EnforcementAction,
    SLAComplianceResult, EnforcementResult, SLAMonitor,
)


class TestSLATerm:
    def test_default(self):
        t = SLATerm()
        assert t.metric == ""
        assert t.target == 0.0
        assert t.penalty_per_breach == 0.0

    def test_custom(self):
        t = SLATerm(metric="completion_rate", target=0.95, penalty_per_breach=500.0)
        assert t.metric == "completion_rate"
        assert t.target == 0.95
        assert t.penalty_per_breach == 500.0

    def test_measurement_periods(self):
        for period in ["per_task", "daily", "weekly"]:
            t = SLATerm(measurement_period=period)
            assert t.measurement_period == period


class TestContract:
    def test_default(self):
        c = Contract()
        assert c.id == ""
        assert c.status == ContractStatus.PENDING
        assert c.terms == []

    def test_custom(self):
        terms = [SLATerm(metric="quality", target=0.9)]
        c = Contract(
            id="C-001", task_id="t1", vessel_id="v1",
            terms=terms, total_value=5000.0,
        )
        assert c.id == "C-001"
        assert c.task_id == "t1"
        assert len(c.terms) == 1
        assert c.total_value == 5000.0


class TestContractStatus:
    def test_enum_values(self):
        assert ContractStatus.ACTIVE.value == "ACTIVE"
        assert ContractStatus.COMPLETED.value == "COMPLETED"
        assert ContractStatus.BREACHED.value == "BREACHED"
        assert ContractStatus.TERMINATED.value == "TERMINATED"
        assert ContractStatus.PENDING.value == "PENDING"


class TestEnforcementAction:
    def test_enum_values(self):
        assert EnforcementAction.NONE.value == "NONE"
        assert EnforcementAction.WARNING.value == "WARNING"
        assert EnforcementAction.PENALTY.value == "PENALTY"
        assert EnforcementAction.TERMINATION.value == "TERMINATION"


class TestSLAMonitor:
    def setup_method(self):
        self.monitor = SLAMonitor()

    def test_create_contract(self):
        class FakeTask:
            id = "task-001"
            reward = 5000.0
            deadline = datetime.utcnow() + timedelta(days=7)
        class FakeVessel:
            vessel_id = "vessel-001"
        contract = self.monitor.create_contract(FakeTask(), FakeVessel())
        assert contract.task_id == "task-001"
        assert contract.vessel_id == "vessel-001"
        assert contract.status == ContractStatus.ACTIVE
        assert contract.total_value == 5000.0
        assert len(contract.terms) == 2

    def test_create_contract_custom_terms(self):
        class FakeTask:
            id = "t1"
            reward = 3000.0
            deadline = None
        class FakeVessel:
            vessel_id = "v1"
        terms = [SLATerm(metric="uptime", target=0.99, penalty_per_breach=1000.0)]
        contract = self.monitor.create_contract(FakeTask(), FakeVessel(), terms)
        assert len(contract.terms) == 1
        assert contract.terms[0].metric == "uptime"

    def test_create_contract_id_format(self):
        class FakeTask:
            id = "task-12345678"
            reward = 1000.0
            deadline = None
        class FakeVessel:
            vessel_id = "vessel-87654321"
        contract = self.monitor.create_contract(FakeTask(), FakeVessel())
        assert "task-123" in contract.id
        assert "vessel-8" in contract.id

    def test_check_sla_compliant(self):
        contract = Contract(
            terms=[SLATerm(metric="quality", target=0.8, penalty_per_breach=200.0)],
        )
        results = self.monitor.check_sla(contract, {"quality": 0.9})
        assert len(results) == 1
        assert results[0].compliant is True
        assert results[0].violation_amount == 0.0

    def test_check_sla_breach(self):
        contract = Contract(
            terms=[SLATerm(metric="quality", target=0.9, penalty_per_breach=200.0)],
        )
        results = self.monitor.check_sla(contract, {"quality": 0.7})
        assert len(results) == 1
        assert results[0].compliant is False
        assert results[0].violation_amount == pytest.approx(0.2, abs=0.01)

    def test_check_sla_missing_metric(self):
        contract = Contract(
            terms=[SLATerm(metric="quality", target=0.8, penalty_per_breach=200.0)],
        )
        results = self.monitor.check_sla(contract, {})
        assert len(results) == 1
        assert results[0].compliant is False
        assert results[0].actual == 0.0

    def test_check_sla_multiple_terms(self):
        contract = Contract(
            terms=[
                SLATerm(metric="quality", target=0.8, penalty_per_breach=200.0),
                SLATerm(metric="speed", target=0.9, penalty_per_breach=300.0),
            ],
        )
        results = self.monitor.check_sla(contract, {"quality": 0.9, "speed": 0.7})
        assert len(results) == 2
        assert results[0].compliant is True
        assert results[1].compliant is False

    def test_check_sla_no_terms(self):
        contract = Contract(terms=[])
        results = self.monitor.check_sla(contract, {"quality": 0.5})
        assert results == []

    def test_compute_penalty_no_violations(self):
        contract = Contract(
            terms=[SLATerm(metric="quality", target=0.8, penalty_per_breach=200.0)],
        )
        violations = [SLAComplianceResult(metric="quality", target=0.8, actual=0.9, compliant=True)]
        penalty = self.monitor.compute_penalty(contract, violations)
        assert penalty == 0.0

    def test_compute_penalty_single_violation(self):
        contract = Contract(
            terms=[SLATerm(metric="quality", target=0.8, penalty_per_breach=500.0)],
        )
        violations = [SLAComplianceResult(metric="quality", target=0.8, actual=0.6, compliant=False)]
        penalty = self.monitor.compute_penalty(contract, violations)
        assert penalty == 500.0

    def test_compute_penalty_multiple_violations(self):
        contract = Contract(
            terms=[
                SLATerm(metric="quality", target=0.8, penalty_per_breach=200.0),
                SLATerm(metric="speed", target=0.9, penalty_per_breach=300.0),
            ],
        )
        violations = [
            SLAComplianceResult(metric="quality", target=0.8, actual=0.6, compliant=False),
            SLAComplianceResult(metric="speed", target=0.9, actual=0.7, compliant=False),
        ]
        penalty = self.monitor.compute_penalty(contract, violations)
        assert penalty == 500.0

    def test_compute_penalty_no_terms(self):
        contract = Contract(terms=[])
        violations = [SLAComplianceResult(metric="x", target=0.8, actual=0.5, compliant=False)]
        assert self.monitor.compute_penalty(contract, violations) == 0.0

    def test_update_contract_status_complete(self):
        contract = Contract(status=ContractStatus.ACTIVE)
        updated = self.monitor.update_contract_status(contract, {"type": "complete"})
        assert updated.status == ContractStatus.COMPLETED

    def test_update_contract_status_breach(self):
        contract = Contract(status=ContractStatus.ACTIVE)
        updated = self.monitor.update_contract_status(contract, {
            "type": "breach", "penalty": 500.0, "metric": "quality",
        })
        assert updated.status == ContractStatus.BREACHED
        assert updated.penalties["quality"] == 500.0

    def test_update_contract_status_terminate(self):
        contract = Contract(status=ContractStatus.ACTIVE)
        updated = self.monitor.update_contract_status(contract, {"type": "terminate"})
        assert updated.status == ContractStatus.TERMINATED

    def test_update_contract_status_activate(self):
        contract = Contract(status=ContractStatus.PENDING)
        updated = self.monitor.update_contract_status(contract, {"type": "activate"})
        assert updated.status == ContractStatus.ACTIVE

    def test_update_contract_status_suspend(self):
        contract = Contract(status=ContractStatus.ACTIVE)
        updated = self.monitor.update_contract_status(contract, {"type": "suspend"})
        assert updated.status == ContractStatus.PENDING

    def test_update_contract_accumulates_penalties(self):
        contract = Contract(status=ContractStatus.ACTIVE, penalties={"quality": 200.0})
        updated = self.monitor.update_contract_status(contract, {
            "type": "breach", "penalty": 300.0, "metric": "quality",
        })
        assert updated.penalties["quality"] == 500.0

    def test_update_contract_does_not_mutate_original(self):
        contract = Contract(status=ContractStatus.ACTIVE)
        self.monitor.update_contract_status(contract, {"type": "complete"})
        assert contract.status == ContractStatus.ACTIVE

    def test_enforce_contract_no_violation(self):
        contract = Contract(
            total_value=5000.0,
            terms=[SLATerm(metric="quality", target=0.8, penalty_per_breach=200.0)],
        )
        result = self.monitor.enforce_contract(contract, {"quality": 0.9})
        assert result.action == EnforcementAction.NONE
        assert result.penalty_amount == 0.0
        assert result.adjusted_value == 5000.0

    def test_enforce_contract_single_violation(self):
        contract = Contract(
            total_value=5000.0,
            terms=[SLATerm(metric="quality", target=0.8, penalty_per_breach=200.0)],
        )
        result = self.monitor.enforce_contract(contract, {"quality": 0.6})
        assert result.action == EnforcementAction.WARNING
        assert result.penalty_amount == 200.0
        assert result.adjusted_value == 4800.0

    def test_enforce_contract_multiple_violations(self):
        contract = Contract(
            total_value=5000.0,
            terms=[
                SLATerm(metric="quality", target=0.8, penalty_per_breach=2000.0),
                SLATerm(metric="speed", target=0.9, penalty_per_breach=1500.0),
            ],
        )
        result = self.monitor.enforce_contract(contract, {"quality": 0.5, "speed": 0.5})
        assert result.action == EnforcementAction.TERMINATION
        assert result.penalty_amount == 3500.0

    def test_enforce_contract_details(self):
        contract = Contract(
            total_value=5000.0,
            terms=[SLATerm(metric="quality", target=0.8, penalty_per_breach=200.0)],
        )
        result = self.monitor.enforce_contract(contract, {"quality": 0.6})
        assert "1 SLA violation" in result.details

    def test_compute_contract_value(self):
        contract = Contract(
            total_value=5000.0,
            terms=[SLATerm(metric="quality", target=0.8, penalty_per_breach=200.0)],
        )
        value = self.monitor.compute_contract_value(contract, {"quality": 0.6})
        assert value == 4800.0

    def test_compute_contract_value_full_compliance(self):
        contract = Contract(
            total_value=5000.0,
            terms=[SLATerm(metric="quality", target=0.8, penalty_per_breach=200.0)],
        )
        value = self.monitor.compute_contract_value(contract, {"quality": 0.9})
        assert value == 5000.0

    def test_compute_contract_value_no_negative(self):
        contract = Contract(
            total_value=100.0,
            terms=[SLATerm(metric="quality", target=0.8, penalty_per_breach=500.0)],
        )
        result = self.monitor.enforce_contract(contract, {"quality": 0.5})
        assert result.adjusted_value >= 0.0
