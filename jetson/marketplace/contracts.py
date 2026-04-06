"""Contract enforcement, SLA monitoring, and penalties."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ContractStatus(Enum):
    """Status of a contract."""
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    BREACHED = "BREACHED"
    TERMINATED = "TERMINATED"
    PENDING = "PENDING"


class EnforcementAction(Enum):
    """Possible enforcement actions."""
    NONE = "NONE"
    WARNING = "WARNING"
    PENALTY = "PENALTY"
    TERMINATION = "TERMINATION"


@dataclass
class SLATerm:
    """A service level agreement term."""
    metric: str = ""
    target: float = 0.0
    measurement_period: str = "per_task"  # per_task, daily, weekly
    penalty_per_breach: float = 0.0


@dataclass
class Contract:
    """A contract between a task poster and a vessel operator."""
    id: str = ""
    task_id: str = ""
    vessel_id: str = ""
    terms: List[SLATerm] = field(default_factory=list)
    start_time: Optional[datetime] = None
    deadline: Optional[datetime] = None
    penalties: Dict[str, float] = field(default_factory=dict)
    status: ContractStatus = ContractStatus.PENDING
    total_value: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SLAComplianceResult:
    """Result of SLA compliance check."""
    metric: str = ""
    target: float = 0.0
    actual: float = 0.0
    compliant: bool = False
    violation_amount: float = 0.0


@dataclass
class EnforcementResult:
    """Result of contract enforcement."""
    action: EnforcementAction = EnforcementAction.NONE
    penalty_amount: float = 0.0
    adjusted_value: float = 0.0
    details: str = ""


class SLAMonitor:
    """Monitors and enforces SLA terms on contracts."""

    def create_contract(
        self,
        task: Any,
        vessel: Any,
        terms: Optional[List[SLATerm]] = None,
    ) -> Contract:
        """Create a new contract from a task and vessel."""
        task_id = getattr(task, "id", "")
        vessel_id = getattr(vessel, "vessel_id", "")
        reward = getattr(task, "reward", 0.0)
        deadline = getattr(task, "deadline", None)

        if terms is None:
            terms = [
                SLATerm(metric="completion_rate", target=1.0, measurement_period="per_task", penalty_per_breach=500.0),
                SLATerm(metric="quality_score", target=0.8, measurement_period="per_task", penalty_per_breach=200.0),
            ]

        contract = Contract(
            id=f"C-{task_id[:8]}-{vessel_id[:8]}",
            task_id=task_id,
            vessel_id=vessel_id,
            terms=terms,
            start_time=datetime.utcnow(),
            deadline=deadline,
            total_value=reward,
            status=ContractStatus.ACTIVE,
        )
        return contract

    def check_sla(
        self,
        contract: Contract,
        actual_performance: Dict[str, float],
    ) -> List[SLAComplianceResult]:
        """Check SLA compliance against actual performance metrics."""
        results: List[SLAComplianceResult] = []
        for term in contract.terms:
            actual = actual_performance.get(term.metric, 0.0)
            compliant = actual >= term.target
            violation = max(0.0, term.target - actual)
            results.append(SLAComplianceResult(
                metric=term.metric,
                target=term.target,
                actual=actual,
                compliant=compliant,
                violation_amount=round(violation, 4),
            ))
        return results

    def compute_penalty(self, contract: Contract, violations: List[SLAComplianceResult]) -> float:
        """Compute total penalty for SLA violations."""
        total = 0.0
        for violation in violations:
            if not violation.compliant:
                # Find matching SLA term
                for term in contract.terms:
                    if term.metric == violation.metric:
                        total += term.penalty_per_breach
                        break
        return round(total, 2)

    def update_contract_status(
        self,
        contract: Contract,
        event: Dict[str, Any],
    ) -> Contract:
        """Update contract status based on an event."""
        event_type = event.get("type", "")
        new_contract = Contract(
            id=contract.id,
            task_id=contract.task_id,
            vessel_id=contract.vessel_id,
            terms=contract.terms,
            start_time=contract.start_time,
            deadline=contract.deadline,
            penalties=dict(contract.penalties),
            status=contract.status,
            total_value=contract.total_value,
            metadata=dict(contract.metadata),
        )

        if event_type == "complete":
            new_contract.status = ContractStatus.COMPLETED
        elif event_type == "breach":
            new_contract.status = ContractStatus.BREACHED
            penalty = event.get("penalty", 0.0)
            metric = event.get("metric", "unknown")
            new_contract.penalties[metric] = new_contract.penalties.get(metric, 0.0) + penalty
        elif event_type == "terminate":
            new_contract.status = ContractStatus.TERMINATED
        elif event_type == "activate":
            new_contract.status = ContractStatus.ACTIVE
        elif event_type == "suspend":
            new_contract.status = ContractStatus.PENDING

        return new_contract

    def enforce_contract(
        self,
        contract: Contract,
        performance: Dict[str, float],
    ) -> EnforcementResult:
        """Enforce contract based on performance data."""
        compliance_results = self.check_sla(contract, performance)
        violations = [r for r in compliance_results if not r.compliant]
        penalty = self.compute_penalty(contract, violations)
        adjusted_value = contract.total_value - penalty

        if not violations:
            action = EnforcementAction.NONE
            details = "All SLA terms met."
        elif len(violations) == 1:
            action = EnforcementAction.WARNING
            details = f"1 SLA violation: {violations[0].metric}"
        elif penalty < contract.total_value * 0.5:
            action = EnforcementAction.PENALTY
            details = f"{len(violations)} SLA violations, penalty applied."
        else:
            action = EnforcementAction.TERMINATION
            details = f"Severe breach: {len(violations)} violations, contract terminated."

        return EnforcementResult(
            action=action,
            penalty_amount=penalty,
            adjusted_value=round(max(0.0, adjusted_value), 2),
            details=details,
        )

    def compute_contract_value(
        self,
        contract: Contract,
        performance: Dict[str, float],
    ) -> float:
        """Compute the adjusted contract value based on performance."""
        enforcement = self.enforce_contract(contract, performance)
        return enforcement.adjusted_value
