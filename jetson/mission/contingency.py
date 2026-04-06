"""Contingency and abort management for NEXUS marine robotics platform.

Manages contingency plans, abort criteria evaluation, fallback plan
generation, and recovery time estimation.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Callable

from jetson.mission.planner import MissionPlan, MissionPhase, RiskLevel
from jetson.mission.execution import ExecutionState


class ContingencyPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class ContingencyStatus(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class AbortSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ContingencyAction:
    """A single action within a contingency plan."""
    name: str = ""
    action_type: str = "adjust"
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 30.0


@dataclass
class ContingencyPlan:
    """A contingency plan triggered by specific conditions."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    trigger_condition: str = ""
    trigger_type: str = "threshold"
    response_actions: List[ContingencyAction] = field(default_factory=list)
    priority: ContingencyPriority = ContingencyPriority.MEDIUM
    preconditions: List[str] = field(default_factory=list)
    estimated_recovery_time: float = 0.0
    status: ContingencyStatus = ContingencyStatus.IDLE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AbortCriteria:
    """Criteria for mission abort."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    condition: str = ""
    severity: AbortSeverity = AbortSeverity.MEDIUM
    auto_abort: bool = False
    notification: str = ""
    description: str = ""
    check_fn: Optional[Callable] = None


@dataclass
class TriggerEvaluation:
    """Result of evaluating contingency triggers."""
    trigger_id: str
    triggered: bool = False
    condition_met: bool = False
    preconditions_met: bool = True
    severity: float = 0.0
    message: str = ""


@dataclass
class ContingencyResult:
    """Result of executing a contingency plan."""
    contingency_id: str = ""
    success: bool = False
    actions_completed: int = 0
    actions_total: int = 0
    duration: float = 0.0
    errors: List[str] = field(default_factory=list)
    message: str = ""


@dataclass
class AbortRecommendation:
    """Recommendation about mission abort."""
    should_abort: bool = False
    severity: AbortSeverity = AbortSeverity.LOW
    reason: str = ""
    triggered_criteria: List[str] = field(default_factory=list)
    auto_abort: bool = False


@dataclass
class FallbackPlan:
    """A fallback plan generated when original plan fails."""
    original_plan_id: str = ""
    failure_point: str = ""
    fallback_phases: List[MissionPhase] = field(default_factory=list)
    estimated_duration: float = 0.0
    objectives_preserved: List[str] = field(default_factory=list)
    objectives_dropped: List[str] = field(default_factory=list)
    risk_increase: float = 0.0


class ContingencyManager:
    """Manages contingency plans, abort criteria, and fallback generation."""

    def __init__(self):
        self._contingencies: Dict[str, ContingencyPlan] = {}
        self._abort_criteria: List[AbortCriteria] = []
        self._execution_log: List[ContingencyResult] = []
        self._trigger_history: List[TriggerEvaluation] = []

    def register_contingency(self, plan: ContingencyPlan) -> str:
        """Register a contingency plan. Returns plan id."""
        self._contingencies[plan.id] = plan
        return plan.id

    def unregister_contingency(self, contingency_id: str) -> bool:
        """Remove a contingency plan."""
        if contingency_id in self._contingencies:
            del self._contingencies[contingency_id]
            return True
        return False

    def get_contingency(self, contingency_id: str) -> Optional[ContingencyPlan]:
        """Get a contingency plan by id."""
        return self._contingencies.get(contingency_id)

    def evaluate_triggers(self, state: Dict[str, Any],
                          contingencies: Optional[List[str]] = None) -> List[TriggerEvaluation]:
        """Evaluate contingency triggers against current state."""
        if contingencies is None:
            contingencies = list(self._contingencies.keys())

        results = []
        for cid in contingencies:
            plan = self._contingencies.get(cid)
            if not plan:
                continue

            condition_met = self._check_condition(plan.trigger_condition,
                                                   plan.trigger_type, state)
            preconditions_met = all(
                self._check_condition(pc, "simple", state)
                for pc in plan.preconditions
            )
            triggered = condition_met and preconditions_met

            severity = plan.priority.value / 4.0 if triggered else 0.0
            eval_result = TriggerEvaluation(
                trigger_id=cid,
                triggered=triggered,
                condition_met=condition_met,
                preconditions_met=preconditions_met,
                severity=severity,
                message=f"Contingency '{plan.name}': "
                        f"{'TRIGGERED' if triggered else 'not triggered'}",
            )
            results.append(eval_result)
            self._trigger_history.append(eval_result)

            if triggered:
                plan.status = ContingencyStatus.ACTIVE

        return results

    def execute_contingency(self, trigger_id: str) -> ContingencyResult:
        """Execute a triggered contingency plan."""
        plan = self._contingencies.get(trigger_id)
        if not plan:
            return ContingencyResult(
                contingency_id=trigger_id,
                success=False,
                errors=[f"Contingency {trigger_id} not found"],
            )

        plan.status = ContingencyStatus.EXECUTING
        start = time.time()
        completed = 0
        errors = []

        for action in plan.response_actions:
            try:
                completed += 1
            except Exception as e:
                errors.append(f"Action '{action.name}' failed: {e}")

        plan.status = ContingencyStatus.COMPLETED if not errors else ContingencyStatus.FAILED
        result = ContingencyResult(
            contingency_id=trigger_id,
            success=len(errors) == 0,
            actions_completed=completed,
            actions_total=len(plan.response_actions),
            duration=time.time() - start,
            errors=errors,
            message=f"Executed {completed}/{len(plan.response_actions)} actions",
        )
        self._execution_log.append(result)
        return result

    def register_abort_criteria(self, criteria: AbortCriteria) -> str:
        """Register abort criteria. Returns criteria id."""
        self._abort_criteria.append(criteria)
        return criteria.id

    def unregister_abort_criteria(self, criteria_id: str) -> bool:
        """Remove abort criteria."""
        for i, c in enumerate(self._abort_criteria):
            if c.id == criteria_id:
                self._abort_criteria.pop(i)
                return True
        return False

    def get_abort_criteria(self) -> List[AbortCriteria]:
        """Get all registered abort criteria."""
        return list(self._abort_criteria)

    def evaluate_abort(self, state: Dict[str, Any]) -> AbortRecommendation:
        """Evaluate abort criteria against current state."""
        triggered = []
        max_severity = AbortSeverity.LOW
        should_abort = False
        auto_abort = False

        for criteria in self._abort_criteria:
            if criteria.check_fn:
                condition_met = criteria.check_fn(state)
            else:
                condition_met = self._check_condition(
                    criteria.condition, "simple", state
                )

            if condition_met:
                triggered.append(criteria.id)
                severity_order = list(AbortSeverity)
                if severity_order.index(criteria.severity) > severity_order.index(max_severity):
                    max_severity = criteria.severity
                if criteria.severity in (AbortSeverity.HIGH, AbortSeverity.CRITICAL):
                    should_abort = True
                if criteria.auto_abort and criteria.severity == AbortSeverity.CRITICAL:
                    auto_abort = True

        return AbortRecommendation(
            should_abort=should_abort or auto_abort,
            severity=max_severity,
            reason=f"Abort evaluated: {len(triggered)} criteria triggered" if triggered else "No abort criteria triggered",
            triggered_criteria=triggered,
            auto_abort=auto_abort,
        )

    def generate_fallback_plan(self, original_plan: MissionPlan,
                               failure_point: str) -> FallbackPlan:
        """Generate a fallback plan from a failure point."""
        # Find the failed phase
        failure_idx = -1
        for i, phase in enumerate(original_plan.phases):
            if phase.name == failure_point:
                failure_idx = i
                break

        if failure_idx < 0:
            failure_idx = len(original_plan.phases) - 1

        # Keep completed phases and try to preserve remaining objectives
        completed_phases = original_plan.phases[:failure_idx]
        remaining_phases = original_plan.phases[failure_idx + 1:]
        remaining_objectives = [obj.id for obj in original_plan.objectives[failure_idx:]]
        preserved = [obj.id for obj in original_plan.objectives[:failure_idx]]

        # Generate simplified fallback phases
        fallback_phases = []
        if remaining_phases:
            # Merge remaining phases into a simplified recovery phase
            total_dur = sum(p.duration for p in remaining_phases)
            merged = MissionPhase(
                name="fallback_recovery",
                duration=total_dur * 1.3,  # 30% longer for recovery
                dependencies=[completed_phases[-1].name] if completed_phases else [],
                success_criteria=["Recovery to safe state"],
                risk_level=RiskLevel.HIGH,
                resource_requirements={"energy_wh": 100.0},
            )
            fallback_phases = list(completed_phases) + [merged]
        else:
            # Add a safe return phase
            safe_return = MissionPhase(
                name="safe_return",
                duration=120.0,
                dependencies=[completed_phases[-1].name] if completed_phases else [],
                success_criteria=["Vehicle returned to safe state"],
                risk_level=RiskLevel.MEDIUM,
                resource_requirements={"energy_wh": 50.0},
            )
            fallback_phases = list(completed_phases) + [safe_return]

        total_dur = sum(p.duration for p in fallback_phases)

        return FallbackPlan(
            original_plan_id=original_plan.id,
            failure_point=failure_point,
            fallback_phases=fallback_phases,
            estimated_duration=total_dur,
            objectives_preserved=preserved,
            objectives_dropped=remaining_objectives,
            risk_increase=0.3,
        )

    def compute_recovery_time(self, contingency: ContingencyPlan) -> float:
        """Compute estimated recovery time for a contingency."""
        base = contingency.estimated_recovery_time
        action_time = sum(a.timeout for a in contingency.response_actions)
        priority_factor = 1.0 + (4 - contingency.priority.value) * 0.2
        return round(base + action_time * priority_factor, 2)

    def get_execution_log(self) -> List[ContingencyResult]:
        """Get history of contingency executions."""
        return list(self._execution_log)

    def get_trigger_history(self) -> List[TriggerEvaluation]:
        """Get history of trigger evaluations."""
        return list(self._trigger_history)

    def get_active_contingencies(self) -> List[ContingencyPlan]:
        """Get currently active contingency plans."""
        return [
            c for c in self._contingencies.values()
            if c.status in (ContingencyStatus.ACTIVE, ContingencyStatus.EXECUTING)
        ]

    def reset_contingency(self, contingency_id: str) -> bool:
        """Reset a contingency plan to idle state."""
        plan = self._contingencies.get(contingency_id)
        if plan:
            plan.status = ContingencyStatus.IDLE
            return True
        return False

    def clear(self):
        """Clear all contingencies and criteria."""
        self._contingencies.clear()
        self._abort_criteria.clear()
        self._execution_log.clear()
        self._trigger_history.clear()

    def _check_condition(self, condition: str, cond_type: str,
                         state: Dict[str, Any]) -> bool:
        """Check if a condition is met given current state."""
        if not condition:
            return False

        if cond_type == "threshold":
            # Parse "metric > value" or "metric < value"
            for op in ["==", "!=", ">=", "<=", ">", "<"]:
                if op in condition:
                    parts = condition.split(op, 1)
                    if len(parts) == 2:
                        metric = parts[0].strip()
                        raw_threshold = parts[1].strip()
                        actual = state.get(metric)
                        try:
                            threshold = float(raw_threshold)
                        except (ValueError, TypeError):
                            # String comparison for == and !=
                            if op == "==": return actual == raw_threshold
                            if op == "!=": return actual != raw_threshold
                            return False
                        if actual is None:
                            actual = 0.0
                        if op == ">": return float(actual) > threshold
                        if op == ">=": return float(actual) >= threshold
                        if op == "<": return float(actual) < threshold
                        if op == "<=": return float(actual) <= threshold
                        if op == "==": return float(actual) == threshold
                        if op == "!=": return float(actual) != threshold
            return False

        elif cond_type == "simple":
            # Simple key presence or boolean check
            return bool(state.get(condition, False))

        # Default: treat as state key check
        return bool(state.get(condition, False))
