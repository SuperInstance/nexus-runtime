"""NEXUS Tripartite Consensus Engine.

Orchestrates Pathos/Logos/Ethos agents with competing voting strategies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .agents import (
    AgentAssessment, DecisionVerdict, EthosAgent, IntentAssessment,
    LogosAgent, PathosAgent, SafetyAssessment,
)


class VotingStrategy(Enum):
    UNANIMOUS = "unanimous"   # all 3 must approve
    MAJORITY = "majority"     # 2 of 3
    WEIGHTED = "weighted"     # Pathos=0.3, Logos=0.4, Ethos=0.3
    VETO = "veto"             # any can veto, requires 2 to approve
    AUTO_ETHOS = "auto_ethos" # ethos has tie-breaking power


@dataclass(frozen=True)
class ConsensusResult:
    """Result of a tripartite consensus evaluation."""
    decision: DecisionVerdict
    strategy: VotingStrategy
    pathos: IntentAssessment
    logos: PlanAssessment
    ethos: SafetyAssessment
    weighted_score: float
    elapsed_ms: float
    decision_log: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def approved(self) -> bool:
        return self.decision == DecisionVerdict.APPROVE

    def summary(self) -> str:
        status = "APPROVED" if self.approved else "REJECTED"
        lines = [f"Consensus: {status} (strategy={self.strategy.value}, score={self.weighted_score:.3f}, {self.elapsed_ms:.1f}ms)"]
        for a in (self.pathos, self.logos, self.ethos):
            lines.append(f"  {a.agent_name}: {a.verdict.value} (conf={a.confidence:.2f}, score={a.score:.2f})")
            for r in a.reasons:
                lines.append(f"    - {r}")
        return "\n".join(lines)


class ConsensusEngine:
    """Orchestrates the three tripartite agents with configurable voting."""

    WEIGHTS = {"pathos": 0.3, "logos": 0.4, "ethos": 0.3}

    def __init__(self, pathos: PathosAgent | None = None, logos: LogosAgent | None = None,
                 ethos: EthosAgent | None = None,
                 strategy: VotingStrategy = VotingStrategy.MAJORITY,
                 timeout_ms: float = 5000.0) -> None:
        self.pathos = pathos or PathosAgent()
        self.logos = logos or LogosAgent()
        self.ethos = ethos or EthosAgent()
        self.strategy = strategy
        self.timeout_ms = timeout_ms
        self._decision_history: list[ConsensusResult] = []

    def evaluate(self, action_description: str,
                 context: dict[str, Any] | None = None) -> ConsensusResult:
        """Run full tripartite consensus evaluation."""
        ctx = context or {}
        start = time.monotonic()
        try:
            pathos_result = self.pathos.evaluate(action_description, ctx)
            logos_result = self.logos.evaluate(action_description, ctx)
            ethos_result = self.ethos.evaluate(action_description, ctx)
        except Exception as exc:
            # On error, reject for safety
            return ConsensusResult(
                decision=DecisionVerdict.REJECT,
                strategy=self.strategy,
                pathos=IntentAssessment(agent_name="pathos", verdict=DecisionVerdict.REJECT, confidence=0.0, score=0.0),
                logos=PlanAssessment(agent_name="logos", verdict=DecisionVerdict.REJECT, confidence=0.0, score=0.0),
                ethos=SafetyAssessment(agent_name="ethos", verdict=DecisionVerdict.REJECT, confidence=0.0, score=0.0),
                weighted_score=0.0,
                elapsed_ms=(time.monotonic() - start) * 1000,
                decision_log=(f"Error during consensus: {exc}",),
            )

        elapsed = (time.monotonic() - start) * 1000
        timeout_hit = elapsed > self.timeout_ms

        if timeout_hit:
            decision = DecisionVerdict.REJECT
            decision_log = ("Consensus timeout — defaulting to safest action",)
        else:
            decision, decision_log = self._apply_strategy(
                pathos_result, logos_result, ethos_result)

        weighted_score = (
            pathos_result.score * self.WEIGHTS["pathos"] +
            logos_result.score * self.WEIGHTS["logos"] +
            ethos_result.score * self.WEIGHTS["ethos"]
        )

        result = ConsensusResult(
            decision=decision, strategy=self.strategy,
            pathos=pathos_result, logos=logos_result, ethos=ethos_result,
            weighted_score=round(weighted_score, 4),
            elapsed_ms=round(elapsed, 2),
            decision_log=decision_log,
        )
        self._decision_history.append(result)
        return result

    def _apply_strategy(self, pathos: IntentAssessment, logos: PlanAssessment,
                        ethos: SafetyAssessment) -> tuple[DecisionVerdict, tuple[str, ...]]:
        agents = [pathos, logos, ethos]
        approvals = sum(1 for a in agents if a.is_approval())
        rejections = sum(1 for a in agents if a.verdict == DecisionVerdict.REJECT)

        if self.strategy == VotingStrategy.UNANIMOUS:
            if approvals == 3:
                return DecisionVerdict.APPROVE, ("All agents approved",)
            return DecisionVerdict.REJECT, (f"Not unanimous: {approvals}/3 approved",)

        elif self.strategy == VotingStrategy.MAJORITY:
            if approvals >= 2:
                return DecisionVerdict.APPROVE, (f"Majority approval ({approvals}/3)",)
            return DecisionVerdict.REJECT, (f"No majority: {approvals}/3 approved",)

        elif self.strategy == VotingStrategy.WEIGHTED:
            weighted = (
                pathos.score * self.WEIGHTS["pathos"] +
                logos.score * self.WEIGHTS["logos"] +
                ethos.score * self.WEIGHTS["ethos"]
            )
            threshold = 0.6
            if weighted >= threshold and rejections == 0:
                return DecisionVerdict.APPROVE, (f"Weighted score {weighted:.3f} >= {threshold}",)
            return DecisionVerdict.REJECT, (f"Weighted score {weighted:.3f} < {threshold} or {rejections} rejection(s)",)

        elif self.strategy == VotingStrategy.VETO:
            if rejections >= 2:
                return DecisionVerdict.REJECT, ("Veto: 2+ agents rejected",)
            if approvals >= 2:
                return DecisionVerdict.APPROVE, ("2+ approved, no double veto",)
            return DecisionVerdict.DELEGATE, ("No clear majority, no double veto",)

        elif self.strategy == VotingStrategy.AUTO_ETHOS:
            # Ethos can break ties
            if approvals == 3:
                return DecisionVerdict.APPROVE, ("All agents approved",)
            if approvals == 2:
                return DecisionVerdict.APPROVE, ("Majority + ethos tiebreaker",)
            if ethos.is_approval():
                return DecisionVerdict.DELEGATE, ("Ethos approved but minority — delegated to human")
            return DecisionVerdict.REJECT, ("Ethos rejected — action blocked",)

        return DecisionVerdict.REJECT, ("Unknown strategy",)

    @property
    def decision_history(self) -> list[ConsensusResult]:
        return list(self._decision_history)

    def clear_history(self) -> None:
        self._decision_history.clear()
