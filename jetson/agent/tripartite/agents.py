"""NEXUS Tripartite Consensus — Pathos, Logos, Ethos agents.

Three agents evaluate safety-critical decisions from different perspectives:
  Pathos (Intent):   "What does the human intend?"
  Logos (Planning):   "Is this plan sound?"
  Ethos (Safety):     "Is this the right thing to do?"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IntentType(Enum):
    ROUTINE = "routine"
    FISHING = "fishing"
    NAVIGATION = "navigation"
    EMERGENCY = "emergency"
    RESCUE = "rescue"
    DOCKING = "docking"
    SURVEY = "survey"
    UNKNOWN = "unknown"


class DecisionVerdict(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    DELEGATE = "delegate"


@dataclass(frozen=True)
class AgentAssessment:
    """Base assessment produced by any tripartite agent."""
    agent_name: str
    verdict: DecisionVerdict
    confidence: float
    score: float
    reasons: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_approval(self) -> bool:
        return self.verdict == DecisionVerdict.APPROVE


@dataclass(frozen=True)
class IntentAssessment(AgentAssessment):
    intent_type: IntentType = IntentType.UNKNOWN
    alignment_score: float = 0.0
    urgency: float = 0.0


class PathosAgent:
    URGENCY_KEYWORDS = {
        "emergency": 1.0, "mayday": 1.0, "rescue": 0.9, "man overboard": 1.0,
        "collision": 1.0, "sinking": 1.0, "fire": 1.0, "abandon": 1.0,
        "immediate": 0.8, "urgent": 0.7, "asap": 0.6,
    }
    INTENT_KEYWORDS = {
        IntentType.FISHING: ["fish", "catch", "trawl", "net", "pot", "longline"],
        IntentType.NAVIGATION: ["navigate", "course", "waypoint", "heading", "route"],
        IntentType.EMERGENCY: ["emergency", "evacuate", "abandon", "mayday", "distress"],
        IntentType.RESCUE: ["rescue", "save", "recover", "assist", "help vessel"],
        IntentType.DOCKING: ["dock", "moor", "berth", "tie up", "approach harbor"],
        IntentType.SURVEY: ["survey", "map", "scan", "measure", "monitor", "sample"],
        IntentType.ROUTINE: ["patrol", "standby", "hold", "loiter", "station"],
    }

    def __init__(self, mission_type: IntentType = IntentType.ROUTINE,
                 mission_description: str = "") -> None:
        self.mission_type = mission_type
        self.mission_description = mission_description.lower()

    def evaluate(self, action_description: str,
                 context: dict[str, Any] | None = None) -> IntentAssessment:
        ctx = context or {}
        action_lower = action_description.lower()
        detected_intent = self._detect_intent(action_lower)
        urgency = self._detect_urgency(action_lower, ctx)
        if detected_intent == self.mission_type:
            alignment = 1.0
        elif detected_intent in (IntentType.EMERGENCY, IntentType.RESCUE):
            alignment = 0.95
        else:
            alignment = 0.3
        word_count = len(action_lower.split())
        clarity = min(1.0, word_count / 5.0)
        confidence = alignment * 0.5 + clarity * 0.3 + urgency * 0.2
        if alignment >= 0.8 and clarity >= 0.4:
            verdict = DecisionVerdict.APPROVE
        elif alignment < 0.3 or clarity < 0.2:
            verdict = DecisionVerdict.REJECT
        else:
            verdict = DecisionVerdict.DELEGATE
        reasons = []
        if alignment < 0.5:
            reasons.append(f"Intent mismatch: {detected_intent.value} vs mission {self.mission_type.value}")
        if clarity < 0.4:
            reasons.append("Action description is ambiguous")
        if urgency > 0.8:
            reasons.append("High urgency — expedited approval recommended")
        if alignment >= 0.8:
            reasons.append("Action aligns with stated mission")
        return IntentAssessment(
            agent_name="pathos", verdict=verdict, confidence=round(confidence, 4),
            score=round(alignment, 4), intent_type=detected_intent,
            alignment_score=round(alignment, 4), urgency=round(urgency, 4),
            reasons=tuple(reasons),
        )

    def _detect_intent(self, text: str) -> IntentType:
        for it, kws in self.INTENT_KEYWORDS.items():
            if any(kw in text for kw in kws):
                return it
        return IntentType.UNKNOWN

    def _detect_urgency(self, text: str, ctx: dict[str, Any]) -> float:
        u = 0.0
        for kw, val in self.URGENCY_KEYWORDS.items():
            if kw in text:
                u = max(u, val)
        if ctx.get("emergency", False):
            u = max(u, 1.0)
        return u


@dataclass(frozen=True)
class PlanAssessment(AgentAssessment):
    is_sound: bool = False
    risk_score: float = 0.0
    resource_check: bool = False
    navigation_check: bool = False
    cycle_budget_ok: bool = False
    stack_budget_ok: bool = False


class LogosAgent:
    MAX_CYCLE_BUDGET = 10000
    MAX_STACK_DEPTH = 64

    def __init__(self, fuel_pct: float = 100.0, battery_pct: float = 100.0,
                 max_speed_knots: float = 10.0, position: tuple[float, float] = (0.0, 0.0),
                 consumption_rate_pct_per_nm: float = 2.0) -> None:
        self.fuel_pct = fuel_pct
        self.battery_pct = battery_pct
        self.max_speed_knots = max_speed_knots
        self.position = position
        self.consumption_rate = consumption_rate_pct_per_nm

    def evaluate(self, action_description: str,
                 context: dict[str, Any] | None = None) -> PlanAssessment:
        ctx = context or {}
        distance_nm = ctx.get("distance_nm", 0.0)
        fuel_needed = distance_nm * self.consumption_rate
        resource_check = (self.fuel_pct - fuel_needed) > 10.0
        target = ctx.get("target_position")
        nav_check = True
        if target:
            dist = self._haversine(self.position, target)
            nav_check = dist <= (self.fuel_pct / self.consumption_rate) * 0.9
        bytecode_len = ctx.get("bytecode_length", 0)
        cycle_budget = bytecode_len <= self.MAX_CYCLE_BUDGET
        stack_budget = ctx.get("max_stack_depth", 0) <= self.MAX_STACK_DEPTH
        risk = 0.0
        if not resource_check:
            risk += 0.4
        if not nav_check:
            risk += 0.3
        if not cycle_budget:
            risk += 0.15
        if not stack_budget:
            risk += 0.15
        speed = ctx.get("requested_speed_knots", 0.0)
        if speed > self.max_speed_knots:
            risk += 0.2
        is_sound = resource_check and nav_check and risk < 0.5
        confidence = 0.5 + (1.0 - risk) * 0.5
        reasons = []
        if not resource_check:
            reasons.append(f"Insufficient fuel: need {fuel_needed:.1f}%, have {self.fuel_pct:.1f}%")
        if not nav_check:
            reasons.append("Navigation target may be out of range")
        if not cycle_budget:
            reasons.append(f"Bytecode exceeds cycle budget ({bytecode_len} > {self.MAX_CYCLE_BUDGET})")
        if not stack_budget:
            reasons.append("Bytecode exceeds stack budget")
        if is_sound:
            reasons.append("Plan is physically and logically sound")
        return PlanAssessment(
            agent_name="logos",
            verdict=DecisionVerdict.APPROVE if is_sound else DecisionVerdict.REJECT,
            confidence=round(min(confidence, 1.0), 4), score=round(1.0 - risk, 4),
            is_sound=is_sound, risk_score=round(min(risk, 1.0), 4),
            resource_check=resource_check, navigation_check=nav_check,
            cycle_budget_ok=cycle_budget, stack_budget_ok=stack_budget,
            reasons=tuple(reasons),
        )

    @staticmethod
    def _haversine(pos1: tuple[float, float], pos2: tuple[float, float]) -> float:
        lat1, lon1 = math.radians(pos1[0]), math.radians(pos1[1])
        lat2, lon2 = math.radians(pos2[0]), math.radians(pos2[1])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return 3440.065 * 2 * math.asin(math.sqrt(a))


@dataclass(frozen=True)
class SafetyAssessment(AgentAssessment):
    is_safe: bool = False
    violations: tuple[str, ...] = ()
    trust_required: int = 0
    ethical_score: float = 0.0


class EthosAgent:
    TRUST_REQUIREMENTS = {
        "read_sensor": 0, "navigate": 1, "deploy_reflex": 2,
        "change_course": 2, "approach_vessel": 3,
        "emergency_stop": 0, "override_safety": 5,
    }

    def __init__(self, trust_level: int = 0, safety_state: str = "normal",
                 no_go_zones: list[dict[str, Any]] | None = None,
                 colregs_rules: bool = True) -> None:
        self.trust_level = trust_level
        self.safety_state = safety_state
        self.no_go_zones = no_go_zones or []
        self.colregs_rules = colregs_rules

    def evaluate(self, action_description: str,
                 context: dict[str, Any] | None = None) -> SafetyAssessment:
        ctx = context or {}
        action_lower = action_description.lower()
        violations = []
        trust_required = self._get_trust_requirement(action_lower)
        ethical_score = 1.0
        trust_ok = self.trust_level >= trust_required
        if not trust_ok:
            violations.append(f"Trust {self.trust_level} insufficient; requires {trust_required}")
        if self.safety_state in ("fault", "safe_state"):
            if action_lower not in ("emergency stop", "diagnostic", "reset"):
                violations.append(f"Vessel in {self.safety_state} — restricted operations only")
                trust_ok = False
        target = ctx.get("target_position")
        if target:
            for zone in self.no_go_zones:
                if self._point_in_zone(target, zone):
                    violations.append(f"Target in no-go zone: {zone.get('name', 'unnamed')}")
        if self.colregs_rules and ctx.get("nearby_vessels", 0) > 0:
            if "collision" in action_lower or "ram" in action_lower:
                violations.append("COLREGs violation: must avoid collision")
        requested_speed = ctx.get("requested_speed_knots", 0.0)
        max_speed = ctx.get("max_speed_knots", 999.0)
        if requested_speed > max_speed:
            violations.append(f"Speed {requested_speed}kn exceeds limit {max_speed}kn")
        if ctx.get("marine_life_nearby", False) and "high speed" in action_lower:
            violations.append("Marine life nearby — reduce speed")
            ethical_score -= 0.3
        if ctx.get("protected_area", False) and any(w in action_lower for w in ("anchor", "fish", "trawl")):
            violations.append("Cannot perform this action in a protected area")
            ethical_score -= 0.5
        ethical_score = max(0.0, min(1.0, ethical_score))
        is_safe = len(violations) == 0 and trust_ok
        confidence = 0.7 if is_safe else 0.9
        if not violations:
            violations = ("none",)
        return SafetyAssessment(
            agent_name="ethos",
            verdict=DecisionVerdict.APPROVE if is_safe else DecisionVerdict.REJECT,
            confidence=round(confidence, 4), score=round(ethical_score, 4),
            is_safe=is_safe, violations=tuple(violations),
            trust_required=trust_required, ethical_score=round(ethical_score, 4),
            reasons=tuple(violations),
        )

    def _get_trust_requirement(self, action: str) -> int:
        for kw, level in self.TRUST_REQUIREMENTS.items():
            if kw in action:
                return level
        return 1

    @staticmethod
    def _point_in_zone(point: tuple[float, float], zone: dict[str, Any]) -> bool:
        lat, lon = point
        b = zone.get("bounds", {})
        return (b.get("south", -90) <= lat <= b.get("north", 90) and
                b.get("west", -180) <= lon <= b.get("east", 180))
