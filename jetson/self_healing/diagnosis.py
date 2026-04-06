"""Root cause analysis module — diagnose faults and identify underlying causes.

Pure Python, zero external dependencies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class Diagnosis:
    """Result of a root-cause analysis for a fault event."""
    fault_id: str
    root_cause: str
    confidence: float  # 0.0 – 1.0
    contributing_factors: List[str] = field(default_factory=list)
    recommended_fix: str = ""
    diagnosis_time: float = field(default_factory=time.time)


@dataclass
class DiagnosticRule:
    """A heuristic rule mapping symptom patterns to probable causes."""
    symptoms_pattern: Dict[str, Any]  # key -> expected value or range
    probable_cause: str
    confidence: float  # baseline confidence 0.0 – 1.0
    fix_recommendation: str = ""
    priority: int = 0  # higher = checked first


# ── Causal graph helpers ─────────────────────────────────────────────────

@dataclass
class CausalNode:
    label: str
    weight: float = 1.0
    children: List[CausalNode] = field(default_factory=list)


@dataclass
class CausalEdge:
    source: str
    target: str
    weight: float = 1.0


@dataclass
class CausalGraph:
    nodes: List[str] = field(default_factory=list)
    edges: List[CausalEdge] = field(default_factory=list)

    def add_edge(self, source: str, target: str, weight: float = 1.0) -> None:
        if source not in self.nodes:
            self.nodes.append(source)
        if target not in self.nodes:
            self.nodes.append(target)
        self.edges.append(CausalEdge(source, target, weight))

    def roots(self) -> List[str]:
        targets = {e.target for e in self.edges}
        return [n for n in self.nodes if n not in targets]

    def leaves(self) -> List[str]:
        sources = {e.source for e in self.edges}
        return [n for n in self.nodes if n not in sources]


# ── RootCauseAnalyzer ────────────────────────────────────────────────────

class RootCauseAnalyzer:
    """Analyses fault events and system state to produce diagnoses."""

    def __init__(self) -> None:
        self._rules: List[DiagnosticRule] = []
        self._history: List[Dict[str, Any]] = []  # past fault→diagnosis→resolution triples

    # ── rule management ──

    def add_rule(self, rule: DiagnosticRule) -> None:
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    @property
    def rules(self) -> List[DiagnosticRule]:
        return list(self._rules)

    # ── diagnosis ──

    def diagnose(self, fault_event: Any, system_state: Optional[Dict[str, Any]] = None) -> Diagnosis:
        """Produce a Diagnosis for *fault_event* using registered rules + heuristics."""
        system_state = system_state or {}

        # Try each rule
        best_match: Optional[DiagnosticRule] = None
        best_conf = 0.0
        matched_count = 0

        for rule in self._rules:
            score, matches = self._match_rule(rule, fault_event, system_state)
            if matches > 0 and score > best_conf:
                best_match = rule
                best_conf = score
                matched_count = matches

        if best_match is not None:
            contributing = self._extract_contributing(fault_event, system_state)
            diag = Diagnosis(
                fault_id=getattr(fault_event, "id", "unknown"),
                root_cause=best_match.probable_cause,
                confidence=min(best_conf, 1.0),
                contributing_factors=contributing,
                recommended_fix=best_match.fix_recommendation,
            )
        else:
            # Fallback heuristic
            root_cause = self._heuristic_cause(fault_event, system_state)
            fix = self._heuristic_fix(fault_event)
            contributing = self._extract_contributing(fault_event, system_state)
            diag = Diagnosis(
                fault_id=getattr(fault_event, "id", "unknown"),
                root_cause=root_cause,
                confidence=0.3,
                contributing_factors=contributing,
                recommended_fix=fix,
            )

        self._history.append({
            "fault_id": diag.fault_id,
            "root_cause": diag.root_cause,
            "confidence": diag.confidence,
            "fault_component": getattr(fault_event, "component", ""),
            "fault_type": getattr(fault_event, "fault_type", ""),
        })
        return diag

    # ── causal graph ──

    def compute_causal_graph(self, symptoms: List[str]) -> CausalGraph:
        """Build a causal graph from a list of symptom strings."""
        graph = CausalGraph()
        if not symptoms:
            return graph

        # Simple chain: root_cause → intermediate → symptoms
        root = "root_cause"
        graph.nodes.append(root)

        intermediates: List[str] = []
        for i, symptom in enumerate(symptoms):
            intermediate = f"factor_{i}"
            intermediates.append(intermediate)
            graph.nodes.append(intermediate)
            graph.add_edge(root, intermediate, weight=1.0 / (i + 1))
            graph.add_edge(intermediate, symptom, weight=1.0)

        # Cross-link intermediates for correlation
        for i in range(len(intermediates) - 1):
            graph.add_edge(intermediates[i], intermediates[i + 1], weight=0.5)

        return graph

    # ── hypothesis ranking ──

    def rank_hypotheses(self, hypotheses: List[Dict[str, Any]],
                        evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank hypotheses by confidence given evidence. Returns sorted list with '_score'."""
        scored: List[Dict[str, Any]] = []
        for h in hypotheses:
            conf = self.compute_confidence(h, evidence)
            entry = dict(h)
            entry["_score"] = conf
            scored.append(entry)
        scored.sort(key=lambda x: x["_score"], reverse=True)
        return scored

    def compute_confidence(self, hypothesis: Dict[str, Any],
                           evidence: Dict[str, Any]) -> float:
        """Compute a confidence score 0.0–1.0 for a hypothesis given evidence."""
        if not evidence:
            return hypothesis.get("base_confidence", 0.5)

        matches = 0
        total_checks = 0
        for key, expected in hypothesis.items():
            if key.startswith("_"):
                continue
            if key in evidence:
                total_checks += 1
                if isinstance(expected, (list, tuple)):
                    if evidence[key] in expected:
                        matches += 1
                elif isinstance(expected, dict):
                    lo = expected.get("min", float("-inf"))
                    hi = expected.get("max", float("inf"))
                    if lo <= evidence[key] <= hi:
                        matches += 1
                else:
                    if evidence[key] == expected:
                        matches += 1

        if total_checks == 0:
            return hypothesis.get("base_confidence", 0.5)

        match_ratio = matches / total_checks
        base = hypothesis.get("base_confidence", 0.5)
        return min(1.0, base * 0.4 + match_ratio * 0.6)

    # ── learning ──

    def learn_from_resolution(self, fault: Any, diagnosis: Any,
                              actual_cause: str) -> List[DiagnosticRule]:
        """Update internal rules based on resolved fault. Returns new/updated rules."""
        fault_symptoms = getattr(fault, "symptoms", [])
        fault_component = getattr(fault, "component", "")
        diag_confidence = getattr(diagnosis, "confidence", 0.0)

        # If the diagnosis was correct, boost existing rule or create new one
        matched = False
        for rule in self._rules:
            if rule.probable_cause == actual_cause:
                # Boost confidence slightly
                rule.confidence = min(1.0, rule.confidence + 0.05)
                rule.priority += 1
                matched = True

        new_rules: List[DiagnosticRule] = []
        if not matched:
            pattern: Dict[str, Any] = {"component": fault_component}
            for s in fault_symptoms:
                pattern[s] = True

            new_rule = DiagnosticRule(
                symptoms_pattern=pattern,
                probable_cause=actual_cause,
                confidence=0.5,
                fix_recommendation=getattr(diagnosis, "recommended_fix", ""),
                priority=1,
            )
            self._rules.append(new_rule)
            self._rules.sort(key=lambda r: r.priority, reverse=True)
            new_rules.append(new_rule)

        # Also adjust if diagnosis was wrong
        if getattr(diagnosis, "root_cause", "") != actual_cause:
            for rule in self._rules:
                if rule.probable_cause == getattr(diagnosis, "root_cause", ""):
                    rule.confidence = max(0.0, rule.confidence - 0.1)
                    rule.priority = max(0, rule.priority - 1)

        return new_rules

    # ── internal helpers ──────────────────────────────────────────────────

    def _match_rule(self, rule: DiagnosticRule, fault: Any,
                    state: Dict[str, Any]) -> Tuple[float, int]:
        """Return (score, match_count) for a rule against fault + state."""
        matches = 0
        total = len(rule.symptoms_pattern)

        for key, expected in rule.symptoms_pattern.items():
            actual = None
            if hasattr(fault, key):
                actual = getattr(fault, key)
            elif key in state:
                actual = state[key]

            if actual is not None:
                if isinstance(expected, (list, tuple)):
                    if actual in expected:
                        matches += 1
                elif isinstance(expected, dict):
                    lo = expected.get("min", float("-inf"))
                    hi = expected.get("max", float("inf"))
                    if lo <= actual <= hi:
                        matches += 1
                elif callable(expected):
                    if expected(actual):
                        matches += 1
                else:
                    if actual == expected:
                        matches += 1

        if total == 0:
            return (rule.confidence, 0)
        ratio = matches / total
        score = rule.confidence * ratio
        return (score, matches)

    def _extract_contributing(self, fault: Any, state: Dict[str, Any]) -> List[str]:
        factors: List[str] = []
        comp = getattr(fault, "component", "")
        if comp:
            factors.append(f"component={comp}")
        ft = getattr(fault, "fault_type", "")
        if ft:
            factors.append(f"fault_type={ft}")
        for k, v in (getattr(fault, "context", {}) or {}).items():
            factors.append(f"ctx.{k}={v}")
        for k in state:
            factors.append(f"state.{k}=present")
        return factors

    def _heuristic_cause(self, fault: Any, state: Dict[str, Any]) -> str:
        comp = getattr(fault, "component", "unknown")
        ft = getattr(fault, "fault_type", "unknown")
        return f"unknown_cause_in_{comp}_{ft}"

    def _heuristic_fix(self, fault: Any) -> str:
        comp = getattr(fault, "component", "component")
        return f"Restart or reconfigure {comp}. Investigate logs for details."

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)
