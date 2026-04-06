"""Tests for diagnosis module — 40 tests."""

import pytest

from jetson.self_healing.diagnosis import (
    CausalEdge,
    CausalGraph,
    Diagnosis,
    DiagnosticRule,
    RootCauseAnalyzer,
)
from jetson.self_healing.fault_detector import FaultEvent, FaultSeverity


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def analyzer():
    return RootCauseAnalyzer()


@pytest.fixture
def sample_fault():
    return FaultEvent(
        id="fault-001",
        component="gps_module",
        fault_type="signal_loss",
        severity=FaultSeverity.HIGH,
        symptoms=["no_signal", "low_snr"],
        context={"snr_db": -5.0, "satellites_visible": 0},
    )


@pytest.fixture
def sample_rule():
    return DiagnosticRule(
        symptoms_pattern={"component": "gps_module", "fault_type": "signal_loss"},
        probable_cause="antenna_obstruction",
        confidence=0.85,
        fix_recommendation="Check antenna mounting and clear line of sight",
        priority=10,
    )


# ── Diagnosis dataclass ──────────────────────────────────────────────────

class TestDiagnosis:
    def test_fields(self):
        d = Diagnosis(
            fault_id="f-1",
            root_cause="overheating",
            confidence=0.9,
            contributing_factors=["high_temp", "fan_failure"],
            recommended_fix="Replace fan",
        )
        assert d.fault_id == "f-1"
        assert d.root_cause == "overheating"
        assert d.confidence == 0.9
        assert len(d.contributing_factors) == 2
        assert d.recommended_fix == "Replace fan"

    def test_defaults(self):
        d = Diagnosis(fault_id="f-1", root_cause="x", confidence=0.5)
        assert d.contributing_factors == []
        assert d.recommended_fix == ""
        assert d.diagnosis_time > 0


# ── DiagnosticRule dataclass ─────────────────────────────────────────────

class TestDiagnosticRule:
    def test_fields(self):
        r = DiagnosticRule(
            symptoms_pattern={"key": "val"},
            probable_cause="cause1",
            confidence=0.8,
            fix_recommendation="fix it",
            priority=5,
        )
        assert r.confidence == 0.8
        assert r.priority == 5

    def test_defaults(self):
        r = DiagnosticRule(symptoms_pattern={}, probable_cause="c", confidence=0.5)
        assert r.fix_recommendation == ""
        assert r.priority == 0


# ── CausalGraph ───────────────────────────────────────────────────────────

class TestCausalGraph:
    def test_add_edge(self):
        g = CausalGraph()
        g.add_edge("A", "B", 1.0)
        assert "A" in g.nodes
        assert "B" in g.nodes
        assert len(g.edges) == 1

    def test_add_edge_existing_nodes(self):
        g = CausalGraph()
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        assert len(g.nodes) == 3
        assert len(g.edges) == 2

    def test_roots(self):
        g = CausalGraph()
        g.add_edge("root", "child1")
        g.add_edge("root", "child2")
        assert g.roots() == ["root"]

    def test_leaves(self):
        g = CausalGraph()
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        leaves = g.leaves()
        assert "B" in leaves
        assert "C" in leaves
        assert "A" not in leaves

    def test_empty_graph(self):
        g = CausalGraph()
        assert g.nodes == []
        assert g.edges == []
        assert g.roots() == []
        assert g.leaves() == []


# ── RootCauseAnalyzer ────────────────────────────────────────────────────

class TestRootCauseAnalyzer:
    def test_add_rule(self, analyzer, sample_rule):
        analyzer.add_rule(sample_rule)
        assert len(analyzer.rules) == 1

    def test_add_rule_sorted_by_priority(self, analyzer):
        r1 = DiagnosticRule({"a": 1}, "cause_a", 0.5, priority=1)
        r2 = DiagnosticRule({"b": 2}, "cause_b", 0.5, priority=5)
        r3 = DiagnosticRule({"c": 3}, "cause_c", 0.5, priority=3)
        analyzer.add_rule(r1)
        analyzer.add_rule(r2)
        analyzer.add_rule(r3)
        assert analyzer.rules[0].priority == 5
        assert analyzer.rules[1].priority == 3
        assert analyzer.rules[2].priority == 1

    def test_diagnose_with_matching_rule(self, analyzer, sample_fault, sample_rule):
        analyzer.add_rule(sample_rule)
        diag = analyzer.diagnose(sample_fault)
        assert diag.root_cause == "antenna_obstruction"
        assert diag.confidence > 0

    def test_diagnose_without_rules(self, analyzer, sample_fault):
        diag = analyzer.diagnose(sample_fault)
        assert diag.confidence == 0.3
        assert "unknown" in diag.root_cause

    def test_diagnose_records_history(self, analyzer, sample_fault):
        analyzer.diagnose(sample_fault)
        assert len(analyzer.history) == 1

    def test_diagnose_with_system_state(self, analyzer):
        rule = DiagnosticRule(
            symptoms_pattern={"component": "motor", "fault_type": "overheat"},
            probable_cause="cooling_failure",
            confidence=0.9,
            priority=5,
        )
        analyzer.add_rule(rule)
        fault = FaultEvent(component="motor", fault_type="overheat")
        state = {"temperature": 95.0}
        diag = analyzer.diagnose(fault, state)
        assert diag.root_cause == "cooling_failure"

    def test_compute_causal_graph(self, analyzer):
        symptoms = ["no_signal", "low_snr", "timeout"]
        graph = analyzer.compute_causal_graph(symptoms)
        assert "root_cause" in graph.nodes
        assert len(graph.roots()) == 1

    def test_compute_causal_graph_empty(self, analyzer):
        graph = analyzer.compute_causal_graph([])
        assert graph.nodes == []
        assert graph.edges == []

    def test_compute_causal_graph_single_symptom(self, analyzer):
        graph = analyzer.compute_causal_graph(["symptom1"])
        assert "symptom1" in graph.nodes

    def test_rank_hypotheses(self, analyzer):
        hypotheses = [
            {"cause": "A", "base_confidence": 0.8},
            {"cause": "B", "base_confidence": 0.6},
        ]
        evidence = {"cause": "A", "extra": "data"}
        ranked = analyzer.rank_hypotheses(hypotheses, evidence)
        assert len(ranked) == 2
        assert "_score" in ranked[0]
        assert ranked[0]["_score"] >= ranked[1]["_score"]

    def test_rank_hypotheses_empty_evidence(self, analyzer):
        hypotheses = [{"cause": "A", "base_confidence": 0.7}]
        ranked = analyzer.rank_hypotheses(hypotheses, {})
        assert ranked[0]["_score"] == 0.7

    def test_rank_hypotheses_empty_list(self, analyzer):
        ranked = analyzer.rank_hypotheses([], {"x": 1})
        assert ranked == []

    def test_compute_confidence_exact_match(self, analyzer):
        hypothesis = {"component": "gps", "base_confidence": 0.5}
        evidence = {"component": "gps"}
        conf = analyzer.compute_confidence(hypothesis, evidence)
        assert conf > 0.5

    def test_compute_confidence_no_match(self, analyzer):
        hypothesis = {"component": "gps", "base_confidence": 0.5}
        evidence = {"component": "motor"}
        conf = analyzer.compute_confidence(hypothesis, evidence)
        assert conf < 0.5

    def test_compute_confidence_range_match(self, analyzer):
        hypothesis = {"temperature": {"min": 80, "max": 120}, "base_confidence": 0.5}
        evidence = {"temperature": 95.0}
        conf = analyzer.compute_confidence(hypothesis, evidence)
        assert conf > 0.5

    def test_compute_confidence_list_match(self, analyzer):
        hypothesis = {"status": ["ok", "degraded"], "base_confidence": 0.5}
        evidence = {"status": "ok"}
        conf = analyzer.compute_confidence(hypothesis, evidence)
        assert conf > 0.5

    def test_compute_confidence_empty_evidence(self, analyzer):
        hypothesis = {"key": "val", "base_confidence": 0.7}
        conf = analyzer.compute_confidence(hypothesis, {})
        assert conf == 0.7

    def test_learn_from_resolution_new_rule(self, analyzer, sample_fault):
        diag = Diagnosis(fault_id="f-1", root_cause="wrong", confidence=0.5, recommended_fix="fix")
        new_rules = analyzer.learn_from_resolution(sample_fault, diag, "actual_cause")
        assert len(new_rules) == 1
        assert new_rules[0].probable_cause == "actual_cause"

    def test_learn_from_resolution_boost_existing(self, analyzer, sample_rule):
        original_conf = sample_rule.confidence
        analyzer.add_rule(sample_rule)
        fault = FaultEvent(component="gps_module", fault_type="signal_loss", symptoms=[])
        diag = Diagnosis(fault_id="f-1", root_cause="antenna_obstruction", confidence=0.8)
        new_rules = analyzer.learn_from_resolution(fault, diag, "antenna_obstruction")
        assert len(new_rules) == 0
        assert analyzer.rules[0].confidence > original_conf

    def test_learn_from_correction_penalty(self, analyzer, sample_rule):
        original_conf = sample_rule.confidence
        analyzer.add_rule(sample_rule)
        fault = FaultEvent(component="gps_module", fault_type="signal_loss", symptoms=[])
        diag = Diagnosis(fault_id="f-1", root_cause="antenna_obstruction", confidence=0.8)
        analyzer.learn_from_resolution(fault, diag, "real_cause")
        # Original rule should have reduced confidence
        found = [r for r in analyzer.rules if r.probable_cause == "antenna_obstruction"]
        assert found[0].confidence < original_conf

    def test_rules_property(self, analyzer, sample_rule):
        analyzer.add_rule(sample_rule)
        rules = analyzer.rules
        assert len(rules) == 1
        # Should be a copy
        rules.append(DiagnosticRule({}, "dummy", 0.1))
        assert len(analyzer.rules) == 1

    def test_history_property(self, analyzer, sample_fault):
        analyzer.diagnose(sample_fault)
        h = analyzer.history
        assert len(h) == 1
        h.append({})
        assert len(analyzer.history) == 1

    def test_diagnose_multiple_symptoms_rule(self, analyzer):
        rule = DiagnosticRule(
            symptoms_pattern={
                "component": "sensor",
                "fault_type": "timeout",
            },
            probable_cause="bus_congestion",
            confidence=0.75,
            priority=8,
        )
        analyzer.add_rule(rule)
        fault = FaultEvent(component="sensor", fault_type="timeout")
        diag = analyzer.diagnose(fault)
        assert diag.root_cause == "bus_congestion"

    def test_diagnose_partial_rule_match(self, analyzer):
        rule = DiagnosticRule(
            symptoms_pattern={"component": "gps", "fault_type": "drift"},
            probable_cause="clock_skew",
            confidence=0.7,
            priority=3,
        )
        analyzer.add_rule(rule)
        fault = FaultEvent(component="gps", fault_type="drift")
        diag = analyzer.diagnose(fault)
        assert diag.root_cause == "clock_skew"

    def test_contributes_factors_include_context(self, analyzer):
        fault = FaultEvent(
            component="motor",
            fault_type="stall",
            symptoms=[],
            context={"rpm": 0, "current": 15.2},
        )
        diag = analyzer.diagnose(fault)
        has_ctx = any("ctx." in f for f in diag.contributing_factors)
        assert has_ctx

    def test_learn_multiple_times(self, analyzer):
        r = DiagnosticRule({"component": "x"}, "cause_x", 0.5, priority=1)
        analyzer.add_rule(r)
        fault = FaultEvent(component="x", symptoms=[])
        diag = Diagnosis(fault_id="f", root_cause="cause_x", confidence=0.5)
        for _ in range(5):
            analyzer.learn_from_resolution(fault, diag, "cause_x")
        assert analyzer.rules[0].confidence <= 1.0
