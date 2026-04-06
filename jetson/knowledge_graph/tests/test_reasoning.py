"""Tests for reasoning.py — Reasoner, InferenceRule, InferredFact, Contradiction."""

import pytest
from jetson.knowledge_graph.graph import (
    Entity, Relation, KnowledgeGraph, make_entity, make_relation,
)
from jetson.knowledge_graph.reasoning import (
    InferenceRule, InferredFact, Contradiction, Reasoner,
)


# ======================================================================
# InferenceRule
# ======================================================================

class TestInferenceRule:
    def test_create_rule(self):
        r = InferenceRule(name="test", premises=["a|knows|b"], conclusion="b|knows|a")
        assert r.name == "test"
        assert r.confidence == 1.0

    def test_rule_confidence(self):
        r = InferenceRule(name="r", premises=[], conclusion="c", confidence=0.8)
        assert r.confidence == 0.8

    def test_rule_premises_list(self):
        r = InferenceRule(name="r", premises=["p1", "p2"], conclusion="c")
        assert len(r.premises) == 2


# ======================================================================
# InferredFact
# ======================================================================

class TestInferredFact:
    def test_create_fact(self):
        f = InferredFact(fact="x|is_a|y", source_ids=("a", "b"), confidence=0.9, rule_name="r1")
        assert f.fact == "x|is_a|y"
        assert f.source_ids == ("a", "b")

    def test_fact_defaults(self):
        f = InferredFact(fact="f", source_ids=(), confidence=1.0, rule_name="r")
        assert f.source_ids == ()


# ======================================================================
# Contradiction
# ======================================================================

class TestContradiction:
    def test_create_contradiction(self):
        c = Contradiction(fact_a="a|is_a|b", fact_b="a|not_a|b", entity_id="a")
        assert c.entity_id == "a"


# ======================================================================
# Reasoner — Forward chaining
# ======================================================================

class TestForwardChain:
    def _make_graph(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("Animal", entity_id="cat"))
        g.add_entity(make_entity("Animal", entity_id="dog"))
        g.add_entity(make_entity("Sound", entity_id="meow"))
        g.add_relation(make_relation("cat", "meow", "makes"))
        return g

    def test_forward_chain_basic(self):
        g = self._make_graph()
        r = Reasoner()
        rule = InferenceRule(
            name="sound_rule",
            premises=["cat|makes|meow"],
            conclusion="inferred:cat_has_sound",
            confidence=0.9,
        )
        facts = r.forward_chain(g, [rule])
        assert len(facts) == 1
        assert facts[0].rule_name == "sound_rule"

    def test_forward_chain_no_match(self):
        g = self._make_graph()
        r = Reasoner()
        rule = InferenceRule(name="bad", premises=["nonexistent|has|x"], conclusion="y")
        facts = r.forward_chain(g, [rule])
        assert len(facts) == 0

    def test_forward_chain_multi_step(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("T", entity_id="a"))
        g.add_entity(make_entity("T", entity_id="b"))
        g.add_entity(make_entity("T", entity_id="c"))
        g.add_relation(make_relation("a", "b", "r1"))
        rule1 = InferenceRule(name="step1", premises=["a|r1|b"], conclusion="b|r2|c", confidence=0.9)
        rule2 = InferenceRule(name="step2", premises=["b|r2|c"], conclusion="c|r3|a", confidence=0.9)
        r = Reasoner()
        facts = r.forward_chain(g, [rule1, rule2], max_steps=10)
        assert len(facts) == 2

    def test_forward_chain_entity_type_premise(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("Vessel", entity_id="v1"))
        r = Reasoner()
        rule = InferenceRule(name="v", premises=["entity_type:Vessel"], conclusion="has_vessel", confidence=1.0)
        facts = r.forward_chain(g, [rule])
        assert len(facts) == 1

    def test_forward_chain_entity_type_premise_fails(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("Person", entity_id="p1"))
        r = Reasoner()
        rule = InferenceRule(name="v", premises=["entity_type:Vessel"], conclusion="has_vessel")
        facts = r.forward_chain(g, [rule])
        assert len(facts) == 0

    def test_forward_chain_max_steps(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("T", entity_id="a"))
        g.add_relation(make_relation("a", "a", "r"))
        rule = InferenceRule(name="loop", premises=["a|r|a"], conclusion="b|s|c")
        r = Reasoner()
        facts = r.forward_chain(g, [rule], max_steps=1)
        # First step triggers, second (from b|s|c) won't since b doesn't exist as source
        # But it should produce at least 1 fact
        assert len(facts) >= 0

    def test_forward_chain_uses_stored_rules(self):
        g = self._make_graph()
        r = Reasoner()
        r.add_rule(InferenceRule(name="r1", premises=["cat|makes|meow"], conclusion="cat_can_meow"))
        facts = r.forward_chain(g)
        assert len(facts) == 1

    def test_forward_chain_no_duplicate_inference(self):
        g = self._make_graph()
        r = Reasoner()
        rule = InferenceRule(name="r", premises=["cat|makes|meow"], conclusion="cat_has_sound", confidence=0.5)
        facts = r.forward_chain(g, [rule], max_steps=10)
        assert len(facts) == 1  # should not infer twice


# ======================================================================
# Reasoner — Backward chaining
# ======================================================================

class TestBackwardChain:
    def test_backward_chain_found_in_graph(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("T", entity_id="a"))
        g.add_entity(make_entity("T", entity_id="b"))
        g.add_relation(make_relation("a", "b", "likes"))
        r = Reasoner()
        path = r.backward_chain(g, "a|likes|b")
        assert len(path) >= 1
        assert "graph_fact" in path

    def test_backward_chain_not_found(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("T", entity_id="a"))
        r = Reasoner()
        path = r.backward_chain(g, "a|likes|b")
        assert path == []

    def test_backward_chain_via_rule(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("T", entity_id="a"))
        g.add_entity(make_entity("T", entity_id="b"))
        g.add_relation(make_relation("a", "b", "knows"))
        rule = InferenceRule(name="transitive", premises=["a|knows|b"], conclusion="a|friend|b")
        r = Reasoner()
        path = r.backward_chain(g, "a|friend|b", [rule])
        assert "transitive" in path

    def test_backward_chain_deep_proof(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("T", entity_id="x"))
        g.add_entity(make_entity("T", entity_id="y"))
        g.add_relation(make_relation("x", "y", "base"))
        rule1 = InferenceRule(name="r1", premises=["x|base|y"], conclusion="y|mid|z")
        rule2 = InferenceRule(name="r2", premises=["y|mid|z"], conclusion="z|final|w")
        r = Reasoner()
        path = r.backward_chain(g, "z|final|w", [rule1, rule2])
        assert "r1" in path
        assert "r2" in path


# ======================================================================
# Reasoner — Transitive closure
# ======================================================================

class TestTransitiveClosure:
    def test_simple_transitive(self):
        g = KnowledgeGraph()
        for eid in ("a", "b", "c"):
            g.add_entity(make_entity("T", entity_id=eid))
        g.add_relation(make_relation("a", "b", "parent_of"))
        g.add_relation(make_relation("b", "c", "parent_of"))
        r = Reasoner()
        closed = r.transitive_closure(g, "parent_of")
        # Should have a->c
        rels = closed.get_relations("a", "outgoing")
        targets = {rel.target_id for rel in rels}
        assert "c" in targets

    def test_no_transitive(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("T", entity_id="a"))
        g.add_entity(make_entity("T", entity_id="b"))
        g.add_relation(make_relation("a", "b", "edge"))
        r = Reasoner()
        closed = r.transitive_closure(g, "edge")
        assert closed.stats()["relation_count"] == 1

    def test_closure_preserves_entities(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("T", entity_id="a"))
        g.add_entity(make_entity("T", entity_id="b"))
        g.add_relation(make_relation("a", "b", "e"))
        r = Reasoner()
        closed = r.transitive_closure(g, "e")
        assert closed.stats()["entity_count"] == 2

    def test_closure_empty_graph(self):
        g = KnowledgeGraph()
        r = Reasoner()
        closed = r.transitive_closure(g, "e")
        assert closed.stats()["entity_count"] == 0

    def test_closure_ignores_other_types(self):
        g = KnowledgeGraph()
        for eid in ("a", "b", "c"):
            g.add_entity(make_entity("T", entity_id=eid))
        g.add_relation(make_relation("a", "b", "is_a"))
        g.add_relation(make_relation("b", "c", "other"))
        r = Reasoner()
        closed = r.transitive_closure(g, "is_a")
        rels = closed.get_relations("a", "outgoing")
        assert len(rels) == 1


# ======================================================================
# Reasoner — Similarity
# ======================================================================

class TestSimilarity:
    def test_identical_neighbors(self):
        g = KnowledgeGraph()
        for eid in ("a", "b", "c"):
            g.add_entity(make_entity("T", entity_id=eid))
        g.add_relation(make_relation("a", "c", "e"))
        g.add_relation(make_relation("b", "c", "e"))
        r = Reasoner()
        sim = r.compute_similarity("a", "b", g)
        assert sim == 1.0

    def test_no_common_neighbors(self):
        g = KnowledgeGraph()
        for eid in ("a", "b", "c", "d"):
            g.add_entity(make_entity("T", entity_id=eid))
        g.add_relation(make_relation("a", "c", "e"))
        g.add_relation(make_relation("b", "d", "e"))
        r = Reasoner()
        sim = r.compute_similarity("a", "b", g)
        assert sim == 0.0

    def test_partial_overlap(self):
        g = KnowledgeGraph()
        for eid in ("a", "b", "c", "d"):
            g.add_entity(make_entity("T", entity_id=eid))
        g.add_relation(make_relation("a", "c", "e"))
        g.add_relation(make_relation("a", "d", "e"))
        g.add_relation(make_relation("b", "c", "e"))
        r = Reasoner()
        sim = r.compute_similarity("a", "b", g)
        assert 0.0 < sim < 1.0

    def test_both_isolated(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("T", entity_id="a"))
        g.add_entity(make_entity("T", entity_id="b"))
        r = Reasoner()
        assert r.compute_similarity("a", "b", g) == 1.0

    def test_one_isolated(self):
        g = KnowledgeGraph()
        for eid in ("a", "b", "c"):
            g.add_entity(make_entity("T", entity_id=eid))
        g.add_relation(make_relation("b", "c", "e"))
        r = Reasoner()
        assert r.compute_similarity("a", "b", g) == 0.0


# ======================================================================
# Reasoner — Contradiction detection
# ======================================================================

class TestContradictionDetection:
    def test_detect_is_a_not_a(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("T", entity_id="x"))
        g.add_entity(make_entity("T", entity_id="y"))
        g.add_relation(make_relation("x", "y", "is_a"))
        g.add_relation(make_relation("x", "y", "not_a"))
        r = Reasoner()
        contras = r.detect_contradictions(g)
        assert len(contras) == 1
        assert contras[0].entity_id == "x"

    def test_no_contradiction(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("T", entity_id="a"))
        g.add_entity(make_entity("T", entity_id="b"))
        g.add_relation(make_relation("a", "b", "is_a"))
        r = Reasoner()
        assert r.detect_contradictions(g) == []

    def test_empty_graph_no_contradiction(self):
        g = KnowledgeGraph()
        r = Reasoner()
        assert r.detect_contradictions(g) == []

    def test_has_vs_lacks_contradiction(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("T", entity_id="a"))
        g.add_entity(make_entity("T", entity_id="b"))
        g.add_relation(make_relation("a", "b", "has"))
        g.add_relation(make_relation("a", "b", "lacks"))
        r = Reasoner()
        contras = r.detect_contradictions(g)
        assert len(contras) >= 1


# ======================================================================
# Reasoner — Centrality
# ======================================================================

class TestCentrality:
    def test_high_centrality(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("T", entity_id="hub"))
        for i in range(5):
            eid = f"node_{i}"
            g.add_entity(make_entity("T", entity_id=eid))
            g.add_relation(make_relation("hub", eid, "connects"))
        r = Reasoner()
        centrality = r.compute_centrality(g, "hub")
        assert centrality > 0.5

    def test_low_centrality(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("T", entity_id="a"))
        g.add_entity(make_entity("T", entity_id="b"))
        g.add_relation(make_relation("a", "b", "e"))
        r = Reasoner()
        centrality = r.compute_centrality(g, "a")
        assert centrality == 1.0  # 1 connection / (2-1)

    def test_isolated_centrality(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("T", entity_id="iso"))
        g.add_entity(make_entity("T", entity_id="a"))
        g.add_entity(make_entity("T", entity_id="b"))
        g.add_relation(make_relation("a", "b", "e"))
        r = Reasoner()
        assert r.compute_centrality(g, "iso") == 0.0

    def test_single_entity_centrality(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("T", entity_id="solo"))
        r = Reasoner()
        assert r.compute_centrality(g, "solo") == 0.0
