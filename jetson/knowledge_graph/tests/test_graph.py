"""Tests for graph.py — KnowledgeGraph, Entity, Relation."""

import pytest
from jetson.knowledge_graph.graph import (
    Entity, Relation, KnowledgeGraph, make_entity, make_relation,
)


# ======================================================================
# Entity
# ======================================================================

class TestEntity:
    def test_create_basic_entity(self):
        e = Entity(id="e1", type="Person")
        assert e.id == "e1"
        assert e.type == "Person"

    def test_entity_with_properties(self):
        e = Entity(id="e2", type="Vessel", properties={"name": "Titan", "speed": 12.5})
        assert e.properties["name"] == "Titan"
        assert e.properties["speed"] == 12.5

    def test_entity_with_labels(self):
        e = Entity(id="e3", type="Sensor", labels=("optical", "primary"))
        assert e.labels == ("optical", "primary")

    def test_entity_defaults(self):
        e = Entity(id="e4", type="Thing")
        assert e.properties == {}
        assert e.labels == ()

    def test_entity_frozen(self):
        e = Entity(id="e5", type="X")
        with pytest.raises(AttributeError):
            e.id = "new_id"

    def test_entity_equality(self):
        e1 = Entity(id="same", type="A")
        e2 = Entity(id="same", type="B")
        assert e1 == e2

    def test_entity_hash(self):
        e1 = Entity(id="h1", type="A")
        e2 = Entity(id="h1", type="B")
        assert hash(e1) == hash(e2)

    def test_entity_not_equal_to_non_entity(self):
        e = Entity(id="x", type="X")
        assert e != "x"


# ======================================================================
# Relation
# ======================================================================

class TestRelation:
    def test_create_basic_relation(self):
        r = Relation(id="r1", source_id="a", target_id="b", type="knows")
        assert r.source_id == "a"
        assert r.target_id == "b"

    def test_relation_weight(self):
        r = Relation(id="r2", source_id="a", target_id="b", type="links", weight=0.7)
        assert r.weight == 0.7

    def test_relation_default_weight(self):
        r = Relation(id="r3", source_id="a", target_id="b", type="links")
        assert r.weight == 1.0

    def test_relation_properties(self):
        r = Relation(id="r4", source_id="a", target_id="b", type="has", properties={"since": 2020})
        assert r.properties["since"] == 2020

    def test_relation_frozen(self):
        r = Relation(id="r5", source_id="a", target_id="b", type="frozen")
        with pytest.raises(AttributeError):
            r.weight = 0.5

    def test_relation_equality(self):
        r1 = Relation(id="same", source_id="a", target_id="b", type="x")
        r2 = Relation(id="same", source_id="c", target_id="d", type="y")
        assert r1 == r2


# ======================================================================
# KnowledgeGraph
# ======================================================================

class TestKnowledgeGraphInit:
    def test_empty_graph(self):
        g = KnowledgeGraph()
        assert g.stats()["entity_count"] == 0
        assert g.stats()["relation_count"] == 0

    def test_stats_keys(self):
        g = KnowledgeGraph()
        s = g.stats()
        assert "entity_count" in s
        assert "relation_count" in s
        assert "entity_types" in s


class TestKnowledgeGraphAdd:
    def test_add_entity(self):
        g = KnowledgeGraph()
        e = Entity(id="e1", type="Person")
        g.add_entity(e)
        assert g.stats()["entity_count"] == 1

    def test_add_entity_retrievable(self):
        g = KnowledgeGraph()
        e = Entity(id="e1", type="Person", properties={"name": "Alice"})
        g.add_entity(e)
        assert g.get_entity("e1").properties["name"] == "Alice"

    def test_add_entity_replace(self):
        g = KnowledgeGraph()
        g.add_entity(Entity(id="e1", type="A"))
        g.add_entity(Entity(id="e1", type="B"))
        assert g.get_entity("e1").type == "B"

    def test_add_relation(self):
        g = KnowledgeGraph()
        g.add_entity(Entity(id="a", type="T"))
        g.add_entity(Entity(id="b", type="T"))
        g.add_relation(Relation(id="r1", source_id="a", target_id="b", type="links"))
        assert g.stats()["relation_count"] == 1

    def test_add_relation_returns_relation(self):
        g = KnowledgeGraph()
        r = Relation(id="r1", source_id="a", target_id="b", type="links")
        result = g.add_relation(r)
        assert result.id == "r1"


class TestKnowledgeGraphGetRelations:
    def test_get_outgoing(self):
        g = KnowledgeGraph()
        g.add_entity(Entity(id="a", type="T"))
        g.add_entity(Entity(id="b", type="T"))
        g.add_relation(Relation(id="r1", source_id="a", target_id="b", type="knows"))
        rels = g.get_relations("a", "outgoing")
        assert len(rels) == 1
        assert rels[0].type == "knows"

    def test_get_incoming(self):
        g = KnowledgeGraph()
        g.add_entity(Entity(id="a", type="T"))
        g.add_entity(Entity(id="b", type="T"))
        g.add_relation(Relation(id="r1", source_id="a", target_id="b", type="knows"))
        rels = g.get_relations("b", "incoming")
        assert len(rels) == 1

    def test_get_both(self):
        g = KnowledgeGraph()
        g.add_entity(Entity(id="a", type="T"))
        g.add_entity(Entity(id="b", type="T"))
        g.add_entity(Entity(id="c", type="T"))
        g.add_relation(Relation(id="r1", source_id="a", target_id="b", type="x"))
        g.add_relation(Relation(id="r2", source_id="c", target_id="b", type="y"))
        rels = g.get_relations("b", "both")
        assert len(rels) == 2

    def test_get_relations_missing_entity(self):
        g = KnowledgeGraph()
        assert g.get_relations("missing", "outgoing") == []

    def test_get_relations_default_direction(self):
        g = KnowledgeGraph()
        g.add_entity(Entity(id="a", type="T"))
        g.add_entity(Entity(id="b", type="T"))
        g.add_relation(Relation(id="r1", source_id="a", target_id="b", type="z"))
        rels = g.get_relations("a")
        assert len(rels) == 1


class TestKnowledgeGraphFind:
    def test_find_by_type(self):
        g = KnowledgeGraph()
        g.add_entity(Entity(id="a", type="Person"))
        g.add_entity(Entity(id="b", type="Vessel"))
        result = g.find_entities(type_filter="Person")
        assert len(result) == 1
        assert result[0].id == "a"

    def test_find_by_property(self):
        g = KnowledgeGraph()
        g.add_entity(Entity(id="a", type="T", properties={"color": "red"}))
        g.add_entity(Entity(id="b", type="T", properties={"color": "blue"}))
        result = g.find_entities(properties={"color": "red"})
        assert len(result) == 1

    def test_find_by_type_and_property(self):
        g = KnowledgeGraph()
        g.add_entity(Entity(id="a", type="T", properties={"x": 1}))
        g.add_entity(Entity(id="b", type="T2", properties={"x": 1}))
        result = g.find_entities(type_filter="T", properties={"x": 1})
        assert len(result) == 1

    def test_find_no_match(self):
        g = KnowledgeGraph()
        g.add_entity(Entity(id="a", type="T"))
        assert g.find_entities(type_filter="Missing") == []

    def test_find_no_filter(self):
        g = KnowledgeGraph()
        g.add_entity(Entity(id="a", type="T"))
        g.add_entity(Entity(id="b", type="T"))
        assert len(g.find_entities()) == 2


class TestKnowledgeGraphPaths:
    def test_find_direct_path(self):
        g = KnowledgeGraph()
        for eid in ("a", "b"):
            g.add_entity(Entity(id=eid, type="T"))
        g.add_relation(Relation(id="r1", source_id="a", target_id="b", type="edge"))
        paths = g.find_paths("a", "b")
        assert len(paths) == 1
        assert len(paths[0]) == 1

    def test_find_multi_hop_path(self):
        g = KnowledgeGraph()
        for eid in ("a", "b", "c"):
            g.add_entity(Entity(id=eid, type="T"))
        g.add_relation(Relation(id="r1", source_id="a", target_id="b", type="e"))
        g.add_relation(Relation(id="r2", source_id="b", target_id="c", type="e"))
        paths = g.find_paths("a", "c")
        assert len(paths) >= 1

    def test_no_path(self):
        g = KnowledgeGraph()
        g.add_entity(Entity(id="a", type="T"))
        g.add_entity(Entity(id="b", type="T"))
        assert g.find_paths("a", "b") == []

    def test_path_missing_entities(self):
        g = KnowledgeGraph()
        assert g.find_paths("x", "y") == []

    def test_path_same_node(self):
        g = KnowledgeGraph()
        g.add_entity(Entity(id="a", type="T"))
        assert g.find_paths("a", "a") == []

    def test_path_max_depth(self):
        g = KnowledgeGraph()
        for eid in ("a", "b", "c", "d"):
            g.add_entity(Entity(id=eid, type="T"))
        g.add_relation(Relation(id="r1", source_id="a", target_id="b", type="e"))
        g.add_relation(Relation(id="r2", source_id="b", target_id="c", type="e"))
        g.add_relation(Relation(id="r3", source_id="c", target_id="d", type="e"))
        paths = g.find_paths("a", "d", max_depth=1)
        assert paths == []


class TestKnowledgeGraphNeighbors:
    def test_direct_neighbors(self):
        g = KnowledgeGraph()
        for eid in ("a", "b", "c"):
            g.add_entity(Entity(id=eid, type="T"))
        g.add_relation(Relation(id="r1", source_id="a", target_id="b", type="e"))
        g.add_relation(Relation(id="r2", source_id="a", target_id="c", type="e"))
        nbrs = g.get_neighbors("a", depth=1)
        assert nbrs == {"b", "c"}

    def test_neighbors_excludes_self(self):
        g = KnowledgeGraph()
        g.add_entity(Entity(id="a", type="T"))
        assert g.get_neighbors("a", depth=1) == set()

    def test_two_hop_neighbors(self):
        g = KnowledgeGraph()
        for eid in ("a", "b", "c"):
            g.add_entity(Entity(id=eid, type="T"))
        g.add_relation(Relation(id="r1", source_id="a", target_id="b", type="e"))
        g.add_relation(Relation(id="r2", source_id="b", target_id="c", type="e"))
        nbrs = g.get_neighbors("a", depth=2)
        assert "b" in nbrs
        assert "c" in nbrs


class TestKnowledgeGraphMerge:
    def test_merge_empty(self):
        g1 = KnowledgeGraph()
        g2 = KnowledgeGraph()
        g1.merge(g2)
        assert g1.stats()["entity_count"] == 0

    def test_merge_entities(self):
        g1 = KnowledgeGraph()
        g2 = KnowledgeGraph()
        g1.add_entity(Entity(id="a", type="T"))
        g2.add_entity(Entity(id="b", type="T"))
        g1.merge(g2)
        assert g1.stats()["entity_count"] == 2

    def test_merge_relations(self):
        g1 = KnowledgeGraph()
        g2 = KnowledgeGraph()
        g1.add_entity(Entity(id="a", type="T"))
        g1.add_entity(Entity(id="b", type="T"))
        g2.add_entity(Entity(id="a", type="T"))
        g2.add_entity(Entity(id="b", type="T"))
        g2.add_relation(Relation(id="r1", source_id="a", target_id="b", type="link"))
        g1.merge(g2)
        assert g1.stats()["relation_count"] == 1

    def test_merge_returns_self(self):
        g1 = KnowledgeGraph()
        result = g1.merge(KnowledgeGraph())
        assert result is g1

    def test_merge_dedup_relations(self):
        g1 = KnowledgeGraph()
        g2 = KnowledgeGraph()
        g1.add_entity(Entity(id="a", type="T"))
        g1.add_entity(Entity(id="b", type="T"))
        g2.add_entity(Entity(id="a", type="T"))
        g2.add_entity(Entity(id="b", type="T"))
        g1.add_relation(Relation(id="r1", source_id="a", target_id="b", type="link"))
        g2.add_relation(Relation(id="r1", source_id="a", target_id="b", type="link"))
        g1.merge(g2)
        assert g1.stats()["relation_count"] == 1


class TestKnowledgeGraphIterators:
    def test_entities_iterator(self):
        g = KnowledgeGraph()
        g.add_entity(Entity(id="a", type="T"))
        g.add_entity(Entity(id="b", type="T"))
        assert len(list(g.entities())) == 2

    def test_relations_iterator(self):
        g = KnowledgeGraph()
        g.add_entity(Entity(id="a", type="T"))
        g.add_entity(Entity(id="b", type="T"))
        g.add_relation(Relation(id="r1", source_id="a", target_id="b", type="e"))
        assert len(list(g.relations())) == 1


# ======================================================================
# Helper functions
# ======================================================================

class TestHelpers:
    def test_make_entity(self):
        e = make_entity("Vessel", properties={"name": "X"})
        assert e.type == "Vessel"
        assert e.id  # auto-generated

    def test_make_entity_with_id(self):
        e = make_entity("Vessel", entity_id="v1")
        assert e.id == "v1"

    def test_make_relation(self):
        r = make_relation("a", "b", "knows")
        assert r.source_id == "a"
        assert r.target_id == "b"
        assert r.type == "knows"

    def test_make_relation_with_weight(self):
        r = make_relation("a", "b", "knows", weight=0.5)
        assert r.weight == 0.5

    def test_make_relation_with_id(self):
        r = make_relation("a", "b", "knows", relation_id="my_rel")
        assert r.id == "my_rel"
