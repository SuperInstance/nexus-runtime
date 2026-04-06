"""Tests for query.py — GraphQueryEngine, QueryResult, PatternMatch, AggregationResult."""

import pytest
from jetson.knowledge_graph.graph import (
    Entity, Relation, KnowledgeGraph, make_entity, make_relation,
)
from jetson.knowledge_graph.query import (
    QueryResult, PatternMatch, AggregationResult, GraphQueryEngine,
)


# ======================================================================
# QueryResult
# ======================================================================

class TestQueryResult:
    def test_empty_result(self):
        qr = QueryResult()
        assert qr.is_empty
        assert qr.total_count == 0

    def test_non_empty_result(self):
        qr = QueryResult(results=[1, 2, 3], total_count=3)
        assert not qr.is_empty

    def test_execution_time(self):
        qr = QueryResult(execution_time=0.001)
        assert qr.execution_time == 0.001

    def test_confidence(self):
        qr = QueryResult(confidence=0.85)
        assert qr.confidence == 0.85


# ======================================================================
# PatternMatch
# ======================================================================

class TestPatternMatch:
    def test_create_pattern_match(self):
        pm = PatternMatch(bindings={"a": Entity(id="x", type="T")})
        assert "a" in pm.bindings

    def test_pattern_match_with_relations(self):
        pm = PatternMatch(
            bindings={},
            relations=[Relation(id="r1", source_id="a", target_id="b", type="e")],
        )
        assert len(pm.relations) == 1

    def test_pattern_match_confidence(self):
        pm = PatternMatch(confidence=0.7)
        assert pm.confidence == 0.7


# ======================================================================
# AggregationResult
# ======================================================================

class TestAggregationResult:
    def test_create(self):
        ar = AggregationResult(group_key="red", value=42, count=5)
        assert ar.group_key == "red"
        assert ar.value == 42


# ======================================================================
# GraphQueryEngine — query()
# ======================================================================

class TestGraphQueryEngineQuery:
    def _make_sample_graph(self):
        g = KnowledgeGraph()
        for i in range(3):
            g.add_entity(make_entity("Vessel", entity_id=f"v{i}", properties={"name": f"Vessel{i}"}))
        g.add_relation(make_relation("v0", "v1", "connects"))
        g.add_relation(make_relation("v1", "v2", "connects"))
        return g

    def test_match_type(self):
        g = self._make_sample_graph()
        engine = GraphQueryEngine()
        qr = engine.query(g, "MATCH (e:Vessel) RETURN e")
        assert qr.total_count == 3

    def test_match_type_with_property(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("Vessel", entity_id="v1", properties={"name": "alpha"}))
        g.add_entity(make_entity("Vessel", entity_id="v2", properties={"name": "Beta"}))
        engine = GraphQueryEngine()
        qr = engine.query(g, "MATCH (e:Vessel {name: Alpha}) RETURN e")
        assert qr.total_count == 1

    def test_find_entities_by_type(self):
        g = self._make_sample_graph()
        engine = GraphQueryEngine()
        qr = engine.query(g, "FIND entities WHERE type = 'Vessel'")
        assert qr.total_count == 3

    def test_find_entity_by_id(self):
        g = self._make_sample_graph()
        engine = GraphQueryEngine()
        qr = engine.query(g, "FIND entity BY id 'v1'")
        assert qr.total_count == 1

    def test_find_entity_by_id_missing(self):
        g = self._make_sample_graph()
        engine = GraphQueryEngine()
        qr = engine.query(g, "FIND entity BY id 'missing'")
        assert qr.total_count == 0

    def test_count_type(self):
        g = self._make_sample_graph()
        engine = GraphQueryEngine()
        qr = engine.query(g, "COUNT (e:Vessel)")
        assert qr.total_count == 3  # count result
        # results contain the count value
        assert qr.results == [3]

    def test_paths_query(self):
        g = self._make_sample_graph()
        engine = GraphQueryEngine()
        qr = engine.query(g, "PATHS FROM 'v0' TO 'v2'")
        assert qr.total_count >= 1

    def test_paths_no_path(self):
        g = self._make_sample_graph()
        engine = GraphQueryEngine()
        qr = engine.query(g, "PATHS FROM 'v0' TO 'missing'")
        assert qr.total_count == 0

    def test_neighbors_query(self):
        g = self._make_sample_graph()
        engine = GraphQueryEngine()
        qr = engine.query(g, "NEIGHBORS OF 'v1'")
        assert qr.total_count >= 1

    def test_neighbors_with_depth(self):
        g = self._make_sample_graph()
        engine = GraphQueryEngine()
        qr = engine.query(g, "NEIGHBORS OF 'v0' DEPTH 2")
        assert qr.total_count >= 2

    def test_relations_outgoing(self):
        g = self._make_sample_graph()
        engine = GraphQueryEngine()
        qr = engine.query(g, "RELATIONS FROM 'v0' DIRECTION outgoing")
        assert qr.total_count == 1

    def test_relations_incoming(self):
        g = self._make_sample_graph()
        engine = GraphQueryEngine()
        qr = engine.query(g, "RELATIONS FROM 'v1' DIRECTION incoming")
        assert qr.total_count == 1

    def test_aggregate_query(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("T", entity_id="a", properties={"cat": "x", "val": 10}))
        g.add_entity(make_entity("T", entity_id="b", properties={"cat": "x", "val": 20}))
        g.add_entity(make_entity("T", entity_id="c", properties={"cat": "y", "val": 5}))
        engine = GraphQueryEngine()
        qr = engine.query(g, "AGGREGATE T BY cat")
        assert qr.total_count >= 1

    def test_unknown_query(self):
        g = KnowledgeGraph()
        engine = GraphQueryEngine()
        qr = engine.query(g, "UNKNOWN COMMAND")
        assert qr.total_count == 0
        assert qr.confidence == 0.0

    def test_execution_time_recorded(self):
        g = self._make_sample_graph()
        engine = GraphQueryEngine()
        qr = engine.query(g, "COUNT (e:Vessel)")
        assert qr.execution_time >= 0.0


# ======================================================================
# GraphQueryEngine — match_pattern
# ======================================================================

class TestMatchPattern:
    def test_simple_pattern(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("Vessel", entity_id="v1"))
        g.add_entity(make_entity("Equipment", entity_id="e1"))
        g.add_relation(make_relation("v1", "e1", "uses"))
        engine = GraphQueryEngine()
        pattern = {
            "nodes": [{"var": "v", "type": "Vessel"}, {"var": "e", "type": "Equipment"}],
            "edges": [{"from": "v", "to": "e", "type": "uses"}],
        }
        matches = engine.match_pattern(g, pattern)
        assert len(matches) >= 1

    def test_no_match(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("Vessel", entity_id="v1"))
        g.add_entity(make_entity("Equipment", entity_id="e1"))
        # No relation
        engine = GraphQueryEngine()
        pattern = {
            "nodes": [{"var": "v", "type": "Vessel"}, {"var": "e", "type": "Equipment"}],
            "edges": [{"from": "v", "to": "e", "type": "uses"}],
        }
        matches = engine.match_pattern(g, pattern)
        assert len(matches) == 0

    def test_empty_pattern(self):
        g = KnowledgeGraph()
        engine = GraphQueryEngine()
        matches = engine.match_pattern(g, {})
        assert matches == []

    def test_missing_type_filter_in_pattern(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("T", entity_id="a"))
        g.add_entity(make_entity("T", entity_id="b"))
        g.add_relation(make_relation("a", "b", "links"))
        engine = GraphQueryEngine()
        pattern = {
            "nodes": [{"var": "x"}, {"var": "y"}],
            "edges": [{"from": "x", "to": "y", "type": "links"}],
        }
        matches = engine.match_pattern(g, pattern)
        assert len(matches) >= 1


# ======================================================================
# GraphQueryEngine — aggregate
# ======================================================================

class TestAggregate:
    def test_group_by_property(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("T", entity_id="a", properties={"color": "red", "size": 5}))
        g.add_entity(make_entity("T", entity_id="b", properties={"color": "red", "size": 3}))
        g.add_entity(make_entity("T", entity_id="c", properties={"color": "blue", "size": 10}))
        engine = GraphQueryEngine()
        results = engine.aggregate(g, "T", "color")
        groups = {r.group_key for r in results}
        assert "red" in groups
        assert "blue" in groups

    def test_aggregate_empty(self):
        g = KnowledgeGraph()
        engine = GraphQueryEngine()
        results = engine.aggregate(g, "Missing", "prop")
        assert results == []

    def test_aggregate_numeric_sum(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("T", entity_id="a", properties={"val": 10}))
        g.add_entity(make_entity("T", entity_id="b", properties={"val": 20}))
        engine = GraphQueryEngine()
        results = engine.aggregate(g, "T", "val")
        assert len(results) == 2  # each in own group (unique values)


# ======================================================================
# GraphQueryEngine — sort & paginate
# ======================================================================

class TestSortAndPaginate:
    def test_sort_by_id(self):
        engine = GraphQueryEngine()
        items = [make_entity("T", entity_id=f"e{i}") for i in [3, 1, 2]]
        sorted_items = engine.sort_results(items, key="id")
        assert sorted_items[0].id == "e1"

    def test_sort_by_property(self):
        engine = GraphQueryEngine()
        items = [
            make_entity("T", entity_id="a", properties={"val": 3}),
            make_entity("T", entity_id="b", properties={"val": 1}),
        ]
        sorted_items = engine.sort_results(items, key="val")
        assert sorted_items[0].properties["val"] == 1

    def test_sort_reverse(self):
        engine = GraphQueryEngine()
        items = [make_entity("T", entity_id=f"e{i}") for i in range(3)]
        sorted_items = engine.sort_results(items, key="id", reverse=True)
        assert sorted_items[0].id == "e2"

    def test_paginate_first_page(self):
        engine = GraphQueryEngine()
        items = list(range(25))
        page = engine.paginate(items, page=1, per_page=10)
        assert len(page) == 10

    def test_paginate_second_page(self):
        engine = GraphQueryEngine()
        items = list(range(25))
        page = engine.paginate(items, page=2, per_page=10)
        assert len(page) == 10
        assert page[0] == 10

    def test_paginate_last_page(self):
        engine = GraphQueryEngine()
        items = list(range(25))
        page = engine.paginate(items, page=3, per_page=10)
        assert len(page) == 5

    def test_paginate_empty(self):
        engine = GraphQueryEngine()
        assert engine.paginate([], page=1) == []


# ======================================================================
# GraphQueryEngine — explain_query
# ======================================================================

class TestExplainQuery:
    def test_explain_match(self):
        engine = GraphQueryEngine()
        exp = engine.explain_query("MATCH (e:Vessel) RETURN e")
        assert exp["strategy"] == "pattern_match"

    def test_explain_find_id(self):
        engine = GraphQueryEngine()
        exp = engine.explain_query("FIND entity BY id 'v1'")
        assert exp["strategy"] == "direct_lookup"

    def test_explain_find_type(self):
        engine = GraphQueryEngine()
        exp = engine.explain_query("FIND entities WHERE type = 'Vessel'")
        assert exp["strategy"] == "type_scan"

    def test_explain_count(self):
        engine = GraphQueryEngine()
        exp = engine.explain_query("COUNT (e:Vessel)")
        assert exp["strategy"] == "type_count"

    def test_explain_paths(self):
        engine = GraphQueryEngine()
        exp = engine.explain_query("PATHS FROM 'a' TO 'b'")
        assert exp["strategy"] == "bfs_shortest_path"

    def test_explain_unknown(self):
        engine = GraphQueryEngine()
        exp = engine.explain_query("FOOBAR")
        assert exp["strategy"] == "unknown"


# ======================================================================
# GraphQueryEngine — History
# ======================================================================

class TestHistory:
    def test_history_recorded(self):
        g = KnowledgeGraph()
        g.add_entity(make_entity("T", entity_id="a"))
        engine = GraphQueryEngine()
        engine.query(g, "COUNT (e:T)")
        engine.query(g, "COUNT (e:Missing)")
        history = engine.get_history()
        assert len(history) == 2

    def test_clear_history(self):
        g = KnowledgeGraph()
        engine = GraphQueryEngine()
        engine.query(g, "COUNT (e:T)")
        engine.clear_history()
        assert engine.get_history() == []
