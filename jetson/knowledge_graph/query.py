"""Graph query engine for NEXUS knowledge graph."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .graph import Entity, KnowledgeGraph, Relation


@dataclass
class QueryResult:
    """Container for query engine results."""
    results: List[Any] = field(default_factory=list)
    total_count: int = 0
    execution_time: float = 0.0
    confidence: float = 1.0

    @property
    def is_empty(self) -> bool:
        return self.total_count == 0


@dataclass
class PatternMatch:
    """A single pattern match result."""
    bindings: Dict[str, Entity] = field(default_factory=dict)
    relations: List[Relation] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class AggregationResult:
    """Result of an aggregation query."""
    group_key: str
    value: Any
    count: int = 0


class GraphQueryEngine:
    """Query engine that supports pattern matching, aggregation, and pagination."""

    def __init__(self) -> None:
        self._query_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Main query entry point
    # ------------------------------------------------------------------

    def query(
        self,
        graph: KnowledgeGraph,
        query_string: str,
    ) -> QueryResult:
        """Parse a simple query string and execute against the graph.

        Supported formats:
        - "MATCH (e:EntityType) RETURN e"
        - "MATCH (e:EntityType {prop: value}) RETURN e"
        - "FIND entities WHERE type = 'EntityType'"
        - "FIND entity BY id 'entity_id'"
        - "COUNT (e:EntityType)"
        - "PATHS FROM 'id1' TO 'id2'"
        - "NEIGHBORS OF 'id'"
        - "RELATIONS FROM 'id' DIRECTION outgoing"
        - "AGGREGATE EntityType BY prop"
        """
        start = time.monotonic()
        results: List[Any] = []
        confidence = 1.0
        query_lower = query_string.strip().lower()

        # MATCH pattern
        match = re.match(
            r"match\s+\((\w+):(\w+)(?:\s*\{(\w+)\s*:\s*'?(\w+)'?\})?\)\s+return\s+(\w+)",
            query_lower,
        )
        if match:
            var, etype, prop_key, prop_val, ret_var = match.groups()
            props = {prop_key: prop_val} if prop_key else None
            entities = graph.find_entities(type_filter=etype.capitalize(), properties=props)
            if ret_var == var:
                results = entities
            confidence = 0.95
            elapsed = time.monotonic() - start
            self._record(query_string, len(results), elapsed)
            return QueryResult(results=results, total_count=len(results), execution_time=elapsed, confidence=confidence)

        # FIND entities WHERE
        match = re.match(r"find\s+entities\s+where\s+type\s*=\s*'(\w+)'", query_lower)
        if match:
            etype = match.group(1).capitalize()
            results = graph.find_entities(type_filter=etype)
            elapsed = time.monotonic() - start
            self._record(query_string, len(results), elapsed)
            return QueryResult(results=results, total_count=len(results), execution_time=elapsed)

        # FIND entity BY id
        match = re.match(r"(?i)find\s+entity\s+by\s+id\s+'([^']+)'", query_string.strip())
        if match:
            eid = match.group(1)
            entity = graph.get_entity(eid)
            results = [entity] if entity else []
            elapsed = time.monotonic() - start
            self._record(query_string, len(results), elapsed)
            return QueryResult(results=results, total_count=len(results), execution_time=elapsed)

        # COUNT
        match = re.match(r"count\s+\(\w+:(\w+)\)", query_lower)
        if match:
            etype = match.group(1).capitalize()
            results = graph.find_entities(type_filter=etype)
            count_val = len(results)
            elapsed = time.monotonic() - start
            self._record(query_string, count_val, elapsed)
            return QueryResult(results=[count_val], total_count=count_val, execution_time=elapsed, confidence=1.0)

        # PATHS
        match = re.match(r"paths\s+from\s+'([^']+)'\s+to\s+'([^']+)'", query_string.strip(), re.IGNORECASE)
        if match:
            src, tgt = match.group(1), match.group(2)
            paths = graph.find_paths(src, tgt)
            elapsed = time.monotonic() - start
            self._record(query_string, len(paths), elapsed)
            return QueryResult(results=paths, total_count=len(paths), execution_time=elapsed)

        # NEIGHBORS
        match = re.match(r"neighbors\s+of\s+'([^']+)'", query_string.strip(), re.IGNORECASE)
        if match:
            eid = match.group(1)
            depth_match = re.search(r"depth\s+(\d+)", query_string.strip(), re.IGNORECASE)
            depth = int(depth_match.group(1)) if depth_match else 1
            neighbors = graph.get_neighbors(eid, depth)
            neighbor_entities = [graph.get_entity(nid) for nid in neighbors if graph.get_entity(nid)]
            elapsed = time.monotonic() - start
            self._record(query_string, len(neighbor_entities), elapsed)
            return QueryResult(results=neighbor_entities, total_count=len(neighbor_entities), execution_time=elapsed)

        # RELATIONS
        match = re.match(
            r"relations\s+from\s+'([^']+)'\s+direction\s+(\w+)",
            query_string.strip(), re.IGNORECASE,
        )
        if match:
            eid, direction = match.group(1), match.group(2).lower()
            rels = graph.get_relations(eid, direction)
            elapsed = time.monotonic() - start
            self._record(query_string, len(rels), elapsed)
            return QueryResult(results=rels, total_count=len(rels), execution_time=elapsed)

        # AGGREGATE
        match = re.match(r"aggregate\s+(\w+)\s+by\s+(\w+)", query_string.strip(), re.IGNORECASE)
        if match:
            etype, prop = match.group(1).capitalize(), match.group(2)
            agg_results = self.aggregate(graph, etype, prop)
            elapsed = time.monotonic() - start
            self._record(query_string, len(agg_results), elapsed)
            return QueryResult(results=agg_results, total_count=len(agg_results), execution_time=elapsed)

        # Fallback: empty result
        elapsed = time.monotonic() - start
        return QueryResult(results=[], total_count=0, execution_time=elapsed, confidence=0.0)

    # ------------------------------------------------------------------
    # Pattern matching
    # ------------------------------------------------------------------

    def match_pattern(
        self,
        graph: KnowledgeGraph,
        pattern_dict: Dict[str, Any],
    ) -> List[PatternMatch]:
        """Match a structural pattern against the graph.

        pattern_dict example:
            {
                "nodes": [{"var": "a", "type": "VesselType"}, {"var": "b", "type": "Equipment"}],
                "edges": [{"from": "a", "to": "b", "type": "uses"}],
            }
        """
        matches: List[PatternMatch] = []

        # Parse nodes
        node_specs: Dict[str, Dict[str, str]] = {}
        for node in pattern_dict.get("nodes", []):
            node_specs[node["var"]] = {"type": node.get("type", ""), "label": node.get("label", "")}

        # Parse edges
        edge_specs = pattern_dict.get("edges", [])

        if not edge_specs or not node_specs:
            return matches

        # Get candidate entities for first node
        first_var = edge_specs[0]["from"] if "from" in edge_specs[0] else list(node_specs.keys())[0]
        spec = node_specs.get(first_var, {})
        candidates = graph.find_entities(
            type_filter=spec.get("type") or None,
        )

        for candidate in candidates:
            binding: Dict[str, Entity] = {first_var: candidate}
            matched_rels = self._match_edges(graph, binding, edge_specs, 0, node_specs)
            if matched_rels is not None:
                matches.append(PatternMatch(bindings=binding, relations=matched_rels))

        return matches

    def _match_edges(
        self,
        graph: KnowledgeGraph,
        binding: Dict[str, Entity],
        edges: List[Dict[str, Any]],
        edge_idx: int,
        node_specs: Dict[str, Dict[str, str]],
    ) -> Optional[List[Relation]]:
        """Recursive edge matcher."""
        if edge_idx >= len(edges):
            return []

        edge = edges[edge_idx]
        from_var = edge.get("from", "")
        to_var = edge.get("to", "")
        rel_type = edge.get("type", "")

        from_entity = binding.get(from_var)
        if from_entity is None:
            return None

        rels = graph.get_relations(from_entity.id, "outgoing")
        for rel in rels:
            if rel_type and rel.type != rel_type:
                continue
            to_entity = graph.get_entity(rel.target_id)
            if to_entity is None:
                continue

            # Check if target matches its spec
            to_spec = node_specs.get(to_var, {})
            if to_spec.get("type") and to_entity.type != to_spec["type"]:
                continue

            # If to_var already bound, must match
            if to_var in binding:
                if binding[to_var].id != to_entity.id:
                    continue
            else:
                binding[to_var] = to_entity

            sub_match = self._match_edges(graph, binding, edges, edge_idx + 1, node_specs)
            if sub_match is not None:
                return [rel] + sub_match
            else:
                # Unbind if we just bound
                if to_var not in {e.get("from", "") for e in edges[:edge_idx+1]}:
                    binding.pop(to_var, None)

        return None

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(
        self,
        graph: KnowledgeGraph,
        grouping: str,
        metric: str,
    ) -> List[AggregationResult]:
        """Group entities by type and compute a metric on a property."""
        entities = graph.find_entities(type_filter=grouping)
        groups: Dict[str, List[Entity]] = {}
        for entity in entities:
            val = entity.properties.get(metric, "__none__")
            groups.setdefault(str(val), []).append(entity)

        results: List[AggregationResult] = []
        for key, ents in groups.items():
            # Compute sum for numeric values
            values = [e.properties.get(metric, 0) for e in ents if isinstance(e.properties.get(metric), (int, float))]
            total = sum(values) if values else 0
            results.append(AggregationResult(
                group_key=key,
                value=total,
                count=len(ents),
            ))
        return results

    # ------------------------------------------------------------------
    # Sorting & Pagination
    # ------------------------------------------------------------------

    @staticmethod
    def sort_results(
        results: List[Any],
        key: str = "id",
        reverse: bool = False,
    ) -> List[Any]:
        """Sort results by an attribute."""
        def _get_sort_val(item: Any) -> Any:
            if isinstance(item, Entity):
                return item.properties.get(key, getattr(item, key, ""))
            if isinstance(item, dict):
                return item.get(key, "")
            return getattr(item, key, "")
        return sorted(results, key=_get_sort_val, reverse=reverse)

    @staticmethod
    def paginate(
        results: List[Any],
        page: int = 1,
        per_page: int = 10,
    ) -> List[Any]:
        """Return a page of results."""
        start = (page - 1) * per_page
        end = start + per_page
        return results[start:end]

    # ------------------------------------------------------------------
    # Explain
    # ------------------------------------------------------------------

    def explain_query(self, query_string: str) -> Dict[str, Any]:
        """Return an explanation of how a query would be executed."""
        explanation: Dict[str, Any] = {
            "query": query_string,
            "strategy": "unknown",
            "estimated_cost": "low",
            "indexes_used": [],
        }
        ql = query_string.strip().lower()

        if ql.startswith("match"):
            explanation["strategy"] = "pattern_match"
            explanation["indexes_used"] = ["type_index", "property_index"]
            explanation["estimated_cost"] = "medium"
        elif ql.startswith("find") and "id" in ql:
            explanation["strategy"] = "direct_lookup"
            explanation["indexes_used"] = ["entity_id_index"]
            explanation["estimated_cost"] = "low"
        elif ql.startswith("find"):
            explanation["strategy"] = "type_scan"
            explanation["indexes_used"] = ["type_index"]
            explanation["estimated_cost"] = "medium"
        elif ql.startswith("count"):
            explanation["strategy"] = "type_count"
            explanation["indexes_used"] = ["type_index"]
            explanation["estimated_cost"] = "low"
        elif ql.startswith("paths"):
            explanation["strategy"] = "bfs_shortest_path"
            explanation["indexes_used"] = ["adjacency_list"]
            explanation["estimated_cost"] = "high"
        elif ql.startswith("neighbors"):
            explanation["strategy"] = "bfs_neighbor_search"
            explanation["indexes_used"] = ["adjacency_list"]
            explanation["estimated_cost"] = "medium"
        elif ql.startswith("relations"):
            explanation["strategy"] = "relation_scan"
            explanation["indexes_used"] = ["adjacency_list"]
            explanation["estimated_cost"] = "low"
        elif ql.startswith("aggregate"):
            explanation["strategy"] = "group_aggregate"
            explanation["indexes_used"] = ["type_index", "property_index"]
            explanation["estimated_cost"] = "medium"

        return explanation

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def _record(self, query: str, result_count: int, exec_time: float) -> None:
        self._query_history.append({
            "query": query,
            "result_count": result_count,
            "execution_time": exec_time,
        })

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self._query_history)

    def clear_history(self) -> None:
        self._query_history.clear()
