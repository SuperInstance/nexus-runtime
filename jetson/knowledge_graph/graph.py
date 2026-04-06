"""Knowledge graph data structure for NEXUS marine robotics platform."""

from __future__ import annotations

import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Set, Tuple


@dataclass(frozen=True)
class Entity:
    """Immutable node in the knowledge graph."""
    id: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    labels: Tuple[str, ...] = ()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass(frozen=True)
class Relation:
    """Immutable directed edge in the knowledge graph."""
    id: str
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Relation):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


class KnowledgeGraph:
    """In-memory knowledge graph with entity/relation storage and graph algorithms."""

    def __init__(self) -> None:
        self._entities: Dict[str, Entity] = {}
        self._relations: Dict[str, Relation] = {}
        self._outgoing: Dict[str, List[str]] = defaultdict(list)   # source -> [rel_ids]
        self._incoming: Dict[str, List[str]] = defaultdict(list)   # target -> [rel_ids]
        self._type_index: Dict[str, Set[str]] = defaultdict(set)   # type -> {entity_ids}
        self._prop_index: Dict[str, Dict[Any, Set[str]]] = defaultdict(lambda: defaultdict(set))

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_entity(self, entity: Entity) -> Entity:
        """Add an entity.  Replaces existing entity with same id."""
        old = self._entities.get(entity.id)
        self._entities[entity.id] = entity
        # rebuild type index
        if old is not None:
            self._type_index[old.type].discard(old.id)
        self._type_index[entity.type].add(entity.id)
        # rebuild property index
        if old is not None:
            for k, v in old.properties.items():
                bucket = self._prop_index.get(k)
                if bucket is not None:
                    bucket.get(v, set()).discard(old.id)
        for k, v in entity.properties.items():
            self._prop_index[k][v].add(entity.id)
        return entity

    def add_relation(self, relation: Relation) -> Relation:
        """Add a directed relation between two entities."""
        self._relations[relation.id] = relation
        self._outgoing[relation.source_id].append(relation.id)
        self._incoming[relation.target_id].append(relation.id)
        return relation

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(entity_id)

    def get_relations(
        self,
        entity_id: str,
        direction: str = "both",
    ) -> List[Relation]:
        """Return relations touching *entity_id*.

        direction: 'outgoing', 'incoming', or 'both'.
        """
        rel_ids: List[str] = []
        if direction in ("outgoing", "both"):
            rel_ids.extend(self._outgoing.get(entity_id, []))
        if direction in ("incoming", "both"):
            rel_ids.extend(self._incoming.get(entity_id, []))
        return [self._relations[rid] for rid in rel_ids if rid in self._relations]

    def find_entities(
        self,
        type_filter: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> List[Entity]:
        """Find entities matching optional type and/or property constraints."""
        candidates: Set[str] = set(self._entities.keys())
        if type_filter is not None:
            candidates &= self._type_index.get(type_filter, set())
        if properties:
            for k, v in properties.items():
                candidates &= self._prop_index.get(k, {}).get(v, set())
        return [self._entities[eid] for eid in candidates if eid in self._entities]

    # ------------------------------------------------------------------
    # Traversal / path finding
    # ------------------------------------------------------------------

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> List[List[Relation]]:
        """BFS shortest-path search (returns all shortest paths up to max_depth)."""
        if source_id not in self._entities or target_id not in self._entities:
            return []
        if source_id == target_id:
            return []

        queue: deque = deque()
        queue.append((source_id, [relation.id for relation in self.get_relations(source_id, "outgoing")], []))
        # Each entry: (current_node, accumulated_rel_ids, path_so_far)
        # Actually let's do a standard BFS tracking visited per path to find *all* shortest
        queue = deque()
        queue.append((source_id, []))  # (node, path of rel ids)

        visited: Dict[str, int] = {source_id: 0}  # node -> shortest depth
        results: List[List[Relation]] = []
        shortest_len = max_depth + 1

        while queue:
            node, path = queue.popleft()
            depth = len(path)
            if depth > max_depth:
                continue
            if depth > shortest_len:
                break

            if node == target_id and path:
                shortest_len = depth
                results.append([self._relations[rid] for rid in path])
                continue  # only keep shortest

            for rid in self._outgoing.get(node, []):
                rel = self._relations[rid]
                nxt = rel.target_id
                if nxt not in visited or visited[nxt] >= depth + 1:
                    visited[nxt] = depth + 1
                    queue.append((nxt, path + [rid]))

        return results

    def get_neighbors(self, entity_id: str, depth: int = 1) -> Set[str]:
        """BFS to collect entity ids within *depth* hops."""
        visited: Set[str] = set()
        queue: deque = deque()
        queue.append((entity_id, 0))
        while queue:
            node, d = queue.popleft()
            if node in visited:
                continue
            if d > depth:
                continue
            visited.add(node)
            for rid in self._outgoing.get(node, []) + self._incoming.get(node, []):
                rel = self._relations.get(rid)
                if rel is None:
                    continue
                other = rel.target_id if rel.source_id == node else rel.source_id
                if other not in visited:
                    queue.append((other, d + 1))
        visited.discard(entity_id)
        return visited

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge(self, other: KnowledgeGraph) -> KnowledgeGraph:
        """Merge *other* graph into this one.  Returns self for chaining."""
        for entity in other._entities.values():
            self.add_entity(entity)
        for relation in other._relations.values():
            # Only add if both endpoints exist in this graph
            if relation.source_id in self._entities and relation.target_id in self._entities:
                if relation.id not in self._relations:
                    self.add_relation(relation)
        return self

    # ------------------------------------------------------------------
    # Stats / iteration
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "entity_count": len(self._entities),
            "relation_count": len(self._relations),
            "entity_types": list(self._type_index.keys()),
        }

    def entities(self) -> Iterator[Entity]:
        return iter(self._entities.values())

    def relations(self) -> Iterator[Relation]:
        return iter(self._relations.values())


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_entity(
    entity_type: str,
    properties: Optional[Dict[str, Any]] = None,
    labels: Optional[Tuple[str, ...]] = None,
    entity_id: Optional[str] = None,
) -> Entity:
    return Entity(
        id=entity_id or str(uuid.uuid4()),
        type=entity_type,
        properties=properties or {},
        labels=labels or (),
    )


def make_relation(
    source_id: str,
    target_id: str,
    rel_type: str,
    properties: Optional[Dict[str, Any]] = None,
    weight: float = 1.0,
    relation_id: Optional[str] = None,
) -> Relation:
    return Relation(
        id=relation_id or str(uuid.uuid4()),
        source_id=source_id,
        target_id=target_id,
        type=rel_type,
        properties=properties or {},
        weight=weight,
    )
