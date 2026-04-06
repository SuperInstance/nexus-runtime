"""Graph reasoning and inference engine for NEXUS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from .graph import Entity, KnowledgeGraph, Relation


@dataclass
class InferenceRule:
    """A single inference rule with premises, conclusion, and confidence."""
    name: str
    premises: List[str]  # list of (source, rel_type, target) triples as strings
    conclusion: str       # description of the inferred fact
    confidence: float = 1.0


@dataclass
class InferredFact:
    """A fact derived by the reasoner."""
    fact: str
    source_ids: Tuple[str, ...]
    confidence: float
    rule_name: str


@dataclass
class Contradiction:
    """A pair of contradictory facts."""
    fact_a: str
    fact_b: str
    entity_id: str


class Reasoner:
    """Forward and backward chaining reasoner over a KnowledgeGraph."""

    def __init__(self) -> None:
        self._rules: List[InferenceRule] = []

    def add_rule(self, rule: InferenceRule) -> None:
        self._rules.append(rule)

    # ------------------------------------------------------------------
    # Forward chaining
    # ------------------------------------------------------------------

    def forward_chain(
        self,
        graph: KnowledgeGraph,
        rules: Optional[List[InferenceRule]] = None,
        max_steps: int = 50,
    ) -> List[InferredFact]:
        """Apply rules iteratively until no new facts or max_steps reached."""
        active_rules = rules or self._rules
        inferred: List[InferredFact] = []
        known_facts: Set[str] = set()
        step = 0

        # Seed with explicit facts from the graph
        for rel in graph.relations():
            fact_str = f"{rel.source_id}|{rel.type}|{rel.target_id}"
            known_facts.add(fact_str)

        changed = True
        while changed and step < max_steps:
            changed = False
            step += 1
            for rule in active_rules:
                # Check if all premises are satisfied
                if self._premises_satisfied(rule.premises, graph, known_facts):
                    conclusion_key = rule.conclusion
                    if conclusion_key not in known_facts:
                        known_facts.add(conclusion_key)
                        sources = self._extract_premise_entities(rule.premises, graph)
                        inferred.append(InferredFact(
                            fact=conclusion_key,
                            source_ids=tuple(sources),
                            confidence=rule.confidence,
                            rule_name=rule.name,
                        ))
                        changed = True
        return inferred

    # ------------------------------------------------------------------
    # Backward chaining
    # ------------------------------------------------------------------

    def backward_chain(
        self,
        graph: KnowledgeGraph,
        goal: str,
        rules: Optional[List[InferenceRule]] = None,
    ) -> List[str]:
        """Attempt to prove *goal* by finding a chain of rules.

        Returns a list of rule names forming the proof path (or empty list).
        """
        active_rules = rules or self._rules

        def _prove(g: str, depth: int, visited: Set[str]) -> List[str]:
            if depth > 10:
                return []
            # Check if goal already in graph
            for rel in graph.relations():
                fact_str = f"{rel.source_id}|{rel.type}|{rel.target_id}"
                if fact_str == g or g in fact_str:
                    return ["graph_fact"]
            # Try each rule whose conclusion matches the goal
            for rule in active_rules:
                if rule.name in visited:
                    continue
                if rule.conclusion == g or g in rule.conclusion:
                    # Try to prove premises
                    all_proved = True
                    proof_path: List[str] = []
                    for premise in rule.premises:
                        sub = _prove(premise, depth + 1, visited | {rule.name})
                        if sub:
                            proof_path.extend(sub)
                        else:
                            all_proved = False
                            break
                    if all_proved:
                        return proof_path + [rule.name]
            return []

        return _prove(goal, 0, set())

    # ------------------------------------------------------------------
    # Transitive closure
    # ------------------------------------------------------------------

    def transitive_closure(
        self,
        graph: KnowledgeGraph,
        relation_type: str,
    ) -> KnowledgeGraph:
        """Compute transitive closure for a given relation type.

        Returns a *new* graph containing the original entities plus
        inferred transitive relations.
        """
        new_graph = KnowledgeGraph()
        for entity in graph.entities():
            new_graph.add_entity(entity)
        for rel in graph.relations():
            new_graph.add_relation(rel)

        changed = True
        while changed:
            changed = False
            for rel in list(new_graph.relations()):
                if rel.type != relation_type:
                    continue
                # Find outgoing of target that match type
                for rel2 in new_graph.get_relations(rel.target_id, "outgoing"):
                    if rel2.type != relation_type:
                        continue
                    # Infer rel.source_id -> rel2.target_id
                    exists = any(
                        r.source_id == rel.source_id
                        and r.target_id == rel2.target_id
                        and r.type == relation_type
                        for r in new_graph.get_relations(rel.source_id, "outgoing")
                    )
                    if not exists:
                        from .graph import make_relation
                        new_graph.add_relation(make_relation(
                            source_id=rel.source_id,
                            target_id=rel2.target_id,
                            rel_type=relation_type,
                            properties={"inferred": True},
                            weight=rel.weight * rel2.weight,
                        ))
                        changed = True
        return new_graph

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    def compute_similarity(
        self,
        entity1: str,
        entity2: str,
        graph: KnowledgeGraph,
    ) -> float:
        """Compute structural similarity between two entities (0-1).

        Uses Jaccard similarity on neighbor sets.
        """
        e1_neighbors = graph.get_neighbors(entity1, depth=1)
        e2_neighbors = graph.get_neighbors(entity2, depth=1)
        if not e1_neighbors and not e2_neighbors:
            return 1.0  # both isolated
        if not e1_neighbors or not e2_neighbors:
            return 0.0
        intersection = e1_neighbors & e2_neighbors
        union = e1_neighbors | e2_neighbors
        return len(intersection) / len(union)

    # ------------------------------------------------------------------
    # Contradiction detection
    # ------------------------------------------------------------------

    def detect_contradictions(self, graph: KnowledgeGraph) -> List[Contradiction]:
        """Detect contradictory relations (e.g. A is-a B and A not-a B)."""
        contradictions: List[Contradiction] = []
        negation_map: Dict[str, str] = {
            "is_a": "not_a",
            "has": "lacks",
            "part_of": "not_part_of",
            "requires": "exempts",
            "enabled": "disabled",
        }
        outgoing_by_type: Dict[str, Dict[str, List[Relation]]] = {}
        for rel in graph.relations():
            outgoing_by_type.setdefault(rel.source_id, {}).setdefault(rel.type, []).append(rel)

        for source_id, types_dict in outgoing_by_type.items():
            for rel_type, rels in types_dict.items():
                neg_type = negation_map.get(rel_type)
                if neg_type and neg_type in types_dict:
                    for r1 in rels:
                        for r2 in types_dict[neg_type]:
                            if r1.target_id == r2.target_id:
                                contradictions.append(Contradiction(
                                    fact_a=f"{r1.source_id}|{r1.type}|{r1.target_id}",
                                    fact_b=f"{r2.source_id}|{r2.type}|{r2.target_id}",
                                    entity_id=source_id,
                                ))
        return contradictions

    # ------------------------------------------------------------------
    # Centrality
    # ------------------------------------------------------------------

    def compute_centrality(
        self,
        graph: KnowledgeGraph,
        entity_id: str,
    ) -> float:
        """Compute normalized degree centrality for an entity (0-1)."""
        rels = graph.get_relations(entity_id, "both")
        if not rels:
            return 0.0
        n = len(list(graph.entities()))
        if n <= 1:
            return 0.0
        degree = len(rels)
        return degree / (n - 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _premises_satisfied(
        premises: List[str],
        graph: KnowledgeGraph,
        known_facts: Set[str],
    ) -> bool:
        for premise in premises:
            # premise could be "entity_type:Foo" or "rel:bar|baz"
            if premise.startswith("entity_type:"):
                etype = premise[len("entity_type:"):]
                if not any(e.type == etype for e in graph.entities()):
                    return False
            elif premise in known_facts:
                continue
            else:
                # Check graph relations
                found = False
                for rel in graph.relations():
                    fact_str = f"{rel.source_id}|{rel.type}|{rel.target_id}"
                    if fact_str == premise or premise in fact_str:
                        found = True
                        break
                if not found:
                    return False
        return True

    @staticmethod
    def _extract_premise_entities(
        premises: List[str],
        graph: KnowledgeGraph,
    ) -> List[str]:
        ids: List[str] = []
        for p in premises:
            if p.startswith("entity_type:"):
                etype = p[len("entity_type:"):]
                for e in graph.find_entities(type_filter=etype):
                    ids.append(e.id)
            elif "|" in p:
                parts = p.split("|")
                if len(parts) >= 2:
                    ids.extend([parts[0], parts[-1]])
        return list(dict.fromkeys(ids))  # deduplicate preserving order
