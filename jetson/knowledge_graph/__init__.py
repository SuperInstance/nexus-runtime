"""NEXUS Knowledge Graph Engine — Phase 5 Round 1."""

from .graph import Entity, Relation, KnowledgeGraph, make_entity, make_relation
from .reasoning import InferenceRule, InferredFact, Contradiction, Reasoner
from .nlp import Token, ParsedIntent, NLPEngine
from .marine_kb import MarineKnowledgeBase
from .query import QueryResult, PatternMatch, AggregationResult, GraphQueryEngine
from .embedding import EmbeddingVector, SimpleEmbedder

__all__ = [
    "Entity", "Relation", "KnowledgeGraph", "make_entity", "make_relation",
    "InferenceRule", "InferredFact", "Contradiction", "Reasoner",
    "Token", "ParsedIntent", "NLPEngine",
    "MarineKnowledgeBase",
    "QueryResult", "PatternMatch", "AggregationResult", "GraphQueryEngine",
    "EmbeddingVector", "SimpleEmbedder",
]
