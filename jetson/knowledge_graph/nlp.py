"""Natural language understanding for NEXUS knowledge graph queries."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Token:
    """A token produced by the tokenizer."""
    text: str
    pos: str = "UNKNOWN"
    lemma: str = ""
    entity_type: str = ""
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.lemma:
            self.lemma = self.text.lower()


@dataclass
class ParsedIntent:
    """Result of intent parsing."""
    intent_type: str
    entities: List[str] = field(default_factory=list)
    slots: Dict[str, str] = field(default_factory=dict)
    confidence: float = 1.0


# Minimal POS tagger based on suffix heuristics
_SUFFIX_POS = {
    "ing": "VERB",
    "tion": "NOUN",
    "ment": "NOUN",
    "ness": "NOUN",
    "able": "ADJ",
    "ible": "ADJ",
    "ful": "ADJ",
    "less": "ADJ",
    "ous": "ADJ",
    "ive": "ADJ",
    "ly": "ADV",
    "ed": "VERB",
    "er": "NOUN",
    "or": "NOUN",
    "ist": "NOUN",
    "ism": "NOUN",
    "al": "ADJ",
    "es": "VERB",
    "ize": "VERB",
    "ify": "VERB",
    "ate": "VERB",
    "ence": "NOUN",
    "ance": "NOUN",
}

_INTENT_PATTERNS = [
    (r"(?i)what\s+(?:is|are)\s+", "query_definition"),
    (r"(?i)find\s+(?:all|the)?\s*", "query_search"),
    (r"(?i)how\s+many\s+", "query_count"),
    (r"(?i)list\s+(?:all|the)?\s*", "query_list"),
    (r"(?i)show\s+(?:me\s+)?(?:the\s+)?", "query_show"),
    (r"(?i)which\s+", "query_which"),
    (r"(?i)compare\s+", "query_compare"),
    (r"(?i)(?:is|are)\s+\w+\s+(?:a|an|the)\s+", "query_verify"),
    (r"(?i)add\s+(?:a|an|the)?\s*", "command_add"),
    (r"(?i)remove\s+(?:a|an|the)?\s*", "command_remove"),
    (r"(?i)connect\s+", "command_connect"),
    (r"(?i)explain\s+", "query_explain"),
    (r"(?i)why\s+", "query_why"),
    (r"(?i)navigate\s+(?:to\s+)?", "command_navigate"),
    (r"(?i)report\s+", "query_report"),
    (r"(?i)status\s+(?:of\s+)?", "query_status"),
]


class NLPEngine:
    """Pure-Python NLP engine with tokenization, intent parsing, and similarity."""

    def __init__(self) -> None:
        self._stopwords: Set[str] = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "shall",
            "should", "may", "might", "can", "could", "of", "in", "to", "for",
            "with", "on", "at", "from", "by", "about", "as", "into", "through",
            "during", "before", "after", "above", "below", "between", "and",
            "but", "or", "not", "no", "nor", "so", "yet", "both", "either",
            "neither", "each", "every", "all", "any", "few", "more", "most",
            "other", "some", "such", "than", "too", "very", "just", "that",
            "this", "these", "those", "it", "its", "i", "me", "my", "we",
            "our", "you", "your", "he", "him", "his", "she", "her", "they",
            "them", "their", "what", "which", "who", "whom", "when", "where",
            "how",
        }

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> List[Token]:
        """Tokenize *text* into a list of Token objects."""
        raw_tokens = re.findall(r"[A-Za-z0-9_]+(?:'[a-z]+)?", text)
        tokens: List[Token] = []
        for word in raw_tokens:
            pos = self._guess_pos(word)
            tokens.append(Token(text=word, pos=pos, lemma=word.lower()))
        return tokens

    def _guess_pos(self, word: str) -> str:
        lower = word.lower()
        # Check longest suffix first
        for suffix_len in range(min(4, len(lower)), 1, -1):
            suffix = lower[-suffix_len:]
            if suffix in _SUFFIX_POS:
                return _SUFFIX_POS[suffix]
        return "NOUN"  # default

    # ------------------------------------------------------------------
    # Intent parsing
    # ------------------------------------------------------------------

    def parse_intent(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ParsedIntent:
        """Parse user text into a structured intent."""
        for pattern, intent_type in _INTENT_PATTERNS:
            match = re.search(pattern, text)
            if match:
                entities = self._extract_intent_entities(text)
                slots = self._extract_slots(text, context)
                return ParsedIntent(
                    intent_type=intent_type,
                    entities=entities,
                    slots=slots,
                    confidence=0.9,
                )
        return ParsedIntent(
            intent_type="unknown",
            entities=self._extract_intent_entities(text),
            slots={},
            confidence=0.3,
        )

    def _extract_intent_entities(self, text: str) -> List[str]:
        """Extract potential entity mentions from text."""
        tokens = self.tokenize(text)
        entities = []
        for tok in tokens:
            if tok.text.lower() not in self._stopwords and tok.pos in ("NOUN", "ADJ"):
                entities.append(tok.lemma)
        return entities

    def _extract_slots(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Extract slot-value pairs from text."""
        slots: Dict[str, str] = {}
        # Detect numbers
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
        if numbers:
            slots["number"] = numbers[0]
        # Detect time references
        time_match = re.search(
            r"\b(?:today|yesterday|tomorrow|now|last|next|current|recent)\b",
            text, re.IGNORECASE,
        )
        if time_match:
            slots["time"] = time_match.group(0).lower()
        # Inherit from context
        if context:
            for k, v in context.items():
                if k not in slots:
                    slots[k] = str(v)
        return slots

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    def extract_entities(
        self,
        text: str,
        known_entities: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Extract known entities mentioned in *text*.

        Returns list of (entity_name, confidence) sorted by confidence desc.
        """
        known = known_entities or []
        if not known:
            return []
        text_lower = text.lower()
        matches: List[Tuple[str, float]] = []
        for entity in known:
            entity_lower = entity.lower()
            # Exact match
            if entity_lower in text_lower:
                confidence = 1.0
            else:
                confidence = self._token_overlap(entity_lower, text_lower)
            if confidence > 0.0:
                matches.append((entity, confidence))
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize(self, text: str) -> str:
        """Normalize text: lowercase, strip punctuation, collapse whitespace."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Jaccard + bigram overlap similarity (0-1)."""
        norm1 = self.normalize(text1)
        norm2 = self.normalize(text2)
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        jaccard = len(words1 & words2) / len(words1 | words2)

        bigrams1 = self._bigrams(norm1)
        bigrams2 = self._bigrams(norm2)
        if not bigrams1 and not bigrams2:
            bigram_sim = 1.0
        elif not bigrams1 or not bigrams2:
            bigram_sim = 0.0
        else:
            bigram_sim = len(bigrams1 & bigrams2) / len(bigrams1 | bigrams2)

        return 0.5 * jaccard + 0.5 * bigram_sim

    @staticmethod
    def _bigrams(text: str) -> set:
        words = text.split()
        return {f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)}

    @staticmethod
    def _token_overlap(a: str, b: str) -> float:
        set_a = set(a.split())
        set_b = set(b.split())
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / max(len(set_a), len(set_b))

    # ------------------------------------------------------------------
    # Fuzzy matching
    # ------------------------------------------------------------------

    def fuzzy_match(
        self,
        query: str,
        candidates: List[str],
        threshold: float = 0.3,
    ) -> List[Tuple[str, float]]:
        """Rank candidates by similarity to query, filtering by threshold."""
        scored: List[Tuple[str, float]] = []
        for candidate in candidates:
            sim = self.compute_similarity(query, candidate)
            if sim >= threshold:
                scored.append((candidate, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
