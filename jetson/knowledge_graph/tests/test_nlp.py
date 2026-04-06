"""Tests for nlp.py — NLPEngine, Token, ParsedIntent."""

import pytest
from jetson.knowledge_graph.nlp import Token, ParsedIntent, NLPEngine


# ======================================================================
# Token
# ======================================================================

class TestToken:
    def test_create_token(self):
        t = Token(text="hello")
        assert t.text == "hello"
        assert t.pos == "UNKNOWN"

    def test_token_lemma_default(self):
        t = Token(text="Running")
        assert t.lemma == "running"

    def test_token_with_fields(self):
        t = Token(text="vessel", pos="NOUN", lemma="vessel", entity_type="Vessel", confidence=0.9)
        assert t.pos == "NOUN"
        assert t.entity_type == "Vessel"


# ======================================================================
# ParsedIntent
# ======================================================================

class TestParsedIntent:
    def test_create_intent(self):
        p = ParsedIntent(intent_type="query_search")
        assert p.intent_type == "query_search"
        assert p.entities == []
        assert p.slots == {}

    def test_intent_with_slots(self):
        p = ParsedIntent(intent_type="query", slots={"location": "harbor"})
        assert p.slots["location"] == "harbor"

    def test_intent_confidence(self):
        p = ParsedIntent(intent_type="q", confidence=0.7)
        assert p.confidence == 0.7


# ======================================================================
# NLPEngine — Tokenization
# ======================================================================

class TestTokenize:
    def test_simple_sentence(self):
        engine = NLPEngine()
        tokens = engine.tokenize("The vessel is moving")
        assert len(tokens) == 4

    def test_empty_string(self):
        engine = NLPEngine()
        assert engine.tokenize("") == []

    def test_numbers_preserved(self):
        engine = NLPEngine()
        tokens = engine.tokenize("depth 100 meters")
        assert tokens[1].text == "100"

    def test_apostrophe_handling(self):
        engine = NLPEngine()
        tokens = engine.tokenize("it's moving")
        texts = [t.text for t in tokens]
        assert "it's" in texts or "it" in texts

    def test_pos_noun_default(self):
        engine = NLPEngine()
        tokens = engine.tokenize("boat")
        assert tokens[0].pos == "NOUN"

    def test_pos_verb_detection(self):
        engine = NLPEngine()
        tokens = engine.tokenize("running")
        assert tokens[0].pos == "VERB"

    def test_pos_adjective_detection(self):
        engine = NLPEngine()
        tokens = engine.tokenize("dangerous")
        assert tokens[0].pos == "ADJ"

    def test_pos_adverb_detection(self):
        engine = NLPEngine()
        tokens = engine.tokenize("quickly")
        assert tokens[0].pos == "ADV"

    def test_lemma_lowercase(self):
        engine = NLPEngine()
        tokens = engine.tokenize("VESSEL")
        assert tokens[0].lemma == "vessel"


# ======================================================================
# NLPEngine — Intent parsing
# ======================================================================

class TestParseIntent:
    def test_what_is_query(self):
        engine = NLPEngine()
        result = engine.parse_intent("What is the depth?")
        assert result.intent_type == "query_definition"

    def test_find_all_query(self):
        engine = NLPEngine()
        result = engine.parse_intent("Find all vessels")
        assert result.intent_type == "query_search"

    def test_how_many_count(self):
        engine = NLPEngine()
        result = engine.parse_intent("How many sensors are installed?")
        assert result.intent_type == "query_count"

    def test_list_command(self):
        engine = NLPEngine()
        result = engine.parse_intent("List all equipment")
        assert result.intent_type == "query_list"

    def test_show_command(self):
        engine = NLPEngine()
        result = engine.parse_intent("Show the map")
        assert result.intent_type == "query_show"

    def test_compare_command(self):
        engine = NLPEngine()
        result = engine.parse_intent("Compare vessel A and vessel B")
        assert result.intent_type == "query_compare"

    def test_add_command(self):
        engine = NLPEngine()
        result = engine.parse_intent("Add a new sensor")
        assert result.intent_type == "command_add"

    def test_remove_command(self):
        engine = NLPEngine()
        result = engine.parse_intent("Remove the old sensor")
        assert result.intent_type == "command_remove"

    def test_navigate_command(self):
        engine = NLPEngine()
        result = engine.parse_intent("Navigate to harbor")
        assert result.intent_type == "command_navigate"

    def test_unknown_intent(self):
        engine = NLPEngine()
        result = engine.parse_intent("xyzzy")
        assert result.intent_type == "unknown"
        assert result.confidence < 0.5

    def test_context_passed_to_slots(self):
        engine = NLPEngine()
        result = engine.parse_intent("Find vessels", context={"zone": "alpha"})
        assert result.slots.get("zone") == "alpha"

    def test_explain_intent(self):
        engine = NLPEngine()
        result = engine.parse_intent("Explain the route")
        assert result.intent_type == "query_explain"


# ======================================================================
# NLPEngine — Entity extraction
# ======================================================================

class TestExtractEntities:
    def test_extract_known_entity(self):
        engine = NLPEngine()
        matches = engine.extract_entities("check the sonar array", ["sonar array", "lidar"])
        assert len(matches) >= 1
        assert matches[0][0] == "sonar array"

    def test_extract_multiple(self):
        engine = NLPEngine()
        matches = engine.extract_entities("sonar and lidar are working", ["sonar", "lidar"])
        assert len(matches) == 2

    def test_no_match(self):
        engine = NLPEngine()
        matches = engine.extract_entities("random text", ["sonar", "lidar"])
        assert len(matches) == 0

    def test_empty_known_entities(self):
        engine = NLPEngine()
        assert engine.extract_entities("text", []) == []

    def test_confidence_ranking(self):
        engine = NLPEngine()
        matches = engine.extract_entities("sonar sensor", ["sonar", "sonar sensor"])
        # More specific should rank higher or equal
        if len(matches) == 2:
            assert matches[0][1] >= matches[1][1]

    def test_case_insensitive(self):
        engine = NLPEngine()
        matches = engine.extract_entities("SONAR is active", ["sonar"])
        assert len(matches) >= 1


# ======================================================================
# NLPEngine — Normalization
# ======================================================================

class TestNormalize:
    def test_lowercase(self):
        engine = NLPEngine()
        assert engine.normalize("VESSEL") == "vessel"

    def test_strip_punctuation(self):
        engine = NLPEngine()
        result = engine.normalize("Hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_collapse_whitespace(self):
        engine = NLPEngine()
        result = engine.normalize("  hello   world  ")
        assert result == "hello world"

    def test_empty_string(self):
        engine = NLPEngine()
        assert engine.normalize("") == ""


# ======================================================================
# NLPEngine — Similarity
# ======================================================================

class TestSimilarity:
    def test_identical_texts(self):
        engine = NLPEngine()
        assert engine.compute_similarity("sonar sensor", "sonar sensor") == 1.0

    def test_completely_different(self):
        engine = NLPEngine()
        sim = engine.compute_similarity("sonar sensor", "banana fruit")
        assert sim < 0.5

    def test_partial_overlap(self):
        engine = NLPEngine()
        sim = engine.compute_similarity("sonar sensor depth", "sonar sensor range")
        assert 0.3 < sim < 1.0

    def test_empty_texts(self):
        engine = NLPEngine()
        assert engine.compute_similarity("", "") == 1.0

    def test_one_empty(self):
        engine = NLPEngine()
        assert engine.compute_similarity("text", "") == 0.0

    def test_single_word(self):
        engine = NLPEngine()
        sim = engine.compute_similarity("vessel", "vessel")
        assert sim == 1.0


# ======================================================================
# NLPEngine — Fuzzy matching
# ======================================================================

class TestFuzzyMatch:
    def test_exact_match(self):
        engine = NLPEngine()
        results = engine.fuzzy_match("sonar sensor", ["sonar sensor", "lidar"])
        assert len(results) >= 1
        assert results[0][0] == "sonar sensor"

    def test_threshold_filtering(self):
        engine = NLPEngine()
        results = engine.fuzzy_match("quantum physics", ["sonar", "lidar", "gps"], threshold=0.5)
        assert len(results) == 0

    def test_ranked_results(self):
        engine = NLPEngine()
        results = engine.fuzzy_match("sonar", ["sonar system", "sonar array", "gps"])
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_candidates(self):
        engine = NLPEngine()
        assert engine.fuzzy_match("query", []) == []

    def test_low_threshold(self):
        engine = NLPEngine()
        results = engine.fuzzy_match("sonar", ["sonar"], threshold=0.0)
        assert len(results) >= 1
