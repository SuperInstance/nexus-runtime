"""Tests for NLParser — tokenization, parsing, entity extraction, normalization."""

import pytest
from jetson.nl_commands.parser import Token, Entity, ParseTree, NLParser


@pytest.fixture
def parser():
    return NLParser()


# ===================================================================
# Token dataclass
# ===================================================================

class TestToken:
    def test_token_creation(self):
        t = Token(text="go", pos_tag="VERB", lemma="go", start=0, end=2)
        assert t.text == "go"
        assert t.pos_tag == "VERB"
        assert t.lemma == "go"
        assert t.start == 0
        assert t.end == 2

    def test_token_fields_mutable(self):
        t = Token(text="run", pos_tag="VERB", lemma="run", start=0, end=3)
        t.text = "ran"
        assert t.text == "ran"


# ===================================================================
# Entity dataclass
# ===================================================================

class TestEntity:
    def test_entity_creation(self):
        e = Entity(entity_type="speed", value=5.0, raw_text="5 knots", start=10, end=17)
        assert e.entity_type == "speed"
        assert e.value == 5.0
        assert e.raw_text == "5 knots"
        assert e.start == 10
        assert e.end == 17
        assert e.confidence == 1.0

    def test_entity_custom_confidence(self):
        e = Entity(entity_type="heading", value=90, raw_text="90°", start=0, end=3, confidence=0.8)
        assert e.confidence == 0.8


# ===================================================================
# ParseTree dataclass
# ===================================================================

class TestParseTree:
    def test_parse_tree_creation(self):
        tokens = [Token(text="go", pos_tag="VERB", lemma="go", start=0, end=2)]
        pt = ParseTree(tokens=tokens, root="go", entities=[], confidence=0.7)
        assert pt.root == "go"
        assert len(pt.tokens) == 1
        assert pt.confidence == 0.7


# ===================================================================
# NLParser.tokenize
# ===================================================================

class TestTokenize:
    def test_simple_sentence(self, parser):
        tokens = parser.tokenize("go to waypoint alpha")
        texts = [t.text for t in tokens]
        assert "go" in texts
        assert "to" in texts
        assert "waypoint" in texts
        assert "alpha" in texts

    def test_empty_string(self, parser):
        tokens = parser.tokenize("")
        assert tokens == []

    def test_whitespace_only(self, parser):
        tokens = parser.tokenize("   ")
        assert tokens == []

    def test_numbers_tokenized(self, parser):
        tokens = parser.tokenize("speed 5 knots")
        texts = [t.text for t in tokens]
        assert "5" in texts

    def test_punctuation(self, parser):
        tokens = parser.tokenize("stop!")
        texts = [t.text for t in tokens]
        assert "!" in texts
        assert "stop" in texts

    def test_pos_tag_verb(self, parser):
        tokens = parser.tokenize("navigate")
        assert tokens[0].pos_tag == "VERB"

    def test_pos_tag_noun(self, parser):
        tokens = parser.tokenize("waypoint")
        assert tokens[0].pos_tag == "NOUN"

    def test_pos_tag_determiner(self, parser):
        tokens = parser.tokenize("the")
        assert tokens[0].pos_tag == "DET"

    def test_pos_tag_preposition(self, parser):
        tokens = parser.tokenize("to")
        assert tokens[0].pos_tag == "PREP"

    def test_pos_tag_number(self, parser):
        tokens = parser.tokenize("42")
        assert tokens[0].pos_tag == "NUM"

    def test_lemma_lowercase(self, parser):
        tokens = parser.tokenize("Navigate")
        assert tokens[0].lemma == "navigate"

    def test_token_start_end_positions(self, parser):
        tokens = parser.tokenize("go to")
        assert tokens[0].start == 0
        assert tokens[0].end == 2
        assert tokens[1].start == 3

    def test_hyphenated_words(self, parser):
        tokens = parser.tokenize("star-board")
        assert any("star-board" in t.text for t in tokens)


# ===================================================================
# NLParser.parse
# ===================================================================

class TestParse:
    def test_parse_basic_command(self, parser):
        result = parser.parse("navigate to waypoint alpha")
        assert result.root == "navigate"
        assert len(result.tokens) > 0
        assert result.confidence > 0

    def test_parse_empty(self, parser):
        result = parser.parse("")
        assert result.tokens == []
        assert result.root == ""
        assert result.confidence == 0.0

    def test_parse_confidence_increases_with_entities(self, parser):
        r1 = parser.parse("go")
        r2 = parser.parse("navigate to waypoint alpha at 5 knots")
        assert r2.confidence >= r1.confidence

    def test_parse_no_verb(self, parser):
        result = parser.parse("alpha bravo charlie")
        assert result.root == ""


# ===================================================================
# NLParser.extract_entities
# ===================================================================

class TestExtractEntities:
    def test_extract_speed(self, parser):
        entities = parser.extract_entities("set speed to 5 knots")
        types = [e.entity_type for e in entities]
        assert "speed" in types
        speed_entity = next(e for e in entities if e.entity_type == "speed")
        assert speed_entity.value == 5.0

    def test_extract_heading(self, parser):
        entities = parser.extract_entities("set heading to 90 degrees")
        types = [e.entity_type for e in entities]
        assert "heading" in types

    def test_extract_duration(self, parser):
        entities = parser.extract_entities("hold position for 30 minutes")
        types = [e.entity_type for e in entities]
        assert "duration" in types

    def test_extract_distance(self, parser):
        entities = parser.extract_entities("navigate 5 nautical miles")
        types = [e.entity_type for e in entities]
        assert "distance" in types
        dist_entity = next(e for e in entities if e.entity_type == "distance")
        assert dist_entity.value == 5.0

    def test_extract_numbers(self, parser):
        entities = parser.extract_entities("3 items")
        types = [e.entity_type for e in entities]
        assert "number" in types

    def test_extract_multiple_speeds(self, parser):
        entities = parser.extract_entities("speed 5 knots then 10 knots")
        speeds = [e for e in entities if e.entity_type == "speed"]
        assert len(speeds) == 2

    def test_entities_sorted_by_position(self, parser):
        entities = parser.extract_entities("speed 5 knots heading 90 degrees")
        positions = [e.start for e in entities]
        assert positions == sorted(positions)

    def test_no_entities(self, parser):
        entities = parser.extract_entities("hello world")
        # "hello" and "world" don't match any entity patterns — only generic numbers
        types = [e.entity_type for e in entities]
        assert "speed" not in types
        assert "heading" not in types

    def test_entity_confidence_default(self, parser):
        entities = parser.extract_entities("5 knots")
        assert all(e.confidence == 1.0 for e in entities)

    def test_coordinate_entity(self, parser):
        entities = parser.extract_entities("go to 37.5, -122.3")
        types = [e.entity_type for e in entities]
        assert "coordinate" in types


# ===================================================================
# NLParser.extract_numbers
# ===================================================================

class TestExtractNumbers:
    def test_single_number(self, parser):
        nums = parser.extract_numbers("speed 5 knots")
        assert 5.0 in nums

    def test_multiple_numbers(self, parser):
        nums = parser.extract_numbers("heading 90 and speed 5")
        assert 90.0 in nums
        assert 5.0 in nums

    def test_negative_number(self, parser):
        nums = parser.extract_numbers("-122.5")
        assert -122.5 in nums

    def test_decimal_number(self, parser):
        nums = parser.extract_numbers("depth 45.7 meters")
        assert 45.7 in nums

    def test_no_numbers(self, parser):
        nums = parser.extract_numbers("navigate to alpha")
        assert nums == []

    def test_zero(self, parser):
        nums = parser.extract_numbers("speed 0 knots")
        assert 0.0 in nums


# ===================================================================
# NLParser.extract_coordinates
# ===================================================================

class TestExtractCoordinates:
    def test_decimal_coordinates(self, parser):
        coords = parser.extract_coordinates("navigate to 37.7749, -122.4194")
        assert len(coords) >= 1
        lat, lon = coords[0]
        assert abs(lat - 37.7749) < 0.01
        assert abs(lon - (-122.4194)) < 0.01

    def test_no_coordinates(self, parser):
        coords = parser.extract_coordinates("navigate to waypoint alpha")
        assert coords == []

    def test_coordinate_validation_range(self, parser):
        # Valid lat/lon
        coords = parser.extract_coordinates("45.0, 90.0")
        assert len(coords) >= 1
        # Invalid lat > 90 (should not match)
        coords2 = parser.extract_coordinates("200.0, 300.0")
        assert coords2 == []

    def test_dms_coordinates(self, parser):
        coords = parser.extract_coordinates("37°45'23\"N 122°27'00\"W")
        assert len(coords) >= 1

    def test_decimal_degree_with_direction(self, parser):
        coords = parser.extract_coordinates("37.5°N, 122.5°W")
        assert len(coords) >= 1
        lat, lon = coords[0]
        assert lat > 0
        assert lon < 0


# ===================================================================
# NLParser.extract_durations
# ===================================================================

class TestExtractDurations:
    def test_minutes(self, parser):
        durations = parser.extract_durations("hold for 30 minutes")
        assert any("30" in d and "minute" in d for d in durations)

    def test_seconds(self, parser):
        durations = parser.extract_durations("wait 45 seconds")
        assert any("45" in d and "second" in d for d in durations)

    def test_hours(self, parser):
        durations = parser.extract_durations("patrol for 2 hours")
        assert any("2" in d and "hour" in d for d in durations)

    def test_multiple_durations(self, parser):
        durations = parser.extract_durations("wait 30 seconds then 5 minutes")
        assert len(durations) == 2

    def test_no_durations(self, parser):
        durations = parser.extract_durations("navigate to alpha")
        assert durations == []

    def test_abbreviation_min(self, parser):
        durations = parser.extract_durations("hold 10 min")
        assert len(durations) >= 1


# ===================================================================
# NLParser.normalize
# ===================================================================

class TestNormalize:
    def test_lowercase(self, parser):
        assert parser.normalize("NAVIGATE To Alpha") == "navigate to alpha"

    def test_collapse_whitespace(self, parser):
        assert parser.normalize("go   to   alpha") == "go to alpha"

    def test_strip_punctuation(self, parser):
        assert parser.normalize("stop!") == "stop"

    def test_expand_nm(self, parser):
        result = parser.normalize("go 5 nm")
        assert "nautical miles" in result

    def test_expand_kn(self, parser):
        result = parser.normalize("speed 5 kn")
        assert "knots" in result

    def test_expand_kts(self, parser):
        result = parser.normalize("speed 10 kts")
        assert "knots" in result

    def test_expand_deg(self, parser):
        result = parser.normalize("heading 90 deg")
        assert "degrees" in result

    def test_expand_hrs(self, parser):
        result = parser.normalize("patrol for 2 hrs")
        assert "hours" in result

    def test_expand_lat_lon(self, parser):
        result = parser.normalize("lat 37 lon 122")
        assert "latitude" in result
        assert "longitude" in result

    def test_empty_string(self, parser):
        assert parser.normalize("") == ""

    def test_whitespace_only(self, parser):
        assert parser.normalize("   ") == ""


# ===================================================================
# NLParser.extract_commands
# ===================================================================

class TestExtractCommands:
    def test_single_command(self, parser):
        cmds = parser.extract_commands("navigate to alpha")
        assert len(cmds) == 1

    def test_semicolon_split(self, parser):
        cmds = parser.extract_commands("go to alpha. set speed 5 knots")
        assert len(cmds) == 2

    def test_then_split(self, parser):
        cmds = parser.extract_commands("go to alpha then set speed 5")
        assert len(cmds) == 2

    def test_and_then_split(self, parser):
        cmds = parser.extract_commands("navigate and then patrol")
        assert len(cmds) == 2

    def test_empty_parts_filtered(self, parser):
        cmds = parser.extract_commands("  ")
        assert cmds == []
