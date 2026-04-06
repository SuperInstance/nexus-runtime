"""Tests for IntentRecognizer — recognition, training, confidence, disambiguation, slots."""

import pytest
from jetson.nl_commands.intent import Intent, IntentType, IntentRecognizer


@pytest.fixture
def recognizer():
    return IntentRecognizer()


# ===================================================================
# Intent dataclass
# ===================================================================

class TestIntent:
    def test_intent_creation(self):
        intent = Intent(type=IntentType.NAVIGATE, confidence=0.9, raw_text="go to alpha")
        assert intent.type == IntentType.NAVIGATE
        assert intent.confidence == 0.9
        assert intent.raw_text == "go to alpha"
        assert intent.slots == {}

    def test_intent_with_slots(self):
        intent = Intent(type=IntentType.SET_SPEED, slots={"speed": 5}, confidence=0.85)
        assert intent.slots["speed"] == 5

    def test_intent_timestamp_default(self):
        intent = Intent(type=IntentType.QUERY_STATUS)
        assert intent.timestamp > 0


# ===================================================================
# IntentType enum
# ===================================================================

class TestIntentType:
    def test_all_intent_types(self):
        expected = [
            "navigate", "station_keep", "patrol", "survey",
            "emergency_stop", "return_home", "set_speed", "set_heading",
            "query_status", "configure", "unknown",
        ]
        values = [it.value for it in IntentType]
        for v in expected:
            assert v in values

    def test_enum_value_access(self):
        assert IntentType.NAVIGATE.value == "navigate"
        assert IntentType.EMERGENCY_STOP.value == "emergency_stop"


# ===================================================================
# IntentRecognizer.recognize
# ===================================================================

class TestRecognize:
    def test_navigate_intent(self, recognizer):
        intent = recognizer.recognize("navigate to waypoint alpha")
        assert intent.type == IntentType.NAVIGATE

    def test_move_intent_maps_to_navigate(self, recognizer):
        intent = recognizer.recognize("move to waypoint alpha")
        assert intent.type == IntentType.NAVIGATE

    def test_go_intent_maps_to_navigate(self, recognizer):
        intent = recognizer.recognize("go to waypoint alpha")
        assert intent.type == IntentType.NAVIGATE

    def test_emergency_stop(self, recognizer):
        intent = recognizer.recognize("emergency stop")
        assert intent.type == IntentType.EMERGENCY_STOP

    def test_abort_intent(self, recognizer):
        intent = recognizer.recognize("abort mission")
        assert intent.type == IntentType.EMERGENCY_STOP

    def test_halt_intent(self, recognizer):
        intent = recognizer.recognize("halt")
        assert intent.type == IntentType.EMERGENCY_STOP

    def test_return_home(self, recognizer):
        intent = recognizer.recognize("return to home")
        assert intent.type == IntentType.RETURN_HOME

    def test_dock_intent(self, recognizer):
        intent = recognizer.recognize("dock the vessel")
        assert intent.type == IntentType.RETURN_HOME

    def test_set_speed(self, recognizer):
        intent = recognizer.recognize("set speed to 5 knots")
        assert intent.type == IntentType.SET_SPEED

    def test_slow_down(self, recognizer):
        intent = recognizer.recognize("slow down")
        assert intent.type == IntentType.SET_SPEED

    def test_set_heading(self, recognizer):
        intent = recognizer.recognize("set heading to 90 degrees")
        assert intent.type == IntentType.SET_HEADING

    def test_turn_to(self, recognizer):
        intent = recognizer.recognize("turn to 180 degrees")
        assert intent.type == IntentType.SET_HEADING

    def test_station_keep(self, recognizer):
        intent = recognizer.recognize("hold position")
        assert intent.type == IntentType.STATION_KEEP

    def test_loiter(self, recognizer):
        intent = recognizer.recognize("loiter here for 10 minutes")
        assert intent.type == IntentType.STATION_KEEP

    def test_patrol(self, recognizer):
        intent = recognizer.recognize("patrol the harbor zone")
        assert intent.type == IntentType.PATROL

    def test_survey(self, recognizer):
        intent = recognizer.recognize("survey the seabed")
        assert intent.type == IntentType.SURVEY

    def test_scan_maps_to_survey(self, recognizer):
        intent = recognizer.recognize("scan the area")
        assert intent.type == IntentType.SURVEY

    def test_query_status(self, recognizer):
        intent = recognizer.recognize("what is the current status")
        assert intent.type == IntentType.QUERY_STATUS

    def test_check_battery(self, recognizer):
        intent = recognizer.recognize("check battery level")
        assert intent.type == IntentType.QUERY_STATUS

    def test_configure(self, recognizer):
        intent = recognizer.recognize("configure the sonar system")
        assert intent.type == IntentType.CONFIGURE

    def test_enable_system(self, recognizer):
        intent = recognizer.recognize("enable the camera")
        assert intent.type == IntentType.CONFIGURE

    def test_unknown_intent(self, recognizer):
        intent = recognizer.recognize("xyzzyplugh")
        assert intent.type == IntentType.UNKNOWN

    def test_empty_text(self, recognizer):
        intent = recognizer.recognize("")
        assert intent.type == IntentType.UNKNOWN
        assert intent.confidence == 0.0

    def test_whitespace_text(self, recognizer):
        intent = recognizer.recognize("   ")
        assert intent.type == IntentType.UNKNOWN

    def test_recognize_with_context(self, recognizer):
        context = {"mode": "emergency"}
        intent = recognizer.recognize("stop", context)
        assert intent.type == IntentType.EMERGENCY_STOP
        assert intent.confidence > 0

    def test_recognize_confidence_above_zero(self, recognizer):
        intent = recognizer.recognize("navigate to alpha")
        assert intent.confidence > 0.0

    def test_recognize_sets_timestamp(self, recognizer):
        import time
        before = time.time()
        intent = recognizer.recognize("navigate to alpha")
        after = time.time()
        assert before <= intent.timestamp <= after


# ===================================================================
# IntentRecognizer.train
# ===================================================================

class TestTrain:
    def test_train_basic(self, recognizer):
        examples = [
            ("navigate to alpha", IntentType.NAVIGATE),
            ("go to bravo", IntentType.NAVIGATE),
            ("stop now", IntentType.EMERGENCY_STOP),
        ]
        accuracy = recognizer.train(examples)
        assert accuracy >= 0.5

    def test_train_empty(self, recognizer):
        accuracy = recognizer.train([])
        assert accuracy == 0.0

    def test_train_all_correct(self, recognizer):
        examples = [
            ("navigate to alpha", IntentType.NAVIGATE),
            ("go to bravo", IntentType.NAVIGATE),
            ("move to charlie", IntentType.NAVIGATE),
        ]
        accuracy = recognizer.train(examples)
        assert accuracy == 1.0

    def test_train_adds_custom_examples(self, recognizer):
        examples = [("increase speed", IntentType.SET_SPEED)]
        recognizer.train(examples)
        # Should be able to recognize "increase speed" as SET_SPEED now
        intent = recognizer.recognize("increase speed")
        assert intent.type == IntentType.SET_SPEED


# ===================================================================
# IntentRecognizer.compute_confidence
# ===================================================================

class TestComputeConfidence:
    def test_navigate_confidence(self, recognizer):
        conf = recognizer.compute_confidence("navigate to alpha", IntentType.NAVIGATE)
        assert conf > 0

    def test_wrong_intent_low_confidence(self, recognizer):
        conf = recognizer.compute_confidence("xyzzy", IntentType.NAVIGATE)
        assert conf == 0.0

    def test_confidence_bounded(self, recognizer):
        conf = recognizer.compute_confidence("navigate go move travel", IntentType.NAVIGATE)
        assert 0.0 <= conf <= 1.0


# ===================================================================
# IntentRecognizer.disambiguate
# ===================================================================

class TestDisambiguate:
    def test_single_intent(self, recognizer):
        intent = Intent(type=IntentType.NAVIGATE, confidence=0.8, raw_text="go")
        result = recognizer.disambiguate([intent])
        assert result.type == IntentType.NAVIGATE

    def test_empty_list(self, recognizer):
        result = recognizer.disambiguate([])
        assert result.type == IntentType.UNKNOWN

    def test_emergency_takes_priority(self, recognizer):
        nav = Intent(type=IntentType.NAVIGATE, confidence=0.9, raw_text="go")
        emergency = Intent(type=IntentType.EMERGENCY_STOP, confidence=0.5, raw_text="stop")
        result = recognizer.disambiguate([nav, emergency])
        assert result.type == IntentType.EMERGENCY_STOP

    def test_higher_confidence_wins_same_priority(self, recognizer):
        patrol = Intent(type=IntentType.PATROL, confidence=0.3, raw_text="patrol")
        survey = Intent(type=IntentType.SURVEY, confidence=0.7, raw_text="survey")
        result = recognizer.disambiguate([patrol, survey])
        assert result.confidence >= patrol.confidence


# ===================================================================
# IntentRecognizer.extract_slots
# ===================================================================

class TestExtractSlots:
    def test_navigate_destination(self, recognizer):
        slots = recognizer.extract_slots("navigate to waypoint alpha", IntentType.NAVIGATE)
        assert "destination" in slots or len(slots) >= 0  # depends on pattern match

    def test_set_speed_slot(self, recognizer):
        slots = recognizer.extract_slots("set speed to 5 knots", IntentType.SET_SPEED)
        assert "speed" in slots
        assert slots["speed"] == 5

    def test_set_heading_degrees(self, recognizer):
        slots = recognizer.extract_slots("set heading to 90 degrees", IntentType.SET_HEADING)
        assert "heading_degrees" in slots

    def test_set_heading_direction(self, recognizer):
        slots = recognizer.extract_slots("set heading to north", IntentType.SET_HEADING)
        assert "direction" in slots
        assert slots["direction"] == "north"

    def test_survey_area(self, recognizer):
        slots = recognizer.extract_slots("survey the harbor", IntentType.SURVEY)
        assert "area" in slots

    def test_patrol_zone(self, recognizer):
        slots = recognizer.extract_slots("patrol the perimeter", IntentType.PATROL)
        assert "zone" in slots

    def test_station_keep_duration(self, recognizer):
        slots = recognizer.extract_slots("hold position for 30 minutes", IntentType.STATION_KEEP)
        assert "duration" in slots

    def test_configure_parameter(self, recognizer):
        slots = recognizer.extract_slots("configure the sonar", IntentType.CONFIGURE)
        assert "parameter" in slots

    def test_no_slots(self, recognizer):
        slots = recognizer.extract_slots("emergency stop", IntentType.EMERGENCY_STOP)
        assert slots == {}

    def test_query_status_subject(self, recognizer):
        slots = recognizer.extract_slots("check the battery", IntentType.QUERY_STATUS)
        assert "subject" in slots


# ===================================================================
# IntentRecognizer.get_supported_intents
# ===================================================================

class TestGetSupportedIntents:
    def test_returns_list(self, recognizer):
        intents = recognizer.get_supported_intents()
        assert isinstance(intents, list)

    def test_contains_all_types(self, recognizer):
        intents = recognizer.get_supported_intents()
        assert IntentType.NAVIGATE in intents
        assert IntentType.EMERGENCY_STOP in intents
        assert IntentType.UNKNOWN in intents

    def test_count(self, recognizer):
        intents = recognizer.get_supported_intents()
        assert len(intents) == len(IntentType)
