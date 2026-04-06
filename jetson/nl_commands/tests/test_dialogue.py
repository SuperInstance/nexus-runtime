"""Tests for DialogueManager — input processing, clarification, confirmation, correction, context."""

import pytest
from jetson.nl_commands.intent import Intent, IntentType, IntentRecognizer
from jetson.nl_commands.executor import CommandExecutor, CommandPriority
from jetson.nl_commands.dialogue import DialogueState, DialogueManager


@pytest.fixture
def recognizer():
    return IntentRecognizer()


@pytest.fixture
def executor():
    return CommandExecutor()


@pytest.fixture
def dm(recognizer, executor):
    return DialogueManager(intent_recognizer=recognizer, executor=executor)


@pytest.fixture
def fresh_state():
    return DialogueState()


# ===================================================================
# DialogueState
# ===================================================================

class TestDialogueState:
    def test_default_state(self):
        state = DialogueState()
        assert state.intent_history == []
        assert state.current_intent is None
        assert state.pending_clarification is None
        assert state.turn_count == 0
        assert state.session_id != ""
        assert state.started_at > 0

    def test_custom_session_id(self):
        state = DialogueState(session_id="abc123")
        assert state.session_id == "abc123"

    def test_mutable_fields(self):
        state = DialogueState()
        state.turn_count = 5
        assert state.turn_count == 5


# ===================================================================
# DialogueManager.process_input
# ===================================================================

class TestProcessInput:
    def test_basic_navigate(self, dm, fresh_state):
        response, state = dm.process_input("navigate to waypoint alpha", fresh_state)
        assert state.turn_count == 1
        assert isinstance(response, str)
        assert len(response) > 0

    def test_empty_input(self, dm, fresh_state):
        response, state = dm.process_input("", fresh_state)
        assert "listening" in response.lower()

    def test_whitespace_input(self, dm, fresh_state):
        response, state = dm.process_input("   ", fresh_state)
        assert "listening" in response.lower()

    def test_state_updated_each_turn(self, dm, fresh_state):
        dm.process_input("navigate to alpha", fresh_state)
        assert fresh_state.turn_count == 1
        dm.process_input("set speed 5 knots", fresh_state)
        assert fresh_state.turn_count == 2

    def test_intent_history_grows(self, dm, fresh_state):
        dm.process_input("navigate to alpha", fresh_state)
        dm.process_input("set heading 90", fresh_state)
        assert len(fresh_state.intent_history) == 2

    def test_unknown_intent_triggers_clarification(self, dm, fresh_state):
        response, state = dm.process_input("xyzzyplugh", fresh_state)
        assert state.pending_clarification is not None

    def test_clarification_then_affirm(self, dm, fresh_state):
        dm.process_input("xyzzyplugh", fresh_state)
        response, state = dm.process_input("yes", fresh_state)
        assert state.pending_clarification is None

    def test_clarification_then_negate(self, dm, fresh_state):
        dm.process_input("xyzzyplugh", fresh_state)
        response, state = dm.process_input("no", fresh_state)
        assert state.pending_clarification is None
        assert "cancel" in response.lower()

    def test_response_string_for_valid_command(self, dm, fresh_state):
        response, _ = dm.process_input("navigate to alpha", fresh_state)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_process_without_state(self, dm):
        response, state = dm.process_input("navigate to alpha")
        assert state.turn_count == 1
        assert state.session_id != ""


# ===================================================================
# DialogueManager.ask_clarification
# ===================================================================

class TestAskClarification:
    def test_unknown_intent_clarification(self, dm):
        intent = Intent(type=IntentType.UNKNOWN, confidence=0.2, raw_text="xyzzy")
        q = dm.ask_clarification(intent)
        assert "rephrase" in q.lower() or "sure" in q.lower()

    def test_navigate_missing_dest(self, dm):
        intent = Intent(type=IntentType.NAVIGATE, slots={}, confidence=0.5, raw_text="navigate")
        q = dm.ask_clarification(intent)
        assert "where" in q.lower() or "destination" in q.lower()

    def test_set_speed_missing_value(self, dm):
        intent = Intent(type=IntentType.SET_SPEED, slots={}, confidence=0.5, raw_text="set speed")
        q = dm.ask_clarification(intent)
        assert "speed" in q.lower()

    def test_set_heading_missing_value(self, dm):
        intent = Intent(type=IntentType.SET_HEADING, slots={}, confidence=0.5, raw_text="set heading")
        q = dm.ask_clarification(intent)
        assert "heading" in q.lower()

    def test_configure_missing_param(self, dm):
        intent = Intent(type=IntentType.CONFIGURE, slots={}, confidence=0.5, raw_text="configure")
        q = dm.ask_clarification(intent)
        assert "configure" in q.lower() or "what" in q.lower()

    def test_patrol_missing_zone(self, dm):
        intent = Intent(type=IntentType.PATROL, slots={}, confidence=0.5, raw_text="patrol")
        q = dm.ask_clarification(intent)
        assert "area" in q.lower() or "zone" in q.lower() or "patrol" in q.lower()

    def test_survey_missing_area(self, dm):
        intent = Intent(type=IntentType.SURVEY, slots={}, confidence=0.5, raw_text="survey")
        q = dm.ask_clarification(intent)
        assert "area" in q.lower() or "survey" in q.lower()

    def test_low_confidence_clarification(self, dm):
        intent = Intent(type=IntentType.NAVIGATE, slots={"destination": "alpha"}, confidence=0.2, raw_text="go")
        q = dm.ask_clarification(intent)
        assert "confirm" in q.lower() or "confident" in q.lower()


# ===================================================================
# DialogueManager.confirm_action
# ===================================================================

class TestConfirmAction:
    def test_navigate_confirm(self, dm):
        intent = Intent(type=IntentType.NAVIGATE, slots={"destination": "alpha"})
        c = dm.confirm_action(intent)
        assert "navigat" in c.lower()

    def test_emergency_stop_confirm(self, dm):
        intent = Intent(type=IntentType.EMERGENCY_STOP)
        c = dm.confirm_action(intent)
        assert "emergency" in c.lower() or "stop" in c.lower()

    def test_return_home_confirm(self, dm):
        intent = Intent(type=IntentType.RETURN_HOME)
        c = dm.confirm_action(intent)
        assert "home" in c.lower() or "return" in c.lower()

    def test_station_keep_confirm(self, dm):
        intent = Intent(type=IntentType.STATION_KEEP)
        c = dm.confirm_action(intent)
        assert "station" in c.lower() or "keep" in c.lower()

    def test_patrol_confirm(self, dm):
        intent = Intent(type=IntentType.PATROL, slots={"zone": "harbor"})
        c = dm.confirm_action(intent)
        assert "patrol" in c.lower()

    def test_survey_confirm(self, dm):
        intent = Intent(type=IntentType.SURVEY, slots={"area": "seabed"})
        c = dm.confirm_action(intent)
        assert "survey" in c.lower()

    def test_set_speed_confirm(self, dm):
        intent = Intent(type=IntentType.SET_SPEED, slots={"speed": 5})
        c = dm.confirm_action(intent)
        assert "speed" in c.lower()

    def test_set_heading_confirm(self, dm):
        intent = Intent(type=IntentType.SET_HEADING, slots={"heading_degrees": 90})
        c = dm.confirm_action(intent)
        assert "heading" in c.lower()

    def test_query_status_confirm(self, dm):
        intent = Intent(type=IntentType.QUERY_STATUS)
        c = dm.confirm_action(intent)
        assert "status" in c.lower() or "retriev" in c.lower()

    def test_unknown_confirm(self, dm):
        intent = Intent(type=IntentType.UNKNOWN)
        c = dm.confirm_action(intent)
        assert "unknown" in c.lower()


# ===================================================================
# DialogueManager.handle_correction
# ===================================================================

class TestHandleCorrection:
    def test_correction_increments_counter(self, dm, fresh_state):
        state = dm.handle_correction(fresh_state, "no, I meant go to bravo")
        assert state.user_corrections == 1

    def test_correction_clears_pending(self, dm, fresh_state):
        fresh_state.pending_clarification = "some question"
        state = dm.handle_correction(fresh_state, "actually, go to bravo")
        assert state.pending_clarification is None

    def test_multiple_corrections(self, dm, fresh_state):
        dm.handle_correction(fresh_state, "no, bravo")
        dm.handle_correction(fresh_state, "actually charlie")
        assert fresh_state.user_corrections == 2

    def test_correction_updates_context(self, dm, fresh_state):
        state = dm.handle_correction(fresh_state, "I meant navigate")
        assert "last_correction" in state.context


# ===================================================================
# DialogueManager.build_response
# ===================================================================

class TestBuildResponse:
    def test_success_response(self, dm):
        intent = _make_intent(IntentType.NAVIGATE)
        result = _make_result(True, "Navigating to alpha")
        response = dm.build_response(intent, result)
        assert "navigat" in response.lower()

    def test_error_response(self, dm):
        intent = _make_intent(IntentType.UNKNOWN)
        result = _make_result(False, "Unknown command")
        response = dm.build_response(intent, result)
        assert "error" in response.lower()

    def test_side_effects_included(self, dm):
        intent = _make_intent(IntentType.NAVIGATE)
        result = _make_result(True, "Done", side_effects=["propulsion_started"])
        response = dm.build_response(intent, result)
        assert "side effect" in response.lower() or "propulsion_started" in response


# ===================================================================
# DialogueManager.manage_context
# ===================================================================

class TestManageContext:
    def test_context_updated(self, dm, fresh_state):
        state = dm.manage_context(fresh_state, {"key": "value"})
        assert state.context["key"] == "value"

    def test_context_overwrite(self, dm, fresh_state):
        fresh_state.context["key"] = "old"
        state = dm.manage_context(fresh_state, {"key": "new"})
        assert state.context["key"] == "new"

    def test_context_bounded(self, dm, fresh_state):
        for i in range(60):
            fresh_state = dm.manage_context(fresh_state, {f"key_{i}": i})
        # manage_context should keep it bounded
        assert len(fresh_state.context) < 60


# ===================================================================
# DialogueManager.detect_user_frustration
# ===================================================================

class TestDetectUserFrustration:
    def test_no_frustration_initial(self, dm, fresh_state):
        assert not dm.detect_user_frustration(fresh_state)

    def test_frustration_from_corrections(self, dm, fresh_state):
        fresh_state.turn_count = 10
        fresh_state.user_corrections = 6
        assert dm.detect_user_frustration(fresh_state)

    def test_frustration_from_failures(self, dm, fresh_state):
        fresh_state.turn_count = 10
        fresh_state.failed_commands = 6
        assert dm.detect_user_frustration(fresh_state)

    def test_frustration_from_clarifications(self, dm, fresh_state):
        fresh_state.clarification_count = 5
        assert dm.detect_user_frustration(fresh_state)

    def test_no_frustration_normal(self, dm, fresh_state):
        fresh_state.turn_count = 3
        fresh_state.user_corrections = 0
        fresh_state.failed_commands = 0
        assert not dm.detect_user_frustration(fresh_state)

    def test_frustration_from_context_inputs(self, dm, fresh_state):
        fresh_state.context["recent_inputs"] = ["no!", "wrong!", "this is stupid"]
        assert dm.detect_user_frustration(fresh_state)

    def test_no_frustration_from_positive_inputs(self, dm, fresh_state):
        fresh_state.context["recent_inputs"] = ["navigate to alpha", "set speed 5", "great"]
        assert not dm.detect_user_frustration(fresh_state)


# ===================================================================
# DialogueManager.compute_dialogue_confidence
# ===================================================================

class TestComputeDialogueConfidence:
    def test_initial_confidence(self, dm, fresh_state):
        conf = dm.compute_dialogue_confidence(fresh_state)
        assert conf == 0.5

    def test_high_confidence_with_good_intent(self, dm, fresh_state):
        fresh_state.turn_count = 1
        fresh_state.current_intent = Intent(type=IntentType.NAVIGATE, confidence=0.95)
        conf = dm.compute_dialogue_confidence(fresh_state)
        assert conf > 0.7

    def test_low_confidence_with_pending_clarification(self, dm, fresh_state):
        fresh_state.turn_count = 1
        fresh_state.current_intent = Intent(type=IntentType.NAVIGATE, confidence=0.8)
        fresh_state.pending_clarification = "Where to?"
        conf = dm.compute_dialogue_confidence(fresh_state)
        conf_no_pending = dm.compute_dialogue_confidence(DialogueState(
            turn_count=1,
            current_intent=Intent(type=IntentType.NAVIGATE, confidence=0.8),
        ))
        assert conf < conf_no_pending

    def test_penalty_for_corrections(self, dm, fresh_state):
        fresh_state.turn_count = 5
        fresh_state.user_corrections = 5
        conf = dm.compute_dialogue_confidence(fresh_state)
        assert conf < 0.5

    def test_confidence_bounded_zero_one(self, dm, fresh_state):
        fresh_state.turn_count = 1
        fresh_state.user_corrections = 100
        conf = dm.compute_dialogue_confidence(fresh_state)
        assert 0.0 <= conf <= 1.0

    def test_penalty_for_failures(self, dm, fresh_state):
        fresh_state.turn_count = 5
        fresh_state.failed_commands = 4
        conf = dm.compute_dialogue_confidence(fresh_state)
        assert conf < 0.5


# ===================================================================
# Integration: multi-turn dialogue
# ===================================================================

class TestMultiTurnDialogue:
    def test_full_conversation(self, dm, fresh_state):
        response1, state = dm.process_input("navigate to waypoint alpha", fresh_state)
        assert state.turn_count == 1

        response2, state = dm.process_input("set speed to 5 knots", state)
        assert state.turn_count == 2

        response3, state = dm.process_input("what is my status", state)
        assert state.turn_count == 3
        assert len(state.intent_history) == 3

    def test_clarification_flow(self, dm, fresh_state):
        # Trigger unknown intent
        resp1, state = dm.process_input("do the thing", fresh_state)
        assert state.pending_clarification is not None

        # Affirm
        resp2, state = dm.process_input("yes", state)
        assert state.pending_clarification is None

    def test_correction_flow(self, dm, fresh_state):
        resp1, state = dm.process_input("navigate to alpha", fresh_state)
        assert state.user_corrections == 0

        resp2, state = dm.process_input("no, I meant go to bravo", state)
        assert state.user_corrections == 1


# ===================================================================
# Helpers
# ===================================================================

def _make_intent(itype: IntentType, **slots):
    return Intent(type=itype, slots=slots, confidence=0.9, raw_text="")


def _make_result(success: bool, message: str, side_effects=None):
    from jetson.nl_commands.executor import ExecutionResult
    return ExecutionResult(
        success=success,
        message=message,
        side_effects=side_effects or [],
    )
