"""
Dialogue Management for NEXUS Marine Robotics Platform.

Manages multi-turn conversations with context tracking, clarification
requests, action confirmation, frustration detection, and context-aware
response generation.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .intent import Intent, IntentType, IntentRecognizer
from .executor import Command, CommandExecutor, CommandPriority, ExecutionResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DialogueState:
    """State of an ongoing dialogue session."""
    intent_history: list[Intent] = field(default_factory=list)
    current_intent: Optional[Intent] = None
    pending_clarification: Optional[str] = None
    context: dict[str, Any] = field(default_factory=dict)
    turn_count: int = 0
    session_id: str = ""
    started_at: float = 0.0
    last_activity: float = 0.0
    user_corrections: int = 0
    failed_commands: int = 0
    clarification_count: int = 0

    def __post_init__(self):
        if not self.session_id:
            self.session_id = _generate_session_id()
        if not self.started_at:
            self.started_at = time.time()
        if not self.last_activity:
            self.last_activity = time.time()


def _generate_session_id() -> str:
    import uuid
    return uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# DialogueManager
# ---------------------------------------------------------------------------

class DialogueManager:
    """Multi-turn dialogue manager for NL command interface.

    Tracks conversation context, handles clarification requests,
    manages confirmations, detects user frustration, and generates
    natural language responses.
    """

    # Frustration indicators
    _FRUSTRATION_PATTERNS = [
        re.compile(r"\b(no!|wrong|incorrect|not what|forget it|never mind|cancel|stop asking)\b", re.IGNORECASE),
        re.compile(r"\b(stupid|useless|broken|dumb|annoying)\b", re.IGNORECASE),
        re.compile(r"[!]{2,}"),  # multiple exclamation marks
        re.compile(r"\b(again|repeat|I said|I told you)\b", re.IGNORECASE),
    ]

    # Correction patterns
    _CORRECTION_PATTERNS = [
        re.compile(r"\b(no,?\s*I meant|i meant)\b", re.IGNORECASE),
        re.compile(r"\b(actually|rather|instead)\b", re.IGNORECASE),
        re.compile(r"\b(cancel that|undo|forget)\b", re.IGNORECASE),
        re.compile(r"\bnot\b.*, (but|instead)\b", re.IGNORECASE),
    ]

    # Affirmation patterns
    _AFFIRMATION_PATTERNS = [
        re.compile(r"^(yes|yeah|yep|sure|ok|okay|confirm|affirmative|correct|right|do it|go ahead|please do|proceed)$", re.IGNORECASE),
        re.compile(r"^(yeah|yep|sure|ok|okay).*$", re.IGNORECASE),
    ]

    # Negation patterns
    _NEGATION_PATTERNS = [
        re.compile(r"^(no|nope|nah|cancel|stop|abort|negative|don'?t|do not)$", re.IGNORECASE),
        re.compile(r"^(no|nope|nah|cancel|stop|abort|negative).*$", re.IGNORECASE),
    ]

    def __init__(
        self,
        intent_recognizer: Optional[IntentRecognizer] = None,
        executor: Optional[CommandExecutor] = None,
    ) -> None:
        self._recognizer = intent_recognizer or IntentRecognizer()
        self._executor = executor or CommandExecutor()

    # -- public API --------------------------------------------------------

    def process_input(self, text: str, state: Optional[DialogueState] = None) -> tuple[str, DialogueState]:
        """Process user input and return (response, updated_state)."""
        if state is None:
            state = DialogueState()

        state.turn_count += 1
        state.last_activity = time.time()
        text_clean = text.strip()

        if not text_clean:
            return self._empty_response(state), state

        # Check for pending clarification
        if state.pending_clarification:
            return self._handle_clarification_response(text_clean, state)

        # Detect corrections
        if self._is_correction(text_clean):
            state = self.handle_correction(state, text_clean)
            return "I've noted the correction. How can I help?", state

        # Recognize intent
        intent = self._recognizer.recognize(text_clean, state.context)
        state.intent_history.append(intent)
        state.current_intent = intent

        # Update context
        state = self.manage_context(state, {"last_user_input": text_clean, "last_intent": intent.type})

        # Handle based on intent type
        if intent.type == IntentType.UNKNOWN:
            state.clarification_count += 1
            question = self.ask_clarification(intent)
            state.pending_clarification = question
            return question, state

        # Build and optionally execute command
        if intent.confidence >= 0.3:
            confirmation = self.confirm_action(intent)
            cmd = Command(
                intent=intent,
                parameters=intent.slots,
                priority=self._intent_to_priority(intent.type),
            )
            result = self._executor.execute(cmd)

            response = self.build_response(intent, result)
            if result.success:
                state.context["last_command_result"] = result
            else:
                state.failed_commands += 1
        else:
            state.clarification_count += 1
            response = self.ask_clarification(intent)
            state.pending_clarification = response

        # Check frustration periodically
        if state.turn_count % 3 == 0 and self.detect_user_frustration(state):
            response = "I sense you may be frustrated. How can I better assist you? " + response

        return response, state

    def ask_clarification(self, ambiguous_intent: Intent) -> str:
        """Generate a clarification question for an ambiguous intent."""
        if ambiguous_intent.type == IntentType.UNKNOWN:
            return "I'm not sure what you'd like me to do. Could you rephrase your command? For example: 'navigate to waypoint alpha' or 'set speed to 5 knots'."

        if ambiguous_intent.type == IntentType.NAVIGATE:
            if not ambiguous_intent.slots.get("destination"):
                return "Where would you like to navigate? Please provide a destination or waypoint name."

        if ambiguous_intent.type == IntentType.SET_SPEED:
            if not ambiguous_intent.slots.get("speed") and not ambiguous_intent.slots.get("level"):
                return "What speed would you like to set? You can say something like '5 knots' or 'half speed'."

        if ambiguous_intent.type == IntentType.SET_HEADING:
            if not ambiguous_intent.slots.get("heading_degrees") and not ambiguous_intent.slots.get("direction"):
                return "What heading should I set? You can give a degree value (0-360) or a direction like 'north' or 'southeast'."

        if ambiguous_intent.type == IntentType.CONFIGURE:
            if not ambiguous_intent.slots.get("parameter"):
                return "What would you like to configure? For example: 'enable sonar' or 'set mode to survey'."

        if ambiguous_intent.type == IntentType.PATROL:
            if not ambiguous_intent.slots.get("zone"):
                return "Which area would you like me to patrol? Please specify a zone name or coordinates."

        if ambiguous_intent.type == IntentType.SURVEY:
            if not ambiguous_intent.slots.get("area"):
                return "What area would you like me to survey? Please specify the survey area."

        if ambiguous_intent.confidence < 0.3:
            return f"I think you want to {ambiguous_intent.type.value.replace('_', ' ')}, but I'm not very confident. Could you confirm?"

        return f"Could you provide more details for the {ambiguous_intent.type.value.replace('_', ' ')} command?"

    def confirm_action(self, intent: Intent) -> str:
        """Generate an action confirmation message."""
        itype = intent.type
        slots = intent.slots

        confirmations: dict[IntentType, str] = {
            IntentType.NAVIGATE: f"Navigating to {slots.get('destination', 'the specified destination')}",
            IntentType.STATION_KEEP: "Engaging station keeping mode",
            IntentType.PATROL: f"Beginning patrol of {slots.get('zone', 'the designated area')}",
            IntentType.SURVEY: f"Starting survey of {slots.get('area', 'the designated area')}",
            IntentType.EMERGENCY_STOP: "⚠️ EXECUTING EMERGENCY STOP — all propulsion halted",
            IntentType.RETURN_HOME: "Returning to home position",
            IntentType.SET_SPEED: f"Setting speed to {slots.get('speed', slots.get('level', 'the specified value'))}",
            IntentType.SET_HEADING: f"Setting heading to {slots.get('heading_degrees', slots.get('direction', 'the specified value'))}",
            IntentType.QUERY_STATUS: "Retrieving current status...",
            IntentType.CONFIGURE: f"Applying configuration: {slots.get('parameter', 'settings')}",
            IntentType.UNKNOWN: "Cannot confirm unknown action",
        }
        return confirmations.get(itype, "Action confirmed")

    def handle_correction(self, state: DialogueState, correction: str) -> DialogueState:
        """Handle a user correction and return updated state."""
        state.user_corrections += 1
        state.pending_clarification = None

        # Clear the last intent if correcting
        if state.intent_history:
            state.current_intent = None

        # Update context with correction info
        state.context["last_correction"] = correction
        state.context["correction_count"] = state.user_corrections

        return state

    def build_response(self, action: Intent, result: ExecutionResult) -> str:
        """Build a natural language response for an action and its result."""
        if result.success:
            base = result.message
            if result.side_effects:
                side = ", ".join(result.side_effects)
                base = f"{base}. Side effects: {side}"
            return base
        else:
            return f"Error: {result.message}"

    def manage_context(self, state: DialogueState, new_info: dict[str, Any]) -> DialogueState:
        """Update dialogue state with new information."""
        state.context.update(new_info)
        # Keep context bounded
        if len(state.context) > 50:
            # Remove oldest non-essential keys
            essential = {"last_user_input", "last_intent", "mode", "vessel_position"}
            keys = list(state.context.keys())
            for k in keys:
                if k not in essential:
                    del state.context[k]
                    break
        return state

    def detect_user_frustration(self, state: DialogueState) -> bool:
        """Detect if the user appears frustrated based on conversation history."""
        # Check recent inputs for frustration markers
        recent_inputs = state.context.get("recent_inputs", [])
        if isinstance(recent_inputs, list):
            for inp in recent_inputs[-3:]:
                if isinstance(inp, str):
                    for pat in self._FRUSTRATION_PATTERNS:
                        if pat.search(inp):
                            return True

        # Heuristic: high correction and failure rate
        if state.turn_count > 5:
            correction_rate = state.user_corrections / state.turn_count
            failure_rate = state.failed_commands / max(1, state.turn_count - 1)
            if correction_rate > 0.5 or failure_rate > 0.5:
                return True

        # Too many clarifications
        if state.clarification_count > 3:
            return True

        return False

    def compute_dialogue_confidence(self, state: DialogueState) -> float:
        """Compute overall confidence in the dialogue state."""
        if state.turn_count == 0:
            return 0.5

        score = 0.5

        # Recent intent confidence
        if state.current_intent:
            score = state.current_intent.confidence * 0.6 + score * 0.4

        # Penalty for pending clarification
        if state.pending_clarification:
            score *= 0.7

        # Penalty for user corrections
        correction_penalty = state.user_corrections * 0.05
        score = max(0.0, score - correction_penalty)

        # Penalty for failed commands
        failure_penalty = state.failed_commands * 0.08
        score = max(0.0, score - failure_penalty)

        return round(min(1.0, score), 4)

    # -- internal helpers ---------------------------------------------------

    def _handle_clarification_response(self, text: str, state: DialogueState) -> tuple[str, DialogueState]:
        """Handle a response to a pending clarification."""
        # Check for affirmation
        if self._is_affirmation(text):
            state.pending_clarification = None
            if state.current_intent and state.current_intent.type != IntentType.UNKNOWN:
                cmd = Command(
                    intent=state.current_intent,
                    parameters=state.current_intent.slots,
                    priority=self._intent_to_priority(state.current_intent.type),
                )
                result = self._executor.execute(cmd)
                response = self.build_response(state.current_intent, result)
                if not result.success:
                    state.failed_commands += 1
                return response, state
            return "Acknowledged. What would you like me to do?", state

        # Check for negation
        if self._is_negation(text):
            state.pending_clarification = None
            state.current_intent = None
            return "Command cancelled. How else can I help?", state

        # It's new input — clear clarification and reprocess
        state.pending_clarification = None
        state.turn_count -= 1  # will be incremented in process_input
        return self.process_input(text, state)

    def _is_affirmation(self, text: str) -> bool:
        text_clean = text.strip().rstrip(".!?,")
        return any(p.match(text_clean) for p in self._AFFIRMATION_PATTERNS)

    def _is_negation(self, text: str) -> bool:
        text_clean = text.strip().rstrip(".!?,")
        return any(p.match(text_clean) for p in self._NEGATION_PATTERNS)

    def _is_correction(self, text: str) -> bool:
        return any(p.search(text) for p in self._CORRECTION_PATTERNS)

    def _empty_response(self, state: DialogueState) -> str:
        if state.pending_clarification:
            return state.pending_clarification
        return "I'm listening. Please give me a command."

    @staticmethod
    def _intent_to_priority(itype: IntentType) -> CommandPriority:
        priority_map = {
            IntentType.EMERGENCY_STOP: CommandPriority.CRITICAL,
            IntentType.NAVIGATE: CommandPriority.HIGH,
            IntentType.RETURN_HOME: CommandPriority.HIGH,
            IntentType.PATROL: CommandPriority.NORMAL,
            IntentType.SURVEY: CommandPriority.NORMAL,
            IntentType.STATION_KEEP: CommandPriority.NORMAL,
            IntentType.SET_SPEED: CommandPriority.NORMAL,
            IntentType.SET_HEADING: CommandPriority.NORMAL,
            IntentType.QUERY_STATUS: CommandPriority.LOW,
            IntentType.CONFIGURE: CommandPriority.NORMAL,
            IntentType.UNKNOWN: CommandPriority.LOW,
        }
        return priority_map.get(itype, CommandPriority.NORMAL)
