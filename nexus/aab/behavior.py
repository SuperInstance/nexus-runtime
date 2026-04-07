"""
NEXUS Autonomous Agent Behavior (AAB) — behavior state machine and A2A opcodes.

The AAB framework provides a state machine for autonomous agents with 5 states
and 29 agent-to-agent (A2A) opcodes for inter-agent communication.

States:
    IDLE → ACTIVE, NEGOTIATING, DELEGATING
    ACTIVE → IDLE, REPORTING
    NEGOTIATING → ACTIVE, IDLE
    DELEGATING → ACTIVE, IDLE
    REPORTING → IDLE, ACTIVE
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# A2A Opcodes (29)
# ---------------------------------------------------------------------------

class A2AOpcodes(enum.IntEnum):
    """29 agent-to-agent opcodes for NEXUS marine coordination."""

    # -- task management --
    REQUEST_TASK = 0x01
    OFFER_CAPABILITY = 0x02
    ASSIGN_TASK = 0x03
    ACCEPT_TASK = 0x04
    REJECT_TASK = 0x05
    COMPLETE_TASK = 0x06
    FAIL_TASK = 0x07
    CANCEL_TASK = 0x08

    # -- negotiation --
    NEGOTIATE = 0x10
    PROPOSE = 0x11
    COUNTER_PROPOSE = 0x12
    ACCEPT_PROPOSAL = 0x13
    REJECT_PROPOSAL = 0x14

    # -- delegation --
    DELEGATE = 0x20
    DELEGATE_RESULT = 0x21
    REQUEST_ASSISTANCE = 0x22
    OFFER_ASSISTANCE = 0x23
    ASSIST_COMPLETE = 0x24

    # -- reporting --
    REPORT_STATUS = 0x30
    REPORT_TELEMETRY = 0x31
    REPORT_ANOMALY = 0x32
    REPORT_COMPLETE = 0x33

    # -- coordination --
    SYNC_CLOCK = 0x40
    SHARE_MAP = 0x41
    SHARE_PATH = 0x42
    FORMATION_UPDATE = 0x43
    HANDOFF = 0x44
    EMERGENCY_STOP = 0x45
    PING = 0x46
    PONG = 0x47


# ---------------------------------------------------------------------------
# Behavior states
# ---------------------------------------------------------------------------

class BehaviorState(enum.Enum):
    """Agent behavior states."""

    IDLE = "idle"
    ACTIVE = "active"
    NEGOTIATING = "negotiating"
    DELEGATING = "delegating"
    REPORTING = "reporting"


# Valid state transitions
_VALID_TRANSITIONS: Dict[BehaviorState, Set[BehaviorState]] = {
    BehaviorState.IDLE: {BehaviorState.ACTIVE, BehaviorState.NEGOTIATING, BehaviorState.DELEGATING},
    BehaviorState.ACTIVE: {BehaviorState.IDLE, BehaviorState.REPORTING, BehaviorState.NEGOTIATING},
    BehaviorState.NEGOTIATING: {BehaviorState.ACTIVE, BehaviorState.IDLE},
    BehaviorState.DELEGATING: {BehaviorState.ACTIVE, BehaviorState.IDLE},
    BehaviorState.REPORTING: {BehaviorState.IDLE, BehaviorState.ACTIVE},
}


# ---------------------------------------------------------------------------
# A2A Message
# ---------------------------------------------------------------------------

@dataclass
class A2AMessage:
    """An agent-to-agent message."""

    opcode: A2AOpcodes
    sender: str = ""
    receiver: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    in_reply_to: str = ""
    priority: int = 0  # 0=low, 5=normal, 10=critical

    def reply(self, opcode: A2AOpcodes, payload: Optional[Dict[str, Any]] = None) -> "A2AMessage":
        """Create a reply message."""
        return A2AMessage(
            opcode=opcode,
            sender=self.receiver,
            receiver=self.sender,
            payload=payload or {},
            conversation_id=self.conversation_id,
            in_reply_to=self.conversation_id,
        )


@dataclass
class A2AResponse:
    """Response from a message handler."""

    accepted: bool = True
    reply_opcode: Optional[A2AOpcodes] = None
    reply_payload: Dict[str, Any] = field(default_factory=dict)
    state_transition: Optional[BehaviorState] = None
    error: str = ""


# ---------------------------------------------------------------------------
# Message handler type
# ---------------------------------------------------------------------------

MessageHandler = Callable[[A2AMessage], A2AResponse]


# ---------------------------------------------------------------------------
# Behavior Engine
# ---------------------------------------------------------------------------

class BehaviorEngine:
    """State machine and message dispatcher for autonomous agent behavior.

    Usage::

        engine = BehaviorEngine(agent_id="AUV-001")
        engine.register_handler(A2AOpcodes.REQUEST_TASK, handle_task_request)
        engine.transition(BehaviorState.ACTIVE)
        response = engine.handle_message(msg)
    """

    def __init__(
        self,
        agent_id: str = "",
        initial_state: BehaviorState = BehaviorState.IDLE,
    ) -> None:
        self.agent_id = agent_id
        self.state = initial_state
        self._handlers: Dict[A2AOpcodes, MessageHandler] = {}
        self._message_log: List[A2AMessage] = []
        self._response_log: List[A2AResponse] = []
        self._task_queue: List[A2AMessage] = []
        self._current_task: Optional[A2AMessage] = None
        self._state_history: List[Tuple[BehaviorState, float]] = [(initial_state, time.time())]
        self._conversations: Dict[str, List[A2AMessage]] = {}

    @property
    def current_task(self) -> Optional[A2AMessage]:
        return self._current_task

    @property
    def message_count(self) -> int:
        return len(self._message_log)

    @property
    def pending_tasks(self) -> int:
        return len(self._task_queue)

    # ----- state management -----

    def transition(self, new_state: BehaviorState) -> bool:
        """Attempt a state transition. Returns True if successful."""
        allowed = _VALID_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            return False
        self.state = new_state
        self._state_history.append((new_state, time.time()))
        return True

    def force_transition(self, new_state: BehaviorState) -> None:
        """Force a state transition (bypasses validation)."""
        self.state = new_state
        self._state_history.append((new_state, time.time()))

    def get_state_history(self) -> List[Tuple[BehaviorState, float]]:
        """Get the history of state transitions."""
        return list(self._state_history)

    def is_valid_transition(self, target: BehaviorState) -> bool:
        """Check if transitioning to *target* is valid from current state."""
        return target in _VALID_TRANSITIONS.get(self.state, set())

    # ----- handler registration -----

    def register_handler(self, opcode: A2AOpcodes, handler: MessageHandler) -> None:
        """Register a message handler for an opcode."""
        self._handlers[opcode] = handler

    def unregister_handler(self, opcode: A2AOpcodes) -> bool:
        """Remove a handler. Returns True if it existed."""
        return self._handlers.pop(opcode, None) is not None

    def has_handler(self, opcode: A2AOpcodes) -> bool:
        """Check if a handler is registered for an opcode."""
        return opcode in self._handlers

    # ----- message handling -----

    def handle_message(self, msg: A2AMessage) -> A2AResponse:
        """Process an incoming A2A message and return a response."""
        self._message_log.append(msg)

        # Track conversation
        conv_id = msg.conversation_id
        if conv_id not in self._conversations:
            self._conversations[conv_id] = []
        self._conversations[conv_id].append(msg)

        # Dispatch to handler
        handler = self._handlers.get(msg.opcode)
        if handler is None:
            response = A2AResponse(accepted=False, error=f"No handler for {msg.opcode.name}")
        else:
            try:
                response = handler(msg)
            except Exception as e:
                response = A2AResponse(accepted=False, error=str(e))

        self._response_log.append(response)

        # Apply state transition if requested
        if response.state_transition is not None:
            self.transition(response.state_transition)

        return response

    def send_message(
        self,
        opcode: A2AOpcodes,
        receiver: str,
        payload: Optional[Dict[str, Any]] = None,
        conversation_id: str = "",
        priority: int = 5,
    ) -> A2AMessage:
        """Create and log an outgoing message."""
        msg = A2AMessage(
            opcode=opcode,
            sender=self.agent_id,
            receiver=receiver,
            payload=payload or {},
            conversation_id=conversation_id or str(uuid.uuid4())[:8],
            priority=priority,
        )
        self._message_log.append(msg)
        return msg

    # ----- task management -----

    def enqueue_task(self, msg: A2AMessage) -> None:
        """Add a task to the queue."""
        self._task_queue.append(msg)

    def next_task(self) -> Optional[A2AMessage]:
        """Pop the next task from the queue."""
        if self._task_queue:
            self._current_task = self._task_queue.pop(0)
            return self._current_task
        return None

    def complete_current_task(self) -> bool:
        """Mark the current task as complete."""
        if self._current_task:
            self._current_task = None
            return True
        return False

    # ----- queries -----

    def get_conversation(self, conv_id: str) -> List[A2AMessage]:
        """Get all messages in a conversation."""
        return list(self._conversations.get(conv_id, []))

    def get_messages_by_opcode(self, opcode: A2AOpcodes) -> List[A2AMessage]:
        """Get all messages of a given opcode type."""
        return [m for m in self._message_log if m.opcode == opcode]

    def reset(self) -> None:
        """Reset the engine to initial state."""
        self.state = BehaviorState.IDLE
        self._message_log.clear()
        self._response_log.clear()
        self._task_queue.clear()
        self._current_task = None
        self._state_history = [(BehaviorState.IDLE, time.time())]
        self._conversations.clear()
