"""Tests for the NEXUS AAB Behavior Engine (25+ tests)."""

import pytest

from nexus.aab.behavior import (
    BehaviorEngine, BehaviorState, A2AOpcodes,
    A2AMessage, A2AResponse,
)


@pytest.fixture
def engine():
    return BehaviorEngine(agent_id="AUV-001")


class TestA2AOpcodes:
    def test_opcode_count(self):
        assert len(A2AOpcodes) >= 29

    def test_task_opcodes(self):
        assert A2AOpcodes.REQUEST_TASK == 0x01
        assert A2AOpcodes.COMPLETE_TASK == 0x06
        assert A2AOpcodes.CANCEL_TASK == 0x08

    def test_negotiation_opcodes(self):
        assert A2AOpcodes.NEGOTIATE == 0x10
        assert A2AOpcodes.ACCEPT_PROPOSAL == 0x13

    def test_delegation_opcodes(self):
        assert A2AOpcodes.DELEGATE == 0x20
        assert A2AOpcodes.ASSIST_COMPLETE == 0x24

    def test_coordination_opcodes(self):
        assert A2AOpcodes.SYNC_CLOCK == 0x40
        assert A2AOpcodes.EMERGENCY_STOP == 0x45


class TestBehaviorState:
    def test_state_count(self):
        assert len(BehaviorState) == 5

    def test_state_values(self):
        assert BehaviorState.IDLE.value == "idle"
        assert BehaviorState.ACTIVE.value == "active"


class TestStateTransitions:
    def test_idle_to_active(self, engine):
        assert engine.transition(BehaviorState.ACTIVE) is True
        assert engine.state == BehaviorState.ACTIVE

    def test_idle_to_negotiating(self, engine):
        assert engine.transition(BehaviorState.NEGOTIATING) is True

    def test_idle_to_delegating(self, engine):
        assert engine.transition(BehaviorState.DELEGATING) is True

    def test_active_to_reporting(self, engine):
        engine.transition(BehaviorState.ACTIVE)
        assert engine.transition(BehaviorState.REPORTING) is True

    def test_active_to_idle(self, engine):
        engine.transition(BehaviorState.ACTIVE)
        assert engine.transition(BehaviorState.IDLE) is True

    def test_invalid_transition(self, engine):
        # IDLE cannot go directly to REPORTING
        assert engine.transition(BehaviorState.REPORTING) is False

    def test_force_transition(self, engine):
        engine.force_transition(BehaviorState.REPORTING)
        assert engine.state == BehaviorState.REPORTING

    def test_is_valid_transition(self, engine):
        assert engine.is_valid_transition(BehaviorState.ACTIVE) is True
        assert engine.is_valid_transition(BehaviorState.REPORTING) is False

    def test_state_history(self, engine):
        engine.transition(BehaviorState.ACTIVE)
        engine.transition(BehaviorState.REPORTING)
        history = engine.get_state_history()
        assert len(history) == 3  # INIT + ACTIVE + REPORTING


class TestMessageHandling:
    def test_handle_message_no_handler(self, engine):
        msg = A2AMessage(opcode=A2AOpcodes.REQUEST_TASK, sender="AUV-002")
        response = engine.handle_message(msg)
        assert response.accepted is False

    def test_register_and_handle(self, engine):
        def handler(msg):
            return A2AResponse(accepted=True, reply_opcode=A2AOpcodes.ACCEPT_TASK)

        engine.register_handler(A2AOpcodes.REQUEST_TASK, handler)
        msg = A2AMessage(opcode=A2AOpcodes.REQUEST_TASK, sender="AUV-002")
        response = engine.handle_message(msg)
        assert response.accepted is True
        assert response.reply_opcode == A2AOpcodes.ACCEPT_TASK

    def test_handler_exception(self, engine):
        def bad_handler(msg):
            raise ValueError("oops")

        engine.register_handler(A2AOpcodes.PING, bad_handler)
        msg = A2AMessage(opcode=A2AOpcodes.PING, sender="AUV-002")
        response = engine.handle_message(msg)
        assert response.accepted is False

    def test_handler_state_transition(self, engine):
        def handler(msg):
            return A2AResponse(accepted=True, state_transition=BehaviorState.ACTIVE)

        engine.register_handler(A2AOpcodes.ASSIGN_TASK, handler)
        msg = A2AMessage(opcode=A2AOpcodes.ASSIGN_TASK, sender="AUV-002")
        engine.handle_message(msg)
        assert engine.state == BehaviorState.ACTIVE

    def test_unregister_handler(self, engine):
        def h(msg):
            return A2AResponse()

        engine.register_handler(A2AOpcodes.PING, h)
        assert engine.has_handler(A2AOpcodes.PING) is True
        assert engine.unregister_handler(A2AOpcodes.PING) is True
        assert engine.has_handler(A2AOpcodes.PING) is False


class TestMessageCreation:
    def test_send_message(self, engine):
        msg = engine.send_message(A2AOpcodes.PING, "AUV-002")
        assert msg.sender == "AUV-001"
        assert msg.receiver == "AUV-002"
        assert msg.opcode == A2AOpcodes.PING

    def test_send_with_priority(self, engine):
        msg = engine.send_message(A2AOpcodes.EMERGENCY_STOP, "ALL", priority=10)
        assert msg.priority == 10

    def test_message_reply(self):
        original = A2AMessage(
            opcode=A2AOpcodes.REQUEST_TASK,
            sender="AUV-001",
            receiver="AUV-002",
            conversation_id="CONV123",
        )
        reply = original.reply(A2AOpcodes.ACCEPT_TASK, {"ready": True})
        assert reply.sender == "AUV-002"
        assert reply.receiver == "AUV-001"
        assert reply.conversation_id == "CONV123"
        assert reply.in_reply_to == "CONV123"


class TestTaskManagement:
    def test_enqueue_task(self, engine):
        msg = A2AMessage(opcode=A2AOpcodes.REQUEST_TASK)
        engine.enqueue_task(msg)
        assert engine.pending_tasks == 1

    def test_next_task(self, engine):
        msg = A2AMessage(opcode=A2AOpcodes.REQUEST_TASK)
        engine.enqueue_task(msg)
        task = engine.next_task()
        assert task is not None
        assert engine.current_task is not None
        assert engine.pending_tasks == 0

    def test_next_task_empty(self, engine):
        assert engine.next_task() is None

    def test_complete_current_task(self, engine):
        msg = A2AMessage(opcode=A2AOpcodes.REQUEST_TASK)
        engine.enqueue_task(msg)
        engine.next_task()
        assert engine.complete_current_task() is True
        assert engine.current_task is None

    def test_complete_no_task(self, engine):
        assert engine.complete_current_task() is False


class TestConversations:
    def test_conversation_tracking(self, engine):
        msg = A2AMessage(opcode=A2AOpcodes.PING, conversation_id="C1")
        engine.handle_message(msg)
        conv = engine.get_conversation("C1")
        assert len(conv) == 1

    def test_messages_by_opcode(self, engine):
        engine.send_message(A2AOpcodes.PING, "A")
        engine.send_message(A2AOpcodes.PING, "B")
        engine.send_message(A2AOpcodes.PONG, "C")
        pings = engine.get_messages_by_opcode(A2AOpcodes.PING)
        assert len(pings) == 2


class TestReset:
    def test_reset(self, engine):
        engine.transition(BehaviorState.ACTIVE)
        engine.send_message(A2AOpcodes.PING, "A")
        engine.reset()
        assert engine.state == BehaviorState.IDLE
        assert engine.message_count == 0
