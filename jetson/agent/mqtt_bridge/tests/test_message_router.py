"""Tests for Message Router — routing, dead letter queue, priority filtering."""

import json
import time
import pytest

from jetson.agent.mqtt_bridge.client import MQTTMessage, MockMQTTClient
from jetson.agent.mqtt_bridge.topics import TopicPattern, TopicHierarchy
from jetson.agent.mqtt_bridge.message_router import (
    MessageRouter,
    HandlerRegistration,
    DeadLetterQueue,
    DeadLetterEntry,
    MessagePriority,
    RouteResult,
)


# ===================================================================
# Helpers
# ===================================================================

def make_message(topic: str, payload: dict | None = None, qos: int = 1) -> MQTTMessage:
    """Create a test MQTTMessage."""
    data = json.dumps(payload or {"test": True}).encode("utf-8")
    return MQTTMessage(topic=topic, payload=data, qos=qos, mid=1)


# ===================================================================
# Dead Letter Queue Tests
# ===================================================================

class TestDeadLetterQueue:
    @pytest.fixture
    def dlq(self):
        return DeadLetterQueue()

    def test_enqueue_and_size(self, dlq):
        msg = make_message("test/topic")
        dlq.enqueue(msg, "no handler")
        assert dlq.size == 1

    def test_dequeue(self, dlq):
        msg = make_message("test/topic", {"key": "val"})
        dlq.enqueue(msg, "reason1")
        entry = dlq.dequeue()
        assert entry is not None
        assert entry.reason == "reason1"
        assert entry.message.topic == "test/topic"
        assert dlq.size == 0

    def test_dequeue_empty(self, dlq):
        assert dlq.dequeue() is None

    def test_peek(self, dlq):
        msg = make_message("t")
        dlq.enqueue(msg, "r")
        entry = dlq.peek()
        assert entry is not None
        assert dlq.size == 1  # Not removed

    def test_fifo_order(self, dlq):
        msgs = [make_message(f"t/{i}") for i in range(5)]
        for m in msgs:
            dlq.enqueue(m, f"reason-{i}")

        for i in range(5):
            entry = dlq.dequeue()
            assert entry.message.topic == f"t/{i}"

    def test_retry_all(self, dlq):
        for i in range(3):
            dlq.enqueue(make_message(f"t/{i}"), "r")
        entries = dlq.retry_all()
        assert len(entries) == 3
        assert dlq.size == 0

    def test_purge(self, dlq):
        for _ in range(5):
            dlq.enqueue(make_message("t"), "r")
        count = dlq.purge()
        assert count == 5
        assert dlq.size == 0

    def test_purge_expired(self, dlq):
        dlq.enqueue(make_message("t"), "old")
        # Manually age the entry
        dlq._queue[0].timestamp = time.time() - 7200  # 2 hours ago

        dlq.enqueue(make_message("t"), "new")
        expired = dlq.purge_expired(max_age_seconds=3600)
        assert expired == 1
        assert dlq.size == 1

    def test_max_size(self):
        dlq = DeadLetterQueue(max_size=3)
        for i in range(5):
            dlq.enqueue(make_message(f"t/{i}"), "r")
        assert dlq.size == 3

    def test_is_empty(self, dlq):
        assert dlq.is_empty
        dlq.enqueue(make_message("t"), "r")
        assert not dlq.is_empty

    def test_stats(self, dlq):
        dlq.enqueue(make_message("t"), "r")
        dlq.enqueue(make_message("t"), "r")
        stats = dlq.stats
        assert stats["total_enqueued"] == 2
        assert stats["current_size"] == 2

    def test_entry_to_dict(self, dlq):
        dlq.enqueue(make_message("test/topic", {"data": 42}), "no handler")
        entry = dlq.dequeue()
        d = entry.to_dict()
        assert d["reason"] == "no handler"
        assert d["topic"] == "test/topic"
        assert "payload_preview" in d

    def test_get_entries(self, dlq):
        dlq.enqueue(make_message("t1"), "r1")
        dlq.enqueue(make_message("t2"), "r2")
        entries = dlq.get_entries()
        assert len(entries) == 2
        assert dlq.size == 2  # Not removed


# ===================================================================
# Message Router Tests
# ===================================================================

class TestMessageRouterRegistration:
    @pytest.fixture
    def router(self):
        return MessageRouter(vessel_id="vessel-001")

    def test_register_handler(self, router):
        handler = lambda msg, pr: None
        reg = router.register("nexus/vessel-001/command", handler, "cmd")
        assert router.handler_count == 1
        assert reg.name == "cmd"

    def test_register_multiple_handlers(self, router):
        router.register("t1", lambda m, p: None, "h1")
        router.register("t2", lambda m, p: None, "h2")
        router.register("t3", lambda m, p: None, "h3")
        assert router.handler_count == 3

    def test_unregister(self, router):
        router.register("t", lambda m, p: None, "h1")
        assert router.unregister("h1") is True
        assert router.handler_count == 0

    def test_unregister_nonexistent(self, router):
        assert router.unregister("nonexistent") is False

    def test_get_handler(self, router):
        router.register("t", lambda m, p: None, "h1")
        reg = router.get_handler("h1")
        assert reg is not None
        assert reg.name == "h1"

    def test_get_handler_nonexistent(self, router):
        assert router.get_handler("nonexistent") is None

    def test_enable_disable_handler(self, router):
        router.register("t", lambda m, p: None, "h1")
        assert router.disable_handler("h1") is True
        reg = router.get_handler("h1")
        assert reg is not None
        assert not reg.enabled
        assert router.enable_handler("h1") is True
        assert reg.enabled


class TestMessageRouterRouting:
    @pytest.fixture
    def router(self):
        return MessageRouter(vessel_id="vessel-001")

    def test_route_to_matching_handler(self, router):
        received = []
        router.register(
            "nexus/vessel-001/command",
            lambda msg, pr: received.append(msg),
            "cmd_handler",
        )
        msg = make_message("nexus/vessel-001/command", {"action": "go"})
        result = router.route(msg)
        assert result.was_routed
        assert result.handler_name == "cmd_handler"
        assert len(received) == 1

    def test_route_unknown_prefix_to_dlq(self, router):
        msg = make_message("invalid/prefix/topic")
        result = router.route(msg)
        assert not result.was_routed
        assert result.reason == "unknown_topic_prefix"
        assert router.dlq.size == 1

    def test_route_no_matching_handler_to_dlq(self, router):
        msg = make_message("nexus/vessel-001/telemetry")
        result = router.route(msg)
        assert not result.was_routed
        assert result.reason == "no_matching_handler"
        assert router.dlq.size == 1

    def test_wildcard_plus_routing(self, router):
        received = []
        router.register(
            "nexus/+/command",
            lambda msg, pr: received.append(msg),
            "cmd",
        )
        router.route(make_message("nexus/vessel-001/command"))
        router.route(make_message("nexus/vessel-002/command"))
        assert len(received) == 2

    def test_wildcard_hash_routing(self, router):
        received = []
        router.register(
            "nexus/vessel-001/#",
            lambda msg, pr: received.append(msg),
            "all_v1",
        )
        router.route(make_message("nexus/vessel-001/telemetry"))
        router.route(make_message("nexus/vessel-001/command"))
        router.route(make_message("nexus/vessel-001/safety/alert"))
        assert len(received) == 3

    def test_hash_not_matching_different_vessel(self, router):
        received = []
        router.register(
            "nexus/vessel-001/#",
            lambda msg, pr: received.append(msg),
            "v1",
        )
        router.route(make_message("nexus/vessel-002/telemetry"))
        assert len(received) == 0

    def test_multiple_handlers_same_topic(self, router):
        r1 = []
        r2 = []
        router.register("t", lambda m, p: r1.append(1), "h1")
        router.register("t", lambda m, p: r2.append(2), "h2")
        router.route(make_message("t"))
        assert len(r1) == 1
        assert len(r2) == 1

    def test_disabled_handler_not_called(self, router):
        received = []
        router.register("t", lambda m, p: received.append(1), "h1")
        router.disable_handler("h1")
        router.route(make_message("t"))
        assert len(received) == 0

    def test_handler_error_does_not_crash(self, router):
        def bad_handler(msg, pr):
            raise RuntimeError("test error")

        router.register("t", bad_handler, "bad")
        result = router.route(make_message("t"))
        assert not result.success
        assert "handler_error" in result.reason

    def test_handler_stats_updated(self, router):
        router.register("t", lambda m, p: None, "h1")
        router.route(make_message("t"))
        router.route(make_message("t"))
        reg = router.get_handler("h1")
        assert reg.message_count == 2

    def test_handler_error_stats_updated(self, router):
        router.register("t", lambda m, p: (_ for _ in ()).throw(RuntimeError()), "h1")
        router.route(make_message("t"))
        reg = router.get_handler("h1")
        assert reg.error_count == 1

    def test_route_many(self, router):
        received = []
        router.register("t", lambda m, p: received.append(1), "h1")
        msgs = [make_message("t") for _ in range(5)]
        results = router.route_many(msgs)
        assert len(results) == 5
        assert all(r.was_routed for r in results)
        assert len(received) == 5


class TestMessageRouterPriority:
    def test_priority_ordering(self):
        router = MessageRouter()
        call_order = []

        router.register("t", lambda m, p: call_order.append("low"), "low", MessagePriority.LOW)
        router.register("t", lambda m, p: call_order.append("high"), "high", MessagePriority.HIGH)
        router.register("t", lambda m, p: call_order.append("normal"), "normal", MessagePriority.NORMAL)

        router.route(make_message("t"))
        # Handlers sorted by priority (highest first): HIGH, NORMAL, LOW
        assert call_order[0] == "high"
        assert call_order[1] == "normal"
        assert call_order[2] == "low"

    def test_critical_priority_from_qos(self):
        assert MessagePriority.from_qos(0) == MessagePriority.LOW
        assert MessagePriority.from_qos(1) == MessagePriority.NORMAL
        assert MessagePriority.from_qos(2) == MessagePriority.CRITICAL


class TestMessageRouterTrustFiltering:
    def test_trust_level_filter_pass(self):
        router = MessageRouter()
        received = []
        router.register(
            "t",
            lambda m, p: received.append(1),
            "h1",
            min_trust_level=3,
        )
        msg = make_message("t", {"trust_level": 5})
        router.route(msg)
        assert len(received) == 1

    def test_trust_level_filter_reject(self):
        router = MessageRouter()
        received = []
        router.register(
            "t",
            lambda m, p: received.append(1),
            "h1",
            min_trust_level=5,
        )
        msg = make_message("t", {"trust_level": 2})
        router.route(msg)
        assert len(received) == 0
        assert router.stats["total_filtered_trust"] == 1

    def test_trust_filter_no_trust_in_payload(self):
        router = MessageRouter()
        received = []
        router.register(
            "t",
            lambda m, p: received.append(1),
            "h1",
            min_trust_level=5,
        )
        # No trust_level in payload, defaults to 0 -> filtered
        msg = make_message("t", {"other": "data"})
        router.route(msg)
        assert len(received) == 0


class TestMessageRouterVesselFiltering:
    def test_vessel_filter_matches(self):
        router = MessageRouter(vessel_id="v-1")
        received = []
        router.register(
            "nexus/+/command",
            lambda m, p: received.append(1),
            "h1",
            vessel_id="v-1",
        )
        router.route(make_message("nexus/v-1/command"))
        assert len(received) == 1

    def test_vessel_filter_rejects_other_vessel(self):
        router = MessageRouter(vessel_id="v-1")
        received = []
        router.register(
            "nexus/+/command",
            lambda m, p: received.append(1),
            "h1",
            vessel_id="v-1",
        )
        router.route(make_message("nexus/v-2/command"))
        assert len(received) == 0
        assert router.stats["total_filtered_vessel"] == 1


class TestMessageRouterStats:
    def test_routing_stats(self):
        router = MessageRouter()
        router.register("t", lambda m, p: None, "h1")
        router.route(make_message("t"))
        router.route(make_message("unknown/x"))  # goes to DLQ

        stats = router.stats
        assert stats["total_routed"] == 1
        assert stats["total_dlq"] == 1

    def test_reset_stats(self):
        router = MessageRouter()
        router.register("t", lambda m, p: None, "h1")
        router.route(make_message("t"))
        router.reset_stats()
        assert router.stats["total_routed"] == 0


class TestMessageRouterDLQDisabled:
    def test_dlq_disabled_drops_messages(self):
        router = MessageRouter(enable_dlq=False)
        router.route(make_message("unknown/x"))
        assert router.dlq.size == 0
        assert router.stats["total_dropped"] == 1


class TestRouteResult:
    def test_was_routed_true(self):
        result = RouteResult(
            message_id="abc",
            topic="t",
            handler_name="h1",
            success=True,
        )
        assert result.was_routed

    def test_was_routed_no_handler(self):
        result = RouteResult(
            message_id="abc",
            topic="t",
        )
        assert not result.was_routed

    def test_was_routed_failed(self):
        result = RouteResult(
            message_id="abc",
            topic="t",
            handler_name="h1",
            success=False,
        )
        assert not result.was_routed
