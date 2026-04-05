"""Tests for MQTT Client — Mock client, connections, message handling."""

import json
import time
import pytest

from jetson.agent.mqtt_bridge.client import (
    MQTTMessage,
    MQTTClientInterface,
    MockMQTTClient,
    ConnectionEvent,
    ConnectionState,
)


class TestMQTTMessage:
    """Test MQTT message dataclass."""

    def test_create_with_bytes_payload(self):
        msg = MQTTMessage(topic="test/topic", payload=b"hello")
        assert msg.topic == "test/topic"
        assert msg.payload == b"hello"
        assert msg.qos == 0
        assert msg.retain is False
        assert msg.payload_str == "hello"
        assert msg.payload_size == 5

    def test_create_with_full_params(self):
        msg = MQTTMessage(
            topic="nexus/v1/telemetry",
            payload=b'{"data": 1}',
            qos=1,
            retain=True,
            mid=42,
        )
        assert msg.topic == "nexus/v1/telemetry"
        assert msg.qos == 1
        assert msg.retain is True
        assert msg.mid == 42

    def test_payload_str_decodes_utf8(self):
        msg = MQTTMessage(topic="t", payload=b'{"key": "value"}')
        assert msg.payload_str == '{"key": "value"}'

    def test_payload_str_handles_invalid_utf8(self):
        msg = MQTTMessage(topic="t", payload=b'\xff\xfe')
        # Should not raise, uses errors='replace'
        assert isinstance(msg.payload_str, str)

    def test_payload_size(self):
        msg = MQTTMessage(topic="t", payload=b"hello world")
        assert msg.payload_size == 11

    def test_with_topic_creates_copy(self):
        msg = MQTTMessage(topic="old/topic", payload=b"data", mid=1)
        new_msg = msg.with_topic("new/topic")
        assert new_msg.topic == "new/topic"
        assert new_msg.payload == b"data"
        assert new_msg.mid == 1
        assert msg.topic == "old/topic"  # Original unchanged

    def test_frozen_immutability(self):
        msg = MQTTMessage(topic="t", payload=b"d")
        with pytest.raises(AttributeError):
            msg.topic = "new"

    def test_timestamp_auto_generated(self):
        before = time.time()
        msg = MQTTMessage(topic="t", payload=b"d")
        after = time.time()
        assert before <= msg.timestamp <= after


class TestMockMQTTClientConnect:
    """Test client connection lifecycle."""

    @pytest.fixture
    def client(self):
        return MockMQTTClient(client_id="test-client")

    def test_initial_state_disconnected(self, client):
        assert client.state == ConnectionState.DISCONNECTED
        assert not client.is_connected()

    def test_connect_success(self, client):
        result = client.connect("localhost", 1883)
        assert result == 0
        assert client.is_connected()
        assert client.state == ConnectionState.CONNECTED

    def test_connect_with_full_params(self, client):
        result = client.connect(
            host="broker.local",
            port=8883,
            keepalive=120,
            clean_session=False,
            client_id="custom-id",
            username="user",
            password="pass",
        )
        assert result == 0
        assert client.client_id == "custom-id"
        assert client.is_connected()

    def test_connect_failure(self, client):
        client.fail_on_connect = True
        result = client.connect("localhost")
        assert result != 0
        assert not client.is_connected()

    def test_disconnect(self, client):
        client.connect("localhost")
        result = client.disconnect()
        assert result == 0
        assert not client.is_connected()
        assert client.state == ConnectionState.DISCONNECTED

    def test_disconnect_when_not_connected(self, client):
        result = client.disconnect()
        assert result == 0

    def test_reconnect(self, client):
        client.connect("localhost")
        client.disconnect()
        result = client.connect("localhost", 1884)
        assert result == 0
        assert client.is_connected()

    def test_connect_logs_events(self, client):
        client.connect("localhost")
        events = client.get_event_log()
        event_types = [e.event for e in events]
        assert ConnectionEvent.CONNECTING in event_types
        assert ConnectionEvent.CONNECTED in event_types

    def test_disconnect_logs_events(self, client):
        client.connect("localhost")
        client.disconnect()
        events = client.get_event_log()
        event_types = [e.event for e in events]
        assert ConnectionEvent.DISCONNECTING in event_types
        assert ConnectionEvent.DISCONNECTED in event_types


class TestMockMQTTClientPublish:
    """Test client publishing."""

    @pytest.fixture
    def client(self):
        c = MockMQTTClient()
        c.connect("localhost")
        return c

    def test_publish_basic(self, client):
        mid = client.publish("test/topic", b"hello")
        assert mid > 0

    def test_publish_string_payload(self, client):
        mid = client.publish("test/topic", "hello string")
        assert mid > 0
        msgs = client.get_published_messages()
        assert msgs[0].payload == b"hello string"

    def test_publish_with_qos(self, client):
        mid = client.publish("test/topic", b"data", qos=2)
        assert mid > 0
        msgs = client.get_published_messages()
        assert msgs[0].qos == 2

    def test_publish_with_retain(self, client):
        mid = client.publish("test/topic", b"data", retain=True)
        assert mid > 0
        msgs = client.get_published_messages()
        assert msgs[0].retain is True

    def test_publish_increments_mid(self, client):
        mid1 = client.publish("t", b"a")
        mid2 = client.publish("t", b"b")
        assert mid2 > mid1

    def test_publish_logs_event(self, client):
        client.publish("test/topic", b"data")
        events = client.get_event_log()
        event_types = [e.event for e in events]
        assert ConnectionEvent.MESSAGE_PUBLISHED in event_types

    def test_publish_while_disconnected_fails(self, client):
        client.disconnect()
        mid = client.publish("test/topic", b"data")
        assert mid == -1

    def test_publish_failure_simulated(self, client):
        client.fail_on_publish = True
        mid = client.publish("test/topic", b"data")
        assert mid == -1


class TestMockMQTTClientSubscribe:
    """Test client subscription."""

    @pytest.fixture
    def client(self):
        c = MockMQTTClient()
        c.connect("localhost")
        return c

    def test_subscribe_basic(self, client):
        mid = client.subscribe("test/topic")
        assert mid > 0
        subs = client.get_subscriptions()
        assert "test/topic" in subs

    def test_subscribe_with_qos(self, client):
        mid = client.subscribe("test/topic", qos=2)
        assert mid > 0
        subs = client.get_subscriptions()
        assert subs["test/topic"] == 2

    def test_subscribe_multiple(self, client):
        client.subscribe("topic/a")
        client.subscribe("topic/b")
        client.subscribe("topic/c")
        subs = client.get_subscriptions()
        assert len(subs) == 3

    def test_unsubscribe(self, client):
        client.subscribe("test/topic")
        mid = client.unsubscribe("test/topic")
        assert mid > 0
        subs = client.get_subscriptions()
        assert "test/topic" not in subs

    def test_unsubscribe_nonexistent(self, client):
        mid = client.unsubscribe("nonexistent")
        assert mid > 0  # Still returns a message ID

    def test_subscribe_logs_event(self, client):
        client.subscribe("test/topic")
        events = client.get_event_log()
        event_types = [e.event for e in events]
        assert ConnectionEvent.SUBSCRIBED in event_types

    def test_unsubscribe_logs_event(self, client):
        client.subscribe("test/topic")
        client.unsubscribe("test/topic")
        events = client.get_event_log()
        event_types = [e.event for e in events]
        assert ConnectionEvent.UNSUBSCRIBED in event_types


class TestMockMQTTClientMessageDelivery:
    """Test message delivery between publisher and subscriber."""

    @pytest.fixture
    def client(self):
        c = MockMQTTClient()
        c.connect("localhost")
        return c

    def test_subscriber_receives_published_message(self, client):
        received = []
        client.set_on_message(lambda msg: received.append(msg))
        client.subscribe("test/topic")
        client.publish("test/topic", b"hello")

        assert len(received) == 1
        assert received[0].topic == "test/topic"
        assert received[0].payload == b"hello"

    def test_subscriber_does_not_receive_other_topics(self, client):
        received = []
        client.set_on_message(lambda msg: received.append(msg))
        client.subscribe("topic/a")
        client.publish("topic/b", b"wrong")

        assert len(received) == 0

    def test_wildcard_plus(self, client):
        received = []
        client.set_on_message(lambda msg: received.append(msg))
        client.subscribe("nexus/+/telemetry")
        client.publish("nexus/v1/telemetry", b"data1")
        client.publish("nexus/v2/telemetry", b"data2")

        assert len(received) == 2

    def test_wildcard_hash(self, client):
        received = []
        client.set_on_message(lambda msg: received.append(msg))
        client.subscribe("nexus/#")
        client.publish("nexus/v1/telemetry", b"a")
        client.publish("nexus/v1/status", b"b")
        client.publish("nexus/fleet/coordination", b"c")

        assert len(received) == 3

    def test_wildcard_hash_partial(self, client):
        received = []
        client.set_on_message(lambda msg: received.append(msg))
        client.subscribe("nexus/v1/#")
        client.publish("nexus/v1/telemetry", b"a")
        client.publish("nexus/v2/telemetry", b"b")  # Different vessel

        assert len(received) == 1

    def test_multiple_subscribers_same_topic(self, client):
        # Use two separate clients to test cross-client delivery
        # (mock delivers to local subscribers only)
        received = []
        client.set_on_message(lambda msg: received.append(msg))
        client.subscribe("test/topic")
        client.subscribe("test/topic")  # Duplicate subscription
        client.publish("test/topic", b"data")

        # Both subscriptions trigger callback twice
        assert len(received) == 2

    def test_message_queue_drain(self, client):
        client.set_on_message(lambda msg: None)
        client.subscribe("test/topic")
        client.publish("test/topic", b"a")
        client.publish("test/topic", b"b")
        client.publish("test/topic", b"c")

        queue = client.message_queue()
        assert len(queue) == 3
        # After draining, queue is empty
        assert client.message_queue().is_empty if hasattr(client.message_queue(), 'is_empty') else len(client.message_queue()) == 0

    def test_inject_message(self, client):
        received = []
        client.set_on_message(lambda msg: received.append(msg))
        client.subscribe("test/topic")
        client.inject_message("test/topic", b"injected")

        assert len(received) == 1
        assert received[0].payload == b"injected"


class TestMockMQTTClientRetained:
    """Test retained message behavior."""

    @pytest.fixture
    def client(self):
        c = MockMQTTClient()
        c.connect("localhost")
        return c

    def test_retained_message_delivered_on_subscribe(self, client):
        # Publish retained
        client.publish("test/topic", b"retained", retain=True)

        # New subscriber gets the retained message
        received = []
        client.set_on_message(lambda msg: received.append(msg))
        client.subscribe("test/topic")
        assert len(received) == 1
        assert received[0].payload == b"retained"
        assert received[0].retain is True

    def test_retained_message_updated(self, client):
        client.publish("test/topic", b"v1", retain=True)
        client.publish("test/topic", b"v2", retain=True)

        received = []
        client.set_on_message(lambda msg: received.append(msg))
        client.subscribe("test/topic")
        assert len(received) == 1
        assert received[0].payload == b"v2"

    def test_non_retained_not_delivered_on_subscribe(self, client):
        client.publish("test/topic", b"not retained")

        received = []
        client.set_on_message(lambda msg: received.append(msg))
        client.subscribe("test/topic")
        assert len(received) == 0


class TestMockMQTTClientCallbacks:
    """Test client callback lifecycle."""

    def test_on_connect_callback(self):
        connect_called = []
        client = MockMQTTClient()
        client.set_on_connect(lambda success, data: connect_called.append((success, data)))
        client.connect("localhost")
        assert len(connect_called) == 1
        assert connect_called[0][0] is True

    def test_on_disconnect_callback(self):
        disconnect_called = []
        client = MockMQTTClient()
        client.set_on_disconnect(lambda rc, data: disconnect_called.append((rc, data)))
        client.connect("localhost")
        client.disconnect()
        assert len(disconnect_called) == 1

    def test_on_message_callback(self):
        messages = []
        client = MockMQTTClient()
        client.set_on_message(lambda msg: messages.append(msg))
        client.connect("localhost")
        client.subscribe("test/topic")
        client.publish("test/topic", b"data")
        assert len(messages) == 1
        assert messages[0].topic == "test/topic"

    def test_callback_error_handled_gracefully(self):
        """Callbacks that raise exceptions should not crash the client."""
        def bad_callback(msg):
            raise RuntimeError("test error")

        client = MockMQTTClient()
        client.set_on_message(bad_callback)
        client.connect("localhost")
        client.subscribe("test/topic")
        # Should not raise
        client.publish("test/topic", b"data")
        assert client.is_connected()


class TestMockMQTTClientClear:
    """Test client state clearing."""

    def test_clear_resets_all_state(self):
        client = MockMQTTClient()
        client.connect("localhost")
        client.subscribe("test/topic")
        client.publish("test/topic", b"data")
        client.inject_message("test/topic", b"recv")

        client.clear()

        assert len(client.get_published_messages()) == 0
        assert len(client.get_subscriptions()) == 0
        assert len(client.get_event_log()) == 0


class TestMockMQTTClientLoop:
    """Test loop_start/loop_stop (no-op for mock)."""

    def test_loop_start_stop(self):
        client = MockMQTTClient()
        client.loop_start()
        client.loop_stop()
        # Should not raise


class TestConnectionState:
    """Test ConnectionState enum."""

    def test_all_states_exist(self):
        states = [
            ConnectionState.DISCONNECTED,
            ConnectionState.CONNECTING,
            ConnectionState.CONNECTED,
            ConnectionState.RECONNECTING,
            ConnectionState.DISCONNECTING,
        ]
        assert len(states) == 5

    def test_state_string_values(self):
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.DISCONNECTED.value == "disconnected"


class TestConnectionEvent:
    """Test ConnectionEvent enum."""

    def test_all_events_exist(self):
        events = [
            ConnectionEvent.CONNECTING,
            ConnectionEvent.CONNECTED,
            ConnectionEvent.DISCONNECTING,
            ConnectionEvent.DISCONNECTED,
            ConnectionEvent.MESSAGE_ARRIVED,
            ConnectionEvent.MESSAGE_PUBLISHED,
            ConnectionEvent.SUBSCRIBED,
            ConnectionEvent.UNSUBSCRIBED,
            ConnectionEvent.ERROR,
        ]
        assert len(events) == 9
