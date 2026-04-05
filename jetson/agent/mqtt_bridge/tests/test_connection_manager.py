"""Tests for Connection Manager — multi-broker, health, bandwidth, offline queue."""

import json
import os
import tempfile
import time
import pytest

from jetson.agent.mqtt_bridge.client import (
    MockMQTTClient,
    MQTTMessage,
    ConnectionState,
)
from jetson.agent.mqtt_bridge.connection_manager import (
    BrokerConfig,
    ConnectionHealth,
    ConnectionManager,
    BandwidthMonitor,
    OfflineQueue,
    QueuedMessage,
)


class TestBrokerConfig:
    def test_default_config(self):
        config = BrokerConfig()
        assert config.host == "localhost"
        assert config.port == 1883
        assert config.keepalive == 60
        assert config.clean_session is True
        assert config.tls_enabled is False

    def test_address(self):
        config = BrokerConfig(host="broker.local", port=8883)
        assert config.address == "broker.local:8883"

    def test_id(self):
        config = BrokerConfig(host="broker.local", port=8883)
        assert config.id == "broker.local:8883"

    def test_priority_sorting(self):
        primary = BrokerConfig(host="primary", priority=0)
        backup1 = BrokerConfig(host="backup1", priority=1)
        backup2 = BrokerConfig(host="backup2", priority=2)
        brokers = sorted([backup2, primary, backup1], key=lambda b: b.priority)
        assert brokers[0].host == "primary"

    def test_tls_fields(self):
        config = BrokerConfig(
            tls_enabled=True,
            tls_ca_cert="/path/to/ca.crt",
            tls_cert="/path/to/client.crt",
            tls_key="/path/to/client.key",
        )
        assert config.tls_enabled
        assert config.tls_ca_cert == "/path/to/ca.crt"
        assert config.tls_insecure is False


class TestConnectionHealth:
    def test_initial_state(self):
        health = ConnectionHealth()
        assert not health.connected
        assert health.messages_sent == 0
        assert health.reconnect_count == 0
        assert health.is_healthy is False

    def test_healthy_when_connected(self):
        health = ConnectionHealth()
        health.connected = True
        health.connect_time = time.time() - 10
        health.last_message_time = time.time() - 5
        assert health.is_healthy

    def test_unhealthy_when_disconnected(self):
        health = ConnectionHealth()
        health.connected = False
        assert not health.is_healthy

    def test_unhealthy_when_stale(self):
        health = ConnectionHealth()
        health.connected = True
        health.connect_time = time.time() - 1000
        health.last_message_time = time.time() - 500
        assert not health.is_healthy

    def test_record_latency(self):
        health = ConnectionHealth()
        health.record_latency(10.0)
        health.record_latency(20.0)
        health.record_latency(30.0)
        assert health.ping_latency_ms == 30.0
        assert health.avg_latency_ms == 20.0

    def test_latency_samples_max(self):
        health = ConnectionHealth()
        for i in range(200):
            health.record_latency(float(i))
        assert len(health._latency_samples) == 100

    def test_to_dict(self):
        health = ConnectionHealth()
        health.connected = True
        health.messages_sent = 42
        d = health.to_dict()
        assert d["connected"] is True
        assert d["messages_sent"] == 42


class TestBandwidthMonitor:
    def test_initial_state(self):
        bw = BandwidthMonitor()
        assert bw.current_send_bps == 0.0
        assert not bw.is_throttled
        assert bw.peak_bps == 0.0

    def test_record_send(self):
        bw = BandwidthMonitor()
        bw.record_send(1024)
        bw.record_send(1024)
        assert bw.total_bytes_sent == 2048

    def test_record_receive(self):
        bw = BandwidthMonitor()
        bw.record_receive(512)
        assert bw.total_bytes_received == 512

    def test_record_messages(self):
        bw = BandwidthMonitor()
        bw.record_message_sent()
        bw.record_message_sent()
        bw.record_message_received()
        assert bw._current_window.messages_sent == 2
        assert bw._current_window.messages_received == 1

    def test_throttling_disabled(self):
        bw = BandwidthMonitor(max_bandwidth_bps=0)
        assert not bw.is_throttled
        assert bw.throttle_ratio == 0.0

    def test_not_throttled_under_limit(self):
        bw = BandwidthMonitor(max_bandwidth_bps=10000)
        bw.record_send(100)
        assert not bw.is_throttled

    def test_to_dict(self):
        bw = BandwidthMonitor(max_bandwidth_bps=1000)
        d = bw.to_dict()
        assert "current_send_bps" in d
        assert "is_throttled" in d


class TestOfflineQueue:
    def test_enqueue_and_size(self):
        q = OfflineQueue()
        assert q.enqueue("t", b"data")
        assert q.size == 1

    def test_dequeue(self):
        q = OfflineQueue()
        q.enqueue("t", b"data", qos=2)
        entry = q.dequeue()
        assert entry is not None
        assert entry.topic == "t"
        assert entry.payload == b"data"
        assert entry.qos == 2
        assert q.size == 0

    def test_dequeue_empty(self):
        q = OfflineQueue()
        assert q.dequeue() is None

    def test_peek(self):
        q = OfflineQueue()
        q.enqueue("t", b"data")
        entry = q.peek()
        assert entry is not None
        assert q.size == 1

    def test_fifo_order(self):
        q = OfflineQueue()
        for i in range(5):
            q.enqueue(f"t/{i}", f"data-{i}".encode())
        for i in range(5):
            entry = q.dequeue()
            assert entry.topic == f"t/{i}"

    def test_max_size(self):
        q = OfflineQueue(max_size=3)
        for i in range(5):
            q.enqueue(f"t/{i}", b"d")
        assert q.size == 3

    def test_enqueue_full_drops(self):
        q = OfflineQueue(max_size=1)
        assert q.enqueue("t", b"a")
        assert not q.enqueue("t", b"b")

    def test_replay_through_client(self):
        q = OfflineQueue()
        q.enqueue("t1", b"d1", qos=1)
        q.enqueue("t2", b"d2", qos=2)
        client = MockMQTTClient()
        client.connect("localhost")
        replayed = q.replay(client)
        assert replayed == 2
        assert q.size == 0
        published = client.get_published_messages()
        assert len(published) == 2

    def test_replay_empty(self):
        q = OfflineQueue()
        client = MockMQTTClient()
        client.connect("localhost")
        assert q.replay(client) == 0

    def test_clear(self):
        q = OfflineQueue()
        for _ in range(5):
            q.enqueue("t", b"d")
        count = q.clear()
        assert count == 5
        assert q.size == 0

    def test_is_empty(self):
        q = OfflineQueue()
        assert q.is_empty
        q.enqueue("t", b"d")
        assert not q.is_empty

    def test_stats(self):
        q = OfflineQueue()
        q.enqueue("t", b"d")
        q.enqueue("t", b"d")
        stats = q.stats
        assert stats["enqueued"] == 2
        assert stats["current_size"] == 2

    def test_persist_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            q = OfflineQueue(persist_path=tmpdir)
            q.enqueue("t1", b"data1")
            q.enqueue("t2", b"data2")
            assert q.persist()
            q2 = OfflineQueue(persist_path=tmpdir)
            loaded = q2.load()
            assert loaded == 2
            assert q2.size == 2

    def test_persist_no_path(self):
        q = OfflineQueue(persist_path=None)
        assert not q.persist()

    def test_load_no_path(self):
        q = OfflineQueue(persist_path=None)
        assert q.load() == 0


class TestConnectionManagerConnect:
    def test_connect_single_broker(self):
        mgr = ConnectionManager(
            vessel_id="v-1",
            brokers=[BrokerConfig(host="localhost")],
        )
        assert mgr.connect() is True
        assert mgr.is_connected

    def test_connect_multiple_brokers(self):
        mgr = ConnectionManager(
            vessel_id="v-1",
            brokers=[
                BrokerConfig(host="primary", priority=0),
                BrokerConfig(host="backup", priority=1),
            ],
        )
        assert mgr.connect() is True

    def test_connect_failure(self):
        client = MockMQTTClient(client_id="nexus-v-1")
        client.fail_on_connect = True
        mgr = ConnectionManager(vessel_id="v-1", client=client)
        assert not mgr.connect()
        assert not mgr.is_connected

    def test_initial_disconnected(self):
        mgr = ConnectionManager(vessel_id="v-1")
        assert not mgr.is_connected

    def test_current_broker(self):
        broker = BrokerConfig(host="my-broker", port=1883)
        mgr = ConnectionManager(vessel_id="v-1", brokers=[broker])
        mgr.connect()
        assert mgr.current_broker is not None
        assert mgr.current_broker.host == "my-broker"

    def test_event_logging(self):
        mgr = ConnectionManager(vessel_id="v-1")
        mgr.connect()
        events = mgr.event_log
        event_types = [e["event"] for e in events]
        assert "connect_attempt" in event_types


class TestConnectionManagerDisconnect:
    def test_disconnect(self):
        mgr = ConnectionManager(vessel_id="v-1")
        mgr.connect()
        mgr.disconnect()
        assert not mgr.is_connected

    def test_disconnect_when_not_connected(self):
        mgr = ConnectionManager(vessel_id="v-1")
        mgr.disconnect()
        assert not mgr.is_connected


class TestConnectionManagerShutdown:
    def test_shutdown(self):
        mgr = ConnectionManager(vessel_id="v-1")
        mgr.connect()
        mgr.shutdown()
        assert not mgr.is_connected
        assert not mgr._running

    def test_shutdown_with_offline_queue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ConnectionManager(
                vessel_id="v-1",
                offline_persist_path=tmpdir,
            )
            mgr.connect()
            mgr.offline_queue.enqueue("t", b"data")
            mgr.shutdown()
            q2 = OfflineQueue(persist_path=tmpdir)
            loaded = q2.load()
            assert loaded == 1


class TestConnectionManagerPublish:
    def test_publish_while_connected(self):
        mgr = ConnectionManager(vessel_id="v-1")
        mgr.connect()
        mid = mgr.publish("test/topic", b"data")
        assert mid > 0
        assert mgr.health.messages_sent == 1

    def test_publish_string_payload(self):
        mgr = ConnectionManager(vessel_id="v-1")
        mgr.connect()
        mid = mgr.publish("test/topic", "string data")
        assert mid > 0

    def test_publish_queues_while_disconnected(self):
        mgr = ConnectionManager(vessel_id="v-1")
        mid = mgr.publish("test/topic", b"data")
        assert mid == -1
        assert mgr.offline_queue.size == 1

    def test_publish_updates_bandwidth(self):
        mgr = ConnectionManager(vessel_id="v-1")
        mgr.connect()
        mgr.publish("t", b"x" * 1024)
        assert mgr.bandwidth.total_bytes_sent == 1024


class TestConnectionManagerSubscribe:
    def test_subscribe(self):
        mgr = ConnectionManager(vessel_id="v-1")
        mgr.connect()
        mid = mgr.subscribe("test/topic", 1)
        assert mid > 0


class TestConnectionManagerHealth:
    def test_check_health_connected(self):
        mgr = ConnectionManager(vessel_id="v-1")
        mgr.connect()
        health = mgr.check_health()
        assert health.connected
        assert health.broker_id != ""

    def test_check_health_disconnected(self):
        mgr = ConnectionManager(vessel_id="v-1")
        health = mgr.check_health()
        assert not health.connected

    def test_measure_latency_connected(self):
        mgr = ConnectionManager(vessel_id="v-1")
        mgr.connect()
        latency = mgr.measure_latency()
        assert latency is not None
        assert latency > 0

    def test_measure_latency_disconnected(self):
        mgr = ConnectionManager(vessel_id="v-1")
        latency = mgr.measure_latency()
        assert latency is None


class TestConnectionManagerMultiBroker:
    def test_get_broker_list(self):
        brokers = [
            BrokerConfig(host="b1", priority=0),
            BrokerConfig(host="b2", priority=1),
        ]
        mgr = ConnectionManager(vessel_id="v-1", brokers=brokers)
        broker_list = mgr.get_broker_list()
        assert len(broker_list) == 2
        assert broker_list[0].host == "b1"

    def test_switch_broker(self):
        brokers = [
            BrokerConfig(host="b1", priority=0),
            BrokerConfig(host="b2", priority=1),
        ]
        mgr = ConnectionManager(vessel_id="v-1", brokers=brokers)
        mgr.connect()
        assert mgr.current_broker.host == "b1"
        result = mgr.switch_broker(1)
        assert result is True
        assert mgr.current_broker.host == "b2"

    def test_switch_broker_invalid_index(self):
        mgr = ConnectionManager(vessel_id="v-1")
        assert mgr.switch_broker(99) is False


class TestConnectionManagerReconnect:
    def test_reconnect_success(self):
        mgr = ConnectionManager(
            vessel_id="v-1",
            reconnect_base_delay=0.01,
        )
        mgr.connect()
        mgr.disconnect()
        result = mgr.reconnect()
        assert result is True
        assert mgr.is_connected

    def test_reconnect_increments_count(self):
        mgr = ConnectionManager(
            vessel_id="v-1",
            reconnect_base_delay=0.01,
        )
        mgr.connect()
        mgr.disconnect()
        mgr.reconnect()
        assert mgr.health.reconnect_count >= 1


class TestConnectionManagerProperties:
    def test_client_property(self):
        mgr = ConnectionManager(vessel_id="v-1")
        assert isinstance(mgr.client, MockMQTTClient)

    def test_health_property(self):
        mgr = ConnectionManager(vessel_id="v-1")
        assert isinstance(mgr.health, ConnectionHealth)

    def test_bandwidth_property(self):
        mgr = ConnectionManager(vessel_id="v-1")
        assert isinstance(mgr.bandwidth, BandwidthMonitor)

    def test_offline_queue_property(self):
        mgr = ConnectionManager(vessel_id="v-1")
        assert isinstance(mgr.offline_queue, OfflineQueue)


class TestQueuedMessage:
    def test_create(self):
        msg = QueuedMessage(topic="test/topic", payload=b"data", qos=1, retain=True)
        assert msg.topic == "test/topic"
        assert msg.payload == b"data"
        assert msg.qos == 1
        assert msg.retain is True
        assert msg.retry_count == 0


class TestIntegrationFlow:
    """Integration tests: full edge-to-cloud message flow."""

    def test_full_flow_observation_to_cloud(self):
        client = MockMQTTClient()
        client.connect("cloud-broker")
        from jetson.agent.mqtt_bridge.telemetry_bridge import TelemetryBridge, TelemetryConfig
        config = TelemetryConfig(vessel_id="vessel-001")
        bridge = TelemetryBridge(config, client)
        bridge.start()

        obs = {
            "vessel_id": "vessel-001",
            "latitude": 37.7749,
            "longitude": -122.4194,
            "heading": 180.0,
            "timestamp_ms": 1000,
        }
        bridge.publish_observation(obs)
        msgs = client.get_published_messages()
        assert len(msgs) >= 1
        published_data = json.loads(msgs[0].payload)
        assert published_data["latitude"] == 37.7749

    def test_command_flow_cloud_to_edge(self):
        client = MockMQTTClient()
        client.connect("broker")
        from jetson.agent.mqtt_bridge.message_router import MessageRouter
        router = MessageRouter(vessel_id="vessel-001")
        commands_received = []
        router.register(
            "nexus/vessel-001/command",
            lambda msg, pr: commands_received.append(json.loads(msg.payload)),
            "command_handler",
        )
        client.subscribe("nexus/vessel-001/command")
        client.set_on_message(lambda msg: router.route(msg))

        command = {"action": "goto_waypoint", "lat": 37.0, "lon": -122.0}
        client.inject_message(
            "nexus/vessel-001/command",
            json.dumps(command).encode(),
        )
        assert len(commands_received) == 1
        assert commands_received[0]["action"] == "goto_waypoint"

    def test_safety_alert_bypasses_filters(self):
        client = MockMQTTClient()
        client.connect("broker")
        from jetson.agent.mqtt_bridge.telemetry_bridge import TelemetryBridge, TelemetryConfig
        config = TelemetryConfig(vessel_id="v-1", rate_limits={"default": 0.1})
        bridge = TelemetryBridge(config, client)
        for i in range(50):
            bridge.publish_safety_alert("RED", "SAFETY", f"Alert {i}")
        stats = bridge.stats
        assert stats["published"] == 50
        assert stats["dropped_rate_limit"] == 0

    def test_offline_queue_replay_on_reconnect(self):
        mgr = ConnectionManager(vessel_id="v-1")
        mgr.publish("nexus/v-1/telemetry", b"queued-data-1")
        mgr.publish("nexus/v-1/status", b"queued-data-2")
        assert mgr.offline_queue.size == 2
        mgr.connect()
        assert mgr.offline_queue.size == 0
