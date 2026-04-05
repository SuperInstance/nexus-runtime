"""Tests for Telemetry Bridge — encoding/decoding, batching, compression, rate limiting."""

import base64
import gzip
import io
import json
import time
import pytest

from jetson.agent.mqtt_bridge.client import MockMQTTClient, MQTTMessage
from jetson.agent.mqtt_bridge.topics import TopicPattern, QoSLevel
from jetson.agent.mqtt_bridge.telemetry_bridge import (
    TelemetryBridge,
    TelemetryConfig,
    BatchPolicy,
    CompressionType,
    RateLimiter,
    compress_payload,
    decompress_payload,
)


# ===================================================================
# Helpers
# ===================================================================

def make_observation(
    vessel_id: str = "vessel-001",
    timestamp_ms: int = 1000,
    **overrides,
) -> dict:
    """Create a mock observation dict."""
    obs = {
        "vessel_id": vessel_id,
        "timestamp_ms": timestamp_ms,
        "latitude": 37.7749,
        "longitude": -122.4194,
        "heading": 180.0,
        "speed_over_ground": 2.5,
        "course_over_ground": 185.0,
        "trust_score": 0.85,
        "trust_level": 4,
        "safety_state": "NOMINAL",
        "safety_flags": 0,
        "program_counter": 42,
        "cycle_count": 1000,
        "uptime_ms": 60000,
    }
    obs.update(overrides)
    return obs


class MockObservation:
    """Mock observation with to_dict()."""
    def __init__(self, data: dict):
        self._data = data

    def to_dict(self) -> dict:
        return dict(self._data)


# ===================================================================
# Compression Tests
# ===================================================================

class TestCompression:
    def test_compress_gzip(self):
        data = b"hello world, this is test data for compression"
        compressed = compress_payload(data, CompressionType.GZIP)
        assert compressed != data
        assert len(compressed) < len(data) or len(data) < 50  # Small data may not compress

    def test_compress_none(self):
        data = b"hello"
        result = compress_payload(data, CompressionType.NONE)
        assert result == data

    def test_decompress_gzip_roundtrip(self):
        original = b"test data for roundtrip compression check"
        compressed = compress_payload(original, CompressionType.GZIP)
        decompressed = decompress_payload(compressed, CompressionType.GZIP)
        assert decompressed == original

    def test_decompress_none(self):
        data = b"hello"
        result = decompress_payload(data, CompressionType.NONE)
        assert result == data

    def test_gzip_large_data(self):
        original = b"x" * 10000
        compressed = compress_payload(original, CompressionType.GZIP)
        decompressed = decompress_payload(compressed, CompressionType.GZIP)
        assert decompressed == original
        assert len(compressed) < len(original)

    def test_invalid_gzip_raises(self):
        with pytest.raises(Exception):
            decompress_payload(b"not gzip data", CompressionType.GZIP)


# ===================================================================
# Rate Limiter Tests
# ===================================================================

class TestRateLimiter:
    def test_initial_tokens(self):
        limiter = RateLimiter(max_rate=10.0, burst_size=5)
        assert limiter.available_tokens == 5.0

    def test_allow_within_rate(self):
        limiter = RateLimiter(max_rate=10.0, burst_size=5)
        assert limiter.allow()
        assert limiter.allow()
        assert limiter.allow()
        assert limiter.allow()
        assert limiter.allow()
        assert not limiter.allow()  # Burst exhausted

    def test_tokens_refill_over_time(self):
        limiter = RateLimiter(max_rate=100.0, burst_size=5)
        # Exhaust tokens
        for _ in range(5):
            limiter.allow()
        assert not limiter.allow()

        # Simulate time passing (0.1 seconds should refill ~10 tokens)
        time.sleep(0.1)
        assert limiter.allow()

    def test_reset(self):
        limiter = RateLimiter(max_rate=10.0, burst_size=5)
        for _ in range(5):
            limiter.allow()
        assert not limiter.allow()
        limiter.reset()
        assert limiter.allow()

    def test_unlimited_rate(self):
        limiter = RateLimiter(max_rate=0.0, burst_size=100)
        # With max_rate=0, only burst matters
        for _ in range(100):
            assert limiter.allow()
        assert not limiter.allow()


# ===================================================================
# Batch Policy Tests
# ===================================================================

class TestBatchPolicy:
    def test_default_values(self):
        policy = BatchPolicy()
        assert policy.max_batch_size == 50
        assert policy.max_batch_bytes == 256 * 1024
        assert policy.max_age_seconds == 5.0

    def test_should_compress_large_data(self):
        policy = BatchPolicy(
            compression=CompressionType.GZIP,
            compression_threshold_bytes=100,
        )
        assert policy.should_compress(200)

    def test_should_not_compress_small_data(self):
        policy = BatchPolicy(
            compression=CompressionType.GZIP,
            compression_threshold_bytes=100,
        )
        assert not policy.should_compress(50)

    def test_no_compression_never_compresses(self):
        policy = BatchPolicy(compression=CompressionType.NONE)
        assert not policy.should_compress(100000)


# ===================================================================
# Telemetry Bridge Tests
# ===================================================================

class TestTelemetryBridgePublish:
    @pytest.fixture
    def bridge(self):
        client = MockMQTTClient()
        client.connect("localhost")
        config = TelemetryConfig(vessel_id="vessel-001")
        return TelemetryBridge(config, client)

    def test_publish_observation_dict(self, bridge):
        obs = make_observation()
        result = bridge.publish_observation(obs)
        assert result is True

        msgs = bridge.client.get_published_messages()
        assert len(msgs) == 1
        data = json.loads(msgs[0].payload)
        assert data["vessel_id"] == "vessel-001"
        assert "latitude" in data

    def test_publish_observation_object(self, bridge):
        obs = MockObservation(make_observation())
        result = bridge.publish_observation(obs)
        assert result is True

    def test_publish_to_telemetry_topic(self, bridge):
        bridge.publish_observation(make_observation())
        msgs = bridge.client.get_published_messages()
        assert msgs[0].topic == "nexus/vessel-001/telemetry"

    def test_publish_status(self, bridge):
        status = {"health": "ok", "battery": 95.5}
        result = bridge.publish_status(status)
        assert result is True

        msgs = bridge.client.get_published_messages()
        data = json.loads(msgs[0].payload)
        assert data["health"] == "ok"
        assert data["battery"] == 95.5

    def test_publish_position(self, bridge):
        result = bridge.publish_position(
            latitude=37.7749,
            longitude=-122.4194,
            heading=180.0,
            speed=2.5,
        )
        assert result is True

        msgs = bridge.client.get_published_messages()
        data = json.loads(msgs[0].payload)
        assert data["lat"] == 37.7749
        assert data["lon"] == -122.4194
        assert data["hdg"] == 180.0

    def test_publish_safety_alert(self, bridge):
        result = bridge.publish_safety_alert(
            level="RED",
            category="SAFETY",
            description="E-Stop triggered",
        )
        assert result is True

        msgs = bridge.client.get_published_messages()
        data = json.loads(msgs[0].payload)
        assert data["level"] == "RED"
        assert data["category"] == "SAFETY"

    def test_publish_trust_event(self, bridge):
        result = bridge.publish_trust_event(
            subsystem="steering",
            event_type="heartbeat_missed",
            old_score=0.85,
            new_score=0.80,
            delta=-0.05,
            branch="penalty",
        )
        assert result is True

        msgs = bridge.client.get_published_messages()
        data = json.loads(msgs[0].payload)
        assert data["subsystem"] == "steering"
        assert data["delta"] == -0.05

    def test_safety_alert_bypasses_rate_limit(self, bridge):
        """Safety alerts should never be rate-limited."""
        # Exhaust the rate limiter
        for _ in range(100):
            bridge.publish_safety_alert("RED", "SAFETY", "test")
        # All should succeed
        assert bridge.stats["published"] >= 100
        assert bridge.stats["dropped_rate_limit"] == 0

    def test_publish_increments_counter(self, bridge):
        bridge.publish_observation(make_observation())
        assert bridge.stats["published"] == 1

    def test_publish_while_disconnected_fails(self):
        client = MockMQTTClient()
        config = TelemetryConfig(vessel_id="v-1")
        bridge = TelemetryBridge(config, client)
        # Not connected
        result = bridge.publish_observation(make_observation())
        assert result is False


class TestTelemetryBridgeBatching:
    @pytest.fixture
    def bridge(self):
        client = MockMQTTClient()
        client.connect("localhost")
        config = TelemetryConfig(
            vessel_id="v-1",
            batch_policy=BatchPolicy(max_batch_size=5),
        )
        return TelemetryBridge(config, client)

    def test_add_to_batch(self, bridge):
        obs = make_observation()
        full = bridge.add_to_batch(obs)
        assert not full  # 1 < 5

    def test_batch_flushes_when_full(self, bridge):
        for i in range(5):
            full = bridge.add_to_batch(make_observation(timestamp_ms=i * 100))
            if i < 4:
                assert not full
            else:
                assert full

    def test_flush_batch(self, bridge):
        for i in range(3):
            bridge.add_to_batch(make_observation(timestamp_ms=i * 100))
        result = bridge.flush_batch()
        assert result is True

        msgs = bridge.client.get_published_messages()
        data = json.loads(msgs[0].payload)
        assert data["type"] == "observation_batch"
        assert data["count"] == 3
        assert data["vessel_id"] == "v-1"

    def test_flush_empty_batch(self, bridge):
        result = bridge.flush_batch()
        assert result is False

    def test_publish_observation_batch(self, bridge):
        obs_list = [make_observation(timestamp_ms=i * 100) for i in range(12)]
        published = bridge.publish_observation_batch(obs_list)
        # 12 obs / max_batch_size 5 = 3 messages (5, 5, 2)
        assert published == 3

    def test_pending_batch_size(self, bridge):
        bridge.add_to_batch(make_observation())
        bridge.add_to_batch(make_observation())
        assert bridge.pending_batch_size == 2

    def test_batch_stats(self, bridge):
        bridge.add_to_batch(make_observation())
        bridge.add_to_batch(make_observation())
        assert bridge.stats["batched"] == 2
        bridge.flush_batch()
        assert bridge.stats["batched"] == 2


class TestTelemetryBridgeRateLimiting:
    def test_telemetry_rate_limited(self):
        client = MockMQTTClient()
        client.connect("localhost")
        config = TelemetryConfig(
            vessel_id="v-1",
            rate_limits={"telemetry": 2.0, "default": 2.0},
        )
        bridge = TelemetryBridge(config, client)

        # First few should succeed (burst), then get rate-limited
        success_count = 0
        for _ in range(20):
            if bridge.publish_observation(make_observation()):
                success_count += 1

        assert bridge.stats["dropped_rate_limit"] > 0

    def test_position_rate_limit(self):
        client = MockMQTTClient()
        client.connect("localhost")
        config = TelemetryConfig(
            vessel_id="v-1",
            rate_limits={"position": 1.0, "default": 1.0},
        )
        bridge = TelemetryBridge(config, client)

        # Some should be rate-limited
        for _ in range(20):
            bridge.publish_position(37.0, -122.0)

        assert bridge.stats["dropped_rate_limit"] > 0


class TestTelemetryBridgeDecode:
    @pytest.fixture
    def bridge(self):
        client = MockMQTTClient()
        client.connect("localhost")
        config = TelemetryConfig(vessel_id="v-1")
        return TelemetryBridge(config, client)

    def test_decode_command_json(self):
        config = TelemetryConfig(vessel_id="v-1")
        client = MockMQTTClient()
        client.connect("localhost")
        bridge = TelemetryBridge(config, client)

        cmd_data = {"action": "waypoint", "lat": 37.0, "lon": -122.0}
        msg = MQTTMessage(
            topic="nexus/v-1/command",
            payload=json.dumps(cmd_data).encode("utf-8"),
            qos=1,
            mid=1,
        )
        result = bridge.decode_command(msg)
        assert result["action"] == "waypoint"
        assert result["lat"] == 37.0
        assert result["_topic"] == "nexus/v-1/command"

    def test_decode_malformed_json_raises(self):
        config = TelemetryConfig(vessel_id="v-1")
        client = MockMQTTClient()
        client.connect("localhost")
        bridge = TelemetryBridge(config, client)

        msg = MQTTMessage(topic="t", payload=b"not json {{{")
        with pytest.raises(ValueError, match="Failed to decode"):
            bridge.decode_command(msg)

    def test_decode_binary_payload_raises(self):
        config = TelemetryConfig(vessel_id="v-1")
        client = MockMQTTClient()
        client.connect("localhost")
        bridge = TelemetryBridge(config, client)

        msg = MQTTMessage(topic="t", payload=b"\x00\x01\x02\xff")
        with pytest.raises(ValueError):
            bridge.decode_command(msg)

    def test_decode_increments_received_stat(self):
        config = TelemetryConfig(vessel_id="v-1")
        client = MockMQTTClient()
        client.connect("localhost")
        bridge = TelemetryBridge(config, client)

        msg = MQTTMessage(topic="t", payload=b'{"key": "val"}')
        bridge.decode_command(msg)
        assert bridge.stats["received"] == 1

    def test_decode_error_increments_counter(self):
        config = TelemetryConfig(vessel_id="v-1")
        client = MockMQTTClient()
        client.connect("localhost")
        bridge = TelemetryBridge(config, client)

        msg = MQTTMessage(topic="t", payload=b"invalid")
        with pytest.raises(ValueError):
            bridge.decode_command(msg)
        assert bridge.stats["decode_errors"] == 1


class TestTelemetryBridgeStats:
    @pytest.fixture
    def bridge(self):
        client = MockMQTTClient()
        client.connect("localhost")
        config = TelemetryConfig(vessel_id="v-1")
        return TelemetryBridge(config, client)

    def test_initial_stats(self, bridge):
        stats = bridge.stats
        assert stats["published"] == 0
        assert stats["dropped_rate_limit"] == 0
        assert stats["decode_errors"] == 0

    def test_reset_stats(self, bridge):
        bridge.publish_observation(make_observation())
        bridge.reset_stats()
        assert bridge.stats["published"] == 0

    def test_reset_rate_limiters(self, bridge):
        # Exhaust rate limiter
        for _ in range(20):
            bridge.publish_observation(make_observation())
        assert bridge.stats["dropped_rate_limit"] > 0
        bridge.reset_rate_limiters()
        # After reset, should be able to publish again (burst size)
        result = bridge.publish_observation(make_observation())
        assert result is True


class TestTelemetryBridgeStartStop:
    def test_start_subscribes(self):
        client = MockMQTTClient()
        client.connect("localhost")
        config = TelemetryConfig(vessel_id="v-1")
        bridge = TelemetryBridge(config, client)

        bridge.start()
        subs = client.get_subscriptions()
        assert len(subs) >= 10  # vessel + fleet inbound topics

    def test_stop_flushes_batch(self):
        client = MockMQTTClient()
        client.connect("localhost")
        config = TelemetryConfig(vessel_id="v-1")
        bridge = TelemetryBridge(config, client)

        bridge.add_to_batch(make_observation())
        assert bridge.pending_batch_size == 1

        bridge.stop()
        assert bridge.pending_batch_size == 0

    def test_oversized_message_dropped(self):
        client = MockMQTTClient()
        client.connect("localhost")
        config = TelemetryConfig(
            vessel_id="v-1",
            max_message_size=10,  # Very small limit
        )
        bridge = TelemetryBridge(config, client)

        result = bridge.publish_observation(make_observation())
        assert result is False
        assert bridge.stats["dropped_oversize"] > 0
