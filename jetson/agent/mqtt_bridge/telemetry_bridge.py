"""NEXUS Telemetry Bridge — Edge <-> Cloud translation.

Handles encoding/decoding of telemetry data for MQTT transport:
  - UnifiedObservation -> JSON -> MQTT publish (Edge -> Cloud)
  - MQTT subscribe -> JSON -> command objects (Cloud -> Edge)
  - Gzip compression for large observation batches
  - Batching: aggregate multiple observations into single message
  - Rate limiting: configurable per-topic publish rates
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .client import MQTTClientInterface, MQTTMessage
from .topics import QoSLevel, TopicPattern, TopicHierarchy, build_topic

logger = logging.getLogger("nexus.mqtt.telemetry")


# ===================================================================
# Compression
# ===================================================================

class CompressionType(str, Enum):
    """Compression types for MQTT payloads."""
    NONE = "none"
    GZIP = "gzip"


def compress_payload(data: bytes, compression: CompressionType) -> bytes:
    """Compress payload bytes using the specified method.

    Args:
        data: Raw payload bytes.
        compression: Compression type.

    Returns:
        Compressed bytes (or original if NONE).
    """
    if compression == CompressionType.GZIP:
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as f:
            f.write(data)
        return buf.getvalue()
    return data


def decompress_payload(data: bytes, compression: CompressionType) -> bytes:
    """Decompress payload bytes.

    Args:
        data: Compressed payload bytes.
        compression: Compression type used.

    Returns:
        Decompressed bytes (or original if NONE).
    """
    if compression == CompressionType.GZIP:
        buf = io.BytesIO(data)
        with gzip.GzipFile(fileobj=buf, mode="rb") as f:
            return f.read()
    return data


# ===================================================================
# Rate Limiter
# ===================================================================

@dataclass
class RateLimiter:
    """Token-bucket rate limiter for per-topic publish throttling.

    Usage:
        limiter = RateLimiter(max_rate=10.0)  # 10 msgs/sec
        if limiter.allow():
            client.publish(topic, payload)
    """

    max_rate: float = 10.0  # messages per second
    burst_size: int = 5
    _tokens: float = field(init=False, default=5.0)
    _last_refill: float = field(init=False, default_factory=time.time)

    def allow(self) -> bool:
        """Check if a message is allowed under the rate limit.

        Returns:
            True if the message can be published.
        """
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(
            self.max_rate * elapsed + self._tokens,
            float(self.burst_size),
        )
        self._last_refill = now

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def reset(self) -> None:
        """Reset the rate limiter to full capacity."""
        self._tokens = float(self.burst_size)
        self._last_refill = time.time()

    @property
    def available_tokens(self) -> float:
        """Current number of available tokens."""
        return self._tokens


# ===================================================================
# Batch Policy
# ===================================================================

@dataclass
class BatchPolicy:
    """Configuration for observation batching.

    Controls when accumulated observations are flushed to MQTT.
    """

    max_batch_size: int = 50  # Max observations per batch
    max_batch_bytes: int = 256 * 1024  # Max batch size in bytes (256 KB)
    max_age_seconds: float = 5.0  # Max time before auto-flush
    compression_threshold_bytes: int = 1024  # Compress if batch > this size
    compression: CompressionType = CompressionType.NONE

    @property
    def should_compress(self, data_size: int) -> bool:
        """Whether to compress data of the given size."""
        return (
            self.compression != CompressionType.NONE
            and data_size > self.compression_threshold_bytes
        )


# ===================================================================
# Telemetry Config
# ===================================================================

@dataclass
class TelemetryConfig:
    """Configuration for the telemetry bridge."""

    vessel_id: str = "unknown"
    batch_policy: BatchPolicy = field(default_factory=BatchPolicy)
    rate_limits: dict[str, float] = field(default_factory=lambda: {
        "telemetry": 10.0,    # 10 msgs/sec
        "position": 20.0,     # 20 msgs/sec (high frequency)
        "status": 1.0,        # 1 msg/sec
        "safety/alert": 0.0,  # unlimited (safety critical)
        "trust/events": 5.0,  # 5 msgs/sec
        "default": 5.0,       # default rate
    })
    max_message_size: int = 512 * 1024  # 512 KB max MQTT message
    include_timestamp: bool = True
    json_indent: int | None = None  # None = compact


# ===================================================================
# Telemetry Bridge
# ===================================================================

class TelemetryBridge:
    """Translates between NEXUS data types and MQTT messages.

    Handles:
    - Edge -> Cloud: Serialize observations/telemetry to JSON and publish
    - Cloud -> Edge: Receive JSON messages and parse into commands
    - Batching: Aggregate multiple observations into single messages
    - Compression: Gzip for large batches
    - Rate limiting: Per-topic throttling

    Usage:
        config = TelemetryConfig(vessel_id="vessel-001")
        bridge = TelemetryBridge(config, mqtt_client)
        bridge.start()

        # Publish telemetry
        bridge.publish_observation(observation)
        bridge.publish_observation_batch(observations)

        # Receive commands
        def on_command(msg):
            cmd = bridge.decode_command(msg)
            # process command...
    """

    def __init__(
        self,
        config: TelemetryConfig,
        client: MQTTClientInterface,
    ) -> None:
        self.config = config
        self.client = client
        self.hierarchy = TopicHierarchy(config.vessel_id)

        # Batching state
        self._batch: list[dict[str, Any]] = []
        self._batch_start_time: float = 0.0
        self._batch_size_bytes: int = 0

        # Rate limiters (per topic name)
        self._rate_limiters: dict[str, RateLimiter] = {}
        for topic_name, rate in config.rate_limits.items():
            self._rate_limiters[topic_name] = RateLimiter(
                max_rate=rate,
                burst_size=max(1, int(rate * 2)),
            )

        # Stats
        self._stats = {
            "published": 0,
            "batched": 0,
            "dropped_rate_limit": 0,
            "dropped_oversize": 0,
            "compressed": 0,
            "received": 0,
            "decode_errors": 0,
        }

        self._logger = logging.getLogger(
            f"nexus.mqtt.telemetry.{config.vessel_id}"
        )

    def start(self) -> None:
        """Start the telemetry bridge (subscribe to inbound topics)."""
        subs = self.hierarchy.get_vessel_subscriptions()
        for topic, qos in subs:
            self.client.subscribe(topic, int(qos))
        self._logger.info(
            "Telemetry bridge started for vessel %s (%d subscriptions)",
            self.config.vessel_id, len(subs),
        )

    def stop(self) -> None:
        """Stop the telemetry bridge and flush pending batch."""
        self.flush_batch()
        self._logger.info(
            "Telemetry bridge stopped for vessel %s", self.config.vessel_id
        )

    # ------------------------------------------------------------------
    # Publishing: Edge -> Cloud
    # ------------------------------------------------------------------

    def publish_observation(
        self,
        observation: Any,
        topic_pattern: TopicPattern = TopicPattern.TELEMETRY,
    ) -> bool:
        """Publish a single observation to MQTT.

        Args:
            observation: A UnifiedObservation (or any object with to_dict()).
            topic_pattern: Topic to publish to.

        Returns:
            True if published successfully, False if rate-limited or error.
        """
        # Rate check
        topic_name = self._get_topic_name(topic_pattern)
        if not self._check_rate_limit(topic_name):
            self._stats["dropped_rate_limit"] += 1
            return False

        # Serialize observation
        payload_dict = self._serialize_observation(observation)

        return self._publish_json(topic_pattern, payload_dict)

    def publish_observation_batch(
        self,
        observations: list[Any],
        topic_pattern: TopicPattern = TopicPattern.TELEMETRY,
    ) -> int:
        """Publish a batch of observations, respecting batching policy.

        If the number of observations exceeds the batch policy,
        they are split into multiple messages.

        Args:
            observations: List of observations.
            topic_pattern: Topic to publish to.

        Returns:
            Number of messages actually published.
        """
        published = 0

        for obs in observations:
            obs_dict = self._serialize_observation(obs)
            self._batch.append(obs_dict)
            self._batch_size_bytes += len(json.dumps(obs_dict))
            self._stats["batched"] += 1

            # Check flush conditions
            should_flush = (
                len(self._batch) >= self.config.batch_policy.max_batch_size
                or self._batch_size_bytes >= self.config.batch_policy.max_batch_bytes
                or (self._batch_start_time > 0
                    and time.time() - self._batch_start_time
                    >= self.config.batch_policy.max_age_seconds)
            )

            if should_flush:
                if self.flush_batch(topic_pattern):
                    published += 1

        return published

    def add_to_batch(self, observation: Any) -> bool:
        """Add an observation to the pending batch.

        Args:
            observation: Observation to add.

        Returns:
            True if the batch is now full and ready to flush.
        """
        obs_dict = self._serialize_observation(observation)
        self._batch.append(obs_dict)
        self._batch_size_bytes += len(json.dumps(obs_dict))
        self._stats["batched"] += 1

        if not self._batch_start_time:
            self._batch_start_time = time.time()

        return (
            len(self._batch) >= self.config.batch_policy.max_batch_size
            or self._batch_size_bytes >= self.config.batch_policy.max_batch_bytes
        )

    def flush_batch(
        self,
        topic_pattern: TopicPattern = TopicPattern.TELEMETRY,
    ) -> bool:
        """Flush the pending observation batch to MQTT.

        Returns:
            True if a message was published.
        """
        if not self._batch:
            return False

        # Rate check
        topic_name = self._get_topic_name(topic_pattern)
        if not self._check_rate_limit(topic_name):
            self._stats["dropped_rate_limit"] += 1
            return False

        batch_data = {
            "type": "observation_batch",
            "vessel_id": self.config.vessel_id,
            "count": len(self._batch),
            "timestamp": time.time(),
            "observations": self._batch,
        }

        result = self._publish_json(topic_pattern, batch_data)

        # Reset batch
        self._batch = []
        self._batch_start_time = 0.0
        self._batch_size_bytes = 0

        return result

    def publish_status(
        self,
        status: dict[str, Any],
        topic_pattern: TopicPattern = TopicPattern.STATUS,
    ) -> bool:
        """Publish a vessel status message.

        Args:
            status: Status dictionary.
            topic_pattern: Topic to publish to.

        Returns:
            True if published.
        """
        topic_name = self._get_topic_name(topic_pattern)
        if not self._check_rate_limit(topic_name):
            self._stats["dropped_rate_limit"] += 1
            return False

        payload = {
            "vessel_id": self.config.vessel_id,
            "timestamp": time.time(),
            **status,
        }
        return self._publish_json(topic_pattern, payload)

    def publish_position(
        self,
        latitude: float,
        longitude: float,
        heading: float = 0.0,
        speed: float = 0.0,
        accuracy: float = 0.0,
        timestamp_ms: int = 0,
    ) -> bool:
        """Publish a high-frequency position update.

        Args:
            latitude: Latitude in decimal degrees.
            longitude: Longitude in decimal degrees.
            heading: Heading in degrees.
            speed: Speed in m/s.
            accuracy: Position accuracy in meters.
            timestamp_ms: Timestamp in milliseconds.

        Returns:
            True if published.
        """
        topic_name = self._get_topic_name(TopicPattern.POSITION)
        if not self._check_rate_limit(topic_name):
            self._stats["dropped_rate_limit"] += 1
            return False

        payload = {
            "vessel_id": self.config.vessel_id,
            "lat": latitude,
            "lon": longitude,
            "hdg": heading,
            "spd": speed,
            "acc": accuracy,
            "ts": timestamp_ms or int(time.time() * 1000),
        }

        topic = self.hierarchy.build(TopicPattern.POSITION)
        # Position uses compact JSON
        data = json.dumps(payload, separators=(",", ":"))
        qos = int(_get_topic_qos_map().get(TopicPattern.POSITION, QoSLevel.AT_MOST_ONCE))

        return self._do_publish(topic, data.encode("utf-8"), qos)

    def publish_safety_alert(
        self,
        level: str,
        category: str,
        description: str,
        details: dict[str, Any] | None = None,
    ) -> bool:
        """Publish a safety alert (bypasses rate limiting).

        Safety alerts are never rate-limited.

        Args:
            level: Alert level (GREEN, YELLOW, ORANGE, RED).
            category: Alert category.
            description: Human-readable description.
            details: Additional details dict.

        Returns:
            True if published.
        """
        payload = {
            "vessel_id": self.config.vessel_id,
            "level": level,
            "category": category,
            "description": description,
            "timestamp": time.time(),
            "details": details or {},
        }
        return self._publish_json(TopicPattern.SAFETY_ALERT, payload)

    def publish_trust_event(
        self,
        subsystem: str,
        event_type: str,
        old_score: float,
        new_score: float,
        delta: float,
        branch: str = "",
    ) -> bool:
        """Publish a trust score change event.

        Args:
            subsystem: Affected subsystem.
            event_type: Type of trust event.
            old_score: Previous trust score.
            new_score: New trust score.
            delta: Trust delta.
            branch: Algorithm branch (gain, penalty, decay).

        Returns:
            True if published.
        """
        topic_name = self._get_topic_name(TopicPattern.TRUST_EVENTS)
        if not self._check_rate_limit(topic_name):
            self._stats["dropped_rate_limit"] += 1
            return False

        payload = {
            "vessel_id": self.config.vessel_id,
            "subsystem": subsystem,
            "event_type": event_type,
            "old_score": old_score,
            "new_score": new_score,
            "delta": delta,
            "branch": branch,
            "timestamp": time.time(),
        }
        return self._publish_json(TopicPattern.TRUST_EVENTS, payload)

    # ------------------------------------------------------------------
    # Receiving: Cloud -> Edge
    # ------------------------------------------------------------------

    def decode_command(self, msg: MQTTMessage) -> dict[str, Any]:
        """Decode an MQTT message into a command object.

        Args:
            msg: Received MQTT message.

        Returns:
            Parsed command dictionary.

        Raises:
            ValueError: If the message cannot be decoded.
        """
        self._stats["received"] += 1

        try:
            # Check for compression header
            payload = msg.payload
            compression = CompressionType.NONE

            # Try JSON decode first
            data = json.loads(payload)
            if isinstance(data, dict):
                # Check for embedded compression
                if data.get("_compression") == "gzip":
                    compression = CompressionType.GZIP
                    raw = base64.b64decode(data["_payload"])
                    payload = decompress_payload(raw, compression)
                    data = json.loads(payload)

            data["_topic"] = msg.topic
            data["_qos"] = msg.qos
            data["_mid"] = msg.mid
            data["_timestamp"] = msg.timestamp

            return data

        except (json.JSONDecodeError, UnicodeDecodeError, Exception) as e:
            self._stats["decode_errors"] += 1
            raise ValueError(f"Failed to decode command: {e}") from e

    def decode_message(self, msg: MQTTMessage) -> dict[str, Any]:
        """Decode any MQTT message payload to a dictionary.

        Similar to decode_command but works for any message type.

        Args:
            msg: MQTT message.

        Returns:
            Parsed message dictionary.
        """
        return self.decode_command(msg)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _serialize_observation(self, observation: Any) -> dict[str, Any]:
        """Serialize an observation to a dictionary."""
        if hasattr(observation, "to_dict"):
            return observation.to_dict()
        if isinstance(observation, dict):
            return observation
        # Fallback: try to get all public attributes
        result = {}
        for attr in dir(observation):
            if not attr.startswith("_") and not callable(getattr(observation, attr)):
                result[attr] = getattr(observation, attr)
        return result

    def _publish_json(
        self,
        topic_pattern: TopicPattern,
        data: dict[str, Any],
    ) -> bool:
        """Publish a JSON payload to a topic.

        Args:
            topic_pattern: Topic pattern to publish to.
            data: Data to serialize and publish.

        Returns:
            True if published successfully.
        """
        topic = self.hierarchy.build(topic_pattern)
        indent = self.config.json_indent
        json_bytes = json.dumps(
            data,
            indent=indent,
            separators=((",", ":") if indent is None else None),
            default=str,
        ).encode("utf-8")

        # Check message size
        if len(json_bytes) > self.config.max_message_size:
            # Try compression
            if self.config.batch_policy.should_compress(len(json_bytes)):
                compressed = compress_payload(
                    json_bytes, self.config.batch_policy.compression
                )
                if len(compressed) < len(json_bytes):
                    # Wrap in compression envelope
                    envelope = json.dumps({
                        "_compression": self.config.batch_policy.compression.value,
                        "_payload": base64.b64encode(compressed).decode("ascii"),
                        "_original_size": len(json_bytes),
                    }, separators=(",", ":")).encode("utf-8")

                    if len(envelope) <= self.config.max_message_size:
                        qos = int(_get_topic_qos_map().get(
                            topic_pattern, QoSLevel.AT_LEAST_ONCE
                        ))
                        self._stats["compressed"] += 1
                        return self._do_publish(topic, envelope, qos)

            self._stats["dropped_oversize"] += 1
            self._logger.warning(
                "Message too large for topic %s: %d bytes (max %d)",
                topic, len(json_bytes), self.config.max_message_size,
            )
            return False

        qos = int(_get_topic_qos_map().get(topic_pattern, QoSLevel.AT_LEAST_ONCE))
        return self._do_publish(topic, json_bytes, qos)

    def _do_publish(self, topic: str, payload: bytes, qos: int) -> bool:
        """Execute the actual publish via the MQTT client."""
        result = self.client.publish(topic, payload, qos=qos)
        if result >= 0:
            self._stats["published"] += 1
            return True
        return False

    def _check_rate_limit(self, topic_name: str) -> bool:
        """Check rate limit for a topic.

        Safety alerts bypass rate limiting.
        """
        # Safety alerts are never rate-limited
        if topic_name == "safety/alert":
            return True

        limiter = self._rate_limiters.get(topic_name)
        if limiter is None:
            limiter = self._rate_limiters.get("default")
        if limiter is None:
            return True
        return limiter.allow()

    @staticmethod
    def _get_topic_name(pattern: TopicPattern) -> str:
        """Extract the short topic name from a pattern."""
        parts = pattern.value.split("/")
        # nexus/{vessel_id}/name or nexus/fleet/name
        if len(parts) >= 3:
            return parts[2] if parts[1] != "fleet" else "fleet/" + parts[2]
        return pattern.value

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pending_batch_size(self) -> int:
        """Number of observations in the current batch."""
        return len(self._batch)

    @property
    def stats(self) -> dict[str, int]:
        """Telemetry bridge statistics."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset all statistics counters."""
        for key in self._stats:
            self._stats[key] = 0

    def reset_rate_limiters(self) -> None:
        """Reset all rate limiters to full capacity."""
        for limiter in self._rate_limiters.values():
            limiter.reset()


# QoS mapping from topic pattern
_TOPIC_QOS_MAP_CACHE: dict[TopicPattern, QoSLevel] | None = None

def _get_topic_qos_map() -> dict[TopicPattern, QoSLevel]:
    global _TOPIC_QOS_MAP_CACHE
    if _TOPIC_QOS_MAP_CACHE is None:
        from .topics import TOPIC_DEFINITIONS
        _TOPIC_QOS_MAP_CACHE = {
            pattern: defn.qos
            for pattern, defn in TOPIC_DEFINITIONS.items()
        }
    return _TOPIC_QOS_MAP_CACHE
