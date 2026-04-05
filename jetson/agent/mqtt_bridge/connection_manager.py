"""NEXUS MQTT Connection Manager — Multi-broker, health, and lifecycle.

Provides:
  - Multi-broker support (primary + backup)
  - Automatic reconnection with exponential backoff
  - Health monitoring (ping/pong, latency measurement)
  - Bandwidth monitoring and throttling
  - Graceful shutdown (send offline status before disconnect)
  - Offline message queue (persist to memory, replay on reconnect)
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .client import (
    MQTTClientInterface,
    MockMQTTClient,
    MQTTMessage,
    ConnectionState,
    ConnectionEvent,
)
from .topics import TopicHierarchy, TopicPattern, QoSLevel, all_vessel_subscriptions

logger = logging.getLogger("nexus.mqtt.connection")


# ===================================================================
# Broker Configuration
# ===================================================================

@dataclass
class BrokerConfig:
    """Configuration for a single MQTT broker."""

    host: str = "localhost"
    port: int = 1883
    keepalive: int = 60
    clean_session: bool = True
    client_id: str = ""
    username: str = ""
    password: str = ""
    tls_enabled: bool = False
    tls_ca_cert: str = ""  # Path to CA certificate
    tls_cert: str = ""     # Path to client certificate
    tls_key: str = ""      # Path to client key
    tls_insecure: bool = False  # Skip certificate verification
    priority: int = 0  # 0 = primary, 1+ = backup (lower = higher priority)

    @property
    def address(self) -> str:
        """Broker address as host:port."""
        return f"{self.host}:{self.port}"

    @property
    def id(self) -> str:
        """Unique broker identifier."""
        return f"{self.host}:{self.port}"


# ===================================================================
# Connection Health
# ===================================================================

@dataclass
class ConnectionHealth:
    """Health metrics for the MQTT connection."""

    connected: bool = False
    broker_id: str = ""
    connect_time: float = 0.0
    last_message_time: float = 0.0
    last_ping_time: float = 0.0
    ping_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    reconnect_count: int = 0
    consecutive_errors: int = 0
    uptime_seconds: float = 0.0
    _latency_samples: deque = field(default_factory=lambda: deque(maxlen=100))

    def record_latency(self, latency_ms: float) -> None:
        """Record a latency sample."""
        self._latency_samples.append(latency_ms)
        if self._latency_samples:
            self.avg_latency_ms = sum(self._latency_samples) / len(self._latency_samples)

    @property
    def is_healthy(self) -> bool:
        """Connection is considered healthy if connected and recent activity."""
        if not self.connected:
            return False
        # Consider healthy if we've had activity in the last 2x keepalive
        if self.last_message_time > 0:
            age = time.time() - self.last_message_time
            return age < 120  # 2 minutes
        if self.last_ping_time > 0:
            age = time.time() - self.last_ping_time
            return age < 120
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize health to dictionary."""
        return {
            "connected": self.connected,
            "broker_id": self.broker_id,
            "connect_time": self.connect_time,
            "last_message_time": self.last_message_time,
            "ping_latency_ms": self.ping_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "reconnect_count": self.reconnect_count,
            "consecutive_errors": self.consecutive_errors,
            "uptime_seconds": self.uptime_seconds,
            "is_healthy": self.is_healthy,
        }


# ===================================================================
# Bandwidth Monitor
# ===================================================================

@dataclass
class BandwidthSample:
    """A single bandwidth measurement window."""
    timestamp: float
    bytes_sent: int
    bytes_received: int
    messages_sent: int
    messages_received: int


class BandwidthMonitor:
    """Tracks bandwidth usage over sliding windows.

    Provides:
    - Current bandwidth (bytes/sec)
    - Peak bandwidth
    - Throttling support
    """

    def __init__(
        self,
        max_bandwidth_bps: int = 0,  # 0 = unlimited
        window_seconds: float = 60.0,
    ) -> None:
        self.max_bandwidth_bps = max_bandwidth_bps
        self.window_seconds = window_seconds
        self._samples: deque[BandwidthSample] = deque(maxlen=3600)
        self._current_window = BandwidthSample(
            timestamp=time.time(), bytes_sent=0, bytes_received=0,
            messages_sent=0, messages_received=0,
        )
        self._total_bytes_sent = 0
        self._total_bytes_received = 0
        self._peak_bps = 0.0

    def record_send(self, byte_count: int) -> None:
        """Record bytes sent."""
        self._rotate_window()
        self._current_window.bytes_sent += byte_count
        self._total_bytes_sent += byte_count

    def record_receive(self, byte_count: int) -> None:
        """Record bytes received."""
        self._rotate_window()
        self._current_window.bytes_received += byte_count
        self._total_bytes_received += byte_count

    def record_message_sent(self) -> None:
        """Record a message sent."""
        self._rotate_window()
        self._current_window.messages_sent += 1

    def record_message_received(self) -> None:
        """Record a message received."""
        self._rotate_window()
        self._current_window.messages_received += 1

    @property
    def current_send_bps(self) -> float:
        """Current send bandwidth in bytes per second."""
        return self._window_rate("bytes_sent")

    @property
    def current_receive_bps(self) -> float:
        """Current receive bandwidth in bytes per second."""
        return self._window_rate("bytes_received")

    @property
    def is_throttled(self) -> bool:
        """Whether bandwidth is being throttled."""
        if self.max_bandwidth_bps <= 0:
            return False
        total_bps = self.current_send_bps + self.current_receive_bps
        return total_bps > self.max_bandwidth_bps

    @property
    def throttle_ratio(self) -> float:
        """Ratio of current bandwidth to max (0.0 to 1.0+)."""
        if self.max_bandwidth_bps <= 0:
            return 0.0
        total_bps = self.current_send_bps + self.current_receive_bps
        return total_bps / self.max_bandwidth_bps

    @property
    def peak_bps(self) -> float:
        """Peak observed bandwidth."""
        return self._peak_bps

    @property
    def total_bytes_sent(self) -> int:
        return self._total_bytes_sent

    @property
    def total_bytes_received(self) -> int:
        return self._total_bytes_received

    def _rotate_window(self) -> None:
        """Start a new measurement window if the current one has expired."""
        now = time.time()
        if now - self._current_window.timestamp >= self.window_seconds:
            # Calculate rate for the completed window
            duration = now - self._current_window.timestamp
            total = self._current_window.bytes_sent + self._current_window.bytes_received
            if duration > 0:
                rate = total / duration
                if rate > self._peak_bps:
                    self._peak_bps = rate
            self._samples.append(self._current_window)
            self._current_window = BandwidthSample(
                timestamp=now, bytes_sent=0, bytes_received=0,
                messages_sent=0, messages_received=0,
            )

    def _window_rate(self, attr: str) -> float:
        """Calculate rate for an attribute over the current window."""
        elapsed = time.time() - self._current_window.timestamp
        if elapsed <= 0:
            return 0.0
        value = getattr(self._current_window, attr, 0)
        return value / elapsed

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "current_send_bps": self.current_send_bps,
            "current_receive_bps": self.current_receive_bps,
            "peak_bps": self.peak_bps,
            "is_throttled": self.is_throttled,
            "throttle_ratio": self.throttle_ratio,
            "total_bytes_sent": self._total_bytes_sent,
            "total_bytes_received": self._total_bytes_received,
            "max_bandwidth_bps": self.max_bandwidth_bps,
        }


# ===================================================================
# Offline Queue — Persist messages during disconnection
# ===================================================================

@dataclass
class QueuedMessage:
    """A message queued for later delivery."""
    topic: str
    payload: bytes
    qos: int
    retain: bool
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0


class OfflineQueue:
    """Queue for messages that need to be sent when connection is restored.

    Can persist to disk and replay on reconnect.
    """

    def __init__(
        self,
        max_size: int = 10000,
        persist_path: str | None = None,
    ) -> None:
        self._queue: deque[QueuedMessage] = deque(maxlen=max_size)
        self._max_size = max_size
        self._persist_path = persist_path
        self._stats = {
            "enqueued": 0,
            "dequeued": 0,
            "replayed": 0,
            "dropped": 0,
            "persisted": 0,
            "loaded": 0,
        }

    def enqueue(
        self,
        topic: str,
        payload: bytes,
        qos: int = 1,
        retain: bool = False,
    ) -> bool:
        """Add a message to the offline queue.

        Returns:
            True if enqueued, False if queue is full.
        """
        if len(self._queue) >= self._max_size:
            self._stats["dropped"] += 1
            return False

        msg = QueuedMessage(
            topic=topic,
            payload=payload,
            qos=qos,
            retain=retain,
        )
        self._queue.append(msg)
        self._stats["enqueued"] += 1
        return True

    def dequeue(self) -> QueuedMessage | None:
        """Remove and return the oldest queued message."""
        if self._queue:
            msg = self._queue.popleft()
            self._stats["dequeued"] += 1
            return msg
        return None

    def peek(self) -> QueuedMessage | None:
        """Look at the oldest message without removing it."""
        return self._queue[0] if self._queue else None

    def replay(self, client: MQTTClientInterface) -> int:
        """Replay all queued messages through the client.

        Args:
            client: Connected MQTT client.

        Returns:
            Number of messages replayed.
        """
        replayed = 0
        while self._queue:
            msg = self._queue.popleft()
            result = client.publish(msg.topic, msg.payload, msg.qos, msg.retain)
            if result >= 0:
                replayed += 1
            else:
                # Re-queue on failure
                msg.retry_count += 1
                if msg.retry_count < 3:
                    self._queue.appendleft(msg)
                break
        self._stats["replayed"] += replayed
        return replayed

    def persist(self) -> bool:
        """Persist the queue to disk.

        Returns:
            True if successful.
        """
        if not self._persist_path:
            return False
        try:
            os.makedirs(self._persist_path, exist_ok=True)
            filepath = os.path.join(self._persist_path, "offline_queue.pkl")
            data = list(self._queue)
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
            self._stats["persisted"] += len(data)
            return True
        except Exception as e:
            logger.error("Failed to persist offline queue: %s", e)
            return False

    def load(self) -> int:
        """Load the queue from disk.

        Returns:
            Number of messages loaded.
        """
        if not self._persist_path:
            return 0
        try:
            filepath = os.path.join(self._persist_path, "offline_queue.pkl")
            if not os.path.exists(filepath):
                return 0
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            count = 0
            for msg in data:
                if len(self._queue) < self._max_size:
                    self._queue.append(msg)
                    count += 1
            self._stats["loaded"] += count
            return count
        except Exception as e:
            logger.error("Failed to load offline queue: %s", e)
            return 0

    def clear(self) -> int:
        """Clear all queued messages.

        Returns:
            Number of messages cleared.
        """
        count = len(self._queue)
        self._queue.clear()
        return count

    @property
    def size(self) -> int:
        return len(self._queue)

    @property
    def is_empty(self) -> bool:
        return len(self._queue) == 0

    @property
    def stats(self) -> dict[str, int]:
        return {**self._stats, "current_size": self.size}


# ===================================================================
# Connection Manager
# ===================================================================

class ConnectionManager:
    """Manages MQTT connections with multi-broker failover.

    Features:
    - Multi-broker support (primary + backup)
    - Automatic reconnection with exponential backoff
    - Health monitoring (ping/pong, latency)
    - Bandwidth monitoring and throttling
    - Graceful shutdown (send offline status before disconnect)
    - Offline message queue with disk persistence

    Usage:
        manager = ConnectionManager(
            vessel_id="vessel-001",
            brokers=[BrokerConfig(host="primary.local"), BrokerConfig(host="backup.local")],
        )
        manager.connect()

        # Publish with automatic offline queuing
        manager.publish("nexus/vessel-001/telemetry", payload)

        # Graceful shutdown
        manager.shutdown()
    """

    def __init__(
        self,
        vessel_id: str,
        brokers: list[BrokerConfig] | None = None,
        client: MQTTClientInterface | None = None,
        reconnect_base_delay: float = 1.0,
        reconnect_max_delay: float = 60.0,
        max_reconnect_attempts: int = 0,  # 0 = infinite
        health_check_interval: float = 30.0,
        max_bandwidth_bps: int = 0,
        offline_persist_path: str | None = None,
    ) -> None:
        self.vessel_id = vessel_id
        self._brokers = sorted(
            brokers or [BrokerConfig()], key=lambda b: b.priority
        )
        self._reconnect_base_delay = reconnect_base_delay
        self._reconnect_max_delay = reconnect_max_delay
        self._max_reconnect_attempts = max_reconnect_attempts
        self._health_check_interval = health_check_interval

        # Use provided client or create mock
        self._client = client or MockMQTTClient(
            client_id=f"nexus-{vessel_id}"
        )
        self._hierarchy = TopicHierarchy(vessel_id)

        # State
        self._current_broker_idx = 0
        self._reconnect_attempt = 0
        self._reconnect_delay = reconnect_base_delay
        self._running = False
        self._shutdown_requested = False
        self._lock = threading.Lock()

        # Subsystems
        self._health = ConnectionHealth()
        self._bandwidth = BandwidthMonitor(max_bandwidth_bps=max_bandwidth_bps)
        self._offline_queue = OfflineQueue(persist_path=offline_persist_path)

        # Event log
        self._event_log: list[dict[str, Any]] = []

        # Wire up client callbacks
        self._client.set_on_connect(self._on_connect)
        self._client.set_on_disconnect(self._on_disconnect)

        self._logger = logging.getLogger(
            f"nexus.mqtt.connmgr.{vessel_id}"
        )

    @property
    def client(self) -> MQTTClientInterface:
        """The underlying MQTT client."""
        return self._client

    @property
    def health(self) -> ConnectionHealth:
        """Current connection health."""
        return self._health

    @property
    def bandwidth(self) -> BandwidthMonitor:
        """Bandwidth monitor."""
        return self._bandwidth

    @property
    def offline_queue(self) -> OfflineQueue:
        """Offline message queue."""
        return self._offline_queue

    @property
    def is_connected(self) -> bool:
        """Whether currently connected to a broker."""
        return self._client.is_connected()

    @property
    def current_broker(self) -> BrokerConfig | None:
        """Currently connected broker config."""
        if not self._brokers:
            return None
        return self._brokers[self._current_broker_idx]

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Connect to the best available broker.

        Tries brokers in priority order until one succeeds.

        Returns:
            True if connected successfully.
        """
        self._running = True
        self._shutdown_requested = False

        # Try loading offline queue
        self._offline_queue.load()

        for i, broker in enumerate(self._brokers):
            self._current_broker_idx = i
            self._log_event("connect_attempt", {
                "broker": broker.address, "priority": broker.priority,
            })

            result = self._client.connect(
                host=broker.host,
                port=broker.port,
                keepalive=broker.keepalive,
                clean_session=broker.clean_session,
                client_id=broker.client_id or f"nexus-{self.vessel_id}",
                username=broker.username,
                password=broker.password,
            )

            if result == 0:
                self._reconnect_attempt = 0
                self._reconnect_delay = self._reconnect_base_delay
                return True

            self._log_event("connect_failed", {
                "broker": broker.address, "error_code": result,
            })

        return False

    def disconnect(self) -> None:
        """Disconnect from the broker."""
        # Send offline status before disconnecting
        self._publish_offline_status()
        self._client.disconnect()

    def shutdown(self) -> None:
        """Graceful shutdown: persist queue, send offline, disconnect."""
        self._shutdown_requested = True
        self._running = False

        # Persist offline queue
        self._offline_queue.persist()

        # Send offline status
        self._publish_offline_status()

        # Disconnect
        self._client.disconnect()

        self._log_event("shutdown", {"vessel_id": self.vessel_id})
        self._logger.info("Connection manager shutdown complete")

    def reconnect(self) -> bool:
        """Attempt to reconnect to the current broker.

        If the current broker fails, tries the next one in the list.

        Returns:
            True if reconnected successfully.
        """
        self._health.reconnect_count += 1

        # Calculate backoff delay
        delay = min(
            self._reconnect_delay,
            self._reconnect_max_delay,
        )
        self._reconnect_delay *= 2  # Exponential backoff

        self._log_event("reconnect_wait", {"delay": delay})

        if self._max_reconnect_attempts > 0:
            if self._reconnect_attempt >= self._max_reconnect_attempts:
                self._logger.warning("Max reconnect attempts reached")
                return False
        self._reconnect_attempt += 1

        # Try current broker first, then failover
        tried = set()
        for i in range(len(self._brokers)):
            idx = (self._current_broker_idx + i) % len(self._brokers)
            if idx in tried:
                continue
            tried.add(idx)

            broker = self._brokers[idx]
            self._log_event("reconnect_attempt", {
                "broker": broker.address, "attempt": self._reconnect_attempt,
            })

            result = self._client.connect(
                host=broker.host,
                port=broker.port,
                keepalive=broker.keepalive,
                clean_session=broker.clean_session,
                client_id=broker.client_id or f"nexus-{self.vessel_id}",
                username=broker.username,
                password=broker.password,
            )

            if result == 0:
                self._current_broker_idx = idx
                self._reconnect_attempt = 0
                self._reconnect_delay = self._reconnect_base_delay
                self._log_event("reconnected", {"broker": broker.address})
                return True

        return False

    def switch_broker(self, broker_index: int) -> bool:
        """Switch to a specific broker.

        Args:
            broker_index: Index in the brokers list.

        Returns:
            True if switched and connected.
        """
        if broker_index < 0 or broker_index >= len(self._brokers):
            return False

        self._client.disconnect()
        self._current_broker_idx = broker_index
        return self.connect()

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(
        self,
        topic: str,
        payload: bytes | str,
        qos: int = 1,
        retain: bool = False,
    ) -> int:
        """Publish a message, queuing offline if disconnected.

        Args:
            topic: MQTT topic.
            payload: Message payload.
            qos: QoS level.
            retain: Retain flag.

        Returns:
            Message ID if published, -1 if queued or error.
        """
        if isinstance(payload, str):
            payload = payload.encode("utf-8")

        if not self._client.is_connected():
            # Queue for offline delivery
            self._offline_queue.enqueue(topic, payload, qos, retain)
            return -1

        result = self._client.publish(topic, payload, qos, retain)
        if result >= 0:
            self._health.messages_sent += 1
            self._health.bytes_sent += len(payload)
            self._bandwidth.record_send(len(payload))
            self._bandwidth.record_message_sent()
        return result

    def subscribe(self, topic: str, qos: int = 0) -> int:
        """Subscribe to a topic.

        Args:
            topic: Topic filter.
            qos: QoS level.

        Returns:
            Message ID.
        """
        return self._client.subscribe(topic, qos)

    # ------------------------------------------------------------------
    # Health monitoring
    # ------------------------------------------------------------------

    def check_health(self) -> ConnectionHealth:
        """Perform a health check and update metrics.

        Returns:
            Current ConnectionHealth snapshot.
        """
        if self._health.connected:
            self._health.uptime_seconds = time.time() - self._health.connect_time
        return self._health

    def measure_latency(self) -> float | None:
        """Measure round-trip latency to the broker.

        Returns:
            Latency in milliseconds, or None if not connected.
        """
        if not self._client.is_connected():
            return None

        # In a real implementation, this would send a ping and measure
        # the response time. For the mock, we simulate ~5ms.
        start = time.time()
        # Mock: no actual network call
        latency = (time.time() - start) * 1000 + 5.0  # simulate 5ms
        self._health.ping_latency_ms = latency
        self._health.record_latency(latency)
        self._health.last_ping_time = time.time()
        return latency

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------

    def _on_connect(self, success: bool, data: dict[str, Any]) -> None:
        """Handle connection event."""
        broker_id = data.get("host", "")
        self._health.connected = success
        self._health.broker_id = broker_id
        self._health.connect_time = time.time()
        self._health.consecutive_errors = 0

        if success:
            self._log_event("connected", {"broker": broker_id})

            # Publish online status
            self._publish_online_status()

            # Replay offline queue
            replayed = self._offline_queue.replay(self._client)
            if replayed > 0:
                self._log_event("offline_replay", {"count": replayed})
        else:
            self._health.consecutive_errors += 1

    def _on_disconnect(self, rc: int, data: dict[str, Any]) -> None:
        """Handle disconnection event."""
        self._health.connected = False
        reason = data.get("reason", "unknown")
        self._log_event("disconnected", {"rc": rc, "reason": reason})

        # Auto-reconnect if running and not shutdown
        if self._running and not self._shutdown_requested:
            self.reconnect()

    def _publish_online_status(self) -> None:
        """Publish vessel online status to the fleet."""
        import json
        topic = self._hierarchy.build(TopicPattern.FLEET_DISCOVERY)
        payload = json.dumps({
            "vessel_id": self.vessel_id,
            "status": "online",
            "timestamp": time.time(),
            "trust_scores": {},
        }, separators=(",", ":"))
        self._client.publish(topic, payload, qos=int(QoSLevel.AT_LEAST_ONCE))

    def _publish_offline_status(self) -> None:
        """Publish vessel offline status to the fleet."""
        if not self._client.is_connected():
            return
        import json
        topic = self._hierarchy.build(TopicPattern.FLEET_DISCOVERY)
        payload = json.dumps({
            "vessel_id": self.vessel_id,
            "status": "offline",
            "timestamp": time.time(),
        }, separators=(",", ":"))
        self._client.publish(topic, payload, qos=int(QoSLevel.AT_LEAST_ONCE))

    def _log_event(self, event_type: str, details: dict[str, Any] | None = None) -> None:
        """Log a connection event."""
        self._event_log.append({
            "event": event_type,
            "timestamp": time.time(),
            "details": details or {},
        })

    @property
    def event_log(self) -> list[dict[str, Any]]:
        """Connection event log."""
        return list(self._event_log)

    def get_broker_list(self) -> list[BrokerConfig]:
        """Get the list of configured brokers."""
        return list(self._brokers)
