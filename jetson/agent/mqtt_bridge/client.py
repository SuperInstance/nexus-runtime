"""NEXUS MQTT Client — Abstract interface and mock implementation.

Provides:
  - MQTTClientInterface: Abstract base class for MQTT operations
  - MockMQTTClient: In-memory mock for testing (no network required)
  - MQTTMessage: Immutable message representation
  - ConnectionEvent/ConnectionState: Connection lifecycle tracking
"""

from __future__ import annotations

import abc
import time
import threading
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable

from .topics import QoSLevel

logger = logging.getLogger("nexus.mqtt.client")


# ===================================================================
# Message
# ===================================================================

@dataclass(frozen=True)
class MQTTMessage:
    """Immutable MQTT message representation."""

    topic: str
    payload: bytes
    qos: int = 0
    retain: bool = False
    mid: int = 0  # Message ID (assigned by broker)
    timestamp: float = field(default_factory=time.time)

    @property
    def payload_str(self) -> str:
        """Decode payload as UTF-8 string."""
        return self.payload.decode("utf-8", errors="replace")

    @property
    def payload_size(self) -> int:
        """Size of the payload in bytes."""
        return len(self.payload)

    def with_topic(self, new_topic: str) -> MQTTMessage:
        """Create a copy of this message with a different topic."""
        return MQTTMessage(
            topic=new_topic,
            payload=self.payload,
            qos=self.qos,
            retain=self.retain,
            mid=self.mid,
            timestamp=self.timestamp,
        )


# ===================================================================
# Connection State
# ===================================================================

class ConnectionState(str, Enum):
    """MQTT client connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    DISCONNECTING = "disconnecting"


class ConnectionEvent(str, Enum):
    """MQTT connection lifecycle events."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    MESSAGE_ARRIVED = "message_arrived"
    MESSAGE_PUBLISHED = "message_published"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"
    ERROR = "error"


@dataclass
class ConnectionEventRecord:
    """Record of a connection event."""
    event: ConnectionEvent
    timestamp: float = field(default_factory=time.time)
    details: str = ""
    broker_id: str = ""


# ===================================================================
# Callback Types
# ===================================================================

OnConnectCallback = Callable[[bool, dict[str, Any]], None]
OnDisconnectCallback = Callable[[int, dict[str, Any]], None]
OnMessageCallback = Callable[["MQTTMessage"], None]
OnPublishCallback = Callable[[int, int], None]  # mid, result
OnSubscribeCallback = Callable[[int, int, list[tuple[int, int]]], None]
OnLogCallback = Callable[[int, str], None]


# ===================================================================
# Abstract MQTT Client Interface
# ===================================================================

class MQTTClientInterface(abc.ABC):
    """Abstract base class for MQTT client implementations.

    All MQTT operations go through this interface. Concrete implementations
    can be paho-mqtt, pure-socket, or the mock client for testing.
    """

    @abc.abstractmethod
    def connect(
        self,
        host: str,
        port: int = 1883,
        keepalive: int = 60,
        clean_session: bool = True,
        client_id: str = "",
        username: str = "",
        password: str = "",
        **kwargs: Any,
    ) -> int:
        """Connect to the MQTT broker.

        Args:
            host: Broker hostname or IP.
            port: Broker port (default 1883).
            keepalive: Keepalive interval in seconds.
            clean_session: Start a clean session.
            client_id: Client identifier.
            username: Optional username.
            password: Optional password.
            **kwargs: Additional broker-specific options.

        Returns:
            0 on success, non-zero on error.
        """
        ...

    @abc.abstractmethod
    def disconnect(self) -> int:
        """Disconnect from the broker gracefully.

        Returns:
            0 on success.
        """
        ...

    @abc.abstractmethod
    def publish(
        self,
        topic: str,
        payload: bytes | str = b"",
        qos: int = 0,
        retain: bool = False,
    ) -> int:
        """Publish a message to a topic.

        Args:
            topic: Topic string.
            payload: Message payload (bytes or string).
            qos: QoS level (0, 1, or 2).
            retain: Whether to set the retained flag.

        Returns:
            Message ID on success, -1 on error.
        """
        ...

    @abc.abstractmethod
    def subscribe(
        self,
        topic: str,
        qos: int = 0,
    ) -> int:
        """Subscribe to a topic.

        Args:
            topic: Topic filter string (may include wildcards).
            qos: Maximum QoS level for this subscription.

        Returns:
            Message ID of the subscribe packet.
        """
        ...

    @abc.abstractmethod
    def unsubscribe(self, topic: str) -> int:
        """Unsubscribe from a topic.

        Args:
            topic: Topic filter string.

        Returns:
            Message ID of the unsubscribe packet.
        """
        ...

    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Check if the client is currently connected."""
        ...

    @property
    @abc.abstractmethod
    def state(self) -> ConnectionState:
        """Current connection state."""
        ...

    @abc.abstractmethod
    def loop_start(self) -> None:
        """Start the network loop (background thread)."""
        ...

    @abc.abstractmethod
    def loop_stop(self) -> None:
        """Stop the network loop."""
        ...

    @abc.abstractmethod
    def message_queue(self) -> deque[MQTTMessage]:
        """Get accumulated messages since last check.

        Returns:
            Deque of received messages.
        """
        ...

    def set_on_connect(self, callback: OnConnectCallback | None) -> None:
        """Set the on_connect callback."""
        self._on_connect = callback

    def set_on_disconnect(self, callback: OnDisconnectCallback | None) -> None:
        """Set the on_disconnect callback."""
        self._on_disconnect = callback

    def set_on_message(self, callback: OnMessageCallback | None) -> None:
        """Set the on_message callback."""
        self._on_message = callback

    # Default callback storage
    _on_connect: OnConnectCallback | None = None
    _on_disconnect: OnDisconnectCallback | None = None
    _on_message: OnMessageCallback | None = None


# ===================================================================
# Mock MQTT Client — In-memory implementation for testing
# ===================================================================

class MockMQTTClient(MQTTClientInterface):
    """In-memory mock MQTT client for testing.

    Simulates a broker without network connections. Messages published
    to topics are immediately delivered to subscribers. Supports all
    QoS levels, retained messages, and connection lifecycle.

    Usage:
        client = MockMQTTClient()
        client.set_on_message(my_handler)
        client.connect("localhost")
        client.subscribe("nexus/vessel-001/telemetry")
        client.publish("nexus/vessel-001/telemetry", b'{"data": 1}')

        # Check received messages
        messages = client.message_queue()
    """

    def __init__(self, client_id: str = "mock-client") -> None:
        self.client_id = client_id
        self._state = ConnectionState.DISCONNECTED
        self._connected = False
        self._host = ""
        self._port = 1883
        self._subscriptions: dict[str, int] = {}  # topic_filter -> qos
        self._retained: dict[str, MQTTMessage] = {}  # topic -> last retained msg
        self._received: deque[MQTTMessage] = deque(maxlen=10000)
        self._published: deque[MQTTMessage] = deque(maxlen=10000)
        self._message_id_counter = 0
        self._event_log: list[ConnectionEventRecord] = []
        self._loop_thread: threading.Thread | None = None
        self._loop_running = False
        self._lock = threading.Lock()
        # Allow injecting failures
        self.fail_on_connect = False
        self.fail_on_publish = False

        # Callbacks
        self._on_connect: OnConnectCallback | None = None
        self._on_disconnect: OnDisconnectCallback | None = None
        self._on_message: OnMessageCallback | None = None

    def connect(
        self,
        host: str,
        port: int = 1883,
        keepalive: int = 60,
        clean_session: bool = True,
        client_id: str = "",
        username: str = "",
        password: str = "",
        **kwargs: Any,
    ) -> int:
        """Connect to the mock broker."""
        if self.fail_on_connect:
            self._state = ConnectionState.DISCONNECTED
            self._log_event(ConnectionEvent.ERROR, "Simulated connect failure")
            return 1

        self._state = ConnectionState.CONNECTING
        self._log_event(ConnectionEvent.CONNECTING, f"Connecting to {host}:{port}")
        self._host = host
        self._port = port
        if client_id:
            self.client_id = client_id

        # Simulate instant connection
        self._connected = True
        self._state = ConnectionState.CONNECTED
        self._log_event(ConnectionEvent.CONNECTED, f"Connected to {host}:{port}")

        # Re-deliver retained messages for existing subscriptions
        with self._lock:
            for topic_filter, qos in list(self._subscriptions.items()):
                for retained_topic, msg in self._retained.items():
                    if self._topic_matches(retained_topic, topic_filter):
                        self._deliver_message(msg)

        # Fire callback
        if self._on_connect:
            try:
                self._on_connect(True, {"host": host, "port": port})
            except Exception as e:
                logger.warning("on_connect callback error: %s", e)

        return 0

    def disconnect(self) -> int:
        """Disconnect from the mock broker."""
        if not self._connected:
            return 0

        self._state = ConnectionState.DISCONNECTING
        self._log_event(ConnectionEvent.DISCONNECTING)

        self._connected = False
        self._state = ConnectionState.DISCONNECTED
        self._log_event(ConnectionEvent.DISCONNECTED)

        if self._on_disconnect:
            try:
                self._on_disconnect(0, {"reason": "client_disconnect"})
            except Exception as e:
                logger.warning("on_disconnect callback error: %s", e)

        return 0

    def publish(
        self,
        topic: str,
        payload: bytes | str = b"",
        qos: int = 0,
        retain: bool = False,
    ) -> int:
        """Publish a message to a topic."""
        if self.fail_on_publish:
            self._log_event(ConnectionEvent.ERROR, "Simulated publish failure")
            return -1

        if not self._connected:
            self._log_event(ConnectionEvent.ERROR, "Publish while disconnected")
            return -1

        self._message_id_counter += 1
        mid = self._message_id_counter

        if isinstance(payload, str):
            payload = payload.encode("utf-8")

        msg = MQTTMessage(
            topic=topic,
            payload=payload,
            qos=qos,
            retain=retain,
            mid=mid,
        )

        # Store retained message
        if retain:
            self._retained[topic] = msg

        # Deliver to local subscribers
        self._deliver_to_subscribers(msg)

        # Record published
        with self._lock:
            self._published.append(msg)

        self._log_event(ConnectionEvent.MESSAGE_PUBLISHED,
                        f"topic={topic}, qos={qos}, size={len(payload)}")

        return mid

    def subscribe(self, topic: str, qos: int = 0) -> int:
        """Subscribe to a topic."""
        self._message_id_counter += 1
        mid = self._message_id_counter

        with self._lock:
            self._subscriptions[topic] = qos

        self._log_event(ConnectionEvent.SUBSCRIBED, f"topic={topic}, qos={qos}")

        # Deliver any retained messages matching this subscription
        if self._connected:
            for retained_topic, msg in self._retained.items():
                if self._topic_matches(retained_topic, topic):
                    self._deliver_message(msg)

        return mid

    def unsubscribe(self, topic: str) -> int:
        """Unsubscribe from a topic."""
        self._message_id_counter += 1
        mid = self._message_id_counter

        with self._lock:
            self._subscriptions.pop(topic, None)

        self._log_event(ConnectionEvent.UNSUBSCRIBED, f"topic={topic}")
        return mid

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    @property
    def state(self) -> ConnectionState:
        """Current connection state."""
        return self._state

    def loop_start(self) -> None:
        """Start the mock loop (no-op — messages delivered immediately)."""
        self._loop_running = True

    def loop_stop(self) -> None:
        """Stop the mock loop."""
        self._loop_running = False

    def message_queue(self) -> deque[MQTTMessage]:
        """Get all received messages since last check."""
        with self._lock:
            messages = deque(self._received)
            self._received.clear()
        return messages

    # ------------------------------------------------------------------
    # Mock-specific methods
    # ------------------------------------------------------------------

    def inject_message(self, topic: str, payload: bytes | str, qos: int = 0) -> None:
        """Inject a message as if received from the broker (for testing).

        Args:
            topic: Message topic.
            payload: Message payload.
            qos: QoS level.
        """
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        self._message_id_counter += 1
        msg = MQTTMessage(
            topic=topic,
            payload=payload,
            qos=qos,
            mid=self._message_id_counter,
        )
        self._deliver_message(msg)

    def get_published_messages(self) -> list[MQTTMessage]:
        """Get all published messages (for test assertions)."""
        with self._lock:
            return list(self._published)

    def get_subscriptions(self) -> dict[str, int]:
        """Get current subscriptions."""
        with self._lock:
            return dict(self._subscriptions)

    def get_event_log(self) -> list[ConnectionEventRecord]:
        """Get connection event log."""
        return list(self._event_log)

    def clear(self) -> None:
        """Clear all state (published, received, events)."""
        with self._lock:
            self._received.clear()
            self._published.clear()
            self._event_log.clear()
            self._subscriptions.clear()
            self._retained.clear()
            self._message_id_counter = 0

    @property
    def event_log(self) -> list[ConnectionEventRecord]:
        """Connection event log."""
        return list(self._event_log)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _deliver_to_subscribers(self, msg: MQTTMessage) -> None:
        """Deliver a published message to matching subscribers."""
        with self._lock:
            for topic_filter, qos in list(self._subscriptions.items()):
                if self._topic_matches(msg.topic, topic_filter):
                    self._deliver_message(msg)

    def _deliver_message(self, msg: MQTTMessage) -> None:
        """Deliver a message to the callback and queue."""
        with self._lock:
            self._received.append(msg)
        if self._on_message:
            try:
                self._on_message(msg)
            except Exception as e:
                logger.warning("on_message callback error: %s", e)

    @staticmethod
    def _topic_matches(topic: str, pattern: str) -> bool:
        """Check if a topic matches a subscription pattern.

        Supports MQTT wildcards:
          '+' matches exactly one level
          '#' matches any number of levels (must be last)
        """
        if pattern == "#" or pattern == topic:
            return True

        topic_parts = topic.split("/")
        pattern_parts = pattern.split("/")

        for i, pp in enumerate(pattern_parts):
            if pp == "#":
                return True
            if i >= len(topic_parts):
                return False
            if pp != "+" and pp != topic_parts[i]:
                return False

        return len(topic_parts) == len(pattern_parts)

    def _log_event(
        self, event: ConnectionEvent, details: str = "", broker_id: str = ""
    ) -> None:
        """Log a connection event."""
        self._event_log.append(ConnectionEventRecord(
            event=event,
            details=details,
            broker_id=broker_id or self._host,
        ))
