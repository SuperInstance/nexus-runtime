"""NEXUS MQTT Message Router — Inbound message dispatch.

Routes incoming MQTT messages to appropriate handlers based on:
  - Topic pattern matching
  - Message priority
  - Trust level filtering
  - Vessel ownership verification

Features:
  - Register handlers for specific topic patterns
  - Wildcard pattern support (+, #)
  - Dead letter queue for unrouteable messages
  - Priority-based message ordering
  - Handler chain (multiple handlers per topic)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable

from .client import MQTTMessage
from .topics import (
    TopicHierarchy,
    TopicPattern,
    TopicParseResult,
    QoSLevel,
    parse_topic,
)

logger = logging.getLogger("nexus.mqtt.router")


# ===================================================================
# Message Priority
# ===================================================================

class MessagePriority(IntEnum):
    """Message priority levels for routing decisions."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

    @staticmethod
    def from_qos(qos: int) -> MessagePriority:
        """Derive priority from QoS level."""
        mapping = {0: MessagePriority.LOW, 1: MessagePriority.NORMAL, 2: MessagePriority.CRITICAL}
        return mapping.get(qos, MessagePriority.NORMAL)


# ===================================================================
# Handler Registration
# ===================================================================

MessageHandler = Callable[[MQTTMessage, TopicParseResult], None]


@dataclass
class HandlerRegistration:
    """Registration of a message handler."""

    handler: MessageHandler
    topic_filter: str
    name: str
    priority: MessagePriority = MessagePriority.NORMAL
    min_trust_level: int = 0  # Minimum trust level to accept
    vessel_id: str | None = None  # Only accept messages for this vessel
    enabled: bool = True
    message_count: int = 0
    error_count: int = 0
    last_message_time: float = 0.0

    def reset_stats(self) -> None:
        """Reset handler statistics."""
        self.message_count = 0
        self.error_count = 0
        self.last_message_time = 0.0


# ===================================================================
# Route Result
# ===================================================================

@dataclass
class RouteResult:
    """Result of routing a single message."""

    message_id: str
    topic: str
    handler_name: str | None = None
    success: bool = True
    reason: str = ""
    priority: MessagePriority = MessagePriority.NORMAL
    latency_ms: float = 0.0
    error: str = ""

    @property
    def was_routed(self) -> bool:
        return self.handler_name is not None and self.success


# ===================================================================
# Dead Letter Queue
# ===================================================================

@dataclass
class DeadLetterEntry:
    """An entry in the dead letter queue."""

    message: MQTTMessage
    reason: str
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    original_topic: str = ""
    parse_result: TopicParseResult | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for inspection."""
        return {
            "reason": self.reason,
            "timestamp": self.timestamp,
            "retry_count": self.retry_count,
            "topic": self.original_topic or self.message.topic,
            "qos": self.message.qos,
            "payload_size": self.message.payload_size,
            "payload_preview": self.message.payload_str[:200],
        }


class DeadLetterQueue:
    """Queue for messages that could not be routed.

    Provides inspection, retry, and cleanup capabilities.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._queue: deque[DeadLetterEntry] = deque(maxlen=max_size)
        self._stats = {
            "total_enqueued": 0,
            "total_retried": 0,
            "total_expired": 0,
            "total_purged": 0,
        }

    def enqueue(
        self,
        message: MQTTMessage,
        reason: str,
        parse_result: TopicParseResult | None = None,
    ) -> None:
        """Add a message to the dead letter queue.

        Args:
            message: The unrouteable message.
            reason: Why it couldn't be routed.
            parse_result: Parsed topic result (if available).
        """
        entry = DeadLetterEntry(
            message=message,
            reason=reason,
            original_topic=message.topic,
            parse_result=parse_result,
        )
        self._queue.append(entry)
        self._stats["total_enqueued"] += 1
        logger.debug(
            "DLQ enqueued: topic=%s, reason=%s",
            message.topic, reason,
        )

    def dequeue(self) -> DeadLetterEntry | None:
        """Remove and return the oldest entry."""
        if self._queue:
            return self._queue.popleft()
        return None

    def peek(self) -> DeadLetterEntry | None:
        """Look at the oldest entry without removing it."""
        if self._queue:
            return self._queue[0]
        return None

    def retry_all(self) -> list[DeadLetterEntry]:
        """Remove all entries for retry (caller should re-route them).

        Returns:
            List of all entries, oldest first.
        """
        entries = list(self._queue)
        count = len(entries)
        self._queue.clear()
        self._stats["total_retried"] += count
        return entries

    def purge(self) -> int:
        """Remove all entries from the queue.

        Returns:
            Number of entries purged.
        """
        count = len(self._queue)
        self._queue.clear()
        self._stats["total_purged"] += count
        return count

    def purge_expired(self, max_age_seconds: float = 3600.0) -> int:
        """Remove entries older than max_age_seconds.

        Returns:
            Number of entries expired and removed.
        """
        now = time.time()
        expired = 0
        new_queue: deque[DeadLetterEntry] = deque(maxlen=self._queue.maxlen)
        for entry in self._queue:
            if now - entry.timestamp > max_age_seconds:
                expired += 1
            else:
                new_queue.append(entry)
        self._queue = new_queue
        self._stats["total_expired"] += expired
        return expired

    @property
    def size(self) -> int:
        """Current queue size."""
        return len(self._queue)

    @property
    def is_empty(self) -> bool:
        """True if the queue is empty."""
        return len(self._queue) == 0

    @property
    def stats(self) -> dict[str, int]:
        """Dead letter queue statistics."""
        return {**self._stats, "current_size": self.size}

    def get_entries(self) -> list[DeadLetterEntry]:
        """Get all entries without removing them."""
        return list(self._queue)

    def get_entry_dicts(self) -> list[dict[str, Any]]:
        """Get all entries as dictionaries for inspection."""
        return [e.to_dict() for e in self._queue]


# ===================================================================
# Message Router
# ===================================================================

class MessageRouter:
    """Routes incoming MQTT messages to registered handlers.

    Features:
    - Pattern-based handler matching (supports MQTT wildcards)
    - Multiple handlers per topic (handler chain)
    - Priority-based ordering
    - Trust level filtering
    - Dead letter queue for unhandled messages
    - Statistics tracking

    Usage:
        router = MessageRouter(vessel_id="vessel-001")

        # Register handlers
        router.register("nexus/vessel-001/command", handle_command, "cmd_handler")
        router.register("nexus/vessel-001/reflex/+", handle_reflex, "reflex_handler")
        router.register("nexus/fleet/#", handle_fleet, "fleet_handler")

        # Route messages
        result = router.route(message)
        if not result.was_routed:
            # Message went to dead letter queue
            pass
    """

    def __init__(
        self,
        vessel_id: str = "",
        enable_dlq: bool = True,
    ) -> None:
        self.vessel_id = vessel_id
        self._handlers: list[HandlerRegistration] = []
        self._dlq = DeadLetterQueue()
        self._enable_dlq = enable_dlq
        self._stats = {
            "total_routed": 0,
            "total_dlq": 0,
            "total_dropped": 0,
            "total_filtered_trust": 0,
            "total_filtered_vessel": 0,
        }
        self._logger = logging.getLogger(
            f"nexus.mqtt.router.{vessel_id}"
        )

    def register(
        self,
        topic_filter: str,
        handler: MessageHandler,
        name: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        min_trust_level: int = 0,
        vessel_id: str | None = None,
    ) -> HandlerRegistration:
        """Register a message handler.

        Args:
            topic_filter: MQTT topic filter (supports + and # wildcards).
            handler: Callable that takes (MQTTMessage, TopicParseResult).
            name: Handler name for identification.
            priority: Message priority level.
            min_trust_level: Minimum trust level to accept messages.
            vessel_id: Only accept messages for this vessel (None = any).

        Returns:
            The HandlerRegistration for later management.
        """
        reg = HandlerRegistration(
            handler=handler,
            topic_filter=topic_filter,
            name=name,
            priority=priority,
            min_trust_level=min_trust_level,
            vessel_id=vessel_id or self.vessel_id,
        )
        self._handlers.append(reg)

        # Keep sorted by priority (highest first)
        self._handlers.sort(key=lambda h: h.priority, reverse=True)

        self._logger.debug(
            "Handler registered: name=%s, topic=%s, priority=%s",
            name, topic_filter, priority.name,
        )
        return reg

    def unregister(self, name: str) -> bool:
        """Unregister a handler by name.

        Args:
            name: Handler name.

        Returns:
            True if handler was found and removed.
        """
        for i, reg in enumerate(self._handlers):
            if reg.name == name:
                del self._handlers[i]
                return True
        return False

    def route(self, message: MQTTMessage) -> RouteResult:
        """Route a single message to matching handlers.

        Args:
            message: The MQTT message to route.

        Returns:
            RouteResult with routing details.
        """
        msg_id = uuid.uuid4().hex[:8]
        start = time.time()
        parse_result = parse_topic(message.topic)

        # Filter: unknown topics go to DLQ
        if parse_result is None:
            self._handle_unrouteable(
                message, "unknown_topic_prefix", parse_result
            )
            return RouteResult(
                message_id=msg_id,
                topic=message.topic,
                success=False,
                reason="unknown_topic_prefix",
                priority=MessagePriority.from_qos(message.qos),
                latency_ms=(time.time() - start) * 1000,
            )

        # Find matching handlers
        matched = self._find_handlers(message, parse_result)

        if not matched:
            self._handle_unrouteable(
                message, "no_matching_handler", parse_result
            )
            return RouteResult(
                message_id=msg_id,
                topic=message.topic,
                success=False,
                reason="no_matching_handler",
                priority=MessagePriority.from_qos(message.qos),
                latency_ms=(time.time() - start) * 1000,
            )

        # Deliver to all matched handlers
        last_result: RouteResult | None = None
        for reg in matched:
            try:
                reg.handler(message, parse_result)
                reg.message_count += 1
                reg.last_message_time = time.time()
                last_result = RouteResult(
                    message_id=msg_id,
                    topic=message.topic,
                    handler_name=reg.name,
                    success=True,
                    priority=reg.priority,
                    latency_ms=(time.time() - start) * 1000,
                )
            except Exception as e:
                reg.error_count += 1
                self._logger.error(
                    "Handler %s error: %s", reg.name, e
                )
                last_result = RouteResult(
                    message_id=msg_id,
                    topic=message.topic,
                    handler_name=reg.name,
                    success=False,
                    reason=f"handler_error: {e}",
                    priority=reg.priority,
                    latency_ms=(time.time() - start) * 1000,
                    error=str(e),
                )

        self._stats["total_routed"] += 1

        return last_result or RouteResult(
            message_id=msg_id,
            topic=message.topic,
            success=True,
            latency_ms=(time.time() - start) * 1000,
        )

    def route_many(self, messages: list[MQTTMessage]) -> list[RouteResult]:
        """Route multiple messages.

        Args:
            messages: List of MQTT messages.

        Returns:
            List of RouteResults, one per message.
        """
        return [self.route(msg) for msg in messages]

    # ------------------------------------------------------------------
    # Handler lookup
    # ------------------------------------------------------------------

    def _find_handlers(
        self,
        message: MQTTMessage,
        parse_result: TopicParseResult,
    ) -> list[HandlerRegistration]:
        """Find all matching handlers for a message."""
        matched: list[HandlerRegistration] = []

        for reg in self._handlers:
            if not reg.enabled:
                continue

            # Check vessel filter
            if reg.vessel_id and parse_result.vessel_id:
                if reg.vessel_id != parse_result.vessel_id:
                    self._stats["total_filtered_vessel"] += 1
                    continue

            # Check trust level (from message payload if available)
            if reg.min_trust_level > 0:
                try:
                    payload = json.loads(message.payload)
                    trust = payload.get("trust_level", 0)
                    if trust < reg.min_trust_level:
                        self._stats["total_filtered_trust"] += 1
                        continue
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

            # Check topic match
            if self._topic_matches(message.topic, reg.topic_filter):
                matched.append(reg)

        return matched

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

    # ------------------------------------------------------------------
    # Unrouteable messages
    # ------------------------------------------------------------------

    def _handle_unrouteable(
        self,
        message: MQTTMessage,
        reason: str,
        parse_result: TopicParseResult | None,
    ) -> None:
        """Handle a message that couldn't be routed."""
        if self._enable_dlq:
            self._dlq.enqueue(message, reason, parse_result)
            self._stats["total_dlq"] += 1
        else:
            self._stats["total_dropped"] += 1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dlq(self) -> DeadLetterQueue:
        """Access the dead letter queue."""
        return self._dlq

    @property
    def handler_count(self) -> int:
        """Number of registered handlers."""
        return len(self._handlers)

    @property
    def handlers(self) -> list[HandlerRegistration]:
        """List of all registered handlers."""
        return list(self._handlers)

    @property
    def stats(self) -> dict[str, int]:
        """Router statistics."""
        return dict(self._stats)

    def get_handler(self, name: str) -> HandlerRegistration | None:
        """Get a handler registration by name."""
        for reg in self._handlers:
            if reg.name == name:
                return reg
        return None

    def enable_handler(self, name: str) -> bool:
        """Enable a handler by name."""
        reg = self.get_handler(name)
        if reg:
            reg.enabled = True
            return True
        return False

    def disable_handler(self, name: str) -> bool:
        """Disable a handler by name."""
        reg = self.get_handler(name)
        if reg:
            reg.enabled = False
            return True
        return False

    def reset_stats(self) -> None:
        """Reset all statistics."""
        for key in self._stats:
            self._stats[key] = 0
        for reg in self._handlers:
            reg.reset_stats()
