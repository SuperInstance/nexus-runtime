"""NEXUS MQTT Topic Hierarchy — 13+ topics as specified in Edge-Native spec.

Topic structure:
  nexus/{vessel_id}/telemetry     — Sensor data and observations       QoS 0
  nexus/{vessel_id}/status        — Vessel status (health, trust)      QoS 1
  nexus/{vessel_id}/reflex/deploy — Bytecode deployment commands       QoS 2
  nexus/{vessel_id}/reflex/result — Reflex execution results           QoS 1
  nexus/{vessel_id}/trust/events  — Trust score changes                QoS 1
  nexus/{vessel_id}/safety/alert  — Safety violations & emergencies    QoS 2
  nexus/{vessel_id}/command       — Command messages (waypoints, etc)  QoS 1
  nexus/{vessel_id}/response      — Command acknowledgments            QoS 1
  nexus/{vessel_id}/position      — GPS/DR position updates (high Hz)  QoS 0
  nexus/{vessel_id}/intention     — Dead reckoning intention broadcasts QoS 0
  nexus/fleet/coordination        — Fleet-wide coordination messages   QoS 1
  nexus/fleet/discovery           — Vessel discovery and heartbeat     QoS 1
  nexus/fleet/sync                — CRDT state synchronization         QoS 2
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any


# ===================================================================
# QoS Levels
# ===================================================================

class QoSLevel(IntEnum):
    """MQTT Quality of Service levels."""
    AT_MOST_ONCE = 0   # Fire and forget — telemetry, position
    AT_LEAST_ONCE = 1  # Acknowledged delivery — commands, status
    EXACTLY_ONCE = 2   # Guaranteed delivery — safety, deployment


# ===================================================================
# Topic Patterns
# ===================================================================

class TopicPattern(str, Enum):
    """All 13 topic patterns in the NEXUS hierarchy."""

    TELEMETRY = "nexus/{vessel_id}/telemetry"
    STATUS = "nexus/{vessel_id}/status"
    REFLEX_DEPLOY = "nexus/{vessel_id}/reflex/deploy"
    REFLEX_RESULT = "nexus/{vessel_id}/reflex/result"
    TRUST_EVENTS = "nexus/{vessel_id}/trust/events"
    SAFETY_ALERT = "nexus/{vessel_id}/safety/alert"
    COMMAND = "nexus/{vessel_id}/command"
    RESPONSE = "nexus/{vessel_id}/response"
    POSITION = "nexus/{vessel_id}/position"
    INTENTION = "nexus/{vessel_id}/intention"
    FLEET_COORDINATION = "nexus/fleet/coordination"
    FLEET_DISCOVERY = "nexus/fleet/discovery"
    FLEET_SYNC = "nexus/fleet/sync"


# ===================================================================
# Topic Definition
# ===================================================================

@dataclass(frozen=True)
class TopicDefinition:
    """Complete definition of an MQTT topic."""

    pattern: TopicPattern
    name: str
    description: str
    qos: QoSLevel
    retained: bool = False
    direction: str = "bidirectional"  # "outbound", "inbound", "bidirectional"
    category: str = "vessel"  # "vessel" or "fleet"


TOPIC_DEFINITIONS: dict[TopicPattern, TopicDefinition] = {
    TopicPattern.TELEMETRY: TopicDefinition(
        pattern=TopicPattern.TELEMETRY,
        name="telemetry",
        description="Sensor data and observations",
        qos=QoSLevel.AT_MOST_ONCE,
        direction="outbound",
    ),
    TopicPattern.STATUS: TopicDefinition(
        pattern=TopicPattern.STATUS,
        name="status",
        description="Vessel status (health, trust, safety state)",
        qos=QoSLevel.AT_LEAST_ONCE,
        direction="bidirectional",
    ),
    TopicPattern.REFLEX_DEPLOY: TopicDefinition(
        pattern=TopicPattern.REFLEX_DEPLOY,
        name="reflex/deploy",
        description="Bytecode deployment commands",
        qos=QoSLevel.EXACTLY_ONCE,
        direction="inbound",
    ),
    TopicPattern.REFLEX_RESULT: TopicDefinition(
        pattern=TopicPattern.REFLEX_RESULT,
        name="reflex/result",
        description="Reflex execution results",
        qos=QoSLevel.AT_LEAST_ONCE,
        direction="outbound",
    ),
    TopicPattern.TRUST_EVENTS: TopicDefinition(
        pattern=TopicPattern.TRUST_EVENTS,
        name="trust/events",
        description="Trust score changes",
        qos=QoSLevel.AT_LEAST_ONCE,
        direction="bidirectional",
    ),
    TopicPattern.SAFETY_ALERT: TopicDefinition(
        pattern=TopicPattern.SAFETY_ALERT,
        name="safety/alert",
        description="Safety violations and emergencies",
        qos=QoSLevel.EXACTLY_ONCE,
        direction="outbound",
    ),
    TopicPattern.COMMAND: TopicDefinition(
        pattern=TopicPattern.COMMAND,
        name="command",
        description="Command messages (waypoints, missions)",
        qos=QoSLevel.AT_LEAST_ONCE,
        direction="inbound",
    ),
    TopicPattern.RESPONSE: TopicDefinition(
        pattern=TopicPattern.RESPONSE,
        name="response",
        description="Command acknowledgments",
        qos=QoSLevel.AT_LEAST_ONCE,
        direction="outbound",
    ),
    TopicPattern.POSITION: TopicDefinition(
        pattern=TopicPattern.POSITION,
        name="position",
        description="GPS/DR position updates (high frequency)",
        qos=QoSLevel.AT_MOST_ONCE,
        direction="outbound",
    ),
    TopicPattern.INTENTION: TopicDefinition(
        pattern=TopicPattern.INTENTION,
        name="intention",
        description="Dead reckoning intention broadcasts",
        qos=QoSLevel.AT_MOST_ONCE,
        direction="outbound",
    ),
    TopicPattern.FLEET_COORDINATION: TopicDefinition(
        pattern=TopicPattern.FLEET_COORDINATION,
        name="fleet/coordination",
        description="Fleet-wide coordination messages",
        qos=QoSLevel.AT_LEAST_ONCE,
        direction="bidirectional",
        category="fleet",
    ),
    TopicPattern.FLEET_DISCOVERY: TopicDefinition(
        pattern=TopicPattern.FLEET_DISCOVERY,
        name="fleet/discovery",
        description="Vessel discovery and heartbeat",
        qos=QoSLevel.AT_LEAST_ONCE,
        direction="bidirectional",
        category="fleet",
    ),
    TopicPattern.FLEET_SYNC: TopicDefinition(
        pattern=TopicPattern.FLEET_SYNC,
        name="fleet/sync",
        description="CRDT state synchronization",
        qos=QoSLevel.EXACTLY_ONCE,
        direction="bidirectional",
        category="fleet",
    ),
}


# ===================================================================
# Topic Hierarchy — Build and Parse
# ===================================================================

class TopicHierarchy:
    """Manages the NEXUS MQTT topic hierarchy.

    Provides methods to build topics for specific vessels and fleets,
    parse incoming topics to determine their type, and enumerate
    all relevant subscriptions.

    Usage:
        hierarchy = TopicHierarchy(vessel_id="vessel-001")
        topic = hierarchy.build(TopicPattern.TELEMETRY)
        # "nexus/vessel-001/telemetry"

        result = TopicHierarchy.parse("nexus/vessel-001/safety/alert")
        # TopicParseResult(pattern=TopicPattern.SAFETY_ALERT, vessel_id="vessel-001", ...)
    """

    TOPIC_PREFIX = "nexus"

    def __init__(self, vessel_id: str) -> None:
        self.vessel_id = vessel_id

    def build(self, pattern: TopicPattern) -> str:
        """Build a fully-qualified topic string.

        Args:
            pattern: The topic pattern to build.

        Returns:
            Fully-qualified MQTT topic string.
        """
        template = pattern.value
        if "{vessel_id}" in template:
            return template.replace("{vessel_id}", self.vessel_id)
        return template

    def build_fleet_topic(self, pattern: TopicPattern) -> str:
        """Build a fleet-level topic.

        Args:
            pattern: The fleet topic pattern.

        Returns:
            Fully-qualified MQTT topic string.
        """
        template = pattern.value
        if "{vessel_id}" in template:
            raise ValueError(
                f"Topic pattern {pattern.name} requires a vessel_id, "
                f"use build() instead"
            )
        return template

    def get_vessel_subscriptions(self) -> list[tuple[str, QoSLevel]]:
        """Get all MQTT subscriptions for this vessel.

        Returns:
            List of (topic_filter, qos) tuples for subscribing.
        """
        subs: list[tuple[str, QoSLevel]] = []
        for pattern, defn in TOPIC_DEFINITIONS.items():
            if defn.category == "vessel":
                if defn.direction in ("inbound", "bidirectional"):
                    # Subscribe to vessel-specific topics
                    topic = self.build(pattern)
                    subs.append((topic, defn.qos))
            elif defn.category == "fleet":
                if defn.direction in ("inbound", "bidirectional"):
                    # Subscribe to fleet topics
                    subs.append((pattern.value, defn.qos))
        return subs

    def get_publish_topics(self) -> list[tuple[str, QoSLevel]]:
        """Get all topics this vessel can publish to.

        Returns:
            List of (topic, qos) tuples for publishing.
        """
        pubs: list[tuple[str, QoSLevel]] = []
        for pattern, defn in TOPIC_DEFINITIONS.items():
            if defn.category == "vessel":
                if defn.direction in ("outbound", "bidirectional"):
                    topic = self.build(pattern)
                    pubs.append((topic, defn.qos))
            elif defn.category == "fleet":
                if defn.direction in ("outbound", "bidirectional"):
                    pubs.append((pattern.value, defn.qos))
        return pubs

    @staticmethod
    def parse(topic: str) -> "TopicParseResult | None":
        """Parse a topic string into its components.

        Args:
            topic: MQTT topic string to parse.

        Returns:
            TopicParseResult with parsed components, or None if not a valid NEXUS topic.
        """
        parts = topic.split("/")
        if len(parts) < 3 or parts[0] != TopicHierarchy.TOPIC_PREFIX:
            return None

        # Fleet topics: nexus/fleet/<name>
        if parts[1] == "fleet":
            topic_suffix = "/".join(parts[2:])
            for pattern, defn in TOPIC_DEFINITIONS.items():
                if defn.category == "fleet" and pattern.value.endswith(topic_suffix):
                    return TopicParseResult(
                        pattern=pattern,
                        definition=defn,
                        vessel_id=None,
                        fleet_topic=True,
                        raw=topic,
                    )
            # Unknown fleet topic
            return TopicParseResult(
                pattern=None,
                definition=None,
                vessel_id=None,
                fleet_topic=True,
                raw=topic,
                unknown=True,
            )

        # Vessel topics: nexus/<vessel_id>/<name>
        vessel_id = parts[1]
        topic_suffix = "/".join(parts[2:])

        for pattern, defn in TOPIC_DEFINITIONS.items():
            if defn.category == "vessel":
                prefix = "nexus/{vessel_id}/"
                expected_suffix = pattern.value[len(prefix):] if pattern.value.startswith(prefix) else ""
                if topic_suffix == expected_suffix:
                    return TopicParseResult(
                        pattern=pattern,
                        definition=defn,
                        vessel_id=vessel_id,
                        fleet_topic=False,
                        raw=topic,
                    )

        # Unknown vessel topic
        return TopicParseResult(
            pattern=None,
            definition=None,
            vessel_id=vessel_id,
            fleet_topic=False,
            raw=topic,
            unknown=True,
        )

    @staticmethod
    def get_all_patterns() -> list[TopicPattern]:
        """Return all defined topic patterns."""
        return list(TopicPattern)

    @staticmethod
    def get_vessel_patterns() -> list[TopicPattern]:
        """Return vessel-scoped topic patterns only."""
        return [
            p for p, d in TOPIC_DEFINITIONS.items()
            if d.category == "vessel"
        ]

    @staticmethod
    def get_fleet_patterns() -> list[TopicPattern]:
        """Return fleet-scoped topic patterns only."""
        return [
            p for p, d in TOPIC_DEFINITIONS.items()
            if d.category == "fleet"
        ]


# ===================================================================
# Topic Parse Result
# ===================================================================

@dataclass
class TopicParseResult:
    """Result of parsing an MQTT topic string."""

    pattern: TopicPattern | None
    definition: TopicDefinition | None
    vessel_id: str | None
    fleet_topic: bool
    raw: str
    unknown: bool = False

    @property
    def is_valid(self) -> bool:
        """True if this is a known, valid NEXUS topic."""
        return self.pattern is not None and not self.unknown

    @property
    def qos(self) -> QoSLevel:
        """QoS level for this topic (default 0 for unknown)."""
        if self.definition:
            return self.definition.qos
        return QoSLevel.AT_MOST_ONCE

    @property
    def category(self) -> str:
        """Topic category: 'vessel' or 'fleet'."""
        if self.definition:
            return self.definition.category
        return "fleet" if self.fleet_topic else "vessel"


# ===================================================================
# Convenience Functions
# ===================================================================

def build_topic(pattern: TopicPattern, vessel_id: str = "") -> str:
    """Build a topic string from a pattern and optional vessel_id.

    Args:
        pattern: Topic pattern.
        vessel_id: Vessel ID for vessel-scoped topics.

    Returns:
        Fully-qualified MQTT topic string.
    """
    if "{vessel_id}" in pattern.value:
        if not vessel_id:
            raise ValueError(
                f"Topic pattern {pattern.name} requires a vessel_id"
            )
        return pattern.value.replace("{vessel_id}", vessel_id)
    return pattern.value


def parse_topic(topic: str) -> TopicParseResult | None:
    """Parse a topic string into its components.

    Args:
        topic: MQTT topic string.

    Returns:
        TopicParseResult or None if not a NEXUS topic.
    """
    return TopicHierarchy.parse(topic)


def vessel_topics(vessel_id: str) -> dict[TopicPattern, str]:
    """Build all vessel-scoped topic strings for a given vessel.

    Args:
        vessel_id: Vessel identifier.

    Returns:
        Dict mapping TopicPattern to fully-qualified topic strings.
    """
    result = {}
    for pattern, defn in TOPIC_DEFINITIONS.items():
        if defn.category == "vessel":
            result[pattern] = pattern.value.replace("{vessel_id}", vessel_id)
    return result


def fleet_topics() -> dict[TopicPattern, str]:
    """Get all fleet-scoped topic strings.

    Returns:
        Dict mapping TopicPattern to fully-qualified topic strings.
    """
    result = {}
    for pattern, defn in TOPIC_DEFINITIONS.items():
        if defn.category == "fleet":
            result[pattern] = pattern.value
    return result


def all_vessel_subscriptions(vessel_id: str) -> list[tuple[str, QoSLevel]]:
    """Get all subscription topics for a vessel.

    Args:
        vessel_id: Vessel identifier.

    Returns:
        List of (topic, qos) tuples.
    """
    hierarchy = TopicHierarchy(vessel_id)
    return hierarchy.get_vessel_subscriptions()
