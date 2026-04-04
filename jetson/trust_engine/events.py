"""NEXUS Trust Engine - Event classification.

15 event types with severity and quality values.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EventDefinition:
    """Definition of a trust event type."""

    name: str
    category: str  # "good", "bad", "neutral"
    severity: float  # 0.0 to 1.0
    quality: float  # 0.0 to 1.0 (for good events)


# Core event definitions (stub - full list in spec)
EVENT_DEFINITIONS: dict[str, EventDefinition] = {
    "reflex_completed": EventDefinition(
        name="reflex_completed",
        category="good",
        severity=0.0,
        quality=0.8,
    ),
    "reflex_error": EventDefinition(
        name="reflex_error",
        category="bad",
        severity=0.6,
        quality=0.0,
    ),
    "sensor_timeout": EventDefinition(
        name="sensor_timeout",
        category="bad",
        severity=0.4,
        quality=0.0,
    ),
    "actuator_overcurrent": EventDefinition(
        name="actuator_overcurrent",
        category="bad",
        severity=0.9,
        quality=0.0,
    ),
    "heartbeat_missed": EventDefinition(
        name="heartbeat_missed",
        category="bad",
        severity=0.3,
        quality=0.0,
    ),
}


def classify_event(event_type: str) -> EventDefinition | None:
    """Look up event definition by type name."""
    return EVENT_DEFINITIONS.get(event_type)
