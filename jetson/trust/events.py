"""NEXUS Trust Engine - Event classification.

15 event types with severity and quality values.
Events are classified as good, bad, or neutral.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EventDefinition:
    """Definition of a trust event type."""

    name: str
    category: str   # "good", "bad", "neutral"
    severity: float # 0.0 to 1.0 (higher = more severe for bad events)
    quality: float  # 0.0 to 1.0 (higher = better for good events)

    @property
    def is_good(self) -> bool:
        return self.category == "good"

    @property
    def is_bad(self) -> bool:
        return self.category == "bad"

    @property
    def is_neutral(self) -> bool:
        return self.category == "neutral"


# ===================================================================
# Complete 15 event definitions
# ===================================================================

EVENT_DEFINITIONS: dict[str, EventDefinition] = {
    # ---- Good events (5) ----
    "heartbeat_ok": EventDefinition(
        name="heartbeat_ok",
        category="good",
        severity=0.0,
        quality=0.7,
    ),
    "sensor_valid": EventDefinition(
        name="sensor_valid",
        category="good",
        severity=0.0,
        quality=0.8,
    ),
    "reflex_completed": EventDefinition(
        name="reflex_completed",
        category="good",
        severity=0.0,
        quality=0.8,
    ),
    "actuator_nominal": EventDefinition(
        name="actuator_nominal",
        category="good",
        severity=0.0,
        quality=0.9,
    ),
    "command_ack": EventDefinition(
        name="command_ack",
        category="good",
        severity=0.0,
        quality=0.6,
    ),

    # ---- Bad events (7) ----
    "heartbeat_missed": EventDefinition(
        name="heartbeat_missed",
        category="bad",
        severity=0.3,
        quality=0.0,
    ),
    "sensor_invalid": EventDefinition(
        name="sensor_invalid",
        category="bad",
        severity=0.5,
        quality=0.0,
    ),
    "reflex_error": EventDefinition(
        name="reflex_error",
        category="bad",
        severity=0.6,
        quality=0.0,
    ),
    "actuator_overrange": EventDefinition(
        name="actuator_overrange",
        category="bad",
        severity=0.7,
        quality=0.0,
    ),
    "trust_violation": EventDefinition(
        name="trust_violation",
        category="bad",
        severity=0.9,
        quality=0.0,
    ),
    "safety_trigger": EventDefinition(
        name="safety_trigger",
        category="bad",
        severity=0.8,
        quality=0.0,
    ),
    "communication_timeout": EventDefinition(
        name="communication_timeout",
        category="bad",
        severity=0.4,
        quality=0.0,
    ),

    # ---- Neutral events (3) ----
    "system_boot": EventDefinition(
        name="system_boot",
        category="neutral",
        severity=0.0,
        quality=0.0,
    ),
    "parameter_change": EventDefinition(
        name="parameter_change",
        category="neutral",
        severity=0.0,
        quality=0.0,
    ),
    "calibration_complete": EventDefinition(
        name="calibration_complete",
        category="neutral",
        severity=0.0,
        quality=0.0,
    ),
}


def classify_event(event_type: str) -> EventDefinition | None:
    """Look up event definition by type name."""
    return EVENT_DEFINITIONS.get(event_type)


def get_good_events() -> list[str]:
    """Return list of all good event type names."""
    return [name for name, defn in EVENT_DEFINITIONS.items() if defn.is_good]


def get_bad_events() -> list[str]:
    """Return list of all bad event type names."""
    return [name for name, defn in EVENT_DEFINITIONS.items() if defn.is_bad]


def get_neutral_events() -> list[str]:
    """Return list of all neutral event type names."""
    return [name for name, defn in EVENT_DEFINITIONS.items() if defn.is_neutral]
