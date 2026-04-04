"""NEXUS Trust Engine - Autonomy level definitions.

6 autonomy levels (L0-L5) with promotion/demotion rules.
Demotion is immediate. Promotion requires sustained safe operation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AutonomyLevel:
    """Definition of an autonomy level."""

    level: int
    name: str
    trust_threshold: float
    min_observation_hours: float
    min_clean_windows: int
    key_criteria: str


# Autonomy level definitions from spec
AUTONOMY_LEVELS: dict[int, AutonomyLevel] = {
    0: AutonomyLevel(
        level=0,
        name="Disabled",
        trust_threshold=0.0,
        min_observation_hours=0.0,
        min_clean_windows=0,
        key_criteria="Default after full reset",
    ),
    1: AutonomyLevel(
        level=1,
        name="Advisory",
        trust_threshold=0.20,
        min_observation_hours=8.0,
        min_clean_windows=4,
        key_criteria="80% clean windows",
    ),
    2: AutonomyLevel(
        level=2,
        name="Supervised",
        trust_threshold=0.40,
        min_observation_hours=48.0,
        min_clean_windows=24,
        key_criteria="No severity >= 0.8 events",
    ),
    3: AutonomyLevel(
        level=3,
        name="Semi-Autonomous",
        trust_threshold=0.60,
        min_observation_hours=168.0,
        min_clean_windows=100,
        key_criteria="No severity >= 0.7 in 48h",
    ),
    4: AutonomyLevel(
        level=4,
        name="High Autonomy",
        trust_threshold=0.80,
        min_observation_hours=336.0,
        min_clean_windows=200,
        key_criteria="No severity >= 0.6 in 72h",
    ),
    5: AutonomyLevel(
        level=5,
        name="Full Autonomy",
        trust_threshold=0.95,
        min_observation_hours=720.0,
        min_clean_windows=500,
        key_criteria="No severity >= 0.5 in 168h",
    ),
}


def get_level_definition(level: int) -> AutonomyLevel | None:
    """Get autonomy level definition by level number."""
    return AUTONOMY_LEVELS.get(level)


def can_promote(trust: float, current_level: int, observation_hours: float,
                clean_windows: int) -> int:
    """Check if promotion to a higher level is possible.

    Returns the highest level achievable, or current level if no promotion.
    """
    highest = current_level
    for lvl in range(current_level + 1, 6):
        defn = AUTONOMY_LEVELS[lvl]
        if (trust >= defn.trust_threshold and
                observation_hours >= defn.min_observation_hours and
                clean_windows >= defn.min_clean_windows):
            highest = lvl
        else:
            break
    return highest
