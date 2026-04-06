"""Autonomy level definitions (L0-L5) and capability management."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


class AutonomyLevel(enum.IntEnum):
    """Six-level autonomy scale from fully manual to fully autonomous."""
    MANUAL = 0
    ASSISTED = 1
    SEMI_AUTO = 2
    AUTO_WITH_SUPERVISION = 3
    FULL_AUTO = 4
    AUTONOMOUS = 5

    def __str__(self) -> str:
        return self.name


@dataclass
class LevelCapabilities:
    """Describes what operations and tolerances are allowed at a given level."""
    allowed_operations: List[str] = field(default_factory=list)
    required_human_approval: List[str] = field(default_factory=list)
    max_risk_tolerance: float = 0.0
    decision_authority: float = 0.0  # 0.0 = human, 1.0 = system


class AutonomyLevelManager:
    """Manages autonomy-level capability lookups and comparisons."""

    # Canonical capabilities per level
    _CAPABILITIES: Dict[AutonomyLevel, LevelCapabilities] = {}

    @classmethod
    def _ensure_capabilities(cls) -> None:
        if cls._CAPABILITIES:
            return
        cls._CAPABILITIES = {
            AutonomyLevel.MANUAL: LevelCapabilities(
                allowed_operations=[
                    "read_sensors", "display_info", "log_data",
                ],
                required_human_approval=["all"],
                max_risk_tolerance=0.0,
                decision_authority=0.0,
            ),
            AutonomyLevel.ASSISTED: LevelCapabilities(
                allowed_operations=[
                    "read_sensors", "display_info", "log_data",
                    "suggest_actions", "provide_warnings",
                ],
                required_human_approval=["all"],
                max_risk_tolerance=0.1,
                decision_authority=0.2,
            ),
            AutonomyLevel.SEMI_AUTO: LevelCapabilities(
                allowed_operations=[
                    "read_sensors", "display_info", "log_data",
                    "suggest_actions", "provide_warnings",
                    "execute_approved_actions", "adjust_parameters",
                ],
                required_human_approval=["critical", "hazardous"],
                max_risk_tolerance=0.3,
                decision_authority=0.4,
            ),
            AutonomyLevel.AUTO_WITH_SUPERVISION: LevelCapabilities(
                allowed_operations=[
                    "read_sensors", "display_info", "log_data",
                    "suggest_actions", "provide_warnings",
                    "execute_approved_actions", "adjust_parameters",
                    "plan_path", "control_speed", "avoid_obstacles",
                ],
                required_human_approval=["critical", "hazardous"],
                max_risk_tolerance=0.5,
                decision_authority=0.6,
            ),
            AutonomyLevel.FULL_AUTO: LevelCapabilities(
                allowed_operations=[
                    "read_sensors", "display_info", "log_data",
                    "suggest_actions", "provide_warnings",
                    "execute_approved_actions", "adjust_parameters",
                    "plan_path", "control_speed", "avoid_obstacles",
                    "navigate", "make_decisions", "adapt_behavior",
                ],
                required_human_approval=["critical"],
                max_risk_tolerance=0.7,
                decision_authority=0.8,
            ),
            AutonomyLevel.AUTONOMOUS: LevelCapabilities(
                allowed_operations=[
                    "read_sensors", "display_info", "log_data",
                    "suggest_actions", "provide_warnings",
                    "execute_approved_actions", "adjust_parameters",
                    "plan_path", "control_speed", "avoid_obstacles",
                    "navigate", "make_decisions", "adapt_behavior",
                    "self_diagnose", "request_help", "full_control",
                ],
                required_human_approval=[],
                max_risk_tolerance=0.9,
                decision_authority=1.0,
            ),
        }

    # ---- public API ----

    def get_capabilities(self, level: AutonomyLevel) -> LevelCapabilities:
        """Return the capability descriptor for *level*."""
        self._ensure_capabilities()
        return self._CAPABILITIES[level]

    def is_operation_allowed(
        self, operation: str, level: AutonomyLevel
    ) -> bool:
        """Check whether *operation* is in the allowed set for *level*."""
        caps = self.get_capabilities(level)
        return operation in caps.allowed_operations

    def get_max_risk_tolerance(self, level: AutonomyLevel) -> float:
        caps = self.get_capabilities(level)
        return caps.max_risk_tolerance

    def compute_decision_authority(self, level: AutonomyLevel) -> float:
        """Return decision-authority as a 0-100 percentage."""
        caps = self.get_capabilities(level)
        return caps.decision_authority * 100.0

    def list_level_operations(self, level: AutonomyLevel) -> List[str]:
        """Return a *copy* of the allowed-operations list."""
        caps = self.get_capabilities(level)
        return list(caps.allowed_operations)

    def compare_levels(
        self, a: AutonomyLevel, b: AutonomyLevel
    ) -> str:
        """Return ``'higher'``, ``'lower'``, or ``'equal'``."""
        if a > b:
            return "higher"
        if a < b:
            return "lower"
        return "equal"
