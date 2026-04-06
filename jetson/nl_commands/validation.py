"""
Command Validation and Safety Checks for NEXUS Marine Robotics Platform.

Implements syntax validation, semantic checks, safety rule enforcement,
permission control, risk estimation, and safe alternative computation.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from .intent import Intent, IntentType
from .executor import Command


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of validating a command."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    safe_to_execute: bool = True


class RiskLevel(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SafeAlternative:
    """A safer alternative to an unsafe command."""
    command_text: str
    description: str
    risk_level: RiskLevel = RiskLevel.LOW
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Role definitions
# ---------------------------------------------------------------------------

_ROLES: dict[str, int] = {
    "admin": 10,
    "operator": 7,
    "observer": 3,
    "guest": 1,
}

_INTENT_PERMISSIONS: dict[IntentType, int] = {
    IntentType.EMERGENCY_STOP: 1,   # Anyone can emergency stop
    IntentType.RETURN_HOME: 1,      # Anyone can return home
    IntentType.QUERY_STATUS: 1,     # Anyone can query status
    IntentType.NAVIGATE: 5,         # Operator+
    IntentType.STATION_KEEP: 5,
    IntentType.PATROL: 5,
    IntentType.SURVEY: 5,
    IntentType.SET_SPEED: 7,        # Operator
    IntentType.SET_HEADING: 7,
    IntentType.CONFIGURE: 10,       # Admin only
    IntentType.UNKNOWN: 0,
}

# Speed limits
_SPEED_LIMITS: dict[str, tuple[float, float]] = {
    "default": (0.0, 15.0),
    "harbor": (0.0, 5.0),
    "open_water": (0.0, 25.0),
    "emergency": (0.0, 30.0),
    "survey": (0.0, 8.0),
}

# Depth limits
_DEPTH_LIMITS: dict[str, tuple[float, float]] = {
    "default": (0.5, 200.0),
    "shallow": (0.5, 10.0),
    "deep": (5.0, 500.0),
}


# ---------------------------------------------------------------------------
# CommandValidator
# ---------------------------------------------------------------------------

class CommandValidator:
    """Multi-layered command validation and safety system.

    Provides syntax validation, semantic checks (against vessel state),
    safety rule enforcement, permission control, risk estimation,
    and safe alternative computation.
    """

    def __init__(self) -> None:
        self._safety_rules: list[dict[str, Any]] = []
        self._vessel_state: dict[str, Any] = {}
        self._load_default_safety_rules()

    # -- public API --------------------------------------------------------

    def validate_syntax(self, command: Command) -> ValidationResult:
        """Validate the syntactic correctness of a command."""
        errors: list[str] = []
        warnings: list[str] = []

        if command is None:
            return ValidationResult(valid=False, errors=["Command is None"], safe_to_execute=False)

        if command.intent is None:
            errors.append("Command has no intent")
            return ValidationResult(valid=False, errors=errors, safe_to_execute=False)

        if command.intent.type == IntentType.UNKNOWN:
            errors.append("Unknown intent type")
            return ValidationResult(valid=False, errors=errors, warnings=warnings, safe_to_execute=False)

        # Check required fields
        if not command.command_id:
            warnings.append("Command has no ID — one will be generated")

        if command.timestamp <= 0:
            errors.append("Invalid command timestamp")

        # Check for empty text
        if not command.intent.raw_text.strip():
            warnings.append("Empty command text")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            safe_to_execute=len(errors) == 0,
        )

    def validate_semantics(self, command: Command, vessel_state: dict[str, Any]) -> ValidationResult:
        """Validate the command semantics against current vessel state."""
        errors: list[str] = []
        warnings: list[str] = []

        self._vessel_state = vessel_state

        if command is None or command.intent is None:
            return ValidationResult(valid=False, errors=["Invalid command"], safe_to_execute=False)

        itype = command.intent.type

        # Check battery level for energy-intensive operations
        battery = vessel_state.get("battery_level", 100.0)
        if battery < 10:
            if itype in (IntentType.NAVIGATE, IntentType.PATROL, IntentType.SURVEY):
                errors.append(f"Insufficient battery ({battery:.0f}%) for {itype.value}")
            elif itype in (IntentType.RETURN_HOME, IntentType.EMERGENCY_STOP):
                warnings.append(f"Low battery ({battery:.0f}%) — prioritizing essential operations")
        elif battery < 25:
            warnings.append(f"Battery at {battery:.0f}% — consider returning home soon")

        # Check engine status
        engine_on = vessel_state.get("engine_on", True)
        if not engine_on and itype in (IntentType.NAVIGATE, IntentType.PATROL, IntentType.SET_SPEED):
            errors.append("Engine is off — cannot execute propulsion commands")

        # Check depth constraints
        current_depth = vessel_state.get("depth", 0.0)
        min_safe_depth = vessel_state.get("min_safe_depth", 0.5)
        max_safe_depth = vessel_state.get("max_safe_depth", 200.0)
        if current_depth < min_safe_depth:
            warnings.append(f"Vessel depth ({current_depth}m) is near minimum safe depth ({min_safe_depth}m)")
        if current_depth > max_safe_depth * 0.9:
            warnings.append(f"Vessel depth ({current_depth}m) is approaching maximum ({max_safe_depth}m)")

        # Check GPS fix
        gps_fix = vessel_state.get("gps_fix", True)
        if not gps_fix and itype == IntentType.NAVIGATE:
            errors.append("No GPS fix — navigation commands unavailable")
        if not gps_fix and itype == IntentType.RETURN_HOME:
            warnings.append("No GPS fix — return home may use dead reckoning")

        # Check for already-at-destination
        if itype == IntentType.NAVIGATE:
            current_pos = vessel_state.get("position", {})
            target = command.parameters.get("destination") or command.intent.slots.get("destination")
            if isinstance(target, str) and current_pos.get("name") == target:
                errors.append(f"Already at destination: {target}")

        # Check speed constraints
        if itype == IntentType.SET_SPEED:
            speed = self._extract_speed(command)
            if speed is not None:
                mode = vessel_state.get("mode", "default")
                limits = _SPEED_LIMITS.get(mode, _SPEED_LIMITS["default"])
                if speed > limits[1]:
                    errors.append(f"Speed {speed} exceeds limit of {limits[1]} for mode '{mode}'")
                if speed < limits[0]:
                    errors.append(f"Speed {speed} is below minimum of {limits[0]} for mode '{mode}'")

        # Check heading constraints
        if itype == IntentType.SET_HEADING:
            heading = self._extract_heading(command)
            if heading is not None:
                if heading < 0 or heading > 360:
                    errors.append(f"Heading {heading} is outside valid range [0, 360]")

        safe = len(errors) == 0
        return ValidationResult(
            valid=safe,
            errors=errors,
            warnings=warnings,
            safe_to_execute=safe,
        )

    def validate_safety(self, command: Command, safety_rules: Optional[list[dict[str, Any]]] = None) -> ValidationResult:
        """Validate command against safety rules."""
        errors: list[str] = []
        warnings: list[str] = []

        rules = safety_rules if safety_rules is not None else self._safety_rules

        if command is None or command.intent is None:
            return ValidationResult(valid=False, errors=["Invalid command"], safe_to_execute=False)

        for rule in rules:
            match = self._evaluate_rule(command, rule)
            if match:
                severity = rule.get("severity", "warning")
                msg = rule.get("message", f"Safety rule violated: {rule.get('name', 'unknown')}")
                if severity == "error":
                    errors.append(msg)
                else:
                    warnings.append(msg)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            safe_to_execute=len(errors) == 0,
        )

    def check_permissions(self, command: Command, user_role: str) -> tuple[bool, str]:
        """Check if *user_role* has permission to execute *command*.

        Returns (allowed: bool, message: str).
        """
        if command is None or command.intent is None:
            return False, "Invalid command"

        role_level = _ROLES.get(user_role.lower(), 0)
        required_level = _INTENT_PERMISSIONS.get(command.intent.type, 10)

        if role_level >= required_level:
            return True, f"Permission granted for '{command.intent.type.value}' (role: {user_role})"
        else:
            return False, f"Permission denied for '{command.intent.type.value}' — requires role level {required_level}, you have {role_level}"

    def estimate_risk(self, command: Command, current_conditions: dict[str, Any]) -> RiskLevel:
        """Estimate the risk level of executing *command* under *current_conditions*."""
        if command is None or command.intent is None:
            return RiskLevel.CRITICAL

        itype = command.intent.type
        risk_score = 0

        # Base risk by intent type
        base_risks = {
            IntentType.EMERGENCY_STOP: 1,
            IntentType.RETURN_HOME: 1,
            IntentType.QUERY_STATUS: 0,
            IntentType.SET_SPEED: 1,
            IntentType.SET_HEADING: 1,
            IntentType.CONFIGURE: 2,
            IntentType.STATION_KEEP: 1,
            IntentType.NAVIGATE: 2,
            IntentType.PATROL: 2,
            IntentType.SURVEY: 2,
            IntentType.UNKNOWN: 5,
        }
        risk_score += base_risks.get(itype, 2)

        # Weather conditions
        sea_state = current_conditions.get("sea_state", 0)  # 0=calm, 5=very rough
        if sea_state >= 4:
            risk_score += 2
        elif sea_state >= 2:
            risk_score += 1

        # Visibility
        visibility = current_conditions.get("visibility", 10.0)  # nautical miles
        if visibility < 1.0:
            risk_score += 2
        elif visibility < 3.0:
            risk_score += 1

        # Traffic density
        traffic = current_conditions.get("traffic_density", "low")
        traffic_risk = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        risk_score += traffic_risk.get(traffic, 0)

        # Proximity to shore
        distance_to_shore = current_conditions.get("distance_to_shore", 10.0)
        if distance_to_shore < 0.5:
            risk_score += 2
        elif distance_to_shore < 2.0:
            risk_score += 1

        # Battery
        battery = current_conditions.get("battery_level", 100)
        if battery < 15:
            risk_score += 2
        elif battery < 30:
            risk_score += 1

        # Nighttime
        is_night = current_conditions.get("is_night", False)
        if is_night:
            risk_score += 1

        # Map to risk levels
        if risk_score <= 1:
            return RiskLevel.LOW
        elif risk_score <= 3:
            return RiskLevel.MEDIUM
        elif risk_score <= 4:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def compute_safe_alternatives(self, unsafe_command: Command) -> list[SafeAlternative]:
        """Compute safer alternatives to an unsafe command."""
        alternatives: list[SafeAlternative] = []

        if unsafe_command is None or unsafe_command.intent is None:
            return alternatives

        itype = unsafe_command.intent.type

        if itype == IntentType.NAVIGATE:
            alternatives.append(SafeAlternative(
                command_text="station keep",
                description="Hold current position instead of navigating in unsafe conditions",
                risk_level=RiskLevel.LOW,
                confidence=0.9,
            ))
            alternatives.append(SafeAlternative(
                command_text="return home",
                description="Return to home position via a safe route",
                risk_level=RiskLevel.LOW,
                confidence=0.8,
            ))

        elif itype == IntentType.PATROL:
            alternatives.append(SafeAlternative(
                command_text="survey current area",
                description="Survey immediate surroundings instead of patrolling in unsafe conditions",
                risk_level=RiskLevel.LOW,
                confidence=0.7,
            ))
            alternatives.append(SafeAlternative(
                command_text="station keep",
                description="Hold position and monitor instead of active patrol",
                risk_level=RiskLevel.LOW,
                confidence=0.85,
            ))

        elif itype == IntentType.SURVEY:
            alternatives.append(SafeAlternative(
                command_text="query status",
                description="Check current systems status before surveying",
                risk_level=RiskLevel.NONE,
                confidence=0.9,
            ))

        elif itype == IntentType.SET_SPEED:
            current_speed = self._extract_speed(unsafe_command)
            if current_speed is not None and current_speed > 10:
                alternatives.append(SafeAlternative(
                    command_text="set speed to 5 knots",
                    description="Reduce speed to a safer value",
                    risk_level=RiskLevel.LOW,
                    confidence=0.85,
                ))

        elif itype == IntentType.CONFIGURE:
            alternatives.append(SafeAlternative(
                command_text="query status",
                description="Check current configuration before making changes",
                risk_level=RiskLevel.NONE,
                confidence=0.9,
            ))

        else:
            alternatives.append(SafeAlternative(
                command_text="station keep",
                description="Hold current position as a safe default",
                risk_level=RiskLevel.LOW,
                confidence=0.5,
            ))

        return alternatives

    def add_safety_rule(self, rule: dict[str, Any]) -> None:
        """Add a custom safety rule."""
        self._safety_rules.append(rule)

    def get_safety_rules(self) -> list[dict[str, Any]]:
        """Return all currently loaded safety rules."""
        return list(self._safety_rules)

    # -- internal helpers ---------------------------------------------------

    def _load_default_safety_rules(self) -> None:
        """Load default safety rules."""
        self._safety_rules = [
            {
                "name": "no_navigation_without_gps",
                "intent_type": IntentType.NAVIGATE,
                "condition": "no_gps_fix",
                "severity": "error",
                "message": "Navigation requires GPS fix",
            },
            {
                "name": "max_speed_in_harbor",
                "intent_type": IntentType.SET_SPEED,
                "condition": "harbor_mode",
                "severity": "warning",
                "message": "Speed limited in harbor mode",
            },
            {
                "name": "no_survey_during_emergency",
                "intent_type": IntentType.SURVEY,
                "condition": "emergency_mode",
                "severity": "error",
                "message": "Cannot conduct survey during emergency",
            },
            {
                "name": "low_battery_warning",
                "intent_type": IntentType.PATROL,
                "condition": "low_battery",
                "severity": "warning",
                "message": "Low battery — patrol may not complete",
            },
            {
                "name": "no_configure_underway",
                "intent_type": IntentType.CONFIGURE,
                "condition": "underway",
                "severity": "warning",
                "message": "Configuration changes while underway may have unintended effects",
            },
        ]

    def _evaluate_rule(self, command: Command, rule: dict[str, Any]) -> bool:
        """Evaluate a safety rule against a command."""
        if rule.get("intent_type") and command.intent:
            if rule["intent_type"] != command.intent.type:
                return False
        return True  # Rule matches intent — condition is checked by caller

    @staticmethod
    def _extract_speed(command: Command) -> Optional[float]:
        """Extract speed value from command parameters or slots."""
        for key in ("speed", "level"):
            val = command.parameters.get(key) or command.intent.slots.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        return None

    @staticmethod
    def _extract_heading(command: Command) -> Optional[float]:
        """Extract heading value from command parameters or slots."""
        for key in ("heading_degrees", "direction"):
            val = command.parameters.get(key) or command.intent.slots.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        return None
