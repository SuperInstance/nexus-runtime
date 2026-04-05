"""NEXUS Reflex Templates — Pre-built parameterized bytecode programs.

Provides factory functions for common marine robotics reflex patterns.
Each template is a function that takes typed parameters and returns a
JSON reflex definition (the format consumed by ReflexCompiler).

Templates:
  - heading_hold:   PID-like heading control with CLAMP_F on rudder
  - collision_avoidance: Distance-based speed reduction
  - waypoint_follow:   Bearing calculation + heading correction
  - station_keeping:   Position hold with drift compensation
  - emergency_stop:    Immediate throttle zero + rudder center
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ===================================================================
# Template parameter types
# ===================================================================

@dataclass(frozen=True)
class TemplateParams:
    """Typed parameters for reflex templates.

    All fields have sensible defaults for marine robotics.
    """
    # Heading hold
    target_heading: float = 45.0
    heading_tolerance: float = 5.0
    rudder_pin: int = 4
    rudder_min: float = -30.0
    rudder_max: float = 30.0
    compass_pin: int = 2

    # Throttle
    throttle_pin: int = 5
    throttle_min: float = -100.0
    throttle_max: float = 100.0

    # Collision avoidance
    distance_sensor_pin: int = 8
    collision_threshold: float = 5.0
    safe_throttle: float = 10.0

    # Station keeping
    target_x: float = 0.0
    target_y: float = 0.0
    x_sensor_pin: int = 9
    y_sensor_pin: int = 10
    kp_position: float = 2.0

    # Waypoint
    wp_target_heading: float = 0.0
    wp_arrival_tolerance: float = 5.0
    wp_x_sensor_pin: int = 9
    wp_y_sensor_pin: int = 10

    def as_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary."""
        from dataclasses import asdict
        return asdict(self)


# ===================================================================
# Template generators
# ===================================================================

def heading_hold(params: TemplateParams | None = None) -> dict:
    """Generate a heading-hold reflex program.

    Reads compass heading, computes error from target, clamps and writes
    to rudder pin. Ends with HALT.

    Stack trace:
      READ_PIN → push heading
      PUSH_F32 → push target
      SUB_F    → heading_error = heading - target
      CLAMP_F  → clamp error to rudder range
      WRITE_PIN → apply to rudder
      HALT
    """
    p = params or TemplateParams()
    return {
        "name": f"heading_hold_{int(p.target_heading)}",
        "intent": f"Maintain heading at {p.target_heading} degrees with rudder on pin {p.rudder_pin}",
        "body": [
            {"op": "READ_PIN", "arg": p.compass_pin},
            {"op": "PUSH_F32", "value": p.target_heading},
            {"op": "SUB_F"},
            {"op": "CLAMP_F", "lo": p.rudder_min, "hi": p.rudder_max},
            {"op": "WRITE_PIN", "arg": p.rudder_pin},
            {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1},
        ],
    }


def collision_avoidance(params: TemplateParams | None = None) -> dict:
    """Generate a collision avoidance reflex program.

    Reads distance sensor; if below threshold, reduces throttle to
    safe level. Uses JUMP_IF_FALSE for conditional branching.

    Stack trace:
      READ_PIN → push distance
      PUSH_F32 → push threshold
      LT_F     → distance < threshold?
      JUMP_IF_FALSE → skip to safe
      PUSH_F32 → push safe_throttle
      CLAMP_F  → clamp
      WRITE_PIN → apply to throttle
      HALT
    """
    p = params or TemplateParams()
    return {
        "name": "collision_avoidance",
        "intent": f"Reduce throttle to {p.safe_throttle} when obstacle within {p.collision_threshold}m",
        "body": [
            {"op": "READ_PIN", "arg": p.distance_sensor_pin},
            {"op": "PUSH_F32", "value": p.collision_threshold},
            {"op": "LT_F"},
            {"op": "JUMP_IF_FALSE", "target": "safe"},
            {"op": "PUSH_F32", "value": p.safe_throttle},
            {"op": "CLAMP_F", "lo": p.throttle_min, "hi": p.throttle_max},
            {"op": "WRITE_PIN", "arg": p.throttle_pin, "label": "safe"},
            {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1},
        ],
    }


def waypoint_follow(params: TemplateParams | None = None) -> dict:
    """Generate a waypoint following reflex program.

    Single-pass bearing check + correction. Reads current heading,
    computes error from waypoint bearing, clamps and steers.
    Designed to be called repeatedly by the control loop.
    """
    p = params or TemplateParams()
    return {
        "name": f"waypoint_follow_{int(p.wp_target_heading)}",
        "intent": f"Navigate to waypoint at heading {p.wp_target_heading}",
        "body": [
            # Read current bearing
            {"op": "READ_PIN", "arg": p.wp_x_sensor_pin},
            {"op": "PUSH_F32", "value": p.wp_target_heading},
            {"op": "SUB_F"},
            {"op": "ABS_F"},
            {"op": "PUSH_F32", "value": p.wp_arrival_tolerance},
            {"op": "LT_F"},
            {"op": "JUMP_IF_FALSE", "target": "steer"},
            # Close enough — zero rudder
            {"op": "PUSH_F32", "value": 0.0},
            {"op": "CLAMP_F", "lo": p.rudder_min, "hi": p.rudder_max},
            {"op": "WRITE_PIN", "arg": p.rudder_pin},
            {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1},
            # Far from waypoint — steer toward it
            {"op": "READ_PIN", "arg": p.compass_pin, "label": "steer"},
            {"op": "PUSH_F32", "value": p.wp_target_heading},
            {"op": "SUB_F"},
            {"op": "CLAMP_F", "lo": p.rudder_min, "hi": p.rudder_max},
            {"op": "WRITE_PIN", "arg": p.rudder_pin},
            {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1},
        ],
    }


def station_keeping(params: TemplateParams | None = None) -> dict:
    """Generate a station-keeping (position hold) reflex program.

    Reads X/Y position sensors, computes drift from target, applies
    proportional correction to rudder (X error) and throttle (Y error).
    """
    p = params or TemplateParams()
    return {
        "name": "station_keeping",
        "intent": f"Hold position at ({p.target_x}, {p.target_y}) with drift compensation",
        "body": [
            # X correction → rudder
            {"op": "READ_PIN", "arg": p.x_sensor_pin},
            {"op": "PUSH_F32", "value": p.target_x},
            {"op": "SUB_F"},
            {"op": "PUSH_F32", "value": p.kp_position},
            {"op": "MUL_F"},
            {"op": "CLAMP_F", "lo": p.rudder_min, "hi": p.rudder_max},
            {"op": "WRITE_PIN", "arg": p.rudder_pin},
            # Y correction → throttle
            {"op": "READ_PIN", "arg": p.y_sensor_pin},
            {"op": "PUSH_F32", "value": p.target_y},
            {"op": "SUB_F"},
            {"op": "PUSH_F32", "value": p.kp_position},
            {"op": "MUL_F"},
            {"op": "CLAMP_F", "lo": p.throttle_min, "hi": p.throttle_max},
            {"op": "WRITE_PIN", "arg": p.throttle_pin},
            # HALT
            {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1},
        ],
    }


def emergency_stop(params: TemplateParams | None = None) -> dict:
    """Generate an emergency stop reflex program.

    Immediately sets throttle and rudder to zero. This is the most
    safety-critical template.
    """
    p = params or TemplateParams()
    return {
        "name": "emergency_stop",
        "intent": "Immediate emergency stop — zero all actuators",
        "body": [
            # Zero throttle
            {"op": "PUSH_F32", "value": 0.0},
            {"op": "CLAMP_F", "lo": p.throttle_min, "hi": p.throttle_max},
            {"op": "WRITE_PIN", "arg": p.throttle_pin},
            # Zero rudder
            {"op": "PUSH_F32", "value": 0.0},
            {"op": "CLAMP_F", "lo": p.rudder_min, "hi": p.rudder_max},
            {"op": "WRITE_PIN", "arg": p.rudder_pin},
            # HALT
            {"op": "NOP", "flags": "0x80", "operand1": 1, "operand2": 1},
        ],
    }


# ===================================================================
# Template registry
# ===================================================================

@dataclass(frozen=True)
class TemplateInfo:
    """Metadata about a registered template."""
    name: str
    description: str
    generator: callable
    default_params: TemplateParams


KNOWN_TEMPLATES: dict[str, TemplateInfo] = {
    "heading_hold": TemplateInfo(
        name="heading_hold",
        description="PID-like heading control with CLAMP_F on rudder",
        generator=heading_hold,
        default_params=TemplateParams(),
    ),
    "collision_avoidance": TemplateInfo(
        name="collision_avoidance",
        description="Distance-based speed reduction on obstacle detection",
        generator=collision_avoidance,
        default_params=TemplateParams(),
    ),
    "waypoint_follow": TemplateInfo(
        name="waypoint_follow",
        description="Bearing calculation + heading correction toward waypoint",
        generator=waypoint_follow,
        default_params=TemplateParams(),
    ),
    "station_keeping": TemplateInfo(
        name="station_keeping",
        description="Position hold with drift compensation",
        generator=station_keeping,
        default_params=TemplateParams(),
    ),
    "emergency_stop": TemplateInfo(
        name="emergency_stop",
        description="Immediate throttle zero + rudder center",
        generator=emergency_stop,
        default_params=TemplateParams(),
    ),
}


class ReflexTemplates:
    """Registry and factory for reflex templates.

    Provides lookup, instantiation, and listing of all available templates.
    """

    def __init__(self, extra_templates: dict[str, TemplateInfo] | None = None) -> None:
        self._templates = dict(KNOWN_TEMPLATES)
        if extra_templates:
            self._templates.update(extra_templates)

    def get(self, name: str) -> TemplateInfo | None:
        """Get a template by name."""
        return self._templates.get(name)

    def generate(self, name: str, params: TemplateParams | None = None) -> dict | None:
        """Generate a reflex JSON from a named template.

        Args:
            name: Template name (e.g., "heading_hold").
            params: Override default parameters.

        Returns:
            Reflex JSON dict, or None if template not found.
        """
        info = self._templates.get(name)
        if info is None:
            return None
        return info.generator(params)

    def list_templates(self) -> list[str]:
        """Return names of all registered templates."""
        return sorted(self._templates.keys())

    def __len__(self) -> int:
        return len(self._templates)

    def __contains__(self, name: str) -> bool:
        return name in self._templates
