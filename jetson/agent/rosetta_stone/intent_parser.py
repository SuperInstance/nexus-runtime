"""Rosetta Stone Layer 1: Natural language intent parser.

Rule-based parser that converts human-readable intent strings into
structured Intent dataclass objects. No LLM — pure regex + pattern matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ===================================================================
# Intent data structures
# ===================================================================

@dataclass
class Intent:
    """Structured representation of a parsed human intent.

    Supports these action types:
        READ        - Read a sensor or variable
        WRITE       - Write to an actuator or variable
        CONDITIONAL - If-then branching on sensor comparison
        LOOP        - Repeat a block of actions N times
        WAIT        - Delay for N cycles
        PID         - Compute PID control on a sensor
        NAVIGATE    - Navigate to a waypoint
        SYSCALL     - System call (snapshot, event emit, halt)
        COMPOUND    - Multi-part intent combining multiple actions
    """

    action: str  # READ, WRITE, CONDITIONAL, LOOP, WAIT, PID, NAVIGATE, SYSCALL, COMPOUND
    target: str  # SENSOR, ACTUATOR, VARIABLE, WAYPOINT, SYSTEM
    pin: int | None = None
    value: float | None = None
    threshold: float | None = None
    operator: str | None = None  # gt, lt, eq, gte, lte
    body: list[Intent] | None = None  # for compound/loop intents
    then_body: list[Intent] | None = None  # for conditional 'then' branch
    params: dict | None = None  # for PID params, waypoint coords, etc
    confidence: float = 1.0  # 1.0 = fully parsed, <1.0 = ambiguous
    raw: str = ""  # original text

    def __post_init__(self) -> None:
        if self.body is None:
            self.body = []
        if self.then_body is None:
            self.then_body = []
        if self.params is None:
            self.params = {}


# ===================================================================
# Operator word -> canonical operator mapping
# ===================================================================

_OPERATOR_MAP: dict[str, str] = {
    "greater than": "gt",
    ">": "gt",
    "less than": "lt",
    "<": "lt",
    "equals": "eq",
    "==": "eq",
    "equal to": "eq",
    "gte": "gte",
    ">=": "gte",
    "greater than or equal": "gte",
    "lte": "lte",
    "<=": "lte",
    "less than or equal": "lte",
}


# ===================================================================
# Regex patterns for intent parsing
# ===================================================================

# "read sensor <n>"
_PATTERN_READ_SENSOR = re.compile(
    r"read\s+sensor\s+(\d+)(?![\d.])", re.IGNORECASE
)

# "set actuator <n> to <value>"
_PATTERN_WRITE_ACTUATOR = re.compile(
    r"set\s+actuator\s+(\d+)(?![\d.])\s+to\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)

# "read variable <n>" or "read var <n>"
_PATTERN_READ_VARIABLE = re.compile(
    r"read\s+(?:variable|var)\s+(\d+)(?![\d.])", re.IGNORECASE
)

# "set variable <n> to <value>"
_PATTERN_WRITE_VARIABLE = re.compile(
    r"set\s+(?:variable|var)\s+(\d+)(?![\d.])\s+to\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)

# "if sensor <n> <op> <threshold> then <action>"
_PATTERN_CONDITIONAL = re.compile(
    r"if\s+sensor\s+(\d+)(?![\d.])\s+"
    r"(greater than or equal|less than or equal|greater than|less than|equal to|equals|"
    r"gte|lte|gt|lt|eq|[><]=?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+then\s+(.+)",
    re.IGNORECASE,
)

# "repeat <n> times: <actions>"
_PATTERN_LOOP = re.compile(
    r"repeat\s+(\d+)\s+times?\s*:\s*(.+)", re.IGNORECASE
)

# "wait <n> cycles"
_PATTERN_WAIT = re.compile(
    r"wait\s+(\d+)\s+cycles?", re.IGNORECASE
)

# "compute pid on sensor <n> with kp=<v> ki=<v> kd=<v>"
_PATTERN_PID = re.compile(
    r"compute\s+pid\s+on\s+sensor\s+(\d+)(?![\d.])\s+with\s+"
    r"kp\s*=\s*([-+]?\d*\.?\d+)\s*,?\s*"
    r"ki\s*=\s*([-+]?\d*\.?\d+)\s*,?\s*"
    r"kd\s*=\s*([-+]?\d*\.?\d+)",
    re.IGNORECASE,
)

# "navigate to waypoint <x>,<y>"
_PATTERN_NAVIGATE = re.compile(
    r"navigate\s+to\s+waypoint\s+"
    r"([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)",
    re.IGNORECASE,
)

# "log snapshot"
_PATTERN_LOG_SNAPSHOT = re.compile(
    r"log\s+snapshot", re.IGNORECASE
)

# "emit event <msg>"
_PATTERN_EMIT_EVENT = re.compile(
    r"emit\s+event\s+(.+)", re.IGNORECASE
)

# "halt" or "stop"
_PATTERN_HALT = re.compile(
    r"^(?:halt|stop)$", re.IGNORECASE
)

# Compound: "monitor sensor <n> and if <op> <threshold> then trigger actuator <m>"
_PATTERN_MONITOR_TRIGGER = re.compile(
    r"monitor\s+sensor\s+(\d+)\s+and\s+if\s+"
    r"(greater than or equal|less than or equal|greater than|less than|equal to|equals|"
    r"gte|lte|gt|lt|eq|[><]=?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+then\s+trigger\s+actuator\s+(\d+)",
    re.IGNORECASE,
)

# Compound: "patrol: read GPS, if distance > 100m return home"
_PATTERN_PATROL = re.compile(
    r"patrol\s*:\s*(.+)", re.IGNORECASE
)


# ===================================================================
# IntentParser
# ===================================================================

class IntentParser:
    """Parse human-readable intent strings into structured Intent objects.

    Supports simple intents (read, write, conditional, loop, wait, pid,
    navigate, syscall) and compound intents (monitor+trigger, patrol).
    """

    def parse(self, text: str) -> Intent:
        """Parse a single intent string.

        Args:
            text: Human-readable intent string.

        Returns:
            Parsed Intent object. If parsing is ambiguous, confidence < 1.0.
            If no pattern matches, returns an Intent with action='UNKNOWN'.

        Raises:
            ValueError: If text is empty or whitespace-only.
        """
        stripped = text.strip()
        if not stripped:
            raise ValueError("Empty intent text")

        # Try compound patterns first (more specific)
        intent = self._try_compound_patterns(stripped)
        if intent is not None:
            return intent

        # Try simple patterns
        intent = self._try_simple_patterns(stripped)
        if intent is not None:
            return intent

        # No match -- return unknown
        return Intent(
            action="UNKNOWN",
            target="SYSTEM",
            confidence=0.0,
            raw=stripped,
        )

    def parse_many(self, texts: list[str]) -> list[Intent]:
        """Parse multiple intent strings.

        Args:
            texts: List of human-readable intent strings.

        Returns:
            List of parsed Intent objects.
        """
        return [self.parse(t) for t in texts]

    # -----------------------------------------------------------------
    # Compound patterns
    # -----------------------------------------------------------------

    def _try_compound_patterns(self, text: str) -> Intent | None:
        """Try compound intent patterns."""
        # Monitor + trigger
        m = _PATTERN_MONITOR_TRIGGER.match(text)
        if m:
            sensor_pin = int(m.group(1))
            op = _OPERATOR_MAP.get(m.group(2).lower(), m.group(2).lower())
            threshold = float(m.group(3))
            actuator_pin = int(m.group(4))

            conditional_intent = Intent(
                action="CONDITIONAL",
                target="SENSOR",
                pin=sensor_pin,
                operator=op,
                threshold=threshold,
                then_body=[
                    Intent(
                        action="WRITE",
                        target="ACTUATOR",
                        pin=actuator_pin,
                        value=1.0,
                        raw=text,
                    )
                ],
                raw=text,
            )

            return Intent(
                action="COMPOUND",
                target="SYSTEM",
                body=[conditional_intent],
                confidence=1.0,
                raw=text,
            )

        # Patrol pattern
        m = _PATTERN_PATROL.match(text)
        if m:
            patrol_desc = m.group(1).strip()
            sub_intents = self._parse_patrol_body(patrol_desc)
            return Intent(
                action="COMPOUND",
                target="SYSTEM",
                body=sub_intents,
                params={"patrol": True},
                confidence=0.8,  # Patrol is somewhat ambiguous
                raw=text,
            )

        return None

    def _parse_patrol_body(self, desc: str) -> list[Intent]:
        """Parse patrol body description into sub-intents.

        Handles: "read GPS, if distance > 100m return home"
        """
        intents: list[Intent] = []

        # Split by comma or "then"
        parts = re.split(r",\s*(?:and\s+)?|\s+then\s+", desc)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # "read GPS" -> READ sensor 0 (GPS on pin 0)
            if re.match(r"read\s+gps", part, re.IGNORECASE):
                intents.append(Intent(
                    action="READ",
                    target="SENSOR",
                    pin=0,
                    raw=part,
                ))
                continue

            # "if distance > 100m return home" -> conditional halt
            dist_match = re.match(
                r"if\s+distance\s+([><]=?|gte|lte|gt|lt|eq)\s+(\d+)\s*\w*\s+return\s+home",
                part, re.IGNORECASE,
            )
            if dist_match:
                op = _OPERATOR_MAP.get(dist_match.group(1).lower(), dist_match.group(1).lower())
                threshold = float(dist_match.group(2))
                intents.append(Intent(
                    action="CONDITIONAL",
                    target="SENSOR",
                    pin=0,
                    operator=op,
                    threshold=threshold,
                    then_body=[Intent(
                        action="SYSCALL",
                        target="SYSTEM",
                        params={"syscall": "halt"},
                        raw=part,
                    )],
                    raw=part,
                ))
                continue

            # Try parsing as a regular intent
            sub = self._try_simple_patterns(part)
            if sub is not None:
                intents.append(sub)

        return intents

    # -----------------------------------------------------------------
    # Simple patterns
    # -----------------------------------------------------------------

    def _try_simple_patterns(self, text: str) -> Intent | None:
        """Try all simple intent patterns in order of specificity."""

        # PID (most specific)
        m = _PATTERN_PID.match(text)
        if m:
            return Intent(
                action="PID",
                target="SENSOR",
                pin=int(m.group(1)),
                params={
                    "kp": float(m.group(2)),
                    "ki": float(m.group(3)),
                    "kd": float(m.group(4)),
                },
                confidence=1.0,
                raw=text,
            )

        # Navigate to waypoint
        m = _PATTERN_NAVIGATE.match(text)
        if m:
            return Intent(
                action="NAVIGATE",
                target="WAYPOINT",
                params={
                    "x": float(m.group(1)),
                    "y": float(m.group(2)),
                },
                confidence=1.0,
                raw=text,
            )

        # Conditional
        m = _PATTERN_CONDITIONAL.match(text)
        if m:
            sensor_pin = int(m.group(1))
            op = _OPERATOR_MAP.get(m.group(2).lower(), m.group(2).lower())
            threshold = float(m.group(3))
            then_text = m.group(4).strip()

            then_intents = self._parse_then_action(then_text)

            return Intent(
                action="CONDITIONAL",
                target="SENSOR",
                pin=sensor_pin,
                operator=op,
                threshold=threshold,
                then_body=then_intents,
                confidence=1.0,
                raw=text,
            )

        # Loop
        m = _PATTERN_LOOP.match(text)
        if m:
            count = int(m.group(1))
            body_text = m.group(2).strip()

            body_intents = self._parse_body_actions(body_text)

            if not body_intents:
                return Intent(
                    action="LOOP",
                    target="SYSTEM",
                    params={"count": count},
                    confidence=0.5,
                    raw=text,
                )

            return Intent(
                action="LOOP",
                target="SYSTEM",
                body=body_intents,
                params={"count": count},
                confidence=1.0,
                raw=text,
            )

        # Wait
        m = _PATTERN_WAIT.match(text)
        if m:
            return Intent(
                action="WAIT",
                target="SYSTEM",
                params={"cycles": int(m.group(1))},
                confidence=1.0,
                raw=text,
            )

        # Write actuator
        m = _PATTERN_WRITE_ACTUATOR.match(text)
        if m:
            return Intent(
                action="WRITE",
                target="ACTUATOR",
                pin=int(m.group(1)),
                value=float(m.group(2)),
                confidence=1.0,
                raw=text,
            )

        # Write variable
        m = _PATTERN_WRITE_VARIABLE.match(text)
        if m:
            return Intent(
                action="WRITE",
                target="VARIABLE",
                pin=int(m.group(1)),
                value=float(m.group(2)),
                raw=text,
            )

        # Read sensor
        m = _PATTERN_READ_SENSOR.match(text)
        if m:
            return Intent(
                action="READ",
                target="SENSOR",
                pin=int(m.group(1)),
                confidence=1.0,
                raw=text,
            )

        # Read variable
        m = _PATTERN_READ_VARIABLE.match(text)
        if m:
            return Intent(
                action="READ",
                target="VARIABLE",
                pin=int(m.group(1)),
                confidence=1.0,
                raw=text,
            )

        # Log snapshot
        m = _PATTERN_LOG_SNAPSHOT.match(text)
        if m:
            return Intent(
                action="SYSCALL",
                target="SYSTEM",
                params={"syscall": "record_snapshot"},
                confidence=1.0,
                raw=text,
            )

        # Emit event
        m = _PATTERN_EMIT_EVENT.match(text)
        if m:
            return Intent(
                action="SYSCALL",
                target="SYSTEM",
                params={"syscall": "emit_event", "message": m.group(1).strip()},
                confidence=1.0,
                raw=text,
            )

        # Halt
        m = _PATTERN_HALT.match(text)
        if m:
            return Intent(
                action="SYSCALL",
                target="SYSTEM",
                params={"syscall": "halt"},
                confidence=1.0,
                raw=text,
            )

        return None

    # -----------------------------------------------------------------
    # Then-action parsing for conditionals
    # -----------------------------------------------------------------

    def _parse_then_action(self, text: str) -> list[Intent]:
        """Parse the 'then' clause of a conditional intent."""
        text = text.strip()

        # "set actuator <n> to <value>"
        m = _PATTERN_WRITE_ACTUATOR.match(text)
        if m:
            return [Intent(
                action="WRITE",
                target="ACTUATOR",
                pin=int(m.group(1)),
                value=float(m.group(2)),
                raw=text,
            )]

        # "trigger actuator <n>" (default value = 1.0)
        m = re.match(r"trigger\s+actuator\s+(\d+)", text, re.IGNORECASE)
        if m:
            return [Intent(
                action="WRITE",
                target="ACTUATOR",
                pin=int(m.group(1)),
                value=1.0,
                raw=text,
            )]

        # "halt" or "stop"
        if re.match(r"^(?:halt|stop)$", text, re.IGNORECASE):
            return [Intent(
                action="SYSCALL",
                target="SYSTEM",
                params={"syscall": "halt"},
                raw=text,
            )]

        # "log snapshot"
        if re.match(r"log\s+snapshot", text, re.IGNORECASE):
            return [Intent(
                action="SYSCALL",
                target="SYSTEM",
                params={"syscall": "record_snapshot"},
                raw=text,
            )]

        # "emit event <msg>"
        m = _PATTERN_EMIT_EVENT.match(text)
        if m:
            return [Intent(
                action="SYSCALL",
                target="SYSTEM",
                params={"syscall": "emit_event", "message": m.group(1).strip()},
                raw=text,
            )]

        # Try generic parse
        sub = self._try_simple_patterns(text)
        if sub is not None:
            return [sub]

        return [Intent(action="UNKNOWN", target="SYSTEM", confidence=0.0, raw=text)]

    # -----------------------------------------------------------------
    # Body-action parsing for loops
    # -----------------------------------------------------------------

    def _parse_body_actions(self, text: str) -> list[Intent]:
        """Parse a comma-separated list of actions for a loop body."""
        # Split by comma, semicolon, or "and"
        parts = re.split(r",\s*(?:and\s+)?|\s*;\s*|\s+and\s+", text)
        intents: list[Intent] = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            sub = self._try_simple_patterns(part)
            if sub is not None:
                intents.append(sub)

        return intents
