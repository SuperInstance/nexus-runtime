"""
Command Execution Engine for NEXUS Marine Robotics Platform.

Handles command lifecycle: validation, planning, execution, sequencing,
undo support, and history tracking. All operations are in-memory simulations
suitable for testing without hardware dependencies.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from .intent import Intent, IntentType


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class CommandPriority(Enum):
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class Command:
    """A ready-to-execute command."""
    intent: Intent
    parameters: dict[str, Any] = field(default_factory=dict)
    priority: CommandPriority = CommandPriority.NORMAL
    source: str = "nl_interface"
    timestamp: float = 0.0
    command_id: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()
        if not self.command_id:
            self.command_id = uuid.uuid4().hex[:12]


@dataclass
class ExecutionResult:
    """Outcome of executing a command."""
    success: bool
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    side_effects: list[str] = field(default_factory=list)


@dataclass
class ImpactAssessment:
    """Pre-execution impact estimate."""
    risk_level: str = "low"       # low, medium, high, critical
    estimated_duration: float = 0.0
    affected_systems: list[str] = field(default_factory=list)
    reversible: bool = True
    energy_cost: float = 0.0      # in arbitrary units
    description: str = ""


@dataclass
class ExecutionStep:
    """A single step in an execution plan."""
    step_number: int
    action: str
    description: str
    estimated_duration: float = 0.0
    dependencies: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CommandExecutor
# ---------------------------------------------------------------------------

class CommandExecutor:
    """Command execution engine for the NEXUS NL interface.

    Provides command validation, impact estimation, sequencing, execution
    simulation, undo support, and history tracking.
    """

    _MAX_HISTORY = 1000

    def __init__(self) -> None:
        self._history: deque[ExecutionResult] = deque(maxlen=self._MAX_HISTORY)
        self._command_log: deque[Command] = deque(maxlen=self._MAX_HISTORY)
        self._undo_stack: deque[Command] = deque(maxlen=100)
        self._exec_count: int = 0

    # -- public API --------------------------------------------------------

    def execute(self, command: Command) -> ExecutionResult:
        """Execute *command* and return the result."""
        start = time.time()
        validation = self.validate_command(command)
        if not validation["valid"]:
            elapsed = time.time() - start
            return ExecutionResult(
                success=False,
                message=f"Command validation failed: {'; '.join(validation['errors'])}",
                execution_time=elapsed,
                side_effects=[],
            )

        # Simulate execution based on intent type
        result = self._simulate_execution(command)
        elapsed = time.time() - start

        result.execution_time = elapsed
        self._command_log.append(command)
        self._history.append(result)
        self._undo_stack.append(command)
        self._exec_count += 1
        return result

    def validate_command(self, command: Command) -> dict[str, Any]:
        """Validate a command and return a dict with 'valid', 'errors', 'warnings'."""
        errors: list[str] = []
        warnings: list[str] = []

        if command.intent is None:
            errors.append("Command has no intent")
            return {"valid": False, "errors": errors, "warnings": warnings}

        if command.intent.type == IntentType.UNKNOWN:
            errors.append("Unknown intent type — cannot execute")
            return {"valid": False, "errors": errors, "warnings": warnings}

        # Intent-specific validation
        if command.intent.type == IntentType.SET_SPEED:
            speed = command.parameters.get("speed", command.intent.slots.get("speed"))
            if speed is not None:
                try:
                    speed_val = float(speed)
                    if speed_val < 0:
                        errors.append("Speed cannot be negative")
                    if speed_val > 50:
                        warnings.append("Very high speed requested")
                except (ValueError, TypeError):
                    errors.append(f"Invalid speed value: {speed}")

        if command.intent.type == IntentType.SET_HEADING:
            heading = command.parameters.get("heading_degrees", command.intent.slots.get("heading_degrees"))
            if heading is not None:
                try:
                    h = float(heading)
                    if h < 0 or h > 360:
                        errors.append("Heading must be between 0 and 360 degrees")
                except (ValueError, TypeError):
                    errors.append(f"Invalid heading value: {heading}")

        if command.intent.type == IntentType.NAVIGATE:
            dest = command.parameters.get("destination", command.intent.slots.get("destination"))
            if not dest:
                warnings.append("Navigation command has no destination specified")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def estimate_impact(self, command: Command) -> ImpactAssessment:
        """Estimate the impact of executing *command*."""
        itype = command.intent.type if command.intent else IntentType.UNKNOWN

        risk = "low"
        duration = 1.0
        systems: list[str] = []
        reversible = True
        energy = 1.0
        description = ""

        impact_map: dict[IntentType, dict[str, Any]] = {
            IntentType.EMERGENCY_STOP: {
                "risk": "high", "duration": 0.1,
                "systems": ["propulsion", "navigation", "all"],
                "reversible": True, "energy": 0.0,
                "description": "Immediately stops all vessel motion",
            },
            IntentType.NAVIGATE: {
                "risk": "medium", "duration": 60.0,
                "systems": ["propulsion", "navigation", "guidance"],
                "reversible": True, "energy": 5.0,
                "description": "Navigate vessel to target destination",
            },
            IntentType.STATION_KEEP: {
                "risk": "low", "duration": 300.0,
                "systems": ["propulsion", "positioning"],
                "reversible": True, "energy": 3.0,
                "description": "Maintain current position",
            },
            IntentType.PATROL: {
                "risk": "medium", "duration": 600.0,
                "systems": ["propulsion", "navigation", "sensors"],
                "reversible": True, "energy": 15.0,
                "description": "Patrol designated area",
            },
            IntentType.SURVEY: {
                "risk": "low", "duration": 900.0,
                "systems": ["propulsion", "sensors", "navigation"],
                "reversible": True, "energy": 10.0,
                "description": "Survey designated area",
            },
            IntentType.RETURN_HOME: {
                "risk": "medium", "duration": 120.0,
                "systems": ["propulsion", "navigation"],
                "reversible": False, "energy": 8.0,
                "description": "Return vessel to home/dock position",
            },
            IntentType.SET_SPEED: {
                "risk": "low", "duration": 1.0,
                "systems": ["propulsion"],
                "reversible": True, "energy": 0.5,
                "description": "Adjust vessel speed",
            },
            IntentType.SET_HEADING: {
                "risk": "low", "duration": 2.0,
                "systems": ["propulsion", "steering"],
                "reversible": True, "energy": 1.0,
                "description": "Adjust vessel heading",
            },
            IntentType.QUERY_STATUS: {
                "risk": "low", "duration": 0.5,
                "systems": [],
                "reversible": True, "energy": 0.1,
                "description": "Query vessel status information",
            },
            IntentType.CONFIGURE: {
                "risk": "medium", "duration": 2.0,
                "systems": ["configuration"],
                "reversible": True, "energy": 0.5,
                "description": "Modify vessel configuration",
            },
            IntentType.UNKNOWN: {
                "risk": "low", "duration": 0.0,
                "systems": [],
                "reversible": True, "energy": 0.0,
                "description": "Unknown command",
            },
        }

        info = impact_map.get(itype, impact_map[IntentType.UNKNOWN])
        return ImpactAssessment(
            risk_level=info["risk"],
            estimated_duration=info["duration"],
            affected_systems=info["systems"],
            reversible=info["reversible"],
            energy_cost=info["energy"],
            description=info["description"],
        )

    def sequence_commands(self, commands: list[Command]) -> list[Command]:
        """Order a list of commands for optimal/safe execution."""
        if not commands:
            return []

        # Emergency stop always first
        emergency = [c for c in commands if c.intent and c.intent.type == IntentType.EMERGENCY_STOP]
        non_emergency = [c for c in commands if c not in emergency]

        # Sort remaining by priority (descending), then timestamp (ascending)
        non_emergency.sort(key=lambda c: (-c.priority.value, c.timestamp))
        return emergency + non_emergency

    def undo_last_command(self) -> ExecutionResult:
        """Attempt to undo the most recent command."""
        if not self._undo_stack:
            return ExecutionResult(
                success=False,
                message="No command to undo",
                execution_time=0.0,
            )

        last_cmd = self._undo_stack.pop()

        if last_cmd.intent and last_cmd.intent.type == IntentType.EMERGENCY_STOP:
            return ExecutionResult(
                success=False,
                message="Emergency stop cannot be undone",
                execution_time=0.001,
                side_effects=["safety_lock"],
            )

        undo_msg = f"Undo command: {last_cmd.intent.type.value}"
        return ExecutionResult(
            success=True,
            message=undo_msg,
            data={"undone_command_id": last_cmd.command_id, "undone_intent": last_cmd.intent.type.value},
            execution_time=0.001,
            side_effects=["state_restored"],
        )

    def get_command_history(self) -> list[dict[str, Any]]:
        """Return the command execution history."""
        history = []
        for i, (cmd, result) in enumerate(zip(self._command_log, self._history)):
            history.append({
                "index": i,
                "command_id": cmd.command_id,
                "intent": cmd.intent.type.value if cmd.intent else "unknown",
                "success": result.success,
                "message": result.message,
                "timestamp": cmd.timestamp,
                "execution_time": result.execution_time,
            })
        return history

    def compute_execution_plan(self, command: Command) -> list[ExecutionStep]:
        """Compute a step-by-step execution plan for *command*."""
        steps: list[ExecutionStep] = []
        itype = command.intent.type if command.intent else IntentType.UNKNOWN

        if itype == IntentType.NAVIGATE:
            steps = [
                ExecutionStep(1, "validate_route", "Validate navigation route safety", 0.1),
                ExecutionStep(2, "check_obstacles", "Check for obstacles along route", 0.2),
                ExecutionStep(3, "compute_path", "Compute optimal path", 0.3),
                ExecutionStep(4, "engage_propulsion", "Engage propulsion system", 0.1),
                ExecutionStep(5, "follow_path", "Follow computed path to destination", 30.0),
                ExecutionStep(6, "confirm_arrival", "Confirm arrival at destination", 0.1),
            ]
        elif itype == IntentType.EMERGENCY_STOP:
            steps = [
                ExecutionStep(1, "cut_throttle", "Cut throttle to zero", 0.01),
                ExecutionStep(2, "disengage_autopilot", "Disengage autopilot", 0.01),
                ExecutionStep(3, "confirm_stop", "Confirm vessel has stopped", 0.05),
            ]
        elif itype == IntentType.STATION_KEEP:
            steps = [
                ExecutionStep(1, "lock_position", "Lock current GPS position", 0.1),
                ExecutionStep(2, "enable_station_keeping", "Enable station keeping mode", 0.1),
                ExecutionStep(3, "monitor_drift", "Continuously monitor position drift", 60.0),
            ]
        elif itype == IntentType.PATROL:
            steps = [
                ExecutionStep(1, "load_patrol_pattern", "Load patrol area and pattern", 0.2),
                ExecutionStep(2, "start_patrol", "Begin patrol route", 0.1),
                ExecutionStep(3, "monitor_area", "Monitor area during patrol", 120.0),
                ExecutionStep(4, "complete_patrol", "Mark patrol as complete", 0.1),
            ]
        elif itype == IntentType.SURVEY:
            steps = [
                ExecutionStep(1, "configure_sensors", "Configure survey sensors", 0.5),
                ExecutionStep(2, "start_survey_pattern", "Begin survey coverage pattern", 0.2),
                ExecutionStep(3, "collect_data", "Collect survey data", 180.0),
                ExecutionStep(4, "process_data", "Process and store collected data", 1.0),
            ]
        elif itype == IntentType.RETURN_HOME:
            steps = [
                ExecutionStep(1, "compute_home_route", "Compute route to home position", 0.5),
                ExecutionStep(2, "begin_return", "Begin return journey", 0.1),
                ExecutionStep(3, "navigate_home", "Navigate to home position", 60.0),
                ExecutionStep(4, "dock", "Execute docking sequence", 5.0),
            ]
        elif itype == IntentType.SET_SPEED:
            steps = [
                ExecutionStep(1, "validate_speed", "Validate requested speed", 0.01),
                ExecutionStep(2, "adjust_throttle", "Adjust throttle to target speed", 0.5),
                ExecutionStep(3, "confirm_speed", "Confirm speed change", 0.1),
            ]
        elif itype == IntentType.SET_HEADING:
            steps = [
                ExecutionStep(1, "validate_heading", "Validate requested heading", 0.01),
                ExecutionStep(2, "adjust_rudder", "Adjust rudder/steering", 1.0),
                ExecutionStep(3, "confirm_heading", "Confirm heading change", 0.1),
            ]
        elif itype == IntentType.QUERY_STATUS:
            steps = [
                ExecutionStep(1, "gather_system_data", "Gather data from all subsystems", 0.1),
                ExecutionStep(2, "compile_report", "Compile status report", 0.2),
            ]
        elif itype == IntentType.CONFIGURE:
            steps = [
                ExecutionStep(1, "validate_config", "Validate configuration change", 0.1),
                ExecutionStep(2, "apply_config", "Apply new configuration", 0.2),
                ExecutionStep(3, "verify_config", "Verify configuration applied correctly", 0.1),
            ]
        else:
            steps = [
                ExecutionStep(1, "unknown", "Unknown command type — no plan available", 0.0),
            ]

        return steps

    # -- internal helpers ---------------------------------------------------

    def _simulate_execution(self, command: Command) -> ExecutionResult:
        """Simulate command execution and return a result."""
        itype = command.intent.type

        success_map: dict[IntentType, tuple[bool, str]] = {
            IntentType.NAVIGATE: (True, f"Navigating to {command.parameters.get('destination', 'target')}")
                                if command.parameters.get('destination')
                                else (True, "Navigation initiated"),
            IntentType.STATION_KEEP: (True, "Station keeping mode engaged"),
            IntentType.PATROL: (True, f"Patrolling {command.parameters.get('zone', 'designated area')}"),
            IntentType.SURVEY: (True, f"Surveying {command.parameters.get('area', 'designated area')}"),
            IntentType.EMERGENCY_STOP: (True, "Emergency stop executed — all propulsion halted"),
            IntentType.RETURN_HOME: (True, "Returning to home position"),
            IntentType.SET_SPEED: (True, f"Speed set to {command.parameters.get('speed', command.intent.slots.get('speed', 'target'))}"),
            IntentType.SET_HEADING: (True, f"Heading set to {command.parameters.get('heading_degrees', command.intent.slots.get('heading_degrees', 'target'))}"),
            IntentType.QUERY_STATUS: (True, "Status report generated"),
            IntentType.CONFIGURE: (True, f"Configuration updated: {command.parameters.get('parameter', 'parameter')}"),
            IntentType.UNKNOWN: (False, "Cannot execute unknown command"),
        }

        ok, msg = success_map.get(itype, (False, "Unhandled intent type"))
        return ExecutionResult(
            success=ok,
            message=msg,
            data={"intent_type": itype.value, "command_id": command.command_id},
            side_effects=["simulated"],
        )
