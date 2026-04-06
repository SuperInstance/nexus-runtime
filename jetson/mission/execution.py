"""Mission execution engine for NEXUS marine robotics platform.

Handles plan execution, phase transitions, pause/resume/abort,
and progress tracking.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Callable

# Import planner types for type compatibility
from jetson.mission.planner import MissionPlan, MissionPhase


class ExecutionState(Enum):
    """Current execution state of a mission or phase."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


@dataclass
class PhaseResult:
    """Result of executing a mission phase."""
    phase_name: str = ""
    state: ExecutionState = ExecutionState.IDLE
    progress: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: float = 0.0
    results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    success_criteria_met: List[str] = field(default_factory=list)
    success_criteria_failed: List[str] = field(default_factory=list)


@dataclass
class PhaseExecution:
    """Tracks execution state of a phase."""
    phase: Optional[MissionPhase] = None
    state: ExecutionState = ExecutionState.IDLE
    progress: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MissionResult:
    """Result of executing a complete mission plan."""
    plan_id: str = ""
    state: ExecutionState = ExecutionState.IDLE
    total_progress: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: float = 0.0
    phase_results: List[PhaseResult] = field(default_factory=list)
    completed_phases: List[str] = field(default_factory=list)
    failed_phases: List[str] = field(default_factory=list)
    aborted: bool = False
    abort_reason: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransitionResult:
    """Result of a phase transition."""
    from_phase: str = ""
    to_phase: str = ""
    success: bool = True
    error: Optional[str] = None
    transition_time: float = 0.0


class MissionExecutor:
    """Executes mission plans, manages phase transitions, and tracks progress."""

    def __init__(self):
        self._state: ExecutionState = ExecutionState.IDLE
        self._current_plan: Optional[MissionPlan] = None
        self._current_phase_idx: int = -1
        self._phase_executions: List[PhaseExecution] = []
        self._phase_results: List[PhaseResult] = []
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._paused_at: Optional[float] = None
        self._pause_accumulated: float = 0.0
        self._abort_reason: Optional[str] = None
        self._transition_log: List[TransitionResult] = []
        self._hooks: Dict[str, List[Callable]] = {}

    def execute_plan(self, plan: MissionPlan) -> MissionResult:
        """Execute a complete mission plan. Returns MissionResult."""
        self._current_plan = plan
        self._state = ExecutionState.RUNNING
        self._start_time = time.time()
        self._end_time = None
        self._abort_reason = None
        self._pause_accumulated = 0.0
        self._phase_results = []
        self._phase_executions = []
        self._current_phase_idx = -1

        self._fire_hook("mission_start", plan)

        for i, phase in enumerate(plan.phases):
            if self._state == ExecutionState.ABORTED:
                break
            self._current_phase_idx = i
            phase_result = self.execute_phase(phase)
            self._phase_results.append(phase_result)
            if phase_result.state == ExecutionState.FAILED:
                self._state = ExecutionState.FAILED
                break

        if self._state == ExecutionState.RUNNING:
            self._state = ExecutionState.COMPLETED

        self._end_time = time.time()
        self._fire_hook("mission_end", self._state)

        return self._build_result()

    def execute_phase(self, phase: MissionPhase) -> PhaseResult:
        """Execute a single mission phase. Returns PhaseResult."""
        phase_exec = PhaseExecution(
            phase=phase,
            state=ExecutionState.RUNNING,
            start_time=time.time(),
            progress=0.0,
        )
        self._phase_executions.append(phase_exec)
        self._fire_hook("phase_start", phase)

        if self._state == ExecutionState.ABORTED:
            phase_exec.state = ExecutionState.ABORTED
            phase_exec.end_time = time.time()
            return PhaseResult(
                phase_name=phase.name,
                state=ExecutionState.ABORTED,
                error="Mission aborted",
            )

        # Simulate phase execution
        result = PhaseResult(
            phase_name=phase.name,
            state=ExecutionState.RUNNING,
            start_time=phase_exec.start_time,
        )

        try:
            # Execute actions
            for action_idx, action in enumerate(phase.actions):
                if self._state == ExecutionState.ABORTED:
                    result.state = ExecutionState.ABORTED
                    result.error = "Mission aborted during phase"
                    break
                # Update progress
                progress = (action_idx + 1) / max(len(phase.actions), 1)
                result.progress = progress
                phase_exec.progress = progress
                result.results[f"action_{action.name}"] = {
                    "status": "completed",
                    "duration": action.duration,
                }

            if result.state == ExecutionState.RUNNING:
                result.state = ExecutionState.COMPLETED
                # Check success criteria
                result.success_criteria_met = list(phase.success_criteria)

        except Exception as e:
            result.state = ExecutionState.FAILED
            result.error = str(e)
            result.success_criteria_failed = list(phase.success_criteria)

        result.end_time = time.time()
        result.duration = (result.end_time - result.start_time) if result.end_time and result.start_time else 0.0

        phase_exec.state = result.state
        phase_exec.end_time = result.end_time
        phase_exec.results = result.results

        self._fire_hook("phase_end", result)
        return result

    def pause_mission(self) -> bool:
        """Pause the current mission. Returns True if paused."""
        if self._state != ExecutionState.RUNNING:
            return False
        self._state = ExecutionState.PAUSED
        self._paused_at = time.time()
        self._fire_hook("mission_pause", self._current_phase_idx)
        return True

    def resume_mission(self) -> bool:
        """Resume a paused mission. Returns True if resumed."""
        if self._state != ExecutionState.PAUSED:
            return False
        if self._paused_at is not None:
            self._pause_accumulated += time.time() - self._paused_at
            self._paused_at = None
        self._state = ExecutionState.RUNNING
        self._fire_hook("mission_resume", self._current_phase_idx)
        return True

    def abort_mission(self, reason: str = "User abort") -> bool:
        """Abort the current mission. Returns True if aborted."""
        if self._state in (ExecutionState.COMPLETED, ExecutionState.FAILED, ExecutionState.ABORTED):
            return False
        self._state = ExecutionState.ABORTED
        self._abort_reason = reason
        self._end_time = time.time()
        self._fire_hook("mission_abort", reason)
        return True

    def get_current_phase(self) -> Optional[MissionPhase]:
        """Get the currently executing phase."""
        if self._current_plan and 0 <= self._current_phase_idx < len(self._current_plan.phases):
            return self._current_plan.phases[self._current_phase_idx]
        return None

    def get_current_phase_index(self) -> int:
        """Get the current phase index."""
        return self._current_phase_idx

    def handle_phase_transition(self, from_phase: str, to_phase: str) -> TransitionResult:
        """Handle transition between phases. Returns TransitionResult."""
        start = time.time()
        self._fire_hook("phase_transition", (from_phase, to_phase))
        result = TransitionResult(
            from_phase=from_phase,
            to_phase=to_phase,
            success=True,
            transition_time=time.time() - start,
        )
        self._transition_log.append(result)
        return result

    def compute_progress(self) -> float:
        """Compute overall mission progress as a percentage (0.0 - 100.0)."""
        if not self._current_plan or not self._current_plan.phases:
            return 0.0
        total = len(self._current_plan.phases)
        completed = len(self._phase_results)
        if completed == 0:
            return 0.0
        return (completed / total) * 100.0

    def get_state(self) -> ExecutionState:
        """Get current execution state."""
        return self._state

    def get_phase_results(self) -> List[PhaseResult]:
        """Get all phase results."""
        return list(self._phase_results)

    def get_transition_log(self) -> List[TransitionResult]:
        """Get the phase transition log."""
        return list(self._transition_log)

    def register_hook(self, event: str, callback: Callable) -> None:
        """Register a callback for an event."""
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)

    def get_plan(self) -> Optional[MissionPlan]:
        """Get the current plan being executed."""
        return self._current_plan

    def reset(self):
        """Reset executor to initial state."""
        self._state = ExecutionState.IDLE
        self._current_plan = None
        self._current_phase_idx = -1
        self._phase_executions = []
        self._phase_results = []
        self._start_time = None
        self._end_time = None
        self._paused_at = None
        self._pause_accumulated = 0.0
        self._abort_reason = None
        self._transition_log = []

    def _build_result(self) -> MissionResult:
        """Build final MissionResult."""
        elapsed = 0.0
        if self._start_time and self._end_time:
            elapsed = self._end_time - self._start_time - self._pause_accumulated

        return MissionResult(
            plan_id=self._current_plan.id if self._current_plan else "",
            state=self._state,
            total_progress=self.compute_progress(),
            start_time=self._start_time,
            end_time=self._end_time,
            duration=elapsed,
            phase_results=list(self._phase_results),
            completed_phases=[r.phase_name for r in self._phase_results if r.state == ExecutionState.COMPLETED],
            failed_phases=[r.phase_name for r in self._phase_results if r.state == ExecutionState.FAILED],
            aborted=self._state == ExecutionState.ABORTED,
            abort_reason=self._abort_reason,
        )

    def _fire_hook(self, event: str, data: Any) -> None:
        """Fire registered hooks for an event."""
        for cb in self._hooks.get(event, []):
            try:
                cb(data)
            except Exception:
                pass
