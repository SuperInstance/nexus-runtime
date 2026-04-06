"""Tests for mission execution engine."""

import pytest
from jetson.mission.planner import (
    MissionObjective, MissionPlan, MissionPhase, MissionAction,
)
from jetson.mission.execution import (
    ExecutionState,
    PhaseResult,
    PhaseExecution,
    MissionResult,
    TransitionResult,
    MissionExecutor,
)


def _make_plan(num_phases=2, actions_per_phase=1):
    """Helper to create a test plan."""
    phases = []
    for i in range(num_phases):
        actions = [
            MissionAction(name=f"act_{i}_{j}", duration=1.0)
            for j in range(actions_per_phase)
        ]
        phase = MissionPhase(
            name=f"phase_{i}",
            actions=actions,
            duration=10.0,
            dependencies=[phases[-1].name] if phases else [],
            success_criteria=[f"phase_{i}_done"],
        )
        phases.append(phase)
    objectives = [MissionObjective(name=f"obj_{i}", type="survey") for i in range(num_phases)]
    return MissionPlan(name="test_plan", objectives=objectives, phases=phases)


class TestExecutionState:
    def test_idle(self):
        assert ExecutionState.IDLE.value == "idle"

    def test_running(self):
        assert ExecutionState.RUNNING.value == "running"

    def test_paused(self):
        assert ExecutionState.PAUSED.value == "paused"

    def test_completed(self):
        assert ExecutionState.COMPLETED.value == "completed"

    def test_aborted(self):
        assert ExecutionState.ABORTED.value == "aborted"

    def test_failed(self):
        assert ExecutionState.FAILED.value == "failed"


class TestPhaseResult:
    def test_default_creation(self):
        r = PhaseResult()
        assert r.state == ExecutionState.IDLE
        assert r.progress == 0.0

    def test_custom_creation(self):
        r = PhaseResult(
            phase_name="p1", state=ExecutionState.COMPLETED,
            progress=100.0, error=None,
        )
        assert r.phase_name == "p1"
        assert r.state == ExecutionState.COMPLETED


class TestPhaseExecution:
    def test_default_creation(self):
        pe = PhaseExecution()
        assert pe.state == ExecutionState.IDLE
        assert pe.progress == 0.0


class TestMissionResult:
    def test_default_creation(self):
        mr = MissionResult()
        assert mr.state == ExecutionState.IDLE
        assert mr.total_progress == 0.0
        assert mr.aborted is False


class TestTransitionResult:
    def test_default_creation(self):
        tr = TransitionResult()
        assert tr.success is True
        assert tr.from_phase == ""

    def test_custom_creation(self):
        tr = TransitionResult(from_phase="a", to_phase="b", success=False, error="fail")
        assert tr.from_phase == "a"
        assert tr.to_phase == "b"
        assert tr.success is False


class TestMissionExecutor:
    def setup_method(self):
        self.executor = MissionExecutor()

    def test_initial_state(self):
        assert self.executor.get_state() == ExecutionState.IDLE

    def test_execute_plan_basic(self):
        plan = _make_plan()
        result = self.executor.execute_plan(plan)
        assert result.state == ExecutionState.COMPLETED
        assert len(result.completed_phases) == 2

    def test_execute_plan_single_phase(self):
        plan = _make_plan(num_phases=1)
        result = self.executor.execute_plan(plan)
        assert result.state == ExecutionState.COMPLETED
        assert len(result.phase_results) == 1

    def test_execute_plan_empty(self):
        plan = MissionPlan(name="empty", objectives=[], phases=[])
        result = self.executor.execute_plan(plan)
        assert result.state == ExecutionState.COMPLETED

    def test_execute_plan_sets_plan(self):
        plan = _make_plan()
        self.executor.execute_plan(plan)
        assert self.executor.get_plan() is not None
        assert self.executor.get_plan().name == "test_plan"

    def test_execute_phase_basic(self):
        phase = MissionPhase(
            name="p1",
            actions=[MissionAction(name="a1", duration=1.0)],
            success_criteria=["done"],
        )
        result = self.executor.execute_phase(phase)
        assert result.state == ExecutionState.COMPLETED
        assert "done" in result.success_criteria_met

    def test_execute_phase_empty_actions(self):
        phase = MissionPhase(name="p1", success_criteria=["done"])
        result = self.executor.execute_phase(phase)
        assert result.state == ExecutionState.COMPLETED

    def test_execute_phase_updates_progress(self):
        actions = [
            MissionAction(name="a1", duration=1.0),
            MissionAction(name="a2", duration=1.0),
        ]
        phase = MissionPhase(name="p1", actions=actions, success_criteria=["done"])
        result = self.executor.execute_phase(phase)
        assert result.progress == 1.0

    def test_pause_mission(self):
        plan = _make_plan(num_phases=3, actions_per_phase=1)
        # Can't easily pause mid-execution in synchronous mode,
        # so test the state machine directly
        assert self.executor.pause_mission() is False  # Can't pause IDLE

    def test_pause_running(self):
        """Test pause state machine by setting state directly."""
        self.executor._state = ExecutionState.RUNNING
        assert self.executor.pause_mission() is True
        assert self.executor.get_state() == ExecutionState.PAUSED

    def test_resume_paused(self):
        self.executor._state = ExecutionState.PAUSED
        assert self.executor.resume_mission() is True
        assert self.executor.get_state() == ExecutionState.RUNNING

    def test_resume_not_paused(self):
        self.executor._state = ExecutionState.RUNNING
        assert self.executor.resume_mission() is False

    def test_abort_mission(self):
        self.executor._state = ExecutionState.RUNNING
        assert self.executor.abort_mission("test") is True
        assert self.executor.get_state() == ExecutionState.ABORTED

    def test_abort_idle(self):
        assert self.executor.abort_mission() is True

    def test_abort_completed(self):
        self.executor._state = ExecutionState.COMPLETED
        assert self.executor.abort_mission() is False

    def test_abort_failed(self):
        self.executor._state = ExecutionState.FAILED
        assert self.executor.abort_mission() is False

    def test_get_current_phase_none(self):
        assert self.executor.get_current_phase() is None

    def test_get_current_phase_during(self):
        plan = _make_plan(num_phases=3)
        self.executor.execute_plan(plan)
        # After execution, current_phase_idx is at the last phase
        assert self.executor.get_current_phase_index() >= 0

    def test_handle_phase_transition(self):
        result = self.executor.handle_phase_transition("phase_0", "phase_1")
        assert result.success is True
        assert result.from_phase == "phase_0"
        assert result.to_phase == "phase_1"
        assert result.transition_time >= 0

    def test_compute_progress_empty(self):
        assert self.executor.compute_progress() == 0.0

    def test_compute_progress_after_execution(self):
        plan = _make_plan(num_phases=4)
        self.executor.execute_plan(plan)
        progress = self.executor.compute_progress()
        assert progress == 100.0

    def test_compute_progress_partial(self):
        plan = _make_plan(num_phases=4)
        self.executor.execute_plan(plan)
        # Mock partial progress
        self.executor._phase_results = self.executor._phase_results[:2]
        progress = self.executor.compute_progress()
        assert progress == 50.0

    def test_get_phase_results(self):
        plan = _make_plan()
        self.executor.execute_plan(plan)
        results = self.executor.get_phase_results()
        assert len(results) == 2

    def test_get_transition_log(self):
        self.executor.handle_phase_transition("a", "b")
        log = self.executor.get_transition_log()
        assert len(log) == 1
        assert log[0].from_phase == "a"

    def test_register_hook(self):
        calls = []
        self.executor.register_hook("mission_start", lambda p: calls.append(p))
        plan = _make_plan()
        self.executor.execute_plan(plan)
        assert len(calls) == 1

    def test_register_hook_phase_start(self):
        calls = []
        self.executor.register_hook("phase_start", lambda p: calls.append(p))
        plan = _make_plan(num_phases=2)
        self.executor.execute_plan(plan)
        assert len(calls) == 2

    def test_register_hook_phase_end(self):
        calls = []
        self.executor.register_hook("phase_end", lambda r: calls.append(r))
        plan = _make_plan(num_phases=3)
        self.executor.execute_plan(plan)
        assert len(calls) == 3

    def test_register_hook_mission_end(self):
        calls = []
        self.executor.register_hook("mission_end", lambda s: calls.append(s))
        plan = _make_plan()
        self.executor.execute_plan(plan)
        assert len(calls) == 1
        assert calls[0] == ExecutionState.COMPLETED

    def test_reset(self):
        plan = _make_plan()
        self.executor.execute_plan(plan)
        self.executor.reset()
        assert self.executor.get_state() == ExecutionState.IDLE
        assert self.executor.get_plan() is None
        assert self.executor.compute_progress() == 0.0

    def test_mission_result_duration(self):
        plan = _make_plan()
        result = self.executor.execute_plan(plan)
        assert result.duration >= 0

    def test_mission_result_plan_id(self):
        plan = _make_plan()
        result = self.executor.execute_plan(plan)
        assert result.plan_id == plan.id

    def test_execute_plan_with_many_actions(self):
        actions = [MissionAction(name=f"a_{i}", duration=0.1) for i in range(10)]
        phase = MissionPhase(name="busy", actions=actions, success_criteria=["done"])
        plan = MissionPlan(
            objectives=[MissionObjective()],
            phases=[phase],
        )
        result = self.executor.execute_plan(plan)
        assert result.state == ExecutionState.COMPLETED
        assert result.phase_results[0].progress == 1.0

    def test_pause_resume_timing(self):
        self.executor._state = ExecutionState.RUNNING
        self.executor.pause_mission()
        # Manually set _paused_at to simulate pause
        import time
        self.executor._paused_at = time.time() - 0.1
        self.executor.resume_mission()
        assert self.executor._pause_accumulated >= 0.0

    def test_hook_exception_does_not_propagate(self):
        def bad_hook(data):
            raise RuntimeError("hook error")
        self.executor.register_hook("mission_start", bad_hook)
        plan = _make_plan()
        result = self.executor.execute_plan(plan)
        assert result.state == ExecutionState.COMPLETED
