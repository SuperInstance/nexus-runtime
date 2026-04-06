"""Tests for CommandExecutor — execution, validation, impact, sequencing, undo, history, planning."""

import pytest
from jetson.nl_commands.intent import Intent, IntentType
from jetson.nl_commands.executor import (
    Command, CommandPriority, ExecutionResult,
    ImpactAssessment, ExecutionStep, CommandExecutor,
)


def _make_intent(itype: IntentType, text: str = "", **slots):
    return Intent(type=itype, slots=slots, confidence=0.9, raw_text=text)


def _make_command(itype: IntentType, text: str = "", params=None, priority=None):
    intent = _make_intent(itype, text)
    cmd = Command(intent=intent, parameters=params or {})
    if priority is not None:
        cmd.priority = priority
    return cmd


@pytest.fixture
def executor():
    return CommandExecutor()


# ===================================================================
# Command dataclass
# ===================================================================

class TestCommand:
    def test_command_creation(self, executor):
        intent = _make_intent(IntentType.NAVIGATE, "go to alpha")
        cmd = Command(intent=intent)
        assert cmd.intent.type == IntentType.NAVIGATE
        assert cmd.priority == CommandPriority.NORMAL
        assert cmd.command_id  # auto-generated

    def test_command_auto_timestamp(self):
        import time
        before = time.time()
        cmd = Command(intent=_make_intent(IntentType.QUERY_STATUS))
        after = time.time()
        assert before <= cmd.timestamp <= after

    def test_command_custom_priority(self):
        cmd = Command(intent=_make_intent(IntentType.EMERGENCY_STOP), priority=CommandPriority.CRITICAL)
        assert cmd.priority == CommandPriority.CRITICAL


# ===================================================================
# ExecutionResult dataclass
# ===================================================================

class TestExecutionResult:
    def test_result_success(self):
        r = ExecutionResult(success=True, message="Done")
        assert r.success
        assert r.message == "Done"

    def test_result_failure(self):
        r = ExecutionResult(success=False, message="Failed")
        assert not r.success


# ===================================================================
# CommandExecutor.execute
# ===================================================================

class TestExecute:
    def test_navigate(self, executor):
        cmd = _make_command(IntentType.NAVIGATE, "go to alpha", {"destination": "alpha"})
        result = executor.execute(cmd)
        assert result.success

    def test_emergency_stop(self, executor):
        cmd = _make_command(IntentType.EMERGENCY_STOP, "emergency stop")
        result = executor.execute(cmd)
        assert result.success
        assert "stop" in result.message.lower()

    def test_station_keep(self, executor):
        cmd = _make_command(IntentType.STATION_KEEP, "hold position")
        result = executor.execute(cmd)
        assert result.success

    def test_patrol(self, executor):
        cmd = _make_command(IntentType.PATROL, "patrol harbor", {"zone": "harbor"})
        result = executor.execute(cmd)
        assert result.success

    def test_survey(self, executor):
        cmd = _make_command(IntentType.SURVEY, "survey area", {"area": "seabed"})
        result = executor.execute(cmd)
        assert result.success

    def test_return_home(self, executor):
        cmd = _make_command(IntentType.RETURN_HOME, "return home")
        result = executor.execute(cmd)
        assert result.success

    def test_set_speed(self, executor):
        cmd = _make_command(IntentType.SET_SPEED, "set speed 5 knots", {"speed": 5})
        result = executor.execute(cmd)
        assert result.success

    def test_set_heading(self, executor):
        cmd = _make_command(IntentType.SET_HEADING, "set heading 90", {"heading_degrees": 90})
        result = executor.execute(cmd)
        assert result.success

    def test_query_status(self, executor):
        cmd = _make_command(IntentType.QUERY_STATUS, "status")
        result = executor.execute(cmd)
        assert result.success

    def test_configure(self, executor):
        cmd = _make_command(IntentType.CONFIGURE, "enable sonar", {"parameter": "sonar"})
        result = executor.execute(cmd)
        assert result.success

    def test_unknown_intent_fails(self, executor):
        cmd = _make_command(IntentType.UNKNOWN, "xyzzy")
        result = executor.execute(cmd)
        assert not result.success

    def test_execution_time_recorded(self, executor):
        cmd = _make_command(IntentType.QUERY_STATUS)
        result = executor.execute(cmd)
        assert result.execution_time >= 0

    def test_data_includes_command_id(self, executor):
        cmd = _make_command(IntentType.NAVIGATE)
        result = executor.execute(cmd)
        assert result.data["command_id"] == cmd.command_id

    def test_side_effects_recorded(self, executor):
        cmd = _make_command(IntentType.NAVIGATE)
        result = executor.execute(cmd)
        assert len(result.side_effects) > 0


# ===================================================================
# CommandExecutor.validate_command
# ===================================================================

class TestValidateCommand:
    def test_valid_command(self, executor):
        cmd = _make_command(IntentType.NAVIGATE)
        result = executor.validate_command(cmd)
        assert result["valid"]

    def test_no_intent(self, executor):
        cmd = Command(intent=None)
        result = executor.validate_command(cmd)
        assert not result["valid"]
        assert any("intent" in e.lower() for e in result["errors"])

    def test_unknown_intent(self, executor):
        cmd = _make_command(IntentType.UNKNOWN)
        result = executor.validate_command(cmd)
        assert not result["valid"]

    def test_negative_speed(self, executor):
        cmd = _make_command(IntentType.SET_SPEED, params={"speed": -5})
        result = executor.validate_command(cmd)
        assert not result["valid"]

    def test_high_speed_warning(self, executor):
        cmd = _make_command(IntentType.SET_SPEED, params={"speed": 60})
        result = executor.validate_command(cmd)
        assert result["valid"]
        assert len(result["warnings"]) > 0

    def test_heading_out_of_range(self, executor):
        cmd = _make_command(IntentType.SET_HEADING, params={"heading_degrees": 400})
        result = executor.validate_command(cmd)
        assert not result["valid"]

    def test_heading_valid_range(self, executor):
        cmd = _make_command(IntentType.SET_HEADING, params={"heading_degrees": 90})
        result = executor.validate_command(cmd)
        assert result["valid"]

    def test_navigation_missing_destination_warning(self, executor):
        cmd = _make_command(IntentType.NAVIGATE)
        result = executor.validate_command(cmd)
        assert result["valid"]
        assert any("destination" in w.lower() for w in result["warnings"])


# ===================================================================
# CommandExecutor.estimate_impact
# ===================================================================

class TestEstimateImpact:
    def test_emergency_stop_impact(self, executor):
        cmd = _make_command(IntentType.EMERGENCY_STOP)
        impact = executor.estimate_impact(cmd)
        assert impact.risk_level == "high"
        assert "propulsion" in impact.affected_systems

    def test_navigate_impact(self, executor):
        cmd = _make_command(IntentType.NAVIGATE)
        impact = executor.estimate_impact(cmd)
        assert impact.risk_level == "medium"
        assert impact.reversible

    def test_query_status_low_risk(self, executor):
        cmd = _make_command(IntentType.QUERY_STATUS)
        impact = executor.estimate_impact(cmd)
        assert impact.risk_level == "low"
        assert impact.energy_cost < 1.0

    def test_return_home_not_reversible(self, executor):
        cmd = _make_command(IntentType.RETURN_HOME)
        impact = executor.estimate_impact(cmd)
        assert not impact.reversible

    def test_impact_has_description(self, executor):
        cmd = _make_command(IntentType.NAVIGATE)
        impact = executor.estimate_impact(cmd)
        assert impact.description != ""

    def test_impact_has_duration(self, executor):
        cmd = _make_command(IntentType.PATROL)
        impact = executor.estimate_impact(cmd)
        assert impact.estimated_duration > 0


# ===================================================================
# CommandExecutor.sequence_commands
# ===================================================================

class TestSequenceCommands:
    def test_emergency_first(self, executor):
        cmds = [
            _make_command(IntentType.NAVIGATE),
            _make_command(IntentType.EMERGENCY_STOP),
            _make_command(IntentType.PATROL),
        ]
        ordered = executor.sequence_commands(cmds)
        assert ordered[0].intent.type == IntentType.EMERGENCY_STOP

    def test_priority_sorting(self, executor):
        cmds = [
            _make_command(IntentType.QUERY_STATUS, priority=CommandPriority.LOW),
            _make_command(IntentType.NAVIGATE, priority=CommandPriority.HIGH),
        ]
        ordered = executor.sequence_commands(cmds)
        assert ordered[0].intent.type == IntentType.NAVIGATE

    def test_empty_list(self, executor):
        assert executor.sequence_commands([]) == []

    def test_single_command(self, executor):
        cmd = _make_command(IntentType.NAVIGATE)
        ordered = executor.sequence_commands([cmd])
        assert len(ordered) == 1

    def test_multiple_emergencies(self, executor):
        cmds = [
            _make_command(IntentType.EMERGENCY_STOP),
            _make_command(IntentType.EMERGENCY_STOP),
            _make_command(IntentType.NAVIGATE),
        ]
        ordered = executor.sequence_commands(cmds)
        assert all(c.intent.type == IntentType.EMERGENCY_STOP for c in ordered[:2])


# ===================================================================
# CommandExecutor.undo_last_command
# ===================================================================

class TestUndoLastCommand:
    def test_undo_success(self, executor):
        cmd = _make_command(IntentType.NAVIGATE)
        executor.execute(cmd)
        result = executor.undo_last_command()
        assert result.success
        assert "undo" in result.message.lower() or "undid" in result.message.lower()

    def test_undo_empty_history(self, executor):
        result = executor.undo_last_command()
        assert not result.success

    def test_undo_emergency_not_allowed(self, executor):
        cmd = _make_command(IntentType.EMERGENCY_STOP)
        executor.execute(cmd)
        result = executor.undo_last_command()
        assert not result.success


# ===================================================================
# CommandExecutor.get_command_history
# ===================================================================

class TestGetCommandHistory:
    def test_empty_history(self, executor):
        history = executor.get_command_history()
        assert history == []

    def test_history_after_execution(self, executor):
        cmd = _make_command(IntentType.NAVIGATE)
        executor.execute(cmd)
        history = executor.get_command_history()
        assert len(history) == 1
        assert history[0]["intent"] == "navigate"

    def test_history_multiple(self, executor):
        executor.execute(_make_command(IntentType.NAVIGATE))
        executor.execute(_make_command(IntentType.STATION_KEEP))
        executor.execute(_make_command(IntentType.QUERY_STATUS))
        history = executor.get_command_history()
        assert len(history) == 3

    def test_history_entry_has_fields(self, executor):
        cmd = _make_command(IntentType.NAVIGATE)
        executor.execute(cmd)
        history = executor.get_command_history()
        entry = history[0]
        assert "index" in entry
        assert "command_id" in entry
        assert "intent" in entry
        assert "success" in entry
        assert "timestamp" in entry


# ===================================================================
# CommandExecutor.compute_execution_plan
# ===================================================================

class TestComputeExecutionPlan:
    def test_navigate_plan(self, executor):
        cmd = _make_command(IntentType.NAVIGATE)
        plan = executor.compute_execution_plan(cmd)
        assert len(plan) >= 3
        assert plan[0].step_number == 1

    def test_emergency_stop_plan(self, executor):
        cmd = _make_command(IntentType.EMERGENCY_STOP)
        plan = executor.compute_execution_plan(cmd)
        assert len(plan) >= 2
        actions = [s.action for s in plan]
        assert "cut_throttle" in actions

    def test_station_keep_plan(self, executor):
        cmd = _make_command(IntentType.STATION_KEEP)
        plan = executor.compute_execution_plan(cmd)
        assert len(plan) >= 2

    def test_patrol_plan(self, executor):
        cmd = _make_command(IntentType.PATROL)
        plan = executor.compute_execution_plan(cmd)
        assert len(plan) >= 2

    def test_survey_plan(self, executor):
        cmd = _make_command(IntentType.SURVEY)
        plan = executor.compute_execution_plan(cmd)
        assert len(plan) >= 2

    def test_return_home_plan(self, executor):
        cmd = _make_command(IntentType.RETURN_HOME)
        plan = executor.compute_execution_plan(cmd)
        assert len(plan) >= 2

    def test_set_speed_plan(self, executor):
        cmd = _make_command(IntentType.SET_SPEED)
        plan = executor.compute_execution_plan(cmd)
        assert len(plan) >= 2

    def test_set_heading_plan(self, executor):
        cmd = _make_command(IntentType.SET_HEADING)
        plan = executor.compute_execution_plan(cmd)
        assert len(plan) >= 2

    def test_query_status_plan(self, executor):
        cmd = _make_command(IntentType.QUERY_STATUS)
        plan = executor.compute_execution_plan(cmd)
        assert len(plan) >= 1

    def test_configure_plan(self, executor):
        cmd = _make_command(IntentType.CONFIGURE)
        plan = executor.compute_execution_plan(cmd)
        assert len(plan) >= 2

    def test_unknown_plan(self, executor):
        cmd = _make_command(IntentType.UNKNOWN)
        plan = executor.compute_execution_plan(cmd)
        assert len(plan) >= 1

    def test_plan_steps_have_descriptions(self, executor):
        cmd = _make_command(IntentType.NAVIGATE)
        plan = executor.compute_execution_plan(cmd)
        for step in plan:
            assert step.description != ""

    def test_plan_steps_sequential(self, executor):
        cmd = _make_command(IntentType.NAVIGATE)
        plan = executor.compute_execution_plan(cmd)
        for i, step in enumerate(plan):
            assert step.step_number == i + 1
