"""Tests for BootstrapManager — Phase 5 Round 10."""

import pytest
from jetson.integration.bootstrap import (
    BootstrapPhase,
    BootstrapStep,
    BootstrapManager,
    BootstrapResult,
)


def _ok_step():
    return lambda: True


def _fail_step():
    return lambda: False


def _error_step():
    return lambda: (_ for _ in ()).throw(RuntimeError("err"))


@pytest.fixture
def manager():
    return BootstrapManager()


@pytest.fixture
def full_sequence():
    mgr = BootstrapManager()
    mgr.add_step(BootstrapStep(name="init_core", phase=BootstrapPhase.CORE,
                               action_fn=lambda: True, required=True))
    mgr.add_step(BootstrapStep(name="load_config", phase=BootstrapPhase.CORE,
                               action_fn=lambda: True, required=True))
    mgr.add_step(BootstrapStep(name="start_http", phase=BootstrapPhase.SERVICES,
                               action_fn=lambda: True, required=True))
    mgr.add_step(BootstrapStep(name="start_ws", phase=BootstrapPhase.SERVICES,
                               action_fn=lambda: False, required=False))
    mgr.add_step(BootstrapStep(name="spawn_agents", phase=BootstrapPhase.AGENTS,
                               action_fn=lambda: True, required=True))
    mgr.add_step(BootstrapStep(name="run_integration", phase=BootstrapPhase.INTEGRATION,
                               action_fn=lambda: True, required=True))
    mgr.add_step(BootstrapStep(name="ready_signal", phase=BootstrapPhase.READY,
                               action_fn=lambda: True, required=True))
    return mgr


# === BootstrapPhase ===

class TestBootstrapPhase:
    def test_values(self):
        phases = list(BootstrapPhase)
        assert len(phases) == 5
        assert BootstrapPhase.CORE.value == "core"
        assert BootstrapPhase.READY.value == "ready"

    def test_order(self):
        order = BootstrapManager.PHASE_ORDER
        assert order[0] == BootstrapPhase.CORE
        assert order[-1] == BootstrapPhase.READY


# === BootstrapStep ===

class TestBootstrapStep:
    def test_defaults(self):
        step = BootstrapStep(name="x", phase=BootstrapPhase.CORE,
                             action_fn=lambda: True)
        assert step.timeout == 30.0
        assert step.required is True
        assert step.category == "general"

    def test_custom(self):
        step = BootstrapStep(name="y", phase=BootstrapPhase.SERVICES,
                             action_fn=lambda: False, timeout=10.0,
                             required=False, category="net")
        assert step.timeout == 10.0
        assert step.required is False
        assert step.category == "net"


# === Step Management ===

class TestStepManagement:
    def test_add_step(self, manager):
        manager.add_step(BootstrapStep(name="a", phase=BootstrapPhase.CORE,
                                       action_fn=_ok_step()))
        assert len(manager.get_boot_sequence()) == 1

    def test_add_multiple(self, manager):
        for i in range(5):
            manager.add_step(BootstrapStep(name=f"s{i}", phase=BootstrapPhase.CORE,
                                           action_fn=_ok_step()))
        assert len(manager.get_boot_sequence()) == 5

    def test_remove_step(self, manager):
        manager.add_step(BootstrapStep(name="a", phase=BootstrapPhase.CORE,
                                       action_fn=_ok_step()))
        assert manager.remove_step("a") is True
        assert len(manager.get_boot_sequence()) == 0

    def test_remove_missing(self, manager):
        assert manager.remove_step("ghost") is False

    def test_boot_sequence_order(self, manager):
        manager.add_step(BootstrapStep(name="b", phase=BootstrapPhase.SERVICES,
                                       action_fn=_ok_step()))
        manager.add_step(BootstrapStep(name="a", phase=BootstrapPhase.CORE,
                                       action_fn=_ok_step()))
        seq = manager.get_boot_sequence()
        assert seq[0].name == "a"
        assert seq[1].name == "b"

    def test_steps_for_phase(self, manager):
        manager.add_step(BootstrapStep(name="a", phase=BootstrapPhase.CORE,
                                       action_fn=_ok_step()))
        manager.add_step(BootstrapStep(name="b", phase=BootstrapPhase.SERVICES,
                                       action_fn=_ok_step()))
        assert len(manager.get_steps_for_phase(BootstrapPhase.CORE)) == 1
        assert len(manager.get_steps_for_phase(BootstrapPhase.READY)) == 0


# === Bootstrap Execution ===

class TestBootstrap:
    def test_all_pass(self, full_sequence):
        result = full_sequence.run_bootstrap()
        assert result.success is True
        assert result.steps_completed == 7

    def test_required_failure(self, manager):
        manager.add_step(BootstrapStep(name="bad", phase=BootstrapPhase.CORE,
                                       action_fn=_fail_step(), required=True))
        result = manager.run_bootstrap()
        assert result.success is False
        assert "bad" in result.failures

    def test_optional_failure_still_passes(self, manager):
        manager.add_step(BootstrapStep(name="opt", phase=BootstrapPhase.CORE,
                                       action_fn=_fail_step(), required=False))
        result = manager.run_bootstrap()
        assert result.success is True

    def test_error_in_step(self, manager):
        manager.add_step(BootstrapStep(name="err", phase=BootstrapPhase.CORE,
                                       action_fn=_error_step(), required=True))
        result = manager.run_bootstrap()
        assert result.success is False
        assert "err" in result.failures

    def test_result_phase(self, full_sequence):
        result = full_sequence.run_bootstrap()
        assert result.phase == BootstrapPhase.READY

    def test_result_duration(self, full_sequence):
        result = full_sequence.run_bootstrap()
        assert result.duration_s >= 0

    def test_empty_bootstrap(self, manager):
        result = manager.run_bootstrap()
        assert result.success is True
        assert result.steps_total == 0


# === Environment Validation ===

class TestValidation:
    def test_valid_env(self, manager):
        manager.add_step(BootstrapStep(name="check", phase=BootstrapPhase.CORE,
                                       action_fn=_ok_step(), required=True))
        valid, missing = manager.validate_environment()
        assert valid is True
        assert missing == []

    def test_invalid_env(self, manager):
        manager.add_step(BootstrapStep(name="check", phase=BootstrapPhase.CORE,
                                       action_fn=_fail_step(), required=True))
        valid, missing = manager.validate_environment()
        assert valid is False
        assert len(missing) == 1

    def test_error_env(self, manager):
        manager.add_step(BootstrapStep(name="check", phase=BootstrapPhase.CORE,
                                       action_fn=_error_step(), required=True))
        valid, missing = manager.validate_environment()
        assert valid is False

    def test_non_core_ignored(self, manager):
        # Non-CORE required steps should not be validated as env deps
        manager.add_step(BootstrapStep(name="svc", phase=BootstrapPhase.SERVICES,
                                       action_fn=_fail_step(), required=True))
        valid, missing = manager.validate_environment()
        assert valid is True


# === Rollback ===

class TestRollback:
    def test_rollback_handler(self, manager):
        rolled_back = []
        manager.register_rollback_handler(
            BootstrapPhase.CORE, lambda: rolled_back.append("core"))
        manager.register_rollback_handler(
            BootstrapPhase.SERVICES, lambda: rolled_back.append("services"))
        manager.rollback(BootstrapPhase.SERVICES)
        assert "services" in rolled_back
        assert "core" in rolled_back

    def test_rollback_partial(self, manager):
        rolled_back = []
        manager.register_rollback_handler(
            BootstrapPhase.CORE, lambda: rolled_back.append("core"))
        manager.rollback(BootstrapPhase.CORE)
        assert rolled_back == ["core"]

    def test_rollback_error(self, manager):
        def bad_handler():
            raise ValueError("oops")
        manager.register_rollback_handler(BootstrapPhase.CORE, bad_handler)
        # Should not raise
        manager.rollback(BootstrapPhase.CORE)
        log = manager.get_bootstrap_log()
        assert any("Rollback error" in l for l in log)

    def test_rollback_log(self, manager):
        manager.register_rollback_handler(
            BootstrapPhase.CORE, lambda: None)
        manager.rollback(BootstrapPhase.CORE)
        log = manager.get_bootstrap_log()
        assert any("Rolled back" in l for l in log)


# === Introspection ===

class TestIntrospection:
    def test_bootstrap_log(self, full_sequence):
        full_sequence.run_bootstrap()
        log = full_sequence.get_bootstrap_log()
        assert len(log) > 0

    def test_log_contains_phases(self, full_sequence):
        full_sequence.run_bootstrap()
        log = full_sequence.get_bootstrap_log()
        assert any("core" in l for l in log)
        assert any("services" in l for l in log)

    def test_completed_phases(self, full_sequence):
        full_sequence.run_bootstrap()
        phases = full_sequence.get_completed_phases()
        assert BootstrapPhase.CORE in phases
        assert BootstrapPhase.READY in phases

    def test_reset(self, manager):
        manager.add_step(BootstrapStep(name="a", phase=BootstrapPhase.CORE,
                                       action_fn=_ok_step()))
        manager.run_bootstrap()
        manager.reset()
        assert manager.get_bootstrap_log() == []
        assert manager.get_completed_phases() == []


# === BootstrapResult ===

class TestBootstrapResult:
    def test_result_fields(self):
        r = BootstrapResult(success=True, phase=BootstrapPhase.READY,
                            steps_completed=5, steps_total=5)
        assert r.failures == []
        assert r.duration_s == 0.0
        assert r.log == []

    def test_result_with_failures(self):
        r = BootstrapResult(success=False, phase=BootstrapPhase.CORE,
                            failures=["step1", "step2"])
        assert len(r.failures) == 2
