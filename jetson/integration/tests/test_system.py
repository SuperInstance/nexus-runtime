"""Tests for SystemOrchestrator — Phase 5 Round 10."""

import time
import pytest
from jetson.integration.system import (
    SubsystemInfo,
    SubsystemStatus,
    SystemState,
    SystemOrchestrator,
)


@pytest.fixture
def orchestrator():
    return SystemOrchestrator()


@pytest.fixture
def sample_info():
    return SubsystemInfo(
        name="sensor_hub",
        version="1.2.0",
        capabilities=["read_sensors", "calibrate"],
        dependencies=[],
    )


# === SubsystemInfo ===

class TestSubsystemInfo:
    def test_create_default(self):
        info = SubsystemInfo(name="x")
        assert info.name == "x"
        assert info.version == "0.0.0"
        assert info.status == SubsystemStatus.REGISTERED
        assert info.capabilities == []
        assert info.dependencies == []

    def test_create_full(self):
        info = SubsystemInfo(
            name="y", version="2.0.0",
            capabilities=["a"], dependencies=["x"])
        assert info.version == "2.0.0"
        assert info.capabilities == ["a"]
        assert info.dependencies == ["x"]

    def test_metadata_default(self):
        info = SubsystemInfo(name="m")
        assert info.metadata == {}

    def test_metadata_custom(self):
        info = SubsystemInfo(name="m", metadata={"k": "v"})
        assert info.metadata["k"] == "v"

    def test_status_enum_values(self):
        for s in SubsystemStatus:
            assert isinstance(s.value, str)


# === SystemState ===

class TestSystemState:
    def test_default(self):
        st = SystemState()
        assert st.subsystems == {}
        assert st.overall_status == SubsystemStatus.UNKNOWN
        assert st.uptime == 0.0
        assert st.mode == "normal"

    def test_with_data(self):
        st = SystemState(mode="maintenance", uptime=123.4)
        assert st.mode == "maintenance"
        assert st.uptime == 123.4


# === Registration ===

class TestRegistration:
    def test_register(self, orchestrator, sample_info):
        orchestrator.register_subsystem(sample_info)
        assert "sensor_hub" in orchestrator.get_system_state().subsystems

    def test_register_multiple(self, orchestrator):
        for i in range(5):
            orchestrator.register_subsystem(SubsystemInfo(name=f"sub_{i}"))
        state = orchestrator.get_system_state()
        assert len(state.subsystems) == 5

    def test_unregister(self, orchestrator, sample_info):
        orchestrator.register_subsystem(sample_info)
        assert orchestrator.unregister_subsystem("sensor_hub") is True
        assert "sensor_hub" not in orchestrator.get_system_state().subsystems

    def test_unregister_missing(self, orchestrator):
        assert orchestrator.unregister_subsystem("ghost") is False

    def test_reregister(self, orchestrator, sample_info):
        orchestrator.register_subsystem(sample_info)
        new_info = SubsystemInfo(name="sensor_hub", version="2.0.0")
        orchestrator.register_subsystem(new_info)
        assert orchestrator.get_system_state().subsystems["sensor_hub"].version == "2.0.0"


# === Initialization ===

class TestInitialization:
    def test_initialize_single(self, orchestrator):
        orchestrator.register_subsystem(SubsystemInfo(name="solo"))
        success, failures = orchestrator.initialize_all()
        assert success is True
        assert failures == []

    def test_initialize_multiple(self, orchestrator):
        for i in range(4):
            orchestrator.register_subsystem(SubsystemInfo(name=f"s{i}"))
        success, failures = orchestrator.initialize_all()
        assert success is True
        assert len(failures) == 0

    def test_initialize_with_deps(self, orchestrator):
        orchestrator.register_subsystem(SubsystemInfo(name="a"))
        orchestrator.register_subsystem(SubsystemInfo(name="b", dependencies=["a"]))
        success, failures = orchestrator.initialize_all()
        assert success is True

    def test_initialize_missing_dep_fails(self, orchestrator):
        orchestrator.register_subsystem(
            SubsystemInfo(name="b", dependencies=["missing_a"]))
        success, failures = orchestrator.initialize_all()
        assert success is False
        assert "b" in failures

    def test_initialize_partial_failure(self, orchestrator):
        orchestrator.register_subsystem(SubsystemInfo(name="good"))
        orchestrator.register_subsystem(
            SubsystemInfo(name="bad", dependencies=["nonexistent"]))
        success, failures = orchestrator.initialize_all()
        assert success is False
        assert "bad" in failures
        assert "good" not in failures

    def test_initialize_sets_running(self, orchestrator):
        orchestrator.register_subsystem(SubsystemInfo(name="x"))
        orchestrator.initialize_all()
        state = orchestrator.get_system_state()
        assert state.subsystems["x"].status == SubsystemStatus.RUNNING


# === Shutdown ===

class TestShutdown:
    def test_shutdown_all(self, orchestrator):
        orchestrator.register_subsystem(SubsystemInfo(name="a"))
        orchestrator.register_subsystem(SubsystemInfo(name="b"))
        orchestrator.initialize_all()
        stopped = orchestrator.shutdown_all()
        assert "a" in stopped
        assert "b" in stopped

    def test_shutdown_sets_stopped(self, orchestrator):
        orchestrator.register_subsystem(SubsystemInfo(name="x"))
        orchestrator.initialize_all()
        orchestrator.shutdown_all()
        assert orchestrator.get_system_state().subsystems["x"].status == SubsystemStatus.STOPPED

    def test_shutdown_empty(self, orchestrator):
        assert orchestrator.shutdown_all() == []

    def test_shutdown_no_running(self, orchestrator):
        orchestrator.register_subsystem(SubsystemInfo(name="x"))
        # Not initialized, so not running
        assert orchestrator.shutdown_all() == []


# === Start / Stop / Restart ===

class TestLifecycle:
    def test_start_subsystem(self, orchestrator):
        orchestrator.register_subsystem(SubsystemInfo(name="x"))
        assert orchestrator.start_subsystem("x") is True
        assert orchestrator.get_system_state().subsystems["x"].status == SubsystemStatus.RUNNING

    def test_start_missing(self, orchestrator):
        assert orchestrator.start_subsystem("ghost") is False

    def test_start_with_unmet_deps(self, orchestrator):
        orchestrator.register_subsystem(
            SubsystemInfo(name="b", dependencies=["a"]))
        assert orchestrator.start_subsystem("b") is False

    def test_stop_subsystem(self, orchestrator):
        orchestrator.register_subsystem(SubsystemInfo(name="x"))
        orchestrator.start_subsystem("x")
        assert orchestrator.stop_subsystem("x") is True

    def test_stop_not_running(self, orchestrator):
        orchestrator.register_subsystem(SubsystemInfo(name="x"))
        assert orchestrator.stop_subsystem("x") is False

    def test_stop_missing(self, orchestrator):
        assert orchestrator.stop_subsystem("ghost") is False

    def test_restart_subsystem(self, orchestrator):
        orchestrator.register_subsystem(SubsystemInfo(name="x"))
        orchestrator.start_subsystem("x")
        assert orchestrator.restart_subsystem("x") is True
        assert orchestrator.get_system_state().subsystems["x"].status == SubsystemStatus.RUNNING

    def test_restart_missing_dep(self, orchestrator):
        orchestrator.register_subsystem(
            SubsystemInfo(name="b", dependencies=["a"]))
        assert orchestrator.restart_subsystem("b") is False


# === System State ===

class TestSystemStateQueries:
    def test_empty_state(self, orchestrator):
        st = orchestrator.get_system_state()
        assert st.overall_status == SubsystemStatus.UNKNOWN

    def test_all_running(self, orchestrator):
        orchestrator.register_subsystem(SubsystemInfo(name="a"))
        orchestrator.register_subsystem(SubsystemInfo(name="b"))
        orchestrator.initialize_all()
        st = orchestrator.get_system_state()
        assert st.overall_status == SubsystemStatus.RUNNING

    def test_uptime_after_init(self, orchestrator):
        orchestrator.register_subsystem(SubsystemInfo(name="a"))
        orchestrator.initialize_all()
        st = orchestrator.get_system_state()
        assert st.uptime >= 0

    def test_mode(self, orchestrator):
        orchestrator.set_mode("maintenance")
        st = orchestrator.get_system_state()
        assert st.mode == "maintenance"

    def test_uptime_increases(self, orchestrator):
        orchestrator.register_subsystem(SubsystemInfo(name="a"))
        orchestrator.initialize_all()
        t1 = orchestrator.get_system_state().uptime
        time.sleep(0.05)
        t2 = orchestrator.get_system_state().uptime
        assert t2 > t1


# === Dependencies ===

class TestDependencies:
    def test_no_missing(self, orchestrator):
        orchestrator.register_subsystem(SubsystemInfo(name="a"))
        orchestrator.register_subsystem(SubsystemInfo(name="b", dependencies=["a"]))
        assert orchestrator.check_dependencies() == []

    def test_missing_deps(self, orchestrator):
        orchestrator.register_subsystem(
            SubsystemInfo(name="x", dependencies=["ghost"]))
        missing = orchestrator.check_dependencies()
        assert len(missing) == 1
        assert "ghost" in missing[0]

    def test_dependency_graph(self, orchestrator):
        orchestrator.register_subsystem(SubsystemInfo(name="a"))
        orchestrator.register_subsystem(SubsystemInfo(name="b", dependencies=["a"]))
        graph = orchestrator.get_dependency_graph()
        assert graph["a"] == []
        assert graph["b"] == ["a"]

    def test_empty_graph(self, orchestrator):
        assert orchestrator.get_dependency_graph() == {}


# === Hooks ===

class TestHooks:
    def test_add_and_fire(self, orchestrator):
        results = []
        orchestrator.add_hook("on_init", lambda: results.append("fired"))
        orchestrator.fire_hook("on_init")
        assert results == ["fired"]

    def test_fire_empty(self, orchestrator):
        assert orchestrator.fire_hook("nope") == []

    def test_multiple_hooks(self, orchestrator):
        results = []
        orchestrator.add_hook("ev", lambda: results.append(1))
        orchestrator.add_hook("ev", lambda: results.append(2))
        orchestrator.fire_hook("ev")
        assert results == [1, 2]

    def test_hook_with_args(self, orchestrator):
        def cb(x, y):
            return x + y
        orchestrator.add_hook("calc", cb)
        res = orchestrator.fire_hook("calc", 3, 4)
        assert res == [7]
