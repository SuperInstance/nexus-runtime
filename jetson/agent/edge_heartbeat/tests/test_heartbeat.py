"""NEXUS Edge Heartbeat — Comprehensive tests.

Tests cover:
  1. Config loading and defaults
  2. VesselState tracking and serialization
  3. Mission loading and execution
  4. Heartbeat cycle execution (all 5 phases)
  5. .agent/next and .agent/done management
  6. Error handling (mission failure, sensor timeout)
  7. Bytecode generation for missions
"""

import json
from pathlib import Path
import os
import tempfile
import time

import pytest

from agent.edge_heartbeat.config import (
    HeartbeatConfig,
    config_to_json,
    default_config,
    load_config,
    save_config,
)
from agent.edge_heartbeat.heartbeat import (
    EdgeHeartbeat,
    HeartbeatPhase,
    HeartbeatResult,
    PhaseResult,
)
from agent.edge_heartbeat.mission_runner import (
    Mission,
    MissionParseError,
    MissionResult,
    MissionRunner,
    MissionStatus,
    MissionType,
    append_done,
    pop_next_mission,
    read_next_queue,
)
from agent.edge_heartbeat.vessel_state import (
    VesselState,
    VesselStateManager,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def tmp_repo(tmp_path):
    """Create a temporary repo directory with .agent structure."""
    agent_dir = tmp_path / ".agent"
    agent_dir.mkdir()
    (agent_dir / "next").write_text("", encoding="utf-8")
    (agent_dir / "done").write_text("", encoding="utf-8")
    (agent_dir / "identity").write_text("", encoding="utf-8")
    return tmp_path


@pytest.fixture
def config_dict(tmp_repo):
    """Standard config dict for testing."""
    return {
        "vessel_id": "test-vessel-42",
        "repo_path": str(tmp_repo),
        "heartbeat_interval": 60,
        "telemetry_batch_size": 50,
        "trust_reporting": True,
        "serial_port": "/dev/ttyTEST0",
        "serial_baud": 9600,
    }


@pytest.fixture
def config_file(tmp_repo, config_dict):
    """Write a config file and return its path."""
    cfg_path = tmp_repo / "vessel_config.json"
    with open(cfg_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    return str(cfg_path)


@pytest.fixture
def heartbeat(config_file, config_dict):
    """Create an EdgeHeartbeat with test config."""
    cfg = HeartbeatConfig(**config_dict)
    return EdgeHeartbeat(config=cfg)


@pytest.fixture
def state_manager():
    """Fresh VesselStateManager."""
    return VesselStateManager(vessel_id="test-vessel-42")


@pytest.fixture
def mission_runner(tmp_repo):
    """Fresh MissionRunner with temp repo."""
    return MissionRunner(repo_path=str(tmp_repo))


# ===================================================================
# Test 1: Config Defaults and Loading
# ===================================================================

class TestConfigDefaults:
    def test_default_config_values(self):
        cfg = default_config()
        assert cfg.vessel_id == "nexus-vessel-001"
        assert cfg.repo_path == "/opt/nexus-runtime"
        assert cfg.heartbeat_interval == 300
        assert cfg.telemetry_batch_size == 100
        assert cfg.trust_reporting is True
        assert cfg.github_token is None
        assert cfg.github_repo is None
        assert cfg.serial_port == "/dev/ttyUSB0"
        assert cfg.serial_baud == 115200
        assert cfg.max_mission_retries == 3
        assert cfg.mission_timeout_seconds == 120
        assert cfg.log_level == "INFO"

    def test_default_config_is_heartbeat_config(self):
        assert isinstance(default_config(), HeartbeatConfig)

    def test_config_all_fields_set(self):
        cfg = default_config()
        # Ensure no field is None (except github_token/repo which can be None)
        for fld_name, fld_val in cfg.__dict__.items():
            if fld_name in ("github_token", "github_repo"):
                assert fld_val is None
            else:
                assert fld_val is not None and fld_val != "", (
                    f"Field {fld_name} should not be empty"
                )


class TestConfigLoading:
    def test_load_from_json_file(self, config_file, config_dict):
        cfg = load_config(config_file)
        assert cfg.vessel_id == "test-vessel-42"
        assert cfg.heartbeat_interval == 60
        assert cfg.serial_port == "/dev/ttyTEST0"
        assert cfg.serial_baud == 9600

    def test_load_partial_config_uses_defaults(self, tmp_repo):
        partial = {"vessel_id": "partial-vessel"}
        cfg_path = tmp_repo / "partial.json"
        with open(cfg_path, "w") as f:
            json.dump(partial, f)
        cfg = load_config(str(cfg_path))
        assert cfg.vessel_id == "partial-vessel"
        assert cfg.heartbeat_interval == 300  # default
        assert cfg.serial_port == "/dev/ttyUSB0"  # default

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/path/config.json")

    def test_load_invalid_json_raises(self, tmp_repo):
        bad_path = tmp_repo / "bad.json"
        bad_path.write_text("not json {{{", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_config(str(bad_path))

    def test_save_and_reload(self, tmp_repo):
        cfg = HeartbeatConfig(vessel_id="save-test", heartbeat_interval=10)
        save_path = str(tmp_repo / "saved.json")
        save_config(cfg, save_path)
        loaded = load_config(save_path)
        assert loaded.vessel_id == "save-test"
        assert loaded.heartbeat_interval == 10


class TestConfigJson:
    def test_config_to_json(self):
        cfg = HeartbeatConfig(vessel_id="json-test")
        result = config_to_json(cfg)
        data = json.loads(result)
        assert data["vessel_id"] == "json-test"
        assert data["heartbeat_interval"] == 300

    def test_config_to_json_is_valid(self):
        result = config_to_json(default_config())
        json.loads(result)  # Should not raise


# ===================================================================
# Test 2: VesselState Tracking and Serialization
# ===================================================================

class TestVesselStateInit:
    def test_initial_state(self, state_manager):
        s = state_manager.state
        assert s.vessel_id == "test-vessel-42"
        assert s.uptime_seconds >= 0.0
        assert s.trust_scores == {}
        assert s.autonomy_level == 0
        assert s.sensor_status == {}
        assert s.current_mission is None
        assert s.pending_missions == []
        assert s.error_count == 0
        assert s.mission_count == 0
        assert s.mission_failures == 0

    def test_custom_vessel_id(self):
        mgr = VesselStateManager(vessel_id="custom-id")
        assert mgr.vessel_id == "custom-id"


class TestVesselStateSensorUpdates:
    def test_update_valid_sensors(self, state_manager):
        state_manager.update_from_sensors({"gps": "ok", "sonar": "degraded", "lidar": "offline"})
        assert state_manager.state.sensor_status == {"gps": "ok", "sonar": "degraded", "lidar": "offline"}

    def test_update_invalid_status_ignored(self, state_manager):
        state_manager.update_from_sensors({"gps": "ok", "bad_sensor": "exploded"})
        assert "bad_sensor" not in state_manager.state.sensor_status
        assert state_manager.state.sensor_status["gps"] == "ok"

    def test_update_overwrites_previous(self, state_manager):
        state_manager.update_from_sensors({"gps": "ok"})
        state_manager.update_from_sensors({"gps": "degraded"})
        assert state_manager.state.sensor_status["gps"] == "degraded"


class TestVesselStateTrust:
    def test_update_trust_scores(self, state_manager):
        state_manager.update_from_trust({"steering": 0.85, "engine": 0.92})
        assert state_manager.state.trust_scores["steering"] == 0.85
        assert state_manager.state.trust_scores["engine"] == 0.92

    def test_update_trust_merges(self, state_manager):
        state_manager.update_from_trust({"steering": 0.5})
        state_manager.update_from_trust({"engine": 0.7})
        assert len(state_manager.state.trust_scores) == 2


class TestVesselStateAutonomy:
    def test_set_autonomy_level(self, state_manager):
        state_manager.set_autonomy_level(3)
        assert state_manager.state.autonomy_level == 3

    def test_set_autonomy_clamps_high(self, state_manager):
        state_manager.set_autonomy_level(10)
        assert state_manager.state.autonomy_level == 5

    def test_set_autonomy_clamps_low(self, state_manager):
        state_manager.set_autonomy_level(-5)
        assert state_manager.state.autonomy_level == 0


class TestVesselStateMissions:
    def test_set_current_mission(self, state_manager):
        state_manager.set_current_mission("deploy_reflex:heading_hold")
        assert state_manager.state.current_mission == "deploy_reflex:heading_hold"

    def test_clear_current_mission(self, state_manager):
        state_manager.set_current_mission("some-mission")
        state_manager.set_current_mission(None)
        assert state_manager.state.current_mission is None

    def test_set_pending_missions(self, state_manager):
        state_manager.set_pending_missions(["m1", "m2", "m3"])
        assert state_manager.state.pending_missions == ["m1", "m2", "m3"]

    def test_record_mission_complete_success(self, state_manager):
        state_manager.record_mission_complete(success=True)
        assert state_manager.state.mission_count == 1
        assert state_manager.state.mission_failures == 0

    def test_record_mission_complete_failure(self, state_manager):
        state_manager.record_mission_complete(success=False)
        assert state_manager.state.mission_count == 1
        assert state_manager.state.mission_failures == 1

    def test_record_error(self, state_manager):
        state_manager.record_error()
        state_manager.record_error()
        assert state_manager.state.error_count == 2


class TestVesselStateHeartbeat:
    def test_record_heartbeat_updates_time(self, state_manager):
        t1 = state_manager.state.last_heartbeat
        time.sleep(0.01)
        state_manager.record_heartbeat()
        t2 = state_manager.state.last_heartbeat
        assert t2 > t1

    def test_uptime_increases(self, state_manager):
        time.sleep(0.01)
        state_manager.record_heartbeat()
        assert state_manager.state.uptime_seconds > 0.0


class TestVesselStateStatusReport:
    def test_all_clear_status(self, state_manager):
        report = state_manager.get_status_report()
        assert report["status_level"] == "ALL_CLEAR"
        assert report["vessel_id"] == "test-vessel-42"

    def test_alert_on_offline_sensor(self, state_manager):
        state_manager.update_from_sensors({"gps": "offline"})
        report = state_manager.get_status_report()
        assert report["status_level"] == "ALERT"
        assert report["sensors"]["offline"] == 1

    def test_attention_on_degraded(self, state_manager):
        state_manager.update_from_sensors({"gps": "degraded"})
        report = state_manager.get_status_report()
        assert report["status_level"] == "ATTENTION"

    def test_executing_when_mission_active(self, state_manager):
        state_manager.update_from_sensors({"gps": "ok"})
        state_manager.set_current_mission("deploy:heading")
        report = state_manager.get_status_report()
        assert report["status_level"] == "EXECUTING"

    def test_alert_on_low_trust(self, state_manager):
        state_manager.update_from_trust({"steering": 0.1})
        report = state_manager.get_status_report()
        assert report["status_level"] == "ALERT"

    def test_report_has_trust_summary(self, state_manager):
        state_manager.update_from_trust({"steering": 0.5, "engine": 0.8})
        report = state_manager.get_status_report()
        assert report["trust"]["min"] == 0.5
        assert report["trust"]["max"] == 0.8
        assert report["trust"]["average"] == 0.65
        assert len(report["trust"]["by_subsystem"]) == 2

    def test_report_has_sensor_counts(self, state_manager):
        state_manager.update_from_sensors({"gps": "ok", "sonar": "degraded", "lidar": "offline"})
        report = state_manager.get_status_report()
        assert report["sensors"]["total"] == 3
        assert report["sensors"]["ok"] == 1
        assert report["sensors"]["degraded"] == 1
        assert report["sensors"]["offline"] == 1


class TestVesselStateSerialization:
    def test_to_json_roundtrip(self, state_manager):
        state_manager.update_from_sensors({"gps": "ok"})
        state_manager.update_from_trust({"steering": 0.75})
        state_manager.set_autonomy_level(2)
        state_manager.record_mission_complete(success=True)

        json_str = state_manager.to_json()
        data = json.loads(json_str)
        assert data["vessel_id"] == "test-vessel-42"
        assert data["sensor_status"]["gps"] == "ok"
        assert data["trust_scores"]["steering"] == 0.75
        assert data["autonomy_level"] == 2
        assert data["mission_count"] == 1

    def test_from_json_restore(self, state_manager):
        state_manager.update_from_trust({"steering": 0.9})
        json_str = state_manager.to_json()

        new_mgr = VesselStateManager(vessel_id="other")
        new_mgr.from_json(json_str)
        assert new_mgr.state.vessel_id == "test-vessel-42"
        assert new_mgr.state.trust_scores["steering"] == 0.9

    def test_to_json_is_valid_json(self, state_manager):
        json_str = state_manager.to_json()
        json.loads(json_str)  # Should not raise


class TestVesselStateReset:
    def test_reset_clears_state(self, state_manager):
        state_manager.update_from_sensors({"gps": "ok"})
        state_manager.record_error()
        state_manager.record_mission_complete(success=False)

        state_manager.reset()
        s = state_manager.state
        assert s.sensor_status == {}
        assert s.error_count == 0
        assert s.mission_count == 0
        assert s.mission_failures == 0
        assert s.trust_scores == {}

    def test_reset_preserves_vessel_id(self, state_manager):
        state_manager.reset()
        assert state_manager.state.vessel_id == "test-vessel-42"


# ===================================================================
# Test 3: Mission Loading and Parsing
# ===================================================================

class TestMissionParsing:
    def test_parse_deploy_reflex(self, mission_runner):
        m = mission_runner.load_mission("deploy_reflex:reflex=heading_hold,target=esp32")
        assert m.mission_type == MissionType.DEPLOY_REFLEX
        assert m.params["reflex"] == "heading_hold"
        assert m.params["target"] == "esp32"

    def test_parse_survey(self, mission_runner):
        m = mission_runner.load_mission("survey:pattern=grid,duration=120")
        assert m.mission_type == MissionType.SURVEY
        assert m.params["pattern"] == "grid"
        assert m.params["duration"] == "120"

    def test_parse_calibrate(self, mission_runner):
        m = mission_runner.load_mission("calibrate:subsystem=imu,mode=auto")
        assert m.mission_type == MissionType.CALIBRATE
        assert m.params["subsystem"] == "imu"

    def test_parse_diagnostic(self, mission_runner):
        m = mission_runner.load_mission("diagnostic:scope=full")
        assert m.mission_type == MissionType.DIAGNOSTIC
        assert m.params["scope"] == "full"

    def test_parse_report(self, mission_runner):
        m = mission_runner.load_mission("report:type=status")
        assert m.mission_type == MissionType.REPORT
        assert m.params["type"] == "status"

    def test_parse_with_description(self, mission_runner):
        m = mission_runner.load_mission(
            "deploy_reflex:reflex=depth_hold,target=esp32,description=Deploy depth hold for reef survey"
        )
        assert m.description == "Deploy depth hold for reef survey"
        assert m.params["reflex"] == "depth_hold"

    def test_parse_minimal(self, mission_runner):
        m = mission_runner.load_mission("report:")
        assert m.mission_type == MissionType.REPORT
        assert m.params == {}

    def test_parse_unknown_type_raises(self, mission_runner):
        with pytest.raises(MissionParseError, match="Unknown mission type"):
            mission_runner.load_mission("explode:target=everything")

    def test_parse_missing_colon_raises(self, mission_runner):
        with pytest.raises(MissionParseError, match="Missing colon"):
            mission_runner.load_mission("no colon here")

    def test_parse_empty_line_raises(self, mission_runner):
        with pytest.raises(MissionParseError, match="Empty or comment"):
            mission_runner.load_mission("")

    def test_parse_comment_raises(self, mission_runner):
        with pytest.raises(MissionParseError, match="Empty or comment"):
            mission_runner.load_mission("# This is a comment")

    def test_raw_line_preserved(self, mission_runner):
        line = "deploy_reflex:reflex=test"
        m = mission_runner.load_mission(line)
        assert m.raw_line == line

    def test_mission_str(self, mission_runner):
        m = mission_runner.load_mission("report:type=health")
        assert "report" in str(m)

    def test_mission_name_truncated(self, mission_runner):
        m = mission_runner.load_mission(
            "report:description=" + "x" * 100
        )
        assert len(m.name) <= 40


# ===================================================================
# Test 4: Mission Execution
# ===================================================================

class TestMissionExecution:
    def test_execute_deploy_reflex(self, mission_runner):
        m = mission_runner.load_mission("deploy_reflex:reflex=heading_hold,target=esp32")
        result = mission_runner.execute_mission(m)
        assert result.status == MissionStatus.SUCCESS
        assert result.bytecode is not None
        assert len(result.bytecode) > 0
        assert "heading_hold" in result.output

    def test_execute_survey(self, mission_runner):
        m = mission_runner.load_mission("survey:pattern=spiral,duration=60")
        result = mission_runner.execute_mission(m)
        assert result.status == MissionStatus.SUCCESS
        assert result.telemetry["pattern"] == "spiral"

    def test_execute_calibrate(self, mission_runner):
        m = mission_runner.load_mission("calibrate:subsystem=compass")
        result = mission_runner.execute_mission(m)
        assert result.status == MissionStatus.SUCCESS
        assert "compass" in result.output

    def test_execute_diagnostic(self, mission_runner):
        m = mission_runner.load_mission("diagnostic:scope=full")
        result = mission_runner.execute_mission(m)
        assert result.status == MissionStatus.SUCCESS
        assert result.telemetry["checks"] == 5

    def test_execute_diagnostic_partial(self, mission_runner):
        m = mission_runner.load_mission("diagnostic:scope=wire_protocol")
        result = mission_runner.execute_mission(m)
        assert result.status == MissionStatus.SUCCESS
        assert result.telemetry["checks"] == 1

    def test_execute_report(self, mission_runner):
        m = mission_runner.load_mission("report:type=status")
        result = mission_runner.execute_mission(m)
        assert result.status == MissionStatus.SUCCESS
        assert result.telemetry["report_type"] == "status"

    def test_execute_has_duration(self, mission_runner):
        m = mission_runner.load_mission("report:")
        result = mission_runner.execute_mission(m)
        assert result.duration_seconds >= 0.0

    def test_execute_result_links_mission(self, mission_runner):
        m = mission_runner.load_mission("survey:pattern=lawnmower")
        result = mission_runner.execute_mission(m)
        assert result.mission is m
        assert result.mission.mission_type == MissionType.SURVEY


# ===================================================================
# Test 5: Bytecode Generation
# ===================================================================

class TestBytecodeGeneration:
    def test_deploy_reflex_generates_aab(self, mission_runner):
        m = mission_runner.load_mission("deploy_reflex:reflex=heading_hold,target=esp32")
        bc = mission_runner.generate_bytecode_for_mission(m)
        assert bc[:4] == b"NXAB"  # AAB magic
        assert len(bc) > 8

    def test_non_deploy_returns_stub(self, mission_runner):
        m = mission_runner.load_mission("survey:pattern=grid")
        bc = mission_runner.generate_bytecode_for_mission(m)
        assert bc == b"\x00\x00\x00\x00\x00\x00\x00\x00"

    def test_report_returns_stub(self, mission_runner):
        m = mission_runner.load_mission("report:")
        bc = mission_runner.generate_bytecode_for_mission(m)
        assert len(bc) == 8

    def test_deploy_bytecode_has_tlv(self, mission_runner):
        m = mission_runner.load_mission(
            "deploy_reflex:reflex=depth,target=esp32,description=Depth control"
        )
        bc = mission_runner.generate_bytecode_for_mission(m)
        # Check for TLV type description tag (0x01)
        assert 0x01 in bc[8:]  # after header
        # Check for TLV end marker
        assert bc[-1] == 0x00

    def test_deploy_bytecode_includes_narrative(self, mission_runner):
        m = mission_runner.load_mission(
            "deploy_reflex:reflex=test,description=Custom description here"
        )
        bc = mission_runner.generate_bytecode_for_mission(m)
        # Narrative tag 0x05 should be present
        assert 0x05 in bc[8:]


# ===================================================================
# Test 6: .agent/next and .agent/done Management
# ===================================================================

class TestAgentQueue:
    def test_read_empty_queue(self, tmp_repo):
        lines = read_next_queue(tmp_repo / ".agent")
        assert lines == []

    def test_read_populated_queue(self, tmp_repo):
        (tmp_repo / ".agent" / "next").write_text(
            "report:type=status\ndiagnostic:scope=full\n", encoding="utf-8"
        )
        lines = read_next_queue(tmp_repo / ".agent")
        assert len(lines) == 2
        assert lines[0] == "report:type=status"
        assert lines[1] == "diagnostic:scope=full"

    def test_read_skips_comments(self, tmp_repo):
        (tmp_repo / ".agent" / "next").write_text(
            "# This is a comment\nreport:type=status\n# Another comment\n",
            encoding="utf-8",
        )
        lines = read_next_queue(tmp_repo / ".agent")
        assert len(lines) == 1
        assert lines[0] == "report:type=status"

    def test_read_skips_empty_lines(self, tmp_repo):
        (tmp_repo / ".agent" / "next").write_text(
            "\n\nreport:type=status\n\n", encoding="utf-8"
        )
        lines = read_next_queue(tmp_repo / ".agent")
        assert len(lines) == 1

    def test_read_nonexistent_returns_empty(self, tmp_repo):
        lines = read_next_queue(tmp_repo / ".agent" / "nonexistent")
        assert lines == []

    def test_pop_single_mission(self, tmp_repo):
        (tmp_repo / ".agent" / "next").write_text(
            "report:type=status\n", encoding="utf-8"
        )
        line = pop_next_mission(tmp_repo / ".agent")
        assert line == "report:type=status"
        # Queue should now be empty
        remaining = read_next_queue(tmp_repo / ".agent")
        assert remaining == []

    def test_pop_preserves_order(self, tmp_repo):
        (tmp_repo / ".agent" / "next").write_text(
            "m1\nm2\nm3\n", encoding="utf-8"
        )
        assert pop_next_mission(tmp_repo / ".agent") == "m1"
        assert pop_next_mission(tmp_repo / ".agent") == "m2"
        assert pop_next_mission(tmp_repo / ".agent") == "m3"
        assert pop_next_mission(tmp_repo / ".agent") is None

    def test_pop_empty_returns_none(self, tmp_repo):
        assert pop_next_mission(tmp_repo / ".agent") is None

    def test_append_done(self, tmp_repo):
        m = Mission(
            raw_line="report:type=status",
            mission_type=MissionType.REPORT,
            params={"type": "status"},
            description="",
        )
        result = MissionResult(mission=m, status=MissionStatus.SUCCESS, output="OK")
        append_done(tmp_repo / ".agent", "report:type=status", result)

        done_content = (tmp_repo / ".agent" / "done").read_text(encoding="utf-8")
        assert "report:type=status" in done_content
        assert "success" in done_content

    def test_append_done_with_error(self, tmp_repo):
        m = Mission(
            raw_line="deploy_reflex:reflex=bad",
            mission_type=MissionType.DEPLOY_REFLEX,
            params={},
            description="",
        )
        result = MissionResult(
            mission=m, status=MissionStatus.FAILED, error="Sensor timeout"
        )
        append_done(tmp_repo / ".agent", "deploy_reflex:reflex=bad", result)

        done_content = (tmp_repo / ".agent" / "done").read_text(encoding="utf-8")
        assert "Sensor timeout" in done_content
        assert "failed" in done_content

    def test_append_done_multiple(self, tmp_repo):
        for i in range(5):
            m = Mission(
                raw_line=f"report:idx={i}",
                mission_type=MissionType.REPORT,
                params={},
                description="",
            )
            result = MissionResult(mission=m, status=MissionStatus.SUCCESS)
            append_done(tmp_repo / ".agent", f"report:idx={i}", result)

        done_lines = (tmp_repo / ".agent" / "done").read_text(
            encoding="utf-8"
        ).strip().split("\n")
        assert len(done_lines) == 5


# ===================================================================
# Test 7: Heartbeat Cycle Execution
# ===================================================================

class TestHeartbeatInit:
    def test_init_with_config_file(self, config_file):
        hb = EdgeHeartbeat(config_path=config_file)
        assert hb.config.vessel_id == "test-vessel-42"
        assert hb.cycle_count == 0

    def test_init_with_config_object(self, config_dict):
        cfg = HeartbeatConfig(**config_dict)
        hb = EdgeHeartbeat(config=cfg)
        assert hb.config.vessel_id == "test-vessel-42"

    def test_init_default_config(self, tmp_repo):
        cfg = HeartbeatConfig(repo_path=str(tmp_repo))
        hb = EdgeHeartbeat(config=cfg)
        assert hb.config.vessel_id == "nexus-vessel-001"
        assert hb.config.heartbeat_interval == 300

    def test_creates_agent_dirs(self, config_dict, tmp_repo):
        # Remove .agent to test creation
        import shutil
        agent_dir = tmp_repo / ".agent"
        if agent_dir.exists():
            shutil.rmtree(agent_dir)

        cfg = HeartbeatConfig(**config_dict)
        EdgeHeartbeat(config=cfg)
        assert agent_dir.exists()
        assert (agent_dir / "next").exists()
        assert (agent_dir / "done").exists()

    def test_has_subsystems(self, heartbeat):
        assert heartbeat.state_manager is not None
        assert heartbeat.mission_runner is not None


class TestHeartbeatIdleCycle:
    def test_run_once_idle(self, heartbeat):
        """Heartbeat with no pending missions should succeed."""
        result = heartbeat.run_once()
        assert result.success is True
        assert result.cycle_number == 1
        assert result.mission_executed is False
        assert result.mission_type is None
        assert result.total_duration >= 0.0

    def test_idle_has_5_phases(self, heartbeat):
        result = heartbeat.run_once()
        assert len(result.phases) == 5

    def test_idle_phase_names(self, heartbeat):
        result = heartbeat.run_once()
        phases = [p.phase for p in result.phases]
        expected = [
            HeartbeatPhase.PERCEIVE,
            HeartbeatPhase.THINK,
            HeartbeatPhase.ACT,
            HeartbeatPhase.REMEMBER,
            HeartbeatPhase.NOTIFY,
        ]
        assert phases == expected

    def test_cycle_count_increments(self, heartbeat):
        heartbeat.run_once()
        assert heartbeat.cycle_count == 1
        heartbeat.run_once()
        assert heartbeat.cycle_count == 2


class TestHeartbeatWithMission:
    def test_run_once_with_single_mission(self, heartbeat):
        # Add a mission to the queue
        next_file = heartbeat.agent_dir / "next"
        next_file.write_text("report:type=health\n", encoding="utf-8")

        result = heartbeat.run_once()
        assert result.success is True
        assert result.mission_executed is True
        assert result.mission_type == "report"

    def test_mission_popped_from_queue(self, heartbeat):
        next_file = heartbeat.agent_dir / "next"
        next_file.write_text("report:type=health\n", encoding="utf-8")

        heartbeat.run_once()
        remaining = read_next_queue(heartbeat.agent_dir)
        assert remaining == []

    def test_mission_recorded_in_done(self, heartbeat):
        next_file = heartbeat.agent_dir / "next"
        next_file.write_text("report:type=health\n", encoding="utf-8")

        heartbeat.run_once()
        done_content = (heartbeat.agent_dir / "done").read_text(encoding="utf-8")
        assert "report:type=health" in done_content
        assert "success" in done_content

    def test_run_processes_one_mission_per_cycle(self, heartbeat):
        next_file = heartbeat.agent_dir / "next"
        next_file.write_text(
            "report:type=health\ndiagnostic:scope=full\n", encoding="utf-8"
        )

        heartbeat.run_once()
        remaining = read_next_queue(heartbeat.agent_dir)
        assert len(remaining) == 1
        assert "diagnostic:scope=full" in remaining[0]

    def test_run_deploy_reflex_mission(self, heartbeat):
        next_file = heartbeat.agent_dir / "next"
        next_file.write_text(
            "deploy_reflex:reflex=heading_hold,target=esp32,description=Deploy heading hold\n",
            encoding="utf-8",
        )

        result = heartbeat.run_once()
        assert result.success is True
        assert result.mission_type == "deploy_reflex"

    def test_survey_mission(self, heartbeat):
        next_file = heartbeat.agent_dir / "next"
        next_file.write_text("survey:pattern=grid,duration=60\n", encoding="utf-8")

        result = heartbeat.run_once()
        assert result.success is True
        assert result.mission_type == "survey"


class TestHeartbeatPhases:
    def test_perceive_phase_data(self, heartbeat):
        next_file = heartbeat.agent_dir / "next"
        next_file.write_text("report:\n", encoding="utf-8")
        result = heartbeat.run_once()

        perceive = result.phase_result(HeartbeatPhase.PERCEIVE)
        assert perceive is not None
        assert perceive.success is True
        assert perceive.data["pending_missions"] == ["report:"]

    def test_think_phase_with_mission(self, heartbeat):
        next_file = heartbeat.agent_dir / "next"
        next_file.write_text("report:type=health\n", encoding="utf-8")
        result = heartbeat.run_once()

        think = result.phase_result(HeartbeatPhase.THINK)
        assert think is not None
        assert think.success is True
        assert think.data["should_execute"] is True
        assert think.data["mission"] is not None

    def test_think_phase_idle(self, heartbeat):
        result = heartbeat.run_once()

        think = result.phase_result(HeartbeatPhase.THINK)
        assert think.data["should_execute"] is False
        assert think.data["action"] == "idle"

    def test_act_phase_idle(self, heartbeat):
        result = heartbeat.run_once()

        act = result.phase_result(HeartbeatPhase.ACT)
        assert act.success is True
        assert act.data["executed"] is False

    def test_act_phase_executes(self, heartbeat):
        next_file = heartbeat.agent_dir / "next"
        next_file.write_text("report:\n", encoding="utf-8")
        result = heartbeat.run_once()

        act = result.phase_result(HeartbeatPhase.ACT)
        assert act.data["executed"] is True
        assert act.data["mission_type"] == "report"

    def test_remember_phase_records(self, heartbeat):
        next_file = heartbeat.agent_dir / "next"
        next_file.write_text("report:\n", encoding="utf-8")
        result = heartbeat.run_once()

        remember = result.phase_result(HeartbeatPhase.REMEMBER)
        assert remember.success is True
        assert "Recorded" in remember.message

    def test_remember_phase_idle(self, heartbeat):
        result = heartbeat.run_once()

        remember = result.phase_result(HeartbeatPhase.REMEMBER)
        assert "No mission" in remember.message

    def test_notify_phase(self, heartbeat):
        result = heartbeat.run_once()

        notify = result.phase_result(HeartbeatPhase.NOTIFY)
        assert notify.success is True
        assert "status_report" in notify.data

    def test_notify_status_level(self, heartbeat):
        result = heartbeat.run_once()

        notify = result.phase_result(HeartbeatPhase.NOTIFY)
        report = notify.data["status_report"]
        assert report["status_level"] in ("ALL_CLEAR", "ATTENTION", "ALERT", "EXECUTING")

    def test_all_phases_have_duration(self, heartbeat):
        result = heartbeat.run_once()
        for phase_result in result.phases:
            assert phase_result.duration_seconds >= 0.0


# ===================================================================
# Test 8: Error Handling
# ===================================================================

class TestErrorHandling:
    def test_invalid_mission_in_queue(self, heartbeat):
        """Invalid mission line in queue should be skipped in THINK phase."""
        next_file = heartbeat.agent_dir / "next"
        next_file.write_text("invalid_mission_no_colon\n", encoding="utf-8")

        result = heartbeat.run_once()
        # THINK phase should succeed (bad mission gets popped and skipped)
        think = result.phase_result(HeartbeatPhase.THINK)
        assert think.success is True
        assert "Skipped" in think.message or "unparseable" in think.message.lower()

        # Bad mission should have been removed from queue
        remaining = read_next_queue(heartbeat.agent_dir)
        assert remaining == []

        # Bad mission should be recorded in done
        done_content = (heartbeat.agent_dir / "done").read_text(encoding="utf-8")
        assert "failed" in done_content

    def test_failed_think_still_completes_cycle(self, heartbeat):
        """Even with an unparseable mission, all 5 phases should be present."""
        next_file = heartbeat.agent_dir / "next"
        next_file.write_text("bad_format\n", encoding="utf-8")

        result = heartbeat.run_once()
        assert len(result.phases) == 5
        # THINK should succeed (skip bad mission), ACT should not execute
        assert result.phase_result(HeartbeatPhase.THINK).success is True
        assert result.mission_executed is False

    def test_vessel_error_count_increments(self, heartbeat):
        next_file = heartbeat.agent_dir / "next"
        next_file.write_text("garbage_no_colon\n", encoding="utf-8")
        heartbeat.run_once()
        assert heartbeat.state_manager.state.error_count >= 1

    def test_multiple_invalid_missions(self, heartbeat):
        """Multiple invalid missions — each should be skipped, queue continues."""
        next_file = heartbeat.agent_dir / "next"
        next_file.write_text("bad1\nbad2\nreport:\n", encoding="utf-8")

        # Cycle 1: bad1 gets popped and skipped
        result1 = heartbeat.run_once()
        assert result1.success is True
        assert result1.mission_executed is False
        assert "Skipped" in result1.phase_result(HeartbeatPhase.THINK).message

        # Cycle 2: bad2 gets popped and skipped
        result2 = heartbeat.run_once()
        assert result2.success is True
        assert result2.mission_executed is False

        # Cycle 3: valid report mission executes
        result3 = heartbeat.run_once()
        assert result3.success is True
        assert result3.mission_executed is True
        assert result3.mission_type == "report"


class TestHeartbeatResultHelpers:
    def test_success_property(self):
        r = HeartbeatResult(phases=[
            PhaseResult(phase=HeartbeatPhase.PERCEIVE, success=True),
            PhaseResult(phase=HeartbeatPhase.THINK, success=True),
            PhaseResult(phase=HeartbeatPhase.ACT, success=True),
            PhaseResult(phase=HeartbeatPhase.REMEMBER, success=True),
            PhaseResult(phase=HeartbeatPhase.NOTIFY, success=True),
        ])
        assert r.success is True

    def test_success_false_on_any_failure(self):
        r = HeartbeatResult(phases=[
            PhaseResult(phase=HeartbeatPhase.PERCEIVE, success=True),
            PhaseResult(phase=HeartbeatPhase.THINK, success=False),
            PhaseResult(phase=HeartbeatPhase.ACT, success=True),
            PhaseResult(phase=HeartbeatPhase.REMEMBER, success=True),
            PhaseResult(phase=HeartbeatPhase.NOTIFY, success=True),
        ])
        assert r.success is False

    def test_failed_phase_property(self):
        r = HeartbeatResult(phases=[
            PhaseResult(phase=HeartbeatPhase.PERCEIVE, success=True),
            PhaseResult(phase=HeartbeatPhase.THINK, success=False, message="Parse error"),
            PhaseResult(phase=HeartbeatPhase.ACT, success=True),
            PhaseResult(phase=HeartbeatPhase.REMEMBER, success=True),
            PhaseResult(phase=HeartbeatPhase.NOTIFY, success=True),
        ])
        assert r.failed_phase == "THINK"

    def test_failed_phase_none_when_success(self):
        r = HeartbeatResult(phases=[
            PhaseResult(phase=p, success=True)
            for p in HeartbeatPhase
        ])
        assert r.failed_phase is None

    def test_phase_result_lookup(self):
        r = HeartbeatResult(phases=[
            PhaseResult(phase=HeartbeatPhase.PERCEIVE, success=True),
            PhaseResult(phase=HeartbeatPhase.THINK, success=True),
            PhaseResult(phase=HeartbeatPhase.ACT, success=True),
            PhaseResult(phase=HeartbeatPhase.REMEMBER, success=True),
            PhaseResult(phase=HeartbeatPhase.NOTIFY, success=True),
        ])
        pr = r.phase_result(HeartbeatPhase.ACT)
        assert pr is not None
        assert pr.phase == HeartbeatPhase.ACT

    def test_phase_result_none_for_missing(self):
        r = HeartbeatResult(phases=[])
        assert r.phase_result(HeartbeatPhase.PERCEIVE) is None


# ===================================================================
# Test 9: Heartbeat Status
# ===================================================================

class TestHeartbeatStatus:
    def test_get_status(self, heartbeat):
        status = heartbeat.get_status()
        assert status["vessel_id"] == "test-vessel-42"
        assert status["cycle_count"] == 0
        assert status["running"] is False
        assert status["interval"] == 60

    def test_get_status_after_cycle(self, heartbeat):
        heartbeat.run_once()
        status = heartbeat.get_status()
        assert status["cycle_count"] == 1

    def test_vessel_state_accessible(self, heartbeat):
        vs = heartbeat.vessel_state
        assert vs.vessel_id == "test-vessel-42"


# ===================================================================
# Test 10: Vessel Template Config
# ===================================================================

class TestVesselTemplate:
    def test_template_is_valid_json(self):
        template_path = Path(__file__).parent.parent / "configs" / "vessel_template.json"
        assert template_path.exists(), "Template file should exist"
        data = json.loads(template_path.read_text(encoding="utf-8"))
        assert "vessel_id" in data
        assert "heartbeat_interval" in data

    def test_template_loads(self, tmp_repo):
        """Load the vessel template config and verify it works."""
        from pathlib import Path
        template_path = (
            Path(__file__).parent.parent / "configs" / "vessel_template.json"
        )
        if template_path.exists():
            cfg = load_config(str(template_path))
            assert cfg.vessel_id == "nexus-vessel-001"
            assert cfg.heartbeat_interval == 300
            assert cfg.serial_baud == 115200
        else:
            pytest.skip("Template file not found in test context")


# ===================================================================
# Test 11: Integration — Full Cycle with Multiple Missions
# ===================================================================

class TestIntegrationFullCycle:
    def test_process_all_missions(self, heartbeat):
        """Process 3 missions across 3 heartbeat cycles."""
        next_file = heartbeat.agent_dir / "next"
        next_file.write_text(
            "report:type=quick\n"
            "diagnostic:scope=full\n"
            "survey:pattern=spiral,duration=120\n",
            encoding="utf-8",
        )

        for i in range(3):
            result = heartbeat.run_once()
            assert result.success is True, f"Cycle {i+1} failed: {result.error}"
            assert result.mission_executed is True

        remaining = read_next_queue(heartbeat.agent_dir)
        assert remaining == []

    def test_state_accumulates_across_cycles(self, heartbeat):
        next_file = heartbeat.agent_dir / "next"
        next_file.write_text(
            "report:\ndiagnostic:\nsurvey:\n", encoding="utf-8"
        )

        heartbeat.run_once()
        heartbeat.run_once()
        heartbeat.run_once()

        state = heartbeat.state_manager.state
        assert state.mission_count == 3
        assert state.mission_failures == 0

    def test_mixed_success_and_failure(self, heartbeat):
        next_file = heartbeat.agent_dir / "next"
        next_file.write_text(
            "report:\nbad_no_colon\nreport:\n", encoding="utf-8"
        )

        # Cycle 1: success (report)
        r1 = heartbeat.run_once()
        assert r1.mission_executed is True

        # Cycle 2: bad mission gets popped and skipped (but cycle succeeds)
        r2 = heartbeat.run_once()
        assert r2.success is True
        assert r2.mission_executed is False
        assert "Skipped" in r2.phase_result(HeartbeatPhase.THINK).message

        # Cycle 3: success (report)
        r3 = heartbeat.run_once()
        assert r3.mission_executed is True

        state = heartbeat.state_manager.state
        assert state.mission_count >= 2
        assert state.error_count >= 1
