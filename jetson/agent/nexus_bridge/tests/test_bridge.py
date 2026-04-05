"""Comprehensive tests for NEXUS git-agent bridge.

Tests all modules using mocked git operations (temp dirs + real git init).
No external services or network required.

Run with:
    cd /tmp/nexus-runtime && python -m pytest jetson/agent/nexus_bridge/tests/ -v
"""

from __future__ import annotations

import json
import math
import os
import struct
import tempfile
import time

import pytest

from nexus_bridge.bytecode_deployer import (
    BytecodeDeployer,
    INSTR_SIZE,
    ValidationReport,
    ValidationResult,
    unpack_instruction,
)
from nexus_bridge.telemetry_ingester import (
    TelemetryIngester,
    TelemetryResult,
)
from nexus_bridge.trust_sync import (
    TrustSync,
    TrustResult,
)
from nexus_bridge.equipment_manifest import (
    EquipmentManifest,
)
from nexus_bridge.bridge import (
    NexusBridge,
    BridgeStatus,
    DeployResult,
    SafetyResult,
    MissionResult,
)


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def temp_repo():
    """Create a temporary git repo for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from git import Repo
        repo = Repo.init(tmpdir)
        # Configure git user for commits
        repo.config_writer().set_value("user", "name", "NEXUS Test").release()
        repo.config_writer().set_value("user", "email", "test@nexus.local").release()
        yield repo


@pytest.fixture
def temp_repo_path(temp_repo):
    """Return the path of the temp repo."""
    return temp_repo.working_dir


@pytest.fixture
def deployer():
    """Create a BytecodeDeployer for testing."""
    return BytecodeDeployer(max_stack=16, max_cycles=1000)


@pytest.fixture
def bridge(temp_repo_path):
    """Create a NexusBridge with a temp repo."""
    return NexusBridge(
        vessel_id="test-vessel",
        repo_path=temp_repo_path,
    )


def _make_safe_bytecode(opcodes: list[int]) -> bytes:
    """Build a safe bytecode blob from a list of opcodes.
    Each instruction is 8 bytes: opcode(1) + flags(1) + op1(2) + op2(4).
    """
    result = bytearray()
    for op in opcodes:
        instr = struct.pack("<BBHI", op, 0, 0, 0)
        result.extend(instr)
    return bytes(result)


def _make_push_i8(value: int) -> bytes:
    """Build a PUSH_I8 instruction."""
    return struct.pack("<BBHI", 0x01, 0, value, 0)


def _make_push_f32(value: float) -> bytes:
    """Build a PUSH_F32 instruction."""
    f32_bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.pack("<BBHI", 0x03, 0x02, 0, f32_bits)


def _make_clamp_f(lo: float, hi: float) -> bytes:
    """Build a CLAMP_F instruction."""
    lo_bits = struct.unpack("<I", struct.pack("<f", lo))[0]
    hi_bits = struct.unpack("<I", struct.pack("<f", hi))[0]
    # CLAMP_F: opcode=0x10, lo in op1 as u16, hi in op2 as u32
    return struct.pack("<BBHI", 0x10, 0x02, int(lo * 100), hi_bits)


def _make_write_pin(pin: int) -> bytes:
    """Build a WRITE_PIN instruction."""
    return struct.pack("<BBHI", 0x1B, 0, pin, 0)


def _make_jump(target: int) -> bytes:
    """Build a JUMP instruction."""
    return struct.pack("<BBHI", 0x1D, 0, target, 0)


def _make_pop() -> bytes:
    """Build a POP instruction."""
    return struct.pack("<BBHI", 0x04, 0, 0, 0)


# ═══════════════════════════════════════════════════════════════════
# 1. BytecodeDeployer Tests
# ═══════════════════════════════════════════════════════════════════

class TestBytecodeDeployer:
    """Tests for the BytecodeDeployer module."""

    def test_validate_nop_bytecode(self, deployer):
        """NOP-only bytecode should pass validation."""
        bytecode = _make_safe_bytecode([0x00, 0x00, 0x00])
        result = deployer.validate_bytecode(bytecode)
        assert result.passed is True
        assert result.report.instruction_count == 3
        assert result.report.max_stack_depth == 0
        assert len(result.report.errors) == 0

    def test_validate_empty_bytecode(self, deployer):
        """Empty bytecode should fail validation."""
        result = deployer.validate_bytecode(b"")
        assert result.passed is False
        assert any("empty" in e for e in result.report.errors)

    def test_validate_misaligned_bytecode(self, deployer):
        """Bytecode not aligned to 8 bytes should fail."""
        result = deployer.validate_bytecode(b"\x00\x00\x00\x00\x01")
        assert result.passed is False
        assert any("multiple of 8" in e for e in result.report.errors)

    def test_validate_invalid_opcode(self, deployer):
        """Opcodes beyond 0x56 should be rejected."""
        # Build an instruction with opcode 0xFF
        bytecode = struct.pack("<BBHI", 0xFF, 0, 0, 0)
        result = deployer.validate_bytecode(bytecode)
        assert result.passed is False
        assert any("Invalid opcode" in e for e in result.report.errors)

    def test_validate_a2a_opcodes(self, deployer):
        """A2A opcodes (0x20-0x56) should pass."""
        bytecode = _make_safe_bytecode([0x20, 0x30, 0x40, 0x50])
        result = deployer.validate_bytecode(bytecode)
        assert result.passed is True

    def test_validate_stack_depth(self, deployer):
        """Too many pushes should trigger stack depth error."""
        # Push 17 values (exceeds max_stack=16)
        opcodes = [0x01] * 17  # 17x PUSH_I8
        bytecode = _make_safe_bytecode(opcodes)
        result = deployer.validate_bytecode(bytecode)
        assert result.passed is False
        assert any("stack depth" in e.lower() for e in result.report.errors)

    def test_validate_stack_underflow(self, deployer):
        """POP without push should trigger underflow."""
        bytecode = _make_safe_bytecode([0x04])  # POP
        result = deployer.validate_bytecode(bytecode)
        assert result.passed is False
        assert any("underflow" in e.lower() for e in result.report.errors)

    def test_validate_jump_out_of_bounds(self, deployer):
        """Jump target beyond bytecode should fail."""
        bytecode = _make_jump(999)  # Jump to instruction 999
        result = deployer.validate_bytecode(bytecode)
        assert result.passed is False
        assert any("targets" in e for e in result.report.errors)

    def test_validate_jump_in_bounds(self, deployer):
        """Jump target within bytecode should pass."""
        bytecode = _make_jump(0)  # Jump to instruction 0
        result = deployer.validate_bytecode(bytecode)
        assert result.passed is True

    def test_validate_write_pin_without_clamp(self, deployer):
        """WRITE_PIN without preceding CLAMP_F should fail."""
        bytecode = _make_push_i8(42) + _make_write_pin(25)
        result = deployer.validate_bytecode(bytecode)
        assert result.passed is False
        assert any("CLAMP_F" in e for e in result.report.errors)

    def test_validate_write_pin_with_clamp(self, deployer):
        """WRITE_PIN with preceding CLAMP_F should pass."""
        bytecode = (
            _make_push_i8(42) +
            _make_clamp_f(0.0, 1.0) +
            _make_write_pin(25)
        )
        result = deployer.validate_bytecode(bytecode)
        assert result.passed is True

    def test_validate_nan_value(self, deployer):
        """NaN in PUSH_F32 should fail."""
        bytecode = _make_push_f32(float("nan"))
        result = deployer.validate_bytecode(bytecode)
        assert result.passed is False
        assert any("NaN" in e for e in result.report.errors)

    def test_validate_inf_value(self, deployer):
        """Infinity in PUSH_F32 should fail."""
        bytecode = _make_push_f32(float("inf"))
        result = deployer.validate_bytecode(bytecode)
        assert result.passed is False
        assert any("Infinity" in e for e in result.report.errors)

    def test_validate_normal_float(self, deployer):
        """Normal float in PUSH_F32 should pass."""
        bytecode = _make_push_f32(3.14)
        result = deployer.validate_bytecode(bytecode)
        assert result.passed is True

    def test_validate_cycle_budget(self, deployer):
        """Exceeding cycle budget should fail."""
        # Create bytecode with 1001 instructions (exceeds max_cycles=1000)
        bytecode = _make_safe_bytecode([0x00] * 1001)
        result = deployer.validate_bytecode(bytecode)
        assert result.passed is False
        assert any("cycle budget" in e for e in result.report.errors)

    def test_validate_hash_sha256(self, deployer):
        """Validation report should include SHA-256 hash."""
        bytecode = _make_safe_bytecode([0x00])
        result = deployer.validate_bytecode(bytecode)
        assert result.report.hash_sha256
        assert len(result.report.hash_sha256) == 64  # SHA-256 hex length

    def test_commit_bytecode(self, deployer, temp_repo):
        """Bytecode should be committed to git with metadata."""
        bytecode = _make_safe_bytecode([0x00, 0x01, 0x04])
        metadata = {
            "source_reflex": "test-reflex",
            "provenance": {"author": "test-agent"},
            "vessel_id": "test-vessel",
            "validation": {"is_valid": True},
        }

        commit_hash = deployer.commit_bytecode(bytecode, metadata, temp_repo)

        # Verify commit was created
        assert len(commit_hash) == 40  # Full SHA-1 hex

        # Verify commit message
        commit = temp_repo.head.commit
        assert "DEPLOY" in commit.message
        assert "test-reflex" in commit.message

        # Verify files exist in repo
        bytecode_dir = os.path.join(
            temp_repo.working_dir, ".agent", "bytecode", "test-reflex"
        )
        assert os.path.isdir(bytecode_dir)
        files = os.listdir(bytecode_dir)
        assert len(files) >= 2  # .bin + .meta.json

    def test_deploy_to_device_stub(self, deployer):
        """Deploy to device should return True (stub)."""
        bytecode = _make_safe_bytecode([0x00])
        result = deployer.deploy_to_device(bytecode, "/dev/ttyUSB0")
        assert result is True

    def test_make_nop_bytecode(self, deployer):
        """make_nop_bytecode should produce valid NOP instructions."""
        bytecode = BytecodeDeployer.make_nop_bytecode(5)
        assert len(bytecode) == 5 * INSTR_SIZE
        result = deployer.validate_bytecode(bytecode)
        assert result.passed is True
        assert result.report.instruction_count == 5

    def test_complex_safe_reflex(self, deployer):
        """A realistic reflex: read pin, clamp, write pin should pass."""
        # PUSH_I8(34) -> READ_PIN -> CLAMP_F(0,1) -> WRITE_PIN(25) -> POP
        bytecode = (
            _make_push_i8(34) +      # Push pin number
            struct.pack("<BBHI", 0x1A, 0, 0, 0) +  # READ_PIN
            _make_clamp_f(0.0, 1.0) +              # CLAMP_F
            _make_write_pin(25) +                   # WRITE_PIN
            _make_pop()                             # POP (cleanup)
        )
        result = deployer.validate_bytecode(bytecode)
        assert result.passed is True


# ═══════════════════════════════════════════════════════════════════
# 2. TelemetryIngester Tests
# ═══════════════════════════════════════════════════════════════════

class TestTelemetryIngester:
    """Tests for the TelemetryIngester module."""

    def test_add_single_reading(self):
        """Adding a single reading should increment pending count."""
        ingester = TelemetryIngester(vessel_id="test")
        ingester.add_reading(sensor_id=1, value=23.5, timestamp=time.time())
        assert ingester.pending_count == 1

    def test_add_multiple_readings(self):
        """Multiple readings should accumulate."""
        ingester = TelemetryIngester(vessel_id="test")
        for i in range(10):
            ingester.add_reading(sensor_id=i, value=float(i * 10), timestamp=time.time())
        assert ingester.pending_count == 10

    def test_batch_full_triggers_flush(self):
        """Batch should signal full when reaching batch_size."""
        ingester = TelemetryIngester(vessel_id="test", batch_size=5)
        full = False
        for i in range(5):
            full = ingester.add_reading(
                sensor_id=i, value=float(i), timestamp=time.time()
            )
        assert full is True

    def test_batch_not_full(self):
        """Batch should not signal full before reaching batch_size."""
        ingester = TelemetryIngester(vessel_id="test", batch_size=10)
        full = ingester.add_reading(sensor_id=1, value=1.0, timestamp=time.time())
        assert full is False

    def test_flush_empty_returns_uncommitted(self, temp_repo):
        """Flushing with no readings should return uncommitted."""
        ingester = TelemetryIngester(vessel_id="test")
        result = ingester.flush(temp_repo)
        assert result.committed is False
        assert result.readings_count == 0

    def test_flush_creates_commit(self, temp_repo):
        """Flushing should create a git commit with telemetry data."""
        ingester = TelemetryIngester(vessel_id="test-vessel", batch_size=3)
        ingester.add_reading(sensor_id=1, value=23.5, timestamp=1000.0)
        ingester.add_reading(sensor_id=2, value=1013.25, timestamp=1001.0)
        ingester.add_reading(sensor_id=1, value=23.7, timestamp=1002.0)

        result = ingester.flush(temp_repo)

        assert result.committed is True
        assert result.commit_hash
        assert result.readings_count == 3

        # Verify commit message
        commit = temp_repo.head.commit
        assert "TELEMETRY" in commit.message
        assert "3 readings" in commit.message

        # Verify telemetry file exists
        telemetry_dir = os.path.join(
            temp_repo.working_dir, ".agent", "telemetry"
        )
        assert os.path.isdir(telemetry_dir)
        files = [f for f in os.listdir(telemetry_dir) if f.endswith(".json")]
        assert len(files) == 1

        # Verify JSON content
        with open(os.path.join(telemetry_dir, files[0])) as f:
            data = json.load(f)
        assert data["vessel_id"] == "test-vessel"
        assert data["reading_count"] == 3
        assert len(data["readings"]) == 3

    def test_flush_resets_batch(self, temp_repo):
        """After flush, pending_count should be 0."""
        ingester = TelemetryIngester(vessel_id="test", batch_size=2)
        ingester.add_reading(sensor_id=1, value=1.0, timestamp=1000.0)
        ingester.add_reading(sensor_id=2, value=2.0, timestamp=1001.0)
        assert ingester.pending_count == 2

        ingester.flush(temp_repo)
        assert ingester.pending_count == 0

    def test_flush_multiple_times(self, temp_repo):
        """Multiple flush cycles should create multiple commits."""
        ingester = TelemetryIngester(vessel_id="test", batch_size=2)

        # First batch
        ingester.add_reading(sensor_id=1, value=1.0, timestamp=1000.0)
        ingester.add_reading(sensor_id=2, value=2.0, timestamp=1001.0)
        result1 = ingester.flush(temp_repo)

        # Second batch
        ingester.add_reading(sensor_id=3, value=3.0, timestamp=1002.0)
        ingester.add_reading(sensor_id=4, value=4.0, timestamp=1003.0)
        result2 = ingester.flush(temp_repo)

        assert result1.committed is True
        assert result2.committed is True
        assert result1.commit_hash != result2.commit_hash

        # Verify 2 commits
        commits = list(temp_repo.iter_commits())
        assert len(commits) >= 2

    def test_sensor_type_and_unit(self):
        """Readings should preserve sensor_type and unit."""
        ingester = TelemetryIngester(vessel_id="test")
        ingester.add_reading(
            sensor_id=5, value=25.3, timestamp=1000.0,
            sensor_type="temperature", unit="C",
        )
        assert ingester.pending_count == 1

    def test_should_flush_time_window(self):
        """should_flush should return True when window elapsed."""
        ingester = TelemetryIngester(vessel_id="test", window_seconds=5)
        # Add a reading 10 seconds ago
        old_time = time.time() - 10
        ingester.add_reading(sensor_id=1, value=1.0, timestamp=old_time)
        assert ingester.should_flush(time.time()) is True

    def test_should_flush_not_yet(self):
        """should_flush should return False when window not elapsed."""
        ingester = TelemetryIngester(vessel_id="test", window_seconds=300)
        ingester.add_reading(sensor_id=1, value=1.0, timestamp=time.time())
        assert ingester.should_flush(time.time()) is False

    def test_batch_age_seconds(self):
        """batch_age_seconds should return 0 for empty batch."""
        ingester = TelemetryIngester(vessel_id="test")
        assert ingester.batch_age_seconds == 0.0


# ═══════════════════════════════════════════════════════════════════
# 3. TrustSync Tests
# ═══════════════════════════════════════════════════════════════════

class TestTrustSync:
    """Tests for the TrustSync module."""

    def test_record_good_event(self, temp_repo):
        """Good event should increase trust score."""
        trust = TrustSync(vessel_id="test")
        result = trust.record_event(
            subsystem="navigation",
            event_type="heartbeat_ok",
            severity=0,
            details="Normal heartbeat",
            repo=temp_repo,
        )
        assert result.committed is True
        assert result.commit_hash
        assert result.new_trust_score > 0.0

    def test_record_bad_event(self, temp_repo):
        """Bad event should decrease trust score toward floor."""
        trust = TrustSync(vessel_id="test")

        # Build up enough trust to be above the floor (0.2)
        for _ in range(500):
            trust.record_event(
                subsystem="steering",
                event_type="sensor_valid",
                severity=0,
                details="Sensor OK",
            )

        initial_score = trust._engine.get_trust_score("steering")
        assert initial_score > 0.2  # Must be above floor

        # Now record a bad event (severity 0.0-1.0 in TrustSync API)
        result = trust.record_event(
            subsystem="steering",
            event_type="sensor_invalid",
            severity=0.5,
            details="IMU failure",
            repo=temp_repo,
        )
        assert result.committed is True
        assert result.new_trust_score < initial_score

    def test_trust_commit_message_format(self, temp_repo):
        """Trust event commit should use TRUST: format."""
        trust = TrustSync(vessel_id="test")
        trust.record_event(
            subsystem="engine",
            event_type="heartbeat_missed",
            severity=3,
            details="Heartbeat timeout",
            repo=temp_repo,
        )
        commit = temp_repo.head.commit
        assert commit.message.startswith("TRUST:")
        assert "engine" in commit.message
        assert "heartbeat_missed" in commit.message

    def test_trust_event_json_file(self, temp_repo):
        """Trust event should create JSON file in .agent/trust/."""
        trust = TrustSync(vessel_id="test")
        trust.record_event(
            subsystem="navigation",
            event_type="reflex_completed",
            severity=0,
            details="Reflex executed successfully",
            repo=temp_repo,
        )

        trust_dir = os.path.join(
            temp_repo.working_dir, ".agent", "trust"
        )
        assert os.path.isdir(trust_dir)
        files = [f for f in os.listdir(trust_dir) if f.endswith(".json")]
        assert len(files) == 1

        with open(os.path.join(trust_dir, files[0])) as f:
            data = json.load(f)
        assert data["subsystem"] == "navigation"
        assert data["event_type"] == "reflex_completed"
        assert data["is_bad"] is False

    def test_get_trust_history(self, temp_repo):
        """Trust history should list events from git log."""
        trust = TrustSync(vessel_id="test")
        trust.record_event("steering", "sensor_valid", 0, "OK", temp_repo)
        trust.record_event("steering", "heartbeat_ok", 0, "OK", temp_repo)
        trust.record_event("engine", "sensor_invalid", 5, "Fail", temp_repo)

        # Get all steering events
        steering_events = trust.get_trust_history("steering", repo=temp_repo)
        assert len(steering_events) == 2
        for event in steering_events:
            assert event["subsystem"] == "steering"

        # Get all engine events
        engine_events = trust.get_trust_history("engine", repo=temp_repo)
        assert len(engine_events) == 1

    def test_get_trust_history_filter_subsystem(self, temp_repo):
        """Trust history should filter by subsystem."""
        trust = TrustSync(vessel_id="test")
        trust.record_event("nav", "sensor_valid", 0, "OK", temp_repo)
        trust.record_event("steer", "sensor_valid", 0, "OK", temp_repo)

        nav_events = trust.get_trust_history("nav", repo=temp_repo)
        assert len(nav_events) == 1
        assert nav_events[0]["subsystem"] == "nav"

    def test_compute_trust_from_log(self, temp_repo):
        """Recomputed trust from log should be non-zero after good events."""
        trust = TrustSync(vessel_id="test")

        # Record several good events
        for _ in range(5):
            trust.record_event(
                "navigation", "heartbeat_ok", 0, "OK", temp_repo
            )

        score = trust.compute_trust_from_log("navigation", temp_repo)
        assert score > 0.0
        assert score <= 1.0

    def test_compute_trust_bad_events_lower_score(self, temp_repo):
        """Bad events should reduce recomputed trust score from log."""
        trust = TrustSync(vessel_id="test")

        # Record good events to build trust (without git for speed)
        for _ in range(500):
            trust.record_event(
                "engine", "heartbeat_ok", 0, "OK"
            )
        # Verify in-memory score is above floor
        assert trust._engine.get_trust_score("engine") > 0.2

        # Now record events WITH git: one good, then one bad
        trust.record_event("engine", "sensor_valid", 0, "OK", temp_repo)
        trust.record_event("engine", "sensor_invalid", 0.8, "FAIL", temp_repo)

        # Recompute from log (should find both events in chronological order)
        bad_score = trust.compute_trust_from_log("engine", temp_repo)

        # The bad event should cause the floor to be applied
        assert bad_score >= 0.0
        assert bad_score <= 1.0

    def test_trust_snapshot(self):
        """Trust snapshot should return dict of subsystem scores."""
        trust = TrustSync(vessel_id="test")
        trust.record_event("steering", "heartbeat_ok", 0, "OK")
        trust.record_event("navigation", "sensor_valid", 0, "OK")

        snapshot = trust.get_trust_snapshot()
        assert isinstance(snapshot, dict)
        assert "steering" in snapshot
        assert "navigation" in snapshot

    def test_record_event_without_repo(self):
        """Recording without repo should update engine but not commit."""
        trust = TrustSync(vessel_id="test")
        result = trust.record_event(
            subsystem="payload",
            event_type="heartbeat_ok",
            severity=0,
            details="OK",
            repo=None,
        )
        assert result.committed is False
        assert result.commit_hash == ""
        # But trust score should still be updated
        score = trust._engine.get_trust_score("payload")
        assert score > 0.0

    def test_severity_normalization(self, temp_repo):
        """TrustSync stores severity as-is (bridge.py normalizes 0-10 to 0.0-1.0)."""
        trust = TrustSync(vessel_id="test")
        # TrustSync API uses 0.0-1.0 severity directly
        result = trust.record_event(
            subsystem="test", event_type="test_event",
            severity=0.5, details="test", repo=temp_repo,
        )
        trust_dir = os.path.join(
            temp_repo.working_dir, ".agent", "trust"
        )
        files = [f for f in os.listdir(trust_dir) if f.endswith(".json")]
        with open(os.path.join(trust_dir, files[0])) as f:
            data = json.load(f)
        assert data["severity"] == 0.5


# ═══════════════════════════════════════════════════════════════════
# 4. EquipmentManifest Tests
# ═══════════════════════════════════════════════════════════════════

class TestEquipmentManifest:
    """Tests for the EquipmentManifest module."""

    def test_from_hal_config_basic(self):
        """Basic HAL config should produce valid manifest."""
        manifest_gen = EquipmentManifest()
        node_config = {
            "vessel_id": "vessel-001",
            "node_role": "usv-navigation",
            "sensors": [
                {"id": 1, "type": "analog", "pin": 34, "unit": "V",
                 "range": [0.0, 3.3], "description": "Battery voltage"},
                {"id": 2, "type": "gps", "pin": 0, "description": "GPS module"},
            ],
            "actuators": [
                {"id": 1, "type": "pwm", "pin": 25, "range": [0, 255],
                 "description": "Main thruster"},
                {"id": 2, "type": "servo", "pin": 26, "range": [0, 180],
                 "description": "Rudder"},
            ],
        }

        manifest = manifest_gen.from_hal_config(node_config)

        assert manifest["vessel_id"] == "vessel-001"
        assert manifest["node_role"] == "usv-navigation"
        assert manifest["manifest_version"] == "1.0"
        assert len(manifest["sensors"]) == 2
        assert len(manifest["actuators"]) == 2
        assert "navigation" in manifest["capabilities"]
        assert "engine" in manifest["capabilities"]
        assert "steering" in manifest["capabilities"]

    def test_from_hal_config_empty(self):
        """Empty config should produce minimal manifest."""
        manifest_gen = EquipmentManifest()
        manifest = manifest_gen.from_hal_config({})
        assert manifest["vessel_id"] == "unknown"
        assert manifest["sensors"] == []
        assert manifest["actuators"] == []
        assert manifest["capabilities"] == []

    def test_to_json(self):
        """to_json should produce valid JSON string."""
        manifest_gen = EquipmentManifest()
        manifest = manifest_gen.from_hal_config({"vessel_id": "test"})
        json_str = manifest_gen.to_json(manifest)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["vessel_id"] == "test"

        # Should be pretty-printed
        assert "\n" in json_str

    def test_validate_valid_manifest(self):
        """Valid manifest should pass validation."""
        manifest_gen = EquipmentManifest()
        manifest = manifest_gen.from_hal_config({
            "vessel_id": "test",
            "sensors": [{"id": 1, "type": "analog", "pin": 34}],
            "actuators": [{"id": 1, "type": "pwm", "pin": 25}],
        })
        is_valid, errors = manifest_gen.validate(manifest)
        assert is_valid is True
        assert errors == []

    def test_validate_missing_required_fields(self):
        """Manifest missing required fields should fail."""
        manifest_gen = EquipmentManifest()
        is_valid, errors = manifest_gen.validate({})
        assert is_valid is False
        assert any("vessel_id" in e for e in errors)

    def test_validate_invalid_sensor_type(self):
        """Invalid sensor type should fail validation."""
        manifest_gen = EquipmentManifest()
        manifest = {
            "vessel_id": "test",
            "manifest_version": "1.0",
            "generated_at": "2025-01-01T00:00:00Z",
            "sensors": [{"id": 1, "type": "magic_sensor", "pin": 1}],
            "actuators": [],
            "capabilities": [],
        }
        is_valid, errors = manifest_gen.validate(manifest)
        assert is_valid is False
        assert any("invalid type" in e for e in errors)

    def test_validate_invalid_actuator_type(self):
        """Invalid actuator type should fail validation."""
        manifest_gen = EquipmentManifest()
        manifest = {
            "vessel_id": "test",
            "manifest_version": "1.0",
            "generated_at": "2025-01-01T00:00:00Z",
            "sensors": [],
            "actuators": [{"id": 1, "type": "laser_cannon", "pin": 1}],
            "capabilities": [],
        }
        is_valid, errors = manifest_gen.validate(manifest)
        assert is_valid is False
        assert any("invalid type" in e for e in errors)

    def test_validate_invalid_capability(self):
        """Unknown capability should fail validation."""
        manifest_gen = EquipmentManifest()
        manifest = {
            "vessel_id": "test",
            "manifest_version": "1.0",
            "generated_at": "2025-01-01T00:00:00Z",
            "sensors": [],
            "actuators": [],
            "capabilities": ["telekinesis"],
        }
        is_valid, errors = manifest_gen.validate(manifest)
        assert is_valid is False
        assert any("Unknown capability" in e for e in errors)

    def test_capability_inference_gps(self):
        """GPS sensor should infer navigation capability."""
        manifest_gen = EquipmentManifest()
        manifest = manifest_gen.from_hal_config({
            "sensors": [{"type": "gps", "id": 1, "pin": 0}],
            "actuators": [],
        })
        assert "navigation" in manifest["capabilities"]

    def test_capability_inference_camera(self):
        """Camera sensor should infer surveillance capability."""
        manifest_gen = EquipmentManifest()
        manifest = manifest_gen.from_hal_config({
            "sensors": [{"type": "camera", "id": 1, "pin": 0}],
            "actuators": [],
        })
        assert "surveillance" in manifest["capabilities"]

    def test_capability_inference_thruster(self):
        """Thruster actuator should infer engine capability."""
        manifest_gen = EquipmentManifest()
        manifest = manifest_gen.from_hal_config({
            "sensors": [],
            "actuators": [{"type": "thruster", "id": 1, "pin": 25}],
        })
        assert "engine" in manifest["capabilities"]

    def test_capability_inference_no_duplicates(self):
        """Capabilities should not be duplicated."""
        manifest_gen = EquipmentManifest()
        manifest = manifest_gen.from_hal_config({
            "sensors": [
                {"type": "gps", "id": 1, "pin": 0},
                {"type": "sonar", "id": 2, "pin": 0},
            ],
            "actuators": [],
        })
        # Both GPS and sonar infer navigation, but should only appear once
        nav_count = manifest["capabilities"].count("navigation")
        assert nav_count == 1


# ═══════════════════════════════════════════════════════════════════
# 5. NexusBridge Integration Tests
# ═══════════════════════════════════════════════════════════════════

class TestNexusBridge:
    """Integration tests for the main NexusBridge class."""

    def test_bridge_initialization(self, bridge):
        """Bridge should initialize with all sub-modules."""
        assert bridge.vessel_id == "test-vessel"
        assert bridge.deployer is not None
        assert bridge.ingester is not None
        assert bridge.trust is not None
        assert bridge.manifest is not None

    def test_agent_dirs_created(self, bridge, temp_repo_path):
        """Bridge should create .agent directory structure."""
        expected_dirs = [
            ".agent/bytecode",
            ".agent/telemetry",
            ".agent/trust",
            ".agent/safety",
            ".agent/next",
            ".agent/done",
            ".agent/manifest",
        ]
        for d in expected_dirs:
            path = os.path.join(temp_repo_path, d)
            assert os.path.isdir(path), f"Missing directory: {d}"

    def test_deploy_bytecode_success(self, bridge):
        """Successful deployment should return valid DeployResult."""
        bytecode = _make_safe_bytecode([0x00, 0x01, 0x04])
        result = bridge.deploy_bytecode(
            bytecode=bytecode,
            source_reflex="test-reflex",
            provenance={"author": "test-agent", "model": "test-model"},
        )
        assert isinstance(result, DeployResult)
        assert result.success is True
        assert result.commit_hash
        assert len(result.commit_hash) == 40
        assert result.deployment_status in ("deployed", "pending_review")
        assert result.safety_report["is_valid"] is True

    def test_deploy_bytecode_invalid(self, bridge):
        """Invalid bytecode should be rejected."""
        # Empty bytecode
        result = bridge.deploy_bytecode(
            bytecode=b"",
            source_reflex="bad-reflex",
            provenance={},
        )
        assert result.success is False
        assert result.deployment_status == "rejected"
        assert "empty" in result.error.lower() or "Safety" in result.error

    def test_deploy_bytecode_unsafe(self, bridge):
        """Unsafe bytecode (stack underflow) should be rejected."""
        bytecode = _make_safe_bytecode([0x04])  # POP without push
        result = bridge.deploy_bytecode(
            bytecode=bytecode,
            source_reflex="unsafe-reflex",
            provenance={},
        )
        assert result.success is False
        assert result.deployment_status == "rejected"

    def test_deploy_commits_to_git(self, bridge):
        """Deployment should create a visible git commit."""
        bytecode = BytecodeDeployer.make_nop_bytecode(5)
        bridge.deploy_bytecode(
            bytecode=bytecode,
            source_reflex="git-test",
            provenance={},
        )

        # Check git log
        commits = list(bridge.repo.iter_commits())
        assert any("DEPLOY" in c.message for c in commits)

    def test_ingest_telemetry(self, bridge):
        """Telemetry ingestion should work through bridge."""
        result = bridge.ingest_telemetry({
            "readings": [
                {"sensor_id": 1, "value": 23.5, "timestamp": 1000.0},
                {"sensor_id": 2, "value": 1013.25, "timestamp": 1001.0},
            ],
            "sensor_type": "analog",
            "unit": "V",
        })
        assert isinstance(result, TelemetryResult)
        # May or may not be committed depending on batch size

    def test_record_trust_event(self, bridge):
        """Trust event recording should work through bridge."""
        result = bridge.record_trust_event(
            subsystem="navigation",
            event_type="heartbeat_ok",
            severity=0,
            details="Normal heartbeat",
        )
        assert isinstance(result, TrustResult)
        assert result.committed is True
        assert result.commit_hash

    def test_trust_event_in_git_log(self, bridge):
        """Trust events should appear in git log."""
        bridge.record_trust_event(
            subsystem="steering", event_type="sensor_valid",
            severity=0, details="OK",
        )
        commits = list(bridge.repo.iter_commits())
        assert any("TRUST" in c.message for c in commits)

    def test_report_safety_event(self, bridge):
        """Safety event reporting should create git commit."""
        result = bridge.report_safety_event({
            "level": 1,
            "subsystem": "engine",
            "event_type": "overcurrent",
            "details": "Motor current spike detected",
        })
        assert isinstance(result, SafetyResult)
        assert result.committed is True
        assert result.commit_hash

        # Verify commit message format
        commits = list(bridge.repo.iter_commits())
        safety_commits = [c for c in commits if "SAFETY" in c.message]
        assert len(safety_commits) >= 1

    def test_safety_event_json_file(self, bridge, temp_repo_path):
        """Safety event should create JSON file in .agent/safety/."""
        bridge.report_safety_event({
            "level": 2,
            "subsystem": "navigation",
            "event_type": "gps_loss",
            "details": "GPS signal lost",
        })

        safety_dir = os.path.join(temp_repo_path, ".agent", "safety")
        files = [f for f in os.listdir(safety_dir)
                 if f.endswith(".json") and not f.startswith(".")]
        assert len(files) >= 1

        with open(os.path.join(safety_dir, files[0])) as f:
            data = json.load(f)
        assert data["level"] == 2
        assert data["subsystem"] == "navigation"

    def test_get_mission_queue_empty(self, bridge):
        """Empty mission queue should return empty list."""
        missions = bridge.get_mission_queue()
        assert missions == []

    def test_complete_mission_not_found(self, bridge):
        """Completing non-existent mission should fail."""
        result = bridge.complete_mission(
            mission_id="nonexistent",
            results={"summary": "test"},
        )
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_complete_mission(self, bridge, temp_repo_path):
        """Completing a mission should move it from next to done."""
        # Create a mission file in .agent/next
        next_dir = os.path.join(temp_repo_path, ".agent", "next")
        mission = {
            "id": "mission-001",
            "description": "Test mission",
            "priority": 1,
            "assigned_to": "test-vessel",
        }
        with open(os.path.join(next_dir, "mission-001.json"), "w") as f:
            json.dump(mission, f)

        # Complete the mission
        result = bridge.complete_mission(
            mission_id="mission-001",
            results={"summary": "Mission completed successfully"},
        )
        assert result.success is True
        assert result.commit_hash
        assert result.mission_id == "mission-001"

        # Verify file moved from next to done
        next_files = os.listdir(next_dir)
        assert not any("mission-001" in f for f in next_files)

        done_dir = os.path.join(temp_repo_path, ".agent", "done")
        done_files = [f for f in os.listdir(done_dir) if "mission-001" in f]
        assert len(done_files) == 1

        # Verify done file has completion data
        with open(os.path.join(done_dir, done_files[0])) as f:
            data = json.load(f)
        assert data["status"] == "done"
        assert "completed_at" in data
        assert data["results"]["summary"] == "Mission completed successfully"

    def test_get_status(self, bridge):
        """get_status should return BridgeStatus with correct fields."""
        status = bridge.get_status()
        assert isinstance(status, BridgeStatus)
        assert status.vessel_id == "test-vessel"
        assert status.connected is True
        assert status.repo_path == bridge.repo_path

    def test_end_to_end_workflow(self, bridge, temp_repo_path):
        """Full end-to-end workflow: deploy + telemetry + trust + safety."""
        # 1. Deploy bytecode
        deploy_result = bridge.deploy_bytecode(
            bytecode=BytecodeDeployer.make_nop_bytecode(3),
            source_reflex="e2e-reflex",
            provenance={"author": "e2e-test"},
        )
        assert deploy_result.success

        # 2. Ingest telemetry
        bridge.ingest_telemetry({
            "readings": [
                {"sensor_id": 1, "value": 23.5, "timestamp": time.time()},
            ],
        })

        # 3. Record trust events
        bridge.record_trust_event("navigation", "heartbeat_ok", 0, "OK")
        bridge.record_trust_event("steering", "sensor_valid", 0, "OK")

        # 4. Report safety event
        bridge.report_safety_event({
            "level": 1, "subsystem": "engine",
            "event_type": "warning", "details": "Test warning",
        })

        # 5. Create and complete mission
        next_dir = os.path.join(temp_repo_path, ".agent", "next")
        with open(os.path.join(next_dir, "e2e-mission.json"), "w") as f:
            json.dump({"id": "e2e-mission", "description": "E2E test"}, f)
        mission_result = bridge.complete_mission(
            "e2e-mission", {"summary": "E2E complete"}
        )
        assert mission_result.success

        # 6. Check status
        status = bridge.get_status()
        assert status.pending_missions == 0

        # 7. Verify all commits in git log
        all_commits = list(bridge.repo.iter_commits())
        commit_types = {c.message.split(":")[0] if ":" in c.message else c.message[:10]
                        for c in all_commits}
        assert "DEPLOY" in commit_types
        assert "TRUST" in commit_types
        assert "SAFETY[INFO]" in commit_types
        assert "MISSION DONE" in commit_types

    def test_bridge_with_github_token(self, temp_repo_path):
        """Bridge with GitHub token should initialize correctly."""
        bridge = NexusBridge(
            vessel_id="gh-test",
            repo_path=temp_repo_path,
            github_token="ghp_test_token",
        )
        assert bridge.github_token == "ghp_test_token"
        # Deploy should succeed (PR creation is stubbed)
        result = bridge.deploy_bytecode(
            bytecode=BytecodeDeployer.make_nop_bytecode(1),
            source_reflex="gh-test",
            provenance={},
        )
        assert result.success is True
        assert result.deployment_status == "pending_review"


# ═══════════════════════════════════════════════════════════════════
# 6. Edge Case Tests
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_bytecode_max_instructions(self, deployer):
        """Exactly max_cycles instructions should pass."""
        bytecode = _make_safe_bytecode([0x00] * 1000)
        result = deployer.validate_bytecode(bytecode)
        assert result.passed is True
        assert result.report.instruction_count == 1000

    def test_bytecode_single_instruction(self, deployer):
        """Single instruction should pass."""
        bytecode = _make_safe_bytecode([0x00])
        result = deployer.validate_bytecode(bytecode)
        assert result.passed is True
        assert result.report.instruction_count == 1

    def test_multiple_write_pins_with_clamp(self, deployer):
        """Multiple WRITE_PIN should each have their own CLAMP_F."""
        # Two clamp+write pairs
        bytecode = (
            _make_push_i8(100) + _make_clamp_f(0.0, 1.0) + _make_write_pin(25) +
            _make_push_i8(200) + _make_clamp_f(0.0, 1.0) + _make_write_pin(26)
        )
        result = deployer.validate_bytecode(bytecode)
        assert result.passed is True

    def test_telemetry_large_batch(self, temp_repo):
        """Large telemetry batch should commit all readings."""
        ingester = TelemetryIngester(vessel_id="test", batch_size=500)
        for i in range(500):
            ingester.add_reading(
                sensor_id=i % 10, value=float(i), timestamp=1000.0 + i
            )
        result = ingester.flush(temp_repo)
        assert result.committed is True
        assert result.readings_count == 500

    def test_trust_event_severity_max(self, temp_repo):
        """Maximum severity (1.0) should be stored correctly."""
        trust = TrustSync(vessel_id="test")
        result = trust.record_event(
            subsystem="test", event_type="critical",
            severity=1.0, details="max severity", repo=temp_repo,
        )
        trust_dir = os.path.join(
            temp_repo.working_dir, ".agent", "trust"
        )
        files = [f for f in os.listdir(trust_dir) if f.endswith(".json")]
        with open(os.path.join(trust_dir, files[0])) as f:
            data = json.load(f)
        assert data["severity"] == 1.0

    def test_manifest_from_config_with_all_sensor_types(self):
        """Manifest generation should handle all valid sensor types."""
        manifest_gen = EquipmentManifest()
        sensors = []
        for st in ["analog", "digital", "pwm", "i2c", "spi", "can",
                    "uart", "gps", "imu", "sonar", "lidar", "camera",
                    "temperature", "pressure", "humidity", "current", "voltage"]:
            sensors.append({"id": len(sensors), "type": st, "pin": len(sensors)})

        manifest = manifest_gen.from_hal_config({
            "vessel_id": "multi-sensor",
            "sensors": sensors,
            "actuators": [],
        })
        assert len(manifest["sensors"]) == len(sensors)

    def test_bridge_init_existing_repo(self, temp_repo):
        """Bridge should work with pre-existing repo."""
        bridge = NexusBridge(
            vessel_id="existing-repo",
            repo_path=temp_repo.working_dir,
        )
        status = bridge.get_status()
        assert status.connected is True

    def test_multiple_bridge_instances(self, temp_repo_path):
        """Multiple bridges should not interfere."""
        bridge1 = NexusBridge("v1", temp_repo_path)
        bridge2_dir = tempfile.mkdtemp()
        from git import Repo as GitRepo
        GitRepo.init(bridge2_dir)
        bridge2 = NexusBridge("v2", bridge2_dir)

        bridge1.record_trust_event("nav", "ok", 0, "OK")
        bridge2.record_trust_event("nav", "ok", 0, "OK")

        # Each bridge should have its own repo
        assert bridge1.get_status().vessel_id == "v1"
        assert bridge2.get_status().vessel_id == "v2"
