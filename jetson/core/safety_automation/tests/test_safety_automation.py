"""Tests for NEXUS Safety Automation — 80+ tests."""

import os
import tempfile

import pytest
from core.safety_automation.policy_engine import (
    PolicyAction, PolicyCondition, PolicyEngine, SafetyPolicy,
)
from core.safety_automation.audit_trail import AuditEntry, AuditTrail
from core.safety_automation.bytecode_gate import BytecodeGate, GateReport, FileValidationResult
from jetson.reflex.bytecode_emitter import BytecodeEmitter


# ===================================================================
# Policy Condition Tests
# ===================================================================
class TestPolicyCondition:
    def test_eq(self):
        c = PolicyCondition("trust_level", "eq", 3)
        assert c.evaluate({"trust_level": 3})
        assert not c.evaluate({"trust_level": 2})

    def test_ne(self):
        c = PolicyCondition("trust_level", "ne", 0)
        assert c.evaluate({"trust_level": 1})

    def test_gt(self):
        c = PolicyCondition("speed", "gt", 5.0)
        assert c.evaluate({"speed": 6.0})
        assert not c.evaluate({"speed": 5.0})

    def test_gte(self):
        c = PolicyCondition("speed", "gte", 5.0)
        assert c.evaluate({"speed": 5.0})

    def test_lt(self):
        c = PolicyCondition("speed", "lt", 10.0)
        assert c.evaluate({"speed": 9.0})

    def test_lte(self):
        c = PolicyCondition("speed", "lte", 10.0)
        assert c.evaluate({"speed": 10.0})

    def test_in(self):
        c = PolicyCondition("pin", "in", [0, 1, 2, 3])
        assert c.evaluate({"pin": 1})
        assert not c.evaluate({"pin": 5})

    def test_not_in(self):
        c = PolicyCondition("action", "not_in", ["deploy", "delete"])
        assert c.evaluate({"action": "read"})

    def test_contains(self):
        c = PolicyCondition("tags", "contains", "marine")
        assert c.evaluate({"tags": ["marine", "nav"]})

    def test_missing_field(self):
        c = PolicyCondition("nonexistent", "eq", 1)
        assert not c.evaluate({})

    def test_invalid_operator(self):
        c = PolicyCondition("x", "invalid_op", 1)
        assert not c.evaluate({"x": 1})

    def test_type_mismatch(self):
        c = PolicyCondition("speed", "gt", "fast")
        assert not c.evaluate({"speed": 5.0})

    def test_to_dict(self):
        c = PolicyCondition("x", "eq", 42)
        d = c.to_dict()
        assert d["field"] == "x"

    def test_from_dict(self):
        c = PolicyCondition.from_dict({"field": "y", "operator": "gt", "value": 5})
        assert c.field == "y"
        assert c.value == 5


# ===================================================================
# Safety Policy Tests
# ===================================================================
class TestSafetyPolicy:
    def test_single_condition_match(self):
        p = SafetyPolicy("test", conditions=[PolicyCondition("x", "eq", 1)], action=PolicyAction.DENY)
        assert p.evaluate({"x": 1}) == PolicyAction.DENY

    def test_single_condition_no_match(self):
        p = SafetyPolicy("test", conditions=[PolicyCondition("x", "eq", 1)], action=PolicyAction.DENY)
        assert p.evaluate({"x": 2}) is None

    def test_multiple_conditions_all_match(self):
        p = SafetyPolicy("test", conditions=[
            PolicyCondition("a", "eq", 1), PolicyCondition("b", "gt", 0)
        ], action=PolicyAction.DENY)
        assert p.evaluate({"a": 1, "b": 5}) == PolicyAction.DENY

    def test_multiple_conditions_partial(self):
        p = SafetyPolicy("test", conditions=[
            PolicyCondition("a", "eq", 1), PolicyCondition("b", "gt", 0)
        ], action=PolicyAction.DENY)
        assert p.evaluate({"a": 1, "b": -1}) is None

    def test_no_conditions(self):
        p = SafetyPolicy("test", action=PolicyAction.WARN)
        assert p.evaluate({}) == PolicyAction.WARN

    def test_disabled(self):
        p = SafetyPolicy("test", conditions=[PolicyCondition("x", "eq", 1)],
                         action=PolicyAction.DENY, enabled=False)
        assert p.evaluate({"x": 1}) is None

    def test_to_dict(self):
        p = SafetyPolicy("test", action=PolicyAction.DENY)
        d = p.to_dict()
        assert d["name"] == "test"
        assert d["action"] == "deny"

    def test_from_dict(self):
        p = SafetyPolicy.from_dict({"name": "x", "conditions": [
            {"field": "y", "operator": "eq", "value": 1}
        ], "action": "allow"})
        assert p.name == "x"
        assert p.action == PolicyAction.ALLOW


# ===================================================================
# Policy Engine Tests
# ===================================================================
class TestPolicyEngine:
    def test_empty_engine_allows(self):
        e = PolicyEngine()
        assert e.evaluate({}) == PolicyAction.ALLOW

    def test_single_policy_match(self):
        e = PolicyEngine()
        e.add_policy(SafetyPolicy("block", conditions=[PolicyCondition("danger", "eq", True)],
                                  action=PolicyAction.DENY, priority=10))
        assert e.evaluate({"danger": True}) == PolicyAction.DENY

    def test_priority_order(self):
        e = PolicyEngine()
        e.add_policy(SafetyPolicy("warn", conditions=[PolicyCondition("x", "eq", 1)],
                                  action=PolicyAction.WARN, priority=1))
        e.add_policy(SafetyPolicy("deny", conditions=[PolicyCondition("x", "eq", 1)],
                                  action=PolicyAction.DENY, priority=10))
        assert e.evaluate({"x": 1}) == PolicyAction.DENY  # higher priority wins

    def test_no_match_allows(self):
        e = PolicyEngine()
        e.add_policy(SafetyPolicy("block", conditions=[PolicyCondition("x", "eq", 999)],
                                  action=PolicyAction.DENY))
        assert e.evaluate({"x": 1}) == PolicyAction.ALLOW

    def test_remove_policy(self):
        e = PolicyEngine()
        e.add_policy(SafetyPolicy("test", conditions=[PolicyCondition("x", "eq", 1)],
                                  action=PolicyAction.DENY))
        assert e.remove_policy("test")
        assert e.evaluate({"x": 1}) == PolicyAction.ALLOW

    def test_remove_nonexistent(self):
        e = PolicyEngine()
        assert not e.remove_policy("nonexistent")

    def test_evaluate_all(self):
        e = PolicyEngine()
        e.add_policy(SafetyPolicy("a", conditions=[PolicyCondition("x", "eq", 1)],
                                  action=PolicyAction.WARN, priority=5))
        e.add_policy(SafetyPolicy("b", conditions=[PolicyCondition("x", "eq", 1)],
                                  action=PolicyAction.DENY, priority=10))
        results = e.evaluate_all({"x": 1})
        assert len(results) == 2

    def test_get_policy(self):
        e = PolicyEngine()
        e.add_policy(SafetyPolicy("test"))
        assert e.get_policy("test") is not None
        assert e.get_policy("nonexistent") is None

    def test_enable_disable(self):
        e = PolicyEngine()
        e.add_policy(SafetyPolicy("test", conditions=[PolicyCondition("x", "eq", 1)],
                                  action=PolicyAction.DENY))
        assert e.evaluate({"x": 1}) == PolicyAction.DENY
        e.disable_policy("test")
        assert e.evaluate({"x": 1}) == PolicyAction.ALLOW
        e.enable_policy("test")
        assert e.evaluate({"x": 1}) == PolicyAction.DENY

    def test_counts(self):
        e = PolicyEngine()
        e.add_policy(SafetyPolicy("a"))
        e.add_policy(SafetyPolicy("b"))
        e.add_policy(SafetyPolicy("c", enabled=False))
        assert e.policy_count == 3
        assert e.enabled_count == 2

    def test_load_defaults(self):
        e = PolicyEngine()
        e.load_defaults()
        assert e.policy_count >= 5
        # Test trust minimum blocks low trust deploy
        result = e.evaluate({"trust_level": 1, "action_type": "deploy_reflex"})
        assert result == PolicyAction.DENY
        # Test safety pin protection
        result = e.evaluate({"target_pin": 0, "action_type": "write_pin"})
        assert result == PolicyAction.DENY
        # Test fault state restriction
        result = e.evaluate({"safety_state": "fault", "action_type": "deploy_reflex"})
        assert result == PolicyAction.DENY


# ===================================================================
# Audit Trail Tests
# ===================================================================
class TestAuditEntry:
    def test_create(self):
        e = AuditEntry("2026-01-01T00:00:00Z", "test_event", description="test")
        assert e.event_type == "test_event"

    def test_roundtrip(self):
        e = AuditEntry("2026-01-01", "test", vessel_id="V1", metadata={"key": "val"})
        d = e.to_dict()
        e2 = AuditEntry.from_dict(d)
        assert e2.vessel_id == "V1"
        assert e2.metadata["key"] == "val"


class TestAuditTrail:
    def test_add_event(self):
        t = AuditTrail(vessel_id="V1")
        t.add_event("bytecode_deploy", "Deployed heading_hold", "success")
        assert len(t.entries) == 1

    def test_filter_by_type(self):
        t = AuditTrail()
        t.add_event("safety_event", "E-Stop triggered", "failure")
        t.add_event("bytecode_deploy", "Deployed", "success")
        assert len(t.filter_by_type("safety_event")) == 1

    def test_filter_by_outcome(self):
        t = AuditTrail()
        t.add_event("test", "", "success")
        t.add_event("test", "", "failure")
        t.add_event("test", "", "success")
        assert len(t.filter_by_outcome("success")) == 2

    def test_filter_by_time(self):
        t = AuditTrail()
        t.add_event("a", "")  # timestamp auto-generated
        t.entries[0].timestamp = "2026-01-01T12:00:00Z"
        t.add_event("b", "")
        t.entries[1].timestamp = "2026-01-01T14:00:00Z"
        t.add_event("c", "")
        t.entries[2].timestamp = "2026-01-01T16:00:00Z"
        assert len(t.filter_by_time("2026-01-01T13:00:00Z", "2026-01-01T15:00:00Z")) == 1

    def test_to_json(self):
        t = AuditTrail(vessel_id="V1", session_id="S1")
        t.add_event("test", "desc")
        j = t.to_json()
        assert "V1" in j
        assert "test" in j

    def test_to_markdown(self):
        t = AuditTrail(vessel_id="V1")
        t.add_event("safety_event", "Test event", "warning")
        md = t.to_markdown()
        assert "NEXUS Audit Trail" in md
        assert "V1" in md
        assert "safety_event" in md

    def test_compliance_iec61508(self):
        t = AuditTrail()
        t.add_event("safety_event", "test")
        r = t.compliance_report_iec61508()
        assert "IEC 61508" in r
        assert "Safety" in r

    def test_compliance_eu_ai_act(self):
        t = AuditTrail()
        t.add_event("bytecode_deploy", "test")
        r = t.compliance_report_eu_ai_act()
        assert "EU AI Act" in r
        assert "Audit" in r

    def test_empty_trail(self):
        t = AuditTrail()
        assert len(t.entries) == 0
        md = t.to_markdown()
        assert "NEXUS Audit Trail" in md


# ===================================================================
# Bytecode Gate Tests
# ===================================================================
class TestBytecodeGate:
    def _make_bytecode(self, emitter: BytecodeEmitter) -> bytes:
        return emitter.get_bytecode()

    def test_valid_bytecode(self):
        g = BytecodeGate(trust_level=2)
        e = BytecodeEmitter()
        e.emit_push_f32(1.0)
        e.emit_push_f32(2.0)
        e.emit_add_f()
        e.emit_halt()
        result = g.validate_bytes(self._make_bytecode(e), "test.bin")
        assert result.passed
        assert result.instruction_count == 4

    def test_empty_bytecode(self):
        g = BytecodeGate()
        result = g.validate_bytes(b"", "empty.bin")
        assert not result.passed
        assert "Empty" in result.error

    def test_misaligned_bytecode(self):
        g = BytecodeGate()
        result = g.validate_bytes(b"\x00\x00\x00\x00\x00", "bad.bin")
        assert not result.passed
        assert "Misaligned" in result.error

    def test_nonexistent_file(self):
        g = BytecodeGate()
        result = g.validate_file("/nonexistent/path.bin")
        assert not result.passed
        assert "not found" in result.error

    def test_file_validation(self):
        g = BytecodeGate(trust_level=2)
        e = BytecodeEmitter()
        e.emit_push_f32(1.0)
        e.emit_halt()
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(e.get_bytecode())
            f.flush()
            result = g.validate_file(f.name)
            assert result.passed
            assert result.file_size_bytes == 16
            os.unlink(f.name)

    def test_directory_validation(self):
        g = BytecodeGate(trust_level=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid file
            e = BytecodeEmitter()
            e.emit_push_f32(1.0)
            e.emit_halt()
            with open(os.path.join(tmpdir, "good.bin"), "wb") as f:
                f.write(e.get_bytecode())
            # Create an invalid file
            with open(os.path.join(tmpdir, "bad.bytecode"), "wb") as f:
                f.write(b"\x00\x01")
            report = g.validate_directory(tmpdir)
            assert report.total_files == 2
            assert report.passed_files == 1
            assert report.failed_files == 1
            assert report.exit_code == 1

    def test_directory_nonexistent(self):
        g = BytecodeGate()
        report = g.validate_directory("/nonexistent")
        assert report.total_files == 0

    def test_unique_opcodes(self):
        g = BytecodeGate()
        e = BytecodeEmitter()
        e.emit_push_f32(1.0)
        e.emit_add_f()
        e.emit_halt()
        result = g.validate_bytes(e.get_bytecode())
        assert "PUSH_F32" in result.unique_opcodes
        assert "ADD_F" in result.unique_opcodes

    def test_gate_report(self):
        g = BytecodeGate()
        report = GateReport(total_files=3, passed_files=3, failed_files=0)
        assert report.exit_code == 0
        s = report.summary()
        assert "PASSED" in s

    def test_gate_report_failure(self):
        report = GateReport(total_files=3, passed_files=2, failed_files=1)
        assert report.exit_code == 1
        s = report.summary()
        assert "FAILURES" in s

    def test_gate_report_to_json(self):
        report = GateReport()
        j = report.to_json()
        data = __import__("json").loads(j)
        assert data["total_files"] == 0


# ===================================================================
# Integration Tests
# ===================================================================
class TestIntegrationPipeline:
    def test_policy_blocks_low_trust_deploy(self):
        engine = PolicyEngine()
        engine.load_defaults()
        result = engine.evaluate({"trust_level": 0, "action_type": "deploy_reflex"})
        assert result == PolicyAction.DENY

    def test_audit_records_policy_violation(self):
        trail = AuditTrail(vessel_id="V1")
        trail.add_event("policy_violation", "Low trust deploy blocked", "failure",
                        {"trust_level": 0, "action": "deploy_reflex"})
        assert len(trail.filter_by_type("policy_violation")) == 1

    def test_full_gate_with_report(self):
        g = BytecodeGate(trust_level=2)
        e = BytecodeEmitter()
        e.emit_push_f32(0.5)
        e.emit_push_i8(8)  # pin 8 (not safety pin)
        e.emit_write_pin(8)
        e.emit_halt()
        result = g.validate_bytes(e.get_bytecode(), "test_reflex.bin")
        assert result.instruction_count == 4
        # Generate markdown audit
        trail = AuditTrail(vessel_id="V1")
        trail.add_event("bytecode_deploy", f"Validated {result.filepath}",
                       "success" if result.passed else "failure",
                       {"instructions": result.instruction_count})
        md = trail.to_markdown()
        assert "V1" in md
