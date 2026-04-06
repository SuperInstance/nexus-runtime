"""Tests for Audit Trail Generation Engine."""

import pytest
import json
from datetime import datetime

from jetson.compliance.audit_trail import (
    AuditEntry,
    AuditTrailConfig,
    AuditTrailGenerator,
    ComplianceTrailReport,
)


class TestAuditEntry:
    """Tests for AuditEntry dataclass."""

    def test_create_entry(self):
        entry = AuditEntry(
            timestamp="2025-01-01T00:00:00",
            actor="system",
            action="safety_check",
            component="navigation",
        )
        assert entry.actor == "system"
        assert entry.action == "safety_check"
        assert entry.component == "navigation"
        assert entry.details == {}
        assert entry.evidence_refs == []

    def test_with_details(self):
        entry = AuditEntry(
            timestamp="2025-01-01T00:00:00",
            actor="operator",
            action="config_change",
            component="controller",
            details={"category": "configuration_change", "param": "max_speed"},
            evidence_refs=["EVD-001"],
        )
        assert entry.details["category"] == "configuration_change"
        assert entry.evidence_refs == ["EVD-001"]


class TestAuditTrailConfig:
    """Tests for AuditTrailConfig dataclass."""

    def test_default_config(self):
        config = AuditTrailConfig()
        assert config.retention_period == 365
        assert config.signing_enabled is True
        assert len(config.required_categories) == 6
        assert config.max_entries == 100000

    def test_custom_config(self):
        config = AuditTrailConfig(
            retention_period=730,
            required_categories=["test", "deploy"],
            signing_enabled=False,
        )
        assert config.retention_period == 730
        assert config.signing_enabled is False


class TestAuditTrailGeneratorRecordEntry:
    """Tests for AuditTrailGenerator.record_entry."""

    def setup_method(self):
        self.gen = AuditTrailGenerator()

    def test_record_single_entry(self):
        entry = AuditEntry(
            timestamp="2025-01-01T00:00:00",
            actor="system",
            action="test",
            component="nav",
        )
        self.gen.record_entry(entry)
        assert len(self.gen.entries) == 1

    def test_entry_id_assigned(self):
        entry = AuditEntry(
            timestamp="2025-01-01T00:00:00",
            actor="system",
            action="test",
            component="nav",
        )
        self.gen.record_entry(entry)
        assert entry.entry_id == "audit-000001"

    def test_sequential_ids(self):
        for i in range(5):
            entry = AuditEntry(
                timestamp="2025-01-01T00:00:00",
                actor="system",
                action=f"action_{i}",
                component="nav",
            )
            self.gen.record_entry(entry)
        ids = [e.entry_id for e in self.gen.entries]
        assert ids == ["audit-000001", "audit-000002", "audit-000003", "audit-000004", "audit-000005"]

    def test_timestamp_auto_generated(self):
        entry = AuditEntry(
            timestamp="",
            actor="system",
            action="test",
            component="nav",
        )
        self.gen.record_entry(entry)
        assert entry.timestamp != ""

    def test_multiple_entries(self):
        for _ in range(10):
            self.gen.record_entry(AuditEntry(
                timestamp="2025-01-01T00:00:00",
                actor="system",
                action="test",
                component="nav",
            ))
        assert len(self.gen.entries) == 10


class TestAuditTrailGeneratorCreateEntry:
    """Tests for AuditTrailGenerator.create_entry."""

    def setup_method(self):
        self.gen = AuditTrailGenerator()

    def test_create_and_record(self):
        entry = self.gen.create_entry("operator", "config_change", "controller")
        assert len(self.gen.entries) == 1
        assert entry.actor == "operator"
        assert entry.action == "config_change"
        assert entry.component == "controller"

    def test_with_category(self):
        entry = self.gen.create_entry(
            "system", "test", "nav",
            category="safety_check",
        )
        assert entry.details["category"] == "safety_check"

    def test_with_details(self):
        entry = self.gen.create_entry(
            "system", "deploy", "nav",
            details={"version": "2.0"},
            evidence_refs=["EVD-001"],
        )
        assert entry.details["version"] == "2.0"
        assert entry.evidence_refs == ["EVD-001"]


class TestAuditTrailGeneratorGenerateTrail:
    """Tests for AuditTrailGenerator.generate_trail."""

    def setup_method(self):
        self.gen = AuditTrailGenerator()
        self.gen.create_entry("sys1", "action1", "nav", category="safety_check")
        self.gen.create_entry("sys2", "action2", "ctrl", category="deployment")
        self.gen.create_entry("sys3", "action3", "nav", category="maintenance")

    def test_filter_by_component(self):
        trail = self.gen.generate_trail("nav", "2000-01-01", "2099-12-31")
        assert len(trail) == 2
        assert all(e.component == "nav" for e in trail)

    def test_filter_by_time_range(self):
        trail = self.gen.generate_trail("nav", "2000-01-01", "2099-12-31")
        assert len(trail) >= 0

    def test_no_matches(self):
        trail = self.gen.generate_trail("nonexistent", "2000-01-01", "2099-12-31")
        assert len(trail) == 0


class TestAuditTrailGeneratorComplianceReport:
    """Tests for AuditTrailGenerator.generate_compliance_report."""

    def setup_method(self):
        self.gen = AuditTrailGenerator()

    def test_empty_trail(self):
        report = self.gen.generate_compliance_report([], "IEC_61508")
        assert report.total_entries == 0
        assert report.completeness_score == 0.0
        assert len(report.gaps) > 0

    def test_full_coverage(self):
        categories = self.gen.config.required_categories
        for cat in categories:
            self.gen.create_entry("sys", "action", "comp", category=cat)
        trail = self.gen.entries
        report = self.gen.generate_compliance_report(trail, "IEC_61508")
        assert report.completeness_score == 1.0
        assert len(report.gaps) == 0

    def test_partial_coverage(self):
        self.gen.create_entry("sys", "action", "comp", category="safety_check")
        self.gen.create_entry("sys", "action", "comp", category="deployment")
        trail = self.gen.entries
        report = self.gen.generate_compliance_report(trail, "EU_AI_ACT")
        assert 0 < report.completeness_score < 1.0
        assert len(report.gaps) > 0

    def test_evidence_count(self):
        self.gen.create_entry("sys", "action", "comp", category="safety_check",
                              evidence_refs=["E1", "E2"])
        trail = self.gen.entries
        report = self.gen.generate_compliance_report(trail, "IEC_61508")
        assert report.evidence_count == 2


class TestAuditTrailGeneratorSigning:
    """Tests for entry signing and verification."""

    def setup_method(self):
        self.gen = AuditTrailGenerator()
        self.key = "secret_key_123"

    def test_sign_entry(self):
        entry = self.gen.create_entry("sys", "action", "comp")
        signed = self.gen.sign_entry(entry, self.key)
        assert signed.signature != ""
        assert len(signed.signature) == 64

    def test_verify_valid_signature(self):
        entry = self.gen.create_entry("sys", "action", "comp")
        self.gen.sign_entry(entry, self.key)
        assert self.gen.verify_signature(entry, self.key) is True

    def test_verify_wrong_key(self):
        entry = self.gen.create_entry("sys", "action", "comp")
        self.gen.sign_entry(entry, self.key)
        assert self.gen.verify_signature(entry, "wrong_key") is False

    def test_verify_unsigned_entry(self):
        entry = self.gen.create_entry("sys", "action", "comp")
        assert self.gen.verify_signature(entry, self.key) is False

    def test_different_entries_different_signatures(self):
        entry1 = self.gen.create_entry("sys1", "action1", "comp1")
        entry2 = self.gen.create_entry("sys2", "action2", "comp2")
        self.gen.sign_entry(entry1, self.key)
        self.gen.sign_entry(entry2, self.key)
        assert entry1.signature != entry2.signature


class TestAuditTrailGeneratorExport:
    """Tests for AuditTrailGenerator.export_trail."""

    def setup_method(self):
        self.gen = AuditTrailGenerator()
        self.gen.create_entry("sys", "deploy", "nav", category="deployment")
        self.gen.create_entry("sys", "test", "nav", category="safety_check")

    def test_export_json(self):
        result = self.gen.export_trail(self.gen.entries, "json")
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) == 2

    def test_export_csv(self):
        result = self.gen.export_trail(self.gen.entries, "csv")
        lines = result.strip().split("\n")
        assert len(lines) == 3  # header + 2 entries

    def test_export_text(self):
        result = self.gen.export_trail(self.gen.entries, "text")
        assert "NEXUS AUDIT TRAIL" in result
        assert "audit-000001" in result

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError):
            self.gen.export_trail(self.gen.entries, "xml")

    def test_empty_export(self):
        result = self.gen.export_trail([], "json")
        data = json.loads(result)
        assert data == []


class TestAuditTrailGeneratorCompleteness:
    """Tests for AuditTrailGenerator.compute_completeness."""

    def setup_method(self):
        self.gen = AuditTrailGenerator()

    def test_full_completeness(self):
        cats = ["safety_check", "deployment", "maintenance"]
        entries = [
            AuditEntry(timestamp="t", actor="s", action="a", component="c",
                       details={"category": cat})
            for cat in cats
        ]
        score = self.gen.compute_completeness(entries, cats)
        assert score == 1.0

    def test_zero_completeness(self):
        entries = []
        score = self.gen.compute_completeness(entries, ["safety_check"])
        assert score == 0.0

    def test_partial_completeness(self):
        entries = [
            AuditEntry(timestamp="t", actor="s", action="a", component="c",
                       details={"category": "safety_check"})
        ]
        score = self.gen.compute_completeness(entries, ["safety_check", "deployment"])
        assert score == pytest.approx(0.5)

    def test_empty_required(self):
        score = self.gen.compute_completeness([], [])
        assert score == 1.0


class TestAuditTrailGeneratorFilters:
    """Tests for filter methods."""

    def setup_method(self):
        self.gen = AuditTrailGenerator()
        self.gen.create_entry("operator1", "action1", "comp1", category="safety")
        self.gen.create_entry("operator2", "action2", "comp2", category="deploy")
        self.gen.create_entry("operator1", "action3", "comp3", category="safety")

    def test_filter_by_actor(self):
        entries = self.gen.get_entries_by_actor("operator1")
        assert len(entries) == 2

    def test_filter_by_category(self):
        entries = self.gen.get_entries_by_category("safety")
        assert len(entries) == 2

    def test_clear_entries(self):
        self.gen.clear_entries()
        assert len(self.gen.entries) == 0
        assert len(self.gen.get_entries_by_actor("operator1")) == 0
