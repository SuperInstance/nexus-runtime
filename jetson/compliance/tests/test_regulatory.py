"""Tests for Regulatory Change Tracking Engine."""

import pytest
from datetime import datetime

from jetson.compliance.regulatory import (
    RegulationVersion,
    ChangeImpact,
    ComplianceGap,
    ComplianceGapReport,
    RegulatoryTracker,
)


class TestRegulationVersion:
    """Tests for RegulationVersion dataclass."""

    def test_create_regulation(self):
        reg = RegulationVersion(
            id="IEC_61508",
            title="Functional Safety of Electrical/Electronic Systems",
            version="2.0",
            effective_date="2024-01-01",
        )
        assert reg.id == "IEC_61508"
        assert reg.version == "2.0"
        assert reg.status == "active"

    def test_with_changes(self):
        reg = RegulationVersion(
            id="EU_AI_ACT",
            title="EU AI Act",
            version="1.0",
            effective_date="2025-01-01",
            changes=[
                {"id": "C1", "description": "New risk categories", "change_type": "new_requirement"},
            ],
        )
        assert len(reg.changes) == 1


class TestChangeImpact:
    """Tests for ChangeImpact dataclass."""

    def test_create_impact(self):
        impact = ChangeImpact(
            regulation="IEC_61508",
            change="Updated SIL requirements",
            affected_components=["nav", "safety"],
            severity="high",
            action_required="Update SIL verification",
        )
        assert impact.severity == "high"
        assert len(impact.affected_components) == 2

    def test_default_severity(self):
        impact = ChangeImpact(regulation="X", change="Y")
        assert impact.severity == "low"


class TestComplianceGap:
    """Tests for ComplianceGap dataclass."""

    def test_create_gap(self):
        gap = ComplianceGap(
            requirement="REQ-001",
            current_status="not_compliant",
            required_status="compliant",
            severity="high",
            remediation="Implement requirement",
        )
        assert gap.severity == "high"


class TestRegulatoryTrackerAddRegulation:
    """Tests for RegulatoryTracker.add_regulation."""

    def setup_method(self):
        self.tracker = RegulatoryTracker()

    def test_add_regulation(self):
        reg = RegulationVersion(
            id="IEC_61508", title="Test", version="1.0", effective_date="2024-01-01",
        )
        self.tracker.add_regulation(reg)
        assert "IEC_61508" in self.tracker.list_regulations()

    def test_add_multiple(self):
        for i in range(5):
            self.tracker.add_regulation(RegulationVersion(
                id=f"REG-{i}", title=f"Regulation {i}", version="1.0",
                effective_date="2024-01-01",
            ))
        assert len(self.tracker.list_regulations()) == 5

    def test_overwrite_regulation(self):
        self.tracker.add_regulation(RegulationVersion(
            id="IEC_61508", title="V1", version="1.0", effective_date="2024-01-01",
        ))
        self.tracker.add_regulation(RegulationVersion(
            id="IEC_61508", title="V2", version="2.0", effective_date="2025-01-01",
        ))
        reg = self.tracker.get_regulation("IEC_61508")
        assert reg.version == "2.0"
        assert reg.title == "V2"


class TestRegulatoryTrackerCheckVersion:
    """Tests for RegulatoryTracker.check_version."""

    def setup_method(self):
        self.tracker = RegulatoryTracker()
        self.tracker.add_regulation(RegulationVersion(
            id="IEC_61508", title="Test", version="2.0", effective_date="2024-01-01",
            changes=[
                {"id": "C1", "description": "Updated SIL tables"},
                {"id": "C2", "description": "New PFD formulas"},
            ],
        ))

    def test_up_to_date(self):
        up_to_date, changes = self.tracker.check_version("IEC_61508", "2.0")
        assert up_to_date is True
        assert changes == []

    def test_outdated(self):
        up_to_date, changes = self.tracker.check_version("IEC_61508", "1.0")
        assert up_to_date is False
        assert len(changes) == 2

    def test_unknown_regulation(self):
        up_to_date, changes = self.tracker.check_version("UNKNOWN", "1.0")
        assert up_to_date is False
        assert any("not found" in c.get("error", "") for c in changes)


class TestRegulatoryTrackerImpact:
    """Tests for RegulatoryTracker.assess_impact."""

    def setup_method(self):
        self.tracker = RegulatoryTracker()

    def test_prohibition_severity(self):
        impact = self.tracker.assess_impact(
            {"description": "Ban social scoring", "change_type": "prohibition", "affected_areas": ["ai_systems"]},
            {"nav": {"categories": ["ai_systems"], "regulated_by": ["EU_AI_ACT"]}},
        )
        assert impact.severity == "critical"

    def test_new_requirement_high(self):
        impact = self.tracker.assess_impact(
            {"description": "New logging requirement", "change_type": "new_requirement", "affected_areas": ["logging"]},
            {"nav": {"categories": ["navigation"], "regulated_by": ["logging"]}},
        )
        assert impact.severity == "high"

    def test_amendment_medium(self):
        impact = self.tracker.assess_impact(
            {"description": "Minor update", "change_type": "amendment", "affected_areas": ["navigation"]},
            {"nav": {"categories": ["navigation"], "regulated_by": []}},
        )
        assert impact.severity == "medium"

    def test_clarification_low(self):
        impact = self.tracker.assess_impact(
            {"description": "Clarified language", "change_type": "clarification", "affected_areas": []},
            {},
        )
        assert impact.severity == "low"

    def test_no_affected_components(self):
        impact = self.tracker.assess_impact(
            {"description": "Minor update", "change_type": "amendment", "affected_areas": ["quantum"]},
            {"nav": {"categories": ["navigation"]}},
        )
        assert impact.affected_components == []

    def test_repeal_medium(self):
        impact = self.tracker.assess_impact(
            {"description": "Removed old rule", "change_type": "repeal", "affected_areas": []},
            {},
        )
        assert impact.severity == "medium"

    def test_amendment_high_multiple_affected(self):
        impact = self.tracker.assess_impact(
            {"description": "Major update", "change_type": "amendment", "affected_areas": ["nav", "safety", "comm"]},
            {
                "nav": {"categories": ["nav"], "regulated_by": []},
                "safety": {"categories": ["safety"], "regulated_by": []},
                "comm": {"categories": ["comm"], "regulated_by": []},
            },
        )
        assert impact.severity == "high"


class TestRegulatoryTrackerGapAnalysis:
    """Tests for RegulatoryTracker.generate_compliance_gap."""

    def setup_method(self):
        self.tracker = RegulatoryTracker()
        self.tracker.add_regulation(RegulationVersion(
            id="IEC_61508", title="Functional Safety", version="2.0",
            effective_date="2024-01-01",
            changes=[
                {"id": "REQ-1", "description": "SIL verification", "change_type": "new_requirement"},
                {"id": "REQ-2", "description": "FMEA documentation", "change_type": "amendment"},
            ],
        ))

    def test_fully_compliant(self):
        system = {
            "name": "Nav System",
            "current_compliance": {"REQ-1": True, "REQ-2": True},
        }
        report = self.tracker.generate_compliance_gap(system, ["IEC_61508"])
        assert report.overall_score == 100.0
        assert report.gaps_found == 0

    def test_partial_compliance(self):
        system = {
            "name": "Nav System",
            "current_compliance": {"REQ-1": True},
        }
        report = self.tracker.generate_compliance_gap(system, ["IEC_61508"])
        assert 0 < report.overall_score < 100
        assert report.gaps_found == 1

    def test_no_compliance(self):
        system = {"name": "Nav System"}
        report = self.tracker.generate_compliance_gap(system, ["IEC_61508"])
        assert report.overall_score == 0.0
        assert report.gaps_found == 2

    def test_unknown_regulation(self):
        system = {"name": "Nav System"}
        report = self.tracker.generate_compliance_gap(system, ["UNKNOWN_REG"])
        assert report.gaps_found >= 1

    def test_multiple_regulations(self):
        self.tracker.add_regulation(RegulationVersion(
            id="EU_AI_ACT", title="EU AI Act", version="1.0",
            effective_date="2025-01-01",
            changes=[
                {"id": "AI-REQ-1", "description": "Risk management", "change_type": "new_requirement"},
            ],
        ))
        system = {
            "name": "Nav System",
            "current_compliance": {"REQ-1": True, "REQ-2": True, "AI-REQ-1": True},
        }
        report = self.tracker.generate_compliance_gap(system, ["IEC_61508", "EU_AI_ACT"])
        assert report.overall_score == 100.0
        assert report.total_requirements == 3


class TestRegulatoryTrackerTrackChanges:
    """Tests for RegulatoryTracker.track_changes."""

    def setup_method(self):
        self.tracker = RegulatoryTracker()
        self.tracker.add_regulation(RegulationVersion(
            id="IEC_61508", title="Functional Safety", version="2.0",
            effective_date="2024-01-01",
            changes=[
                {"id": "C1", "description": "Change 1", "change_type": "amendment", "date": "2024-06-01"},
                {"id": "C2", "description": "Change 2", "change_type": "new_requirement", "date": "2024-12-01"},
            ],
        ))
        self.tracker.add_regulation(RegulationVersion(
            id="EU_AI_ACT", title="EU AI Act", version="1.0",
            effective_date="2025-01-01",
            changes=[
                {"id": "A1", "description": "Change A", "change_type": "amendment", "date": "2025-06-01"},
                {"id": "A2", "description": "Change B", "change_type": "clarification", "date": "2024-03-01"},
            ],
        ))

    def test_changes_since_date(self):
        changes = self.tracker.track_changes("2024-06-01")
        assert len(changes) >= 2  # C2, A1, and potentially C1

    def test_no_changes_recent(self):
        changes = self.tracker.track_changes("2099-01-01")
        assert len(changes) == 0

    def test_all_changes(self):
        changes = self.tracker.track_changes("2000-01-01")
        assert len(changes) == 4

    def test_sorted_descending(self):
        changes = self.tracker.track_changes("2000-01-01")
        for i in range(len(changes) - 1):
            assert changes[i]["change_date"] >= changes[i + 1]["change_date"]

    def test_change_fields(self):
        changes = self.tracker.track_changes("2000-01-01")
        c = changes[0]
        assert "regulation_id" in c
        assert "change_description" in c
        assert "change_type" in c


class TestRegulatoryTrackerScore:
    """Tests for RegulatoryTracker.compute_compliance_score."""

    def setup_method(self):
        self.tracker = RegulatoryTracker()
        self.tracker.add_regulation(RegulationVersion(
            id="IEC_61508", title="Safety", version="2.0",
            effective_date="2024-01-01",
            changes=[
                {"id": "R1", "description": "Req 1"},
                {"id": "R2", "description": "Req 2"},
                {"id": "R3", "description": "Req 3"},
            ],
        ))

    def test_full_score(self):
        system = {"name": "X", "current_compliance": {"R1": True, "R2": True, "R3": True}}
        score = self.tracker.compute_compliance_score(system, ["IEC_61508"])
        assert score == 100.0

    def test_partial_score(self):
        system = {"name": "X", "current_compliance": {"R1": True, "R2": True}}
        score = self.tracker.compute_compliance_score(system, ["IEC_61508"])
        assert abs(score - 200.0/3.0) < 0.1

    def test_zero_score(self):
        system = {"name": "X"}
        score = self.tracker.compute_compliance_score(system, ["IEC_61508"])
        assert score == 0.0

    def test_no_regulations(self):
        system = {"name": "X"}
        score = self.tracker.compute_compliance_score(system, [])
        assert score == 100.0  # No requirements = fully compliant


class TestRegulatoryTrackerHelpers:
    """Tests for RegulatoryTracker helper methods."""

    def setup_method(self):
        self.tracker = RegulatoryTracker()
        self.tracker.add_regulation(RegulationVersion(
            id="IEC_61508", title="Safety", version="2.0",
            effective_date="2024-01-01",
        ))

    def test_get_regulation(self):
        reg = self.tracker.get_regulation("IEC_61508")
        assert reg is not None
        assert reg.version == "2.0"

    def test_get_missing_regulation(self):
        reg = self.tracker.get_regulation("MISSING")
        assert reg is None

    def test_remove_regulation(self):
        assert self.tracker.remove_regulation("IEC_61508") is True
        assert self.tracker.get_regulation("IEC_61508") is None

    def test_remove_missing(self):
        assert self.tracker.remove_regulation("MISSING") is False

    def test_regulations_property(self):
        regs = self.tracker.regulations
        assert "IEC_61508" in regs


class TestComplianceGapReport:
    """Tests for ComplianceGapReport dataclass."""

    def test_create_report(self):
        report = ComplianceGapReport(
            system_name="Test",
            regulation="IEC_61508",
            overall_score=85.5,
            total_requirements=10,
            met_requirements=8,
            gaps_found=2,
        )
        assert report.system_name == "Test"
        assert report.overall_score == 85.5

    def test_with_gaps(self):
        gap = ComplianceGap(
            requirement="REQ-001",
            current_status="not_compliant",
            required_status="compliant",
        )
        report = ComplianceGapReport(
            system_name="Test",
            regulation="EU_AI_ACT",
            gaps=[gap],
            gaps_found=1,
        )
        assert len(report.gaps) == 1
