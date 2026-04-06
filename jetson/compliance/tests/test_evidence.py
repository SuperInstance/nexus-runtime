"""Tests for Evidence Collection Engine."""

import pytest

from jetson.compliance.evidence import (
    EvidenceItem,
    EvidenceCollection,
    EvidenceCollector,
    ValidationResult,
    EvidenceMatrix,
)


class TestEvidenceItem:
    """Tests for EvidenceItem dataclass."""

    def test_create_item(self):
        item = EvidenceItem(
            item_id="EVD-001",
            category="test_execution",
            description="Test results collected",
        )
        assert item.item_id == "EVD-001"
        assert item.category == "test_execution"
        assert item.validity is True
        assert item.artifact_path == ""

    def test_with_artifact(self):
        item = EvidenceItem(
            item_id="EVD-002",
            category="safety",
            description="Safety report",
            artifact_path="/reports/safety.pdf",
            checksum="abc123",
        )
        assert item.artifact_path == "/reports/safety.pdf"
        assert item.checksum == "abc123"

    def test_invalid_item(self):
        item = EvidenceItem(
            item_id="EVD-003",
            category="test",
            description="Failed test",
            validity=False,
        )
        assert item.validity is False


class TestEvidenceCollection:
    """Tests for EvidenceCollection dataclass."""

    def test_create_collection(self):
        coll = EvidenceCollection(
            items=[],
            target_regulation="IEC_61508",
        )
        assert coll.target_regulation == "IEC_61508"
        assert coll.completeness == 0.0

    def test_with_items(self):
        item = EvidenceItem(item_id="EVD-001", category="test", description="Test")
        coll = EvidenceCollection(items=[item], target_regulation="EU_AI_ACT")
        assert len(coll.items) == 1


class TestEvidenceCollectorTestEvidence:
    """Tests for EvidenceCollector.collect_test_evidence."""

    def setup_method(self):
        self.collector = EvidenceCollector()

    def test_basic_test_results(self):
        results = self.collector.collect_test_evidence({
            "total": 100, "passed": 100, "failed": 0, "skipped": 0,
        })
        assert len(results) == 1
        assert results[0].category == "test_execution"
        assert results[0].validity is True

    def test_failed_tests(self):
        results = self.collector.collect_test_evidence({
            "total": 100, "passed": 95, "failed": 5, "skipped": 0,
        })
        assert results[0].validity is False
        assert "95" in results[0].description

    def test_with_coverage(self):
        results = self.collector.collect_test_evidence(
            {"total": 50, "passed": 50, "failed": 0, "skipped": 0},
            {"line_coverage": 0.85, "branch_coverage": 0.70, "module_coverage": 0.90},
        )
        assert len(results) == 2
        coverage_item = results[1]
        assert coverage_item.category == "test_coverage"
        assert coverage_item.metadata["line_coverage"] == 0.85

    def test_low_coverage_invalid(self):
        results = self.collector.collect_test_evidence(
            {"total": 50, "passed": 50, "failed": 0, "skipped": 0},
            {"line_coverage": 0.60},
        )
        coverage_item = results[1]
        assert coverage_item.validity is False

    def test_high_coverage_valid(self):
        results = self.collector.collect_test_evidence(
            {"total": 50, "passed": 50, "failed": 0, "skipped": 0},
            {"line_coverage": 0.90},
        )
        coverage_item = results[1]
        assert coverage_item.validity is True

    def test_unique_ids(self):
        r1 = self.collector.collect_test_evidence({"total": 10, "passed": 10, "failed": 0, "skipped": 0})
        r2 = self.collector.collect_test_evidence({"total": 20, "passed": 20, "failed": 0, "skipped": 0})
        assert r1[0].item_id != r2[0].item_id


class TestEvidenceCollectorSafetyEvidence:
    """Tests for EvidenceCollector.collect_safety_evidence."""

    def setup_method(self):
        self.collector = EvidenceCollector()

    def test_full_safety(self):
        results = self.collector.collect_safety_evidence({
            "sil_verified": "SIL_2",
            "hazard_analysis": {"hazards_identified": 5, "mitigated": 5},
            "fmea_completed": True,
            "fmea_components": 20,
            "risk_assessment": {"overall_score": 0.1},
        })
        assert len(results) == 4
        for r in results:
            assert r.validity is True

    def test_no_sil(self):
        results = self.collector.collect_safety_evidence({
            "sil_verified": "N/A",
            "hazard_analysis": {"hazards_identified": 3, "mitigated": 2},
            "fmea_completed": False,
            "fmea_components": 0,
            "risk_assessment": {"overall_score": 0.8},
        })
        sil_item = results[0]
        assert sil_item.validity is False

    def test_fmea_not_completed(self):
        results = self.collector.collect_safety_evidence({
            "fmea_completed": False,
            "fmea_components": 0,
        })
        fmea_item = results[2]
        assert fmea_item.validity is False

    def test_high_risk_unacceptable(self):
        results = self.collector.collect_safety_evidence({
            "risk_assessment": {"overall_score": 0.9},
        })
        risk_item = results[3]
        assert risk_item.validity is False

    def test_sil_item_category(self):
        results = self.collector.collect_safety_evidence({"sil_verified": "SIL_1"})
        assert results[0].category == "safety_sil"

    def test_hazard_item_has_counts(self):
        results = self.collector.collect_safety_evidence({
            "hazard_analysis": {"hazards_identified": 10, "mitigated": 7},
        })
        hazard_item = results[1]
        assert hazard_item.metadata["hazards_identified"] == 10
        assert hazard_item.validity is False  # 7 < 10


class TestEvidenceCollectorDeploymentEvidence:
    """Tests for EvidenceCollector.collect_deployment_evidence."""

    def setup_method(self):
        self.collector = EvidenceCollector()

    def test_successful_deployment(self):
        results = self.collector.collect_deployment_evidence({
            "version": "2.1.0",
            "environment": "production",
            "deployed_by": "admin",
            "health_checks_passed": True,
            "health_checks": {"nav": True, "ctrl": True, "comm": True},
            "rollback_available": True,
        })
        assert len(results) == 3
        for r in results:
            assert r.validity is True

    def test_failed_health_checks(self):
        results = self.collector.collect_deployment_evidence({
            "version": "2.0.0",
            "environment": "staging",
            "deployed_by": "ci",
            "health_checks_passed": False,
            "health_checks": {"nav": True, "ctrl": False},
            "rollback_available": False,
        })
        health_item = results[1]
        assert health_item.validity is False

    def test_no_rollback(self):
        results = self.collector.collect_deployment_evidence({
            "rollback_available": False,
        })
        rollback_item = results[2]
        assert rollback_item.validity is False

    def test_deployment_record_category(self):
        results = self.collector.collect_deployment_evidence({
            "version": "1.0",
        })
        assert results[0].category == "deployment_record"


class TestEvidenceCollectorValidate:
    """Tests for EvidenceCollector.validate_evidence."""

    def setup_method(self):
        self.collector = EvidenceCollector()

    def test_valid_item(self):
        item = EvidenceItem(
            item_id="EVD-001",
            category="test",
            description="A comprehensive test evidence description",
            collected_at="2025-01-01T00:00:00",
            checksum="abc",
            validity=True,
        )
        result = self.collector.validate_evidence(item)
        assert result.valid is True

    def test_invalid_short_description(self):
        item = EvidenceItem(
            item_id="EVD-001",
            category="test",
            description="Hi",
            collected_at="2025-01-01T00:00:00",
        )
        result = self.collector.validate_evidence(item)
        assert result.valid is False

    def test_no_timestamp(self):
        item = EvidenceItem(
            item_id="EVD-001",
            category="test",
            description="A comprehensive test evidence description",
        )
        result = self.collector.validate_evidence(item)
        assert result.valid is False

    def test_custom_criteria(self):
        item = EvidenceItem(
            item_id="EVD-001",
            category="test",
            description="Good description here",
            collected_at="2025-01-01",
            checksum="abc",
        )
        result = self.collector.validate_evidence(
            item, {"required_category": "safety"}
        )
        assert result.valid is False

    def test_score_range(self):
        item = EvidenceItem(
            item_id="EVD-001",
            category="test",
            description="Short",
            collected_at="2025-01-01",
        )
        result = self.collector.validate_evidence(item)
        assert 0.0 <= result.score <= 1.0


class TestEvidenceCollectorCoverage:
    """Tests for EvidenceCollector.compute_collection_coverage."""

    def setup_method(self):
        self.collector = EvidenceCollector()

    def test_full_coverage(self):
        coll = EvidenceCollection(items=[
            EvidenceItem(item_id="EVD-001", category="test", description="T"),
            EvidenceItem(item_id="EVD-002", category="safety", description="S"),
        ])
        coverage = self.collector.compute_collection_coverage(
            coll, ["test", "safety"]
        )
        assert coverage == 1.0

    def test_partial_coverage(self):
        coll = EvidenceCollection(items=[
            EvidenceItem(item_id="EVD-001", category="test", description="T"),
        ])
        coverage = self.collector.compute_collection_coverage(
            coll, ["test", "safety", "deployment"]
        )
        assert coverage == pytest.approx(1.0 / 3.0)

    def test_no_coverage(self):
        coll = EvidenceCollection(items=[
            EvidenceItem(item_id="EVD-001", category="other", description="O"),
        ])
        coverage = self.collector.compute_collection_coverage(
            coll, ["test", "safety"]
        )
        assert coverage == 0.0

    def test_empty_required(self):
        coll = EvidenceCollection(items=[])
        coverage = self.collector.compute_collection_coverage(coll, [])
        assert coverage == 1.0


class TestEvidenceCollectorMatrix:
    """Tests for EvidenceCollector.generate_evidence_matrix."""

    def setup_method(self):
        self.collector = EvidenceCollector()

    def test_basic_matrix(self):
        coll = EvidenceCollection(items=[
            EvidenceItem(item_id="EVD-001", category="test", description="T"),
            EvidenceItem(item_id="EVD-002", category="safety", description="S"),
        ])
        requirements = [
            {"id": "REQ-1", "description": "Tests", "categories": ["test"]},
            {"id": "REQ-2", "description": "Safety", "categories": ["safety"]},
        ]
        matrix = self.collector.generate_evidence_matrix(coll, requirements)
        assert len(matrix) == 2
        assert matrix[0].coverage == 1.0
        assert matrix[1].coverage == 1.0
        assert not matrix[0].gap

    def test_gap_detected(self):
        coll = EvidenceCollection(items=[
            EvidenceItem(item_id="EVD-001", category="test", description="T"),
        ])
        requirements = [
            {"id": "REQ-1", "description": "Tests", "categories": ["test"]},
            {"id": "REQ-2", "description": "Docs", "categories": ["documentation"]},
        ]
        matrix = self.collector.generate_evidence_matrix(coll, requirements)
        assert matrix[1].gap is True
        assert matrix[1].coverage == 0.0


class TestEvidenceCollectorExport:
    """Tests for EvidenceCollector.export_evidence_package."""

    def setup_method(self):
        self.collector = EvidenceCollector()

    def test_basic_export(self):
        coll = EvidenceCollection(
            items=[
                EvidenceItem(item_id="EVD-001", category="test", description="T", validity=True),
                EvidenceItem(item_id="EVD-002", category="test", description="F", validity=False),
            ],
            target_regulation="IEC_61508",
        )
        pkg = self.collector.export_evidence_package(coll)
        assert pkg["target_regulation"] == "IEC_61508"
        assert pkg["total_items"] == 2
        assert pkg["valid_items"] == 1
        assert pkg["invalid_items"] == 1
        assert len(pkg["items"]) == 2

    def test_empty_export(self):
        coll = EvidenceCollection(items=[], target_regulation="EU_AI_ACT")
        pkg = self.collector.export_evidence_package(coll)
        assert pkg["total_items"] == 0


class TestEvidenceCollectorCreateCollection:
    """Tests for EvidenceCollector.create_collection."""

    def setup_method(self):
        self.collector = EvidenceCollector()

    def test_create_empty(self):
        coll = self.collector.create_collection("IEC_61508")
        assert coll.target_regulation == "IEC_61508"
        assert coll.items == []

    def test_create_with_items(self):
        items = [EvidenceItem(item_id="EVD-001", category="test", description="T")]
        coll = self.collector.create_collection("EU_AI_ACT", items)
        assert len(coll.items) == 1
        assert coll.collection_date != ""
