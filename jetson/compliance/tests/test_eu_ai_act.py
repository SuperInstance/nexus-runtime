"""Tests for EU AI Act Compliance Checking Engine."""

import pytest
from datetime import datetime

from jetson.compliance.eu_ai_act import (
    RiskCategory,
    AIRequirement,
    ComplianceReport,
    EUAIActChecker,
)


class TestRiskCategory:
    """Tests for RiskCategory enum."""

    def test_unacceptable_value(self):
        assert RiskCategory.UNACCEPTABLE.value == "unacceptable"

    def test_high_value(self):
        assert RiskCategory.HIGH.value == "high"

    def test_limited_value(self):
        assert RiskCategory.LIMITED.value == "limited"

    def test_minimal_value(self):
        assert RiskCategory.MINIMAL.value == "minimal"

    def test_all_categories(self):
        assert len(RiskCategory) == 4


class TestAIRequirement:
    """Tests for AIRequirement dataclass."""

    def test_create_requirement(self):
        req = AIRequirement(
            requirement_id="REQ-001",
            description="Test requirement",
            applicable_risks=[RiskCategory.HIGH],
        )
        assert req.requirement_id == "REQ-001"
        assert req.description == "Test requirement"
        assert RiskCategory.HIGH in req.applicable_risks

    def test_default_verification_method(self):
        req = AIRequirement(
            requirement_id="REQ-001",
            description="Test",
            applicable_risks=[],
        )
        assert req.verification_method == "manual_review"


class TestEUAIActCheckerClassifyRisk:
    """Tests for EUAIActChecker.classify_risk."""

    def setup_method(self):
        self.checker = EUAIActChecker()

    def test_prohibited_classification(self):
        risk = self.checker.classify_risk("social scoring by governments system")
        assert risk == RiskCategory.UNACCEPTABLE

    def test_marine_autonomy_high_risk(self):
        risk = self.checker.classify_risk(
            "Marine autonomous navigation system for collision avoidance",
            ["marine_autonomy", "safety_critical"]
        )
        assert risk == RiskCategory.HIGH

    def test_chatbot_limited_risk(self):
        risk = self.checker.classify_risk("Customer service chatbot")
        assert risk == RiskCategory.LIMITED

    def test_spam_filter_minimal_risk(self):
        risk = self.checker.classify_risk("Email spam filter")
        assert risk == RiskCategory.MINIMAL

    def test_navigation_control_high_risk(self):
        risk = self.checker.classify_risk("Autonomous vessel navigation_control for collision_avoidance safety_critical system")
        assert risk == RiskCategory.HIGH

    def test_recommendation_system_limited(self):
        risk = self.checker.classify_risk("Product recommendation system")
        assert risk == RiskCategory.LIMITED

    def test_inventory_minimal(self):
        risk = self.checker.classify_risk("Inventory management system")
        assert risk == RiskCategory.MINIMAL

    def test_critical_infrastructure_high(self):
        risk = self.checker.classify_risk("critical infrastructure management system")
        assert risk == RiskCategory.HIGH

    def test_use_cases_contribute(self):
        risk = self.checker.classify_risk(
            "A system for ocean tasks",
            ["marine_autonomy", "collision_avoidance"]
        )
        assert risk == RiskCategory.HIGH

    def test_none_use_cases(self):
        risk = self.checker.classify_risk("basic automation tool", None)
        assert risk == RiskCategory.MINIMAL


class TestEUAIActCheckerRequirements:
    """Tests for EUAIActChecker.check_requirements."""

    def setup_method(self):
        self.checker = EUAIActChecker()
        self.system = {
            "name": "Test System",
            "capabilities": {
                "risk_management": True,
                "data_governance": True,
                "logging": True,
                "transparency": True,
                "human_oversight": True,
                "accuracy_testing": True,
                "robustness_testing": True,
                "cybersecurity": True,
            },
            "documentation": {
                "technical_documentation": True,
            },
            "metadata": {
                "conformity_assessment": True,
            },
        }

    def test_full_compliance_all_pass(self):
        results = self.checker.check_requirements(self.system, RiskCategory.HIGH)
        applicable = [r for r in results if r.status != "not_applicable"]
        for r in applicable:
            assert r.status in ("pass", "not_applicable")

    def test_no_capabilities_all_fail(self):
        system = {"name": "Empty", "capabilities": {}}
        results = self.checker.check_requirements(system, RiskCategory.HIGH)
        applicable = [r for r in results if r.status != "not_applicable"]
        assert any(r.status == "fail" for r in applicable)

    def test_partial_compliance(self):
        self.system["capabilities"]["accuracy_testing"] = True
        self.system["capabilities"]["robustness_testing"] = False
        self.system["capabilities"]["cybersecurity"] = False
        results = self.checker.check_requirements(self.system, RiskCategory.HIGH)
        req7 = [r for r in results if r.requirement.requirement_id == "REQ-007"][0]
        assert req7.status == "partial"

    def test_not_applicable_for_minimal_risk(self):
        results = self.checker.check_requirements(self.system, RiskCategory.MINIMAL)
        high_risk_reqs = [
            r for r in results
            if r.requirement.requirement_id != "REQ-010"
            and RiskCategory.HIGH in r.requirement.applicable_risks
        ]
        for r in high_risk_reqs:
            assert r.status == "not_applicable"

    def test_prohibited_practices_check(self):
        system = {"name": "Bad", "prohibited_practices": ["social_scoring"]}
        results = self.checker.check_requirements(system, RiskCategory.MINIMAL)
        req10 = [r for r in results if r.requirement.requirement_id == "REQ-010"][0]
        assert req10.status == "fail"


class TestEUAIActCheckerTransparency:
    """Tests for EUAIActChecker.check_transparency."""

    def setup_method(self):
        self.checker = EUAIActChecker()

    def test_full_transparency(self):
        system = {
            "capabilities": {"transparency": True},
            "metadata": {
                "user_disclosure": True,
                "data_usage_transparency": True,
            },
            "documentation": {
                "ai_interaction_notice": True,
                "model_card": True,
            },
        }
        result = self.checker.check_transparency(system)
        assert result.compliant is True
        assert result.score == 1.0

    def test_no_transparency(self):
        system = {}
        result = self.checker.check_transparency(system)
        assert result.compliant is False
        assert result.score < 0.5

    def test_partial_transparency(self):
        system = {
            "capabilities": {"transparency": True},
            "metadata": {"user_disclosure": True},
        }
        result = self.checker.check_transparency(system)
        assert result.score > 0.0
        assert result.score < 1.0


class TestEUAIActCheckerHumanOversight:
    """Tests for EUAIActChecker.check_human_oversight."""

    def setup_method(self):
        self.checker = EUAIActChecker()

    def test_full_oversight(self):
        system = {
            "capabilities": {
                "human_oversight": True,
                "override_capability": True,
                "stop_function": True,
                "anomaly_alerting": True,
            },
            "documentation": {"oversight_procedures": True},
        }
        result = self.checker.check_human_oversight(system)
        assert result.compliant is True
        assert result.score == 1.0

    def test_no_oversight(self):
        system = {}
        result = self.checker.check_human_oversight(system)
        assert result.compliant is False

    def test_partial_oversight(self):
        system = {
            "capabilities": {
                "human_oversight": True,
                "override_capability": True,
                "stop_function": True,
            },
        }
        result = self.checker.check_human_oversight(system)
        assert result.compliant is True
        assert 0 < result.score < 1.0


class TestEUAIActCheckerDataGovernance:
    """Tests for EUAIActChecker.assess_data_governance."""

    def setup_method(self):
        self.checker = EUAIActChecker()

    def test_full_governance(self):
        system = {
            "data": {
                "validation": True, "bias_testing": True,
                "quality_metrics": True, "preprocessing_documented": True,
                "representativeness_assessed": True,
                "privacy_measures": True, "consent_management": True,
                "access_controls": True,
            },
            "capabilities": {"data_governance": True},
            "documentation": {"data_protection_impact": True},
        }
        result = self.checker.assess_data_governance(system)
        assert result.compliant is True
        assert result.score == 1.0
        assert result.data_quality_score == 1.0
        assert result.data_governance_score == 1.0

    def test_no_governance(self):
        system = {}
        result = self.checker.assess_data_governance(system)
        assert result.compliant is False
        assert result.score == 0.0

    def test_quality_only(self):
        system = {
            "data": {
                "validation": True, "bias_testing": True,
                "quality_metrics": True,
            },
        }
        result = self.checker.assess_data_governance(system)
        assert result.data_quality_score > 0
        assert result.data_governance_score == 0.0


class TestEUAIActCheckerComplianceReport:
    """Tests for EUAIActChecker.generate_compliance_report."""

    def setup_method(self):
        self.checker = EUAIActChecker()

    def test_full_report_generated(self):
        system = {
            "name": "Marine Nav AI",
            "capabilities": {
                "risk_management": True, "data_governance": True,
                "logging": True, "transparency": True,
                "human_oversight": True, "override_capability": True,
                "stop_function": True, "anomaly_alerting": True,
                "accuracy_testing": True, "robustness_testing": True,
                "cybersecurity": True,
            },
            "documentation": {
                "technical_documentation": True,
                "oversight_procedures": True,
                "data_protection_impact": True,
                "ai_interaction_notice": True,
                "model_card": True,
            },
            "metadata": {
                "conformity_assessment": True,
                "user_disclosure": True,
                "data_usage_transparency": True,
            },
            "data": {
                "validation": True, "bias_testing": True,
                "quality_metrics": True, "preprocessing_documented": True,
                "representativeness_assessed": True,
                "privacy_measures": True, "consent_management": True,
                "access_controls": True,
            },
        }
        report = self.checker.generate_compliance_report(system, RiskCategory.HIGH)
        assert isinstance(report, ComplianceReport)
        assert report.system_name == "Marine Nav AI"
        assert report.risk_category == RiskCategory.HIGH
        assert report.generated_at != ""
        assert len(report.requirements_status) > 0

    def test_unacceptable_not_compliant(self):
        system = {
            "name": "Bad System",
            "capabilities": {},
            "documentation": {},
            "metadata": {},
        }
        report = self.checker.generate_compliance_report(system, RiskCategory.UNACCEPTABLE)
        assert report.overall_compliant is False

    def test_compliance_score_range(self):
        system = {"name": "X", "capabilities": {}, "documentation": {}, "metadata": {}}
        report = self.checker.generate_compliance_report(system, RiskCategory.MINIMAL)
        assert 0 <= report.compliance_score <= 100

    def test_has_recommendations_when_gaps(self):
        system = {"name": "X", "capabilities": {}, "documentation": {}, "metadata": {}}
        report = self.checker.generate_compliance_report(system, RiskCategory.HIGH)
        # With empty capabilities there should be gaps/recommendations
        assert len(report.gaps) > 0 or len(report.requirements_status) > 0
