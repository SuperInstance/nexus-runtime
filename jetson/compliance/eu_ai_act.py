"""EU AI Act Compliance Checking Engine.

Implements risk classification, requirement checking, transparency
verification, human oversight assessment, data governance evaluation,
and compliance report generation per the EU Artificial Intelligence Act.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime


class RiskCategory(Enum):
    """EU AI Act risk categories."""
    UNACCEPTABLE = "unacceptable"
    HIGH = "high"
    LIMITED = "limited"
    MINIMAL = "minimal"


# High-risk AI system categories per EU AI Act Annex III
HIGH_RISK_CATEGORIES = {
    "remote biometric identification",
    "critical infrastructure management",
    "education and vocational training",
    "employment and worker management",
    "essential private and public services",
    "law enforcement",
    "migration and border control",
    "justice and democratic processes",
    "marine navigation safety",
    "autonomous vehicle control",
}

# Prohibited / unacceptable AI practices
PROHIBITED_PRACTICES = {
    "social scoring by governments",
    "real-time remote biometric identification in public spaces (law enforcement exceptions apply)",
    "exploitation of vulnerabilities of specific groups",
    "manipulation of behaviour through subliminal techniques",
}

LIMITED_RISK_INDICATORS = {
    "chatbot",
    "recommendation system",
    "content filtering",
    "deepfake generation",
    "sentiment analysis",
}

MINIMAL_RISK_INDICATORS = {
    "spam filter",
    "search engine optimization",
    "inventory management",
    "internal data processing",
    "basic automation",
}

HIGH_RISK_INDICATORS = {
    "marine_autonomy",
    "collision_avoidance",
    "navigation_control",
    "safety_critical",
    "real_time_control",
    "biometric",
    "law_enforcement",
    "critical_infrastructure",
}


@dataclass
class AIRequirement:
    """A compliance requirement from the EU AI Act."""
    requirement_id: str
    description: str
    applicable_risks: List[RiskCategory]
    verification_method: str = "manual_review"


@dataclass
class RequirementResult:
    """Result of checking a single requirement."""
    requirement: AIRequirement
    status: str  # "pass", "fail", "not_applicable", "partial"
    findings: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)


@dataclass
class TransparencyResult:
    """Result of transparency checks."""
    compliant: bool
    findings: List[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class OversightResult:
    """Result of human oversight assessment."""
    compliant: bool
    findings: List[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class GovernanceResult:
    """Result of data governance assessment."""
    compliant: bool
    findings: List[str] = field(default_factory=list)
    score: float = 0.0
    data_quality_score: float = 0.0
    data_governance_score: float = 0.0


@dataclass
class ComplianceReport:
    """Full compliance report for a system."""
    system_name: str
    risk_category: RiskCategory
    requirements_status: List[RequirementResult]
    gaps: List[str] = field(default_factory=list)
    overall_compliant: bool = False
    compliance_score: float = 0.0
    generated_at: str = ""
    recommendations: List[str] = field(default_factory=list)


class EUAIActChecker:
    """EU AI Act compliance checking engine."""

    def __init__(self) -> None:
        self._requirements = self._build_requirements()

    def _build_requirements(self) -> List[AIRequirement]:
        """Build the full list of EU AI Act requirements."""
        high = [RiskCategory.HIGH]
        limited = [RiskCategory.LIMITED]
        all_risks = [RiskCategory.UNACCEPTABLE, RiskCategory.HIGH, RiskCategory.LIMITED, RiskCategory.MINIMAL]

        return [
            AIRequirement(
                requirement_id="REQ-001",
                description="Risk management system established and maintained",
                applicable_risks=high,
                verification_method="documentation_review",
            ),
            AIRequirement(
                requirement_id="REQ-002",
                description="Data governance and data quality management",
                applicable_risks=high,
                verification_method="data_audit",
            ),
            AIRequirement(
                requirement_id="REQ-003",
                description="Technical documentation maintained and kept up-to-date",
                applicable_risks=high,
                verification_method="documentation_review",
            ),
            AIRequirement(
                requirement_id="REQ-004",
                description="Record-keeping (logging) capabilities implemented",
                applicable_risks=high,
                verification_method="system_inspection",
            ),
            AIRequirement(
                requirement_id="REQ-005",
                description="Transparency and information to deployers provided",
                applicable_risks=high,
                verification_method="documentation_review",
            ),
            AIRequirement(
                requirement_id="REQ-006",
                description="Human oversight measures implemented",
                applicable_risks=high,
                verification_method="system_inspection",
            ),
            AIRequirement(
                requirement_id="REQ-007",
                description="Accuracy, robustness, and cybersecurity ensured",
                applicable_risks=high,
                verification_method="testing",
            ),
            AIRequirement(
                requirement_id="REQ-008",
                description="Conformity assessment before market placement",
                applicable_risks=high,
                verification_method="assessment_report",
            ),
            AIRequirement(
                requirement_id="REQ-009",
                description="Transparency obligations met for limited-risk AI",
                applicable_risks=limited,
                verification_method="documentation_review",
            ),
            AIRequirement(
                requirement_id="REQ-010",
                description="No prohibited AI practices employed",
                applicable_risks=all_risks,
                verification_method="system_inspection",
            ),
        ]

    def classify_risk(
        self,
        system_description: str,
        use_cases: Optional[List[str]] = None,
    ) -> RiskCategory:
        """Classify a system's risk category per EU AI Act.

        Args:
            system_description: Description of the AI system
            use_cases: List of intended use cases

        Returns:
            RiskCategory classification
        """
        desc_lower = system_description.lower()
        use_cases_lower = [uc.lower() for uc in (use_cases or [])]
        all_text = desc_lower + " " + " ".join(use_cases_lower)

        # Check for prohibited practices
        for practice in PROHIBITED_PRACTICES:
            if practice.lower() in all_text:
                return RiskCategory.UNACCEPTABLE

        # Check for high-risk indicators
        high_risk_score = 0
        for indicator in HIGH_RISK_CATEGORIES:
            if indicator in all_text:
                high_risk_score += 2
        for indicator in HIGH_RISK_INDICATORS:
            if indicator in all_text:
                high_risk_score += 1

        if high_risk_score >= 2:
            return RiskCategory.HIGH

        # Check for limited risk
        for indicator in LIMITED_RISK_INDICATORS:
            if indicator in all_text:
                return RiskCategory.LIMITED

        # Check for minimal risk
        for indicator in MINIMAL_RISK_INDICATORS:
            if indicator in all_text:
                return RiskCategory.MINIMAL

        return RiskCategory.MINIMAL

    def check_requirements(
        self,
        system: Dict[str, Any],
        risk_category: RiskCategory,
    ) -> List[RequirementResult]:
        """Check all applicable requirements for a system.

        Args:
            system: System description dict with capabilities
            risk_category: System's risk classification

        Returns:
            List of requirement check results
        """
        results = []

        for req in self._requirements:
            if risk_category in req.applicable_risks:
                status, findings = self._check_single_requirement(req, system, risk_category)
                results.append(RequirementResult(
                    requirement=req,
                    status=status,
                    findings=findings,
                ))
            else:
                results.append(RequirementResult(
                    requirement=req,
                    status="not_applicable",
                    findings=[f"Not applicable for {risk_category.value} risk"],
                ))

        return results

    def _check_single_requirement(
        self,
        req: AIRequirement,
        system: Dict[str, Any],
        risk_category: RiskCategory,
    ) -> Tuple[str, List[str]]:
        """Check a single requirement against the system."""
        findings = []

        capabilities = system.get("capabilities", {})
        docs = system.get("documentation", {})
        metadata = system.get("metadata", {})

        if req.requirement_id == "REQ-001":
            if capabilities.get("risk_management"):
                return "pass", ["Risk management system confirmed"]
            findings.append("No risk management system identified")
            return "fail", findings

        elif req.requirement_id == "REQ-002":
            if capabilities.get("data_governance"):
                return "pass", ["Data governance framework in place"]
            findings.append("Data governance measures not documented")
            return "fail", findings

        elif req.requirement_id == "REQ-003":
            if docs.get("technical_documentation"):
                return "pass", ["Technical documentation available"]
            findings.append("Technical documentation incomplete or missing")
            return "fail", findings

        elif req.requirement_id == "REQ-004":
            if capabilities.get("logging"):
                return "pass", ["Record-keeping capabilities confirmed"]
            findings.append("Logging/record-keeping not implemented")
            return "fail", findings

        elif req.requirement_id == "REQ-005":
            if capabilities.get("transparency"):
                return "pass", ["Transparency measures in place"]
            findings.append("Transparency measures insufficient")
            return "partial", findings

        elif req.requirement_id == "REQ-006":
            if capabilities.get("human_oversight"):
                return "pass", ["Human oversight measures implemented"]
            findings.append("Human oversight mechanism not identified")
            return "fail", findings

        elif req.requirement_id == "REQ-007":
            score = 0
            if capabilities.get("accuracy_testing"):
                score += 1
                findings.append("Accuracy testing conducted")
            else:
                findings.append("Accuracy testing not confirmed")
            if capabilities.get("robustness_testing"):
                score += 1
                findings.append("Robustness testing conducted")
            else:
                findings.append("Robustness testing not confirmed")
            if capabilities.get("cybersecurity"):
                score += 1
                findings.append("Cybersecurity measures in place")
            else:
                findings.append("Cybersecurity measures not confirmed")

            if score >= 2:
                return "pass", findings
            elif score >= 1:
                return "partial", findings
            return "fail", findings

        elif req.requirement_id == "REQ-008":
            if metadata.get("conformity_assessment"):
                return "pass", ["Conformity assessment completed"]
            findings.append("Conformity assessment not conducted")
            return "fail", findings

        elif req.requirement_id == "REQ-009":
            if capabilities.get("transparency"):
                return "pass", ["Transparency obligations met"]
            findings.append("Transparency disclosure missing")
            return "fail", findings

        elif req.requirement_id == "REQ-010":
            prohibited = system.get("prohibited_practices", [])
            if not prohibited:
                return "pass", ["No prohibited practices detected"]
            findings.append(f"Prohibited practices found: {', '.join(prohibited)}")
            return "fail", findings

        return "pass", ["Requirement satisfied by default"]

    def check_transparency(
        self,
        system: Dict[str, Any],
    ) -> TransparencyResult:
        """Check transparency obligations.

        Args:
            system: System description dict

        Returns:
            TransparencyResult with findings
        """
        findings = []
        score = 0.0
        total_checks = 5

        capabilities = system.get("capabilities", {})
        docs = system.get("documentation", {})
        metadata = system.get("metadata", {})

        if capabilities.get("transparency"):
            score += 1
            findings.append("Transparency mechanism implemented")
        else:
            findings.append("No transparency mechanism found")

        if metadata.get("user_disclosure"):
            score += 1
            findings.append("User disclosure provided")
        else:
            findings.append("User disclosure not provided")

        if docs.get("ai_interaction_notice"):
            score += 1
            findings.append("AI interaction notice included")
        else:
            findings.append("AI interaction notice missing")

        if metadata.get("data_usage_transparency"):
            score += 1
            findings.append("Data usage transparency documented")
        else:
            findings.append("Data usage transparency not documented")

        if docs.get("model_card"):
            score += 1
            findings.append("Model card available")
        else:
            findings.append("Model card not available")

        compliant = score >= 3
        return TransparencyResult(
            compliant=compliant,
            findings=findings,
            score=score / total_checks,
        )

    def check_human_oversight(
        self,
        system: Dict[str, Any],
    ) -> OversightResult:
        """Check human oversight measures.

        Args:
            system: System description dict

        Returns:
            OversightResult with findings
        """
        findings = []
        score = 0.0
        total_checks = 5

        capabilities = system.get("capabilities", {})
        docs = system.get("documentation", {})

        if capabilities.get("human_oversight"):
            score += 1
            findings.append("Human oversight mechanism in place")
        else:
            findings.append("Human oversight mechanism not found")

        if capabilities.get("override_capability"):
            score += 1
            findings.append("Human override capability exists")
        else:
            findings.append("No human override capability detected")

        if capabilities.get("stop_function"):
            score += 1
            findings.append("Emergency stop function available")
        else:
            findings.append("Emergency stop function not detected")

        if docs.get("oversight_procedures"):
            score += 1
            findings.append("Oversight procedures documented")
        else:
            findings.append("Oversight procedures not documented")

        if capabilities.get("anomaly_alerting"):
            score += 1
            findings.append("Anomaly alerting to human operators implemented")
        else:
            findings.append("No anomaly alerting mechanism detected")

        compliant = score >= 3
        return OversightResult(
            compliant=compliant,
            findings=findings,
            score=score / total_checks,
        )

    def assess_data_governance(
        self,
        system: Dict[str, Any],
    ) -> GovernanceResult:
        """Assess data governance practices.

        Args:
            system: System description dict

        Returns:
            GovernanceResult with findings and scores
        """
        findings = []
        quality_checks = 5
        quality_score = 0.0
        governance_checks = 5
        governance_score = 0.0

        data_info = system.get("data", {})
        capabilities = system.get("capabilities", {})
        docs = system.get("documentation", {})

        # Data quality checks
        if data_info.get("validation"):
            quality_score += 1
            findings.append("Data validation implemented")
        else:
            findings.append("No data validation found")

        if data_info.get("bias_testing"):
            quality_score += 1
            findings.append("Bias testing conducted")
        else:
            findings.append("Bias testing not conducted")

        if data_info.get("quality_metrics"):
            quality_score += 1
            findings.append("Data quality metrics tracked")
        else:
            findings.append("Data quality metrics not tracked")

        if data_info.get("preprocessing_documented"):
            quality_score += 1
            findings.append("Preprocessing pipeline documented")
        else:
            findings.append("Preprocessing pipeline not documented")

        if data_info.get("representativeness_assessed"):
            quality_score += 1
            findings.append("Data representativeness assessed")
        else:
            findings.append("Data representativeness not assessed")

        # Data governance checks
        if capabilities.get("data_governance"):
            governance_score += 1
            findings.append("Data governance framework established")
        else:
            findings.append("No data governance framework found")

        if data_info.get("privacy_measures"):
            governance_score += 1
            findings.append("Privacy protection measures in place")
        else:
            findings.append("Privacy protection measures missing")

        if data_info.get("consent_management"):
            governance_score += 1
            findings.append("Consent management implemented")
        else:
            findings.append("Consent management not implemented")

        if data_info.get("access_controls"):
            governance_score += 1
            findings.append("Data access controls defined")
        else:
            findings.append("Data access controls not defined")

        if docs.get("data_protection_impact"):
            governance_score += 1
            findings.append("Data protection impact assessment completed")
        else:
            findings.append("Data protection impact assessment missing")

        data_quality_score = quality_score / quality_checks
        data_governance_score = governance_score / governance_checks
        overall_score = (data_quality_score + data_governance_score) / 2.0
        compliant = overall_score >= 0.5

        return GovernanceResult(
            compliant=compliant,
            findings=findings,
            score=overall_score,
            data_quality_score=data_quality_score,
            data_governance_score=data_governance_score,
        )

    def generate_compliance_report(
        self,
        system: Dict[str, Any],
        risk_category: RiskCategory,
    ) -> ComplianceReport:
        """Generate a full compliance report.

        Args:
            system: System description dict
            risk_category: System's risk classification

        Returns:
            Full ComplianceReport
        """
        req_results = self.check_requirements(system, risk_category)
        transparency = self.check_transparency(system)
        oversight = self.check_human_oversight(system)
        governance = self.assess_data_governance(system)

        applicable_results = [r for r in req_results if r.status != "not_applicable"]
        passing = sum(1 for r in applicable_results if r.status == "pass")
        total = len(applicable_results)

        req_score = passing / total if total > 0 else 1.0
        overall_score = (
            req_score * 0.4
            + transparency.score * 0.2
            + oversight.score * 0.2
            + governance.score * 0.2
        )

        gaps = []
        recommendations = []

        for result in req_results:
            if result.status == "fail":
                gaps.append(
                    f"[{result.requirement.requirement_id}] {result.requirement.description}: "
                    + "; ".join(result.findings)
                )
                recommendations.append(
                    f"Address {result.requirement.requirement_id}: {result.requirement.description}"
                )
            elif result.status == "partial":
                gaps.append(
                    f"[PARTIAL] {result.requirement.requirement_id}] {result.requirement.description}: "
                    + "; ".join(result.findings)
                )

        if not transparency.compliant:
            gaps.append("Transparency requirements not fully met")
            recommendations.append("Implement transparency measures including user disclosure")
        if not oversight.compliant:
            gaps.append("Human oversight requirements not fully met")
            recommendations.append("Implement human override and anomaly alerting mechanisms")
        if not governance.compliant:
            gaps.append("Data governance requirements not fully met")
            recommendations.append("Establish data governance framework with quality metrics")

        overall_compliant = (
            overall_score >= 0.7
            and len([r for r in req_results if r.status == "fail"]) == 0
            and risk_category != RiskCategory.UNACCEPTABLE
        )

        return ComplianceReport(
            system_name=system.get("name", "Unknown System"),
            risk_category=risk_category,
            requirements_status=req_results,
            gaps=gaps,
            overall_compliant=overall_compliant,
            compliance_score=round(overall_score * 100, 1),
            generated_at=datetime.utcnow().isoformat(),
            recommendations=recommendations,
        )
