"""Evidence Collection Engine.

Provides structured evidence collection, validation, coverage analysis,
requirement-evidence mapping, and package export for regulatory compliance
of marine robotics systems.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
import hashlib


@dataclass
class EvidenceItem:
    """Single evidence item."""
    item_id: str
    category: str
    description: str
    artifact_path: str = ""
    collected_at: str = ""
    validity: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""


@dataclass
class EvidenceCollection:
    """Collection of evidence items."""
    items: List[EvidenceItem] = field(default_factory=list)
    target_regulation: str = ""
    collection_date: str = ""
    completeness: float = 0.0


@dataclass
class ValidationResult:
    """Result of validating an evidence item."""
    valid: bool
    issues: List[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class EvidenceMatrix:
    """Requirement-evidence mapping matrix."""
    requirement_id: str
    description: str
    evidence_items: List[str] = field(default_factory=list)
    coverage: float = 0.0
    gap: bool = False


class EvidenceCollector:
    """Evidence collection and management engine."""

    def __init__(self) -> None:
        self._collections: List[EvidenceCollection] = []
        self._item_counter = 0

    def _generate_id(self) -> str:
        """Generate a unique evidence item ID."""
        self._item_counter += 1
        return f"EVD-{self._item_counter:06d}"

    def _compute_checksum(self, content: str) -> str:
        """Compute SHA-256 checksum for evidence integrity."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def collect_test_evidence(
        self,
        test_results: Dict[str, Any],
        coverage_data: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceItem]:
        """Collect evidence from test execution.

        Args:
            test_results: Dict with keys: total, passed, failed, skipped, report_path
            coverage_data: Dict with keys: line_coverage, branch_coverage, module_coverage

        Returns:
            List of EvidenceItem objects
        """
        items = []
        coverage_data = coverage_data or {}

        # Test execution evidence
        total = test_results.get("total", 0)
        passed = test_results.get("passed", 0)
        failed = test_results.get("failed", 0)
        skipped = test_results.get("skipped", 0)
        pass_rate = passed / total if total > 0 else 0.0

        items.append(EvidenceItem(
            item_id=self._generate_id(),
            category="test_execution",
            description=(
                f"Test execution results: {passed}/{total} passed "
                f"({pass_rate:.1%}), {failed} failed, {skipped} skipped"
            ),
            artifact_path=test_results.get("report_path", ""),
            collected_at=datetime.utcnow().isoformat(),
            validity=failed == 0,
            metadata={
                "total": total, "passed": passed, "failed": failed,
                "skipped": skipped, "pass_rate": pass_rate,
            },
            checksum=self._compute_checksum(f"test:{total}:{passed}"),
        ))

        # Coverage evidence
        if coverage_data:
            line_cov = coverage_data.get("line_coverage", 0.0)
            branch_cov = coverage_data.get("branch_coverage", 0.0)
            module_cov = coverage_data.get("module_coverage", 0.0)

            items.append(EvidenceItem(
                item_id=self._generate_id(),
                category="test_coverage",
                description=(
                    f"Code coverage: lines {line_cov:.1%}, "
                    f"branches {branch_cov:.1%}, modules {module_cov:.1%}"
                ),
                artifact_path=coverage_data.get("report_path", ""),
                collected_at=datetime.utcnow().isoformat(),
                validity=line_cov >= 0.80,
                metadata={
                    "line_coverage": line_cov,
                    "branch_coverage": branch_cov,
                    "module_coverage": module_cov,
                },
                checksum=self._compute_checksum(f"cov:{line_cov}:{branch_cov}"),
            ))

        return items

    def collect_safety_evidence(
        self,
        safety_validations: Dict[str, Any],
    ) -> List[EvidenceItem]:
        """Collect evidence from safety validations.

        Args:
            safety_validations: Dict with keys: sil_verified, hazard_analysis,
                              fmea_completed, risk_assessment, validation_date

        Returns:
            List of EvidenceItem objects
        """
        items = []

        # SIL verification evidence
        sil_level = safety_validations.get("sil_verified", "N/A")
        items.append(EvidenceItem(
            item_id=self._generate_id(),
            category="safety_sil",
            description=f"SIL verification: achieved {sil_level}",
            artifact_path=safety_validations.get("sil_report_path", ""),
            collected_at=datetime.utcnow().isoformat(),
            validity=sil_level not in ("N/A", "NOT_ACHIEVED"),
            metadata={"sil_level": str(sil_level)},
            checksum=self._compute_checksum(f"sil:{sil_level}"),
        ))

        # Hazard analysis evidence
        hazard_count = safety_validations.get("hazard_analysis", {}).get("hazards_identified", 0)
        mitigated = safety_validations.get("hazard_analysis", {}).get("mitigated", 0)
        items.append(EvidenceItem(
            item_id=self._generate_id(),
            category="safety_hazard",
            description=(
                f"Hazard analysis: {hazard_count} hazards identified, "
                f"{mitigated} mitigated"
            ),
            artifact_path=safety_validations.get("hazard_report_path", ""),
            collected_at=datetime.utcnow().isoformat(),
            validity=mitigated >= hazard_count if hazard_count > 0 else True,
            metadata={
                "hazards_identified": hazard_count,
                "mitigated": mitigated,
            },
            checksum=self._compute_checksum(f"haz:{hazard_count}:{mitigated}"),
        ))

        # FMEA evidence
        fmea = safety_validations.get("fmea_completed", False)
        fmea_components = safety_validations.get("fmea_components", 0)
        items.append(EvidenceItem(
            item_id=self._generate_id(),
            category="safety_fmea",
            description=(
                f"FMEA: {'completed' if fmea else 'not completed'}, "
                f"{fmea_components} components analyzed"
            ),
            artifact_path=safety_validations.get("fmea_report_path", ""),
            collected_at=datetime.utcnow().isoformat(),
            validity=fmea,
            metadata={"completed": fmea, "components": fmea_components},
            checksum=self._compute_checksum(f"fmea:{fmea}:{fmea_components}"),
        ))

        # Risk assessment evidence
        risk_score = safety_validations.get("risk_assessment", {}).get("overall_score", 0.0)
        risk_acceptable = risk_score < 0.3
        items.append(EvidenceItem(
            item_id=self._generate_id(),
            category="safety_risk",
            description=(
                f"Risk assessment: overall score {risk_score:.2f}, "
                f"{'acceptable' if risk_acceptable else 'not acceptable'}"
            ),
            artifact_path=safety_validations.get("risk_report_path", ""),
            collected_at=datetime.utcnow().isoformat(),
            validity=risk_acceptable,
            metadata={"overall_score": risk_score, "acceptable": risk_acceptable},
            checksum=self._compute_checksum(f"risk:{risk_score}"),
        ))

        return items

    def collect_deployment_evidence(
        self,
        deployment_records: Dict[str, Any],
    ) -> List[EvidenceItem]:
        """Collect evidence from deployment activities.

        Args:
            deployment_records: Dict with keys: version, environment, deployed_by,
                              rollback_available, health_checks_passed

        Returns:
            List of EvidenceItem objects
        """
        items = []

        version = deployment_records.get("version", "unknown")
        environment = deployment_records.get("environment", "unknown")
        deployed_by = deployment_records.get("deployed_by", "unknown")

        # Deployment record
        items.append(EvidenceItem(
            item_id=self._generate_id(),
            category="deployment_record",
            description=(
                f"Deployment v{version} to {environment} by {deployed_by}"
            ),
            artifact_path=deployment_records.get("deployment_log", ""),
            collected_at=datetime.utcnow().isoformat(),
            validity=True,
            metadata={
                "version": version, "environment": environment,
                "deployed_by": deployed_by,
            },
            checksum=self._compute_checksum(f"deploy:{version}:{environment}"),
        ))

        # Health check evidence
        health_passed = deployment_records.get("health_checks_passed", False)
        checks = deployment_records.get("health_checks", {})
        total_checks = len(checks)
        passed_checks = sum(1 for v in checks.values() if v)

        items.append(EvidenceItem(
            item_id=self._generate_id(),
            category="deployment_health",
            description=(
                f"Health checks: {passed_checks}/{total_checks} passed, "
                f"overall {'PASS' if health_passed else 'FAIL'}"
            ),
            artifact_path=deployment_records.get("health_report", ""),
            collected_at=datetime.utcnow().isoformat(),
            validity=health_passed,
            metadata={
                "health_passed": health_passed,
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "checks": checks,
            },
            checksum=self._compute_checksum(f"health:{passed_checks}:{total_checks}"),
        ))

        # Rollback evidence
        rollback = deployment_records.get("rollback_available", False)
        items.append(EvidenceItem(
            item_id=self._generate_id(),
            category="deployment_rollback",
            description=f"Rollback capability: {'available' if rollback else 'NOT available'}",
            artifact_path="",
            collected_at=datetime.utcnow().isoformat(),
            validity=rollback,
            metadata={"rollback_available": rollback},
            checksum=self._compute_checksum(f"rollback:{rollback}"),
        ))

        return items

    def validate_evidence(
        self,
        item: EvidenceItem,
        criteria: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Validate an evidence item against criteria.

        Args:
            item: EvidenceItem to validate
            criteria: Dict with validation criteria (category, min_description_length, etc.)

        Returns:
            ValidationResult with validity assessment
        """
        criteria = criteria or {}
        issues = []
        score = 0.0
        total_checks = 5

        # Check basic validity
        if item.validity:
            score += 1
        else:
            issues.append("Evidence item marked as invalid")

        # Check description
        min_length = criteria.get("min_description_length", 10)
        if len(item.description) >= min_length:
            score += 1
        else:
            issues.append(f"Description too short (min {min_length} chars)")

        # Check timestamp
        if item.collected_at:
            score += 1
        else:
            issues.append("Missing collection timestamp")

        # Check category
        if item.category:
            score += 1
        else:
            issues.append("Missing category")

        # Check artifact path or checksum
        if item.artifact_path or item.checksum:
            score += 1
        else:
            issues.append("Missing artifact path or checksum")

        # Category-specific checks
        required_category = criteria.get("required_category")
        if required_category and item.category != required_category:
            issues.append(f"Wrong category: expected {required_category}, got {item.category}")

        valid = score >= 3 and len(issues) == 0
        return ValidationResult(
            valid=valid,
            issues=issues,
            score=score / total_checks,
        )

    def compute_collection_coverage(
        self,
        collection: EvidenceCollection,
        required_categories: List[str],
    ) -> float:
        """Compute evidence collection coverage.

        Args:
            collection: EvidenceCollection to evaluate
            required_categories: List of categories that must be present

        Returns:
            Coverage percentage (0.0-1.0)
        """
        if not required_categories:
            return 1.0

        found_categories = set()
        for item in collection.items:
            found_categories.add(item.category)

        required_set = set(required_categories)
        if not required_set:
            return 1.0

        covered = required_set & found_categories
        return len(covered) / len(required_set)

    def generate_evidence_matrix(
        self,
        collection: EvidenceCollection,
        requirements: List[Dict[str, Any]],
    ) -> List[EvidenceMatrix]:
        """Generate a requirement-evidence mapping matrix.

        Args:
            collection: EvidenceCollection with evidence items
            requirements: List of dicts with id, description, categories

        Returns:
            List of EvidenceMatrix entries
        """
        matrix = []

        for req in requirements:
            req_id = req.get("id", "")
            req_desc = req.get("description", "")
            req_categories = req.get("categories", [])
            req_evidence = []

            for item in collection.items:
                if item.category in req_categories:
                    req_evidence.append(item.item_id)

            coverage = 1.0 if req_evidence else 0.0
            gap = len(req_evidence) == 0

            matrix.append(EvidenceMatrix(
                requirement_id=req_id,
                description=req_desc,
                evidence_items=req_evidence,
                coverage=coverage,
                gap=gap,
            ))

        return matrix

    def export_evidence_package(
        self,
        collection: EvidenceCollection,
    ) -> Dict[str, Any]:
        """Export evidence collection as a structured package.

        Args:
            collection: EvidenceCollection to export

        Returns:
            Dict representation of the evidence package
        """
        items_data = []
        for item in collection.items:
            items_data.append({
                "item_id": item.item_id,
                "category": item.category,
                "description": item.description,
                "artifact_path": item.artifact_path,
                "collected_at": item.collected_at,
                "validity": item.validity,
                "metadata": item.metadata,
                "checksum": item.checksum,
            })

        valid_count = sum(1 for i in collection.items if i.validity)
        total_count = len(collection.items)

        return {
            "target_regulation": collection.target_regulation,
            "collection_date": collection.collection_date,
            "completeness": collection.completeness,
            "total_items": total_count,
            "valid_items": valid_count,
            "invalid_items": total_count - valid_count,
            "items": items_data,
        }

    def create_collection(
        self,
        target_regulation: str,
        items: Optional[List[EvidenceItem]] = None,
    ) -> EvidenceCollection:
        """Create a new evidence collection.

        Args:
            target_regulation: Regulation this collection targets
            items: Optional list of items to include

        Returns:
            EvidenceCollection
        """
        return EvidenceCollection(
            items=items or [],
            target_regulation=target_regulation,
            collection_date=datetime.utcnow().isoformat(),
        )
