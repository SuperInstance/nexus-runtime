"""Regulatory Change Tracking Engine.

Provides regulation version management, impact assessment, compliance gap
analysis, change tracking, and compliance scoring for marine robotics systems.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, date


@dataclass
class RegulationVersion:
    """Version of a regulation."""
    id: str
    title: str
    version: str
    effective_date: str
    changes: List[Dict[str, str]] = field(default_factory=list)
    status: str = "active"


@dataclass
class ChangeImpact:
    """Impact assessment of a regulatory change."""
    regulation: str
    change: str
    affected_components: List[str] = field(default_factory=list)
    severity: str = "low"  # "low", "medium", "high", "critical"
    action_required: str = ""


@dataclass
class ComplianceGap:
    """A single compliance gap."""
    requirement: str
    current_status: str
    required_status: str
    severity: str = "medium"
    remediation: str = ""


@dataclass
class ComplianceGapReport:
    """Full compliance gap analysis report."""
    system_name: str
    regulation: str
    gaps: List[ComplianceGap] = field(default_factory=list)
    overall_score: float = 0.0
    total_requirements: int = 0
    met_requirements: int = 0
    gaps_found: int = 0
    generated_at: str = ""


class RegulatoryTracker:
    """Regulatory change tracking and compliance management engine."""

    def __init__(self) -> None:
        self._regulations: Dict[str, RegulationVersion] = {}

    @property
    def regulations(self) -> Dict[str, RegulationVersion]:
        return dict(self._regulations)

    def add_regulation(self, regulation: RegulationVersion) -> None:
        """Add or update a regulation in the tracker.

        Args:
            regulation: RegulationVersion to add
        """
        self._regulations[regulation.id] = regulation

    def check_version(
        self,
        regulation_id: str,
        current_version: str,
    ) -> Tuple[bool, List[Dict[str, str]]]:
        """Check if a regulation is up to date.

        Args:
            regulation_id: ID of the regulation to check
            current_version: Current installed/compliant version

        Returns:
            (is_up_to_date, list_of_new_changes)
        """
        regulation = self._regulations.get(regulation_id)
        if regulation is None:
            return False, [{"error": f"Regulation {regulation_id} not found"}]

        if regulation.version == current_version:
            return True, []

        # If versions differ, return all changes from the regulation
        return False, list(regulation.changes)

    def assess_impact(
        self,
        change: Dict[str, str],
        system_components: Dict[str, Any],
    ) -> ChangeImpact:
        """Assess the impact of a regulatory change on system components.

        Args:
            change: Dict with keys: description, affected_areas, change_type
            system_components: Dict mapping component names to their metadata

        Returns:
            ChangeImpact with detailed assessment
        """
        description = change.get("description", "Unknown change")
        affected_areas = change.get("affected_areas", [])
        change_type = change.get("change_type", "amendment")

        affected = []
        severity = "low"
        action = ""

        # Determine affected components
        for comp_name, comp_meta in system_components.items():
            comp_categories = comp_meta.get("categories", [])
            comp_regulations = comp_meta.get("regulated_by", [])

            for area in affected_areas:
                if area.lower() in [c.lower() for c in comp_categories]:
                    affected.append(comp_name)
                if area.lower() in [r.lower() for r in comp_regulations]:
                    affected.append(comp_name)

        # Deduplicate
        affected = list(dict.fromkeys(affected))

        # Determine severity
        if change_type == "prohibition":
            severity = "critical"
            action = "Immediate action required: review and potentially remove affected functionality"
        elif change_type == "new_requirement":
            severity = "high"
            action = "Implement new requirements before effective date"
        elif change_type == "amendment":
            if len(affected) >= 3:
                severity = "high"
                action = "Update multiple affected components"
            elif len(affected) >= 1:
                severity = "medium"
                action = "Update affected component(s): " + ", ".join(affected)
            else:
                severity = "low"
                action = "Review change for potential future impact"
        elif change_type == "clarification":
            severity = "low"
            action = "Review and confirm existing compliance"
        elif change_type == "repeal":
            severity = "medium"
            action = "Review removed requirements; update documentation"
        else:
            severity = "medium"
            action = "Review change and assess impact"

        return ChangeImpact(
            regulation=change.get("regulation_id", "unknown"),
            change=description,
            affected_components=affected,
            severity=severity,
            action_required=action,
        )

    def generate_compliance_gap(
        self,
        system: Dict[str, Any],
        regulations: List[str],
    ) -> ComplianceGapReport:
        """Generate a compliance gap analysis for a system.

        Args:
            system: Dict with system metadata, components, current_compliance
            regulations: List of regulation IDs to check

        Returns:
            ComplianceGapReport with gap analysis
        """
        gaps = []
        total_reqs = 0
        met_reqs = 0

        system_name = system.get("name", "Unknown System")
        components = system.get("components", {})
        current_compliance = system.get("current_compliance", {})

        for reg_id in regulations:
            regulation = self._regulations.get(reg_id)
            if regulation is None:
                gaps.append(ComplianceGap(
                    requirement=f"{reg_id}: registration",
                    current_status="not_registered",
                    required_status="registered and tracked",
                    severity="medium",
                    remediation=f"Register regulation {reg_id} in the tracker",
                ))
                total_reqs += 1
                continue

            # Check each change/requirement
            for change in regulation.changes:
                total_reqs += 1
                change_id = change.get("id", change.get("description", "unknown"))

                if change_id in current_compliance:
                    met_reqs += 1
                else:
                    severity = "medium"
                    remediation = f"Implement: {change.get('description', '')}"

                    if change.get("type") == "prohibition":
                        severity = "critical"
                        remediation = "URGENT: Review and remove prohibited functionality"
                    elif change.get("type") == "new_requirement":
                        severity = "high"
                        remediation = "Implement new requirement before effective date"

                    gaps.append(ComplianceGap(
                        requirement=f"{reg_id}: {change_id}",
                        current_status="not_compliant",
                        required_status="compliant",
                        severity=severity,
                        remediation=remediation,
                    ))

        score = met_reqs / total_reqs if total_reqs > 0 else 1.0

        return ComplianceGapReport(
            system_name=system_name,
            regulation=", ".join(regulations),
            gaps=gaps,
            overall_score=round(score * 100, 1),
            total_requirements=total_reqs,
            met_requirements=met_reqs,
            gaps_found=len(gaps),
            generated_at=datetime.utcnow().isoformat(),
        )

    def track_changes(
        self,
        since_date: str,
    ) -> List[Dict[str, Any]]:
        """Track all regulatory changes since a given date.

        Args:
            since_date: ISO format date string to track changes since

        Returns:
            List of regulatory changes with metadata
        """
        changes = []

        for reg_id, regulation in self._regulations.items():
            for change in regulation.changes:
                change_date = change.get("date", regulation.effective_date)
                if change_date >= since_date:
                    changes.append({
                        "regulation_id": reg_id,
                        "regulation_title": regulation.title,
                        "regulation_version": regulation.version,
                        "change_id": change.get("id", ""),
                        "change_description": change.get("description", ""),
                        "change_type": change.get("type", "amendment"),
                        "change_date": change_date,
                        "effective_date": regulation.effective_date,
                    })

        # Sort by date descending (most recent first)
        changes.sort(key=lambda x: x.get("change_date", ""), reverse=True)
        return changes

    def compute_compliance_score(
        self,
        system: Dict[str, Any],
        regulations: List[str],
    ) -> float:
        """Compute overall compliance score for a system against regulations.

        Args:
            system: System metadata and compliance state
            regulations: List of regulation IDs to score against

        Returns:
            Compliance score 0.0-100.0
        """
        gap_report = self.generate_compliance_gap(system, regulations)
        return gap_report.overall_score

    def get_regulation(self, regulation_id: str) -> Optional[RegulationVersion]:
        """Get a regulation by ID.

        Args:
            regulation_id: Regulation ID

        Returns:
            RegulationVersion if found, None otherwise
        """
        return self._regulations.get(regulation_id)

    def list_regulations(self) -> List[str]:
        """List all tracked regulation IDs.

        Returns:
            List of regulation IDs
        """
        return list(self._regulations.keys())

    def remove_regulation(self, regulation_id: str) -> bool:
        """Remove a regulation from the tracker.

        Args:
            regulation_id: Regulation ID to remove

        Returns:
            True if removed, False if not found
        """
        if regulation_id in self._regulations:
            del self._regulations[regulation_id]
            return True
        return False
