"""Audit Trail Generation Engine.

Provides tamper-evident audit trail generation, signing, verification,
export, and compliance reporting for safety-critical marine robotics systems.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import json


@dataclass
class AuditEntry:
    """Single audit trail entry."""
    timestamp: str
    actor: str
    action: str
    component: str
    details: Dict[str, Any] = field(default_factory=dict)
    evidence_refs: List[str] = field(default_factory=list)
    signature: str = ""
    entry_id: str = ""


@dataclass
class AuditTrailConfig:
    """Configuration for audit trail generation."""
    retention_period: int = 365  # days
    required_categories: List[str] = field(default_factory=lambda: [
        "safety_check", "configuration_change", "deployment",
        "compliance", "incident", "maintenance",
    ])
    signing_enabled: bool = True
    max_entries: int = 100000


@dataclass
class ComplianceTrailReport:
    """Report generated from audit trail for a specific regulation."""
    regulation: str
    total_entries: int = 0
    entries_by_category: Dict[str, int] = field(default_factory=dict)
    completeness_score: float = 0.0
    gaps: List[str] = field(default_factory=list)
    evidence_count: int = 0
    generated_at: str = ""


class AuditTrailGenerator:
    """Audit trail generation and management engine."""

    def __init__(self, config: Optional[AuditTrailConfig] = None) -> None:
        self._config = config or AuditTrailConfig()
        self._entries: List[AuditEntry] = []
        self._entry_counter = 0

    @property
    def config(self) -> AuditTrailConfig:
        return self._config

    @property
    def entries(self) -> List[AuditEntry]:
        return list(self._entries)

    def record_entry(self, entry: AuditEntry) -> None:
        """Record a new audit trail entry.

        Args:
            entry: AuditEntry to record
        """
        self._entry_counter += 1
        entry.entry_id = f"audit-{self._entry_counter:06d}"
        if not entry.timestamp:
            entry.timestamp = datetime.utcnow().isoformat()
        self._entries.append(entry)

    def create_entry(
        self,
        actor: str,
        action: str,
        component: str,
        details: Optional[Dict[str, Any]] = None,
        evidence_refs: Optional[List[str]] = None,
        category: Optional[str] = None,
    ) -> AuditEntry:
        """Create and record a new audit entry.

        Args:
            actor: Who performed the action
            action: What action was performed
            component: Which component was affected
            details: Additional details about the action
            evidence_refs: References to evidence artifacts
            category: Category of the audit entry

        Returns:
            The created AuditEntry
        """
        entry_details = details or {}
        if category:
            entry_details["category"] = category

        entry = AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            actor=actor,
            action=action,
            component=component,
            details=entry_details,
            evidence_refs=evidence_refs or [],
        )
        self.record_entry(entry)
        return entry

    def generate_trail(
        self,
        component: str,
        start_time: str,
        end_time: str,
    ) -> List[AuditEntry]:
        """Generate filtered audit trail for a component within a time range.

        Args:
            component: Component name to filter by
            start_time: ISO format start timestamp
            end_time: ISO format end timestamp

        Returns:
            List of matching AuditEntry objects
        """
        filtered = []
        for entry in self._entries:
            if entry.component != component:
                continue
            if entry.timestamp < start_time:
                continue
            if entry.timestamp > end_time:
                continue
            filtered.append(entry)
        return filtered

    def generate_compliance_report(
        self,
        trail: List[AuditEntry],
        regulation: str,
    ) -> ComplianceTrailReport:
        """Generate a compliance report from an audit trail.

        Args:
            trail: List of AuditEntry objects
            regulation: Regulation identifier (e.g., "IEC_61508", "EU_AI_ACT")

        Returns:
            ComplianceTrailReport with analysis
        """
        entries_by_category: Dict[str, int] = {}
        evidence_count = 0

        for entry in trail:
            category = entry.details.get("category", "uncategorized")
            entries_by_category[category] = entries_by_category.get(category, 0) + 1
            evidence_count += len(entry.evidence_refs)

        completeness = self.compute_completeness(trail, self._config.required_categories)

        gaps = []
        for cat in self._config.required_categories:
            if cat not in entries_by_category or entries_by_category[cat] == 0:
                gaps.append(f"No audit entries for category: {cat}")

        return ComplianceTrailReport(
            regulation=regulation,
            total_entries=len(trail),
            entries_by_category=entries_by_category,
            completeness_score=completeness,
            gaps=gaps,
            evidence_count=evidence_count,
            generated_at=datetime.utcnow().isoformat(),
        )

    def sign_entry(self, entry: AuditEntry, key: str) -> AuditEntry:
        """Sign an audit entry with a key for tamper-evidence.

        Args:
            entry: AuditEntry to sign
            key: Signing key (secret)

        Returns:
            Signed AuditEntry (signature field populated)
        """
        data = self._entry_to_canonical(entry)
        signature = hashlib.sha256((data + key).encode()).hexdigest()
        entry.signature = signature
        return entry

    def verify_signature(self, entry: AuditEntry, key: str) -> bool:
        """Verify the signature of an audit entry.

        Args:
            entry: AuditEntry to verify
            key: Signing key (must match the key used for signing)

        Returns:
            True if signature is valid
        """
        if not entry.signature:
            return False

        data = self._entry_to_canonical(entry)
        expected = hashlib.sha256((data + key).encode()).hexdigest()
        return entry.signature == expected

    def _entry_to_canonical(self, entry: AuditEntry) -> str:
        """Convert an entry to a canonical string representation for signing."""
        data = {
            "timestamp": entry.timestamp,
            "actor": entry.actor,
            "action": entry.action,
            "component": entry.component,
            "details": entry.details,
            "evidence_refs": entry.evidence_refs,
            "entry_id": entry.entry_id,
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    def export_trail(
        self,
        entries: List[AuditEntry],
        format: str = "json",
    ) -> str:
        """Export audit trail entries in specified format.

        Args:
            entries: List of AuditEntry objects
            format: Export format — "json", "csv", "text"

        Returns:
            Formatted string representation
        """
        if format == "json":
            return self._export_json(entries)
        elif format == "csv":
            return self._export_csv(entries)
        elif format == "text":
            return self._export_text(entries)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_json(self, entries: List[AuditEntry]) -> str:
        """Export as JSON array."""
        data = []
        for entry in entries:
            data.append({
                "entry_id": entry.entry_id,
                "timestamp": entry.timestamp,
                "actor": entry.actor,
                "action": entry.action,
                "component": entry.component,
                "details": entry.details,
                "evidence_refs": entry.evidence_refs,
                "signature": entry.signature,
            })
        return json.dumps(data, indent=2)

    def _export_csv(self, entries: List[AuditEntry]) -> str:
        """Export as CSV."""
        lines = ["entry_id,timestamp,actor,action,component,category,signature"]
        for entry in entries:
            category = entry.details.get("category", "")
            lines.append(
                f'"{entry.entry_id}","{entry.timestamp}","{entry.actor}",'
                f'"{entry.action}","{entry.component}","{category}",'
                f'"{entry.signature}"'
            )
        return "\n".join(lines)

    def _export_text(self, entries: List[AuditEntry]) -> str:
        """Export as human-readable text."""
        lines = []
        lines.append("=" * 70)
        lines.append("NEXUS AUDIT TRAIL")
        lines.append("=" * 70)
        lines.append(f"Generated: {datetime.utcnow().isoformat()}")
        lines.append(f"Total entries: {len(entries)}")
        lines.append("=" * 70)

        for entry in entries:
            lines.append(f"[{entry.entry_id}] {entry.timestamp}")
            lines.append(f"  Actor:    {entry.actor}")
            lines.append(f"  Action:   {entry.action}")
            lines.append(f"  Component: {entry.component}")
            if entry.details:
                lines.append(f"  Details:  {json.dumps(entry.details, default=str)}")
            if entry.evidence_refs:
                lines.append(f"  Evidence: {', '.join(entry.evidence_refs)}")
            if entry.signature:
                lines.append(f"  Signature: {entry.signature[:16]}...")
            lines.append("-" * 70)

        return "\n".join(lines)

    def compute_completeness(
        self,
        entries: List[AuditEntry],
        required_categories: Optional[List[str]] = None,
    ) -> float:
        """Compute completeness score of audit trail.

        Args:
            entries: List of AuditEntry objects
            required_categories: Categories that must be present

        Returns:
            Completeness score 0.0-1.0
        """
        if required_categories is None:
            required_categories = self._config.required_categories

        if not required_categories:
            return 1.0

        found_categories = set()
        for entry in entries:
            category = entry.details.get("category", "uncategorized")
            found_categories.add(category)

        required_set = set(required_categories)
        if not required_set:
            return 1.0

        covered = required_set & found_categories
        return len(covered) / len(required_set)

    def get_entries_by_actor(self, actor: str) -> List[AuditEntry]:
        """Get all entries by a specific actor.

        Args:
            actor: Actor name to filter by

        Returns:
            List of matching entries
        """
        return [e for e in self._entries if e.actor == actor]

    def get_entries_by_category(self, category: str) -> List[AuditEntry]:
        """Get all entries in a specific category.

        Args:
            category: Category to filter by

        Returns:
            List of matching entries
        """
        return [
            e for e in self._entries
            if e.details.get("category") == category
        ]

    def clear_entries(self) -> None:
        """Clear all audit entries."""
        self._entries.clear()
        self._entry_counter = 0
