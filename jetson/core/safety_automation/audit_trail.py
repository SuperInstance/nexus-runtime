"""NEXUS Audit Trail Generator — generates audit documents from structured data.

Produces JSON audit logs and human-readable markdown summaries.
Includes compliance templates for IEC 61508 and EU AI Act.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class AuditEntry:
    """Single audit event."""
    timestamp: str
    event_type: str  # bytecode_deploy, trust_change, safety_event, policy_violation, reflex_update
    vessel_id: str = ""
    agent_id: str = ""
    description: str = ""
    outcome: str = ""  # success, failure, warning
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AuditEntry:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AuditTrail:
    """Collection of audit entries with generation capabilities."""
    entries: list[AuditEntry] = field(default_factory=list)
    vessel_id: str = ""
    session_id: str = ""

    def add_entry(self, entry: AuditEntry) -> None:
        self.entries.append(entry)

    def add_event(self, event_type: str, description: str = "",
                  outcome: str = "", metadata: dict[str, Any] | None = None) -> AuditEntry:
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type, vessel_id=self.vessel_id,
            description=description, outcome=outcome,
            metadata=metadata or {},
        )
        self.entries.append(entry)
        return entry

    def filter_by_type(self, event_type: str) -> list[AuditEntry]:
        return [e for e in self.entries if e.event_type == event_type]

    def filter_by_outcome(self, outcome: str) -> list[AuditEntry]:
        return [e for e in self.entries if e.outcome == outcome]

    def filter_by_time(self, start: str, end: str) -> list[AuditEntry]:
        return [e for e in self.entries if start <= e.timestamp <= end]

    def to_json(self) -> str:
        return json.dumps({
            "vessel_id": self.vessel_id,
            "session_id": self.session_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "entry_count": len(self.entries),
            "entries": [e.to_dict() for e in self.entries],
        }, indent=2, default=str)

    def to_markdown(self) -> str:
        lines = [
            f"# NEXUS Audit Trail",
            f"",
            f"**Vessel**: {self.vessel_id}  ",
            f"**Session**: {self.session_id}  ",
            f"**Generated**: {datetime.now(timezone.utc).isoformat()}  ",
            f"**Entries**: {len(self.entries)}  ",
            f"",
            f"## Summary",
            f"",
        ]
        # Count by type
        type_counts: dict[str, int] = {}
        outcome_counts: dict[str, int] = {}
        for e in self.entries:
            type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1
            outcome_counts[e.outcome] = outcome_counts.get(e.outcome, 0) + 1

        lines.append(f"| Event Type | Count |")
        lines.append(f"|---|---|")
        for etype, count in sorted(type_counts.items()):
            lines.append(f"| {etype} | {count} |")
        lines.append("")
        lines.append(f"| Outcome | Count |")
        lines.append(f"|---|---|")
        for outcome, count in sorted(outcome_counts.items()):
            lines.append(f"| {outcome} | {count} |")
        lines.append("")

        # Detailed entries
        lines.append(f"## Detailed Events")
        lines.append("")
        for entry in self.entries[-50:]:  # last 50 entries
            lines.append(f"### [{entry.timestamp}] {entry.event_type}")
            lines.append(f"")
            if entry.description:
                lines.append(f"{entry.description}")
                lines.append(f"")
            if entry.outcome:
                lines.append(f"**Outcome**: {entry.outcome}")
                lines.append(f"")
            if entry.metadata:
                for k, v in entry.metadata.items():
                    lines.append(f"- **{k}**: {v}")
                lines.append(f"")
        return "\n".join(lines)

    def compliance_report_iec61508(self) -> str:
        """Generate IEC 61508 compliance scaffold."""
        lines = [
            "# IEC 61508 Compliance Report — NEXUS Audit Trail",
            "",
            f"**Vessel**: {self.vessel_id}",
            f"**Report Date**: {datetime.now(timezone.utc).isoformat()}",
            f"**Audit Period**: {self.entries[0].timestamp if self.entries else 'N/A'} to {self.entries[-1].timestamp if self.entries else 'N/A'}",
            "",
            "## 1. Safety Requirements Specification",
            "",
            "This section documents the safety requirements allocated to the NEXUS system.",
            "For each safety function, the following shall be specified:",
            "- Safety function description",
            "- Required Safety Integrity Level (SIL)",
            "- Response time requirements",
            "- Safe state definition",
            "",
            "## 2. Safety Validation",
            "",
            f"Total audited events: {len(self.entries)}",
            f"Safety events: {len(self.filter_by_type('safety_event'))}",
            f"Policy violations: {len(self.filter_by_type('policy_violation'))}",
            f"Failed operations: {len(self.filter_by_outcome('failure'))}",
            "",
            "## 3. Safety Assessment",
            "",
            "This audit trail serves as evidence for safety assessment activities.",
            "All bytecode deployments, trust changes, and safety events are recorded",
            "with timestamps and outcomes for traceability.",
            "",
            "## 4. Recommendations",
            "",
            "- Review all policy_violation events for systematic causes",
            "- Verify that all safety_event outcomes resolved to safe states",
            "- Ensure bytecode_deploy events have corresponding validation records",
            "",
        ]
        return "\n".join(lines)

    def compliance_report_eu_ai_act(self) -> str:
        """Generate EU AI Act compliance scaffold."""
        lines = [
            "# EU AI Act Compliance Report — NEXUS Audit Trail",
            "",
            f"**Vessel**: {self.vessel_id}",
            f"**Report Date**: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## 1. Risk Classification",
            "",
            "NEXUS autonomous marine vessels may fall under 'High-Risk AI System' classification",
            "under the EU AI Act when used in commercial maritime operations.",
            "",
            "## 2. Transparency Requirements",
            "",
            f"Total recorded decisions: {len(self.entries)}",
            f"Reflex deployments: {len(self.filter_by_type('bytecode_deploy'))}",
            f"Trust score changes: {len(self.filter_by_type('trust_change'))}",
            "",
            "## 3. Human Oversight",
            "",
            "The NEXUS tripartite consensus system (Pathos/Logos/Ethos) provides multi-agent",
            "oversight for all safety-critical decisions. Human override capability is maintained",
            "at all trust levels.",
            "",
            "## 4. Accuracy and Robustness",
            "",
            f"Failed operations: {len(self.filter_by_outcome('failure'))}",
            f"Warnings issued: {len(self.filter_by_outcome('warning'))}",
            "",
            "## 5. Audit Trail Completeness",
            "",
            "This document serves as the required audit trail under Article 12 of the EU AI Act.",
            "All system decisions are logged with timestamps, outcomes, and contextual metadata.",
            "",
        ]
        return "\n".join(lines)
