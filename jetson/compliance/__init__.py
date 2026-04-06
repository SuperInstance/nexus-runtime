"""NEXUS Compliance Automation Engine — Phase 4 Round 9.

Provides IEC 61508 SIL verification, EU AI Act compliance checking,
COLREGs rule verification, audit trail generation, evidence collection,
and regulatory change tracking for marine robotics systems.
"""

from .iec61508 import (
    SILLevel,
    SILTarget,
    SILVerificationResult,
    SILVerifier,
)
from .eu_ai_act import (
    RiskCategory,
    AIRequirement,
    ComplianceReport,
    EUAIActChecker,
)
from .colregs import (
    VesselSituation,
    RuleReference,
    COLREGsResult,
    COLREGsChecker,
)
from .audit_trail import (
    AuditEntry,
    AuditTrailConfig,
    AuditTrailGenerator,
)
from .evidence import (
    EvidenceItem,
    EvidenceCollection,
    EvidenceCollector,
)
from .regulatory import (
    RegulationVersion,
    ChangeImpact,
    RegulatoryTracker,
)

__all__ = [
    "SILLevel", "SILTarget", "SILVerificationResult", "SILVerifier",
    "RiskCategory", "AIRequirement", "ComplianceReport", "EUAIActChecker",
    "VesselSituation", "RuleReference", "COLREGsResult", "COLREGsChecker",
    "AuditEntry", "AuditTrailConfig", "AuditTrailGenerator",
    "EvidenceItem", "EvidenceCollection", "EvidenceCollector",
    "RegulationVersion", "ChangeImpact", "RegulatoryTracker",
]
