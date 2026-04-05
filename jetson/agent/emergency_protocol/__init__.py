"""NEXUS Emergency Protocol Bridge — Package initialization.

Connects NEXUS safety events to git-agent fleet coordination through
emergency detection, automated response, and fleet-wide alerting.

Modules:
    protocol:  EmergencyProtocol, Incident, EmergencyAssessment, etc.
    detectors: SensorFailureDetector, TrustCollapseDetector, etc.
    response:  EmergencyResponder
"""

from .protocol import (
    DeescalationResult,
    EmergencyAssessment,
    EmergencyLevel,
    EmergencyProtocol,
    EscalationResult,
    Incident,
    IncidentCategory,
    generate_incident_id,
)
from .detectors import (
    CommunicationLossDetector,
    MissionTimeoutDetector,
    SafetyViolationDetector,
    SensorFailureDetector,
    TrustCollapseDetector,
)
from .response import EmergencyResponder

__all__ = [
    # Protocol
    "EmergencyProtocol",
    "EmergencyLevel",
    "EmergencyAssessment",
    "EscalationResult",
    "DeescalationResult",
    "Incident",
    "IncidentCategory",
    "generate_incident_id",
    # Detectors
    "SensorFailureDetector",
    "TrustCollapseDetector",
    "CommunicationLossDetector",
    "SafetyViolationDetector",
    "MissionTimeoutDetector",
    # Response
    "EmergencyResponder",
]
