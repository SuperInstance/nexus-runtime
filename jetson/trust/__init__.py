"""NEXUS trust package.

Provides the INCREMENTS trust algorithm, autonomy level definitions,
event classification, trust propagation, and cryptographic attestation.
"""

from trust.increments import (
    IncrementTrustEngine,
    TrustParams,
    TrustEvent,
    SubsystemTrust,
    TrustUpdateResult,
    SUBSYSTEMS,
    IncrementsParams,
    IncrementsEngine,
)
from trust.levels import (
    AutonomyLevel,
    AUTONOMY_LEVELS,
    get_level_definition,
    can_promote,
)
from trust.events import (
    EventDefinition,
    EVENT_DEFINITIONS,
    classify_event,
    get_good_events,
    get_bad_events,
    get_neutral_events,
)
from trust.propagation import (
    TrustPropagator,
    VesselTrust,
    AgentTrust,
    FleetTrust,
    DEFAULT_ATTENUATION,
    MAX_PROPAGATION_RADIUS,
    TRUST_MERGE_ALPHA,
)
from trust.attestation import (
    TrustAttestation,
    AttestationPayload,
)
from trust.test_vectors import (
    TrustVectorSpec,
    GoldenReference,
    TRUST_TEST_VECTORS,
    TEST_VECTOR_MAP,
    run_vector,
    run_vector_detailed,
    regenerate_golden_references,
)

__all__ = [
    # Engine
    "IncrementTrustEngine",
    "TrustParams",
    "TrustEvent",
    "SubsystemTrust",
    "TrustUpdateResult",
    "SUBSYSTEMS",
    "IncrementsParams",
    "IncrementsEngine",
    # Levels
    "AutonomyLevel",
    "AUTONOMY_LEVELS",
    "get_level_definition",
    "can_promote",
    # Events
    "EventDefinition",
    "EVENT_DEFINITIONS",
    "classify_event",
    "get_good_events",
    "get_bad_events",
    "get_neutral_events",
    # Propagation
    "TrustPropagator",
    "VesselTrust",
    "AgentTrust",
    "FleetTrust",
    "DEFAULT_ATTENUATION",
    "MAX_PROPAGATION_RADIUS",
    "TRUST_MERGE_ALPHA",
    # Attestation
    "TrustAttestation",
    "AttestationPayload",
    # Test vectors
    "TrustVectorSpec",
    "GoldenReference",
    "TRUST_TEST_VECTORS",
    "TEST_VECTOR_MAP",
    "run_vector",
    "run_vector_detailed",
    "regenerate_golden_references",
]
