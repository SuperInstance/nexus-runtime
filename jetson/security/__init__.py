"""NEXUS Security Module — adversarial robustness, fault injection, trust boundaries, HMAC auth, Byzantine tolerance."""

from .attack_detection import AnomalyRecord, AttackSignature, SensorAnomalyDetector, CommandInjector
from .fault_injection import FaultConfig, FaultScenario, FaultInjector, FaultTestRunner
from .trust_boundary import TrustBoundary, TrustPolicy, TrustBoundaryEnforcer
from .authentication import MessageEnvelope, AuthResult, MessageAuthenticator, KeyManager
from .byzantine import Vote, ConsensusResult, ByzantineConsensus
from .safety_monitor import SafetyInvariant, SafetyViolation, SafetyInvariantMonitor

__all__ = [
    "AnomalyRecord", "AttackSignature", "SensorAnomalyDetector", "CommandInjector",
    "FaultConfig", "FaultScenario", "FaultInjector", "FaultTestRunner",
    "TrustBoundary", "TrustPolicy", "TrustBoundaryEnforcer",
    "MessageEnvelope", "AuthResult", "MessageAuthenticator", "KeyManager",
    "Vote", "ConsensusResult", "ByzantineConsensus",
    "SafetyInvariant", "SafetyViolation", "SafetyInvariantMonitor",
]
