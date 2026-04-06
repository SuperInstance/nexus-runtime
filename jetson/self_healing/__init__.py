"""Self-healing systems — fault detection, root cause analysis, recovery, resilience, adaptation."""

from .fault_detector import (
    FaultEvent,
    FaultCategory,
    FaultSeverity,
    HealthIndicator,
    IndicatorStatus,
    DegradationReport,
    FaultDetector,
)
from .diagnosis import (
    Diagnosis,
    DiagnosticRule,
    CausalEdge,
    CausalGraph,
    RootCauseAnalyzer,
)
from .recovery import (
    RecoveryAction,
    RecoveryResult,
    RecoveryType,
    RecoveryStrategy,
    Urgency,
    RecoveryManager,
)
from .resilience import (
    MetricTrend,
    ResilienceMetric,
    ResilienceReport,
    SystemResilience,
)
from .adaptation import (
    AdaptationPlan,
    AdaptationResult,
    AdaptationRiskLevel,
    SystemAdapter,
)

__all__ = [
    # fault_detector
    "FaultEvent",
    "FaultCategory",
    "FaultSeverity",
    "HealthIndicator",
    "IndicatorStatus",
    "DegradationReport",
    "FaultDetector",
    # diagnosis
    "Diagnosis",
    "DiagnosticRule",
    "CausalEdge",
    "CausalGraph",
    "RootCauseAnalyzer",
    # recovery
    "RecoveryAction",
    "RecoveryResult",
    "RecoveryType",
    "RecoveryStrategy",
    "Urgency",
    "RecoveryManager",
    # resilience
    "MetricTrend",
    "ResilienceMetric",
    "ResilienceReport",
    "SystemResilience",
    # adaptation
    "AdaptationPlan",
    "AdaptationResult",
    "AdaptationRiskLevel",
    "SystemAdapter",
]
