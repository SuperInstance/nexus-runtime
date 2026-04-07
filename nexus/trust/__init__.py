"""NEXUS Trust Engine — INCREMENTS multi-dimensional trust for multi-agent cooperation."""

from nexus.trust.increments import (
    TrustWeights, TrustDimensions, HistoryTracker,
    compute_history, compute_capability, compute_latency,
    compute_consistency, compute_composite,
)

__all__ = [
    "TrustEngine", "AgentRecord", "TrustProfile", "TrustWeights",
    "TrustDimensions", "HistoryTracker",
    "compute_history", "compute_capability", "compute_latency",
    "compute_consistency", "compute_composite",
]
