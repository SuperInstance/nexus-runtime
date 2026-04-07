"""
NEXUS Trust Increments — formula implementations for the INCREMENTS trust model.

Dimensions:
    T_history    — Exponential moving average of binary outcomes
    T_capability — Capability match score (0.0 - 1.0)
    T_latency    — Communication latency factor (inverse relationship)
    T_consistency — Behavioral consistency (1.0 - coefficient of variation)

Composite:
    T(a,b,t) = α·T_history + β·T_capability + γ·T_latency + δ·T_consistency
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

# TrustWeights defined below to avoid circular imports


# ---------------------------------------------------------------------------
# Trust dimensions
# ---------------------------------------------------------------------------

@dataclass
class TrustDimensions:
    """Container for the four trust dimensions."""

    history: float = 0.5
    capability: float = 0.5
    latency: float = 0.5
    consistency: float = 0.5


# ---------------------------------------------------------------------------
# History tracker (exponential moving average)
# ---------------------------------------------------------------------------

@dataclass
class HistoryTracker:
    """Tracks interaction history using exponential moving average (EMA).

    Parameters
    ----------
    alpha : float
        Smoothing factor for EMA (0.0-1.0). Higher = more weight on recent.
    max_samples : int
        Maximum number of raw samples to retain for consistency calculation.
    """

    alpha: float = 0.3
    max_samples: int = 100
    _samples: Deque[float] = field(default_factory=deque)
    _latencies: Deque[float] = field(default_factory=deque)
    _ema: float = 0.5
    _count: int = 0

    @property
    def count(self) -> int:
        return self._count

    @property
    def ema(self) -> float:
        return self._ema

    def record(self, success: bool, latency_ms: float = 0.0) -> None:
        """Record a single interaction outcome."""
        value = 1.0 if success else 0.0

        # Update EMA
        if self._count == 0:
            self._ema = value
        else:
            self._ema = self.alpha * value + (1 - self.alpha) * self._ema

        self._count += 1

        # Store samples
        self._samples.append(value)
        if len(self._samples) > self.max_samples:
            self._samples.popleft()

        if latency_ms > 0:
            self._latencies.append(latency_ms)
            if len(self._latencies) > self.max_samples:
                self._latencies.popleft()

    def get_ema(self) -> float:
        """Get current exponential moving average of trust."""
        return self._ema

    def get_mean(self) -> float:
        """Get arithmetic mean of all recorded outcomes."""
        if not self._samples:
            return 0.5
        return sum(self._samples) / len(self._samples)

    def get_variance(self) -> float:
        """Get variance of recorded outcomes."""
        if len(self._samples) < 2:
            return 0.0
        mean = self.get_mean()
        return sum((x - mean) ** 2 for x in self._samples) / len(self._samples)

    def get_stddev(self) -> float:
        """Get standard deviation of recorded outcomes."""
        return math.sqrt(self.get_variance())

    def get_consistency(self) -> float:
        """Compute consistency score (1.0 = perfectly consistent, 0.0 = random).

        Uses: consistency = 1.0 - coefficient_of_variation
        where CV = stddev / mean (clamped to avoid division by zero).
        """
        if len(self._samples) < 2:
            return 0.5
        mean = self.get_mean()
        if mean < 0.001:
            return 0.5
        cv = self.get_stddev() / mean
        return max(0.0, min(1.0, 1.0 - cv))

    def get_latency_stats(self) -> dict:
        """Get latency statistics."""
        if not self._latencies:
            return {"mean": 0.0, "stddev": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        lats = list(self._latencies)
        mean = sum(lats) / len(lats)
        variance = sum((x - mean) ** 2 for x in lats) / len(lats)
        return {
            "mean": mean,
            "stddev": math.sqrt(variance),
            "min": min(lats),
            "max": max(lats),
            "count": len(lats),
        }

    def reset(self) -> None:
        """Clear all history."""
        self._samples.clear()
        self._latencies.clear()
        self._ema = 0.5
        self._count = 0


# ---------------------------------------------------------------------------
# Dimension formulas
# ---------------------------------------------------------------------------

def compute_history(tracker: HistoryTracker) -> float:
    """Compute T_history from a history tracker (EMA-based)."""
    return tracker.get_ema()


def compute_capability(
    agent_capabilities: dict,
    required_capabilities: dict,
) -> float:
    """Compute T_capability based on capability matching.

    Parameters
    ----------
    agent_capabilities : dict
        Agent's capability scores {domain: float}.
    required_capabilities : dict
        Required capability scores {domain: float}.

    Returns
    -------
    float
        Match score 0.0-1.0.
    """
    if not required_capabilities:
        return 1.0

    scores: List[float] = []
    for domain, required in required_capabilities.items():
        if required <= 0:
            scores.append(1.0)
        else:
            agent_val = agent_capabilities.get(domain, 0.0)
            scores.append(min(agent_val, required) / required)

    return sum(scores) / len(scores) if scores else 0.0


def compute_latency(
    latency_ms: float,
    target_latency_ms: float = 100.0,
    max_latency_ms: float = 1000.0,
) -> float:
    """Compute T_latency from observed latency.

    Returns a value between 0.0 (worst) and 1.0 (best).
    """
    if latency_ms <= 0:
        return 0.5  # unknown
    if latency_ms <= target_latency_ms:
        return 1.0
    if latency_ms >= max_latency_ms:
        return 0.0
    # Linear interpolation between target and max
    return 1.0 - (latency_ms - target_latency_ms) / (max_latency_ms - target_latency_ms)


def compute_consistency(samples: List[float]) -> float:
    """Compute T_consistency from a list of outcome samples.

    Higher consistency = more predictable behavior = more trust.
    """
    if len(samples) < 2:
        return 0.5
    mean = sum(samples) / len(samples)
    if mean < 0.001:
        return 0.5
    variance = sum((x - mean) ** 2 for x in samples) / len(samples)
    stddev = math.sqrt(variance)
    cv = stddev / mean
    return max(0.0, min(1.0, 1.0 - cv))




# ---------------------------------------------------------------------------
# Trust weights
# ---------------------------------------------------------------------------

@dataclass
class TrustWeights:
    """Configurable weights for the INCREMENTS trust formula."""

    alpha: float = 0.35   # T_history weight
    beta: float = 0.25    # T_capability weight
    gamma: float = 0.20   # T_latency weight
    delta: float = 0.20   # T_consistency weight

    def normalize(self) -> "TrustWeights":
        """Normalize weights so they sum to 1.0."""
        total = self.alpha + self.beta + self.gamma + self.delta
        if total == 0:
            return TrustWeights(0.25, 0.25, 0.25, 0.25)
        return TrustWeights(
            alpha=self.alpha / total,
            beta=self.beta / total,
            gamma=self.gamma / total,
            delta=self.delta / total,
        )

def compute_composite(
    dims: TrustDimensions,
    weights: Optional[TrustWeights] = None,
) -> float:
    """Compute the composite trust score using INCREMENTS formula.

    T(a,b,t) = α·T_history + β·T_capability + γ·T_latency + δ·T_consistency
    """
    w = weights or TrustWeights()
    score = (
        w.alpha * dims.history
        + w.beta * dims.capability
        + w.gamma * dims.latency
        + w.delta * dims.consistency
    )
    return max(0.0, min(1.0, score))
