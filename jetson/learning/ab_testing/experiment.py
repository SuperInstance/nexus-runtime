"""NEXUS A/B Testing — Experiment definition and management.

Defines the experiment with variants, metrics, duration, and power analysis.
Each variant carries bytecode and metadata. Metrics tracked: cycle_time_ms,
accuracy, trust_delta, safety_events, error_rate.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MetricType(Enum):
    """Types of metrics tracked in A/B experiments."""

    CYCLE_TIME_MS = "cycle_time_ms"
    ACCURACY = "accuracy"
    TRUST_DELTA = "trust_delta"
    SAFETY_EVENTS = "safety_events"
    ERROR_RATE = "error_rate"


@dataclass
class MetricRecord:
    """A single metric observation for a variant iteration."""

    metric_type: MetricType
    value: float
    timestamp_ms: int = 0
    iteration: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_type": self.metric_type.value,
            "value": self.value,
            "timestamp_ms": self.timestamp_ms,
            "iteration": self.iteration,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MetricRecord:
        return cls(
            metric_type=MetricType(d["metric_type"]),
            value=d["value"],
            timestamp_ms=d.get("timestamp_ms", 0),
            iteration=d.get("iteration", 0),
        )


@dataclass
class ExperimentVariant:
    """A single variant in an A/B test experiment.

    Carries bytecode, metadata, and accumulated metric records.
    """

    name: str
    bytecode: bytes = b""
    metadata: dict[str, str] = field(default_factory=dict)
    metrics: list[MetricRecord] = field(default_factory=list)

    def record_metric(
        self,
        metric_type: MetricType,
        value: float,
        timestamp_ms: int = 0,
        iteration: int = 0,
    ) -> None:
        """Record a metric observation for this variant."""
        self.metrics.append(
            MetricRecord(
                metric_type=metric_type,
                value=value,
                timestamp_ms=timestamp_ms,
                iteration=iteration,
            )
        )

    def get_metric_values(self, metric_type: MetricType) -> list[float]:
        """Get all recorded values for a specific metric."""
        return [m.value for m in self.metrics if m.metric_type == metric_type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bytecode": self.bytecode.hex(),
            "bytecode_len": len(self.bytecode),
            "metadata": self.metadata,
            "metrics": [m.to_dict() for m in self.metrics],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentVariant:
        bc = bytes.fromhex(d.get("bytecode", "")) if "bytecode" in d else b""
        metrics = [MetricRecord.from_dict(m) for m in d.get("metrics", [])]
        return cls(
            name=d["name"],
            bytecode=bc,
            metadata=d.get("metadata", {}),
            metrics=metrics,
        )


@dataclass
class PowerAnalysisResult:
    """Result of statistical power analysis for sample size calculation."""

    effect_size: float
    alpha: float
    power: float
    min_sample_size: int
    test_name: str = "welch_t"

    def to_dict(self) -> dict[str, Any]:
        return {
            "effect_size": self.effect_size,
            "alpha": self.alpha,
            "power": self.power,
            "min_sample_size": self.min_sample_size,
            "test_name": self.test_name,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PowerAnalysisResult:
        return cls(
            effect_size=d["effect_size"],
            alpha=d["alpha"],
            power=d["power"],
            min_sample_size=d["min_sample_size"],
            test_name=d.get("test_name", "welch_t"),
        )


@dataclass
class ABTestResult:
    """Result of a completed A/B test."""

    experiment_name: str
    winner: str
    recommendation: str  # "A wins", "B wins", "inconclusive", "need_more_data"
    metric_summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    statistical_reports: dict[str, dict[str, Any]] = field(default_factory=dict)
    bonferroni_corrected: bool = False
    total_iterations: int = 0
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "winner": self.winner,
            "recommendation": self.recommendation,
            "metric_summaries": self.metric_summaries,
            "statistical_reports": self.statistical_reports,
            "bonferroni_corrected": self.bonferroni_corrected,
            "total_iterations": self.total_iterations,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ABTestResult:
        return cls(
            experiment_name=d["experiment_name"],
            winner=d["winner"],
            recommendation=d["recommendation"],
            metric_summaries=d.get("metric_summaries", {}),
            statistical_reports=d.get("statistical_reports", {}),
            bonferroni_corrected=d.get("bonferroni_corrected", False),
            total_iterations=d.get("total_iterations", 0),
            timestamp=d.get("timestamp", ""),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class ABTestSuite:
    """Manages the full lifecycle of an A/B test experiment.

    Defines the experiment with name, description, variants (A/B/C...),
    metrics to track, and duration. Supports serialization, power analysis,
    and result generation.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        metrics: list[MetricType] | None = None,
        duration_seconds: int = 3600,
        alpha: float = 0.05,
        target_power: float = 0.80,
    ) -> None:
        self.name = name
        self.description = description
        self.metrics = metrics or list(MetricType)
        self.duration_seconds = duration_seconds
        self.alpha = alpha
        self.target_power = target_power
        self.variants: dict[str, ExperimentVariant] = {}
        self.results: ABTestResult | None = None
        self._iteration_counter: int = 0

    def add_variant(
        self,
        name: str,
        bytecode: bytes = b"",
        metadata: dict[str, str] | None = None,
    ) -> ExperimentVariant:
        """Add a variant to the experiment."""
        variant = ExperimentVariant(
            name=name,
            bytecode=bytecode,
            metadata=metadata or {},
        )
        self.variants[name] = variant
        return variant

    def get_variant(self, name: str) -> ExperimentVariant | None:
        """Get a variant by name."""
        return self.variants.get(name)

    def variant_names(self) -> list[str]:
        """Return list of variant names."""
        return list(self.variants.keys())

    def record_metric(
        self,
        variant_name: str,
        metric_type: MetricType,
        value: float,
        timestamp_ms: int = 0,
    ) -> None:
        """Record a metric for a variant."""
        if variant_name not in self.variants:
            raise KeyError(f"Unknown variant: {variant_name}")
        self._iteration_counter += 1
        self.variants[variant_name].record_metric(
            metric_type=metric_type,
            value=value,
            timestamp_ms=timestamp_ms,
            iteration=self._iteration_counter,
        )

    def get_metric_values(self, variant_name: str, metric_type: MetricType) -> list[float]:
        """Get all metric values for a variant."""
        variant = self.variants.get(variant_name)
        if variant is None:
            return []
        return variant.get_metric_values(metric_type)

    def total_observations(self) -> int:
        """Total number of metric observations across all variants."""
        return sum(len(v.metrics) for v in self.variants.values())

    def observations_per_variant(self) -> dict[str, int]:
        """Number of observations per variant."""
        return {name: len(v.metrics) for name, v in self.variants.items()}

    def min_observations(self) -> int:
        """Minimum number of observations across all variants."""
        counts = self.observations_per_variant()
        return min(counts.values()) if counts else 0

    def max_observations(self) -> int:
        """Maximum number of observations across all variants."""
        counts = self.observations_per_variant()
        return max(counts.values()) if counts else 0

    def compute_power_analysis(
        self,
        effect_size: float = 0.5,
        alpha: float | None = None,
        power: float | None = None,
    ) -> PowerAnalysisResult:
        """Compute minimum sample size using power analysis.

        Uses a simplified formula for Welch's t-test two-sample case:
        n >= 2 * ((z_alpha + z_beta) / effect_size)^2

        This is a standard approximation for the two-sample t-test.
        """
        alpha = alpha if alpha is not None else self.alpha
        power = power if power is not None else self.target_power

        # z-value for alpha (two-tailed)
        z_alpha = _normal_ppf(1.0 - alpha / 2.0)
        # z-value for power
        z_beta = _normal_ppf(power)

        min_sample = math.ceil(2.0 * ((z_alpha + z_beta) / effect_size) ** 2)
        return PowerAnalysisResult(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            min_sample_size=min_sample,
            test_name="welch_t",
        )

    def has_sufficient_data(self, effect_size: float = 0.5) -> bool:
        """Check if enough data has been collected per metric per variant."""
        pa = self.compute_power_analysis(effect_size=effect_size)
        min_obs = self.min_observations()
        # Each variant needs at least min_sample_size observations per metric
        # For simplicity, check total observations per variant
        return min_obs >= pa.min_sample_size

    def set_result(self, result: ABTestResult) -> None:
        """Set the experiment result."""
        self.results = result

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "metrics": [m.value for m in self.metrics],
            "duration_seconds": self.duration_seconds,
            "alpha": self.alpha,
            "target_power": self.target_power,
            "variants": {k: v.to_dict() for k, v in self.variants.items()},
            "results": self.results.to_dict() if self.results else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ABTestSuite:
        suite = cls(
            name=d["name"],
            description=d.get("description", ""),
            metrics=[MetricType(m) for m in d.get("metrics", [])],
            duration_seconds=d.get("duration_seconds", 3600),
            alpha=d.get("alpha", 0.05),
            target_power=d.get("target_power", 0.80),
        )
        for vname, vdata in d.get("variants", {}).items():
            suite.variants[vname] = ExperimentVariant.from_dict(vdata)
        if d.get("results"):
            suite.results = ABTestResult.from_dict(d["results"])
        return suite

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> ABTestSuite:
        return cls.from_dict(json.loads(json_str))

    def clear_metrics(self) -> None:
        """Clear all recorded metrics from all variants."""
        self._iteration_counter = 0
        for v in self.variants.values():
            v.metrics.clear()

    def summary_stats(
        self, variant_name: str, metric_type: MetricType
    ) -> dict[str, float]:
        """Compute summary statistics for a variant's metric."""
        values = self.get_metric_values(variant_name, metric_type)
        if not values:
            return {
                "n": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
            }
        n = len(values)
        mean = sum(values) / n
        if n > 1:
            var = sum((x - mean) ** 2 for x in values) / (n - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        sorted_vals = sorted(values)
        median = sorted_vals[n // 2] if n % 2 == 1 else (
            sorted_vals[n // 2 - 1] + sorted_vals[n // 2]
        ) / 2.0
        return {
            "n": n,
            "mean": mean,
            "std": std,
            "min": min(values),
            "max": max(values),
            "median": median,
        }


def _normal_ppf(p: float) -> float:
    """Approximate inverse of the standard normal CDF (quantile function).

    Uses the rational approximation algorithm (Abramowitz and Stegun 26.2.23)
    with Horner form for |p| <= 0.5.
    """
    if p <= 0.0:
        return -10.0
    if p >= 1.0:
        return 10.0

    # For p > 0.5, use symmetry: ppf(1-p) = -ppf(p)
    sign = 1.0
    q = p
    if p > 0.5:
        q = 1.0 - p
        sign = -1.0

    t = math.sqrt(-2.0 * math.log(q))
    # Rational approximation coefficients
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    x = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
    return sign * x
