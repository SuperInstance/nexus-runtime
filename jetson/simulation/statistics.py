"""Monte Carlo statistics and simulation analysis for NEXUS.

Provides statistical tools for running repeated simulations,
computing confidence intervals, distribution comparison, reliability
metrics, and report generation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class MonteCarloResult:
    """Result of a Monte Carlo statistical analysis."""

    metric_name: str
    mean: float = 0.0
    std_dev: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    samples: int = 0


@dataclass
class SimulationRun:
    """A single simulation run with results and metrics."""

    run_id: int
    config: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


class SimulationStatistics:
    """Statistical analysis tools for simulation results."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._runs: List[SimulationRun] = []

    def run_monte_carlo(
        self,
        scenario_fn: Callable[[], Dict[str, float]],
        num_runs: int = 100,
        config: Optional[dict] = None,
    ) -> List[MonteCarloResult]:
        """Run a scenario function multiple times and compute statistics.

        scenario_fn returns a dict mapping metric names to values.
        """
        if config is None:
            config = {}
        self._runs = []
        all_samples: Dict[str, List[float]] = {}

        for run_id in range(num_runs):
            result = scenario_fn()
            run = SimulationRun(run_id=run_id, config=config, results=result)
            self._runs.append(run)
            for metric_name, value in result.items():
                if metric_name not in all_samples:
                    all_samples[metric_name] = []
                all_samples[metric_name].append(float(value))

        results = []
        for metric_name, samples in all_samples.items():
            mc_result = self._compute_stats(metric_name, samples)
            results.append(mc_result)

        return results

    def _compute_stats(self, metric_name: str, samples: List[float]) -> MonteCarloResult:
        if not samples:
            return MonteCarloResult(metric_name=metric_name, mean=0.0, std_dev=0.0, samples=0)
        mean = self._mean(samples)
        std_dev = self._std_dev(samples, mean)
        ci = self.compute_confidence_interval(samples, confidence=0.95)
        return MonteCarloResult(
            metric_name=metric_name,
            mean=mean,
            std_dev=std_dev,
            confidence_interval=ci,
            samples=len(samples),
        )

    def compute_confidence_interval(
        self,
        samples: List[float],
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Compute confidence interval using t-distribution approximation."""
        if len(samples) < 2:
            m = samples[0] if samples else 0.0
            return (m, m)

        n = len(samples)
        mean = self._mean(samples)
        std = self._std_dev(samples, mean)
        se = std / math.sqrt(n)

        # Approximate t-value for common confidence levels
        # Using a simplified lookup for normal approximation (large n)
        t_values = {
            0.90: 1.645,
            0.95: 1.960,
            0.99: 2.576,
            0.995: 2.807,
            0.999: 3.291,
        }
        # For small samples, use slightly larger t-value
        t_value = t_values.get(confidence, 1.960)
        if n < 30:
            # Simple correction for small samples
            t_value *= 1.0 + (30 - n) / 60.0

        margin = t_value * se
        return (mean - margin, mean + margin)

    def compute_percentiles(
        self, samples: List[float], percentiles: List[float]
    ) -> Dict[float, float]:
        """Compute percentiles from a list of samples."""
        if not samples:
            return {p: 0.0 for p in percentiles}
        sorted_samples = sorted(samples)
        n = len(sorted_samples)
        result = {}
        for p in percentiles:
            # Linear interpolation
            rank = (p / 100.0) * (n - 1)
            lower = int(math.floor(rank))
            upper = int(math.ceil(rank))
            if lower == upper or upper >= n:
                result[p] = sorted_samples[min(lower, n - 1)]
            else:
                fraction = rank - lower
                value = sorted_samples[lower] * (1.0 - fraction) + sorted_samples[upper] * fraction
                result[p] = value
        return result

    def compare_distributions(
        self,
        samples_a: List[float],
        samples_b: List[float],
    ) -> Tuple[float, float]:
        """Compare two distributions using a non-parametric test.

        Returns (statistic, p_value_approx).
        Uses a simplified Mann-Whitney U-like approach.
        """
        if not samples_a or not samples_b:
            return (0.0, 1.0)

        n_a = len(samples_a)
        n_b = len(samples_b)

        # Compute rank-sum test statistic
        combined = sorted(
            [(v, "a") for v in samples_a] + [(v, "b") for v in samples_b],
            key=lambda x: (x[0], 0 if x[1] == "a" else 1),
        )

        # Assign ranks
        ranks_a = 0.0
        i = 0
        while i < len(combined):
            # Find all tied values
            j = i
            while j < len(combined) and combined[j][0] == combined[i][0]:
                j += 1
            avg_rank = (i + j - 1) / 2.0 + 1.0
            for k in range(i, j):
                if combined[k][1] == "a":
                    ranks_a += avg_rank
            i = j

        u_a = ranks_a - n_a * (n_a + 1) / 2.0
        u_b = n_a * n_b - u_a
        u_stat = min(u_a, u_b)

        # Normalize statistic to [0, 1] range for comparison
        n = n_a + n_b
        mean_u = n_a * n_b / 2.0
        std_u = math.sqrt(n_a * n_b * (n + 1) / 12.0)
        if std_u > 0:
            z_stat = abs(u_stat - mean_u) / std_u
        else:
            z_stat = 0.0

        # Approximate p-value from normal distribution
        # P(|Z| > z) using complementary error function approximation
        p_value = self._normal_survival(z_stat)

        return (u_stat, p_value)

    def _normal_survival(self, z: float) -> float:
        """Approximate two-tailed P(|Z| > z) for standard normal."""
        if z <= 0:
            return 1.0
        # Abramowitz and Stegun approximation
        t = 1.0 / (1.0 + 0.2316419 * z)
        d = 0.3989422804014327  # 1/sqrt(2*pi)
        p = d * math.exp(-z * z / 2.0) * (
            t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.8212560 + t * 1.3302744))))
        )
        return 2.0 * p

    def compute_reliability(
        self,
        run_results: List[Dict[str, Any]],
        success_criteria: Callable[[Dict[str, Any]], bool],
    ) -> float:
        """Compute reliability as the fraction of runs meeting success criteria."""
        if not run_results:
            return 0.0
        successes = sum(1 for r in run_results if success_criteria(r))
        return successes / len(run_results)

    def generate_report(self, results: List[MonteCarloResult]) -> dict:
        """Generate a summary report from Monte Carlo results."""
        if not results:
            return {
                "total_metrics": 0,
                "metrics": [],
                "summary": "No results to report.",
            }

        metrics_data = []
        for r in results:
            metrics_data.append({
                "name": r.metric_name,
                "mean": r.mean,
                "std_dev": r.std_dev,
                "confidence_interval": {
                    "lower": r.confidence_interval[0],
                    "upper": r.confidence_interval[1],
                },
                "samples": r.samples,
                "margin_of_error": (
                    r.confidence_interval[1] - r.confidence_interval[0]
                ) / 2.0,
            })

        total_samples = sum(r.samples for r in results)
        avg_std = sum(r.std_dev for r in results) / len(results)

        return {
            "total_metrics": len(results),
            "total_samples": total_samples,
            "average_std_dev": avg_std,
            "metrics": metrics_data,
            "summary": (
                f"Analyzed {total_samples} samples across {len(results)} metrics. "
                f"Average standard deviation: {avg_std:.4f}."
            ),
        }

    def add_run(self, run: SimulationRun) -> None:
        self._runs.append(run)

    def clear_runs(self) -> None:
        self._runs = []

    @property
    def runs(self) -> List[SimulationRun]:
        return list(self._runs)

    @property
    def run_count(self) -> int:
        return len(self._runs)

    # ---- Internal helper methods ----

    @staticmethod
    def _mean(samples: List[float]) -> float:
        if not samples:
            return 0.0
        return sum(samples) / len(samples)

    @staticmethod
    def _std_dev(samples: List[float], mean: Optional[float] = None) -> float:
        if len(samples) < 2:
            return 0.0
        if mean is None:
            m = sum(samples) / len(samples)
        else:
            m = mean
        variance = sum((s - m) ** 2 for s in samples) / (len(samples) - 1)
        return math.sqrt(variance)
