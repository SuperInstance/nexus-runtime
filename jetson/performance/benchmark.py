"""Benchmarking framework.

Provides statistically robust benchmark execution, comparison,
regression detection, and reporting.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark."""

    name: str = "unnamed"
    iterations: int = 100
    warmup: int = 5
    timeout: float = 30.0
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    mean_time: float
    std_dev: float
    min_time: float
    max_time: float
    iterations: int
    ops_per_sec: float

    @property
    def median_time(self) -> float:
        return self.mean_time

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mean_time": self.mean_time,
            "std_dev": self.std_dev,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "iterations": self.iterations,
            "ops_per_sec": self.ops_per_sec,
        }


class BenchmarkRunner:
    """Execute benchmarks with warmup, timeout, and statistical analysis."""

    def run_benchmark(self, fn: Callable[..., Any], config: Optional[BenchmarkConfig] = None) -> BenchmarkResult:
        """Run *fn* according to *config* and return a BenchmarkResult."""
        if config is None:
            config = BenchmarkConfig()
        # Warmup
        for _ in range(config.warmup):
            fn()

        times: List[float] = []
        deadline = time.perf_counter() + config.timeout
        for _ in range(config.iterations):
            start = time.perf_counter()
            fn()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            if time.perf_counter() > deadline:
                break

        n = len(times)
        mean_t = sum(times) / n
        if n > 1:
            variance = sum((t - mean_t) ** 2 for t in times) / (n - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0
        ops = (1.0 / mean_t) if mean_t > 0 else float("inf")
        return BenchmarkResult(
            name=config.name,
            mean_time=mean_t,
            std_dev=std,
            min_time=min(times),
            max_time=max(times),
            iterations=n,
            ops_per_sec=ops,
        )

    def compare_implementations(
        self,
        implementations: List[Callable[..., Any]],
        config: Optional[BenchmarkConfig] = None,
    ) -> List[BenchmarkResult]:
        """Benchmark multiple implementations and rank by mean_time (fastest first)."""
        if config is None:
            config = BenchmarkConfig()
        results: List[BenchmarkResult] = []
        for idx, fn in enumerate(implementations):
            cfg = BenchmarkConfig(
                name=f"{config.name}_impl_{idx}",
                iterations=config.iterations,
                warmup=config.warmup,
                timeout=config.timeout,
                parameters=dict(config.parameters),
            )
            results.append(self.run_benchmark(fn, cfg))
        results.sort(key=lambda r: r.mean_time)
        return results

    def run_regression_suite(
        self,
        suite: Dict[str, Callable[..., Any]],
        baseline: Optional[Dict[str, BenchmarkResult]] = None,
    ) -> Dict[str, Any]:
        """Run a named suite and optionally compare against a baseline."""
        results: Dict[str, BenchmarkResult] = {}
        for name, fn in suite.items():
            cfg = BenchmarkConfig(name=name, iterations=20, warmup=2, timeout=10.0)
            results[name] = self.run_benchmark(fn, cfg)

        report: Dict[str, Any] = {"results": {n: r.as_dict() for n, r in results.items()}, "regressions": []}
        if baseline:
            for name, cur in results.items():
                if name in baseline:
                    base = baseline[name]
                    if cur.mean_time > base.mean_time * 1.2:
                        report["regressions"].append({
                            "name": name,
                            "baseline_mean": base.mean_time,
                            "current_mean": cur.mean_time,
                            "ratio": cur.mean_time / base.mean_time,
                        })
        return report

    @staticmethod
    def compute_statistical_significance(
        result_a: BenchmarkResult, result_b: BenchmarkResult,
    ) -> bool:
        """Return True if the two results are likely from different distributions.

        Uses a simple z-test on the means with pooled std-dev.
        """
        n_a = result_a.iterations
        n_b = result_b.iterations
        if n_a < 2 or n_b < 2:
            return abs(result_a.mean_time - result_b.mean_time) > 0
        pooled_std = math.sqrt(
            ((n_a - 1) * result_a.std_dev ** 2 + (n_b - 1) * result_b.std_dev ** 2)
            / (n_a + n_b - 2)
        )
        se = pooled_std * math.sqrt(1.0 / n_a + 1.0 / n_b)
        if se == 0:
            return result_a.mean_time != result_b.mean_time
        z = abs(result_a.mean_time - result_b.mean_time) / se
        return z > 1.96  # ~95% confidence

    @staticmethod
    def generate_benchmark_report(results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate a summary report from multiple results."""
        if not results:
            return {"total_benchmarks": 0, "benchmarks": []}
        fastest = min(results, key=lambda r: r.mean_time)
        slowest = max(results, key=lambda r: r.mean_time)
        return {
            "total_benchmarks": len(results),
            "fastest": fastest.name,
            "slowest": slowest.name,
            "benchmarks": [r.as_dict() for r in results],
        }
