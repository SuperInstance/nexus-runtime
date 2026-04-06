"""NEXUS Performance Engineering Module — Phase 6 Round 10.

Provides code profiling, benchmarking, optimization analysis,
caching strategies, and resource monitoring for Jetson platforms.
"""

from jetson.performance.profiler import ProfileEntry, ProfileReport, Profiler
from jetson.performance.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
)
from jetson.performance.optimization import (
    OptimizationSuggestion,
    CodeAnalyzer,
)
from jetson.performance.caching import CacheEntry, CachePolicy, CacheManager
from jetson.performance.resource_monitor import (
    ResourceSnapshot,
    ResourceAlert,
    ResourceMonitor,
)

__all__ = [
    "ProfileEntry",
    "ProfileReport",
    "Profiler",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkRunner",
    "OptimizationSuggestion",
    "CodeAnalyzer",
    "CacheEntry",
    "CachePolicy",
    "CacheManager",
    "ResourceSnapshot",
    "ResourceAlert",
    "ResourceMonitor",
]
