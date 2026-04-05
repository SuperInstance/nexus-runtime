"""NEXUS A/B Testing Framework for reflex comparison.

Statistical engine, VM simulation, and git-branch integration for
comparing different reflex bytecode variants.
"""

from .experiment import (
    ABTestResult,
    ABTestSuite,
    ExperimentVariant,
    MetricRecord,
    MetricType,
    PowerAnalysisResult,
)
from .git_integration import BranchIntegration
from .statistical_engine import (
    BonferroniResult,
    StatisticalEngine,
    TestMethod,
    TestResult,
)
from .vm_simulator import ReflexComparator, SimulationIteration

__all__ = [
    "ABTestSuite",
    "ABTestResult",
    "ExperimentVariant",
    "MetricRecord",
    "MetricType",
    "PowerAnalysisResult",
    "StatisticalEngine",
    "TestMethod",
    "TestResult",
    "BonferroniResult",
    "ReflexComparator",
    "SimulationIteration",
    "BranchIntegration",
]
