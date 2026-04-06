"""NEXUS Explainable AI — feature attribution, decision logging, model interpretation,
counterfactual analysis, and explanation report generation.

Pure Python, zero external dependencies.
"""

from .attribution import FeatureImportance, AttributionResult, FeatureAttributor
from .decision_log import DecisionRecord, DecisionLog
from .interpret import ModelInsight, ModelInterpreter
from .counterfactual import CounterfactualExample, CounterfactualGenerator
from .report import ExplanationSection, ExplanationReport

__all__ = [
    "FeatureImportance",
    "AttributionResult",
    "FeatureAttributor",
    "DecisionRecord",
    "DecisionLog",
    "ModelInsight",
    "ModelInterpreter",
    "CounterfactualExample",
    "CounterfactualGenerator",
    "ExplanationSection",
    "ExplanationReport",
]
