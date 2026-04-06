"""Explanation report generation for NEXUS explainable AI.

Assembles structured and text reports from attribution results,
model insights, and actionable recommendations. Pure Python, no deps.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .attribution import AttributionResult, FeatureImportance


@dataclass
class ExplanationSection:
    """A section in an explanation report."""
    title: str
    content: str
    importance_level: str = "normal"  # "critical", "high", "normal", "low"

    def __post_init__(self):
        if self.importance_level not in ("critical", "high", "normal", "low"):
            self.importance_level = "normal"


class ExplanationReport:
    """Generate comprehensive explanation reports."""

    def __init__(self, title: str = "NEXUS Explanation Report", model_name: str = "unknown"):
        self.title = title
        self.model_name = model_name
        self.sections: List[ExplanationSection] = []
        self.visualizations: List[Dict[str, Any]] = []
        self.created_at = time.time()

    def add_section(self, section: ExplanationSection) -> None:
        """Add a section to the report."""
        self.sections.append(section)

    def generate_text_report(self) -> str:
        """Generate a human-readable text report."""
        lines = [
            f"{'=' * 60}",
            f"{self.title}",
            f"Model: {self.model_name}",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.created_at))}",
            f"{'=' * 60}",
            "",
        ]

        for section in self.sections:
            marker = {"critical": "!!!", "high": "!!", "normal": " ", "low": "~"}.get(
                section.importance_level, " "
            )
            lines.append(f"{marker} [{section.importance_level.upper()}] {section.title}")
            lines.append("-" * len(section.title) + "-" * 20)
            lines.append(section.content)
            lines.append("")

        if self.visualizations:
            lines.append("Visualizations:")
            for viz in self.visualizations:
                lines.append(f"  - {viz.get('type', 'unknown')}: {viz.get('description', '')}")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    def generate_structured_report(self) -> Dict[str, Any]:
        """Generate a structured (dict) report."""
        return {
            "title": self.title,
            "model_name": self.model_name,
            "created_at": self.created_at,
            "sections": [
                {
                    "title": s.title,
                    "content": s.content,
                    "importance_level": s.importance_level,
                }
                for s in self.sections
            ],
            "visualizations": self.visualizations,
            "summary": self.generate_summary(),
        }

    def add_visualization(
        self,
        data: Any,
        chart_type: str,
        description: str = "",
    ) -> None:
        """Add a visualization entry to the report."""
        self.visualizations.append({
            "type": chart_type,
            "data": data,
            "description": description,
        })

    def generate_summary(self) -> str:
        """Generate a brief summary of the report."""
        if not self.sections:
            return "No sections in report."
        critical = sum(1 for s in self.sections if s.importance_level == "critical")
        high = sum(1 for s in self.sections if s.importance_level == "high")
        normal = sum(1 for s in self.sections if s.importance_level == "normal")
        low = sum(1 for s in self.sections if s.importance_level == "low")

        parts = [f"Report contains {len(self.sections)} sections."]
        if critical > 0:
            parts.append(f"{critical} critical finding(s).")
        if high > 0:
            parts.append(f"{high} high-priority finding(s).")
        parts.append(f"Model: {self.model_name}.")

        # Extract key findings from critical/high sections
        key_sections = [s for s in self.sections if s.importance_level in ("critical", "high")]
        if key_sections:
            parts.append(f"Key findings: {', '.join(s.title for s in key_sections[:3])}.")

        return " ".join(parts)

    def generate_technical_details(
        self,
        attributions: List[FeatureImportance],
        model_info: Dict[str, Any],
    ) -> str:
        """Generate technical details section from attributions and model info."""
        lines = []
        lines.append("Technical Details")
        lines.append("-" * 40)

        lines.append(f"\nModel: {model_info.get('name', 'unknown')}")
        lines.append(f"Version: {model_info.get('version', 'unknown')}")
        lines.append(f"Type: {model_info.get('type', 'unknown')}")
        lines.append(f"Accuracy: {model_info.get('accuracy', 'N/A')}")

        if attributions:
            lines.append("\nFeature Attributions:")
            sorted_attrs = sorted(attributions, key=lambda a: a.importance_score, reverse=True)
            for i, attr in enumerate(sorted_attrs, 1):
                lines.append(
                    f"  {i}. {attr.feature_name}: "
                    f"importance={attr.importance_score:.4f}, "
                    f"direction={attr.direction}, "
                    f"contribution={attr.contribution:.4f}"
                )

        return "\n".join(lines)

    def generate_actionable_recommendations(
        self,
        analysis: Dict[str, Any],
    ) -> List[str]:
        """Generate actionable recommendations based on analysis results."""
        recommendations = []

        # Check for bias
        biased = analysis.get("biased_features", [])
        if biased:
            recommendations.append(
                f"Address bias in features: {', '.join(biased)}. "
                f"Consider retraining with balanced data or applying fairness constraints."
            )

        # Check for low confidence
        avg_conf = analysis.get("avg_confidence", 1.0)
        if avg_conf < 0.7:
            recommendations.append(
                f"Average prediction confidence ({avg_conf:.2f}) is below 0.7. "
                f"Consider gathering more training data in uncertain regions."
            )

        # Check for high complexity
        complexity = analysis.get("complexity_score", 0.5)
        if complexity > 0.8:
            recommendations.append(
                f"Model complexity ({complexity:.2f}) is high. "
                f"Consider simplifying the model or using regularization."
            )

        # Check for anomalies
        anomalies = analysis.get("anomaly_count", 0)
        if anomalies > 0:
            recommendations.append(
                f"{anomalies} anomalous decisions detected. "
                f"Review decision logs and consider adding additional safety constraints."
            )

        # Check fairness
        fairness = analysis.get("fairness_scores", {})
        low_fairness = [f for f, s in fairness.items() if s < 0.8]
        if low_fairness:
            recommendations.append(
                f"Low fairness scores for: {', '.join(low_fairness)}. "
                f"Apply adversarial debiasing or reweight training samples."
            )

        # Check for data quality issues
        data_quality = analysis.get("data_quality_score", 1.0)
        if data_quality < 0.8:
            recommendations.append(
                f"Data quality score ({data_quality:.2f}) is below threshold. "
                f"Clean and validate input data before training."
            )

        if not recommendations:
            recommendations.append("No immediate actions required. Model performance is within acceptable bounds.")

        return recommendations

    def export_report(self, format: str = "json") -> str:
        """Export the report in specified format."""
        if format == "json":
            return json.dumps(self.generate_structured_report(), indent=2, default=str)
        elif format == "text":
            return self.generate_text_report()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    @property
    def section_count(self) -> int:
        """Number of sections in the report."""
        return len(self.sections)

    @property
    def has_critical_findings(self) -> bool:
        """Whether the report contains critical findings."""
        return any(s.importance_level == "critical" for s in self.sections)
