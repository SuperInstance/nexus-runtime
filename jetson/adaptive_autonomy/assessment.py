"""Situation assessment for autonomous level recommendation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from jetson.adaptive_autonomy.levels import AutonomyLevel


@dataclass
class SituationAssessment:
    """Snapshot of the assessed situation."""
    complexity: float = 0.0       # 0-1
    risk: float = 0.0             # 0-1
    uncertainty: float = 0.0      # 0-1
    human_workload: float = 0.0   # 0-1
    system_confidence: float = 0.0  # 0-1
    recommended_level: AutonomyLevel = AutonomyLevel.MANUAL


class SituationAssessor:
    """Assesses the environment and recommends an autonomy level."""

    def __init__(self) -> None:
        # Thresholds can be adapted by the learner
        self._complexity_thresholds = {
            "low": 0.2,
            "medium": 0.5,
            "high": 0.8,
        }
        self._risk_thresholds = {
            "low": 0.15,
            "medium": 0.4,
            "high": 0.7,
        }

    # ---- public API ----

    def assess(
        self,
        current_level: AutonomyLevel,
        sensor_data: Dict[str, object],
        environment: Dict[str, object],
        task: Dict[str, object],
    ) -> SituationAssessment:
        """Produce a full SituationAssessment from the given inputs."""
        complexity = self.compute_complexity(environment, task)
        risk = self.compute_risk(
            environment.get("obstacles", []),
            environment.get("weather", "clear"),
            environment.get("traffic", []),
        )
        uncertainty = self.compute_uncertainty(
            sensor_data.get("sensor_quality", 1.0),
            sensor_data.get("model_confidence", 1.0),
        )
        human_workload = self.compute_human_workload(
            environment.get("alerts", []),
            environment.get("active_tasks", []),
        )
        system_confidence = self.compute_system_confidence(
            sensor_data.get("model_accuracy", 1.0),
            sensor_data.get("sensor_health", 1.0),
        )

        recommended = self.recommend_autonomy_level(
            SituationAssessment(
                complexity=complexity,
                risk=risk,
                uncertainty=uncertainty,
                human_workload=human_workload,
                system_confidence=system_confidence,
            )
        )
        return SituationAssessment(
            complexity=complexity,
            risk=risk,
            uncertainty=uncertainty,
            human_workload=human_workload,
            system_confidence=system_confidence,
            recommended_level=recommended,
        )

    def compute_complexity(
        self, environment: Dict[str, object], task: Dict[str, object]
    ) -> float:
        """Compute a 0-1 complexity score."""
        score = 0.0

        # Number of dynamic objects
        dyn = environment.get("dynamic_objects", 0)
        if isinstance(dyn, (int, float)):
            score += min(dyn / 20.0, 0.4)

        # Terrain roughness
        terrain = environment.get("terrain_roughness", 0.0)
        if isinstance(terrain, (int, float)):
            score += min(float(terrain) / 10.0, 0.3)

        # Task complexity
        task_complex = task.get("complexity", 0.0)
        if isinstance(task_complex, (int, float)):
            score += min(float(task_complex) / 10.0, 0.3)

        return round(max(0.0, min(1.0, score)), 4)

    def compute_risk(
        self,
        obstacles: object,
        weather: object,
        traffic: object,
    ) -> float:
        """Compute a 0-1 risk score."""
        score = 0.0

        # Obstacle proximity (list of distances)
        if isinstance(obstacles, list):
            for dist in obstacles:
                if isinstance(dist, (int, float)) and dist < 5.0:
                    score += (5.0 - min(dist, 5.0)) / 5.0 * 0.2
            score = min(score, 0.4)

        # Weather degradation
        weather_map = {
            "clear": 0.0,
            "cloudy": 0.1,
            "rain": 0.2,
            "heavy_rain": 0.3,
            "fog": 0.35,
            "snow": 0.4,
        }
        w = weather_map.get(str(weather).lower(), 0.15)
        score += w

        # Traffic density
        if isinstance(traffic, (int, float)):
            score += min(float(traffic) / 50.0, 0.2)
        elif isinstance(traffic, list):
            score += min(len(traffic) / 50.0, 0.2)

        return round(max(0.0, min(1.0, score)), 4)

    def compute_uncertainty(
        self, sensor_quality: object, model_confidence: object
    ) -> float:
        """Compute a 0-1 uncertainty score (higher = more uncertain)."""
        sq = 1.0
        mc = 1.0
        if isinstance(sensor_quality, (int, float)):
            sq = max(0.0, min(1.0, float(sensor_quality)))
        if isinstance(model_confidence, (int, float)):
            mc = max(0.0, min(1.0, float(model_confidence)))
        combined = (sq + mc) / 2.0
        return round(1.0 - combined, 4)

    def compute_human_workload(
        self, alerts: object, tasks: object
    ) -> float:
        """Compute a 0-1 human-workload score."""
        score = 0.0
        if isinstance(alerts, list):
            score += min(len(alerts) / 10.0, 0.5)
        elif isinstance(alerts, (int, float)):
            score += min(float(alerts) / 10.0, 0.5)
        if isinstance(tasks, list):
            score += min(len(tasks) / 10.0, 0.5)
        elif isinstance(tasks, (int, float)):
            score += min(float(tasks) / 10.0, 0.5)
        return round(max(0.0, min(1.0, score)), 4)

    def compute_system_confidence(
        self, model_accuracy: object, sensor_health: object
    ) -> float:
        """Compute a 0-1 system-confidence score."""
        ma = 1.0
        sh = 1.0
        if isinstance(model_accuracy, (int, float)):
            ma = max(0.0, min(1.0, float(model_accuracy)))
        if isinstance(sensor_health, (int, float)):
            sh = max(0.0, min(1.0, float(sensor_health)))
        return round((ma + sh) / 2.0, 4)

    def recommend_autonomy_level(
        self, assessment: SituationAssessment
    ) -> AutonomyLevel:
        """Map the assessment scores to an autonomy level."""
        # Weighted composite "difficulty" score
        difficulty = (
            assessment.risk * 0.35
            + assessment.complexity * 0.25
            + assessment.uncertainty * 0.20
            + (1.0 - assessment.system_confidence) * 0.10
            + (1.0 - assessment.human_workload) * 0.10
        )
        # difficulty is 0-1; higher = more difficult => lower autonomy

        if difficulty <= 0.15:
            return AutonomyLevel.AUTONOMOUS
        if difficulty <= 0.30:
            return AutonomyLevel.FULL_AUTO
        if difficulty <= 0.45:
            return AutonomyLevel.AUTO_WITH_SUPERVISION
        if difficulty <= 0.60:
            return AutonomyLevel.SEMI_AUTO
        if difficulty <= 0.75:
            return AutonomyLevel.ASSISTED
        return AutonomyLevel.MANUAL
