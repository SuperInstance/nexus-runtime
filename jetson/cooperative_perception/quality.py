"""Perception data quality assessment.

Assesses completeness, accuracy, freshness, consistency, and coverage
of cooperative perception data, and computes an overall quality score.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any


@dataclass
class QualityMetrics:
    """Quality assessment metrics for perception data."""
    completeness: float
    accuracy: float
    freshness: float
    consistency: float
    coverage: float
    overall_quality: float


class PerceptionQuality:
    """Assesses quality of cooperative perception data."""

    def assess_completeness(
        self,
        perception_data: List[dict],
        expected_coverage: float,
    ) -> float:
        """Assess how completely the perception data covers expected detections.

        Args:
            perception_data: list of detection dicts, each with optional 'confidence'.
            expected_coverage: expected number of objects in the scene.

        Returns:
            Completeness score from 0.0 to 1.0.
        """
        if expected_coverage <= 0:
            return 1.0

        detected = len(perception_data)
        ratio = detected / expected_coverage
        if ratio >= 1.0:
            # Check average confidence of detected objects
            if perception_data:
                avg_conf = sum(
                    d.get("confidence", 0.5) for d in perception_data
                ) / len(perception_data)
                return avg_conf
            return 1.0

        return max(0.0, ratio)

    def assess_accuracy(
        self,
        detections: List[dict],
        ground_truth: List[dict],
    ) -> float:
        """Assess detection accuracy against ground truth.

        Matches detections to ground truth by proximity and measures
        type agreement and positional error.

        Args:
            detections: list of detection dicts with 'id', 'type', 'position'.
            ground_truth: list of ground truth dicts with 'id', 'type', 'position'.

        Returns:
            Accuracy score from 0.0 to 1.0.
        """
        if not ground_truth:
            return 1.0 if not detections else 0.0

        if not detections:
            return 0.0

        matched = 0
        total_pos_error = 0.0
        type_correct = 0
        match_threshold = 10.0  # meters

        used_gt: set = set()
        for det in detections:
            det_pos = det.get("position", (0.0, 0.0, 0.0))
            best_gt_idx = -1
            best_dist = float("inf")

            for i, gt in enumerate(ground_truth):
                if i in used_gt:
                    continue
                gt_pos = gt.get("position", (0.0, 0.0, 0.0))
                dist = self._distance3d(det_pos, gt_pos)
                if dist < match_threshold and dist < best_dist:
                    best_dist = dist
                    best_gt_idx = i

            if best_gt_idx >= 0:
                matched += 1
                used_gt.add(best_gt_idx)
                total_pos_error += best_dist
                if det.get("type") == ground_truth[best_gt_idx].get("type"):
                    type_correct += 1

        # Precision component
        precision = matched / len(detections)
        # Recall component
        recall = matched / len(ground_truth)

        # F1 score
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)

        # Type accuracy among matched
        type_acc = type_correct / matched if matched > 0 else 0.0

        # Position accuracy (inverse normalized error)
        pos_acc = max(0.0, 1.0 - (total_pos_error / (matched * match_threshold))) if matched > 0 else 0.0

        # Weighted combination
        accuracy = 0.4 * f1 + 0.3 * type_acc + 0.3 * pos_acc
        return max(0.0, min(1.0, accuracy))

    def assess_freshness(self, timestamp: float, max_age: float) -> float:
        """Assess data freshness based on age.

        Args:
            timestamp: Unix timestamp of the data.
            max_age: maximum acceptable age in seconds.

        Returns:
            Freshness score from 0.0 to 1.0.
        """
        now = time.time()
        age = now - timestamp
        if age < 0:
            return 1.0  # Future timestamp is fine (clock skew)
        if age >= max_age:
            return 0.0
        return 1.0 - (age / max_age)

    def assess_consistency(
        self,
        multi_vessel_observations: Dict[str, List[dict]],
    ) -> float:
        """Assess consistency of observations across multiple vessels.

        Checks type agreement and position agreement for commonly observed objects.

        Args:
            multi_vessel_observations: mapping vessel_id -> list of observation dicts
                with 'id', 'type', 'position'.

        Returns:
            Consistency score from 0.0 to 1.0.
        """
        vessels = list(multi_vessel_observations.keys())
        if len(vessels) < 2:
            return 1.0  # Can't measure consistency with one source

        # Group observations by object id across all vessels
        obj_groups: Dict[str, List[Tuple[str, dict]]] = {}
        for vid, obs_list in multi_vessel_observations.items():
            for obs in obs_list:
                oid = obs.get("id", "unknown")
                obj_groups.setdefault(oid, []).append((vid, obs))

        # Only evaluate objects seen by 2+ vessels
        multi_seen = {
            oid: group for oid, group in obj_groups.items()
            if len(set(v for v, _ in group)) >= 2
        }

        if not multi_seen:
            return 1.0  # No overlapping observations to compare

        type_scores = []
        pos_scores = []

        for oid, group in multi_seen.items():
            types = [obs.get("type", "unknown") for _, obs in group]
            type_agreement = max(
                types.count(t) for t in set(types)
            ) / len(types)
            type_scores.append(type_agreement)

            positions = [obs.get("position", (0, 0, 0)) for _, obs in group]
            if len(positions) >= 2:
                pos_spread = self._max_pairwise_distance(positions)
                # Normalize: spread < 5m is excellent, > 20m is poor
                pos_score = max(0.0, 1.0 - pos_spread / 20.0)
                pos_scores.append(pos_score)

        avg_type = sum(type_scores) / len(type_scores) if type_scores else 1.0
        avg_pos = sum(pos_scores) / len(pos_scores) if pos_scores else 1.0

        return 0.5 * avg_type + 0.5 * avg_pos

    def assess_coverage(
        self,
        fields_of_view: List[Tuple[float, float, float, float]],
        area_of_interest: Tuple[float, float, float, float],
    ) -> float:
        """Assess how well sensor fields of view cover the area of interest.

        Args:
            fields_of_view: list of (x1, y1, x2, y2) bounding boxes for each sensor.
            area_of_interest: (x1, y1, x2, y2) bounding box of interest area.

        Returns:
            Coverage score from 0.0 to 1.0.
        """
        if not area_of_interest or not fields_of_view:
            return 0.0

        # Compute area of interest
        aoi_area = self._bbox_area(area_of_interest)
        if aoi_area <= 0:
            return 1.0  # Degenerate area

        # Compute union of all FOVs clipped to area of interest
        # Use sampling approach for simplicity
        samples = 100
        covered = 0
        x_min, y_min, x_max, y_max = area_of_interest

        for _ in range(samples):
            sx = x_min + (x_max - x_min) * (_ / samples)
            sy = y_min + (y_max - y_min) * ((_ * 7 + 3) % samples / samples)
            for fov in fields_of_view:
                fx1, fy1, fx2, fy2 = (
                    min(fov[0], fov[2]),
                    min(fov[1], fov[3]),
                    max(fov[0], fov[2]),
                    max(fov[1], fov[3]),
                )
                if fx1 <= sx <= fx2 and fy1 <= sy <= fy2:
                    covered += 1
                    break

        return covered / samples

    def compute_overall_quality(self, metrics: QualityMetrics) -> float:
        """Compute overall quality score from individual metrics.

        Uses weighted average: completeness (0.2), accuracy (0.25),
        freshness (0.2), consistency (0.15), coverage (0.2).

        Args:
            metrics: QualityMetrics instance.

        Returns:
            Overall quality score from 0.0 to 1.0.
        """
        score = (
            0.20 * metrics.completeness
            + 0.25 * metrics.accuracy
            + 0.20 * metrics.freshness
            + 0.15 * metrics.consistency
            + 0.20 * metrics.coverage
        )
        return max(0.0, min(1.0, score))

    def compare_quality(
        self,
        quality_a: QualityMetrics,
        quality_b: QualityMetrics,
    ) -> str:
        """Compare two quality metric sets.

        Args:
            quality_a: first QualityMetrics.
            quality_b: second QualityMetrics.

        Returns:
            'better' if a is better than b, 'worse' if a is worse,
            'equal' if they are equivalent.
        """
        overall_a = self.compute_overall_quality(quality_a)
        overall_b = self.compute_overall_quality(quality_b)

        threshold = 0.01
        if overall_a > overall_b + threshold:
            return "better"
        elif overall_a < overall_b - threshold:
            return "worse"
        else:
            return "equal"

    def full_assessment(
        self,
        perception_data: List[dict],
        expected_coverage: float,
        ground_truth: Optional[List[dict]] = None,
        max_age: float = 30.0,
        multi_vessel: Optional[Dict[str, List[dict]]] = None,
        fovs: Optional[List[Tuple[float, float, float, float]]] = None,
        aoi: Optional[Tuple[float, float, float, float]] = None,
    ) -> QualityMetrics:
        """Perform a full quality assessment with all metrics.

        Args:
            perception_data: list of detection dicts.
            expected_coverage: expected number of objects.
            ground_truth: optional ground truth for accuracy.
            max_age: maximum acceptable age for freshness.
            multi_vessel: optional multi-vessel observations for consistency.
            fovs: optional fields of view for coverage.
            aoi: optional area of interest for coverage.

        Returns:
            QualityMetrics with all scores populated.
        """
        completeness = self.assess_completeness(perception_data, expected_coverage)

        if ground_truth is not None:
            accuracy = self.assess_accuracy(perception_data, ground_truth)
        else:
            accuracy = 1.0  # Unknown

        # Freshness from latest detection timestamp
        if perception_data:
            latest_ts = max(d.get("timestamp", time.time()) for d in perception_data)
            freshness = self.assess_freshness(latest_ts, max_age)
        else:
            freshness = 0.0

        if multi_vessel is not None:
            consistency = self.assess_consistency(multi_vessel)
        else:
            consistency = 1.0  # Unknown

        if fovs is not None and aoi is not None:
            coverage = self.assess_coverage(fovs, aoi)
        else:
            coverage = 1.0  # Unknown

        metrics = QualityMetrics(
            completeness=completeness,
            accuracy=accuracy,
            freshness=freshness,
            consistency=consistency,
            coverage=coverage,
            overall_quality=0.0,  # computed below
        )
        metrics.overall_quality = self.compute_overall_quality(metrics)
        return metrics

    def _distance3d(
        self,
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
    ) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _max_pairwise_distance(
        self, positions: List[Tuple[float, float, float]]
    ) -> float:
        if len(positions) < 2:
            return 0.0
        max_d = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                d = self._distance3d(positions[i], positions[j])
                if d > max_d:
                    max_d = d
        return max_d

    def _bbox_area(self, bbox: Tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = bbox
        return abs((x2 - x1) * (y2 - y1))
