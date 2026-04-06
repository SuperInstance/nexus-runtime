"""Tests for perception quality assessment."""

import time
import pytest

from jetson.cooperative_perception.quality import (
    QualityMetrics,
    PerceptionQuality,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def quality():
    return PerceptionQuality()


# ── QualityMetrics dataclass tests ───────────────────────────────────────────

class TestQualityMetrics:

    def test_creation(self):
        qm = QualityMetrics(
            completeness=0.8, accuracy=0.9, freshness=1.0,
            consistency=0.7, coverage=0.85, overall_quality=0.85,
        )
        assert qm.completeness == 0.8
        assert qm.overall_quality == 0.85

    def test_defaults_none(self):
        # Dataclass requires all fields
        qm = QualityMetrics(0, 0, 0, 0, 0, 0)
        assert qm.completeness == 0


# ── assess_completeness tests ────────────────────────────────────────────────

class TestAssessCompleteness:

    def test_full_detection(self, quality):
        data = [{"confidence": 0.9}] * 5
        assert quality.assess_completeness(data, expected_coverage=5) == pytest.approx(0.9)

    def test_partial_detection(self, quality):
        data = [{"confidence": 0.8}] * 3
        result = quality.assess_completeness(data, expected_coverage=5)
        assert result == pytest.approx(0.6)

    def test_over_detection(self, quality):
        data = [{"confidence": 1.0}] * 10
        assert quality.assess_completeness(data, expected_coverage=5) == 1.0

    def test_no_detection(self, quality):
        result = quality.assess_completeness([], expected_coverage=5)
        assert result == 0.0

    def test_zero_expected(self, quality):
        assert quality.assess_completeness([], expected_coverage=0) == 1.0

    def test_negative_expected(self, quality):
        assert quality.assess_completeness([], expected_coverage=-1) == 1.0

    def test_varying_confidence(self, quality):
        data = [{"confidence": 0.5}, {"confidence": 0.9}]
        result = quality.assess_completeness(data, expected_coverage=2)
        assert result == pytest.approx(0.7)

    def test_empty_data_high_expected(self, quality):
        result = quality.assess_completeness([], expected_coverage=10)
        assert result == 0.0


# ── assess_accuracy tests ───────────────────────────────────────────────────

class TestAssessAccuracy:

    def test_perfect_match(self, quality):
        detections = [
            {"id": "o1", "type": "vessel", "position": (10.0, 20.0, 0.0)},
            {"id": "o2", "type": "buoy", "position": (30.0, 40.0, 0.0)},
        ]
        ground_truth = [
            {"id": "o1", "type": "vessel", "position": (10.0, 20.0, 0.0)},
            {"id": "o2", "type": "buoy", "position": (30.0, 40.0, 0.0)},
        ]
        result = quality.assess_accuracy(detections, ground_truth)
        assert result == pytest.approx(1.0)

    def test_no_detections(self, quality):
        gt = [{"id": "o1", "type": "vessel", "position": (0, 0, 0)}]
        assert quality.assess_accuracy([], gt) == 0.0

    def test_no_ground_truth(self, quality):
        dets = [{"id": "o1", "type": "vessel", "position": (0, 0, 0)}]
        assert quality.assess_accuracy(dets, []) == 0.0

    def test_both_empty(self, quality):
        assert quality.assess_accuracy([], []) == 1.0

    def test_type_mismatch(self, quality):
        detections = [
            {"id": "o1", "type": "buoy", "position": (10.0, 20.0, 0.0)},
        ]
        ground_truth = [
            {"id": "o1", "type": "vessel", "position": (10.0, 20.0, 0.0)},
        ]
        result = quality.assess_accuracy(detections, ground_truth)
        assert result < 1.0
        assert result > 0.0  # Position is still correct

    def test_position_error(self, quality):
        detections = [
            {"id": "o1", "type": "vessel", "position": (12.0, 20.0, 0.0)},
        ]
        ground_truth = [
            {"id": "o1", "type": "vessel", "position": (10.0, 20.0, 0.0)},
        ]
        result = quality.assess_accuracy(detections, ground_truth)
        assert result < 1.0

    def test_extra_detections(self, quality):
        detections = [
            {"id": "o1", "type": "vessel", "position": (10, 10, 0)},
            {"id": "o2", "type": "buoy", "position": (100, 100, 0)},  # Far away
        ]
        ground_truth = [
            {"id": "o1", "type": "vessel", "position": (10, 10, 0)},
        ]
        result = quality.assess_accuracy(detections, ground_truth)
        assert result < 1.0  # Precision penalty

    def test_range_0_to_1(self, quality):
        for _ in range(20):
            dets = [{"id": f"o{i}", "type": "vessel",
                      "position": (i * 10, 0, 0)} for i in range(5)]
            gt = [{"id": f"o{i}", "type": "vessel",
                   "position": (i * 10 + 1, 0, 0)} for i in range(5)]
            result = quality.assess_accuracy(dets, gt)
            assert 0.0 <= result <= 1.0


# ── assess_freshness tests ──────────────────────────────────────────────────

class TestAssessFreshness:

    def test_very_fresh(self, quality):
        assert quality.assess_freshness(time.time(), max_age=60.0) > 0.9

    def test_exactly_stale(self, quality):
        ts = time.time() - 30.0
        assert quality.assess_freshness(ts, max_age=30.0) == pytest.approx(0.0)

    def test_beyond_stale(self, quality):
        ts = time.time() - 100.0
        assert quality.assess_freshness(ts, max_age=30.0) == 0.0

    def test_half_life(self, quality):
        ts = time.time() - 15.0
        result = quality.assess_freshness(ts, max_age=30.0)
        assert result == pytest.approx(0.5)

    def test_future_timestamp(self, quality):
        result = quality.assess_freshness(time.time() + 10, max_age=30.0)
        assert result == 1.0

    def test_zero_max_age_fresh(self, quality):
        # max_age=0 means age >= max_age immediately true for any recent timestamp
        result = quality.assess_freshness(time.time(), max_age=0.0)
        assert result == 0.0

    def test_negative_max_age(self, quality):
        result = quality.assess_freshness(time.time(), max_age=-1)
        assert result == 0.0


# ── assess_consistency tests ────────────────────────────────────────────────

class TestAssessConsistency:

    def test_single_vessel(self, quality):
        result = quality.assess_consistency({
            "v_A": [{"id": "o1", "type": "vessel", "position": (10, 20, 0)}],
        })
        assert result == 1.0

    def test_empty_vessels(self, quality):
        result = quality.assess_consistency({})
        assert result == 1.0

    def test_perfect_agreement(self, quality):
        result = quality.assess_consistency({
            "v_A": [{"id": "o1", "type": "vessel", "position": (10, 20, 0)}],
            "v_B": [{"id": "o1", "type": "vessel", "position": (10.1, 20.1, 0)}],
        })
        assert result > 0.8

    def test_type_disagreement(self, quality):
        result = quality.assess_consistency({
            "v_A": [{"id": "o1", "type": "vessel", "position": (10, 20, 0)}],
            "v_B": [{"id": "o1", "type": "buoy", "position": (10, 20, 0)}],
        })
        assert result < 1.0

    def test_position_disagreement(self, quality):
        result = quality.assess_consistency({
            "v_A": [{"id": "o1", "type": "vessel", "position": (0, 0, 0)}],
            "v_B": [{"id": "o1", "type": "vessel", "position": (50, 0, 0)}],
        })
        assert result < 1.0

    def test_no_overlap(self, quality):
        result = quality.assess_consistency({
            "v_A": [{"id": "o1", "type": "vessel", "position": (0, 0, 0)}],
            "v_B": [{"id": "o2", "type": "buoy", "position": (100, 100, 0)}],
        })
        # No overlapping objects -> can't compare -> default 1.0
        assert result == 1.0

    def test_three_vessels_agree(self, quality):
        result = quality.assess_consistency({
            "v_A": [{"id": "o1", "type": "vessel", "position": (10, 20, 0)}],
            "v_B": [{"id": "o1", "type": "vessel", "position": (10.1, 20.1, 0)}],
            "v_C": [{"id": "o1", "type": "vessel", "position": (10.2, 20.2, 0)}],
        })
        assert result > 0.7


# ── assess_coverage tests ───────────────────────────────────────────────────

class TestAssessCoverage:

    def test_full_coverage(self, quality):
        fovs = [(0, 0, 100, 100)]
        aoi = (0, 0, 100, 100)
        result = quality.assess_coverage(fovs, aoi)
        assert result > 0.8

    def test_no_fovs(self, quality):
        result = quality.assess_coverage([], (0, 0, 100, 100))
        assert result == 0.0

    def test_no_aoi(self, quality):
        result = quality.assess_coverage([(0, 0, 100, 100)], (0, 0, 0, 0))
        assert result == 1.0

    def test_partial_coverage(self, quality):
        fovs = [(0, 0, 50, 100)]
        aoi = (0, 0, 100, 100)
        result = quality.assess_coverage(fovs, aoi)
        assert 0.3 < result < 0.7

    def test_disjoint_fov(self, quality):
        fovs = [(200, 200, 300, 300)]
        aoi = (0, 0, 100, 100)
        result = quality.assess_coverage(fovs, aoi)
        assert result == 0.0

    def test_multiple_overlapping_fovs(self, quality):
        fovs = [(0, 0, 80, 100), (20, 0, 100, 100)]
        aoi = (0, 0, 100, 100)
        result = quality.assess_coverage(fovs, aoi)
        assert result > 0.5


# ── compute_overall_quality tests ───────────────────────────────────────────

class TestComputeOverallQuality:

    def test_perfect_quality(self, quality):
        qm = QualityMetrics(1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
        result = quality.compute_overall_quality(qm)
        assert result == pytest.approx(1.0)

    def test_zero_quality(self, quality):
        qm = QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        result = quality.compute_overall_quality(qm)
        assert result == pytest.approx(0.0)

    def test_weights_sum_to_one(self, quality):
        # Verify the weights add up
        qm = QualityMetrics(1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
        assert quality.compute_overall_quality(qm) == pytest.approx(1.0)

    def test_mixed_quality(self, quality):
        qm = QualityMetrics(0.5, 0.8, 1.0, 0.6, 0.9, 0.0)
        result = quality.compute_overall_quality(qm)
        expected = 0.20*0.5 + 0.25*0.8 + 0.20*1.0 + 0.15*0.6 + 0.20*0.9
        assert result == pytest.approx(expected)

    def test_clamped(self, quality):
        qm = QualityMetrics(2.0, 2.0, 2.0, 2.0, 2.0, 0.0)
        result = quality.compute_overall_quality(qm)
        assert result <= 1.0


# ── compare_quality tests ───────────────────────────────────────────────────

class TestCompareQuality:

    def test_better(self, quality):
        qm_a = QualityMetrics(1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
        qm_b = QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert quality.compare_quality(qm_a, qm_b) == "better"

    def test_worse(self, quality):
        qm_a = QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        qm_b = QualityMetrics(1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
        assert quality.compare_quality(qm_a, qm_b) == "worse"

    def test_equal(self, quality):
        qm_a = QualityMetrics(0.5, 0.5, 0.5, 0.5, 0.5, 0.0)
        qm_b = QualityMetrics(0.5, 0.5, 0.5, 0.5, 0.5, 0.0)
        assert quality.compare_quality(qm_a, qm_b) == "equal"

    def test_within_threshold(self, quality):
        qm_a = QualityMetrics(0.501, 0.501, 0.501, 0.501, 0.501, 0.0)
        qm_b = QualityMetrics(0.500, 0.500, 0.500, 0.500, 0.500, 0.0)
        assert quality.compare_quality(qm_a, qm_b) == "equal"


# ── full_assessment tests ───────────────────────────────────────────────────

class TestFullAssessment:

    def test_full_with_all_params(self, quality):
        data = [{"confidence": 0.9, "timestamp": time.time()}] * 5
        gt = [{"id": "o1", "type": "vessel", "position": (10, 10, 0)}]
        dets = [{"id": "o1", "type": "vessel", "position": (10.2, 10.1, 0)}]
        mv = {
            "v_A": [{"id": "o1", "type": "vessel", "position": (10, 10, 0)}],
            "v_B": [{"id": "o1", "type": "vessel", "position": (10.1, 10.1, 0)}],
        }
        fovs = [(0, 0, 100, 100)]
        aoi = (0, 0, 100, 100)

        result = quality.full_assessment(
            perception_data=dets,
            expected_coverage=1,
            ground_truth=gt,
            max_age=60.0,
            multi_vessel=mv,
            fovs=fovs,
            aoi=aoi,
        )
        assert isinstance(result, QualityMetrics)
        assert 0.0 <= result.overall_quality <= 1.0

    def test_full_minimal(self, quality):
        result = quality.full_assessment(
            perception_data=[], expected_coverage=0,
        )
        assert isinstance(result, QualityMetrics)
        assert result.completeness == 1.0
        assert result.accuracy == 1.0  # Unknown defaults to 1

    def test_full_no_timestamps(self, quality):
        result = quality.full_assessment(
            perception_data=[{"confidence": 0.9}], expected_coverage=1,
        )
        assert isinstance(result, QualityMetrics)

    def test_overall_quality_computed(self, quality):
        result = quality.full_assessment(
            perception_data=[{"confidence": 0.5, "timestamp": time.time()}],
            expected_coverage=1,
        )
        assert result.overall_quality > 0.0
