"""Tests for jetson.adaptive_autonomy.assessment."""

import pytest

from jetson.adaptive_autonomy.levels import AutonomyLevel
from jetson.adaptive_autonomy.assessment import (
    SituationAssessment,
    SituationAssessor,
)


# ── SituationAssessment ───────────────────────────────────────────

class TestSituationAssessment:
    def test_defaults(self):
        sa = SituationAssessment()
        assert sa.complexity == 0.0
        assert sa.risk == 0.0
        assert sa.uncertainty == 0.0
        assert sa.human_workload == 0.0
        assert sa.system_confidence == 0.0
        assert sa.recommended_level == AutonomyLevel.MANUAL

    def test_custom_values(self):
        sa = SituationAssessment(
            complexity=0.5, risk=0.3, uncertainty=0.1,
            human_workload=0.8, system_confidence=0.9,
            recommended_level=AutonomyLevel.FULL_AUTO,
        )
        assert sa.complexity == 0.5
        assert sa.recommended_level == AutonomyLevel.FULL_AUTO


# ── SituationAssessor ─────────────────────────────────────────────

class TestSituationAssessor:

    @pytest.fixture()
    def assessor(self):
        return SituationAssessor()

    # assess
    def test_assess_returns_situation_assessment(self, assessor):
        result = assessor.assess(
            AutonomyLevel.MANUAL,
            {"sensor_quality": 1.0, "model_confidence": 1.0,
             "model_accuracy": 1.0, "sensor_health": 1.0},
            {"obstacles": [], "weather": "clear", "traffic": [],
             "alerts": [], "active_tasks": [], "dynamic_objects": 0,
             "terrain_roughness": 0.0},
            {"complexity": 0.0},
        )
        assert isinstance(result, SituationAssessment)

    def test_assess_all_fields_populated(self, assessor):
        result = assessor.assess(
            AutonomyLevel.MANUAL,
            {"sensor_quality": 1.0, "model_confidence": 1.0,
             "model_accuracy": 1.0, "sensor_health": 1.0},
            {},
            {},
        )
        assert 0.0 <= result.complexity <= 1.0
        assert 0.0 <= result.risk <= 1.0
        assert 0.0 <= result.uncertainty <= 1.0
        assert 0.0 <= result.human_workload <= 1.0
        assert 0.0 <= result.system_confidence <= 1.0

    def test_assess_safe_scenario_recommends_high(self, assessor):
        result = assessor.assess(
            AutonomyLevel.MANUAL,
            {"sensor_quality": 1.0, "model_confidence": 1.0,
             "model_accuracy": 1.0, "sensor_health": 1.0},
            {"obstacles": [], "weather": "clear", "traffic": [],
             "alerts": [], "active_tasks": []},
            {"complexity": 0.0},
        )
        assert result.recommended_level.value >= AutonomyLevel.FULL_AUTO.value

    # compute_complexity
    def test_complexity_empty(self, assessor):
        c = assessor.compute_complexity({}, {})
        assert c == 0.0

    def test_complexity_dynamic_objects(self, assessor):
        c = assessor.compute_complexity({"dynamic_objects": 20}, {})
        assert c >= 0.35

    def test_complexity_terrain(self, assessor):
        c = assessor.compute_complexity({"terrain_roughness": 10.0}, {})
        assert c >= 0.25

    def test_complexity_task(self, assessor):
        c = assessor.compute_complexity({}, {"complexity": 10.0})
        assert c >= 0.25

    def test_complexity_capped_at_1(self, assessor):
        c = assessor.compute_complexity(
            {"dynamic_objects": 999, "terrain_roughness": 999},
            {"complexity": 999},
        )
        assert c <= 1.0

    def test_complexity_nonnumeric_ignored(self, assessor):
        c = assessor.compute_complexity(
            {"dynamic_objects": "invalid"},
            {},
        )
        assert c == 0.0

    # compute_risk
    def test_risk_clear_weather_no_obstacles(self, assessor):
        r = assessor.compute_risk([], "clear", 0)
        assert r == 0.0

    def test_risk_close_obstacles(self, assessor):
        r = assessor.compute_risk([1.0, 0.5], "clear", 0)
        assert r > 0.0

    def test_risk_bad_weather(self, assessor):
        r = assessor.compute_risk([], "heavy_rain", 0)
        assert r > 0.0

    def test_risk_snow(self, assessor):
        r = assessor.compute_risk([], "snow", 0)
        assert r == 0.4

    def test_risk_fog(self, assessor):
        r = assessor.compute_risk([], "fog", 0)
        assert r == 0.35

    def test_risk_traffic_int(self, assessor):
        r = assessor.compute_risk([], "clear", 50)
        assert r >= 0.2

    def test_risk_traffic_list(self, assessor):
        r = assessor.compute_risk([], "clear", [1, 2, 3])
        assert r >= 0.06

    def test_risk_capped(self, assessor):
        r = assessor.compute_risk([0.0, 0.0, 0.0], "snow", 999)
        assert r <= 1.0

    def test_risk_far_obstacles_ignored(self, assessor):
        r = assessor.compute_risk([100.0], "clear", 0)
        assert r == 0.0

    # compute_uncertainty
    def test_uncertainty_perfect_sensors(self, assessor):
        u = assessor.compute_uncertainty(1.0, 1.0)
        assert u == 0.0

    def test_uncertainty_bad_sensors(self, assessor):
        u = assessor.compute_uncertainty(0.0, 0.0)
        assert u == 1.0

    def test_uncertainty_partial(self, assessor):
        u = assessor.compute_uncertainty(0.5, 0.5)
        assert u == 0.5

    def test_uncertainty_nonnumeric_defaults_to_1(self, assessor):
        u = assessor.compute_uncertainty("bad", "bad")
        assert u == 0.0

    # compute_human_workload
    def test_workload_empty(self, assessor):
        w = assessor.compute_human_workload([], [])
        assert w == 0.0

    def test_workload_alerts(self, assessor):
        w = assessor.compute_human_workload(["a"] * 10, [])
        assert w == 0.5

    def test_workload_tasks(self, assessor):
        w = assessor.compute_human_workload([], ["t"] * 10)
        assert w == 0.5

    def test_workload_combined(self, assessor):
        w = assessor.compute_human_workload(["a"] * 5, ["t"] * 5)
        assert w == 1.0

    def test_workload_capped(self, assessor):
        w = assessor.compute_human_workload(["a"] * 99, ["t"] * 99)
        assert w <= 1.0

    def test_workload_int_input(self, assessor):
        w = assessor.compute_human_workload(5, 3)
        assert w > 0.0

    # compute_system_confidence
    def test_confidence_perfect(self, assessor):
        c = assessor.compute_system_confidence(1.0, 1.0)
        assert c == 1.0

    def test_confidence_low(self, assessor):
        c = assessor.compute_system_confidence(0.0, 0.0)
        assert c == 0.0

    def test_confidence_average(self, assessor):
        c = assessor.compute_system_confidence(1.0, 0.0)
        assert c == 0.5

    def test_confidence_clamped(self, assessor):
        c = assessor.compute_system_confidence(2.0, -1.0)
        assert c == 0.5  # (1.0 + 0.0) / 2

    # recommend_autonomy_level
    def test_recommend_low_difficulty(self, assessor):
        sa = SituationAssessment(
            risk=0.0, complexity=0.0, uncertainty=0.0,
            system_confidence=1.0, human_workload=0.0,
        )
        assert assessor.recommend_autonomy_level(sa) == AutonomyLevel.AUTONOMOUS

    def test_recommend_high_difficulty(self, assessor):
        sa = SituationAssessment(
            risk=1.0, complexity=1.0, uncertainty=1.0,
            system_confidence=0.0, human_workload=0.0,
        )
        assert assessor.recommend_autonomy_level(sa) == AutonomyLevel.MANUAL

    def test_recommend_medium_difficulty(self, assessor):
        sa = SituationAssessment(
            risk=0.4, complexity=0.4, uncertainty=0.3,
            system_confidence=0.7, human_workload=0.5,
        )
        level = assessor.recommend_autonomy_level(sa)
        assert AutonomyLevel.SEMI_AUTO <= level <= AutonomyLevel.AUTO_WITH_SUPERVISION
