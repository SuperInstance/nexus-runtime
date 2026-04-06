"""Tests for jetson.adaptive_autonomy.learning."""

import pytest

from jetson.adaptive_autonomy.levels import AutonomyLevel
from jetson.adaptive_autonomy.learning import (
    TransitionExperience,
    AutonomyLearner,
)


# ── TransitionExperience ──────────────────────────────────────────

class TestTransitionExperience:
    def test_defaults(self):
        exp = TransitionExperience(
            from_level=AutonomyLevel.MANUAL,
            to_level=AutonomyLevel.ASSISTED,
            outcome="success",
            satisfaction=0.9,
        )
        assert exp.context_snapshot == {}

    def test_with_context(self):
        exp = TransitionExperience(
            from_level=AutonomyLevel.MANUAL,
            to_level=AutonomyLevel.ASSISTED,
            outcome="success",
            satisfaction=0.8,
            context_snapshot={"weather": "clear"},
        )
        assert exp.context_snapshot["weather"] == "clear"


# ── AutonomyLearner ───────────────────────────────────────────────

class TestAutonomyLearner:

    @pytest.fixture()
    def learner(self):
        return AutonomyLearner()

    @pytest.fixture()
    def populated_learner(self):
        al = AutonomyLearner()
        al.record_experience(TransitionExperience(
            from_level=AutonomyLevel.MANUAL,
            to_level=AutonomyLevel.ASSISTED,
            outcome="success",
            satisfaction=0.9,
            context_snapshot={"weather": "clear"},
        ))
        al.record_experience(TransitionExperience(
            from_level=AutonomyLevel.ASSISTED,
            to_level=AutonomyLevel.SEMI_AUTO,
            outcome="success",
            satisfaction=0.7,
            context_snapshot={"weather": "clear"},
        ))
        al.record_experience(TransitionExperience(
            from_level=AutonomyLevel.SEMI_AUTO,
            to_level=AutonomyLevel.MANUAL,
            outcome="failure",
            satisfaction=0.2,
            context_snapshot={"weather": "rain"},
        ))
        return al

    # record_experience
    def test_record_single(self, learner):
        exp = TransitionExperience(
            from_level=AutonomyLevel.MANUAL,
            to_level=AutonomyLevel.ASSISTED,
            outcome="success",
            satisfaction=0.8,
        )
        learner.record_experience(exp)
        stats = learner.get_learning_statistics()
        assert stats["total_experiences"] == 1

    def test_record_multiple(self, learner):
        for i in range(5):
            learner.record_experience(TransitionExperience(
                from_level=AutonomyLevel.MANUAL,
                to_level=AutonomyLevel.ASSISTED,
                outcome="success",
                satisfaction=0.5 + i * 0.1,
            ))
        stats = learner.get_learning_statistics()
        assert stats["total_experiences"] == 5

    def test_record_with_context_indexes(self, learner):
        learner.record_experience(TransitionExperience(
            from_level=AutonomyLevel.MANUAL,
            to_level=AutonomyLevel.ASSISTED,
            outcome="success",
            satisfaction=0.9,
            context_snapshot={"weather": "clear", "terrain": "flat"},
        ))
        result = learner.compute_optimal_level({"weather": "clear"})
        assert result is not None

    # compute_optimal_level
    def test_optimal_empty(self, learner):
        assert learner.compute_optimal_level({}) is None

    def test_optimal_returns_level(self, populated_learner):
        result = populated_learner.compute_optimal_level({"weather": "clear"})
        assert result in AutonomyLevel

    def test_optimal_best_matching_context(self, populated_learner):
        result = populated_learner.compute_optimal_level({"weather": "clear"})
        # Both clear experiences are high-satisfaction, one ASSISTED (0.9) one SEMI_AUTO (0.7)
        # But ASSISTED has higher satisfaction
        assert result == AutonomyLevel.ASSISTED

    def test_optimal_no_matching_context(self, populated_learner):
        result = populated_learner.compute_optimal_level({"weather": "snow"})
        # No matching key in context
        assert result is None

    # analyze_transition_satisfaction
    def test_analyze_empty(self, learner):
        assert learner.analyze_transition_satisfaction() == []

    def test_analyze_returns_list(self, populated_learner):
        result = populated_learner.analyze_transition_satisfaction()
        assert isinstance(result, list)

    def test_analyze_sorted_descending(self, populated_learner):
        result = populated_learner.analyze_transition_satisfaction()
        for i in range(len(result) - 1):
            assert result[i]["avg_satisfaction"] >= result[i + 1]["avg_satisfaction"]

    def test_analyze_has_keys(self, populated_learner):
        result = populated_learner.analyze_transition_satisfaction()
        for r in result:
            assert "from_level" in r
            assert "to_level" in r
            assert "count" in r
            assert "avg_satisfaction" in r

    def test_analyze_custom_list(self, learner):
        exps = [
            TransitionExperience(AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED, "success", 0.9),
            TransitionExperience(AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED, "success", 0.7),
        ]
        result = learner.analyze_transition_satisfaction(exps)
        assert len(result) == 1
        assert result[0]["avg_satisfaction"] == 0.8

    # adapt_thresholds
    def test_adapt_empty(self, learner):
        t = learner.adapt_thresholds()
        assert t["risk_up"] == 0.0
        assert t["risk_down"] == 0.0

    def test_adapt_all_success(self, learner):
        for _ in range(10):
            learner.record_experience(TransitionExperience(
                AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED, "success", 0.9,
            ))
        t = learner.adapt_thresholds()
        assert t["risk_up"] > 0.0
        assert t["risk_down"] == 0.0
        assert t["confidence_up"] > 0.0

    def test_adapt_all_failure(self, learner):
        for _ in range(10):
            learner.record_experience(TransitionExperience(
                AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED, "failure", 0.1,
            ))
        t = learner.adapt_thresholds()
        assert t["risk_up"] == 0.0
        assert t["risk_down"] > 0.0
        assert t["confidence_down"] > 0.0

    def test_adapt_mixed(self, learner):
        learner.record_experience(TransitionExperience(
            AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED, "success", 0.8,
        ))
        learner.record_experience(TransitionExperience(
            AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED, "failure", 0.2,
        ))
        t = learner.adapt_thresholds()
        # success_rate = 0.5, delta = 0 => both 0
        assert t["risk_up"] == 0.0
        assert t["risk_down"] == 0.0

    def test_adapt_custom_list(self, learner):
        exps = [
            TransitionExperience(AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED, "success", 0.9),
        ]
        t = learner.adapt_thresholds(exps)
        assert t["risk_up"] > 0.0

    # predict_satisfaction
    def test_predict_no_data(self, learner):
        pred = learner.predict_satisfaction(
            AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED, {}
        )
        assert pred == 0.5

    def test_predict_matching_context(self, populated_learner):
        pred = populated_learner.predict_satisfaction(
            AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED, {"weather": "clear"},
        )
        assert pred == 0.9

    def test_predict_no_context_match(self, populated_learner):
        pred = populated_learner.predict_satisfaction(
            AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED, {"weather": "snow"},
        )
        # Falls back to average for same from/to pair
        assert pred == 0.9  # only one MANUAL->ASSISTED exists

    def test_predict_no_match_at_all(self, populated_learner):
        pred = populated_learner.predict_satisfaction(
            AutonomyLevel.FULL_AUTO, AutonomyLevel.AUTONOMOUS, {},
        )
        assert pred == 0.5

    # get_learning_statistics
    def test_stats_empty(self, learner):
        stats = learner.get_learning_statistics()
        assert stats["total_experiences"] == 0
        assert stats["avg_satisfaction"] == 0.0
        assert stats["success_rate"] == 0.0
        assert stats["most_common_transition"] is None

    def test_stats_populated(self, populated_learner):
        stats = populated_learner.get_learning_statistics()
        assert stats["total_experiences"] == 3
        assert stats["unique_transitions"] == 3
        assert stats["avg_satisfaction"] > 0.0

    def test_stats_success_rate(self, populated_learner):
        stats = populated_learner.get_learning_statistics()
        # 2 successes out of 3
        assert stats["success_rate"] == pytest.approx(2 / 3, rel=1e-4)

    def test_stats_most_common(self, populated_learner):
        # Add duplicate
        populated_learner.record_experience(TransitionExperience(
            AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED, "success", 0.8,
        ))
        stats = populated_learner.get_learning_statistics()
        assert stats["most_common_transition"] == (AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED)

    def test_stats_best_worst(self, populated_learner):
        stats = populated_learner.get_learning_statistics()
        assert stats["best_transition"] == (AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED)
        assert stats["worst_transition"] == (AutonomyLevel.SEMI_AUTO, AutonomyLevel.MANUAL)
