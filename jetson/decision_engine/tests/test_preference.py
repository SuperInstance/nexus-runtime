"""Tests for preference.py — PreferenceModel, Preference."""

import pytest

from jetson.decision_engine.preference import Preference, PreferenceModel


# ============================================================
# Preference dataclass
# ============================================================

class TestPreference:
    def test_preference_defaults(self):
        p = Preference(criterion="cost", direction="min")
        assert p.importance == 1.0
        assert p.indifference_threshold == 0.0
        assert p.preference_threshold == 0.0

    def test_preference_full(self):
        p = Preference(
            criterion="quality",
            direction="max",
            importance=2.5,
            indifference_threshold=0.1,
            preference_threshold=0.5,
        )
        assert p.criterion == "quality"
        assert p.direction == "max"
        assert p.importance == pytest.approx(2.5)


# ============================================================
# PreferenceModel — construction & add_preference
# ============================================================

class TestPreferenceModelBasics:
    def test_empty_model(self):
        model = PreferenceModel()
        assert model.preferences == []

    def test_add_preference(self):
        model = PreferenceModel()
        p = Preference(criterion="cost", direction="min")
        model.add_preference(p)
        assert len(model.preferences) == 1
        assert model.preferences[0].criterion == "cost"

    def test_add_multiple_preferences(self):
        model = PreferenceModel()
        model.add_preference(Preference(criterion="cost", direction="min"))
        model.add_preference(Preference(criterion="quality", direction="max"))
        assert len(model.preferences) == 2


# ============================================================
# PreferenceModel.compare_alternatives
# ============================================================

class TestCompareAlternatives:
    def setup_method(self):
        self.model = PreferenceModel()
        self.model.add_preference(
            Preference(criterion="cost", direction="min", importance=1.0,
                        indifference_threshold=0.0, preference_threshold=1.0)
        )
        self.model.add_preference(
            Preference(criterion="quality", direction="max", importance=1.0,
                        indifference_threshold=0.0, preference_threshold=1.0)
        )

    def test_compare_a_better(self):
        a = {"cost": 5, "quality": 8}
        b = {"cost": 10, "quality": 5}
        preferred, strength = self.model.compare_alternatives(a, b)
        assert preferred == "a"
        assert strength > 0

    def test_compare_b_better(self):
        a = {"cost": 10, "quality": 5}
        b = {"cost": 5, "quality": 8}
        preferred, strength = self.model.compare_alternatives(a, b)
        assert preferred == "b"
        assert strength > 0

    def test_compare_tie(self):
        a = {"cost": 5, "quality": 5}
        b = {"cost": 5, "quality": 5}
        preferred, strength = self.model.compare_alternatives(a, b)
        assert preferred == "tie"

    def test_compare_empty_model(self):
        model = PreferenceModel()
        preferred, strength = model.compare_alternatives({"x": 1}, {"x": 2})
        assert preferred == "tie"
        assert strength == 0.0

    def test_compare_with_thresholds(self):
        model = PreferenceModel()
        model.add_preference(
            Preference(criterion="cost", direction="min", importance=1.0,
                        indifference_threshold=2.0, preference_threshold=5.0)
        )
        a = {"cost": 10.0}
        b = {"cost": 10.5}
        preferred, strength = model.compare_alternatives(a, b)
        # Within indifference zone
        assert preferred == "tie"

    def test_compare_a_beyond_preference_threshold(self):
        model = PreferenceModel()
        model.add_preference(
            Preference(criterion="cost", direction="min", importance=1.0,
                        indifference_threshold=1.0, preference_threshold=3.0)
        )
        a = {"cost": 5.0}
        b = {"cost": 15.0}
        preferred, strength = self.model.compare_alternatives(a, b)
        assert preferred == "a"

    def test_compare_missing_criterion(self):
        a = {"cost": 5}
        b = {"quality": 8}
        # Neither alternative has both criteria — neither can be compared
        preferred, _ = self.model.compare_alternatives(a, b)
        assert preferred == "tie"


# ============================================================
# PreferenceModel.compute_outranking
# ============================================================

class TestComputeOutranking:
    def test_outranking_matrix_size(self):
        model = PreferenceModel()
        model.add_preference(Preference(criterion="cost", direction="min"))
        alts = [{"cost": i} for i in range(4)]
        matrix = model.compute_outranking(alts)
        assert len(matrix) == 4
        assert len(matrix[0]) == 4

    def test_outranking_diagonal(self):
        model = PreferenceModel()
        model.add_preference(Preference(criterion="cost", direction="min"))
        alts = [{"cost": i} for i in range(3)]
        matrix = model.compute_outranking(alts)
        for i in range(3):
            assert matrix[i][i] == 1.0

    def test_outranking_empty(self):
        model = PreferenceModel()
        model.add_preference(Preference(criterion="cost", direction="min"))
        matrix = model.compute_outranking([])
        assert matrix == []

    def test_outranking_two_alternatives(self):
        model = PreferenceModel()
        model.add_preference(Preference(criterion="cost", direction="min"))
        alts = [{"cost": 5}, {"cost": 10}]
        matrix = model.compute_outranking(alts)
        # alt 0 should outrank alt 1 (lower cost)
        assert matrix[0][1] > matrix[1][0]

    def test_outranking_empty_preferences(self):
        model = PreferenceModel()
        alts = [{"x": 1}, {"x": 2}]
        matrix = model.compute_outranking(alts)
        assert matrix[0][1] == 0.0


# ============================================================
# PreferenceModel.compute_ranking
# ============================================================

class TestComputeRanking:
    def test_ranking_basic(self):
        model = PreferenceModel()
        model.add_preference(Preference(criterion="cost", direction="min"))
        alts = [{"cost": 5}, {"cost": 10}, {"cost": 1}]
        matrix = model.compute_outranking(alts)
        ranking = model.compute_ranking(matrix)
        assert ranking[0] == 2  # cost=1 is best

    def test_ranking_empty(self):
        model = PreferenceModel()
        ranking = model.compute_ranking([])
        assert ranking == []

    def test_ranking_single(self):
        model = PreferenceModel()
        matrix = [[1.0]]
        ranking = model.compute_ranking(matrix)
        assert ranking == [0]


# ============================================================
# PreferenceModel.learn_preferences
# ============================================================

class TestLearnPreferences:
    def test_learn_basic(self):
        model = PreferenceModel()
        pairs = [
            ({"cost": 5, "quality": 8}, {"cost": 10, "quality": 3}),
            ({"cost": 3, "quality": 9}, {"cost": 7, "quality": 2}),
        ]
        result = model.learn_preferences(pairs)
        assert result is model
        assert len(model.preferences) == 2

    def test_learn_weights_positive(self):
        model = PreferenceModel()
        pairs = [
            ({"a": 5, "b": 1}, {"a": 1, "b": 5}),
            ({"a": 4, "b": 2}, {"a": 2, "b": 4}),
        ]
        model.learn_preferences(pairs)
        for p in model.preferences:
            assert p.importance >= 0

    def test_learn_empty_pairs(self):
        model = PreferenceModel()
        model.learn_preferences([])
        assert model.preferences == []

    def test_learn_updates_preferences(self):
        model = PreferenceModel()
        model.add_preference(Preference(criterion="old", direction="min"))
        pairs = [
            ({"new_a": 5, "new_b": 1}, {"new_a": 1, "new_b": 5}),
        ]
        model.learn_preferences(pairs)
        names = [p.criterion for p in model.preferences]
        assert "old" not in names
        assert "new_a" in names


# ============================================================
# PreferenceModel.validate_preferences
# ============================================================

class TestValidatePreferences:
    def test_valid_preferences(self):
        model = PreferenceModel()
        model.add_preference(Preference(criterion="cost", direction="min"))
        score = model.validate_preferences()
        assert score == pytest.approx(1.0)

    def test_negative_indifference(self):
        prefs = [Preference(criterion="x", direction="min", indifference_threshold=-1.0)]
        model = PreferenceModel()
        score = model.validate_preferences(prefs)
        assert score < 1.0

    def test_preference_below_indifference(self):
        prefs = [
            Preference(criterion="x", direction="min",
                        indifference_threshold=2.0, preference_threshold=1.0)
        ]
        model = PreferenceModel()
        score = model.validate_preferences(prefs)
        assert score < 1.0

    def test_negative_importance(self):
        prefs = [Preference(criterion="x", direction="min", importance=-1.0)]
        model = PreferenceModel()
        score = model.validate_preferences(prefs)
        assert score < 1.0

    def test_duplicate_criteria(self):
        prefs = [
            Preference(criterion="cost", direction="min"),
            Preference(criterion="cost", direction="max"),
        ]
        model = PreferenceModel()
        score = model.validate_preferences(prefs)
        assert score < 1.0

    def test_empty_preferences(self):
        model = PreferenceModel()
        score = model.validate_preferences()
        assert score == 1.0

    def test_validate_uses_model_prefs(self):
        model = PreferenceModel()
        model.add_preference(Preference(criterion="cost", direction="min"))
        model.add_preference(
            Preference(criterion="cost", direction="min")  # duplicate
        )
        score = model.validate_preferences()
        assert score < 1.0
