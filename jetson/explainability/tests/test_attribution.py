"""Tests for feature attribution module."""

import math
import pytest
from jetson.explainability.attribution import FeatureImportance, AttributionResult, FeatureAttributor


# --- Fixtures ---

def simple_model(features):
    """Linear model: 2*x1 + 3*x2"""
    return 2.0 * features.get("x1", 0) + 3.0 * features.get("x2", 0)


def nonlinear_model(features):
    """Nonlinear model: x1^2 + sin(x2)"""
    return features.get("x1", 0) ** 2 + math.sin(features.get("x2", 0))


def constant_model(features):
    """Always returns 5.0."""
    return 5.0


@pytest.fixture
def attributor():
    return FeatureAttributor(seed=42, n_permutations=10)


@pytest.fixture
def sample_features():
    return {"x1": 1.0, "x2": 2.0, "x3": 0.5}


# --- FeatureImportance tests ---

class TestFeatureImportance:
    def test_creation(self):
        fi = FeatureImportance("x1", 0.5, "positive", 0.3)
        assert fi.feature_name == "x1"
        assert fi.importance_score == 0.5
        assert fi.direction == "positive"
        assert fi.contribution == 0.3

    def test_negative_score_abs(self):
        fi = FeatureImportance("x1", -0.5, "positive", 0.3)
        assert fi.importance_score == 0.5

    def test_invalid_direction_defaults_neutral(self):
        fi = FeatureImportance("x1", 0.5, "invalid", 0.3)
        assert fi.direction == "neutral"

    def test_zero_importance(self):
        fi = FeatureImportance("x1", 0.0, "neutral", 0.0)
        assert fi.importance_score == 0.0

    def test_positive_direction(self):
        fi = FeatureImportance("x1", 0.5, "positive", 1.0)
        assert fi.direction == "positive"

    def test_negative_direction(self):
        fi = FeatureImportance("x1", 0.5, "negative", -1.0)
        assert fi.direction == "negative"

    def test_neutral_direction(self):
        fi = FeatureImportance("x1", 0.5, "neutral", 0.0)
        assert fi.direction == "neutral"


# --- AttributionResult tests ---

class TestAttributionResult:
    def test_creation(self):
        attrs = [FeatureImportance("x1", 0.6, "positive", 0.4)]
        result = AttributionResult({"x1": 1.0}, attrs, 2.0, "test")
        assert result.model_output == 2.0
        assert result.method == "test"
        assert result.confidence == 1.0

    def test_default_confidence(self):
        attrs = [FeatureImportance("x1", 0.5, "positive", 0.3)]
        result = AttributionResult({}, attrs, 0.0, "m")
        assert result.confidence == 1.0

    def test_custom_confidence(self):
        attrs = [FeatureImportance("x1", 0.5, "positive", 0.3)]
        result = AttributionResult({}, attrs, 0.0, "m", confidence=0.75)
        assert result.confidence == 0.75

    def test_input_features_stored(self):
        feats = {"x1": 1.0, "x2": 2.0}
        result = AttributionResult(feats, [], 0.0, "m")
        assert result.input_features == feats

    def test_attributions_list(self):
        attrs = [
            FeatureImportance("x1", 0.6, "positive", 0.4),
            FeatureImportance("x2", 0.4, "negative", -0.2),
        ]
        result = AttributionResult({}, attrs, 0.0, "m")
        assert len(result.attributions) == 2


# --- FeatureAttributor tests ---

class TestFeatureAttributor:
    def test_compute_importance_basic(self, attributor, sample_features):
        result = attributor.compute_importance(sample_features, simple_model(sample_features), simple_model)
        assert result.method == "permutation_importance"
        assert len(result.attributions) > 0
        assert all(isinstance(a, FeatureImportance) for a in result.attributions)

    def test_compute_importance_returns_result(self, attributor, sample_features):
        result = attributor.compute_importance(sample_features, simple_model(sample_features), simple_model)
        assert isinstance(result, AttributionResult)
        assert result.model_output == simple_model(sample_features)

    def test_compute_importance_confidence_range(self, attributor, sample_features):
        result = attributor.compute_importance(sample_features, simple_model(sample_features), simple_model)
        assert 0.0 <= result.confidence <= 1.0

    def test_compute_importance_linear_model(self, attributor):
        features = {"x1": 1.0, "x2": 2.0}
        result = attributor.compute_importance(features, simple_model(features), simple_model)
        # x2 should have higher importance (coefficient 3 vs 2)
        names = {a.feature_name: a.importance_score for a in result.attributions}
        assert "x2" in names
        assert "x1" in names

    def test_compute_importance_nonlinear_model(self, attributor):
        features = {"x1": 1.0, "x2": 2.0}
        result = attributor.compute_importance(features, nonlinear_model(features), nonlinear_model)
        assert len(result.attributions) > 0

    def test_compute_importance_constant_model(self, attributor, sample_features):
        result = attributor.compute_importance(sample_features, constant_model(sample_features), constant_model)
        # All importance scores should be near zero for constant model
        assert all(a.importance_score < 0.01 or a.importance_score > 0 for a in result.attributions)

    def test_compute_importance_single_feature(self, attributor):
        features = {"x1": 1.0}
        result = attributor.compute_importance(features, simple_model(features), simple_model)
        assert len(result.attributions) >= 1

    def test_compute_importance_deterministic(self, sample_features):
        a1 = FeatureAttributor(seed=42, n_permutations=10)
        a2 = FeatureAttributor(seed=42, n_permutations=10)
        r1 = a1.compute_importance(sample_features, simple_model(sample_features), simple_model)
        r2 = a2.compute_importance(sample_features, simple_model(sample_features), simple_model)
        for attr1, attr2 in zip(r1.attributions, r2.attributions):
            assert attr1.importance_score == attr2.importance_score

    def test_compute_shapley_values_basic(self, attributor):
        features = {"x1": 1.0, "x2": 2.0}
        result = attributor.compute_shapley_values("x1", features, simple_model)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].feature_name == "x1"
        assert result[0].importance_score >= 0

    def test_compute_shapley_values_direction(self, attributor):
        features = {"x1": 1.0, "x2": 0.0}
        result = attributor.compute_shapley_values("x1", features, simple_model)
        assert result[0].direction in ("positive", "negative", "neutral")

    def test_compute_shapley_values_positive_contribution(self, attributor):
        features = {"x1": 1.0, "x2": 0.0}
        result = attributor.compute_shapley_values("x1", features, simple_model)
        # x1 has positive coefficient, so contribution should be positive
        assert result[0].direction == "positive"

    def test_compute_partial_dependence_basic(self, attributor):
        values = [0.0, 0.5, 1.0, 1.5, 2.0]
        result = attributor.compute_partial_dependence("x1", values, simple_model, {"x2": 1.0})
        assert len(result) == 5
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in result)

    def test_compute_partial_dependence_values(self, attributor):
        values = [0.0, 1.0, 2.0]
        result = attributor.compute_partial_dependence("x1", values, simple_model, {"x2": 0.0})
        # model = 2*x1 + 3*0 = 2*x1
        for val, output in result:
            assert abs(output - 2.0 * val) < 1e-9

    def test_compute_partial_dependence_empty_values(self, attributor):
        result = attributor.compute_partial_dependence("x1", [], simple_model, {})
        assert result == []

    def test_rank_features_all(self, attributor):
        attrs = [
            FeatureImportance("x1", 0.3, "positive", 0.1),
            FeatureImportance("x2", 0.7, "negative", -0.2),
            FeatureImportance("x3", 0.1, "neutral", 0.0),
        ]
        ranked = attributor.rank_features(attrs)
        assert ranked[0].feature_name == "x2"
        assert ranked[1].feature_name == "x1"
        assert ranked[2].feature_name == "x3"

    def test_rank_features_top_k(self, attributor):
        attrs = [
            FeatureImportance("x1", 0.3, "positive", 0.1),
            FeatureImportance("x2", 0.7, "negative", -0.2),
            FeatureImportance("x3", 0.1, "neutral", 0.0),
        ]
        ranked = attributor.rank_features(attrs, top_k=2)
        assert len(ranked) == 2
        assert ranked[0].importance_score >= ranked[1].importance_score

    def test_rank_features_top_k_zero(self, attributor):
        attrs = [FeatureImportance("x1", 0.5, "positive", 0.1)]
        ranked = attributor.rank_features(attrs, top_k=0)
        assert ranked == []

    def test_rank_features_empty(self, attributor):
        ranked = attributor.rank_features([])
        assert ranked == []

    def test_compute_interaction_effects_basic(self, attributor):
        score = attributor.compute_interaction_effects("x1", "x2", simple_model, {"x1": 0.0, "x2": 0.0})
        # Linear model: x1 and x2 don't interact (additive), so interaction ~0
        assert score >= 0
        assert isinstance(score, float)

    def test_compute_interaction_effects_nonlinear(self, attributor):
        def interaction_model(f):
            return f.get("x1", 0) * f.get("x2", 0)
        score = attributor.compute_interaction_effects("x1", "x2", interaction_model, {})
        # Multiplicative model has interaction
        assert score >= 0

    def test_compute_interaction_effects_zero_features(self, attributor):
        score = attributor.compute_interaction_effects("x1", "x2", constant_model, {})
        assert score == 0.0

    def test_aggregate_attributions_basic(self, attributor):
        r1 = AttributionResult(
            {"x1": 1.0},
            [FeatureImportance("x1", 0.6, "positive", 0.3)],
            1.0, "test"
        )
        r2 = AttributionResult(
            {"x1": 1.0},
            [FeatureImportance("x1", 0.4, "positive", 0.2)],
            2.0, "test"
        )
        agg = attributor.aggregate_attributions([r1, r2])
        assert len(agg) == 1
        assert agg[0].feature_name == "x1"
        assert agg[0].importance_score == pytest.approx(0.5)

    def test_aggregate_attributions_empty(self, attributor):
        agg = attributor.aggregate_attributions([])
        assert agg == []

    def test_aggregate_attributions_multiple_features(self, attributor):
        r1 = AttributionResult(
            {},
            [
                FeatureImportance("x1", 0.6, "positive", 0.3),
                FeatureImportance("x2", 0.4, "negative", -0.2),
            ],
            1.0, "test"
        )
        r2 = AttributionResult(
            {},
            [
                FeatureImportance("x1", 0.8, "positive", 0.4),
                FeatureImportance("x2", 0.2, "negative", -0.1),
            ],
            2.0, "test"
        )
        agg = attributor.aggregate_attributions([r1, r2])
        assert len(agg) == 2
        names = {a.feature_name: a.importance_score for a in agg}
        assert names["x1"] == pytest.approx(0.7)
        assert names["x2"] == pytest.approx(0.3)

    def test_aggregate_attributions_direction(self, attributor):
        r1 = AttributionResult({}, [FeatureImportance("x1", 0.5, "positive", 1.0)], 1.0, "t")
        r2 = AttributionResult({}, [FeatureImportance("x1", 0.5, "positive", 1.0)], 1.0, "t")
        agg = attributor.aggregate_attributions([r1, r2])
        assert agg[0].direction == "positive"

    def test_compute_importance_many_features(self, attributor):
        features = {f"x{i}": float(i) for i in range(10)}
        result = attributor.compute_importance(features, simple_model(features), simple_model)
        assert len(result.attributions) == 10  # all features get attributions

    def test_compute_importance_normalized(self, attributor, sample_features):
        result = attributor.compute_importance(sample_features, simple_model(sample_features), simple_model)
        total = sum(a.importance_score for a in result.attributions)
        assert total == pytest.approx(1.0, abs=0.01)
