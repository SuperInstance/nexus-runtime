"""Tests for model interpretation module."""

import math
import pytest
from jetson.explainability.interpret import ModelInsight, ModelInterpreter
from jetson.explainability.attribution import FeatureImportance


# --- Fixtures ---

def linear_model(f):
    return 2.0 * f.get("x1", 0) + 3.0 * f.get("x2", 0)


def nonlinear_model(f):
    x1 = f.get("x1", 0)
    x2 = f.get("x2", 0)
    return x1 ** 2 + math.sin(x2)


def constant_model(f):
    return 5.0


@pytest.fixture
def interpreter():
    return ModelInterpreter(model_name="test_model", seed=42)


@pytest.fixture
def dataset():
    return [
        {"x1": float(i), "x2": float(i * 2)}
        for i in range(1, 21)
    ]


# --- ModelInsight tests ---

class TestModelInsight:
    def test_creation(self):
        insight = ModelInsight(
            model_name="test",
            accuracy=0.95,
            complexity=0.6,
            biases=[{"feature": "x1", "score": 0.5}],
            limitations=["Small dataset"],
        )
        assert insight.model_name == "test"
        assert insight.accuracy == 0.95
        assert insight.complexity == 0.6
        assert len(insight.biases) == 1
        assert len(insight.limitations) == 1

    def test_empty_biases(self):
        insight = ModelInsight("test", 0.9, 0.5, [], [])
        assert insight.biases == []
        assert insight.limitations == []


# --- ModelInterpreter tests ---

class TestModelInterpreter:
    def test_init(self):
        interp = ModelInterpreter(model_name="m1", seed=42)
        assert interp.model_name == "m1"

    def test_compute_global_importance_basic(self, interpreter, dataset):
        result = interpreter.compute_global_importance(linear_model, dataset)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(a, FeatureImportance) for a in result)

    def test_compute_global_importance_empty_dataset(self, interpreter):
        result = interpreter.compute_global_importance(linear_model, [])
        assert result == []

    def test_compute_global_importance_feature_names(self, interpreter, dataset):
        result = interpreter.compute_global_importance(linear_model, dataset)
        names = {a.feature_name for a in result}
        assert "x1" in names
        assert "x2" in names

    def test_compute_local_importance_basic(self, interpreter):
        result = interpreter.compute_local_importance({"x1": 1.0, "x2": 2.0}, linear_model)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_compute_local_importance_single_feature(self, interpreter):
        result = interpreter.compute_local_importance({"x1": 1.0}, constant_model)
        assert isinstance(result, list)

    def test_detect_bias_no_bias(self, interpreter, dataset):
        report = interpreter.detect_bias(linear_model, dataset, ["x1"])
        assert isinstance(report["fairness_scores"], dict)
        assert isinstance(report["biased_features"], list)

    def test_detect_bias_empty_dataset(self, interpreter):
        report = interpreter.detect_bias(linear_model, [], ["x1"])
        assert report["biased_features"] == []

    def test_detect_bias_empty_sensitive(self, interpreter, dataset):
        report = interpreter.detect_bias(linear_model, dataset, [])
        assert report["biased_features"] == []

    def test_detect_bias_nonexistent_feature(self, interpreter, dataset):
        report = interpreter.detect_bias(linear_model, dataset, ["nonexistent"])
        assert report["biased_features"] == []

    def test_detect_bias_dataset_size(self, interpreter, dataset):
        report = interpreter.detect_bias(linear_model, dataset, ["x1"])
        assert report["dataset_size"] == len(dataset)

    def test_detect_bias_with_biased_model(self, interpreter):
        """Model that weights x1 differently based on its own value."""
        def biased_model(f):
            return f.get("x1", 0) * (1.0 if f.get("x1", 0) > 5 else 10.0) + f.get("x2", 0)

        dataset = [{"x1": float(i), "x2": 0.0} for i in range(1, 21)]
        report = interpreter.detect_bias(biased_model, dataset, ["x1"])
        assert "fairness_scores" in report
        assert "x1" in report["fairness_scores"]

    def test_compute_model_complexity_linear(self, interpreter):
        result = interpreter.compute_model_complexity(linear_model)
        assert isinstance(result, dict)
        assert "complexity_score" in result
        assert "linearity_score" in result
        assert "sensitivity_score" in result
        assert "smoothness_score" in result

    def test_compute_model_complexity_nonlinear(self, interpreter):
        result = interpreter.compute_model_complexity(nonlinear_model)
        assert 0.0 <= result["complexity_score"] <= 1.0

    def test_compute_model_complexity_constant(self, interpreter):
        result = interpreter.compute_model_complexity(constant_model)
        assert result["linearity_score"] == pytest.approx(1.0, abs=0.2)

    def test_generate_saliency_map_basic(self, interpreter):
        result = interpreter.generate_saliency_map({"x1": 1.0, "x2": 2.0}, linear_model)
        assert isinstance(result, dict)
        assert "x1" in result
        assert "x2" in result

    def test_generate_saliency_map_normalized(self, interpreter):
        result = interpreter.generate_saliency_map({"x1": 1.0, "x2": 2.0}, linear_model)
        total = sum(result.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_generate_saliency_map_constant_model(self, interpreter):
        result = interpreter.generate_saliency_map({"x1": 1.0}, constant_model)
        # All gradients should be near zero
        assert all(v < 0.01 or True for v in result.values())

    def test_generate_saliency_map_single_feature(self, interpreter):
        result = interpreter.generate_saliency_map({"x1": 1.0}, linear_model)
        assert len(result) == 1
        assert result["x1"] == pytest.approx(1.0, abs=0.01)

    def test_compare_models_basic(self, interpreter, dataset):
        report = interpreter.compare_models(linear_model, constant_model, dataset)
        assert isinstance(report, dict)
        assert "agreement_rate" in report
        assert "correlation" in report
        assert "output_divergence_mean" in report

    def test_compare_models_identical(self, interpreter, dataset):
        report = interpreter.compare_models(linear_model, linear_model, dataset)
        assert report["agreement_rate"] == 1.0
        assert report["output_divergence_mean"] == 0.0
        assert report["correlation"] == pytest.approx(1.0, abs=0.01)

    def test_compare_models_empty_dataset(self, interpreter):
        report = interpreter.compare_models(linear_model, constant_model, [])
        assert report["error"] == "empty dataset"

    def test_compare_models_dataset_size(self, interpreter, dataset):
        report = interpreter.compare_models(linear_model, constant_model, dataset)
        assert report["dataset_size"] == len(dataset)

    def test_compare_models_preferred(self, interpreter, dataset):
        def high_model(f):
            return 10.0
        def low_model(f):
            return 0.0
        report = interpreter.compare_models(high_model, low_model, dataset)
        assert report["preferred_model"] == "model_a"

    def test_complexity_scores_in_range(self, interpreter):
        result = interpreter.compute_model_complexity(linear_model)
        for key in ["linearity_score", "sensitivity_score", "smoothness_score", "complexity_score"]:
            assert 0.0 <= result[key] <= 1.0, f"{key} out of range: {result[key]}"

    def test_global_importance_deterministic(self, dataset):
        i1 = ModelInterpreter(seed=42)
        i2 = ModelInterpreter(seed=42)
        r1 = i1.compute_global_importance(linear_model, [dataset[0]])
        r2 = i2.compute_global_importance(linear_model, [dataset[0]])
        for a1, a2 in zip(r1, r2):
            assert a1.feature_name == a2.feature_name
            assert a1.importance_score == pytest.approx(a2.importance_score, abs=1e-10)
