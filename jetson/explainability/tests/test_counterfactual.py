"""Tests for counterfactual explanation module."""

import math
import pytest
from jetson.explainability.counterfactual import CounterfactualExample, CounterfactualGenerator


# --- Fixtures ---

def linear_model(f):
    return 2.0 * f.get("x1", 0) + 3.0 * f.get("x2", 0)


def nonlinear_model(f):
    x1 = f.get("x1", 0)
    x2 = f.get("x2", 0)
    return x1 ** 2 + x2


def constant_model(f):
    return 5.0


@pytest.fixture
def generator():
    return CounterfactualGenerator(seed=42, step_size=0.2, max_iterations=50)


@pytest.fixture
def input_data():
    return {"x1": 1.0, "x2": 1.0}


# --- CounterfactualExample tests ---

class TestCounterfactualExample:
    def test_creation(self):
        cf = CounterfactualExample(
            original_input={"x1": 1.0},
            modified_input={"x1": 2.0},
            original_output=2.0,
            counterfactual_output=4.0,
            changed_features=["x1"],
            plausibility=0.8,
        )
        assert cf.changed_features == ["x1"]
        assert cf.plausibility == 0.8

    def test_plausibility_clamped_high(self):
        cf = CounterfactualExample({}, {}, 0, 0, [], 1.5)
        assert cf.plausibility == 1.0

    def test_plausibility_clamped_low(self):
        cf = CounterfactualExample({}, {}, 0, 0, [], -0.5)
        assert cf.plausibility == 0.0

    def test_fields_immutable(self):
        cf = CounterfactualExample({"x1": 1.0}, {"x1": 2.0}, 2, 4, ["x1"], 0.8)
        assert cf.original_input == {"x1": 1.0}
        assert cf.modified_input == {"x1": 2.0}

    def test_empty_changed_features(self):
        cf = CounterfactualExample({}, {}, 0, 0, [], 1.0)
        assert cf.changed_features == []


# --- CounterfactualGenerator tests ---

class TestCounterfactualGenerator:
    def test_init(self):
        gen = CounterfactualGenerator(seed=42, step_size=0.1, max_iterations=100)
        assert gen.seed == 42
        assert gen.step_size == 0.1
        assert gen.max_iterations == 100

    def test_generate_basic(self, generator, input_data):
        cf = generator.generate(input_data, 10.0, linear_model)
        assert isinstance(cf, CounterfactualExample)
        assert cf.original_input == input_data
        assert cf.original_output == linear_model(input_data)

    def test_generate_changed_features(self, generator, input_data):
        cf = generator.generate(input_data, 10.0, linear_model)
        assert isinstance(cf.changed_features, list)

    def test_generate_plausibility_range(self, generator, input_data):
        cf = generator.generate(input_data, 10.0, linear_model)
        assert 0.0 <= cf.plausibility <= 1.0

    def test_generate_counterfactual_output(self, generator, input_data):
        target = 20.0
        cf = generator.generate(input_data, target, linear_model)
        assert isinstance(cf.counterfactual_output, float)

    def test_generate_with_max_changes(self, generator, input_data):
        cf = generator.generate(input_data, 10.0, linear_model, max_changes=1)
        assert len(cf.changed_features) <= 1

    def test_generate_zero_max_changes(self, generator, input_data):
        cf = generator.generate(input_data, 10.0, linear_model, max_changes=0)
        assert cf.changed_features == []

    def test_generate_same_target(self, generator, input_data):
        current = linear_model(input_data)
        cf = generator.generate(input_data, current, linear_model)
        assert cf.counterfactual_output == pytest.approx(current, abs=0.1)

    def test_generate_constant_model(self, generator, input_data):
        cf = generator.generate(input_data, 10.0, constant_model)
        # Cannot reach different target with constant model
        assert isinstance(cf, CounterfactualExample)

    def test_find_minimum_changes_basic(self, generator, input_data):
        cf = generator.find_minimum_changes(input_data, 20.0, linear_model)
        assert isinstance(cf, CounterfactualExample)
        assert len(cf.changed_features) >= 1

    def test_find_minimum_changes_no_change_needed(self, generator, input_data):
        current = linear_model(input_data)
        cf = generator.find_minimum_changes(input_data, current, linear_model)
        # Target is already met, should be minimal
        assert isinstance(cf, CounterfactualExample)

    def test_find_minimum_changes_single_feature(self, generator):
        input_data = {"x1": 1.0}
        cf = generator.find_minimum_changes(input_data, 10.0, linear_model)
        assert isinstance(cf, CounterfactualExample)
        assert "x1" in input_data

    def test_compute_plausibility_no_change(self, generator, input_data):
        score = generator.compute_plausibility(input_data, dict(input_data), {})
        assert score == pytest.approx(1.0, abs=0.01)

    def test_compute_plausibility_with_stats(self, generator, input_data):
        modified = {"x1": 1.5, "x2": 1.0}
        stats = {"x1": {"mean": 1.0, "std": 0.5}, "x2": {"mean": 1.0, "std": 0.5}}
        score = generator.compute_plausibility(input_data, modified, stats)
        assert 0.0 <= score <= 1.0

    def test_compute_plausibility_large_change(self, generator, input_data):
        modified = {"x1": 100.0, "x2": 100.0}
        score = generator.compute_plausibility(input_data, modified, {})
        assert score < 0.1

    def test_compute_plausibility_empty_stats(self, generator, input_data):
        modified = {"x1": 2.0, "x2": 2.0}
        score = generator.compute_plausibility(input_data, modified, {})
        assert 0.0 <= score <= 1.0

    def test_compute_plausibility_zero_std(self, generator, input_data):
        modified = {"x1": 2.0, "x2": 1.0}
        stats = {"x1": {"mean": 1.0, "std": 0.0}, "x2": {"mean": 1.0, "std": 0.5}}
        score = generator.compute_plausibility(input_data, modified, stats)
        assert 0.0 <= score <= 1.0

    def test_generate_diverse_explanations(self, generator, input_data):
        results = generator.generate_diverse_explanations(input_data, 20.0, linear_model, num_explanations=5)
        assert len(results) == 5
        assert all(isinstance(cf, CounterfactualExample) for cf in results)

    def test_generate_diverse_explanations_single(self, generator, input_data):
        results = generator.generate_diverse_explanations(input_data, 20.0, linear_model, num_explanations=1)
        assert len(results) == 1

    def test_generate_diverse_explanations_different(self, generator, input_data):
        results = generator.generate_diverse_explanations(input_data, 20.0, linear_model, num_explanations=3)
        # They should produce different modifications (with different seeds)
        modified_inputs = [tuple(sorted(cf.modified_input.items())) for cf in results]
        # At least some should differ
        assert len(set(modified_inputs)) >= 1

    def test_evaluate_counterfactual_valid(self, generator, input_data):
        cf = generator.generate(input_data, 20.0, linear_model)
        evaluation = generator.evaluate_counterfactual(cf, linear_model)
        assert isinstance(evaluation, dict)
        assert "valid" in evaluation
        assert "sparsity" in evaluation
        assert "plausibility" in evaluation

    def test_evaluate_counterfactual_fields(self, generator, input_data):
        cf = generator.generate(input_data, 20.0, linear_model)
        evaluation = generator.evaluate_counterfactual(cf, linear_model)
        assert "output_matches_model" in evaluation
        assert "original_matches_model" in evaluation
        assert "total_change" in evaluation
        assert "changed_features" in evaluation

    def test_evaluate_counterfactual_sparsity(self, generator, input_data):
        cf = generator.generate(input_data, 20.0, linear_model, max_changes=1)
        evaluation = generator.evaluate_counterfactual(cf, linear_model)
        # max_changes=1 means at most 1 feature changed out of 2
        assert evaluation["sparsity"] >= 0.5

    def test_evaluate_counterfactual_plausibility(self, generator, input_data):
        cf = generator.generate(input_data, 20.0, linear_model)
        evaluation = generator.evaluate_counterfactual(cf, linear_model)
        assert 0.0 <= evaluation["plausibility"] <= 1.0

    def test_generate_nonlinear_model(self, generator, input_data):
        cf = generator.generate(input_data, 10.0, nonlinear_model)
        assert isinstance(cf, CounterfactualExample)
        assert cf.original_output == nonlinear_model(input_data)

    def test_generate_many_features(self):
        gen = CounterfactualGenerator(seed=42, step_size=0.5, max_iterations=100)
        features = {f"x{i}": 0.5 for i in range(5)}
        cf = gen.generate(features, 5.0, linear_model)
        assert isinstance(cf, CounterfactualExample)

    def test_generate_diverse_num_explanations_zero(self, generator, input_data):
        results = generator.generate_diverse_explanations(input_data, 20.0, linear_model, num_explanations=0)
        assert results == []
