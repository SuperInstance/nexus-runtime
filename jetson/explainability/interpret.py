"""Model interpretation and analysis for NEXUS explainable AI.

Provides global/local importance, bias detection, complexity scoring,
saliency maps, and model comparison. Pure Python, no external dependencies.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .attribution import FeatureImportance, FeatureAttributor


@dataclass
class ModelInsight:
    """Comprehensive model insight report."""
    model_name: str
    accuracy: float
    complexity: float
    biases: List[Dict[str, Any]]
    limitations: List[str]


class ModelInterpreter:
    """Interpret ML models with global/local analysis and bias detection."""

    def __init__(self, model_name: str = "unknown", seed: int = 42):
        self.model_name = model_name
        self.seed = seed
        self._attributor = FeatureAttributor(seed=seed)

    def compute_global_importance(
        self,
        model_fn: Callable[[Dict[str, float]], float],
        dataset: List[Dict[str, float]],
    ) -> List[FeatureImportance]:
        """Compute global feature importance across entire dataset."""
        if not dataset:
            return []
        all_attributions: List[FeatureImportance] = []
        for sample in dataset:
            result = self._attributor.compute_importance(sample, model_fn(sample), model_fn)
            all_attributions.extend(result.attributions)
        # Average per feature
        return self._attributor.aggregate_per_feature(
            all_attributions, list(dataset[0].keys())
        )

    def compute_local_importance(
        self,
        input_data: Dict[str, float],
        model_fn: Callable[[Dict[str, float]], float],
    ) -> List[FeatureImportance]:
        """Compute feature importance for a single input (local explanation)."""
        result = self._attributor.compute_importance(input_data, model_fn(input_data), model_fn)
        return result.attributions

    def detect_bias(
        self,
        model_fn: Callable[[Dict[str, float]], float],
        dataset: List[Dict[str, float]],
        sensitive_features: List[str],
    ) -> Dict[str, Any]:
        """Detect bias related to sensitive features."""
        if not dataset or not sensitive_features:
            return {"biased_features": [], "fairness_scores": {}}

        fairness_scores: Dict[str, float] = {}
        biased_features: List[str] = []

        for sf in sensitive_features:
            if sf not in dataset[0]:
                continue
            # Split dataset by median of sensitive feature
            values = sorted(set(d[sf] for d in dataset))
            if len(values) < 2:
                fairness_scores[sf] = 1.0
                continue

            median_val = values[len(values) // 2]
            group_low = [d for d in dataset if d[sf] <= median_val]
            group_high = [d for d in dataset if d[sf] > median_val]

            if not group_low or not group_high:
                fairness_scores[sf] = 1.0
                continue

            # Compare outputs
            outputs_low = [model_fn(d) for d in group_low]
            outputs_high = [model_fn(d) for d in group_high]

            mean_low = sum(outputs_low) / len(outputs_low)
            mean_high = sum(outputs_high) / len(outputs_high)
            std_all = self._std([model_fn(d) for d in dataset])

            if std_all > 0:
                disparity = abs(mean_high - mean_low) / std_all
            else:
                disparity = 0.0

            # Fairness score: 1.0 = perfectly fair, 0.0 = maximally biased
            fairness = max(0.0, 1.0 - disparity)
            fairness_scores[sf] = fairness

            if fairness < 0.8:  # threshold for bias detection
                biased_features.append(sf)

        return {
            "biased_features": biased_features,
            "fairness_scores": fairness_scores,
            "dataset_size": len(dataset),
            "sensitive_features_checked": len(sensitive_features),
        }

    def compute_model_complexity(
        self,
        model_fn: Callable[[Dict[str, float]], float],
    ) -> Dict[str, float]:
        """Estimate model complexity via sensitivity analysis and nonlinearity."""
        rng = random.Random(self.seed)
        n_samples = 50
        features = ["x1", "x2", "x3", "x4", "x5"]

        # Linearity test
        linear_score = self._test_linearity(model_fn, features, rng, n_samples)
        # Sensitivity (how much output changes with input perturbation)
        sensitivity = self._test_sensitivity(model_fn, features, rng, n_samples)
        # Smoothness (output variance over nearby inputs)
        smoothness = self._test_smoothness(model_fn, features, rng, n_samples)

        # Overall complexity: higher = more complex
        complexity = (1.0 - linear_score) * 0.4 + sensitivity * 0.3 + (1.0 - smoothness) * 0.3

        return {
            "linearity_score": linear_score,  # 1.0 = perfectly linear
            "sensitivity_score": sensitivity,  # 0-1, higher = more sensitive
            "smoothness_score": smoothness,    # 1.0 = perfectly smooth
            "complexity_score": max(0.0, min(1.0, complexity)),
        }

    def generate_saliency_map(
        self,
        input_data: Dict[str, float],
        model_fn: Callable[[Dict[str, float]], float],
    ) -> Dict[str, float]:
        """Generate saliency map: gradient-like importance per feature."""
        base_output = model_fn(input_data)
        saliency: Dict[str, float] = {}

        for fname, fval in input_data.items():
            delta = max(0.001, abs(fval) * 0.01 + 0.001)
            perturbed_plus = dict(input_data)
            perturbed_plus[fname] = fval + delta
            perturbed_minus = dict(input_data)
            perturbed_minus[fname] = fval - delta

            grad = (model_fn(perturbed_plus) - model_fn(perturbed_minus)) / (2 * delta)
            saliency[fname] = abs(grad)

        # Normalize
        total = sum(saliency.values()) or 1.0
        for k in saliency:
            saliency[k] /= total
        return saliency

    def compare_models(
        self,
        model_a_fn: Callable[[Dict[str, float]], float],
        model_b_fn: Callable[[Dict[str, float]], float],
        dataset: List[Dict[str, float]],
    ) -> Dict[str, Any]:
        """Compare two models on a shared dataset."""
        if not dataset:
            return {"error": "empty dataset"}

        outputs_a = [model_a_fn(d) for d in dataset]
        outputs_b = [model_b_fn(d) for d in dataset]

        mean_a = sum(outputs_a) / len(outputs_a)
        mean_b = sum(outputs_b) / len(outputs_b)

        # Agreement rate
        agreements = sum(1 for a, b in zip(outputs_a, outputs_b) if (a > 0) == (b > 0))
        agreement_rate = agreements / len(dataset)

        # Correlation (Pearson)
        correlation = self._pearson_correlation(outputs_a, outputs_b)

        # Output divergence
        divergences = [abs(a - b) for a, b in zip(outputs_a, outputs_b)]
        mean_divergence = sum(divergences) / len(divergences)
        max_divergence = max(divergences)

        return {
            "model_a_mean_output": mean_a,
            "model_b_mean_output": mean_b,
            "output_divergence_mean": mean_divergence,
            "output_divergence_max": max_divergence,
            "agreement_rate": agreement_rate,
            "correlation": correlation,
            "dataset_size": len(dataset),
            "preferred_model": "model_a" if mean_a > mean_b else "model_b",
        }

    # --- private helpers ---

    def _std(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(variance)

    def _test_linearity(
        self,
        model_fn: Callable,
        features: List[str],
        rng: random.Random,
        n: int,
    ) -> float:
        """Test how linear the model is: 1.0 = perfectly linear."""
        errors = []
        for _ in range(n):
            x = {f: rng.uniform(-1, 1) for f in features}
            y = {f: rng.uniform(-1, 1) for f in features}
            t = rng.random()

            # Expected if linear: model(tx + (1-t)y) = t*model(x) + (1-t)*model(y)
            interp = {f: t * x[f] + (1 - t) * y[f] for f in features}
            actual = model_fn(interp)
            expected = t * model_fn(x) + (1 - t) * model_fn(y)
            errors.append(abs(actual - expected))

        max_err = max(errors) if errors else 0
        return max(0.0, 1.0 - min(1.0, max_err * 2))

    def _test_sensitivity(
        self,
        model_fn: Callable,
        features: List[str],
        rng: random.Random,
        n: int,
    ) -> float:
        """Test input sensitivity: how much output changes with perturbation."""
        deltas = []
        for _ in range(n):
            x = {f: rng.uniform(-1, 1) for f in features}
            base = model_fn(x)
            perturbed = {f: x[f] + rng.gauss(0, 0.1) for f in features}
            perturbed_out = model_fn(perturbed)
            deltas.append(abs(perturbed_out - base))

        mean_delta = sum(deltas) / len(deltas) if deltas else 0
        return min(1.0, mean_delta)

    def _test_smoothness(
        self,
        model_fn: Callable,
        features: List[str],
        rng: random.Random,
        n: int,
    ) -> float:
        """Test smoothness: how consistent outputs are for nearby inputs."""
        variances = []
        for _ in range(n):
            center = {f: rng.uniform(-1, 1) for f in features}
            outputs = []
            for _ in range(10):
                perturbed = {f: center[f] + rng.gauss(0, 0.05) for f in features}
                outputs.append(model_fn(perturbed))
            if len(outputs) >= 2:
                mean_out = sum(outputs) / len(outputs)
                var = sum((o - mean_out) ** 2 for o in outputs) / len(outputs)
                variances.append(math.sqrt(var))

        mean_var = sum(variances) / len(variances) if variances else 0
        return max(0.0, 1.0 - min(1.0, mean_var * 5))

    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Compute Pearson correlation coefficient."""
        n = len(x)
        if n < 2:
            return 0.0
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
        std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
        if std_x == 0 or std_y == 0:
            return 0.0
        return max(-1.0, min(1.0, cov / (std_x * std_y)))
