"""Counterfactual explanation generation for NEXUS explainable AI.

Generates 'what-if' explanations showing minimal input changes
needed to achieve a different model output. Pure Python, no external deps.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class CounterfactualExample:
    """A single counterfactual explanation."""
    original_input: Dict[str, float]
    modified_input: Dict[str, float]
    original_output: float
    counterfactual_output: float
    changed_features: List[str]
    plausibility: float  # 0.0 to 1.0

    def __post_init__(self):
        self.plausibility = max(0.0, min(1.0, self.plausibility))


class CounterfactualGenerator:
    """Generate counterfactual explanations for model predictions."""

    def __init__(self, seed: int = 42, step_size: float = 0.1, max_iterations: int = 100):
        self.seed = seed
        self.step_size = step_size
        self.max_iterations = max_iterations

    def generate(
        self,
        input_data: Dict[str, float],
        target_output: float,
        model_fn: Callable[[Dict[str, float]], float],
        max_changes: int = 5,
    ) -> CounterfactualExample:
        """Generate a counterfactual: modify input to reach target output."""
        rng = random.Random(self.seed)
        current = dict(input_data)
        original_output = model_fn(input_data)
        feature_names = list(input_data.keys())
        changed: List[str] = []

        for _ in range(self.max_iterations):
            if len(changed) >= max_changes:
                break
            current_output = model_fn(current)
            if self._reached_target(current_output, target_output):
                break

            # Find the feature whose perturbation moves output closest to target
            best_feature = None
            best_dist = abs(current_output - target_output)

            for fname in feature_names:
                for delta_sign in [1, -1]:
                    candidate = dict(current)
                    candidate[fname] += delta_sign * self.step_size
                    candidate_output = model_fn(candidate)
                    dist = abs(candidate_output - target_output)
                    if dist < best_dist:
                        best_dist = dist
                        best_feature = (fname, delta_sign)

            if best_feature is None:
                break

            fname, sign = best_feature
            current[fname] += sign * self.step_size
            if fname not in changed:
                changed.append(fname)

        cf_output = model_fn(current)
        plausibility = self._compute_simple_plausibility(input_data, current)

        return CounterfactualExample(
            original_input=dict(input_data),
            modified_input=current,
            original_output=original_output,
            counterfactual_output=cf_output,
            changed_features=changed,
            plausibility=plausibility,
        )

    def find_minimum_changes(
        self,
        input_data: Dict[str, float],
        target_output: float,
        model_fn: Callable[[Dict[str, float]], float],
    ) -> CounterfactualExample:
        """Find the minimum number of feature changes to reach target."""
        feature_names = list(input_data.keys())
        original_output = model_fn(input_data)

        # Try increasing numbers of changed features
        for n_changes in range(1, len(feature_names) + 1):
            # Try all combinations of n_changes features
            result = self._try_n_changes(
                input_data, target_output, model_fn, feature_names, n_changes
            )
            if result is not None:
                cf_output = model_fn(result)
                plausibility = self._compute_simple_plausibility(input_data, result)
                changed = [f for f in feature_names if result[f] != input_data[f]]
                return CounterfactualExample(
                    original_input=dict(input_data),
                    modified_input=result,
                    original_output=original_output,
                    counterfactual_output=cf_output,
                    changed_features=changed,
                    plausibility=plausibility,
                )

        # If no solution found, return best-effort with all features
        cf = self.generate(input_data, target_output, model_fn, max_changes=len(feature_names))
        return cf

    def compute_plausibility(
        self,
        original: Dict[str, float],
        counterfactual: Dict[str, float],
        dataset_stats: Dict[str, Dict[str, float]],
    ) -> float:
        """Compute plausibility score based on dataset statistics.

        dataset_stats: {feature_name: {"mean": ..., "std": ...}}
        """
        if not dataset_stats:
            return self._compute_simple_plausibility(original, counterfactual)

        total_distance = 0.0
        n_features = 0

        for fname, cf_val in counterfactual.items():
            orig_val = original.get(fname, cf_val)
            change = abs(cf_val - orig_val)
            if fname in dataset_stats:
                stats = dataset_stats[fname]
                std = stats.get("std", 1.0)
                if std > 0:
                    normalized_change = change / std
                else:
                    normalized_change = change
                total_distance += normalized_change
                n_features += 1

        if n_features == 0:
            return 0.5

        avg_distance = total_distance / n_features
        # Plausibility decreases with distance; 1.0 = no change, 0.0 = far away
        plausibility = 1.0 / (1.0 + avg_distance)
        return max(0.0, min(1.0, plausibility))

    def generate_diverse_explanations(
        self,
        input_data: Dict[str, float],
        target: float,
        model_fn: Callable[[Dict[str, float]], float],
        num_explanations: int = 5,
    ) -> List[CounterfactualExample]:
        """Generate diverse counterfactual explanations with different seeds."""
        results = []
        for i in range(num_explanations):
            gen = CounterfactualGenerator(
                seed=self.seed + i * 1000 + 1,
                step_size=self.step_size * (0.5 + i * 0.25),
                max_iterations=self.max_iterations,
            )
            cf = gen.generate(input_data, target, model_fn)
            results.append(cf)
        return results

    def evaluate_counterfactual(
        self,
        cf: CounterfactualExample,
        model_fn: Callable[[Dict[str, float]], float],
    ) -> Dict[str, Any]:
        """Evaluate a counterfactual: validity, distance, sparsity."""
        # Verify counterfactual output matches model prediction
        actual_cf_output = model_fn(cf.modified_input)
        output_matches = abs(actual_cf_output - cf.counterfactual_output) < 0.01
        original_matches = abs(model_fn(cf.original_input) - cf.original_output) < 0.01

        # Compute feature distance
        total_change = 0.0
        changed_count = 0
        for fname in cf.original_input:
            change = abs(cf.modified_input.get(fname, 0) - cf.original_input.get(fname, 0))
            total_change += change
            if change > 1e-9:
                changed_count += 1

        # Sparsity: fraction of unchanged features
        total_features = len(cf.original_input)
        sparsity = (total_features - changed_count) / total_features if total_features > 0 else 1.0

        return {
            "valid": output_matches and original_matches,
            "output_matches_model": output_matches,
            "original_matches_model": original_matches,
            "total_change": total_change,
            "changed_features": changed_count,
            "sparsity": sparsity,
            "plausibility": cf.plausibility,
        }

    # --- private helpers ---

    def _reached_target(self, current_output: float, target_output: float) -> bool:
        """Check if current output is close enough to target."""
        tolerance = abs(target_output) * 0.05 + 0.01
        return abs(current_output - target_output) < tolerance

    def _compute_simple_plausibility(
        self,
        original: Dict[str, float],
        counterfactual: Dict[str, float],
    ) -> float:
        """Simple plausibility based on L2 distance."""
        total_sq = 0.0
        for fname in original:
            diff = counterfactual.get(fname, 0) - original.get(fname, 0)
            total_sq += diff ** 2
        distance = math.sqrt(total_sq)
        return max(0.0, 1.0 / (1.0 + distance))

    def _try_n_changes(
        self,
        input_data: Dict[str, float],
        target_output: float,
        model_fn: Callable,
        feature_names: List[str],
        n_changes: int,
    ) -> Optional[Dict[str, float]]:
        """Try to find counterfactual by modifying exactly n_changes features."""
        # Use greedy approach: for each subset size, try different perturbation scales
        from itertools import combinations

        for combo in combinations(feature_names, n_changes):
            current = dict(input_data)
            found = False
            for _ in range(self.max_iterations):
                output = model_fn(current)
                if self._reached_target(output, target_output):
                    return current

                # Perturb the selected features
                for fname in combo:
                    # Gradient-like: try moving toward target
                    for delta_sign in [1, -1]:
                        candidate = dict(current)
                        candidate[fname] += delta_sign * self.step_size
                        candidate_output = model_fn(candidate)
                        if abs(candidate_output - target_output) < abs(output - target_output):
                            current[fname] = candidate[fname]
                            found = True
                            break

                if not found:
                    break

        return None
