"""Feature attribution and importance analysis for NEXUS explainable AI.

Provides permutation importance, Shapley value approximation, partial dependence,
interaction effects, and feature ranking. Pure Python, no external dependencies.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class FeatureImportance:
    """Single feature importance entry."""
    feature_name: str
    importance_score: float
    direction: str  # "positive", "negative", "neutral"
    contribution: float  # raw contribution value

    def __post_init__(self):
        if self.direction not in ("positive", "negative", "neutral"):
            self.direction = "neutral"
        if self.importance_score < 0:
            self.importance_score = abs(self.importance_score)


@dataclass
class AttributionResult:
    """Complete attribution result for a single prediction."""
    input_features: Dict[str, float]
    attributions: List[FeatureImportance]
    model_output: float
    method: str
    confidence: float = 1.0


class FeatureAttributor:
    """Compute feature attributions using permutation importance and Shapley approximation."""

    def __init__(self, seed: int = 42, n_permutations: int = 50):
        self.seed = seed
        self.n_permutations = n_permutations

    def compute_importance(
        self,
        input_features: Dict[str, float],
        model_output: float,
        model_fn: Callable[[Dict[str, float]], float],
    ) -> AttributionResult:
        """Permutation importance: permute each feature, measure output change."""
        rng = random.Random(self.seed)
        feature_names = list(input_features.keys())
        base_output = model_fn(input_features)
        attributions = []

        for _ in range(self.n_permutations):
            for fname in feature_names:
                original_val = input_features[fname]
                # Perturb: sample near original or random in [-1, 1]
                perturbed_val = original_val + rng.gauss(0, max(0.1, abs(original_val) * 0.1 + 0.01))
                perturbed = dict(input_features)
                perturbed[fname] = perturbed_val
                perturbed_output = model_fn(perturbed)
                delta = abs(perturbed_output - base_output)
                direction = "positive" if perturbed_output > base_output else (
                    "negative" if perturbed_output < base_output else "neutral"
                )
                contrib = perturbed_output - base_output
                attributions.append(FeatureImportance(
                    feature_name=fname,
                    importance_score=delta,
                    direction=direction,
                    contribution=contrib,
                ))

        # Aggregate: mean per feature
        aggregated = self.aggregate_per_feature(attributions, feature_names)
        total = sum(a.importance_score for a in aggregated) or 1.0
        for a in aggregated:
            a.importance_score /= total

        return AttributionResult(
            input_features=input_features,
            attributions=aggregated,
            model_output=model_output,
            method="permutation_importance",
            confidence=self._compute_confidence(aggregated, base_output),
        )

    def compute_shapley_values(
        self,
        feature: str,
        all_features: Dict[str, float],
        model_fn: Callable[[Dict[str, float]], float],
    ) -> List[FeatureImportance]:
        """Approximate Shapley values using marginal contribution sampling."""
        rng = random.Random(self.seed)
        feature_names = [f for f in all_features if f != feature]
        n_samples = min(len(feature_names) * 3, 20)
        contributions = []

        baseline = model_fn({f: 0.0 for f in all_features})
        full_val = model_fn(all_features)

        for _ in range(n_samples):
            # Random subset of other features
            subset_size = rng.randint(0, len(feature_names))
            subset = rng.sample(feature_names, subset_size)
            without_feature = {f: all_features[f] if f in subset else 0.0 for f in all_features}
            without_feature[feature] = 0.0

            with_feature = dict(without_feature)
            with_feature[feature] = all_features[feature]

            marginal = model_fn(with_feature) - model_fn(without_feature)
            contributions.append(marginal)

        avg_contribution = sum(contributions) / len(contributions) if contributions else 0.0
        direction = "positive" if avg_contribution > 0 else ("negative" if avg_contribution < 0 else "neutral")

        return [FeatureImportance(
            feature_name=feature,
            importance_score=abs(avg_contribution),
            direction=direction,
            contribution=avg_contribution,
        )]

    def compute_partial_dependence(
        self,
        feature: str,
        values: List[float],
        model_fn: Callable[[Dict[str, float]], float],
        other_features: Dict[str, float],
    ) -> List[Tuple[float, float]]:
        """Compute partial dependence: average model output across feature values."""
        results = []
        for val in values:
            feats = dict(other_features)
            feats[feature] = val
            output = model_fn(feats)
            results.append((val, output))
        return results

    def rank_features(
        self,
        attributions: List[FeatureImportance],
        top_k: Optional[int] = None,
    ) -> List[FeatureImportance]:
        """Rank features by importance score (descending)."""
        ranked = sorted(attributions, key=lambda a: a.importance_score, reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked

    def compute_interaction_effects(
        self,
        feature_a: str,
        feature_b: str,
        model_fn: Callable[[Dict[str, float]], float],
        other_features: Dict[str, float],
    ) -> float:
        """Compute interaction score between two features using H-statistic approximation."""
        rng = random.Random(self.seed)
        n_samples = 30
        interaction_sum = 0.0

        # Sample values for both features
        all_features_keys = list(other_features.keys()) + [feature_a, feature_b]
        vals_a = [rng.uniform(-1, 1) for _ in range(n_samples)]
        vals_b = [rng.uniform(-1, 1) for _ in range(n_samples)]

        base_output = model_fn(other_features)

        for va, vb in zip(vals_a, vals_b):
            feats_ab = dict(other_features)
            feats_ab[feature_a] = va
            feats_ab[feature_b] = vb
            out_ab = model_fn(feats_ab)

            feats_a = dict(other_features)
            feats_a[feature_a] = va
            out_a = model_fn(feats_a)

            feats_b = dict(other_features)
            feats_b[feature_b] = vb
            out_b = model_fn(feats_b)

            # Interaction: deviation from additivity
            interaction = abs(out_ab - out_a - out_b + base_output)
            interaction_sum += interaction

        return interaction_sum / n_samples if n_samples > 0 else 0.0

    def aggregate_attributions(
        self,
        results: List[AttributionResult],
    ) -> List[FeatureImportance]:
        """Aggregate multiple attribution results into mean importance per feature."""
        feature_scores: Dict[str, List[float]] = {}
        feature_contribs: Dict[str, List[float]] = {}

        for result in results:
            for attr in result.attributions:
                feature_scores.setdefault(attr.feature_name, []).append(attr.importance_score)
                feature_contribs.setdefault(attr.feature_name, []).append(attr.contribution)

        aggregated = []
        for fname in feature_scores:
            scores = feature_scores[fname]
            contribs = feature_contribs[fname]
            avg_score = sum(scores) / len(scores)
            avg_contrib = sum(contribs) / len(contribs)
            direction = "positive" if avg_contrib > 0 else ("negative" if avg_contrib < 0 else "neutral")
            aggregated.append(FeatureImportance(
                feature_name=fname,
                importance_score=avg_score,
                direction=direction,
                contribution=avg_contrib,
            ))
        return aggregated

    # --- private helpers ---

    def aggregate_per_feature(
        self,
        attributions: List[FeatureImportance],
        feature_names: List[str],
    ) -> List[FeatureImportance]:
        """Group attributions by feature name, compute mean."""
        grouped: Dict[str, List[FeatureImportance]] = {f: [] for f in feature_names}
        for attr in attributions:
            grouped[attr.feature_name].append(attr)

        result = []
        for fname, attrs in grouped.items():
            if not attrs:
                continue
            avg_score = sum(a.importance_score for a in attrs) / len(attrs)
            avg_contrib = sum(a.contribution for a in attrs) / len(attrs)
            pos_count = sum(1 for a in attrs if a.direction == "positive")
            neg_count = sum(1 for a in attrs if a.direction == "negative")
            if pos_count > neg_count:
                direction = "positive"
            elif neg_count > pos_count:
                direction = "negative"
            else:
                direction = "neutral"
            result.append(FeatureImportance(
                feature_name=fname,
                importance_score=avg_score,
                direction=direction,
                contribution=avg_contrib,
            ))
        return result

    def _compute_confidence(
        self,
        attributions: List[FeatureImportance],
        base_output: float,
    ) -> float:
        """Heuristic confidence based on attribution distribution."""
        if not attributions:
            return 0.5
        total = sum(a.importance_score for a in attributions)
        if total == 0:
            return 0.5
        # Higher concentration = higher confidence
        max_score = max(a.importance_score for a in attributions)
        concentration = max_score / total
        return min(1.0, 0.3 + 0.7 * concentration)
