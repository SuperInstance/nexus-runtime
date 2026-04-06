"""Decision under uncertainty: expected values, VaR, minimax, info gain, sensitivity."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class UncertainValue:
    """A value with uncertainty metadata."""
    mean: float = 0.0
    std_dev: float = 0.0
    distribution_type: str = "normal"  # normal, uniform, triangular
    confidence: float = 0.95


@dataclass
class DecisionScenario:
    """A decision scenario with alternatives, outcomes, and uncertainties."""
    alternatives: List[str] = field(default_factory=list)
    outcomes: List[List[float]] = field(default_factory=list)
    probabilities: List[float] = field(default_factory=list)
    uncertainties: List[UncertainValue] = field(default_factory=list)


class UncertaintyManager:
    """Manages decision-making under uncertainty."""

    # ------------------------------------------------------------------
    # Expected value & utility
    # ------------------------------------------------------------------

    def compute_expected_value(
        self,
        outcomes: List[float],
        probabilities: List[float],
    ) -> float:
        """Compute E[X] = Σ p_i * x_i."""
        if len(outcomes) != len(probabilities):
            raise ValueError("Outcomes and probabilities must have equal length")
        total_p = sum(probabilities)
        if abs(total_p - 1.0) > 1e-6:
            # Normalize
            probabilities = [p / total_p for p in probabilities]
        return sum(o * p for o, p in zip(outcomes, probabilities))

    def compute_expected_utility(
        self,
        outcomes: List[float],
        probabilities: List[float],
        utility_fn: Callable[[float], float],
    ) -> float:
        """Compute E[U(X)] = Σ p_i * U(x_i)."""
        if len(outcomes) != len(probabilities):
            raise ValueError("Outcomes and probabilities must have equal length")
        total_p = sum(probabilities)
        if abs(total_p - 1.0) > 1e-6:
            probabilities = [p / total_p for p in probabilities]
        return sum(utility_fn(o) * p for o, p in zip(outcomes, probabilities))

    # ------------------------------------------------------------------
    # Risk measures
    # ------------------------------------------------------------------

    def compute_value_at_risk(
        self,
        outcomes: List[float],
        confidence: float = 0.95,
    ) -> float:
        """Compute Value at Risk (VaR) at the given confidence level.

        VaR is the worst outcome not exceeded with probability *confidence*.
        For a list of outcomes with equal probability, this is a quantile.
        """
        if not outcomes:
            return 0.0
        sorted_outcomes = sorted(outcomes)
        n = len(sorted_outcomes)
        # Quantile index
        index = math.ceil((1.0 - confidence) * n) - 1
        index = max(0, min(index, n - 1))
        return sorted_outcomes[index]

    def minimax_regret(
        self,
        alternatives: List[List[float]],
        scenarios: Optional[List[str]] = None,
    ) -> int:
        """Select the alternative that minimizes the maximum regret.

        *alternatives*: list of alternatives, each being a list of payoffs
        per scenario (all alternatives must have the same number of scenarios).

        Returns the index of the best alternative.
        """
        if not alternatives:
            return -1
        n_scenarios = len(alternatives[0])
        if n_scenarios == 0:
            return 0

        # Find best outcome in each scenario
        best_in_scenario: List[float] = []
        for s in range(n_scenarios):
            col = [alt[s] for alt in alternatives if len(alt) > s]
            best_in_scenario.append(max(col) if col else 0.0)

        # Compute regret for each alternative
        regrets: List[float] = []
        for alt in alternatives:
            regret = 0.0
            for s in range(len(alt)):
                regret = max(regret, best_in_scenario[s] - alt[s])
            regrets.append(regret)

        # Pick the one with minimum maximum regret
        min_regret = min(regrets)
        return regrets.index(min_regret)

    # ------------------------------------------------------------------
    # Information gain
    # ------------------------------------------------------------------

    def compute_info_gain(
        self,
        current_belief: List[float],
        potential_observation: List[List[float]],
    ) -> float:
        """Compute expected information gain from a potential observation.

        *current_belief*: prior probability distribution over states.
        *potential_observation*: likelihood matrix, where
            potential_observation[i][j] = P(obs=i | state=j).

        Returns expected KL-divergence (information gain in nats).
        """
        n_states = len(current_belief)
        if n_states == 0:
            return 0.0

        n_obs = len(potential_observation)
        if n_obs == 0:
            return 0.0

        # Normalize belief
        total_belief = sum(current_belief)
        if total_belief <= 0:
            return 0.0
        belief = [b / total_belief for b in current_belief]

        expected_kl = 0.0
        for o in range(n_obs):
            # P(obs=o) = Σ_j P(obs=o|state=j) * P(state=j)
            p_obs = sum(
                potential_observation[o][j] * belief[j] for j in range(n_states)
                if j < len(potential_observation[o])
            )
            if p_obs <= 0:
                continue

            # Posterior: P(state=j | obs=o)
            posterior: List[float] = []
            valid = True
            for j in range(n_states):
                if j < len(potential_observation[o]):
                    post = potential_observation[o][j] * belief[j] / p_obs
                else:
                    post = 0.0
                posterior.append(post)

            # KL divergence: D_KL(posterior || belief)
            kl = 0.0
            for j in range(n_states):
                if posterior[j] > 0 and belief[j] > 0:
                    kl += posterior[j] * math.log(posterior[j] / belief[j])
            expected_kl += p_obs * kl

        return max(0.0, expected_kl)

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        decision: Callable[[Dict[str, float]], float],
        parameter_ranges: Dict[str, Tuple[float, float]],
        base_values: Optional[Dict[str, float]] = None,
        num_samples: int = 100,
    ) -> List[Tuple[str, float]]:
        """Perform one-at-a-time sensitivity analysis.

        Varies each parameter across its range (while holding others at
        base values) and measures the change in the decision output.

        Returns parameters sorted by sensitivity (most sensitive first).
        """
        if base_values is None:
            base_values = {}
            for param, (lo, hi) in parameter_ranges.items():
                base_values[param] = (lo + hi) / 2.0

        base_result = decision(base_values)

        sensitivities: List[Tuple[str, float]] = []
        for param, (lo, hi) in parameter_ranges.items():
            if hi == lo:
                sensitivities.append((param, 0.0))
                continue

            # Evaluate at lo and hi
            params_lo = dict(base_values)
            params_lo[param] = lo
            params_hi = dict(base_values)
            params_hi[param] = hi

            result_lo = decision(params_lo)
            result_hi = decision(params_hi)
            variation = abs(result_hi - result_lo)
            range_size = abs(hi - lo)

            # Normalized sensitivity: change in output per unit change in param
            sensitivity = variation / range_size if range_size > 0 else 0.0
            sensitivities.append((param, sensitivity))

        sensitivities.sort(key=lambda x: -x[1])
        return sensitivities
