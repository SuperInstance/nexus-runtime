"""Utility theory: exponential/power utility, certainty equivalents, risk premiums."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class RiskAttitude(Enum):
    """Enum for risk attitude classification."""
    RISK_AVERSE = "risk_averse"
    RISK_NEUTRAL = "risk_neutral"
    RISK_SEEKING = "risk_seeking"


@dataclass
class UtilityFunction:
    """A utility function specification."""
    type: str  # "exponential", "power", "linear", "logarithmic"
    parameters: Dict[str, float] = field(default_factory=dict)
    domain: Tuple[float, float] = (0.0, float("inf"))


class UtilityTheory:
    """Core utility theory computations."""

    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------

    @staticmethod
    def exponential_utility(value: float, risk_aversion: float = 1.0) -> float:
        """Exponential utility: U(x) = 1 - exp(-risk_aversion * x).

        Appropriate for constant absolute risk aversion (CARA).
        """
        return 1.0 - math.exp(-risk_aversion * value)

    @staticmethod
    def power_utility(value: float, risk_aversion: float = 0.5) -> float:
        """Power (CRRA) utility.

        U(x) = x^(1-risk_aversion) / (1-risk_aversion)  for risk_aversion != 1
        U(x) = ln(x)  for risk_aversion == 1
        """
        if risk_aversion == 1.0:
            if value <= 0:
                return float("-inf")
            return math.log(value)
        if value <= 0 and (1.0 - risk_aversion) > 0:
            return float("-inf")
        coefficient = 1.0 - risk_aversion
        if coefficient == 0:
            return math.log(value) if value > 0 else float("-inf")
        return (value ** coefficient) / coefficient

    # ------------------------------------------------------------------
    # Certainty equivalent & risk premium
    # ------------------------------------------------------------------

    @staticmethod
    def compute_certainty_equivalent(
        utility_fn: Callable[[float], float],
        outcomes: List[float],
        probabilities: List[float],
    ) -> float:
        """Compute the certainty equivalent of a lottery.

        CE: U(CE) = E[U(X)]  →  CE = U^{-1}(E[U(X)])

        Uses numerical bisection to invert the utility function.
        """
        if not outcomes or not probabilities:
            return 0.0

        total_p = sum(probabilities)
        if abs(total_p - 1.0) > 1e-6:
            probabilities = [p / total_p for p in probabilities]

        # Expected utility
        eu = sum(utility_fn(o) * p for o, p in zip(outcomes, probabilities))

        # Binary search for CE such that U(CE) ≈ EU
        lo = min(outcomes) - 1.0
        hi = max(outcomes) + 1.0
        last_mid = (lo + hi) / 2.0

        for _ in range(100):
            mid = (lo + hi) / 2.0
            last_mid = mid
            u_mid = utility_fn(mid)
            if u_mid < eu:
                lo = mid
            else:
                hi = mid
            if abs(u_mid - eu) < 1e-10:
                return mid

        return last_mid

    @staticmethod
    def compute_risk_premium(
        certainty_equivalent: float,
        expected_value: float,
    ) -> float:
        """Compute the risk premium: π = EV - CE.

        Positive π indicates risk aversion.
        """
        return expected_value - certainty_equivalent

    # ------------------------------------------------------------------
    # Lottery comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare_lotteries(
        lottery_a: Tuple[List[float], List[float]],
        lottery_b: Tuple[List[float], List[float]],
        risk_attitude: RiskAttitude = RiskAttitude.RISK_NEUTRAL,
    ) -> str:
        """Compare two lotteries given a risk attitude.

        Each lottery is (outcomes, probabilities).
        Returns "a", "b", or "tie".
        """
        outcomes_a, probs_a = lottery_a
        outcomes_b, probs_b = lottery_b

        # Choose utility function based on risk attitude
        if risk_attitude == RiskAttitude.RISK_AVERSE:
            u = UtilityTheory.exponential_utility
        elif risk_attitude == RiskAttitude.RISK_SEEKING:
            u = lambda x: -UtilityTheory.exponential_utility(x)
        else:
            u = lambda x: x  # Risk neutral: U(x) = x

        total_pa = sum(probs_a)
        total_pb = sum(probs_b)
        if abs(total_pa - 1.0) > 1e-6:
            probs_a = [p / total_pa for p in probs_a]
        if abs(total_pb - 1.0) > 1e-6:
            probs_b = [p / total_pb for p in probs_b]

        eu_a = sum(u(o) * p for o, p in zip(outcomes_a, probs_a))
        eu_b = sum(u(o) * p for o, p in zip(outcomes_b, probs_b))

        tol = 1e-9
        if eu_a > eu_b + tol:
            return "a"
        elif eu_b > eu_a + tol:
            return "b"
        return "tie"

    # ------------------------------------------------------------------
    # Utility function construction from indifference points
    # ------------------------------------------------------------------

    @staticmethod
    def construct_utility_function(
        indifference_points: List[Tuple[float, float]],
    ) -> UtilityFunction:
        """Construct a utility function from indifference points.

        Each point (x, p) means the decision maker is indifferent between:
          - receiving x for sure, and
          - a lottery with probability p of the best outcome and
            (1-p) of the worst outcome.

        Uses least-squares fitting to an exponential function.
        Returns a UtilityFunction with fitted parameters.
        """
        if len(indifference_points) < 2:
            return UtilityFunction(type="linear", parameters={"slope": 1.0})

        # Extract x values and utility values
        xs = [pt[0] for pt in indifference_points]
        us = [pt[1] for pt in indifference_points]

        # Try to fit exponential: U(x) = a * (1 - exp(-b * (x - c)))
        # Simplified: U(x) = 1 - exp(-b * x)
        # Fit b using log: -ln(1-U) = b*x

        valid_points = []
        for x, u in zip(xs, us):
            if u < 1.0 - 1e-9:
                valid_points.append((x, u))
            elif u >= 1.0:
                valid_points.append((x, 0.9999))

        if len(valid_points) < 2:
            # Fall back to linear
            n = len(xs)
            sum_x = sum(xs)
            sum_u = sum(us)
            sum_xu = sum(x * u for x, u in zip(xs, us))
            sum_xx = sum(x * x for x in xs)
            denom = n * sum_xx - sum_x * sum_x
            if abs(denom) < 1e-12:
                return UtilityFunction(type="linear", parameters={"slope": 1.0})
            slope = (n * sum_xu - sum_x * sum_u) / denom
            return UtilityFunction(type="linear", parameters={"slope": slope})

        # Fit exponential: U(x) = 1 - exp(-b*x)
        # → ln(1-U(x)) = -b*x
        # → b = -mean(ln(1-U_i) / x_i)
        b_values = []
        for x, u in valid_points:
            if abs(x) > 1e-9:
                val = -math.log(1.0 - u) / x
                b_values.append(val)

        if not b_values:
            return UtilityFunction(type="linear", parameters={"slope": 1.0})

        b = sum(b_values) / len(b_values)
        if b <= 0:
            b = 0.01  # Small positive risk aversion

        # Compute R-squared for goodness of fit
        ss_res = 0.0
        ss_tot = 0.0
        mean_u = sum(us) / len(us)
        for x, u in zip(xs, us):
            predicted = 1.0 - math.exp(-b * x)
            ss_res += (u - predicted) ** 2
            ss_tot += (u - mean_u) ** 2

        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # If exponential fit is poor, use linear
        if r_squared < 0.5:
            n = len(xs)
            sum_x = sum(xs)
            sum_u = sum(us)
            sum_xu = sum(x * u for x, u in zip(xs, us))
            sum_xx = sum(x * x for x in xs)
            denom = n * sum_xx - sum_x * sum_x
            if abs(denom) < 1e-12:
                return UtilityFunction(type="linear", parameters={"slope": 1.0})
            slope = (n * sum_xu - sum_x * sum_u) / denom
            return UtilityFunction(type="linear", parameters={"slope": slope})

        domain_min = min(xs) if xs else 0.0
        domain_max = max(xs) if xs else 1.0

        return UtilityFunction(
            type="exponential",
            parameters={"risk_aversion": b, "r_squared": r_squared},
            domain=(domain_min, domain_max),
        )
