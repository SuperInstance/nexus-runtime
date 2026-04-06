"""Bayesian estimation utilities for sensor fusion.

Pure Python — no external dependencies. Implements Gaussian conjugate updates,
sequential filtering, and evidence computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class BayesEstimate:
    """Result of a Bayesian estimation step."""
    mean: float
    variance: float
    confidence_interval: Tuple[float, float]
    samples: int = 1


class BayesianEstimator:
    """Gaussian Bayesian estimator using conjugate priors.

    Implements the standard Gaussian-Gaussian conjugate update:
        posterior_precision = prior_precision + likelihood_precision
        posterior_mean = (prior_precision * prior_mean + likelihood_precision * obs_mean) / posterior_precision
    """

    def update_prior(
        self,
        prior_mean: float,
        prior_var: float,
        obs_mean: float,
        obs_var: float,
    ) -> Tuple[float, float]:
        """Gaussian conjugate update — alias for compute_posterior."""
        return self.compute_posterior(prior_mean, prior_var, obs_mean, obs_var)

    def compute_posterior(
        self,
        prior_mean: float,
        prior_var: float,
        obs_mean: float,
        obs_var: float,
    ) -> Tuple[float, float]:
        """Compute posterior mean and variance (Gaussian conjugate).

        Returns (posterior_mean, posterior_variance).
        """
        prior_prec = 1.0 / prior_var if prior_var > 1e-15 else 1e15
        obs_prec = 1.0 / obs_var if obs_var > 1e-15 else 1e15
        post_prec = prior_prec + obs_prec
        post_var = 1.0 / post_prec if post_prec > 1e-15 else 1e15
        post_mean = (prior_prec * prior_mean + obs_prec * obs_mean) / post_prec
        return (post_mean, post_var)

    def compute_confidence(
        self,
        posterior_mean: float,
        posterior_var: float,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float]:
        """Compute confidence interval for a Gaussian posterior.

        Uses the standard normal z-score approximation.
        Returns (lower, upper).
        """
        z = self._z_score(confidence_level)
        std = math.sqrt(max(posterior_var, 0.0))
        half_width = z * std
        return (posterior_mean - half_width, posterior_mean + half_width)

    def sequential_update(
        self,
        prior_mean: float,
        prior_var: float,
        observations: List[float],
        obs_variances: Optional[List[float]] = None,
    ) -> BayesEstimate:
        """Sequentially update posterior with multiple observations.

        If obs_variances is None, all observations share the prior's implicit
        observation variance (set to prior_var as default).
        """
        post_mean = prior_mean
        post_var = prior_var
        n = len(observations)
        for i, obs in enumerate(observations):
            ov = obs_variances[i] if obs_variances is not None and i < len(obs_variances) else post_var
            post_mean, post_var = self.compute_posterior(post_mean, post_var, obs, ov)
        ci = self.compute_confidence(post_mean, post_var)
        return BayesEstimate(
            mean=post_mean,
            variance=post_var,
            confidence_interval=ci,
            samples=n,
        )

    def predict(
        self,
        post_mean: float,
        post_var: float,
        process_noise: float,
    ) -> Tuple[float, float]:
        """Predictive distribution: prior for next time step.

        Adds process_noise to the posterior variance.
        Returns (predicted_mean, predicted_variance).
        """
        return (post_mean, post_var + process_noise)

    def compute_evidence(
        self,
        obs_mean: float,
        obs_var: float,
        prior_mean: float,
        prior_var: float,
    ) -> float:
        """Compute marginal likelihood (evidence) p(z|model).

        p(z) = N(z; prior_mean, prior_var + obs_var)
        """
        total_var = prior_var + obs_var
        if total_var < 1e-15:
            total_var = 1e-15
        # Log evidence
        log_evidence = -0.5 * math.log(2.0 * math.pi * total_var) \
                       - 0.5 * ((obs_mean - prior_mean) ** 2) / total_var
        return math.exp(log_evidence)

    # -- helper ------------------------------------------------------------

    @staticmethod
    def _z_score(confidence_level: float) -> float:
        """Approximate z-score for a given confidence level.

        Uses a rational approximation of the inverse normal CDF.
        """
        if confidence_level <= 0.0:
            return 0.0
        if confidence_level >= 1.0:
            return 10.0

        # Map confidence to two-tailed p
        p = (1.0 + confidence_level) / 2.0

        # Abramowitz and Stegun approximation 26.2.23
        if p < 0.5:
            p = 1.0 - p

        t = math.sqrt(-2.0 * math.log(1.0 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
