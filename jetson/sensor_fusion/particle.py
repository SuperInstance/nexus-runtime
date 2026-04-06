"""Particle filter for non-linear / non-Gaussian state estimation.

Pure Python — no external dependencies.
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class Particle:
    """A single particle with state and weight."""
    state: List[float]
    weight: float = 1.0


class ParticleFilter:
    """Sequential Monte Carlo (particle) filter.

    Parameters
    ----------
    num_particles : int
        Number of particles to maintain.
    state_dim : int
        Dimensionality of the state vector.
    """

    def __init__(self, num_particles: int, state_dim: int) -> None:
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.process_noise_stddev: List[float] = [1.0] * state_dim
        self.particles: List[Particle] = []
        self._init_particles()

    def _init_particles(self) -> None:
        """Initialize particles uniformly around zero."""
        self.particles = [
            Particle(state=[0.0] * self.state_dim, weight=1.0 / self.num_particles)
            for _ in range(self.num_particles)
        ]

    def initialize_particles(self, initial_states: List[List[float]]) -> None:
        """Set particles to specific initial states with uniform weights."""
        if len(initial_states) != self.num_particles:
            raise ValueError(
                f"Expected {self.num_particles} initial states, got {len(initial_states)}"
            )
        for i, s in enumerate(initial_states):
            if len(s) != self.state_dim:
                raise ValueError(
                    f"Particle {i} has dimension {len(s)}, expected {self.state_dim}"
                )
        self.particles = [
            Particle(state=s[:], weight=1.0 / self.num_particles)
            for s in initial_states
        ]

    # -- public API --------------------------------------------------------

    def predict(self, dt: float, process_fn: Optional[Callable[[List[float], float], List[float]]] = None) -> None:
        """Propagate each particle through the process model + noise."""
        for p in self.particles:
            if process_fn is not None:
                p.state = process_fn(p.state, dt)
            # Add process noise
            for j in range(self.state_dim):
                p.state[j] += random.gauss(0.0, self.process_noise_stddev[j] * math.sqrt(max(dt, 0.0)))

    def update(self, measurement: List[float], likelihood_fn: Callable[[List[float], List[float]], float]) -> None:
        """Update particle weights based on measurement likelihood."""
        total = 0.0
        for p in self.particles:
            p.weight *= likelihood_fn(p.state, measurement)
            total += p.weight
        if total > 1e-300:
            for p in self.particles:
                p.weight /= total
        else:
            # Degeneracy guard: redistribute uniformly
            for p in self.particles:
                p.weight = 1.0 / self.num_particles

    def resample(self, method: str = 'systematic') -> None:
        """Resample particles to avoid degeneracy.

        Supported methods: 'systematic', 'multinomial', 'stratified'.
        """
        self.normalize_weights()
        weights = [p.weight for p in self.particles]

        if method == 'systematic':
            indices = self._systematic_resample(weights)
        elif method == 'multinomial':
            indices = self._multinomial_resample(weights)
        elif method == 'stratified':
            indices = self._stratified_resample(weights)
        else:
            raise ValueError(f"Unknown resampling method: {method}")

        new_particles = [Particle(state=self.particles[i].state[:],
                                  weight=1.0 / self.num_particles)
                         for i in indices]
        self.particles = new_particles

    def get_state_estimate(self) -> dict:
        """Compute weighted mean and covariance of particles."""
        mean = [0.0] * self.state_dim
        for p in self.particles:
            for j in range(self.state_dim):
                mean[j] += p.weight * p.state[j]

        cov = self._compute_covariance(mean)
        return {"mean": mean, "covariance": cov}

    def _compute_covariance(self, mean: List[float]) -> List[List[float]]:
        """Compute weighted covariance matrix."""
        cov = [[0.0] * self.state_dim for _ in range(self.state_dim)]
        for p in self.particles:
            diff = [p.state[j] - mean[j] for j in range(self.state_dim)]
            for i in range(self.state_dim):
                for j in range(self.state_dim):
                    cov[i][j] += p.weight * diff[i] * diff[j]
        return cov

    def get_effective_particles(self) -> float:
        """Compute the effective number of particles (N_eff)."""
        sum_w2 = sum(p.weight ** 2 for p in self.particles)
        if sum_w2 < 1e-300:
            return 0.0
        return 1.0 / sum_w2

    def set_process_noise(self, stddev: List[float]) -> None:
        """Set process noise standard deviations per dimension."""
        if len(stddev) != self.state_dim:
            raise ValueError(
                f"Expected {self.state_dim} stddev values, got {len(stddev)}"
            )
        self.process_noise_stddev = stddev[:]

    def normalize_weights(self) -> None:
        """Normalize particle weights so they sum to 1."""
        total = sum(p.weight for p in self.particles)
        if total > 1e-300:
            for p in self.particles:
                p.weight /= total
        else:
            for p in self.particles:
                p.weight = 1.0 / self.num_particles

    # -- resampling strategies ---------------------------------------------

    @staticmethod
    def _cumulative_sum(weights: List[float]) -> List[float]:
        cs = [0.0] * len(weights)
        cs[0] = weights[0]
        for i in range(1, len(weights)):
            cs[i] = cs[i - 1] + weights[i]
        return cs

    def _systematic_resample(self, weights: List[float]) -> List[int]:
        """Systematic resampling."""
        N = len(weights)
        cs = self._cumulative_sum(weights)
        u0 = random.random() / N
        indices = []
        j = 0
        for i in range(N):
            u = u0 + i / N
            while j < N - 1 and u > cs[j]:
                j += 1
            indices.append(j)
        return indices

    def _multinomial_resample(self, weights: List[float]) -> List[int]:
        """Multinomial resampling."""
        N = len(weights)
        cs = self._cumulative_sum(weights)
        indices = []
        for _ in range(N):
            u = random.random()
            j = 0
            while j < N - 1 and u > cs[j]:
                j += 1
            indices.append(j)
        return indices

    def _stratified_resample(self, weights: List[float]) -> List[int]:
        """Stratified resampling."""
        N = len(weights)
        cs = self._cumulative_sum(weights)
        indices = []
        j = 0
        for i in range(N):
            u = (i + random.random()) / N
            while j < N - 1 and u > cs[j]:
                j += 1
            indices.append(j)
        return indices
