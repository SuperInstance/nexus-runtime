"""
Experience replay management: FIFO, prioritized, and episode buffers.
"""

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Transition:
    """Single transition tuple with optional priority."""
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool
    priority: float = 1.0


class ReplayBuffer:
    """FIFO replay buffer with fixed capacity."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)

    def add(self, state: Any, action: Any, reward: float, next_state: Any, done: bool):
        t = Transition(state=state, action=action, reward=reward,
                       next_state=next_state, done=done)
        self._buffer.append(t)

    def add_transition(self, t: Transition):
        self._buffer.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        n = min(batch_size, len(self._buffer))
        return random.sample(list(self._buffer), n)

    def __len__(self) -> int:
        return len(self._buffer)

    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    def clear(self):
        self._buffer.clear()

    def get_all(self) -> List[Transition]:
        return list(self._buffer)

    def can_sample(self, batch_size: int) -> bool:
        return len(self._buffer) >= batch_size


class PrioritizedReplayBuffer:
    """Prioritized experience replay with importance sampling weights."""

    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4,
                 beta_anneal: float = 0.001, eps: float = 1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_anneal = beta_anneal
        self.eps = eps
        self._buffer: List[Transition] = []
        self._priorities: List[float] = []
        self._max_priority = 1.0
        self._pos = 0

    def add(self, state: Any, action: Any, reward: float, next_state: Any, done: bool,
            priority: Optional[float] = None):
        if priority is None:
            priority = self._max_priority
        t = Transition(state=state, action=action, reward=reward,
                       next_state=next_state, done=done, priority=priority)
        if len(self._buffer) < self.capacity:
            self._buffer.append(t)
            self._priorities.append(priority)
        else:
            self._buffer[self._pos] = t
            self._priorities[self._pos] = priority
        self._pos = (self._pos + 1) % self.capacity
        self._max_priority = max(self._max_priority, priority)

    def _get_probabilities(self) -> List[float]:
        if not self._priorities:
            return []
        scaled = [p ** self.alpha for p in self._priorities]
        total = sum(scaled)
        if total == 0:
            return [1.0 / len(scaled)] * len(scaled)
        return [s / total for s in scaled]

    def _get_importance_weights(self, probabilities: List[float], batch_size: int) -> List[float]:
        n = len(self._buffer)
        if n == 0:
            return []
        weights = [(n * p) ** (-self.beta) for p in probabilities]
        max_w = max(weights) if weights else 1.0
        return [w / max_w for w in weights]

    def sample(self, batch_size: int) -> Tuple[List[Transition], List[float], List[int]]:
        """Sample transitions. Returns (transitions, importance_weights, indices)."""
        if not self._buffer:
            return [], [], []
        n = min(batch_size, len(self._buffer))
        probs = self._get_probabilities()
        weights = self._get_importance_weights(probs, n)
        # Weighted random sampling without replacement
        indices = list(range(len(self._buffer)))
        sampled_indices = []
        remaining = list(range(len(self._buffer)))
        remaining_probs = list(probs)
        for _ in range(n):
            total = sum(remaining_probs)
            if total == 0:
                sampled_indices.append(remaining.pop(0))
                remaining_probs.pop(0)
                continue
            r = random.random() * total
            cumulative = 0.0
            for j in range(len(remaining)):
                cumulative += remaining_probs[j]
                if r < cumulative:
                    sampled_indices.append(remaining.pop(j))
                    remaining_probs.pop(j)
                    break
        transitions = [self._buffer[i] for i in sampled_indices]
        sample_weights = [weights[i] for i in sampled_indices]
        return transitions, sample_weights, sampled_indices

    def update_priorities(self, indices: List[int], priorities: List[float]):
        for idx, prio in zip(indices, priorities):
            if 0 <= idx < len(self._priorities):
                self._priorities[idx] = prio
                self._max_priority = max(self._max_priority, prio)

    def anneal_beta(self):
        self.beta = min(1.0, self.beta + self.beta_anneal)

    def __len__(self) -> int:
        return len(self._buffer)

    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    def clear(self):
        self._buffer = []
        self._priorities = []
        self._pos = 0
        self._max_priority = 1.0

    def get_all(self) -> List[Transition]:
        return list(self._buffer)

    def get_priorities(self) -> List[float]:
        return list(self._priorities)

    def get_beta(self) -> float:
        return self.beta


class EpisodeBuffer:
    """Stores complete episodes and computes returns / GAE."""

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._episodes: List[List[Transition]] = []
        self._current_episode: List[Transition] = []

    def add(self, state: Any, action: Any, reward: float, next_state: Any, done: bool):
        t = Transition(state=state, action=action, reward=reward,
                       next_state=next_state, done=done)
        self._current_episode.append(t)
        if done:
            self._episodes.append(list(self._current_episode))
            self._current_episode = []

    def add_transition(self, t: Transition):
        self._current_episode.append(t)
        if t.done:
            self._episodes.append(list(self._current_episode))
            self._current_episode = []

    def compute_returns(self, episode: Optional[List[Transition]] = None) -> List[float]:
        ep = episode or self._current_episode
        returns = []
        g = 0.0
        for t in reversed(ep):
            g = t.reward + self.gamma * g
            returns.insert(0, g)
        return returns

    def compute_gae(self, values: List[float], episode: Optional[List[Transition]] = None) -> List[float]:
        """Compute Generalized Advantage Estimation."""
        ep = episode or self._current_episode
        if len(ep) != len(values):
            return []
        advantages = []
        gae = 0.0
        for t in reversed(range(len(ep))):
            if t == len(ep) - 1:
                next_val = 0.0
            else:
                next_val = values[t + 1]
            delta = ep[t].reward + self.gamma * next_val - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
        return advantages

    def finalize_episode(self):
        if self._current_episode:
            self._episodes.append(list(self._current_episode))
            self._current_episode = []

    def get_current_episode(self) -> List[Transition]:
        return list(self._current_episode)

    def get_episodes(self) -> List[List[Transition]]:
        return list(self._episodes)

    def num_episodes(self) -> int:
        return len(self._episodes)

    def clear(self):
        self._episodes = []
        self._current_episode = []

    def __len__(self) -> int:
        return sum(len(ep) for ep in self._episodes) + len(self._current_episode)

    def is_empty(self) -> bool:
        return len(self._current_episode) == 0 and len(self._episodes) == 0

    def current_length(self) -> int:
        return len(self._current_episode)
