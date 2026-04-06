"""
RL agents: tabular Q-learning, deep Q-learning (dict-simulated),
policy gradient, and PPO-lite. Pure Python — no numpy/torch.
"""

import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _softmax(values: List[float], temperature: float = 1.0) -> List[float]:
    """Numerically stable softmax."""
    if not values:
        return []
    t = max(temperature, 1e-8)
    max_v = max(values)
    exps = [math.exp((v - max_v) / t) for v in values]
    s = sum(exps)
    if s == 0:
        n = len(values)
        return [1.0 / n] * n
    return [e / s for e in exps]


def _relu(x: float) -> float:
    return max(0.0, x)


def _tanh(x: float) -> float:
    return math.tanh(x)


def _sigm(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _make_hashable(state: Any) -> Any:
    if isinstance(state, (list, tuple)):
        return tuple(state)
    return state


# ---------------------------------------------------------------------------
# TabularQLearning
# ---------------------------------------------------------------------------

class TabularQLearning:
    """Classic epsilon-greedy tabular Q-learning."""

    def __init__(self, n_actions: int, lr: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_min: float = 0.01, epsilon_decay: float = 0.995):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: Dict[Any, List[float]] = defaultdict(lambda: [0.0] * n_actions)
        self._steps = 0

    def select_action(self, state: Any) -> int:
        s = _make_hashable(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        q_vals = self.q_table[s]
        max_q = max(q_vals)
        # Break ties randomly
        best = [a for a in range(self.n_actions) if q_vals[a] == max_q]
        return random.choice(best)

    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool):
        s = _make_hashable(state)
        ns = _make_hashable(next_state)
        current = self.q_table[s][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.q_table[ns])
        self.q_table[s][action] = current + self.lr * (target - current)
        self._steps += 1

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_table(self) -> Dict[Any, List[float]]:
        return dict(self.q_table)

    def get_q_values(self, state: Any) -> List[float]:
        s = _make_hashable(state)
        return list(self.q_table[s])

    def get_state_count(self) -> int:
        return len(self.q_table)


# ---------------------------------------------------------------------------
# DeepQLearning  (dict-based simulated neural network)
# ---------------------------------------------------------------------------

class DeepQLearning:
    """Q-learning with a dict-simulated deep network and target network."""

    def __init__(self, state_dim: int, n_actions: int, lr: float = 0.001,
                 gamma: float = 0.99, epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995, hidden_size: int = 64,
                 target_update_freq: int = 100):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.hidden_size = hidden_size
        self.target_update_freq = target_update_freq
        self._steps = 0
        # Weights stored as flat dicts keyed by layer name
        self._weights: Dict[str, List[float]] = {}
        self._target_weights: Dict[str, List[float]] = {}
        self._init_weights()

    def _init_weights(self):
        random.seed(0)
        self._weights["W1"] = [random.gauss(0, 0.1) for _ in range(self.state_dim * self.hidden_size)]
        self._weights["b1"] = [0.0] * self.hidden_size
        self._weights["W2"] = [random.gauss(0, 0.1) for _ in range(self.hidden_size * self.n_actions)]
        self._weights["b2"] = [0.0] * self.n_actions
        self._target_weights = {k: list(v) for k, v in self._weights.items()}
        random.seed()

    def _forward(self, state: List[float], weights: Dict[str, List[float]]) -> List[float]:
        # Layer 1: linear + ReLU
        hidden = [0.0] * self.hidden_size
        for j in range(self.hidden_size):
            s = weights["b1"][j]
            for i in range(min(self.state_dim, len(state))):
                s += state[i] * weights["W1"][i * self.hidden_size + j]
            hidden[j] = _relu(s)
        # Layer 2: linear
        output = [0.0] * self.n_actions
        for a in range(self.n_actions):
            s = weights["b2"][a]
            for j in range(self.hidden_size):
                s += hidden[j] * weights["W2"][j * self.n_actions + a]
            output[a] = s
        return output

    def _q_values(self, state: List[float]) -> List[float]:
        return self._forward(state, self._weights)

    def select_action(self, state: List[float]) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        q = self._q_values(state)
        max_q = max(q)
        best = [a for a in range(self.n_actions) if q[a] == max_q]
        return random.choice(best)

    def update(self, state: List[float], action: int, reward: float, next_state: List[float], done: bool):
        current_q = self._q_values(state)[action]
        target_q_vals = self._forward(next_state, self._target_weights)
        target = reward if done else reward + self.gamma * max(target_q_vals)
        # Simple gradient: nudge the weight toward target (simplified)
        # We update the output bias for the selected action
        self._weights["b2"][action] += self.lr * (target - current_q) * 0.01
        self._steps += 1

    def update_batch(self, batch: List[Tuple]) -> List[float]:
        """Update on a batch of (state, action, reward, next_state, done). Returns list of TD errors."""
        errors = []
        for s, a, r, ns, d in batch:
            current = self._q_values(s)[a]
            target_q = self._forward(ns, self._target_weights)
            target = r if d else r + self.gamma * max(target_q)
            td = target - current
            self._weights["b2"][a] += self.lr * td * 0.01
            errors.append(abs(td))
            self._steps += 1
        return errors

    def target_update(self):
        """Hard copy online weights to target."""
        self._target_weights = {k: list(v) for k, v in self._weights.items()}

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_weights(self) -> Dict[str, List[float]]:
        return self._weights

    def get_step_count(self) -> int:
        return self._steps


# ---------------------------------------------------------------------------
# PolicyGradientAgent
# ---------------------------------------------------------------------------

class PolicyGradientAgent:
    """REINFORCE-style policy gradient with softmax policy."""

    def __init__(self, state_dim: int, n_actions: int, lr: float = 0.01, gamma: float = 0.99):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        # Policy parameters: (state_dim x n_actions) weights
        self._policy_weights: List[List[float]] = [[0.01] * n_actions for _ in range(state_dim)]
        self._transitions: List[Tuple[List[float], int, float]] = []  # (state, action, reward)

    def _policy_probs(self, state: List[float]) -> List[float]:
        logits = [0.0] * self.n_actions
        for i in range(min(self.state_dim, len(state))):
            for a in range(self.n_actions):
                logits[a] += state[i] * self._policy_weights[i][a]
        return _softmax(logits)

    def select_action(self, state: List[float]) -> int:
        probs = self._policy_probs(state)
        r = random.random()
        cumulative = 0.0
        for a, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                return a
        return self.n_actions - 1

    def store_transition(self, state: List[float], action: int, reward: float):
        self._transitions.append((list(state), action, reward))

    def _compute_returns(self) -> List[float]:
        returns = []
        g = 0.0
        for _, _, r in reversed(self._transitions):
            g = r + self.gamma * g
            returns.insert(0, g)
        return returns

    def update_policy(self) -> float:
        """Run REINFORCE update. Returns total loss magnitude."""
        if not self._transitions:
            return 0.0
        returns = self._compute_returns()
        total_loss = 0.0
        for (state, action, _), G in zip(self._transitions, returns):
            probs = self._policy_probs(state)
            for i in range(min(self.state_dim, len(state))):
                grad = -G * (1.0 - probs[action]) if action == len(probs) - 1 and probs[action] < 0.5 else -G
                for a in range(self.n_actions):
                    indicator = 1.0 if a == action else 0.0
                    self._policy_weights[i][a] += self.lr * G * (indicator - probs[a]) * state[i] * 0.001
            total_loss += abs(G)
        self._transitions = []
        return total_loss / len(returns)

    def get_policy_probs(self, state: List[float]) -> List[float]:
        return self._policy_probs(state)

    def clear_transitions(self):
        self._transitions = []


# ---------------------------------------------------------------------------
# PPOLiteAgent
# ---------------------------------------------------------------------------

class PPOLiteAgent:
    """PPO-lite with clipped surrogate objective and a value function."""

    def __init__(self, state_dim: int, n_actions: int, lr: float = 0.001,
                 gamma: float = 0.99, clip_ratio: float = 0.2, epochs: int = 3,
                 hidden_size: int = 32, gae_lambda: float = 0.95):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.gae_lambda = gae_lambda
        # Policy: state_dim x n_actions
        self._policy_w: List[List[float]] = [[0.01] * n_actions for _ in range(state_dim)]
        # Value: state_dim -> scalar
        self._value_w: List[float] = [0.01] * state_dim
        self._value_b: float = 0.0
        # Buffer
        self._buffer: List[Dict[str, Any]] = []

    def _policy_probs(self, state: List[float]) -> List[float]:
        logits = [0.0] * self.n_actions
        for i in range(min(self.state_dim, len(state))):
            for a in range(self.n_actions):
                logits[a] += state[i] * self._policy_w[i][a]
        return _softmax(logits)

    def _value(self, state: List[float]) -> float:
        s = self._value_b
        for i in range(min(self.state_dim, len(state))):
            s += self._value_w[i] * state[i]
        return _tanh(s)

    def select_action(self, state: List[float]) -> int:
        probs = self._policy_probs(state)
        r = random.random()
        cumulative = 0.0
        for a, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                return a
        return self.n_actions - 1

    def store_transition(self, state: List[float], action: int, reward: float,
                         value: float, log_prob: float):
        self._buffer.append({
            "state": list(state), "action": action, "reward": reward,
            "value": value, "log_prob": log_prob,
        })

    def compute_advantages(self, last_value: float = 0.0) -> List[float]:
        """Compute GAE advantages from buffer."""
        if not self._buffer:
            return []
        advantages = []
        gae = 0.0
        for t in reversed(range(len(self._buffer))):
            if t == len(self._buffer) - 1:
                next_val = last_value
            else:
                next_val = self._buffer[t + 1]["value"]
            delta = self._buffer[t]["reward"] + self.gamma * next_val - self._buffer[t]["value"]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
        return advantages

    def _normalize(self, vals: List[float]) -> List[float]:
        if not vals:
            return []
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) + 1e-8
        return [(v - mean) / std for v in vals]

    def update_policy_batch(self) -> float:
        """PPO clipped surrogate update. Returns mean loss."""
        if not self._buffer:
            return 0.0
        advantages = self.compute_advantages()
        if not advantages:
            return 0.0
        adv_norm = self._normalize(advantages)

        total_loss = 0.0
        for _ in range(self.epochs):
            for i, trans in enumerate(self._buffer):
                state = trans["state"]
                action = trans["action"]
                old_log_prob = trans["log_prob"]
                adv = adv_norm[i]

                probs = self._policy_probs(state)
                new_prob = max(probs[action], 1e-8)
                new_log_prob = math.log(new_prob)

                ratio = math.exp(new_log_prob - old_log_prob)
                clipped = max(self.clip_ratio, 1 - self.clip_ratio)
                surrogate1 = ratio * adv
                surrogate2 = clipped * adv
                loss = -min(surrogate1, surrogate2)

                # Update policy weights
                for ii in range(min(self.state_dim, len(state))):
                    indicator = 1.0 if ii == action else 0.0
                    grad = -(indicator - probs[ii]) * adv * 0.001
                    # Apply gradient for the action dimension
                    for a in range(self.n_actions):
                        ind = 1.0 if a == action else 0.0
                        self._policy_w[ii][a] += self.lr * (ind - probs[a]) * adv * state[ii] * 0.01

                total_loss += abs(loss)

        return total_loss / len(self._buffer)

    def update_value(self) -> float:
        """Update value function via MSE on buffer. Returns mean loss."""
        if not self._buffer:
            return 0.0
        advantages = self.compute_advantages()
        if not advantages:
            return 0.0
        returns = [self._buffer[i]["value"] + advantages[i] for i in range(len(self._buffer))]
        ret_norm = self._normalize(returns)

        total_loss = 0.0
        for i, trans in enumerate(self._buffer):
            state = trans["state"]
            target = ret_norm[i]
            pred = self._value(state)
            error = target - pred
            for ii in range(min(self.state_dim, len(state))):
                self._value_w[ii] += self.lr * error * state[ii] * 0.01
            self._value_b += self.lr * error * 0.01
            total_loss += error ** 2
        return total_loss / len(self._buffer)

    def get_policy_probs(self, state: List[float]) -> List[float]:
        return self._policy_probs(state)

    def get_value(self, state: List[float]) -> float:
        return self._value(state)

    def clear_buffer(self):
        self._buffer = []

    def buffer_size(self) -> int:
        return len(self._buffer)

    def compute_returns(self, last_value: float = 0.0) -> List[float]:
        """Compute discounted returns from buffer."""
        returns = []
        g = 0.0
        for t in reversed(range(len(self._buffer))):
            g = self._buffer[t]["reward"] + self.gamma * g
            returns.insert(0, g)
        return returns
