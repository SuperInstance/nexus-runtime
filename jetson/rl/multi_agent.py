"""
Multi-agent reinforcement learning: environments, independent learners,
centralized critic, and communication protocols.
"""

import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


class MultiAgentEnv:
    """Simple multi-agent grid environment."""

    def __init__(self, agents: List[str], grid_size: int = 10,
                 max_steps: int = 200):
        self.agent_ids = list(agents)
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.n_actions = 5  # N, E, S, W, Stay
        self._positions: Dict[str, Tuple[int, int]] = {}
        self._goals: Dict[str, Tuple[int, int]] = {}
        self._step_count = 0
        self._reset_positions()

    def _reset_positions(self):
        for i, aid in enumerate(self.agent_ids):
            self._positions[aid] = (i * 2 % self.grid_size, i * 2 % self.grid_size)
            self._goals[aid] = (self.grid_size - 1 - i, self.grid_size - 1 - i)

    def reset(self) -> Dict[str, Any]:
        self._step_count = 0
        self._reset_positions()
        return {aid: self._obs(aid) for aid in self.agent_ids}

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, Any], Dict[str, float],
                                                      Dict[str, bool], Dict[str, Dict]]:
        self._step_count += 1
        observations = {}
        rewards = {}
        dones = {}
        infos = {}

        for aid in self.agent_ids:
            action = actions.get(aid, 4)
            dx, dy = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)][action]
            x, y = self._positions[aid]
            nx = max(0, min(self.grid_size - 1, x + dx))
            ny = max(0, min(self.grid_size - 1, y + dy))
            self._positions[aid] = (nx, ny)

            dist = math.sqrt((nx - self._goals[aid][0]) ** 2 +
                             (ny - self._goals[aid][1]) ** 2)
            reward = -1.0
            done = False

            if (nx, ny) == self._goals[aid]:
                reward = 100.0
                done = True

            if self._step_count >= self.max_steps:
                done = True

            observations[aid] = self._obs(aid)
            rewards[aid] = reward
            dones[aid] = done
            infos[aid] = {"distance": dist, "step": self._step_count}

        return observations, rewards, dones, infos

    def _obs(self, aid: str) -> List[float]:
        x, y = self._positions[aid]
        gx, gy = self._goals[aid]
        return [float(x), float(y), float(gx - x), float(gy - y)]

    def get_positions(self) -> Dict[str, Tuple[int, int]]:
        return dict(self._positions)

    def get_goals(self) -> Dict[str, Tuple[int, int]]:
        return dict(self._goals)

    def set_goals(self, goals: Dict[str, Tuple[int, int]]):
        self._goals = dict(goals)

    def num_agents(self) -> int:
        return len(self.agent_ids)


class IndependentLearner:
    """Per-agent independent tabular Q-learning."""

    def __init__(self, agent_ids: List[str], n_actions: int = 5, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 1.0, epsilon_decay: float = 0.995):
        self.agent_ids = list(agent_ids)
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self._q_tables: Dict[str, Dict[Any, List[float]]] = {
            aid: defaultdict(lambda: [0.0] * n_actions) for aid in agent_ids
        }

    def select_actions(self, observations: Dict[str, Any]) -> Dict[str, int]:
        actions = {}
        for aid, obs in observations.items():
            if aid not in self.agent_ids:
                continue
            if random.random() < self.epsilon:
                actions[aid] = random.randint(0, self.n_actions - 1)
            else:
                s = tuple(obs) if isinstance(obs, (list, tuple)) else obs
                q = self._q_tables[aid][s]
                max_q = max(q)
                best = [a for a in range(self.n_actions) if q[a] == max_q]
                actions[aid] = random.choice(best)
        return actions

    def update(self, aid: str, state: Any, action: int, reward: float,
               next_state: Any, done: bool):
        if aid not in self._q_tables:
            return
        s = tuple(state) if isinstance(state, (list, tuple)) else state
        ns = tuple(next_state) if isinstance(next_state, (list, tuple)) else next_state
        current = self._q_tables[aid][s][action]
        target = reward if done else reward + self.gamma * max(self._q_tables[aid][ns])
        self._q_tables[aid][s][action] = current + self.lr * (target - current)

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    def get_q_table(self, aid: str) -> Dict[Any, List[float]]:
        return dict(self._q_tables.get(aid, {}))

    def get_epsilon(self) -> float:
        return self.epsilon


class CentralizedCritic:
    """Centralized critic that computes joint value and per-agent advantages."""

    def __init__(self, agent_ids: List[str], state_dim: int = 4, lr: float = 0.01,
                 gamma: float = 0.99):
        self.agent_ids = list(agent_ids)
        self.state_dim = state_dim
        self.lr = lr
        self.gamma = gamma
        # Joint value weights: sum of all agent states
        total_dim = state_dim * len(agent_ids)
        self._value_weights: List[float] = [0.01] * total_dim
        self._value_bias: float = 0.0
        self._returns_buffer: Dict[str, List[float]] = {aid: [] for aid in agent_ids}

    def _joint_state(self, observations: Dict[str, Any]) -> List[float]:
        joint = []
        for aid in self.agent_ids:
            obs = observations.get(aid, [0.0] * self.state_dim)
            for v in (obs if isinstance(obs, (list, tuple)) else [obs]):
                joint.append(float(v))
        return joint

    def joint_value(self, observations: Dict[str, Any]) -> float:
        joint = self._joint_state(observations)
        total = self._value_bias
        for i in range(min(len(self._value_weights), len(joint))):
            total += self._value_weights[i] * joint[i]
        return math.tanh(total)

    def advantage_per_agent(self, aid: str, observations: Dict[str, Any],
                            rewards: Dict[str, float]) -> float:
        jv = self.joint_value(observations)
        agent_reward = rewards.get(aid, 0.0)
        return agent_reward - jv

    def store_returns(self, aid: str, returns: List[float]):
        if aid in self._returns_buffer:
            self._returns_buffer[aid].extend(returns)

    def update(self, observations: Dict[str, Any], target_value: float):
        """Single-step value update."""
        joint = self._joint_state(observations)
        pred = self.joint_value(observations)
        error = target_value - pred
        for i in range(min(len(self._value_weights), len(joint))):
            self._value_weights[i] += self.lr * error * joint[i] * 0.01
        self._value_bias += self.lr * error * 0.01

    def get_value_weights(self) -> List[float]:
        return list(self._value_weights)

    def get_returns_buffer(self, aid: str) -> List[float]:
        return list(self._returns_buffer.get(aid, []))

    def clear_buffers(self):
        self._returns_buffer = {aid: [] for aid in self.agent_ids}


class CommunicationProtocol:
    """Simulated inter-agent communication with broadcast and shared memory."""

    def __init__(self, agents: List[str], max_message_size: int = 100,
                 channel_capacity: int = 50):
        self.agent_ids = list(agents)
        self.max_message_size = max_message_size
        self.channel_capacity = channel_capacity
        self._messages: List[Dict[str, Any]] = []
        self._shared_memory: Dict[str, Dict[str, Any]] = {
            aid: {} for aid in agents
        }
        self._inbox: Dict[str, List[Dict[str, Any]]] = {
            aid: [] for aid in agents
        }

    def send_message(self, sender: str, receiver: str, content: Any) -> bool:
        if sender not in self.agent_ids or receiver not in self.agent_ids:
            return False
        msg = {"sender": sender, "receiver": receiver, "content": content}
        self._inbox[receiver].append(msg)
        return True

    def broadcast(self, sender: str, content: Any) -> int:
        if sender not in self.agent_ids:
            return 0
        count = 0
        for aid in self.agent_ids:
            if aid != sender:
                msg = {"sender": sender, "receiver": aid, "content": content,
                       "broadcast": True}
                self._inbox[aid].append(msg)
                count += 1
        return count

    def receive(self, agent_id: str) -> List[Dict[str, Any]]:
        if agent_id not in self._inbox:
            return []
        msgs = list(self._inbox[agent_id])
        self._inbox[agent_id].clear()
        return msgs

    def peek(self, agent_id: str) -> List[Dict[str, Any]]:
        if agent_id not in self._inbox:
            return []
        return list(self._inbox[agent_id])

    def write_shared(self, agent_id: str, key: str, value: Any):
        if agent_id in self._shared_memory:
            self._shared_memory[agent_id][key] = value

    def read_shared(self, agent_id: str, key: str, default: Any = None) -> Any:
        if agent_id in self._shared_memory:
            return self._shared_memory[agent_id].get(key, default)
        return default

    def read_all_shared(self, agent_id: str) -> Dict[str, Any]:
        if agent_id in self._shared_memory:
            return dict(self._shared_memory[agent_id])
        return {}

    def clear_shared(self, agent_id: str):
        if agent_id in self._shared_memory:
            self._shared_memory[agent_id] = {}

    def clear_all(self):
        self._inbox = {aid: [] for aid in self.agent_ids}
        self._shared_memory = {aid: {} for aid in self.agent_ids}
        self._messages = []

    def message_count(self, agent_id: str) -> int:
        return len(self._inbox.get(agent_id, []))

    def total_messages(self) -> int:
        return sum(len(v) for v in self._inbox.values())
