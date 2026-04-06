"""
Marine environments with gym-like API for reinforcement learning.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Space & Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ActionSpace:
    """Discrete action space."""
    n: int
    actions: Optional[List[Any]] = None

    def __post_init__(self):
        if self.actions is None:
            self.actions = list(range(self.n))

    def sample(self) -> Any:
        return random.choice(self.actions)

    def contains(self, x: Any) -> bool:
        return x in self.actions


@dataclass
class ObservationSpace:
    """Box-like observation space with shape and bounds."""
    shape: Tuple[int, ...]
    low: float = -float('inf')
    high: float = float('inf')

    def sample(self) -> List[float]:
        return [random.uniform(self.low, self.high) for _ in range(int(math.prod(self.shape)))]

    def contains(self, x: Any) -> bool:
        if isinstance(x, (list, tuple)):
            return all(self.low <= v <= self.high for v in x)
        return self.low <= x <= self.high


@dataclass
class StepResult:
    """Result of env.step(action)."""
    observation: Any
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        return iter((self.observation, self.reward, self.done, self.info))

    def __getitem__(self, idx):
        return (self.observation, self.reward, self.done, self.info)[idx]


# ---------------------------------------------------------------------------
# Base helpers
# ---------------------------------------------------------------------------

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# ---------------------------------------------------------------------------
# MarineNavigationEnv
# ---------------------------------------------------------------------------

class MarineNavigationEnv:
    """Grid-based marine navigation with obstacles and a goal.

    Actions: 0=North, 1=East, 2=South, 3=West, 4=Stay
    Observation: [x, y, goal_dx, goal_dy]
    """

    def __init__(self, grid_size: int = 10, obstacles: Optional[List[Tuple[int, int]]] = None,
                 max_steps: int = 200, stochastic: bool = False):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.stochastic = stochastic
        self.obstacles: List[Tuple[int, int]] = obstacles or []
        self.action_space = ActionSpace(n=5, actions=[0, 1, 2, 3, 4])
        self.observation_space = ObservationSpace(shape=(4,), low=0, high=grid_size - 1)
        self._step_count = 0
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.goal: Tuple[int, int] = (grid_size - 1, grid_size - 1)
        self._render_buf: Optional[List[str]] = None

    def reset(self) -> Any:
        self._step_count = 0
        while True:
            self.agent_pos = (random.randint(0, self.grid_size - 1),
                              random.randint(0, self.grid_size - 1))
            if self.agent_pos not in self.obstacles and self.agent_pos != self.goal:
                break
        return self._get_obs()

    def step(self, action: int) -> StepResult:
        self._step_count += 1
        dx, dy = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)][action]
        nx = _clamp(self.agent_pos[0] + dx, 0, self.grid_size - 1)
        ny = _clamp(self.agent_pos[1] + dy, 0, self.grid_size - 1)

        # Stochastic slip
        if self.stochastic and random.random() < 0.1:
            dx2, dy2 = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            nx = _clamp(nx + dx2, 0, self.grid_size - 1)
            ny = _clamp(ny + dy2, 0, self.grid_size - 1)

        if (nx, ny) not in self.obstacles:
            self.agent_pos = (nx, ny)

        reward = -1.0  # step penalty
        done = False
        info = {}

        if self.agent_pos == self.goal:
            reward = 100.0
            done = True
            info["reason"] = "goal_reached"
        elif self._step_count >= self.max_steps:
            done = True
            info["reason"] = "max_steps"

        dist = _euclidean(self.agent_pos, self.goal)
        info["distance_to_goal"] = dist
        return StepResult(observation=self._get_obs(), reward=reward, done=done, info=info)

    def _get_obs(self) -> List[float]:
        return [
            float(self.agent_pos[0]),
            float(self.agent_pos[1]),
            float(self.goal[0] - self.agent_pos[0]),
            float(self.goal[1] - self.agent_pos[1]),
        ]

    def render(self) -> Optional[List[str]]:
        buf = []
        for y in range(self.grid_size - 1, -1, -1):
            row = ""
            for x in range(self.grid_size):
                if (x, y) == self.agent_pos:
                    row += "A"
                elif (x, y) == self.goal:
                    row += "G"
                elif (x, y) in self.obstacles:
                    row += "#"
                else:
                    row += "."
            buf.append(row)
        self._render_buf = buf
        return buf

    def set_agent_pos(self, pos: Tuple[int, int]):
        self.agent_pos = pos

    def set_goal(self, pos: Tuple[int, int]):
        self.goal = pos

    def get_obstacles(self) -> List[Tuple[int, int]]:
        return list(self.obstacles)

    def add_obstacle(self, pos: Tuple[int, int]):
        self.obstacles.append(pos)

    def remove_obstacle(self, pos: Tuple[int, int]):
        if pos in self.obstacles:
            self.obstacles.remove(pos)


# ---------------------------------------------------------------------------
# MarinePatrolEnv
# ---------------------------------------------------------------------------

class MarinePatrolEnv:
    """Patrol environment: visit waypoints in order under a fuel constraint.

    Actions: 0=N, 1=E, 2=S, 3=W, 4=Stay
    Observation: [x, y, next_wp_x, next_wp_y, fuel_remaining, wps_visited]
    """

    def __init__(self, grid_size: int = 8, waypoints: Optional[List[Tuple[int, int]]] = None,
                 max_fuel: float = 100.0, fuel_per_step: float = 1.0, max_steps: int = 500):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.max_fuel = max_fuel
        self.fuel_per_step = fuel_per_step
        self.waypoints: List[Tuple[int, int]] = waypoints or [
            (0, 0), (grid_size - 1, 0), (grid_size - 1, grid_size - 1), (0, grid_size - 1)
        ]
        self.action_space = ActionSpace(n=5, actions=[0, 1, 2, 3, 4])
        self.observation_space = ObservationSpace(shape=(6,), low=0, high=max(grid_size, max_fuel))
        self._step_count = 0
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.current_wp_idx: int = 0
        self.fuel: float = max_fuel

    def reset(self) -> Any:
        self._step_count = 0
        self.agent_pos = self.waypoints[0]
        self.current_wp_idx = 1  # start at first waypoint, advance
        self.fuel = self.max_fuel
        return self._get_obs()

    def step(self, action: int) -> StepResult:
        self._step_count += 1
        dx, dy = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)][action]
        nx = _clamp(self.agent_pos[0] + dx, 0, self.grid_size - 1)
        ny = _clamp(self.agent_pos[1] + dy, 0, self.grid_size - 1)
        self.agent_pos = (nx, ny)
        self.fuel -= self.fuel_per_step

        reward = -0.1
        done = False
        info: Dict[str, Any] = {"fuel": self.fuel, "wp_index": self.current_wp_idx}

        if self.agent_pos == self.waypoints[self.current_wp_idx]:
            self.current_wp_idx += 1
            reward = 10.0
            info["waypoint_reached"] = True
            if self.current_wp_idx >= len(self.waypoints):
                reward = 50.0
                done = True
                info["reason"] = "patrol_complete"

        if self.fuel <= 0:
            done = True
            reward -= 20.0
            info["reason"] = "out_of_fuel"

        if self._step_count >= self.max_steps:
            done = True
            info["reason"] = "max_steps"

        info["waypoints_visited"] = self.current_wp_idx
        return StepResult(observation=self._get_obs(), reward=reward, done=done, info=info)

    def _get_obs(self) -> List[float]:
        wp = self.waypoints[min(self.current_wp_idx, len(self.waypoints) - 1)]
        return [
            float(self.agent_pos[0]),
            float(self.agent_pos[1]),
            float(wp[0]),
            float(wp[1]),
            float(self.fuel),
            float(self.current_wp_idx),
        ]

    def render(self) -> Optional[List[str]]:
        buf = []
        for y in range(self.grid_size - 1, -1, -1):
            row = ""
            for x in range(self.grid_size):
                if (x, y) == self.agent_pos:
                    row += "A"
                elif (x, y) in self.waypoints:
                    idx = self.waypoints.index((x, y))
                    row += str(idx) if idx < 10 else "W"
                else:
                    row += "."
            buf.append(row)
        return buf

    def get_fuel(self) -> float:
        return self.fuel

    def get_waypoints_visited(self) -> int:
        return self.current_wp_idx

    def get_remaining_waypoints(self) -> int:
        return max(0, len(self.waypoints) - self.current_wp_idx)


# ---------------------------------------------------------------------------
# CollisionAvoidanceEnv
# ---------------------------------------------------------------------------

class CollisionAvoidanceEnv:
    """Navigation among static and dynamic obstacles.

    Actions: 0=N, 1=E, 2=S, 3=W, 4=Stay
    Observation: [x, y, heading, min_obstacle_dist, num_nearby, goal_dx, goal_dy]
    """

    def __init__(self, grid_size: int = 12, num_static: int = 5, num_dynamic: int = 2,
                 proximity_threshold: float = 2.0, max_steps: int = 300):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.proximity_threshold = proximity_threshold
        self.action_space = ActionSpace(n=5, actions=[0, 1, 2, 3, 4])
        self.observation_space = ObservationSpace(shape=(7,), low=-grid_size, high=grid_size)
        self._step_count = 0
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.agent_heading: float = 0.0
        self.goal: Tuple[int, int] = (grid_size - 1, grid_size - 1)
        self.static_obstacles: List[Tuple[int, int]] = []
        self.dynamic_obstacles: List[Tuple[int, int]] = []
        self._num_static = num_static
        self._num_dynamic = num_dynamic
        self._init_obstacles()

    def _init_obstacles(self):
        self.static_obstacles = []
        rng = random.Random(42)
        while len(self.static_obstacles) < self._num_static:
            p = (rng.randint(1, self.grid_size - 2), rng.randint(1, self.grid_size - 2))
            if p != self.goal and p != (0, 0) and p not in self.static_obstacles:
                self.static_obstacles.append(p)
        self.dynamic_obstacles = []
        while len(self.dynamic_obstacles) < self._num_dynamic:
            p = (rng.randint(2, self.grid_size - 3), rng.randint(2, self.grid_size - 3))
            if p != self.goal and p not in self.dynamic_obstacles and p not in self.static_obstacles:
                self.dynamic_obstacles.append(list(p))

    def reset(self) -> Any:
        self._step_count = 0
        self.agent_pos = (0, 0)
        self.agent_heading = 0.0
        self._init_obstacles()
        return self._get_obs()

    def step(self, action: int) -> StepResult:
        self._step_count += 1
        dx, dy = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)][action]
        self.agent_heading = math.atan2(dy, dx) if action != 4 else self.agent_heading

        nx = _clamp(self.agent_pos[0] + dx, 0, self.grid_size - 1)
        ny = _clamp(self.agent_pos[1] + dy, 0, self.grid_size - 1)
        all_obs = set(self.static_obstacles) | {tuple(o) for o in self.dynamic_obstacles}
        if (nx, ny) not in all_obs:
            self.agent_pos = (nx, ny)

        # Move dynamic obstacles
        for i in range(len(self.dynamic_obstacles)):
            ddx, ddy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            new_p = (_clamp(self.dynamic_obstacles[i][0] + ddx, 0, self.grid_size - 1),
                     _clamp(self.dynamic_obstacles[i][1] + ddy, 0, self.grid_size - 1))
            self.dynamic_obstacles[i] = list(new_p)

        # Rewards
        min_dist = self._min_obstacle_dist()
        reward = 0.0
        if min_dist < 1.0:
            reward -= 10.0
        elif min_dist < self.proximity_threshold:
            reward -= 2.0
        else:
            reward += 0.5

        done = False
        info: Dict[str, Any] = {"min_obstacle_dist": min_dist}

        if self.agent_pos == self.goal:
            reward += 100.0
            done = True
            info["reason"] = "goal_reached"
        elif min_dist < 1.0:
            done = True
            info["reason"] = "collision"
        elif self._step_count >= self.max_steps:
            done = True
            info["reason"] = "max_steps"

        info["step"] = self._step_count
        return StepResult(observation=self._get_obs(), reward=reward, done=done, info=info)

    def _min_obstacle_dist(self) -> float:
        all_obs = self.static_obstacles + [tuple(o) for o in self.dynamic_obstacles]
        if not all_obs:
            return float('inf')
        return min(_euclidean(self.agent_pos, o) for o in all_obs)

    def _nearby_count(self) -> int:
        all_obs = self.static_obstacles + [tuple(o) for o in self.dynamic_obstacles]
        return sum(1 for o in all_obs if _euclidean(self.agent_pos, o) < self.proximity_threshold * 2)

    def _get_obs(self) -> List[float]:
        return [
            float(self.agent_pos[0]),
            float(self.agent_pos[1]),
            self.agent_heading,
            self._min_obstacle_dist(),
            float(self._nearby_count()),
            float(self.goal[0] - self.agent_pos[0]),
            float(self.goal[1] - self.agent_pos[1]),
        ]

    def render(self) -> Optional[List[str]]:
        buf = []
        all_obs = set(self.static_obstacles) | {tuple(o) for o in self.dynamic_obstacles}
        for y in range(self.grid_size - 1, -1, -1):
            row = ""
            for x in range(self.grid_size):
                if (x, y) == self.agent_pos:
                    row += "A"
                elif (x, y) == self.goal:
                    row += "G"
                elif (x, y) in all_obs:
                    row += "#"
                else:
                    row += "."
            buf.append(row)
        return buf

    def get_agent_pos(self) -> Tuple[int, int]:
        return self.agent_pos

    def get_dynamic_obstacles(self) -> List[Tuple[int, int]]:
        return [tuple(o) for o in self.dynamic_obstacles]

    def get_static_obstacles(self) -> List[Tuple[int, int]]:
        return list(self.static_obstacles)
