"""
Marine-specific reward shaping functions.
"""

import math
from typing import Any, Dict, List, Optional, Tuple


class NavigationRewardShaper:
    """Reward shaping for marine navigation tasks."""

    def __init__(self, goal_weight: float = 10.0, heading_weight: float = 2.0,
                 speed_weight: float = 1.0, collision_penalty: float = -50.0,
                 boundary_penalty: float = -10.0, step_penalty: float = -0.1,
                 goal_radius: float = 1.0):
        self.goal_weight = goal_weight
        self.heading_weight = heading_weight
        self.speed_weight = speed_weight
        self.collision_penalty = collision_penalty
        self.boundary_penalty = boundary_penalty
        self.step_penalty = step_penalty
        self.goal_radius = goal_radius

    def distance_reward(self, agent_pos: Tuple[float, float],
                        goal_pos: Tuple[float, float]) -> float:
        dist = math.sqrt((agent_pos[0] - goal_pos[0]) ** 2 +
                         (agent_pos[1] - goal_pos[1]) ** 2)
        if dist <= self.goal_radius:
            return self.goal_weight
        # Shaped reward: closer = higher
        return -dist * 0.1

    def heading_reward(self, agent_heading: float, target_heading: float) -> float:
        diff = abs(agent_heading - target_heading)
        if diff > math.pi:
            diff = 2 * math.pi - diff
        alignment = 1.0 - (diff / math.pi)
        return self.heading_weight * alignment

    def speed_reward(self, current_speed: float, optimal_speed: float) -> float:
        if optimal_speed == 0:
            return 0.0
        ratio = current_speed / optimal_speed
        if ratio <= 1.0:
            return self.speed_weight * ratio
        return self.speed_weight / ratio  # penalize too fast

    def collision_reward(self, is_collision: bool, min_distance: float) -> float:
        if is_collision:
            return self.collision_penalty
        if min_distance < 2.0:
            return -5.0 * (1.0 - min_distance / 2.0)
        return 0.0

    def boundary_reward(self, pos: Tuple[float, float], grid_size: float) -> float:
        margin = 1.0
        if (pos[0] <= margin or pos[0] >= grid_size - margin or
                pos[1] <= margin or pos[1] >= grid_size - margin):
            return self.boundary_penalty
        return 0.0

    def compute(self, agent_pos: Tuple[float, float], goal_pos: Tuple[float, float],
                agent_heading: float = 0.0, target_heading: float = 0.0,
                current_speed: float = 1.0, optimal_speed: float = 1.0,
                is_collision: bool = False, min_distance: float = 999.0,
                grid_size: float = 10.0) -> Dict[str, float]:
        rewards = {
            "distance": self.distance_reward(agent_pos, goal_pos),
            "heading": self.heading_reward(agent_heading, target_heading),
            "speed": self.speed_reward(current_speed, optimal_speed),
            "collision": self.collision_reward(is_collision, min_distance),
            "boundary": self.boundary_reward(agent_pos, grid_size),
            "step": self.step_penalty,
        }
        rewards["total"] = sum(rewards.values())
        return rewards


class PatrolRewardShaper:
    """Reward shaping for patrol missions."""

    def __init__(self, waypoint_reward: float = 10.0, efficiency_weight: float = 1.0,
                 fuel_weight: float = 0.5, idle_penalty: float = -0.5,
                 completion_bonus: float = 50.0):
        self.waypoint_reward = waypoint_reward
        self.efficiency_weight = efficiency_weight
        self.fuel_weight = fuel_weight
        self.idle_penalty = idle_penalty
        self.completion_bonus = completion_bonus

    def coverage_reward(self, visited_positions: set, grid_size: int) -> float:
        total = grid_size * grid_size
        visited_ratio = len(visited_positions) / total
        return visited_ratio * self.efficiency_weight * 10.0

    def efficiency_reward(self, steps_taken: int, waypoints_visited: int,
                          total_waypoints: int) -> float:
        if total_waypoints == 0 or waypoints_visited == 0:
            return 0.0
        return self.efficiency_weight * waypoints_visited / max(steps_taken, 1)

    def fuel_reward(self, fuel_remaining: float, max_fuel: float) -> float:
        ratio = fuel_remaining / max(max_fuel, 1e-8)
        return self.fuel_weight * ratio

    def waypoint_reward_fn(self, just_visited: bool, all_visited: bool = False) -> float:
        if all_visited:
            return self.completion_bonus
        if just_visited:
            return self.waypoint_reward
        return 0.0

    def idle_penalty_fn(self, is_idle: bool) -> float:
        return self.idle_penalty if is_idle else 0.0

    def compute(self, visited_positions: set, grid_size: int, steps_taken: int,
                waypoints_visited: int, total_waypoints: int,
                fuel_remaining: float, max_fuel: float,
                just_visited_wp: bool = False, all_visited: bool = False,
                is_idle: bool = False) -> Dict[str, float]:
        rewards = {
            "coverage": self.coverage_reward(visited_positions, grid_size),
            "efficiency": self.efficiency_reward(steps_taken, waypoints_visited, total_waypoints),
            "fuel": self.fuel_reward(fuel_remaining, max_fuel),
            "waypoint": self.waypoint_reward_fn(just_visited_wp, all_visited),
            "idle": self.idle_penalty_fn(is_idle),
        }
        rewards["total"] = sum(rewards.values())
        return rewards


class MultiObjectiveReward:
    """Weighted multi-objective reward with Pareto filtering."""

    def __init__(self):
        self._objectives: Dict[str, Tuple[float, float]] = {}  # name -> (weight, value)
        self._history: List[Dict[str, float]] = []

    def add_objective(self, name: str, weight: float = 1.0, value: float = 0.0):
        self._objectives[name] = (weight, value)

    def remove_objective(self, name: str):
        if name in self._objectives:
            del self._objectives[name]

    def set_weight(self, name: str, weight: float):
        if name in self._objectives:
            self._objectives[name] = (weight, self._objectives[name][1])

    def set_value(self, name: str, value: float):
        if name in self._objectives:
            self._objectives[name] = (self._objectives[name][0], value)

    def get_objectives(self) -> Dict[str, Tuple[float, float]]:
        return dict(self._objectives)

    def compute_weighted(self) -> float:
        total = 0.0
        for name, (w, v) in self._objectives.items():
            total += w * v
        return total

    def compute_objective_vector(self) -> Dict[str, float]:
        return {name: w * v for name, (w, v) in self._objectives.items()}

    def record(self):
        """Record current objective values to history."""
        self._history.append({name: v for name, (_, v) in self._objectives.items()})

    def pareto_filter(self) -> List[Dict[str, float]]:
        """Return Pareto-optimal solutions from history."""
        if not self._history:
            return []
        pareto = []
        for i, sol_i in enumerate(self._history):
            dominated = False
            for j, sol_j in enumerate(self._history):
                if i == j:
                    continue
                # sol_j dominates sol_i if it's >= in all objectives and > in at least one
                if all(sol_j.get(k, 0) >= sol_i.get(k, 0) for k in sol_i):
                    if any(sol_j.get(k, 0) > sol_i.get(k, 0) for k in sol_i):
                        dominated = True
                        break
            if not dominated:
                pareto.append(sol_i)
        return pareto

    def clear_history(self):
        self._history = []

    def get_history(self) -> List[Dict[str, float]]:
        return list(self._history)

    def num_objectives(self) -> int:
        return len(self._objectives)
