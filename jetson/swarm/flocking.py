"""
Flocking Module (Reynolds Rules)
=================================
Implements separation, alignment, and cohesion behaviors for marine
swarm agents, plus obstacle avoidance and full flock simulation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class FlockingParams:
    """Immutable parameters controlling Reynolds flocking behavior."""
    separation_weight: float = 1.5
    alignment_weight: float = 1.0
    cohesion_weight: float = 1.0
    max_speed: float = 5.0
    max_force: float = 3.0
    perception_radius: float = 50.0
    separation_radius: float = 15.0
    obstacle_avoidance_radius: float = 20.0
    obstacle_avoidance_weight: float = 2.0


@dataclass
class Agent:
    """A single flocking agent with position, velocity and acceleration."""
    agent_id: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    max_speed: float = 5.0
    max_force: float = 3.0

    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    def limit_velocity(self) -> None:
        s = self.speed()
        if s > self.max_speed and s > 0:
            scale = self.max_speed / s
            self.vx *= scale
            self.vy *= scale

    def limit_force(self, fx: float, fy: float) -> Tuple[float, float]:
        mag = math.hypot(fx, fy)
        if mag > self.max_force and mag > 0:
            scale = self.max_force / mag
            fx *= scale
            fy *= scale
        return fx, fy

    def apply_force(self, fx: float, fy: float) -> None:
        fx, fy = self.limit_force(fx, fy)
        self.ax += fx
        self.ay += fy

    def update(self, dt: float = 1.0) -> None:
        self.vx += self.ax * dt
        self.vy += self.ay * dt
        self.limit_velocity()
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.ax = 0.0
        self.ay = 0.0


@dataclass(frozen=True)
class Obstacle:
    """An obstacle in the environment."""
    x: float
    y: float
    radius: float = 5.0


class FlockingBehavior:
    """
    Computes Reynolds flocking forces: separation, alignment, cohesion,
    and obstacle avoidance for individual agents.
    """

    def __init__(self, params: Optional[FlockingParams] = None):
        self.params = params or FlockingParams()

    # ------------------------------------------------------------------
    # Core Reynolds rules
    # ------------------------------------------------------------------

    def compute_separation(
        self, agent: Agent, neighbors: List[Agent]
    ) -> Tuple[float, float]:
        """
        Steering force to avoid crowding nearby flock-mates.
        Weight inversely by distance.
        """
        steer_x, steer_y = 0.0, 0.0
        count = 0
        sep_r = self.params.separation_radius

        for other in neighbors:
            if other.agent_id == agent.agent_id:
                continue
            dx = agent.x - other.x
            dy = agent.y - other.y
            dist = math.hypot(dx, dy)
            if 0 < dist < sep_r:
                steer_x += dx / dist / dist
                steer_y += dy / dist / dist
                count += 1

        if count > 0:
            steer_x /= count
            steer_y /= count
            mag = math.hypot(steer_x, steer_y)
            if mag > 0:
                steer_x = steer_x / mag * self.params.max_speed - agent.vx
                steer_y = steer_y / mag * self.params.max_speed - agent.vy
                agent.limit_force(steer_x, steer_y)
                return agent.limit_force(steer_x, steer_y)
        return (0.0, 0.0)

    def compute_alignment(
        self, agent: Agent, neighbors: List[Agent]
    ) -> Tuple[float, float]:
        """
        Steering force to align velocity with nearby flock-mates.
        """
        avg_vx, avg_vy, count = 0.0, 0.0, 0

        for other in neighbors:
            if other.agent_id == agent.agent_id:
                continue
            dist = math.hypot(agent.x - other.x, agent.y - other.y)
            if dist < self.params.perception_radius:
                avg_vx += other.vx
                avg_vy += other.vy
                count += 1

        if count > 0:
            avg_vx /= count
            avg_vy /= count
            mag = math.hypot(avg_vx, avg_vy)
            if mag > 0:
                avg_vx = avg_vx / mag * self.params.max_speed
                avg_vy = avg_vy / mag * self.params.max_speed
            steer_x = avg_vx - agent.vx
            steer_y = avg_vy - agent.vy
            return agent.limit_force(steer_x, steer_y)
        return (0.0, 0.0)

    def compute_cohesion(
        self, agent: Agent, neighbors: List[Agent]
    ) -> Tuple[float, float]:
        """
        Steering force to move toward the centre of mass of nearby flock-mates.
        """
        centre_x, centre_y, count = 0.0, 0.0, 0

        for other in neighbors:
            if other.agent_id == agent.agent_id:
                continue
            dist = math.hypot(agent.x - other.x, agent.y - other.y)
            if dist < self.params.perception_radius:
                centre_x += other.x
                centre_y += other.y
                count += 1

        if count > 0:
            centre_x /= count
            centre_y /= count
            desired_x = centre_x - agent.x
            desired_y = centre_y - agent.y
            mag = math.hypot(desired_x, desired_y)
            if mag > 0:
                desired_x = desired_x / mag * self.params.max_speed
                desired_y = desired_y / mag * self.params.max_speed
            steer_x = desired_x - agent.vx
            steer_y = desired_y - agent.vy
            return agent.limit_force(steer_x, steer_y)
        return (0.0, 0.0)

    def compute_flocking_force(
        self, agent: Agent, neighbors: List[Agent]
    ) -> Tuple[float, float]:
        """
        Combined flocking force: weighted sum of separation, alignment,
        cohesion, and obstacle avoidance.
        """
        sep = self.compute_separation(agent, neighbors)
        ali = self.compute_alignment(agent, neighbors)
        coh = self.compute_cohesion(agent, neighbors)

        p = self.params
        fx = (sep[0] * p.separation_weight
              + ali[0] * p.alignment_weight
              + coh[0] * p.cohesion_weight)
        fy = (sep[1] * p.separation_weight
              + ali[1] * p.alignment_weight
              + coh[1] * p.cohesion_weight)
        return fx, fy

    def obstacle_avoidance_force(
        self, agent: Agent, obstacles: List[Obstacle]
    ) -> Tuple[float, float]:
        """
        Repulsive force steering the agent away from obstacles.
        """
        steer_x, steer_y = 0.0, 0.0
        for obs in obstacles:
            dx = agent.x - obs.x
            dy = agent.y - obs.y
            dist = math.hypot(dx, dy)
            avoid_dist = obs.radius + self.params.obstacle_avoidance_radius
            if 0 < dist < avoid_dist:
                strength = (avoid_dist - dist) / avoid_dist
                steer_x += (dx / dist) * strength * self.params.obstacle_avoidance_weight
                steer_y += (dy / dist) * strength * self.params.obstacle_avoidance_weight
        return agent.limit_force(steer_x, steer_y)


class FlockSimulation:
    """
    Full multi-agent flocking simulation with optional obstacles.
    """

    def __init__(
        self,
        agents: List[Agent],
        obstacles: Optional[List[Obstacle]] = None,
        params: Optional[FlockingParams] = None,
    ):
        self.agents: List[Agent] = agents
        self.obstacles: List[Obstacle] = obstacles or []
        self.behavior = FlockingBehavior(params or FlockingParams())
        self.step_count: int = 0

    def step(self, dt: float = 1.0) -> None:
        """Advance simulation by one time-step."""
        for agent in self.agents:
            flock_force = self.behavior.compute_flocking_force(agent, self.agents)
            obs_force = self.behavior.obstacle_avoidance_force(agent, self.obstacles)
            agent.apply_force(flock_force[0] + obs_force[0],
                              flock_force[1] + obs_force[1])
            agent.update(dt)
        self.step_count += 1

    def run(self, steps: int, dt: float = 1.0) -> List[Dict[str, Tuple[float, float]]]:
        """
        Run *steps* iterations and return a history of agent positions.

        Each entry is a dict mapping agent_id -> (x, y).
        """
        history: List[Dict[str, Tuple[float, float]]] = []
        for _ in range(steps):
            self.step(dt)
            snapshot: Dict[str, Tuple[float, float]] = {}
            for a in self.agents:
                snapshot[a.agent_id] = (a.x, a.y)
            history.append(snapshot)
        return history

    def get_agent_states(self) -> List[Dict]:
        """Return current state of every agent as a list of dicts."""
        return [
            {
                "agent_id": a.agent_id,
                "x": a.x,
                "y": a.y,
                "vx": a.vx,
                "vy": a.vy,
                "speed": a.speed(),
            }
            for a in self.agents
        ]

    def add_agent(self, agent: Agent) -> None:
        self.agents.append(agent)

    def remove_agent(self, agent_id: str) -> bool:
        for i, a in enumerate(self.agents):
            if a.agent_id == agent_id:
                self.agents.pop(i)
                return True
        return False

    def add_obstacle(self, obstacle: Obstacle) -> None:
        self.obstacles.append(obstacle)

    def remove_obstacle(self, index: int) -> bool:
        if 0 <= index < len(self.obstacles):
            self.obstacles.pop(index)
            return True
        return False

    def reset(self) -> None:
        """Reset the step counter (agents keep their current state)."""
        self.step_count = 0
