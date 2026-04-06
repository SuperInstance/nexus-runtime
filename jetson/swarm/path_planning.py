"""
Path Planning Module
====================
RRT* (optimal rapidly-exploring random tree), Voronoi decomposition
for area coverage, and consensus-based multi-agent planning.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


@dataclass(frozen=True)
class Point:
    """Immutable 2-D point."""
    x: float
    y: float


@dataclass(frozen=True)
class Obstacle:
    """Circular obstacle in the planning space."""
    x: float
    y: float
    radius: float = 5.0


@dataclass
class RRTStarConfig:
    """Configuration for RRT* planner."""
    max_iterations: int = 500
    step_size: float = 5.0
    goal_tolerance: float = 2.0
    rewire_radius: float = 15.0
    bias_factor: float = 0.1  # probability of sampling the goal


# ======================================================================
# RRT* Planner
# ======================================================================

class RRTStarPlanner:
    """
    Optimal Rapidly-exploring Random Tree (RRT*) path planner.

    Features incremental rewiring for asymptotic optimality, path
    smoothing, and multi-agent collision-free planning.
    """

    def __init__(self, config: Optional[RRTStarConfig] = None):
        self.config = config or RRTStarConfig()
        self.tree: List[Point] = []
        self.parents: Dict[int, int] = {}
        self.costs: Dict[int, float] = {}

    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacles: Optional[List[Obstacle]] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[Tuple[float, float]]:
        """
        Plan a path from *start* to *goal* avoiding *obstacles*.
        Returns the path as a list of (x, y) waypoints.
        """
        obstacles = obstacles or []
        x_min, y_min, x_max, y_max = bounds or (0.0, 0.0, 100.0, 100.0)

        self.tree = [Point(*start)]
        self.parents = {0: -1}
        self.costs = {0: 0.0}
        goal_pt = Point(*goal)

        for _ in range(self.config.max_iterations):
            # Sample
            if random.random() < self.config.bias_factor:
                sample = goal_pt
            else:
                sample = Point(
                    random.uniform(x_min, x_max),
                    random.uniform(y_min, y_max),
                )

            # Nearest node
            nearest_idx = self._nearest(sample)
            nearest = self.tree[nearest_idx]

            # Steer
            new_pt = self._steer(nearest, sample)

            # Collision check
            if self._collides(nearest, new_pt, obstacles):
                continue

            new_idx = len(self.tree)
            self.tree.append(new_pt)
            dist = self._dist(nearest, new_pt)

            # Find best parent (rewire)
            best_parent = nearest_idx
            best_cost = self.costs[nearest_idx] + dist

            for i in range(new_idx):
                if self._dist(self.tree[i], new_pt) < self.config.rewire_radius:
                    if self._collides(self.tree[i], new_pt, obstacles):
                        continue
                    c = self.costs[i] + self._dist(self.tree[i], new_pt)
                    if c < best_cost:
                        best_cost = c
                        best_parent = i

            self.parents[new_idx] = best_parent
            self.costs[new_idx] = best_cost

            # Rewire neighbours
            self._rewire(new_idx, obstacles)

            # Check if we reached the goal
            if self._dist(new_pt, goal_pt) < self.config.goal_tolerance:
                goal_idx = len(self.tree)
                self.tree.append(goal_pt)
                self.parents[goal_idx] = new_idx
                self.costs[goal_idx] = best_cost + self._dist(new_pt, goal_pt)
                return self._extract_path(goal_idx)

        # No path found — return best-effort path to nearest node to goal
        best_idx = min(range(len(self.tree)), key=lambda i: self._dist(self.tree[i], goal_pt))
        return self._extract_path(best_idx)

    def rewire(self, node_idx: int, obstacles: List[Obstacle]) -> None:
        """Manually rewire the tree from a given node index."""
        self._rewire(node_idx, obstacles)

    def smooth_path(
        self,
        path: List[Tuple[float, float]],
        obstacles: Optional[List[Obstacle]] = None,
        max_iterations: int = 100,
    ) -> List[Tuple[float, float]]:
        """
        Smooth a path by removing unnecessary waypoints.
        Uses line-of-sight checks to skip intermediate points.
        """
        if len(path) <= 2:
            return list(path)
        obstacles = obstacles or []
        smoothed = [path[0]]
        current = 0
        for _ in range(max_iterations):
            if current >= len(path) - 1:
                break
            # Try to skip ahead
            furthest = current + 1
            for j in range(len(path) - 1, current, -1):
                if not self._segment_collides(
                    path[current], path[j], obstacles
                ):
                    furthest = j
                    break
            smoothed.append(path[furthest])
            current = furthest
        return smoothed

    def plan_multi_agent(
        self,
        starts: List[Tuple[float, float]],
        goals: List[Tuple[float, float]],
        obstacles: Optional[List[Obstacle]] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[List[Tuple[float, float]]]:
        """
        Plan collision-free paths for multiple agents sequentially.
        Earlier agents' paths are treated as dynamic obstacles for later ones.
        """
        obstacles = list(obstacles or [])
        paths: List[List[Tuple[float, float]]] = []
        for start, goal in zip(starts, goals):
            path = self.plan(start, goal, obstacles, bounds)
            paths.append(path)
            # Treat waypoints as small obstacles to avoid collisions
            for px, py in path[1:-1]:
                obstacles.append(Obstacle(px, py, radius=3.0))
        return paths

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _nearest(self, p: Point) -> int:
        return min(range(len(self.tree)), key=lambda i: self._dist(self.tree[i], p))

    def _steer(self, from_pt: Point, to_pt: Point) -> Point:
        dx = to_pt.x - from_pt.x
        dy = to_pt.y - from_pt.y
        dist = math.hypot(dx, dy)
        if dist <= self.config.step_size:
            return to_pt
        scale = self.config.step_size / dist
        return Point(from_pt.x + dx * scale, from_pt.y + dy * scale)

    def _dist(self, a: Point, b: Point) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def _collides(self, a: Point, b: Point, obstacles: List[Obstacle]) -> bool:
        return self._segment_collides((a.x, a.y), (b.x, b.y), obstacles)

    def _segment_collides(
        self, p1: Tuple[float, float], p2: Tuple[float, float], obstacles: List[Obstacle]
    ) -> bool:
        for obs in obstacles:
            if self._point_to_segment_dist(p1, p2, (obs.x, obs.y)) < obs.radius:
                return True
        return False

    @staticmethod
    def _point_to_segment_dist(
        a: Tuple[float, float],
        b: Tuple[float, float],
        p: Tuple[float, float],
    ) -> float:
        ax, ay = a
        bx, by = b
        px, py = p
        dx, dy = bx - ax, by - ay
        len_sq = dx * dx + dy * dy
        if len_sq == 0:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    def _rewire(self, node_idx: int, obstacles: List[Obstacle]) -> None:
        node = self.tree[node_idx]
        for i in range(len(self.tree)):
            if i == node_idx:
                continue
            if self._dist(self.tree[i], node) < self.config.rewire_radius:
                if self._collides(self.tree[i], node, obstacles):
                    continue
                new_cost = self.costs[node_idx] + self._dist(self.tree[i], node)
                if new_cost < self.costs.get(i, float("inf")):
                    self.parents[i] = node_idx
                    self.costs[i] = new_cost

    def _extract_path(self, idx: int) -> List[Tuple[float, float]]:
        path: List[Tuple[float, float]] = []
        while idx >= 0:
            pt = self.tree[idx]
            path.append((pt.x, pt.y))
            idx = self.parents.get(idx, -1)
        path.reverse()
        return path


# ======================================================================
# Voronoi Decomposer
# ======================================================================

class VoronoiDecomposer:
    """
    Area decomposition for multi-agent coverage using Voronoi-like
    partitioning (brute-force nearest-centroid without scipy).
    """

    def __init__(self, bounds: Tuple[float, float, float, float] = (0.0, 0.0, 100.0, 100.0)):
        self.bounds = bounds

    def decompose(
        self,
        agents: List[Tuple[float, float]],
        resolution: float = 1.0,
    ) -> Dict[int, List[Tuple[float, float]]]:
        """
        Assign every grid point to the nearest agent.

        Returns a dict mapping agent index to a list of grid points.
        """
        if not agents:
            return {}
        x_min, y_min, x_max, y_max = self.bounds
        cells: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(len(agents))}
        x = x_min
        while x <= x_max:
            y = y_min
            while y <= y_max:
                nearest = self._nearest_agent(x, y, agents)
                cells[nearest].append((x, y))
                y += resolution
            x += resolution
        return cells

    def compute_centroids(
        self, cells: Dict[int, List[Tuple[float, float]]]
    ) -> Dict[int, Tuple[float, float]]:
        """Compute the centroid of each Voronoi cell."""
        centroids: Dict[int, Tuple[float, float]] = {}
        for idx, pts in cells.items():
            if not pts:
                centroids[idx] = (0.0, 0.0)
                continue
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            centroids[idx] = (cx, cy)
        return centroids

    def lloyd_relaxation(
        self,
        agents: List[Tuple[float, float]],
        iterations: int = 10,
        resolution: float = 2.0,
    ) -> List[Tuple[float, float]]:
        """
        Run Lloyd's relaxation algorithm to iteratively improve agent
        placement for area coverage.

        Returns the relaxed agent positions.
        """
        positions = list(agents)
        for _ in range(iterations):
            cells = self.decompose(positions, resolution)
            centroids = self.compute_centroids(cells)
            positions = [centroids[i] for i in range(len(positions))]
        return positions

    def area_coverage(
        self,
        agents: List[Tuple[float, float]],
        resolution: float = 1.0,
    ) -> Dict[int, float]:
        """
        Compute the area (in square units) of each agent's Voronoi cell.
        """
        cells = self.decompose(agents, resolution)
        areas: Dict[int, float] = {}
        for idx, pts in cells.items():
            areas[idx] = len(pts) * resolution * resolution
        return areas

    @staticmethod
    def _nearest_agent(x: float, y: float, agents: List[Tuple[float, float]]) -> int:
        best = 0
        best_d = float("inf")
        for i, (ax, ay) in enumerate(agents):
            d = math.hypot(x - ax, y - ay)
            if d < best_d:
                best_d = d
                best = i
        return best


# ======================================================================
# Consensus Planner
# ======================================================================

class ConsensusPlanner:
    """
    Consensus-based multi-agent path planning.

    Agents propose plans, vote on them, and conflicts are resolved
    through weighted voting and priority-based tie-breaking.
    """

    def __init__(self, agents: Optional[List[str]] = None):
        self.agents: List[str] = agents or []
        self._proposals: Dict[str, List[Dict]] = {}
        self._votes: Dict[str, Dict[str, int]] = {}
        self._current_plan: Optional[Dict] = None

    def add_agent(self, agent_id: str) -> None:
        if agent_id not in self.agents:
            self.agents.append(agent_id)

    def remove_agent(self, agent_id: str) -> None:
        if agent_id in self.agents:
            self.agents.remove(agent_id)

    def propose_plan(
        self, agent_id: str, plan_id: str, waypoints: List[Tuple[float, float]],
        priority: int = 0,
    ) -> bool:
        """
        An agent proposes a plan.
        Returns True if the agent is registered.
        """
        if agent_id not in self.agents:
            return False
        if plan_id not in self._proposals:
            self._proposals[plan_id] = []
            self._votes[plan_id] = {}
        self._proposals[plan_id].append({
            "agent_id": agent_id,
            "waypoints": waypoints,
            "priority": priority,
        })
        return True

    def vote_on_proposals(
        self, plan_id: str, agent_id: str, proposal_idx: int
    ) -> bool:
        """
        An agent casts a vote for a specific proposal index within a plan.
        Returns True if the vote was recorded.
        """
        if plan_id not in self._proposals or agent_id not in self.agents:
            return False
        proposals = self._proposals[plan_id]
        if proposal_idx < 0 or proposal_idx >= len(proposals):
            return False
        votes = self._votes.setdefault(plan_id, {})
        votes[agent_id] = proposal_idx
        return True

    def resolve_conflicts(self, plan_id: str) -> Optional[Dict]:
        """
        Resolve conflicts by selecting the proposal with the most votes.
        Tie-breaking by highest priority, then earliest agent (lexicographic).
        Returns the winning proposal or None.
        """
        proposals = self._proposals.get(plan_id, [])
        votes = self._votes.get(plan_id, {})
        if not proposals:
            return None

        tally: Dict[int, int] = {i: 0 for i in range(len(proposals))}
        for voter, choice in votes.items():
            if choice in tally:
                tally[choice] += 1

        max_votes = max(tally.values()) if tally else 0
        winners = [i for i, v in tally.items() if v == max_votes]

        # Tie-breaking: highest priority, then earliest agent
        def sort_key(idx: int) -> Tuple[int, str]:
            p = proposals[idx]
            return (-p["priority"], p["agent_id"])

        winners.sort(key=sort_key)
        best = winners[0]
        self._current_plan = {
            "plan_id": plan_id,
            "proposal": proposals[best],
            "votes": max_votes,
        }
        return self._current_plan

    def get_current_plan(self) -> Optional[Dict]:
        return self._current_plan

    def get_proposals(self, plan_id: str) -> List[Dict]:
        return list(self._proposals.get(plan_id, []))

    def get_votes(self, plan_id: str) -> Dict[str, int]:
        return dict(self._votes.get(plan_id, {}))

    def clear(self, plan_id: Optional[str] = None) -> None:
        """Clear proposals and votes. If plan_id is None, clear everything."""
        if plan_id is not None:
            self._proposals.pop(plan_id, None)
            self._votes.pop(plan_id, None)
            if self._current_plan and self._current_plan.get("plan_id") == plan_id:
                self._current_plan = None
        else:
            self._proposals.clear()
            self._votes.clear()
            self._current_plan = None
