"""
Swarm Metrics Module
====================
Computes health, efficiency, robustness, and other diagnostics for
a marine robot swarm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class AgentSnapshot:
    """Immutable snapshot of an agent's current state."""
    agent_id: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    energy: float = 100.0
    connected: bool = True


class SwarmMetrics:
    """
    Computes swarm-level metrics for monitoring and diagnostics.

    Metrics include spatial spread, velocity alignment, cohesion,
    communication connectivity, task efficiency, and overall robustness.
    """

    def __init__(self, comm_range: float = 50.0, cohesion_threshold: float = 30.0):
        self.comm_range = comm_range
        self.cohesion_threshold = cohesion_threshold

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def compute_spread(self, agents: List[AgentSnapshot]) -> float:
        """
        Spatial spread: standard deviation of distances from centroid.
        Returns 0.0 for fewer than 2 agents.
        """
        if len(agents) < 2:
            return 0.0
        cx, cy = self._centroid(agents)
        dists = [math.hypot(a.x - cx, a.y - cy) for a in agents]
        mean_d = sum(dists) / len(dists)
        variance = sum((d - mean_d) ** 2 for d in dists) / len(dists)
        return math.sqrt(variance)

    def compute_alignment(self, agents: List[AgentSnapshot]) -> float:
        """
        Velocity alignment: average cosine similarity of agent headings.
        Returns 1.0 for fewer than 2 agents (perfect alignment).
        Returns 0.0 if all agents are stationary.
        """
        if len(agents) < 2:
            return 1.0
        speeds = [math.hypot(a.vx, a.vy) for a in agents]
        total_speed = sum(speeds)
        if total_speed == 0:
            return 0.0
        avg_vx = sum(a.vx for a in agents) / len(agents)
        avg_vy = sum(a.vy for a in agents) / len(agents)
        avg_speed = total_speed / len(agents)
        if avg_speed == 0:
            return 0.0
        mag = math.hypot(avg_vx, avg_vy)
        return min(mag / avg_speed, 1.0)

    def compute_cohesion(self, agents: List[AgentSnapshot]) -> float:
        """
        Cohesion metric: inverse of average distance from centroid,
        normalised to [0, 1] using the cohesion_threshold.
        """
        if len(agents) < 2:
            return 1.0
        cx, cy = self._centroid(agents)
        avg_dist = sum(math.hypot(a.x - cx, a.y - cy) for a in agents) / len(agents)
        return max(0.0, 1.0 - avg_dist / self.cohesion_threshold)

    def compute_connectivity(self, agents: List[AgentSnapshot]) -> float:
        """
        Communication connectivity: fraction of connected agent pairs
        within comm_range.
        """
        if len(agents) < 2:
            return 1.0
        connected_agents = [a for a in agents if a.connected]
        if len(connected_agents) < 2:
            return 0.0
        total_pairs = len(connected_agents) * (len(connected_agents) - 1) / 2
        connected_pairs = 0
        for i in range(len(connected_agents)):
            for j in range(i + 1, len(connected_agents)):
                d = math.hypot(
                    connected_agents[i].x - connected_agents[j].x,
                    connected_agents[i].y - connected_agents[j].y,
                )
                if d <= self.comm_range:
                    connected_pairs += 1
        if total_pairs == 0:
            return 1.0
        return connected_pairs / total_pairs

    def compute_efficiency(
        self,
        agents: List[AgentSnapshot],
        tasks_completed: int = 0,
        tasks_total: int = 0,
    ) -> float:
        """
        Swarm efficiency: composite of alignment, cohesion, and task
        completion rate on [0, 1].
        """
        if not agents:
            return 0.0
        alignment = self.compute_alignment(agents)
        cohesion = self.compute_cohesion(agents)
        task_rate = tasks_completed / tasks_total if tasks_total > 0 else 1.0
        return 0.4 * alignment + 0.3 * cohesion + 0.3 * task_rate

    def compute_robustness(self, agents: List[AgentSnapshot]) -> float:
        """
        Robustness: how well the swarm maintains structure with failures.
        Based on connectivity redundancy and agent energy levels.
        """
        if not agents:
            return 0.0
        connected = [a for a in agents if a.connected]
        if not connected:
            return 0.0
        # Average energy
        avg_energy = sum(a.energy for a in agents) / len(agents) / 100.0
        # Connectivity of connected subset
        conn = self.compute_connectivity(connected)
        # Redundancy: fraction of connected agents
        redundancy = len(connected) / len(agents)
        return 0.4 * conn + 0.3 * min(avg_energy, 1.0) + 0.3 * redundancy

    def compute_swarm_health(
        self,
        agents: List[AgentSnapshot],
        tasks_completed: int = 0,
        tasks_total: int = 0,
    ) -> Dict[str, float]:
        """
        Compute a full health report: spread, alignment, cohesion,
        connectivity, efficiency, robustness.
        """
        if not agents:
            return {
                "spread": 0.0,
                "alignment": 0.0,
                "cohesion": 0.0,
                "connectivity": 0.0,
                "efficiency": 0.0,
                "robustness": 0.0,
                "num_agents": 0,
            }
        return {
            "spread": self.compute_spread(agents),
            "alignment": self.compute_alignment(agents),
            "cohesion": self.compute_cohesion(agents),
            "connectivity": self.compute_connectivity(agents),
            "efficiency": self.compute_efficiency(agents, tasks_completed, tasks_total),
            "robustness": self.compute_robustness(agents),
            "num_agents": len(agents),
        }

    def generate_report(
        self,
        agents: List[AgentSnapshot],
        tasks_completed: int = 0,
        tasks_total: int = 0,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive swarm diagnostic report.
        """
        health = self.compute_swarm_health(agents, tasks_completed, tasks_total)
        status = "OPTIMAL"
        score = (health["alignment"] + health["cohesion"]
                 + health["connectivity"] + health["efficiency"]
                 + health["robustness"]) / 5.0

        if score < 0.3:
            status = "CRITICAL"
        elif score < 0.6:
            status = "DEGRADED"
        elif score < 0.8:
            status = "NOMINAL"

        return {
            "status": status,
            "overall_score": round(score, 4),
            "metrics": health,
            "recommendations": self._recommendations(health),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _centroid(agents: List[AgentSnapshot]) -> Tuple[float, float]:
        if not agents:
            return (0.0, 0.0)
        cx = sum(a.x for a in agents) / len(agents)
        cy = sum(a.y for a in agents) / len(agents)
        return (cx, cy)

    @staticmethod
    def _recommendations(health: Dict[str, float]) -> List[str]:
        recs: List[str] = []
        if health["alignment"] < 0.5:
            recs.append("Low alignment: agents are heading in different directions")
        if health["cohesion"] < 0.5:
            recs.append("Low cohesion: agents are too spread out")
        if health["connectivity"] < 0.5:
            recs.append("Low connectivity: communication links are fragile")
        if health["efficiency"] < 0.5:
            recs.append("Low efficiency: consider re-allocating tasks")
        if health["robustness"] < 0.5:
            recs.append("Low robustness: swarm is vulnerable to failures")
        if not recs:
            recs.append("Swarm is operating within normal parameters")
        return recs
