"""NEXUS Trust Engine - Trust propagation across hierarchy layers.

Handles trust propagation between edge, agent, and fleet layers.

Rules:
  - Trust earned on edge propagates UP (edge -> agent -> fleet)
  - Trust lost at fleet propagates DOWN (fleet -> agent -> edge)
  - Attenuation: 0.85x per hop
  - Maximum propagation radius: 3 hops
  - Fleet trust is weighted average of all vessel trusts
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Propagation constants
DEFAULT_ATTENUATION = 0.85
MAX_PROPAGATION_RADIUS = 3
TRUST_MERGE_ALPHA = 0.3  # blending factor for propagated trust


@dataclass
class VesselTrust:
    """Trust state for a single vessel (edge node)."""

    vessel_id: str
    subsystems: dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0

    def get_composite_score(self) -> float:
        """Compute composite trust score as average of all subsystem scores."""
        if not self.subsystems:
            return 0.0
        return sum(self.subsystems.values()) / len(self.subsystems)

    def get_min_score(self) -> float:
        """Get minimum subsystem score (weakest link)."""
        if not self.subsystems:
            return 0.0
        return min(self.subsystems.values())


@dataclass
class AgentTrust:
    """Trust state for an agent (mid-level orchestrator)."""

    agent_id: str
    direct_trust: dict[str, float] = field(default_factory=dict)
    propagated_from_edges: dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0

    def get_composite_score(self) -> float:
        """Compute composite agent trust."""
        all_scores = list(self.direct_trust.values()) + list(
            self.propagated_from_edges.values()
        )
        if not all_scores:
            return 0.0
        return sum(all_scores) / len(all_scores)


@dataclass
class FleetTrust:
    """Trust state for the entire fleet."""

    fleet_id: str = "default_fleet"
    vessel_scores: dict[str, float] = field(default_factory=dict)
    composite_score: float = 0.0
    timestamp: float = 0.0

    def get_vessel_count(self) -> int:
        return len(self.vessel_scores)

    def get_avg_score(self) -> float:
        if not self.vessel_scores:
            return 0.0
        return sum(self.vessel_scores.values()) / len(self.vessel_scores)

    def get_min_score(self) -> float:
        if not self.vessel_scores:
            return 0.0
        return min(self.vessel_scores.values())


class TrustPropagator:
    """Handles trust propagation between edge, agent, and fleet layers.

    Trust hierarchy:
      Fleet (top)
        -> Agent 1, Agent 2, ...
           -> Edge 1, Edge 2, ... (vessels)

    Propagation rules:
      - Up: edge trust propagates to agent and fleet (attenuated per hop)
      - Down: fleet trust directives propagate to agents and edges (attenuated)
      - Attenuation factor: 0.85 per hop (configurable)
      - Max radius: 3 hops
    """

    def __init__(
        self,
        attenuation: float = DEFAULT_ATTENUATION,
        merge_alpha: float = TRUST_MERGE_ALPHA,
        max_radius: int = MAX_PROPAGATION_RADIUS,
    ) -> None:
        self.attenuation = attenuation
        self.merge_alpha = merge_alpha
        self.max_radius = max_radius

    def propagate_up(
        self,
        edge_trust: dict[str, float],
        current_agent_trust: dict[str, float],
        current_fleet_trust: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Propagate edge trust up to agent and fleet levels.

        Args:
            edge_trust: dict of subsystem_name -> trust_score from edge node
            current_agent_trust: current agent-level trust scores
            current_fleet_trust: current fleet-level trust scores

        Returns:
            Tuple of (updated_agent_trust, updated_fleet_trust)

        Trust propagation:
          edge -> agent: attenuated by 1 hop (0.85x)
          edge -> fleet: attenuated by 2 hops (0.85^2 = 0.7225x)

        Merge formula:
          agent_score = (1 - alpha) * current_agent + alpha * (edge * atten)
        """
        updated_agent = dict(current_agent_trust)
        updated_fleet = dict(current_fleet_trust)

        if not edge_trust:
            return updated_agent, updated_fleet

        agent_atten = self.attenuation ** 1  # 1 hop: edge -> agent
        fleet_atten = self.attenuation ** 2  # 2 hops: edge -> agent -> fleet

        for subsystem, edge_score in edge_trust.items():
            propagated_to_agent = edge_score * agent_atten

            # Merge with existing agent trust
            if subsystem in updated_agent:
                merged = (
                    (1 - self.merge_alpha) * updated_agent[subsystem]
                    + self.merge_alpha * propagated_to_agent
                )
                updated_agent[subsystem] = merged
            else:
                updated_agent[subsystem] = propagated_to_agent

            # Propagate further to fleet (2 hops)
            propagated_to_fleet = edge_score * fleet_atten
            if subsystem in updated_fleet:
                merged = (
                    (1 - self.merge_alpha) * updated_fleet[subsystem]
                    + self.merge_alpha * propagated_to_fleet
                )
                updated_fleet[subsystem] = merged
            else:
                updated_fleet[subsystem] = propagated_to_fleet

        return updated_agent, updated_fleet

    def propagate_down(
        self,
        fleet_directive: dict[str, float],
        current_agent_trust: dict[str, float],
        current_edge_trust: dict[str, float],
    ) -> dict[str, float]:
        """Propagate fleet-level trust changes down to edge.

        Args:
            fleet_directive: dict of subsystem_name -> trust_score from fleet
            current_agent_trust: current agent-level trust scores
            current_edge_trust: current edge-level trust scores

        Returns:
            Updated edge trust scores.

        Trust propagation:
          fleet -> agent: attenuated by 1 hop
          agent -> edge: attenuated by 1 more hop

        For DOWN propagation, fleet directive acts as a trust ceiling:
          edge_score = min(current_edge, fleet_directive * attenuation^2)

        Empty fleet directive returns edge trust unchanged.
        """
        if not fleet_directive:
            return dict(current_edge_trust)

        updated_edge = dict(current_edge_trust)

        # Total attenuation: fleet -> agent -> edge = 2 hops
        total_atten = self.attenuation ** 2

        for subsystem, fleet_score in fleet_directive.items():
            propagated = fleet_score * total_atten
            if subsystem in updated_edge:
                updated_edge[subsystem] = min(updated_edge[subsystem], propagated)
            else:
                updated_edge[subsystem] = propagated

        return updated_edge

    def compute_fleet_trust(
        self,
        vessel_trusts: list[dict[str, float]],
        weights: list[float] | None = None,
    ) -> dict[str, float]:
        """Compute fleet-level trust as weighted average of all vessels.

        Args:
            vessel_trusts: list of dicts, each mapping subsystem -> trust_score
            weights: optional per-vessel weights. If None, equal weighting.

        Returns:
            Dict mapping subsystem_name -> fleet_trust_score

        Raises:
            ValueError: if vessel_trusts is empty
        """
        if not vessel_trusts:
            return {}

        n_vessels = len(vessel_trusts)
        if weights is None:
            weights = [1.0 / n_vessels] * n_vessels

        if len(weights) != n_vessels:
            raise ValueError(
                f"Weight count ({len(weights)}) must match vessel count ({n_vessels})"
            )

        total_weight = sum(weights)
        if total_weight == 0:
            return {}

        # Collect all subsystem names across all vessels
        all_subsystems: set[str] = set()
        for vt in vessel_trusts:
            all_subsystems.update(vt.keys())

        # Compute weighted average per subsystem
        fleet_scores: dict[str, float] = {}
        for subsystem in all_subsystems:
            weighted_sum = 0.0
            for vessel_trust, weight in zip(vessel_trusts, weights):
                score = vessel_trust.get(subsystem, 0.0)
                weighted_sum += score * weight
            fleet_scores[subsystem] = weighted_sum / total_weight

        return fleet_scores

    def propagate_single_hop(
        self,
        source_trust: dict[str, float],
        attenuation: float | None = None,
    ) -> dict[str, float]:
        """Apply single-hop attenuation to trust scores.

        Args:
            source_trust: dict of subsystem -> score
            attenuation: attenuation factor (default: self.attenuation)

        Returns:
            Attenuated trust scores.
        """
        atten = attenuation if attenuation is not None else self.attenuation
        return {k: v * atten for k, v in source_trust.items()}

    def compute_attenuation(self, hops: int) -> float:
        """Compute total attenuation factor for given number of hops.

        Args:
            hops: number of hops (max: max_radius)

        Returns:
            Attenuation factor (e.g., 0.85^2 = 0.7225 for 2 hops)
        """
        return self.attenuation ** min(hops, self.max_radius)
