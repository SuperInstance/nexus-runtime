"""
NEXUS Trust Engine — INCREMENTS multi-dimensional trust model.

Trust formula:
    T(a,b,t) = α·T_history + β·T_capability + γ·T_latency + δ·T_consistency

Where:
    T_history    — weighted historical trust (exponential moving average)
    T_capability — assessed capability match for the required task
    T_latency    — communication latency factor
    T_consistency — consistency of past behavior (low variance = high trust)
    α, β, γ, δ  — configurable weights (default: 0.35, 0.25, 0.20, 0.20)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from nexus.trust.increments import (
    TrustDimensions,
    HistoryTracker,
    TrustWeights,
    compute_composite,
)


# ---------------------------------------------------------------------------
# Agent capability profiles
# ---------------------------------------------------------------------------

@dataclass
class CapabilityProfile:
    """Describes an agent's capabilities across various domains."""

    navigation: float = 0.0       # 0.0 - 1.0
    sensing: float = 0.0
    communication: float = 0.0
    computation: float = 0.0
    endurance: float = 0.0
    manipulation: float = 0.0
    payload_capacity: float = 0.0
    speed: float = 0.0

    def match_score(self, required: "CapabilityProfile") -> float:
        """Compute how well this profile matches a required profile (0.0-1.0)."""
        scores = [
            min(self.navigation, required.navigation) if required.navigation > 0 else 1.0,
            min(self.sensing, required.sensing) if required.sensing > 0 else 1.0,
            min(self.communication, required.communication) if required.communication > 0 else 1.0,
            min(self.computation, required.computation) if required.computation > 0 else 1.0,
            min(self.endurance, required.endurance) if required.endurance > 0 else 1.0,
            min(self.manipulation, required.manipulation) if required.manipulation > 0 else 1.0,
            min(self.payload_capacity, required.payload_capacity) if required.payload_capacity > 0 else 1.0,
            min(self.speed, required.speed) if required.speed > 0 else 1.0,
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "navigation": self.navigation,
            "sensing": self.sensing,
            "communication": self.communication,
            "computation": self.computation,
            "endurance": self.endurance,
            "manipulation": self.manipulation,
            "payload_capacity": self.payload_capacity,
            "speed": self.speed,
        }


# ---------------------------------------------------------------------------
# Trust profile and history
# ---------------------------------------------------------------------------

@dataclass
class TrustProfile:
    """Trust state between two agents (trustor → trustee)."""

    trustor_id: str
    trustee_id: str
    composite_score: float = 0.5  # initial neutral trust
    dimensions: TrustDimensions = field(default_factory=TrustDimensions)
    history: HistoryTracker = field(default_factory=HistoryTracker)
    last_updated: float = field(default_factory=time.time)
    interaction_count: int = 0

    # Decay parameters
    decay_rate: float = 0.01  # per second
    max_age_seconds: float = 86400.0  # 24h before max decay

    def decay(self, now: Optional[float] = None) -> float:
        """Apply time-based trust decay. Returns decayed score."""
        if now is None:
            now = time.time()
        elapsed = now - self.last_updated
        if elapsed <= 0:
            return self.composite_score

        # Exponential decay toward 0.5 (neutral)
        decay_factor = 1.0 - min(self.decay_rate * elapsed / self.max_age_seconds, 0.5)
        self.composite_score = 0.5 + (self.composite_score - 0.5) * decay_factor
        self.last_updated = now
        return self.composite_score


# ---------------------------------------------------------------------------
# Agent record
# ---------------------------------------------------------------------------

class AgentStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class AgentRecord:
    """Registry entry for an agent in the trust network."""

    agent_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    status: AgentStatus = AgentStatus.UNKNOWN
    capabilities: CapabilityProfile = field(default_factory=CapabilityProfile)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trust Engine
# ---------------------------------------------------------------------------

class TrustEngine:
    """INCREMENTS trust engine for multi-agent marine robotics.

    Usage::

        engine = TrustEngine()
        engine.register_agent("AUV-001", capabilities=CapabilityProfile(navigation=0.9))
        engine.register_agent("AUV-002", capabilities=CapabilityProfile(sensing=0.8))
        engine.record_interaction("AUV-001", "AUV-002", success=True, latency_ms=50.0)
        score = engine.get_trust("AUV-001", "AUV-002")
    """

    def __init__(
        self,
        weights: Optional[TrustWeights] = None,
        decay_rate: float = 0.01,
        initial_trust: float = 0.5,
    ) -> None:
        self.weights: TrustWeights = (weights or TrustWeights()).normalize()
        self._agents: Dict[str, AgentRecord] = {}
        self._trust_profiles: Dict[Tuple[str, str], TrustProfile] = {}
        self._decay_rate = decay_rate
        self._initial_trust = initial_trust
        self._required_capabilities: Dict[str, CapabilityProfile] = {}

    # ----- agent registry -----

    def register_agent(
        self,
        agent_id: str,
        name: str = "",
        capabilities: Optional[CapabilityProfile] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentRecord:
        """Register a new agent or update an existing one."""
        now = time.time()
        if agent_id in self._agents:
            record = self._agents[agent_id]
            record.name = name or record.name
            record.capabilities = capabilities or record.capabilities
            record.status = AgentStatus.ONLINE
            record.last_seen = now
            if metadata:
                record.metadata.update(metadata)
            return record

        record = AgentRecord(
            agent_id=agent_id,
            name=name or agent_id,
            status=AgentStatus.ONLINE,
            capabilities=capabilities or CapabilityProfile(),
            first_seen=now,
            last_seen=now,
            metadata=metadata or {},
        )
        self._agents[agent_id] = record
        return record

    def unregister_agent(self, agent_id: str) -> bool:
        """Remove an agent. Returns True if it existed."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            # Remove related trust profiles
            keys_to_remove = [k for k in self._trust_profiles if agent_id in k]
            for k in keys_to_remove:
                del self._trust_profiles[k]
            return True
        return False

    def get_agent(self, agent_id: str) -> Optional[AgentRecord]:
        """Look up an agent by ID."""
        return self._agents.get(agent_id)

    def list_agents(self) -> List[AgentRecord]:
        """Return all registered agents."""
        return list(self._agents.values())

    def set_agent_status(self, agent_id: str, status: AgentStatus) -> None:
        """Update an agent's status."""
        if agent_id in self._agents:
            self._agents[agent_id].status = status
            self._agents[agent_id].last_seen = time.time()

    # ----- trust computation -----

    def _get_or_create_profile(self, trustor_id: str, trustee_id: str) -> TrustProfile:
        """Get or create a trust profile between two agents."""
        key = (trustor_id, trustee_id)
        if key not in self._trust_profiles:
            self._trust_profiles[key] = TrustProfile(
                trustor_id=trustor_id,
                trustee_id=trustee_id,
                composite_score=self._initial_trust,
                dimensions=TrustDimensions(),
                history=HistoryTracker(),
                decay_rate=self._decay_rate,
            )
        return self._trust_profiles[key]

    def get_trust(self, trustor_id: str, trustee_id: str) -> float:
        """Get the current composite trust score (trustor → trustee)."""
        profile = self._get_or_create_profile(trustor_id, trustee_id)
        profile.decay()  # apply time-based decay
        return profile.composite_score

    def record_interaction(
        self,
        trustor_id: str,
        trustee_id: str,
        success: bool,
        latency_ms: float = 0.0,
        task_type: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Record an interaction and update trust. Returns new composite score."""
        profile = self._get_or_create_profile(trustor_id, trustee_id)
        profile.decay()

        # Update history
        profile.history.record(success, latency_ms)

        # Compute dimensions
        profile.dimensions.history = profile.history.get_ema() if profile.history.count > 0 else self._initial_trust

        # Capability dimension
        trustor_agent = self._agents.get(trustor_id)
        trustee_agent = self._agents.get(trustee_id)
        if trustee_agent and task_type and task_type in self._required_capabilities:
            profile.dimensions.capability = trustee_agent.capabilities.match_score(
                self._required_capabilities[task_type]
            )
        elif trustee_agent:
            profile.dimensions.capability = 0.7  # default moderate capability

        # Latency dimension
        if latency_ms > 0:
            target_latency = 100.0  # ms
            profile.dimensions.latency = max(0.0, 1.0 - (latency_ms / target_latency))
        else:
            profile.dimensions.latency = 0.5

        # Consistency dimension
        if profile.history.count > 1:
            profile.dimensions.consistency = profile.history.get_consistency()
        else:
            profile.dimensions.consistency = self._initial_trust

        # Composite
        profile.composite_score = compute_composite(
            profile.dimensions, self.weights
        )
        profile.last_updated = time.time()
        profile.interaction_count += 1

        return profile.composite_score

    def set_required_capabilities(self, task_type: str, caps: CapabilityProfile) -> None:
        """Set required capabilities for a task type."""
        self._required_capabilities[task_type] = caps

    def get_trust_profile(self, trustor_id: str, trustee_id: str) -> Optional[TrustProfile]:
        """Get the full trust profile between two agents."""
        key = (trustor_id, trustee_id)
        return self._trust_profiles.get(key)

    def get_most_trusted(
        self,
        trustor_id: str,
        candidate_ids: Optional[List[str]] = None,
        task_type: str = "",
    ) -> Optional[Tuple[str, float]]:
        """Find the most trusted agent for *trustor_id*.

        Returns (agent_id, trust_score) or None if no candidates.
        """
        candidates = candidate_ids or [a.agent_id for a in self._agents.values() if a.agent_id != trustor_id]

        best_id: Optional[str] = None
        best_score = -1.0

        for cid in candidates:
            score = self.get_trust(trustor_id, cid)
            if score > best_score:
                best_score = score
                best_id = cid

        if best_id is not None:
            return (best_id, best_score)
        return None

    def decay_all(self, now: Optional[float] = None) -> None:
        """Apply decay to all trust profiles."""
        for profile in self._trust_profiles.values():
            profile.decay(now)

    @property
    def agent_count(self) -> int:
        return len(self._agents)

    @property
    def profile_count(self) -> int:
        return len(self._trust_profiles)
