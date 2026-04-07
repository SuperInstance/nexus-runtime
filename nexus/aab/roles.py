"""
NEXUS Role Management — role definitions, matching, assignment, and rotation.

Roles define functional responsibilities in a marine swarm:
    PILOT, SENSOR_OPERATOR, NAVIGATOR, COMM_RELAY, SAFETY_OFFICER,
    MAPPING_SPECIALIST, SEARCH_COORDINATOR, RESCUE_LEAD, DATA_COLLECTOR,
    COMMS_HUB, DECK_ENGINEER, MISSION_COMMANDER
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Role types
# ---------------------------------------------------------------------------

class RoleType(enum.Enum):
    """Marine robotics role types."""

    PILOT = "pilot"
    SENSOR_OPERATOR = "sensor_operator"
    NAVIGATOR = "navigator"
    COMM_RELAY = "comm_relay"
    SAFETY_OFFICER = "safety_officer"
    MAPPING_SPECIALIST = "mapping_specialist"
    SEARCH_COORDINATOR = "search_coordinator"
    RESCUE_LEAD = "rescue_lead"
    DATA_COLLECTOR = "data_collector"
    COMMS_HUB = "comms_hub"
    DECK_ENGINEER = "deck_engineer"
    MISSION_COMMANDER = "mission_commander"


# ---------------------------------------------------------------------------
# Role definition
# ---------------------------------------------------------------------------

@dataclass
class Role:
    """Definition of a role with required capabilities."""

    role_type: RoleType
    name: str = ""
    description: str = ""
    required_capabilities: Dict[str, float] = field(default_factory=dict)
    priority: int = 5  # 0-10, higher = more important
    is_singleton: bool = False  # only one agent can hold this role
    max_assignees: int = 1  # max agents that can hold this role simultaneously
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.role_type.value.replace("_", " ").title()

    def matches_capabilities(self, agent_caps: Dict[str, float]) -> float:
        """Compute how well agent capabilities match this role (0.0-1.0)."""
        if not self.required_capabilities:
            return 1.0
        scores: List[float] = []
        for domain, required in self.required_capabilities.items():
            if required <= 0:
                scores.append(1.0)
            else:
                agent_val = agent_caps.get(domain, 0.0)
                scores.append(min(agent_val / required, 1.0))
        return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Role assignment
# ---------------------------------------------------------------------------

@dataclass
class RoleAssignment:
    """Tracks a role assigned to an agent."""

    assignment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    role: Role = field(default_factory=lambda: Role(role_type=RoleType.PILOT))
    agent_id: str = ""
    assigned_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    active: bool = True

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def __repr__(self) -> str:
        return f"RoleAssignment({self.role.role_type.value} → {self.agent_id})"


# ---------------------------------------------------------------------------
# Role Registry
# ---------------------------------------------------------------------------

class RoleRegistry:
    """Registry for role definitions and assignments.

    Usage::

        registry = RoleRegistry()
        registry.define_role(Role(role_type=RoleType.PILOT, required_capabilities={"navigation": 0.8}))
        registry.assign_role(RoleType.PILOT, "AUV-001")
        best = registry.find_best_candidate(RoleType.PILOT, candidates_caps)
    """

    def __init__(self) -> None:
        self._roles: Dict[RoleType, Role] = {}
        self._assignments: Dict[str, List[RoleAssignment]] = {}  # agent_id -> assignments
        self._role_assignments: Dict[RoleType, List[RoleAssignment]] = {}  # role -> assignments
        self._rotation_history: List[Tuple[str, RoleType, RoleType, float]] = []

        # Define default roles
        self._define_defaults()

    def _define_defaults(self) -> None:
        """Define built-in marine robotics roles."""
        defaults = {
            RoleType.PILOT: Role(
                role_type=RoleType.PILOT,
                required_capabilities={"navigation": 0.8, "speed": 0.6},
                priority=8, is_singleton=False, max_assignees=5,
            ),
            RoleType.SENSOR_OPERATOR: Role(
                role_type=RoleType.SENSOR_OPERATOR,
                required_capabilities={"sensing": 0.7, "computation": 0.5},
                priority=7, is_singleton=False, max_assignees=10,
            ),
            RoleType.NAVIGATOR: Role(
                role_type=RoleType.NAVIGATOR,
                required_capabilities={"navigation": 0.9, "communication": 0.6},
                priority=9, is_singleton=True, max_assignees=1,
            ),
            RoleType.COMM_RELAY: Role(
                role_type=RoleType.COMM_RELAY,
                required_capabilities={"communication": 0.8},
                priority=6, is_singleton=False, max_assignees=5,
            ),
            RoleType.SAFETY_OFFICER: Role(
                role_type=RoleType.SAFETY_OFFICER,
                required_capabilities={"navigation": 0.6, "sensing": 0.6, "communication": 0.7},
                priority=10, is_singleton=True, max_assignees=1,
            ),
            RoleType.MAPPING_SPECIALIST: Role(
                role_type=RoleType.MAPPING_SPECIALIST,
                required_capabilities={"sensing": 0.8, "computation": 0.7},
                priority=5, is_singleton=False, max_assignees=3,
            ),
            RoleType.SEARCH_COORDINATOR: Role(
                role_type=RoleType.SEARCH_COORDINATOR,
                required_capabilities={"navigation": 0.7, "communication": 0.8, "computation": 0.6},
                priority=8, is_singleton=True, max_assignees=1,
            ),
            RoleType.RESCUE_LEAD: Role(
                role_type=RoleType.RESCUE_LEAD,
                required_capabilities={"navigation": 0.8, "manipulation": 0.7, "communication": 0.7},
                priority=10, is_singleton=True, max_assignees=1,
            ),
            RoleType.DATA_COLLECTOR: Role(
                role_type=RoleType.DATA_COLLECTOR,
                required_capabilities={"sensing": 0.6, "computation": 0.5, "payload_capacity": 0.5},
                priority=4, is_singleton=False, max_assignees=10,
            ),
            RoleType.COMMS_HUB: Role(
                role_type=RoleType.COMMS_HUB,
                required_capabilities={"communication": 0.9, "computation": 0.7},
                priority=7, is_singleton=True, max_assignees=1,
            ),
            RoleType.DECK_ENGINEER: Role(
                role_type=RoleType.DECK_ENGINEER,
                required_capabilities={"manipulation": 0.8, "computation": 0.6},
                priority=5, is_singleton=False, max_assignees=3,
            ),
            RoleType.MISSION_COMMANDER: Role(
                role_type=RoleType.MISSION_COMMANDER,
                required_capabilities={"navigation": 0.7, "communication": 0.9, "computation": 0.8},
                priority=10, is_singleton=True, max_assignees=1,
            ),
        }
        self._roles.update(defaults)

    # ----- role definition -----

    def define_role(self, role: Role) -> None:
        """Register a role definition."""
        self._roles[role.role_type] = role

    def get_role(self, role_type: RoleType) -> Optional[Role]:
        """Get a role definition."""
        return self._roles.get(role_type)

    def list_roles(self) -> List[Role]:
        """List all defined roles."""
        return list(self._roles.values())

    # ----- assignment -----

    def assign_role(
        self,
        role_type: RoleType,
        agent_id: str,
        expires_at: Optional[float] = None,
    ) -> Optional[RoleAssignment]:
        """Assign a role to an agent. Returns the assignment or None on conflict."""
        role = self._roles.get(role_type)
        if role is None:
            return None

        current = self.get_assignments_for_role(role_type)
        active_count = sum(1 for a in current if a.active and not a.is_expired)

        if role.is_singleton and active_count >= 1:
            return None
        if active_count >= role.max_assignees:
            return None

        assignment = RoleAssignment(
            role=role,
            agent_id=agent_id,
            expires_at=expires_at,
        )

        self._assignments.setdefault(agent_id, []).append(assignment)
        self._role_assignments.setdefault(role_type, []).append(assignment)
        return assignment

    def release_role(self, agent_id: str, role_type: RoleType) -> bool:
        """Release a role assignment. Returns True if found."""
        assignments = self._assignments.get(agent_id, [])
        for a in assignments:
            if a.role.role_type == role_type and a.active:
                a.active = False
                return True
        return False

    def release_all_roles(self, agent_id: str) -> int:
        """Release all roles for an agent. Returns count of released."""
        assignments = self._assignments.get(agent_id, [])
        count = 0
        for a in assignments:
            if a.active:
                a.active = False
                count += 1
        return count

    def get_agent_roles(self, agent_id: str) -> List[RoleAssignment]:
        """Get all role assignments for an agent."""
        return [a for a in self._assignments.get(agent_id, []) if a.active and not a.is_expired]

    def get_assignments_for_role(self, role_type: RoleType) -> List[RoleAssignment]:
        """Get all active assignments for a role."""
        return [a for a in self._role_assignments.get(role_type, []) if a.active and not a.is_expired]

    # ----- matching -----

    def find_best_candidate(
        self,
        role_type: RoleType,
        candidates: Dict[str, Dict[str, float]],
    ) -> Optional[Tuple[str, float]]:
        """Find the best candidate for a role.

        Parameters
        ----------
        role_type : RoleType
            The role to fill.
        candidates : dict
            {agent_id: {capability_domain: score}} for each candidate.

        Returns
        -------
        (agent_id, match_score) or None
        """
        role = self._roles.get(role_type)
        if role is None:
            return None

        best_id: Optional[str] = None
        best_score = -1.0

        already_assigned = {a.agent_id for a in self.get_assignments_for_role(role_type)}

        for agent_id, caps in candidates.items():
            if agent_id in already_assigned:
                continue
            score = role.matches_capabilities(caps)
            if score > best_score:
                best_score = score
                best_id = agent_id

        if best_id is not None and best_score > 0:
            return (best_id, best_score)
        return None

    # ----- rotation -----

    def rotate_role(self, role_type: RoleType, new_agent_id: str) -> bool:
        """Rotate a singleton role to a new agent."""
        current = self.get_assignments_for_role(role_type)
        for a in current:
            a.active = False
            self._rotation_history.append((a.agent_id, role_type, role_type, time.time()))

        assignment = self.assign_role(role_type, new_agent_id)
        return assignment is not None

    def get_rotation_history(self) -> List[Tuple[str, RoleType, RoleType, float]]:
        """Get the history of role rotations."""
        return list(self._rotation_history)
