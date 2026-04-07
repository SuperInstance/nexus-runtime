"""NEXUS Autonomous Agent Behavior (AAB) framework for marine multi-agent coordination."""

from nexus.aab.behavior import (
    BehaviorEngine, BehaviorState, A2AOpcodes,
    A2AMessage, A2AResponse,
)
from nexus.aab.roles import (
    Role, RoleType, RoleRegistry, RoleAssignment,
)

__all__ = [
    "BehaviorEngine", "BehaviorState", "A2AOpcodes",
    "A2AMessage", "A2AResponse",
    "Role", "RoleType", "RoleRegistry", "RoleAssignment",
]
