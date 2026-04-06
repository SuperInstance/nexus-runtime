"""
NEXUS Swarm Intelligence Module
================================
Marine robotics swarm coordination including:
- Formation control (LINE, WEDGE, CIRCLE, GRID, V_SHAPE)
- Reynolds flocking (separation, alignment, cohesion)
- Contract-net task allocation and auction-based dispatch
- RRT*, Voronoi, and consensus-based path planning
- Swarm health metrics and diagnostics
"""

from .formation import FormationType, FormationController
from .flocking import FlockingParams, FlockingBehavior, FlockSimulation
from .task_allocation import Task, TaskPriority, TaskType, ContractNetProtocol, AuctionEngine
from .path_planning import RRTStarPlanner, VoronoiDecomposer, ConsensusPlanner
from .metrics import SwarmMetrics

__all__ = [
    # Formation
    "FormationType",
    "FormationController",
    # Flocking
    "FlockingParams",
    "FlockingBehavior",
    "FlockSimulation",
    # Task Allocation
    "Task",
    "TaskPriority",
    "TaskType",
    "ContractNetProtocol",
    "AuctionEngine",
    # Path Planning
    "RRTStarPlanner",
    "VoronoiDecomposer",
    "ConsensusPlanner",
    # Metrics
    "SwarmMetrics",
]
