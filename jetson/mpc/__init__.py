"""
jetson.mpc — Model Predictive Control module for NEXUS.

Pure-Python MPC formulation, QP solving, trajectory planning,
multi-objective optimisation, constraint handling, and adaptive tuning.
"""

from .formulation import (
    MPCConfig,
    CostFunction,
    MPCProblem,
    MPCFormulator,
    ConstraintType,
    ConstraintSpec,
)
from .solver import (
    QPProblem,
    QPSolution,
    QuadraticProgramSolver,
    InteriorPointSolver,
    ActiveSetSolver,
)
from .trajectory import (
    Waypoint,
    TrajectoryPoint,
    Trajectory,
    Obstacle,
    Pose2D,
    TrajectoryPlanner,
)
from .multi_objective import (
    Objective,
    ObjectivePriority,
    ParetoPoint,
    MultiObjectiveOptimizer,
)
from .constraints import (
    Constraint,
    ConstraintKind,
    ConstraintSet,
    ConstraintHandler,
    MarineConstraints,
    VesselState,
)
from .adaptive import (
    HorizonConfig,
    PerformanceMetrics,
    AdaptedParameters,
    AdaptiveController,
)

__all__ = [
    # formulation
    "MPCConfig", "CostFunction", "MPCProblem", "MPCFormulator",
    "ConstraintType", "ConstraintSpec",
    # solver
    "QPProblem", "QPSolution", "QuadraticProgramSolver",
    "InteriorPointSolver", "ActiveSetSolver",
    # trajectory
    "Waypoint", "TrajectoryPoint", "Trajectory", "Obstacle", "Pose2D",
    "TrajectoryPlanner",
    # multi_objective
    "Objective", "ObjectivePriority", "ParetoPoint",
    "MultiObjectiveOptimizer",
    # constraints
    "Constraint", "ConstraintKind", "ConstraintSet",
    "ConstraintHandler", "MarineConstraints", "VesselState",
    # adaptive
    "HorizonConfig", "PerformanceMetrics", "AdaptedParameters",
    "AdaptiveController",
]
