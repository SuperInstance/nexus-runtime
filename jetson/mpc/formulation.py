"""
MPC Formulation — cost functions, constraints, problem definition.

Pure Python implementation using math, dataclasses, enum.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Callable


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MPCConfig:
    """Configuration for a Model Predictive Controller."""
    horizon: int = 10
    dt: float = 0.1
    state_dim: int = 4
    control_dim: int = 2
    weights: Optional[List[float]] = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = [1.0] * self.state_dim


class ConstraintType(Enum):
    EQUALITY = "equality"
    INEQUALITY = "inequality"
    BOUND = "bound"


@dataclass
class CostFunction:
    """Quadratic cost specification for MPC."""
    state_weight_matrix: List[List[float]] = field(default_factory=list)
    control_weight_matrix: List[List[float]] = field(default_factory=list)
    terminal_weight: List[List[float]] = field(default_factory=list)

    def __post_init__(self):
        if not self.state_weight_matrix:
            self.state_weight_matrix = [[1.0]]
        if not self.control_weight_matrix:
            self.control_weight_matrix = [[1.0]]
        if not self.terminal_weight:
            self.terminal_weight = [[1.0]]


@dataclass
class ConstraintSpec:
    """Single constraint specification."""
    name: str = ""
    ctype: ConstraintType = ConstraintType.INEQUALITY
    lower: float = 0.0
    upper: float = float("inf")
    matrix: List[List[float]] = field(default_factory=list)


@dataclass
class MPCProblem:
    """Complete MPC optimisation problem."""
    current_state: List[float] = field(default_factory=list)
    reference_trajectory: List[List[float]] = field(default_factory=list)
    constraints: List[ConstraintSpec] = field(default_factory=list)
    cost: CostFunction = field(default_factory=CostFunction)
    horizon: int = 10
    state_dim: int = 4
    control_dim: int = 2


# ---------------------------------------------------------------------------
# Helper matrix ops (pure Python)
# ---------------------------------------------------------------------------

def _mat_zero(n: int, m: int) -> List[List[float]]:
    return [[0.0] * m for _ in range(n)]


def _mat_eye(n: int) -> List[List[float]]:
    M = _mat_zero(n, n)
    for i in range(n):
        M[i][i] = 1.0
    return M


def _mat_scale(M: List[List[float]], s: float) -> List[List[float]]:
    return [[s * M[i][j] for j in range(len(M[0]))] for i in range(len(M))]


def _vec_dot(a: List[float], b: List[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def _vec_sub(a: List[float], b: List[float]) -> List[float]:
    return [ai - bi for ai, bi in zip(a, b)]


def _vec_norm(a: List[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


# ---------------------------------------------------------------------------
# MPCFormulator
# ---------------------------------------------------------------------------

class MPCFormulator:
    """Build MPCProblem instances from various specifications."""

    def __init__(self, config: Optional[MPCConfig] = None):
        self.config = config or MPCConfig()

    # ----- public API -----

    def formulate_tracking(
        self,
        reference: List[List[float]],
        weights: Optional[List[float]] = None,
    ) -> MPCProblem:
        """Create a trajectory-tracking MPC problem."""
        w = weights or self.config.weights
        n = len(w)
        Q = _mat_eye(n)
        for i in range(n):
            Q[i][i] = w[i] if i < len(w) else 1.0
        m = self.config.control_dim
        R = [[1.0] * m for _ in range(m)]
        P = _mat_scale(Q, 2.0)
        cost = CostFunction(
            state_weight_matrix=Q,
            control_weight_matrix=R,
            terminal_weight=P,
        )
        current = reference[0] if reference else [0.0] * n
        return MPCProblem(
            current_state=current,
            reference_trajectory=list(reference),
            constraints=[],
            cost=cost,
            horizon=self.config.horizon,
            state_dim=n,
            control_dim=m,
        )

    def formulate_regulation(
        self,
        setpoint: List[float],
        weights: Optional[List[float]] = None,
    ) -> MPCProblem:
        """Create a regulation (setpoint-tracking) MPC problem."""
        w = weights or self.config.weights
        n = len(setpoint)
        Q = _mat_eye(n)
        for i in range(n):
            Q[i][i] = w[i] if i < len(w) else 1.0
        m = self.config.control_dim
        R = [[0.1] * m for _ in range(m)]
        P = _mat_scale(Q, 10.0)
        cost = CostFunction(
            state_weight_matrix=Q,
            control_weight_matrix=R,
            terminal_weight=P,
        )
        ref = [list(setpoint)] * (self.config.horizon + 1)
        return MPCProblem(
            current_state=[0.0] * n,
            reference_trajectory=ref,
            constraints=[],
            cost=cost,
            horizon=self.config.horizon,
            state_dim=n,
            control_dim=m,
        )

    def formulate_economic(
        self,
        objective_fn: Callable[[List[float], List[float]], float],
        constraints: List[ConstraintSpec],
    ) -> MPCProblem:
        """Create an economic-MPC problem (non-quadratic stage cost wrapped)."""
        n = self.config.state_dim
        m = self.config.control_dim
        Q = _mat_eye(n)
        R = [[1.0] * m for _ in range(m)]
        P = _mat_eye(n)
        cost = CostFunction(
            state_weight_matrix=Q,
            control_weight_matrix=R,
            terminal_weight=P,
        )
        return MPCProblem(
            current_state=[0.0] * n,
            reference_trajectory=[[0.0] * n] * (self.config.horizon + 1),
            constraints=constraints,
            cost=cost,
            horizon=self.config.horizon,
            state_dim=n,
            control_dim=m,
        )

    def compute_stage_cost(
        self,
        state: List[float],
        control: List[float],
        reference: List[float],
        weights: List[float],
    ) -> float:
        """Quadratic stage cost  (x-x_ref)^T Q (x-x_ref) + u^T R u."""
        err = _vec_sub(state, reference)
        w = weights[: len(err)] if len(weights) >= len(err) else weights + [1.0] * (len(err) - len(weights))
        state_cost = sum(w[i] * err[i] ** 2 for i in range(len(err)))
        control_cost = sum(u * u for u in control)
        return state_cost + control_cost

    def compute_terminal_cost(
        self,
        state: List[float],
        terminal_weight: List[List[float]],
    ) -> float:
        """Quadratic terminal cost  x^T P x."""
        n = len(state)
        total = 0.0
        for i in range(min(n, len(terminal_weight))):
            for j in range(min(n, len(terminal_weight[i]))):
                total += terminal_weight[i][j] * state[i] * state[j]
        return total

    def build_cost_matrices(
        self,
        horizon: int,
        state_dim: int,
        control_dim: int,
    ) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        """
        Return (Q, R, P) block-diagonal cost matrices suitable for the
        condensed QP.

        Q is (horizon+1)*state_dim × state_dim diagonal,
        R is horizon*control_dim × control_dim diagonal,
        P is state_dim × state_dim terminal weight.
        """
        n = state_dim
        m = control_dim
        rows_Q = (horizon + 1) * n
        Q = _mat_eye(rows_Q)
        for i in range(rows_Q):
            Q[i][i] = 1.0
        rows_R = horizon * m
        R = [[0.1] * rows_R for _ in range(rows_R)]
        for i in range(rows_R):
            R[i][i] = 0.1
        P = _mat_eye(n)
        for i in range(n):
            P[i][i] = 10.0
        return Q, R, P
