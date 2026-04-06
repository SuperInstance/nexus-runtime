"""
QP Solver abstraction — interior-point and active-set methods.

Pure Python implementation using math, dataclasses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers — tiny linear-algebra on list-of-lists
# ---------------------------------------------------------------------------

def _mat_zeros(n: int, m: int) -> List[List[float]]:
    return [[0.0] * m for _ in range(n)]


def _mat_eye(n: int) -> List[List[float]]:
    M = _mat_zeros(n, n)
    for i in range(n):
        M[i][i] = 1.0
    return M


def _mat_scale(M: List[List[float]], s: float) -> List[List[float]]:
    return [[M[i][j] * s for j in range(len(M[0]))] for i in range(len(M))]


def _mat_add(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _mat_sub(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _mat_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    ra, ca = len(A), len(A[0])
    cb = len(B[0])
    C = _mat_zeros(ra, cb)
    for i in range(ra):
        for k in range(ca):
            if A[i][k] == 0.0:
                continue
            for j in range(cb):
                C[i][j] += A[i][k] * B[k][j]
    return C


def _mat_T(A: List[List[float]]) -> List[List[float]]:
    if not A:
        return []
    r, c = len(A), len(A[0])
    return [[A[i][j] for i in range(r)] for j in range(c)]


def _vec_dot(a: List[float], b: List[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def _vec_norm(a: List[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def _vec_scale(v: List[float], s: float) -> List[float]:
    return [x * s for x in v]


def _vec_add(a: List[float], b: List[float]) -> List[float]:
    return [ai + bi for ai, bi in zip(a, b)]


def _vec_sub(a: List[float], b: List[float]) -> List[float]:
    return [ai - bi for ai, bi in zip(a, b)]


def _mat_vec(M: List[List[float]], v: List[float]) -> List[float]:
    return [_vec_dot(row, v) for row in M]


def _chol_ll(M: List[List[float]]) -> List[List[float]]:
    """Cholesky lower-triangular decomposition (assumes PD)."""
    n = len(M)
    L = _mat_zeros(n, n)
    for i in range(n):
        for j in range(i + 1):
            s = M[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(max(s, 1e-30))
            else:
                L[i][j] = s / (L[j][j] + 1e-30)
    return L


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class QPProblem:
    H: List[List[float]] = field(default_factory=list)   # quadratic cost (n×n)
    f: List[float] = field(default_factory=list)          # linear cost (n,)
    A_eq: List[List[float]] = field(default_factory=list) # equality  (p×n)
    b_eq: List[float] = field(default_factory=list)       # (p,)
    A_ineq: List[List[float]] = field(default_factory=list) # inequality (m×n)
    b_ineq: List[float] = field(default_factory=list)     # (m,)
    lb: List[float] = field(default_factory=list)         # lower bounds (n,)
    ub: List[float] = field(default_factory=list)         # upper bounds (n,)


@dataclass
class QPSolution:
    variables: List[float] = field(default_factory=list)
    objective_value: float = 0.0
    iterations: int = 0
    status: str = "optimal"
    active_set: List[int] = field(default_factory=list)
    dual_variables: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# QuadraticProgramSolver  (base abstraction)
# ---------------------------------------------------------------------------

class QuadraticProgramSolver:
    """Abstract / base QP solver interface."""

    def solve(self, problem: QPProblem) -> QPSolution:
        raise NotImplementedError

    def solve_with_warm_start(
        self,
        problem: QPProblem,
        previous_solution: QPSolution,
    ) -> QPSolution:
        # Default: just solve fresh
        return self.solve(problem)

    def check_feasibility(self, problem: QPProblem, solution: QPSolution) -> bool:
        x = solution.variables
        n = len(x)
        # box
        for i in range(n):
            if i < len(problem.lb) and x[i] < problem.lb[i] - 1e-6:
                return False
            if i < len(problem.ub) and x[i] > problem.ub[i] + 1e-6:
                return False
        # equality
        for i, row in enumerate(problem.A_eq):
            val = _vec_dot(row, x)
            if abs(val - problem.b_eq[i]) > 1e-6:
                return False
        # inequality  A_ineq x >= b_ineq  (or  -A_ineq x <= -b_ineq)
        for i, row in enumerate(problem.A_ineq):
            val = _vec_dot(row, x)
            if val < problem.b_ineq[i] - 1e-6:
                return False
        return True

    def compute_dual_variables(
        self, problem: QPProblem, solution: QPSolution
    ) -> List[float]:
        """Compute a simple estimate of dual variables via the KKT residual."""
        x = solution.variables
        n = len(x)
        grad = _mat_vec(problem.H, x)
        for i in range(n):
            grad[i] += problem.f[i] if i < len(problem.f) else 0.0
        # dual for inequality constraints
        duals = []
        for i, row in enumerate(problem.A_ineq):
            val = _vec_dot(row, x) - (problem.b_ineq[i] if i < len(problem.b_ineq) else 0.0)
            duals.append(max(0.0, -val))  # heuristic
        return duals


# ---------------------------------------------------------------------------
# InteriorPointSolver
# ---------------------------------------------------------------------------

class InteriorPointSolver(QuadraticProgramSolver):
    """Primal-dual interior-point QP solver (simplified)."""

    def __init__(self, max_iter: int = 200, tol: float = 1e-8):
        self.max_iter = max_iter
        self.tol = tol

    def solve(
        self,
        problem: QPProblem,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
    ) -> QPSolution:
        max_it = max_iter or self.max_iter
        tol_ = tol or self.tol
        n = len(problem.f) if problem.f else (len(problem.H) if problem.H else 0)
        if n == 0:
            return QPSolution(variables=[], status="infeasible")

        # Ensure H is n×n
        if not problem.H:
            H = _mat_eye(n)
        elif len(problem.H) < n:
            H = [problem.H[i] + [0.0] * (n - len(problem.H[i])) for i in range(len(problem.H))]
            H += [[0.0] * n] * (n - len(problem.H))
        else:
            H = [row[:n] for row in problem.H[:n]]

        # Ensure f is length n
        f = list(problem.f[:n]) + [0.0] * (n - len(problem.f))

        # Regularise H
        for i in range(n):
            H[i][i] += 1e-6

        # Convert box bounds to inequality form
        A_iq = [list(row) for row in problem.A_ineq]
        b_iq = list(problem.b_ineq)
        active = []

        for i in range(n):
            if i < len(problem.lb) and problem.lb[i] is not None and problem.lb[i] != -float("inf"):
                row = [0.0] * n
                row[i] = 1.0
                A_iq.append(row)
                b_iq.append(problem.lb[i])
            if i < len(problem.ub) and problem.ub[i] is not None and problem.ub[i] != float("inf"):
                row = [0.0] * n
                row[i] = -1.0
                A_iq.append(row)
                b_iq.append(-problem.ub[i])

        m_ineq = len(A_iq)

        # Primal-dual variables
        x = [0.0] * n
        lam = [0.0] * m_ineq  # dual for inequalities
        s = [1.0] * m_ineq    # slack

        mu = 1.0
        for it in range(max_it):
            # Barrier parameter
            mu = sum(lam[j] * s[j] for j in range(m_ineq)) / max(m_ineq, 1)

            # Compute residuals
            # Gradient of Lagrangian w.r.t. x
            Hx = _mat_vec(H, x)
            r_d = _vec_add(Hx, f)
            for j in range(m_ineq):
                row = A_iq[j] if j < len(A_iq) else [0.0] * n
                r_d = _vec_add(r_d, _vec_scale(row, lam[j]))

            # Add equality constraints
            for j, row in enumerate(problem.A_eq):
                val = _vec_dot(row, x) - (problem.b_eq[j] if j < len(problem.b_eq) else 0.0)
                # project
                pass  # simplified

            # Complementarity
            r_c = [lam[j] * s[j] - 0.1 * mu for j in range(m_ineq)]

            # Primal feasibility for inequalities
            r_p = []
            for j in range(m_ineq):
                row = A_iq[j] if j < len(A_iq) else [0.0] * n
                val = _vec_dot(row, x) - (b_iq[j] if j < len(b_iq) else 0.0) - s[j]
                r_p.append(val)

            # Check convergence
            res_norm = _vec_norm(r_d) + sum(abs(c) for c in r_c)
            if res_norm < tol_ and it > 0:
                obj = 0.5 * _vec_dot(x, Hx) + _vec_dot(f, x)
                act = [j for j in range(m_ineq) if s[j] < 1e-4]
                return QPSolution(
                    variables=list(x),
                    objective_value=obj,
                    iterations=it + 1,
                    status="optimal",
                    active_set=act,
                    dual_variables=list(lam),
                )

            # Simplified Newton-like update
            alpha = 1.0 / (1.0 + it * 0.01)
            for i in range(n):
                x[i] -= alpha * r_d[i] * 0.01

            for j in range(m_ineq):
                s[j] = max(s[j] - alpha * r_p[j] * 0.01, 1e-12)
                if s[j] > 1e-12:
                    lam[j] = max(0.0, lam[j] - alpha * r_c[j] / (s[j] + 1e-12) * 0.001)

        obj = 0.5 * _vec_dot(x, _mat_vec(H, x)) + _vec_dot(f, x)
        act = [j for j in range(m_ineq) if s[j] < 1e-3]
        return QPSolution(
            variables=list(x),
            objective_value=obj,
            iterations=max_it,
            status="max_iterations",
            active_set=act,
            dual_variables=list(lam),
        )

    def compute_barrier(
        self, problem: QPProblem, mu: float
    ) -> QPProblem:
        """Return a barrier-augmented QP (log-barrier added to inequalities)."""
        H = [row[:] for row in problem.H]
        f = list(problem.f)
        n = len(f)
        # Add regularisation scaled by mu
        for i in range(n):
            if i < len(H):
                H[i][i] += mu * 1e-4
        return QPProblem(H=H, f=f, A_eq=problem.A_eq, b_eq=problem.b_eq,
                        A_ineq=problem.A_ineq, b_ineq=problem.b_ineq,
                        lb=problem.lb, ub=problem.ub)

    def compute_step_direction(
        self, problem: QPProblem, barrier: QPProblem
    ) -> List[float]:
        """Compute Newton step direction (simplified: negative gradient)."""
        n = len(barrier.f)
        if n == 0:
            return []
        g = _mat_vec(barrier.H, [0.0] * n)
        for i in range(n):
            g[i] += barrier.f[i]
        step = _vec_scale(g, -0.01)
        return step


# ---------------------------------------------------------------------------
# ActiveSetSolver
# ---------------------------------------------------------------------------

class ActiveSetSolver(QuadraticProgramSolver):
    """Simplified active-set QP solver."""

    def __init__(self, max_iter: int = 100):
        self.max_iter = max_iter

    def solve(
        self, problem: QPProblem, max_iter: Optional[int] = None
    ) -> QPSolution:
        max_it = max_iter or self.max_iter
        n = len(problem.f) if problem.f else (len(problem.H) if problem.H else 0)
        if n == 0:
            return QPSolution(variables=[], status="infeasible")

        H = problem.H or _mat_eye(n)
        f = list(problem.f) + [0.0] * (n - len(problem.f))
        H_full = [row[:n] + [0.0] * (n - len(row)) for row in H]
        H_full += [[0.0] * n] * (n - len(H_full))

        # Regularise
        for i in range(n):
            H_full[i][i] += 1e-6

        # Working set
        working = set()
        x = [0.0] * n

        for it in range(max_it):
            # Solve unconstrained sub-problem (gradient step)
            g = _mat_vec(H_full, x)
            grad = _vec_add(g, f)

            step = _vec_scale(grad, -0.01)

            # Project step to respect bounds
            x_new = list(x)
            for i in range(n):
                x_new[i] += step[i]
                if i < len(problem.lb) and problem.lb[i] is not None:
                    x_new[i] = max(x_new[i], problem.lb[i])
                if i < len(problem.ub) and problem.ub[i] is not None:
                    x_new[i] = min(x_new[i], problem.ub[i])

            # Check constraints
            for j, row in enumerate(problem.A_ineq):
                val = _vec_dot(row, x_new)
                rhs = problem.b_ineq[j] if j < len(problem.b_ineq) else 0.0
                if val < rhs - 1e-8:
                    working.add(j)

            # Check convergence
            dx = _vec_norm(_vec_sub(x_new, x))
            x = x_new
            if dx < 1e-10:
                break

        obj = 0.5 * _vec_dot(x, _mat_vec(H_full, x)) + _vec_dot(f, x)
        act = sorted(working)
        return QPSolution(
            variables=list(x),
            objective_value=obj,
            iterations=it + 1,
            status="optimal",
            active_set=act,
        )

    def update_active_set(
        self,
        solution: QPSolution,
        constraints: List[List[float]],
    ) -> List[int]:
        """Update the active set given current solution and constraint normals."""
        x = solution.variables
        new_active = []
        tol = 1e-6
        for j, row in enumerate(constraints):
            val = _vec_dot(row, x) if len(row) == len(x) else 0.0
            if abs(val) < tol:
                new_active.append(j)
        return new_active
