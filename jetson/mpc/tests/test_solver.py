"""Tests for jetson.mpc.solver — 33 tests."""
import math
import pytest
from jetson.mpc.solver import (
    QPProblem,
    QPSolution,
    QuadraticProgramSolver,
    InteriorPointSolver,
    ActiveSetSolver,
    _mat_zeros,
    _mat_eye,
    _mat_scale,
    _mat_add,
    _mat_sub,
    _mat_mul,
    _mat_T,
    _vec_dot,
    _vec_norm,
    _vec_scale,
    _vec_add,
    _vec_sub,
    _mat_vec,
    _chol_ll,
)


# ---- matrix helpers ----

class TestSolverMatOps:
    def test_mat_zeros(self):
        M = _mat_zeros(2, 3)
        assert M == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    def test_mat_eye(self):
        E = _mat_eye(3)
        assert E[0][0] == 1.0 and E[1][1] == 1.0 and E[2][2] == 1.0
        assert E[0][1] == 0.0

    def test_mat_scale(self):
        M = _mat_scale([[1, 2], [3, 4]], -1.0)
        assert M == [[-1, -2], [-3, -4]]

    def test_mat_add(self):
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        C = _mat_add(A, B)
        assert C == [[6, 8], [10, 12]]

    def test_mat_sub(self):
        C = _mat_sub([[5, 6], [7, 8]], [[1, 2], [3, 4]])
        assert C == [[4, 4], [4, 4]]

    def test_mat_mul_identity(self):
        I = _mat_eye(2)
        A = [[1, 2], [3, 4]]
        C = _mat_mul(I, A)
        assert C == A

    def test_mat_mul_2x2(self):
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        C = _mat_mul(A, B)
        assert C == [[19, 22], [43, 50]]

    def test_mat_transpose(self):
        A = [[1, 2, 3], [4, 5, 6]]
        T = _mat_T(A)
        assert T == [[1, 4], [2, 5], [3, 6]]

    def test_mat_vec(self):
        M = [[1, 0], [0, 1]]
        assert _mat_vec(M, [7, 8]) == [7, 8]

    def test_chol_ll_2x2(self):
        M = [[4, 2], [2, 3]]
        L = _chol_ll(M)
        assert abs(L[0][0] - 2.0) < 1e-9
        assert abs(L[1][1] - math.sqrt(2)) < 1e-9

    def test_vec_norm(self):
        assert abs(_vec_norm([3, 4]) - 5.0) < 1e-9

    def test_vec_scale(self):
        assert _vec_scale([1, 2], 3.0) == [3.0, 6.0]

    def test_vec_add(self):
        assert _vec_add([1, 2], [3, 4]) == [4, 6]


# ---- QPProblem / QPSolution ----

class TestQPDataClasses:
    def test_qp_problem_defaults(self):
        p = QPProblem()
        assert p.H == []
        assert p.f == []

    def test_qp_problem_fields(self):
        p = QPProblem(
            H=[[1]], f=[2],
            A_eq=[[1]], b_eq=[0],
            A_ineq=[[-1]], b_ineq=[-10],
            lb=[0], ub=[10],
        )
        assert p.lb == [0]
        assert p.ub == [10]

    def test_qp_solution_defaults(self):
        s = QPSolution()
        assert s.variables == []
        assert s.status == "optimal"
        assert s.iterations == 0

    def test_qp_solution_with_data(self):
        s = QPSolution(variables=[1.0, 2.0], objective_value=5.0,
                       iterations=10, status="max_iterations",
                       active_set=[0, 2], dual_variables=[0.1, 0.0, 0.3])
        assert len(s.variables) == 2
        assert s.objective_value == 5.0
        assert s.active_set == [0, 2]


# ---- QuadraticProgramSolver ----

class TestQuadraticProgramSolver:
    def test_solve_raises(self):
        s = QuadraticProgramSolver()
        with pytest.raises(NotImplementedError):
            s.solve(QPProblem())

    def test_solve_with_warm_start(self):
        s = QuadraticProgramSolver()
        with pytest.raises(NotImplementedError):
            s.solve_with_warm_start(QPProblem(), QPSolution())

    def test_check_feasibility_empty(self):
        s = QuadraticProgramSolver()
        assert s.check_feasibility(QPProblem(), QPSolution(variables=[]))

    def test_check_feasibility_box_ok(self):
        s = QuadraticProgramSolver()
        prob = QPProblem(lb=[0, 0], ub=[10, 10])
        sol = QPSolution(variables=[5, 5])
        assert s.check_feasibility(prob, sol)

    def test_check_feasibility_box_violated(self):
        s = QuadraticProgramSolver()
        prob = QPProblem(lb=[0, 0], ub=[10, 10])
        sol = QPSolution(variables=[15, 5])
        assert not s.check_feasibility(prob, sol)

    def test_check_feasibility_equality(self):
        s = QuadraticProgramSolver()
        prob = QPProblem(A_eq=[[1, 1]], b_eq=[10])
        sol = QPSolution(variables=[3, 7])
        assert s.check_feasibility(prob, sol)

    def test_check_feasibility_equality_violated(self):
        s = QuadraticProgramSolver()
        prob = QPProblem(A_eq=[[1, 1]], b_eq=[10])
        sol = QPSolution(variables=[3, 8])
        assert not s.check_feasibility(prob, sol)

    def test_check_feasibility_inequality(self):
        s = QuadraticProgramSolver()
        prob = QPProblem(A_ineq=[[1, 0]], b_ineq=[2])  # x1 >= 2
        sol = QPSolution(variables=[3, 0])
        assert s.check_feasibility(prob, sol)

    def test_check_feasibility_inequality_violated(self):
        s = QuadraticProgramSolver()
        prob = QPProblem(A_ineq=[[1, 0]], b_ineq=[2])
        sol = QPSolution(variables=[1, 0])
        assert not s.check_feasibility(prob, sol)

    def test_compute_dual_variables(self):
        s = QuadraticProgramSolver()
        prob = QPProblem(H=[[2, 0], [0, 2]], f=[0, 0],
                         A_ineq=[[1, 0], [0, 1]], b_ineq=[0, 0])
        sol = QPSolution(variables=[1.0, 1.0])
        duals = s.compute_dual_variables(prob, sol)
        assert len(duals) == 2


# ---- InteriorPointSolver ----

class TestInteriorPointSolver:
    def test_construct(self):
        s = InteriorPointSolver(max_iter=50, tol=1e-6)
        assert s.max_iter == 50
        assert s.tol == 1e-6

    def test_solve_simple(self):
        s = InteriorPointSolver(max_iter=50)
        prob = QPProblem(
            H=[[2.0, 0.0], [0.0, 2.0]],
            f=[-4.0, -6.0],
            lb=[0, 0], ub=[10, 10],
        )
        sol = s.solve(prob)
        assert sol.status in ("optimal", "max_iterations")
        assert len(sol.variables) == 2

    def test_solve_with_bounds(self):
        s = InteriorPointSolver(max_iter=30)
        prob = QPProblem(
            H=[[1.0]], f=[-1.0],
            lb=[0], ub=[5],
        )
        sol = s.solve(prob)
        assert len(sol.variables) == 1
        assert sol.variables[0] >= -1e-3  # near 0..5 range

    def test_solve_empty(self):
        s = InteriorPointSolver()
        sol = s.solve(QPProblem())
        assert sol.status == "infeasible"
        assert sol.variables == []

    def test_solve_with_inequality(self):
        s = InteriorPointSolver(max_iter=20)
        prob = QPProblem(
            H=[[2.0, 0.0], [0.0, 2.0]],
            f=[0, 0],
            A_ineq=[[1.0, 1.0]],
            b_ineq=[1.0],
        )
        sol = s.solve(prob)
        assert len(sol.variables) == 2

    def test_compute_barrier(self):
        s = InteriorPointSolver()
        prob = QPProblem(H=[[2, 0], [0, 2]], f=[1, 1])
        bp = s.compute_barrier(prob, mu=0.1)
        assert bp.H[0][0] == pytest.approx(2.00001, abs=1e-6)

    def test_compute_step_direction(self):
        s = InteriorPointSolver()
        prob = QPProblem(H=[[2, 0], [0, 2]], f=[1, 1])
        bp = s.compute_barrier(prob, mu=0.1)
        step = s.compute_step_direction(prob, bp)
        assert len(step) == 2
        assert step[0] != 0.0


# ---- ActiveSetSolver ----

class TestActiveSetSolver:
    def test_construct(self):
        s = ActiveSetSolver(max_iter=50)
        assert s.max_iter == 50

    def test_solve_simple(self):
        s = ActiveSetSolver(max_iter=50)
        prob = QPProblem(
            H=[[2.0, 0.0], [0.0, 2.0]],
            f=[-4.0, -6.0],
            lb=[0, 0], ub=[10, 10],
        )
        sol = s.solve(prob)
        assert sol.status == "optimal"
        assert len(sol.variables) == 2

    def test_solve_empty(self):
        s = ActiveSetSolver()
        sol = s.solve(QPProblem())
        assert sol.status == "infeasible"

    def test_solve_with_bounds(self):
        s = ActiveSetSolver(max_iter=30)
        prob = QPProblem(
            H=[[1.0]], f=[-2.0],
            lb=[0], ub=[5],
        )
        sol = s.solve(prob)
        assert len(sol.variables) == 1

    def test_solve_upper_bound(self):
        s = ActiveSetSolver(max_iter=30)
        prob = QPProblem(
            H=[[1.0]], f=[2.0],
            lb=[0], ub=[5],
        )
        sol = s.solve(prob)
        # Minimise x^2 + 2x, so minimum at x=0
        assert sol.variables[0] <= 0.1

    def test_update_active_set(self):
        s = ActiveSetSolver()
        sol = QPSolution(variables=[0.0, 0.0])
        constraints = [[1, 0], [0, 1]]
        active = s.update_active_set(sol, constraints)
        assert 0 in active
        assert 1 in active

    def test_update_active_set_no_binding(self):
        s = ActiveSetSolver()
        sol = QPSolution(variables=[5.0, 5.0])
        constraints = [[1, 0], [0, 1]]
        active = s.update_active_set(sol, constraints)
        # vals are 5 and 5, not near zero
        assert len(active) == 0

    def test_solve_returns_objective(self):
        s = ActiveSetSolver(max_iter=30)
        prob = QPProblem(H=[[2.0]], f=[0.0], lb=[0], ub=[10])
        sol = s.solve(prob)
        assert sol.objective_value >= 0.0
