"""Tests for jetson.mpc.formulation — 32 tests."""
import math
import pytest
from jetson.mpc.formulation import (
    MPCConfig,
    CostFunction,
    MPCProblem,
    MPCFormulator,
    ConstraintType,
    ConstraintSpec,
    _mat_zero,
    _mat_eye,
    _mat_scale,
    _vec_dot,
    _vec_sub,
    _vec_norm,
)


# ---- helpers ----

class TestHelperMatOps:
    def test_mat_zero_3x3(self):
        M = _mat_zero(3, 3)
        assert M == [[0.0] * 3 for _ in range(3)]

    def test_mat_eye_4(self):
        E = _mat_eye(4)
        for i in range(4):
            for j in range(4):
                assert E[i][j] == (1.0 if i == j else 0.0)

    def test_mat_scale(self):
        M = [[1, 2], [3, 4]]
        S = _mat_scale(M, 2.0)
        assert S == [[2, 4], [6, 8]]

    def test_vec_dot(self):
        assert _vec_dot([1, 2, 3], [4, 5, 6]) == 32

    def test_vec_sub(self):
        assert _vec_sub([5, 6], [1, 2]) == [4, 4]

    def test_vec_norm(self):
        assert abs(_vec_norm([3, 4]) - 5.0) < 1e-9


# ---- MPCConfig ----

class TestMPCConfig:
    def test_defaults(self):
        cfg = MPCConfig()
        assert cfg.horizon == 10
        assert cfg.dt == pytest.approx(0.1)
        assert cfg.state_dim == 4
        assert cfg.control_dim == 2

    def test_custom(self):
        cfg = MPCConfig(horizon=20, dt=0.05, state_dim=6, control_dim=3)
        assert cfg.horizon == 20
        assert cfg.dt == pytest.approx(0.05)

    def test_default_weights_match_state_dim(self):
        cfg = MPCConfig(state_dim=5)
        assert len(cfg.weights) == 5
        assert all(w == 1.0 for w in cfg.weights)

    def test_custom_weights_preserved(self):
        cfg = MPCConfig(weights=[2.0, 3.0])
        assert cfg.weights == [2.0, 3.0]


# ---- CostFunction ----

class TestCostFunction:
    def test_default_matrices(self):
        cf = CostFunction()
        assert cf.state_weight_matrix == [[1.0]]
        assert cf.control_weight_matrix == [[1.0]]
        assert cf.terminal_weight == [[1.0]]

    def test_custom_matrices(self):
        cf = CostFunction(
            state_weight_matrix=[[2.0, 0.0], [0.0, 2.0]],
            control_weight_matrix=[[1.0]],
            terminal_weight=[[5.0]],
        )
        assert cf.state_weight_matrix[0][0] == 2.0
        assert cf.terminal_weight[0][0] == 5.0


# ---- MPCProblem ----

class TestMPCProblem:
    def test_default(self):
        p = MPCProblem()
        assert p.current_state == []
        assert p.horizon == 10

    def test_with_data(self):
        p = MPCProblem(
            current_state=[1.0, 2.0],
            horizon=5,
            state_dim=2,
            control_dim=1,
        )
        assert p.current_state == [1.0, 2.0]
        assert p.state_dim == 2


# ---- ConstraintType / ConstraintSpec ----

class TestConstraintSpec:
    def test_equality_type(self):
        cs = ConstraintSpec(ctype=ConstraintType.EQUALITY)
        assert cs.ctype == ConstraintType.EQUALITY

    def test_inequality_type(self):
        cs = ConstraintSpec(ctype=ConstraintType.INEQUALITY, lower=-10, upper=10)
        assert cs.lower == -10
        assert cs.upper == 10


# ---- MPCFormulator ----

class TestMPCFormulator:
    def test_init_default(self):
        f = MPCFormulator()
        assert f.config.horizon == 10

    def test_init_custom_config(self):
        cfg = MPCConfig(horizon=5, state_dim=3, control_dim=1)
        f = MPCFormulator(config=cfg)
        assert f.config.horizon == 5

    def test_formulate_tracking(self):
        f = MPCFormulator(MPCConfig(horizon=5, state_dim=2, control_dim=1))
        ref = [[i, i + 1] for i in range(6)]
        prob = f.formulate_tracking(ref)
        assert prob.horizon == 5
        assert prob.state_dim == 2
        assert len(prob.reference_trajectory) == 6
        assert prob.cost.state_weight_matrix is not None

    def test_formulate_tracking_custom_weights(self):
        f = MPCFormulator(MPCConfig(state_dim=2, control_dim=1))
        ref = [[0.0, 0.0], [1.0, 1.0]]
        prob = f.formulate_tracking(ref, weights=[10.0, 1.0])
        assert prob.cost.state_weight_matrix[0][0] == 10.0
        assert prob.cost.state_weight_matrix[1][1] == 1.0

    def test_formulate_tracking_empty_ref(self):
        f = MPCFormulator(MPCConfig(state_dim=3))
        prob = f.formulate_tracking([])
        assert prob.current_state == [0.0, 0.0, 0.0]

    def test_formulate_regulation(self):
        f = MPCFormulator(MPCConfig(horizon=8, state_dim=3, control_dim=2))
        sp = [1.0, 2.0, 3.0]
        prob = f.formulate_regulation(sp)
        assert prob.horizon == 8
        assert len(prob.reference_trajectory) == 9
        assert prob.cost.terminal_weight[0][0] == 10.0

    def test_formulate_regulation_custom_weights(self):
        f = MPCFormulator(MPCConfig(state_dim=2))
        prob = f.formulate_regulation([5.0, 5.0], weights=[3.0, 1.0])
        assert prob.cost.state_weight_matrix[0][0] == 3.0

    def test_formulate_economic(self):
        f = MPCFormulator(MPCConfig(state_dim=2, control_dim=1, horizon=5))
        cs = ConstraintSpec(ctype=ConstraintType.INEQUALITY, lower=0.0)
        prob = f.formulate_economic(lambda s, u: s[0], [cs])
        assert prob.horizon == 5
        assert len(prob.constraints) == 1

    def test_formulate_economic_objective_fn_called(self):
        f = MPCFormulator(MPCConfig())
        calls = []
        def obj(s, u):
            calls.append((s, u))
            return 0.0
        f.formulate_economic(obj, [])
        assert len(calls) == 0  # formulation stores fn reference, not called yet

    def test_compute_stage_cost_zero(self):
        f = MPCFormulator()
        cost = f.compute_stage_cost([1, 1], [0], [1, 1], [1.0, 1.0])
        assert cost == pytest.approx(0.0)

    def test_compute_stage_cost_nonzero(self):
        f = MPCFormulator()
        cost = f.compute_stage_cost([2, 1], [1], [1, 1], [1.0, 1.0])
        assert cost == pytest.approx(2.0)  # 1^2 + 1^2 (state) + 1 (control)

    def test_compute_stage_cost_control(self):
        f = MPCFormulator()
        cost = f.compute_stage_cost([0, 0], [3], [0, 0], [1, 1])
        assert cost == pytest.approx(9.0)

    def test_compute_stage_cost_weights(self):
        f = MPCFormulator()
        cost = f.compute_stage_cost([2, 0], [0], [0, 0], [5.0, 1.0])
        assert cost == pytest.approx(20.0)

    def test_compute_terminal_cost_identity(self):
        f = MPCFormulator()
        P = [[1.0, 0.0], [0.0, 1.0]]
        cost = f.compute_terminal_cost([3, 4], P)
        assert cost == pytest.approx(25.0)

    def test_compute_terminal_cost_scaled(self):
        f = MPCFormulator()
        P = [[2.0, 0.0], [0.0, 2.0]]
        cost = f.compute_terminal_cost([1, 1], P)
        assert cost == pytest.approx(4.0)

    def test_compute_terminal_cost_zero(self):
        f = MPCFormulator()
        P = [[5.0, 0.0], [0.0, 5.0]]
        cost = f.compute_terminal_cost([0, 0], P)
        assert cost == pytest.approx(0.0)

    def test_build_cost_matrices_shape(self):
        f = MPCFormulator()
        Q, R, P = f.build_cost_matrices(10, 4, 2)
        assert len(Q) == (10 + 1) * 4
        assert len(R) == 10 * 2
        assert len(P) == 4

    def test_build_cost_matrices_identity(self):
        f = MPCFormulator()
        Q, R, P = f.build_cost_matrices(3, 2, 1)
        assert Q[0][0] == pytest.approx(1.0)
        assert P[0][0] == pytest.approx(10.0)
        assert R[0][0] == pytest.approx(0.1)
