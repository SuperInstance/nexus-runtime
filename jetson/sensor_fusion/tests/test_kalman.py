"""Tests for kalman.py — LinearKalmanFilter, ExtendedKalmanFilter, KalmanState."""

import math
import pytest

from jetson.sensor_fusion.kalman import (
    KalmanState,
    LinearKalmanFilter,
    ExtendedKalmanFilter,
    _mat_zeros, _mat_identity, _mat_mul, _mat_transpose,
    _mat_add, _mat_sub, _mat_scale, _mat_vec_mul,
    _vec_add, _vec_sub, _mat_inverse,
)


# ===================================================================
# Matrix utility tests
# ===================================================================

class TestMatZeros:
    def test_2x3(self):
        m = _mat_zeros(2, 3)
        assert m == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    def test_1x1(self):
        assert _mat_zeros(1, 1) == [[0.0]]

    def test_0x0(self):
        assert _mat_zeros(0, 0) == []


class TestMatIdentity:
    def test_3x3(self):
        I = _mat_identity(3)
        for i in range(3):
            for j in range(3):
                assert I[i][j] == (1.0 if i == j else 0.0)

    def test_1x1(self):
        assert _mat_identity(1) == [[1.0]]


class TestMatMul:
    def test_identity(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        I = _mat_identity(2)
        C = _mat_mul(A, I)
        assert C == A

    def test_2x2(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[5.0, 6.0], [7.0, 8.0]]
        C = _mat_mul(A, B)
        assert C == [[19.0, 22.0], [43.0, 50.0]]

    def test_rectangular(self):
        A = [[1.0, 2.0, 3.0]]
        B = [[4.0], [5.0], [6.0]]
        C = _mat_mul(A, B)
        assert C == [[32.0]]


class TestMatTranspose:
    def test_square(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        T = _mat_transpose(A)
        assert T == [[1.0, 3.0], [2.0, 4.0]]

    def test_rectangular(self):
        A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        T = _mat_transpose(A)
        assert T == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]


class TestMatAddSub:
    def test_add(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[5.0, 6.0], [7.0, 8.0]]
        C = _mat_add(A, B)
        assert C == [[6.0, 8.0], [10.0, 12.0]]

    def test_sub(self):
        A = [[5.0, 6.0], [7.0, 8.0]]
        B = [[1.0, 2.0], [3.0, 4.0]]
        C = _mat_sub(A, B)
        assert C == [[4.0, 4.0], [4.0, 4.0]]


class TestMatScale:
    def test_scale(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        C = _mat_scale(A, 2.0)
        assert C == [[2.0, 4.0], [6.0, 8.0]]

    def test_scale_zero(self):
        A = [[1.0, 2.0]]
        C = _mat_scale(A, 0.0)
        assert C == [[0.0, 0.0]]


class TestMatVecMul:
    def test_identity(self):
        A = _mat_identity(3)
        v = [1.0, 2.0, 3.0]
        assert _mat_vec_mul(A, v) == [1.0, 2.0, 3.0]

    def test_simple(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        v = [5.0, 6.0]
        assert _mat_vec_mul(A, v) == [17.0, 39.0]


class TestVecAddSub:
    def test_add(self):
        assert _vec_add([1.0, 2.0], [3.0, 4.0]) == [4.0, 6.0]

    def test_sub(self):
        assert _vec_sub([5.0, 6.0], [3.0, 4.0]) == [2.0, 2.0]


class TestMatInverse:
    def test_1x1(self):
        assert _mat_inverse([[4.0]]) == [[0.25]]

    def test_2x2(self):
        M = [[4.0, 7.0], [2.0, 6.0]]
        inv = _mat_inverse(M)
        assert inv is not None
        # M * inv ≈ I
        prod = _mat_mul(M, inv)
        for i in range(2):
            for j in range(2):
                assert abs(prod[i][j] - (1.0 if i == j else 0.0)) < 1e-10

    def test_3x3(self):
        M = [[2.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 2.0]]
        inv = _mat_inverse(M)
        assert inv is not None
        prod = _mat_mul(M, inv)
        for i in range(3):
            for j in range(3):
                assert abs(prod[i][j] - (1.0 if i == j else 0.0)) < 1e-10

    def test_singular_returns_none(self):
        M = [[1.0, 2.0], [2.0, 4.0]]  # singular
        assert _mat_inverse(M) is None


# ===================================================================
# KalmanState tests
# ===================================================================

class TestKalmanState:
    def test_creation(self):
        s = KalmanState(mean=[1.0, 2.0], covariance=[[3.0, 0.0], [0.0, 4.0]], timestamp=1.5)
        assert s.mean == [1.0, 2.0]
        assert s.timestamp == 1.5

    def test_default_timestamp(self):
        s = KalmanState(mean=[0.0], covariance=[[1.0]])
        assert s.timestamp == 0.0

    def test_copy(self):
        s = KalmanState(mean=[1.0], covariance=[[5.0]])
        s2 = s.copy()
        s2.mean[0] = 99.0
        assert s.mean[0] == 1.0  # original unchanged
        s2.covariance[0][0] = 100.0
        assert s.covariance[0][0] == 5.0


# ===================================================================
# LinearKalmanFilter tests
# ===================================================================

class TestLinearKalmanFilter:
    def test_creation_default(self):
        kf = LinearKalmanFilter(state_dim=2, measurement_dim=2)
        state = kf.get_state()
        assert state.mean == [0.0, 0.0]
        assert state.timestamp == 0.0

    def test_creation_with_initial(self):
        kf = LinearKalmanFilter(
            state_dim=2, measurement_dim=2,
            initial_state=[5.0, 3.0],
            initial_covariance=[[10.0, 0.0], [0.0, 10.0]],
        )
        state = kf.get_state()
        assert state.mean == [5.0, 3.0]

    def test_predict_identity(self):
        kf = LinearKalmanFilter(state_dim=2, measurement_dim=2, initial_state=[1.0, 2.0])
        kf.predict(dt=1.0)
        # With identity F and no control, state stays same
        state = kf.get_state()
        assert abs(state.mean[0] - 1.0) < 1e-10
        assert abs(state.mean[1] - 2.0) < 1e-10
        assert state.timestamp == 1.0

    def test_predict_with_control(self):
        kf = LinearKalmanFilter(state_dim=2, measurement_dim=2, initial_state=[0.0, 0.0])
        kf.predict(dt=1.0, control_input=[3.0, 4.0])
        state = kf.get_state()
        assert state.mean == [3.0, 4.0]

    def test_predict_increases_covariance(self):
        kf = LinearKalmanFilter(
            state_dim=1, measurement_dim=1,
            initial_state=[0.0], initial_covariance=[[1.0]],
            process_noise=[[0.5]],
        )
        state_before = kf.get_state().covariance[0][0]
        kf.predict(dt=1.0)
        state_after = kf.get_state().covariance[0][0]
        assert state_after > state_before

    def test_update_reduces_uncertainty(self):
        kf = LinearKalmanFilter(
            state_dim=1, measurement_dim=1,
            initial_state=[0.0], initial_covariance=[[10.0]],
            measurement_noise=[[1.0]],
        )
        cov_before = kf.get_state().covariance[0][0]
        kf.update(measurement=[0.0])
        cov_after = kf.get_state().covariance[0][0]
        assert cov_after < cov_before

    def test_update_moves_toward_measurement(self):
        kf = LinearKalmanFilter(
            state_dim=1, measurement_dim=1,
            initial_state=[0.0], initial_covariance=[[10.0]],
            measurement_noise=[[10.0]],
        )
        kf.update(measurement=[10.0])
        state = kf.get_state()
        assert 0.0 < state.mean[0] < 10.0

    def test_set_process_noise(self):
        kf = LinearKalmanFilter(state_dim=2, measurement_dim=2)
        kf.set_process_noise([[2.0, 0.0], [0.0, 2.0]])
        kf.predict(dt=1.0)
        state = kf.get_state()
        assert state.covariance[0][0] == pytest.approx(3.0, abs=1e-9)

    def test_set_measurement_noise(self):
        kf = LinearKalmanFilter(state_dim=2, measurement_dim=2)
        kf.set_measurement_noise([[5.0, 0.0], [0.0, 5.0]])
        kf.update(measurement=[1.0, 1.0])
        # Should not raise

    def test_compute_innovation(self):
        kf = LinearKalmanFilter(
            state_dim=1, measurement_dim=1,
            initial_state=[5.0],
        )
        innov = kf.compute_innovation([8.0])
        assert abs(innov[0] - 3.0) < 1e-10

    def test_custom_state_transition(self):
        F = [[1.0, 1.0], [0.0, 1.0]]  # constant velocity model
        kf = LinearKalmanFilter(
            state_dim=2, measurement_dim=2,
            state_transition=F,
            initial_state=[0.0, 1.0],
            process_noise=[[0.0, 0.0], [0.0, 0.0]],
        )
        kf.predict(dt=1.0)
        state = kf.get_state()
        assert state.mean[0] == pytest.approx(1.0, abs=1e-10)
        assert state.mean[1] == pytest.approx(1.0, abs=1e-10)

    def test_custom_measurement_matrix(self):
        H = [[1.0, 0.0]]  # only observe first state
        kf = LinearKalmanFilter(
            state_dim=2, measurement_dim=1,
            measurement_matrix=H,
            initial_state=[0.0, 5.0],
            initial_covariance=[[10.0, 0.0], [0.0, 10.0]],
            measurement_noise=[[1.0]],
        )
        kf.update(measurement=[3.0])
        state = kf.get_state()
        # First state should move toward 3.0
        assert state.mean[0] > 0.0
        assert state.mean[0] < 3.0
        # Second state unchanged
        assert state.mean[1] == pytest.approx(5.0, abs=1e-10)

    def test_multiple_predict_update_cycles(self):
        kf = LinearKalmanFilter(
            state_dim=1, measurement_dim=1,
            initial_state=[0.0], initial_covariance=[[100.0]],
            measurement_noise=[[1.0]], process_noise=[[0.1]],
        )
        true_value = 10.0
        for i in range(20):
            kf.predict(dt=0.1)
            kf.update(measurement=[true_value + 0.5 * ((-1) ** i)])  # noisy measurements
        state = kf.get_state()
        assert abs(state.mean[0] - true_value) < 2.0
        assert state.covariance[0][0] < 1.0

    def test_update_with_custom_noise(self):
        kf = LinearKalmanFilter(
            state_dim=1, measurement_dim=1,
            initial_state=[0.0], initial_covariance=[[10.0]],
            measurement_noise=[[100.0]],
        )
        kf.update(measurement=[10.0], measurement_noise=[[0.1]])
        state = kf.get_state()
        # With very low measurement noise, state should be very close to 10
        assert state.mean[0] > 9.0

    def test_get_state_returns_copy(self):
        kf = LinearKalmanFilter(
            state_dim=1, measurement_dim=1,
            initial_state=[5.0], initial_covariance=[[2.0]],
        )
        s1 = kf.get_state()
        s1.mean[0] = 999.0
        s2 = kf.get_state()
        assert s2.mean[0] == 5.0  # internal state unchanged


# ===================================================================
# ExtendedKalmanFilter tests
# ===================================================================

class TestExtendedKalmanFilter:
    def test_creation(self):
        ekf = ExtendedKalmanFilter(state_dim=2, measurement_dim=2)
        state = ekf.get_state()
        assert state.mean == [0.0, 0.0]

    def test_creation_with_initial(self):
        ekf = ExtendedKalmanFilter(
            state_dim=2, measurement_dim=2,
            initial_state=[1.0, 2.0],
            initial_covariance=[[5.0, 0.0], [0.0, 5.0]],
        )
        state = ekf.get_state()
        assert state.mean == [1.0, 2.0]

    def test_predict_identity(self):
        ekf = ExtendedKalmanFilter(state_dim=2, measurement_dim=2, initial_state=[3.0, 4.0])
        ekf.predict(dt=1.0)
        state = ekf.get_state()
        assert abs(state.mean[0] - 3.0) < 1e-6
        assert abs(state.mean[1] - 4.0) < 1e-6

    def test_predict_with_control(self):
        ekf = ExtendedKalmanFilter(state_dim=2, measurement_dim=2)
        ekf.predict(dt=1.0, control_input=[1.0, 2.0])
        state = ekf.get_state()
        assert state.mean == [1.0, 2.0]

    def test_predict_with_state_transition_fn(self):
        def constant_velocity(state, dt, control):
            # state = [pos, vel]
            return [state[0] + state[1] * dt, state[1]]

        ekf = ExtendedKalmanFilter(
            state_dim=2, measurement_dim=2,
            initial_state=[0.0, 2.0],
            process_noise=[[0.01, 0.0], [0.0, 0.01]],
        )
        ekf.predict(dt=1.0, state_transition_fn=constant_velocity)
        state = ekf.get_state()
        assert state.mean[0] == pytest.approx(2.0, abs=1e-4)
        assert state.mean[1] == pytest.approx(2.0, abs=1e-4)

    def test_predict_with_process_noise_fn(self):
        def custom_Q(state, dt):
            return [[dt * 0.5, 0.0], [0.0, dt * 0.1]]

        ekf = ExtendedKalmanFilter(state_dim=2, measurement_dim=2)
        ekf.predict(dt=2.0, process_noise_fn=custom_Q)
        state = ekf.get_state()
        assert state.covariance[0][0] == pytest.approx(1.0 + 1.0, abs=1e-4)

    def test_update_linear_measurement(self):
        ekf = ExtendedKalmanFilter(
            state_dim=1, measurement_dim=1,
            initial_state=[0.0], initial_covariance=[[10.0]],
            measurement_noise=[[1.0]],
        )
        ekf.update(measurement=[5.0], measurement_fn=lambda s: [s[0]])
        state = ekf.get_state()
        assert 0.0 < state.mean[0] < 5.0

    def test_update_nonlinear_measurement(self):
        # Measure range from origin: z = sqrt(x^2)
        ekf = ExtendedKalmanFilter(
            state_dim=1, measurement_dim=1,
            initial_state=[3.0], initial_covariance=[[1.0]],
            measurement_noise=[[0.1]],
        )
        ekf.update(measurement=[5.0], measurement_fn=lambda s: [math.sqrt(s[0] ** 2)])
        state = ekf.get_state()
        assert state.mean[0] > 3.0  # Should move toward 5

    def test_compute_jacobian_identity(self):
        ekf = ExtendedKalmanFilter(state_dim=3, measurement_dim=3)
        J = ekf.compute_jacobian(lambda s: s, [1.0, 2.0, 3.0])
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert abs(J[i][j] - expected) < 1e-4

    def test_compute_jacobian_linear(self):
        ekf = ExtendedKalmanFilter(state_dim=2, measurement_dim=1)
        # z = x + 2y
        J = ekf.compute_jacobian(lambda s: [s[0] + 2 * s[1]], [0.0, 0.0])
        assert abs(J[0][0] - 1.0) < 1e-4
        assert abs(J[0][1] - 2.0) < 1e-4

    def test_compute_jacobian_quadratic(self):
        ekf = ExtendedKalmanFilter(state_dim=1, measurement_dim=1)
        # z = x^2
        J = ekf.compute_jacobian(lambda s: [s[0] ** 2], [3.0])
        assert abs(J[0][0] - 6.0) < 1e-3

    def test_compute_jacobian_custom_delta(self):
        ekf = ExtendedKalmanFilter(state_dim=1, measurement_dim=1)
        J1 = ekf.compute_jacobian(lambda s: [s[0] ** 2], [3.0], delta=1e-6)
        J2 = ekf.compute_jacobian(lambda s: [s[0] ** 2], [3.0], delta=1e-3)
        # Both should approximate 2*3 = 6
        assert abs(J1[0][0] - 6.0) < 0.01
        assert abs(J2[0][0] - 6.0) < 0.01

    def test_predict_update_cycle(self):
        def cv(state, dt, control):
            return [state[0] + state[1] * dt, state[1]]

        ekf = ExtendedKalmanFilter(
            state_dim=2, measurement_dim=1,
            initial_state=[0.0, 1.0],
            initial_covariance=[[10.0, 0.0], [0.0, 1.0]],
            measurement_noise=[[0.5]],
            process_noise=[[0.01, 0.0], [0.0, 0.01]],
        )
        H = [[1.0, 0.0]]
        for t in range(5):
            ekf.predict(dt=1.0, state_transition_fn=cv)
            ekf.update(measurement=[float(t + 1)], measurement_fn=lambda s: [s[0]])

        state = ekf.get_state()
        # Position should be close to the true trajectory
        assert abs(state.mean[0] - 5.0) < 3.0

    def test_update_reduces_covariance(self):
        ekf = ExtendedKalmanFilter(
            state_dim=1, measurement_dim=1,
            initial_state=[0.0], initial_covariance=[[100.0]],
            measurement_noise=[[1.0]],
        )
        cov_before = ekf.get_state().covariance[0][0]
        ekf.update(measurement=[0.0], measurement_fn=lambda s: [s[0]])
        cov_after = ekf.get_state().covariance[0][0]
        assert cov_after < cov_before

    def test_get_state_returns_copy(self):
        ekf = ExtendedKalmanFilter(state_dim=1, measurement_dim=1, initial_state=[7.0])
        s = ekf.get_state()
        s.mean[0] = 999.0
        assert ekf.get_state().mean[0] == 7.0
