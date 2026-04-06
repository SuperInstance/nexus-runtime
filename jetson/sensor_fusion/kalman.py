"""Extended Kalman Filter implementations for sensor fusion.

Pure Python — no external dependencies.
Uses math and copy for matrix operations.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, List, Optional


# ---------------------------------------------------------------------------
# Matrix utilities (pure Python, no numpy)
# ---------------------------------------------------------------------------

def _mat_zeros(n: int, m: int) -> List[List[float]]:
    """Create an n×m zero matrix."""
    return [[0.0] * m for _ in range(n)]


def _mat_identity(n: int) -> List[List[float]]:
    """Create an n×n identity matrix."""
    m = _mat_zeros(n, n)
    for i in range(n):
        m[i][i] = 1.0
    return m


def _mat_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Multiply two matrices A (p×q) and B (q×r) -> (p×r)."""
    p = len(A)
    q = len(A[0])
    r = len(B[0])
    C = _mat_zeros(p, r)
    for i in range(p):
        for j in range(r):
            s = 0.0
            for k in range(q):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


def _mat_transpose(A: List[List[float]]) -> List[List[float]]:
    """Transpose matrix A."""
    n = len(A)
    m = len(A[0])
    T = _mat_zeros(m, n)
    for i in range(n):
        for j in range(m):
            T[j][i] = A[i][j]
    return T


def _mat_add(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Add two matrices element-wise."""
    n = len(A)
    m = len(A[0])
    C = _mat_zeros(n, m)
    for i in range(n):
        for j in range(m):
            C[i][j] = A[i][j] + B[i][j]
    return C


def _mat_sub(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Subtract B from A element-wise."""
    n = len(A)
    m = len(A[0])
    C = _mat_zeros(n, m)
    for i in range(n):
        for j in range(m):
            C[i][j] = A[i][j] - B[i][j]
    return C


def _mat_scale(A: List[List[float]], s: float) -> List[List[float]]:
    """Scale matrix A by scalar s."""
    n = len(A)
    m = len(A[0])
    C = _mat_zeros(n, m)
    for i in range(n):
        for j in range(m):
            C[i][j] = A[i][j] * s
    return C


def _mat_vec_mul(A: List[List[float]], v: List[float]) -> List[float]:
    """Multiply matrix A (n×m) by column vector v (m) -> (n)."""
    n = len(A)
    m = len(v)
    result = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(m):
            s += A[i][j] * v[j]
        result[i] = s
    return result


def _vec_add(a: List[float], b: List[float]) -> List[float]:
    """Add two vectors."""
    return [ai + bi for ai, bi in zip(a, b)]


def _vec_sub(a: List[float], b: List[float]) -> List[float]:
    """Subtract vector b from a."""
    return [ai - bi for ai, bi in zip(a, b)]


def _mat_invert_2x2(M: List[List[float]]) -> List[List[float]]:
    """Invert a 2×2 matrix."""
    a, b = M[0][0], M[0][1]
    c, d = M[1][0], M[1][1]
    det = a * d - b * c
    if abs(det) < 1e-12:
        return None
    return [[d / det, -b / det], [-c / det, a / det]]


def _mat_invert_general(M: List[List[float]]) -> Optional[List[List[float]]]:
    """Invert a general square matrix using Gauss-Jordan elimination."""
    n = len(M)
    # Augmented matrix [M | I]
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
           for i, row in enumerate(M)]
    for col in range(n):
        # Find pivot
        max_val = abs(aug[col][col])
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]
        if abs(aug[col][col]) < 1e-12:
            return None
        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= factor * aug[col][j]
    return [row[n:] for row in aug]


def _mat_inverse(M: List[List[float]]) -> Optional[List[List[float]]]:
    """Invert a square matrix, choosing the best method."""
    n = len(M)
    if n == 1:
        if abs(M[0][0]) < 1e-12:
            return None
        return [[1.0 / M[0][0]]]
    if n == 2:
        return _mat_invert_2x2(M)
    return _mat_invert_general(M)


# ---------------------------------------------------------------------------
# KalmanState
# ---------------------------------------------------------------------------

@dataclass
class KalmanState:
    """State estimate for a Kalman filter."""
    mean: List[float]
    covariance: List[List[float]]
    timestamp: float = 0.0

    def copy(self) -> KalmanState:
        return KalmanState(
            mean=self.mean[:],
            covariance=deepcopy(self.covariance),
            timestamp=self.timestamp,
        )


# ---------------------------------------------------------------------------
# LinearKalmanFilter
# ---------------------------------------------------------------------------

class LinearKalmanFilter:
    """Standard (linear) Kalman filter.

    Supports configurable state-transition matrix F, measurement matrix H,
    process noise Q, and measurement noise R.
    """

    def __init__(
        self,
        state_dim: int,
        measurement_dim: int,
        state_transition: Optional[List[List[float]]] = None,
        measurement_matrix: Optional[List[List[float]]] = None,
        process_noise: Optional[List[List[float]]] = None,
        measurement_noise: Optional[List[List[float]]] = None,
        initial_state: Optional[List[float]] = None,
        initial_covariance: Optional[List[List[float]]] = None,
    ) -> None:
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # Default F = identity
        if state_transition is not None:
            self.F = deepcopy(state_transition)
        else:
            self.F = _mat_identity(state_dim)

        # Default H: measures first measurement_dim states directly
        if measurement_matrix is not None:
            self.H = deepcopy(measurement_matrix)
        else:
            self.H = _mat_zeros(measurement_dim, state_dim)
            for i in range(min(measurement_dim, state_dim)):
                self.H[i][i] = 1.0

        # Noise covariances
        self.Q = process_noise if process_noise is not None else _mat_identity(state_dim)
        self.R = measurement_noise if measurement_noise is not None else _mat_identity(measurement_dim)

        # State
        if initial_state is not None:
            self.x = initial_state[:]
        else:
            self.x = [0.0] * state_dim

        if initial_covariance is not None:
            self.P = deepcopy(initial_covariance)
        else:
            self.P = _mat_identity(state_dim)

        self.timestamp = 0.0

    # -- public API --------------------------------------------------------

    def predict(self, dt: float, control_input: Optional[List[float]] = None) -> KalmanState:
        """Predict step: project state and covariance forward by dt."""
        # For a linear filter with constant F, dt scales the noise.
        # x = F * x (+ B*u if provided)
        if control_input is not None:
            # Assume B = I for simplicity; control adds directly
            u = control_input[:self.state_dim]
            self.x = _vec_add(_mat_vec_mul(self.F, self.x), u)
        else:
            self.x = _mat_vec_mul(self.F, self.x)

        # P = F * P * F^T + Q * dt
        Ft = _mat_transpose(self.F)
        scaled_Q = _mat_scale(self.Q, dt)
        self.P = _mat_add(_mat_mul(_mat_mul(self.F, self.P), Ft), scaled_Q)

        self.timestamp += dt
        return self.get_state()

    def update(
        self,
        measurement: List[float],
        measurement_noise: Optional[List[List[float]]] = None,
        measurement_matrix: Optional[List[List[float]]] = None,
    ) -> KalmanState:
        """Update step: incorporate a measurement."""
        H = measurement_matrix if measurement_matrix is not None else self.H
        R = measurement_noise if measurement_noise is not None else self.R

        # Innovation
        z = measurement[:self.measurement_dim]
        y = _vec_sub(z, _mat_vec_mul(H, self.x))

        # Innovation covariance: S = H * P * H^T + R
        Ht = _mat_transpose(H)
        S = _mat_add(_mat_mul(_mat_mul(H, self.P), Ht), R)

        # Kalman gain: K = P * H^T * S^{-1}
        S_inv = _mat_inverse(S)
        if S_inv is None:
            return self.get_state()
        K = _mat_mul(_mat_mul(self.P, Ht), S_inv)

        # State update: x = x + K * y
        Ky = _mat_vec_mul(K, y)
        self.x = _vec_add(self.x, Ky)

        # Covariance update: P = (I - K*H) * P
        n = self.state_dim
        KH = _mat_mul(K, H)
        I_KH = _mat_sub(_mat_identity(n), KH)
        self.P = _mat_mul(I_KH, self.P)

        return self.get_state()

    def get_state(self) -> KalmanState:
        """Return current state estimate."""
        return KalmanState(
            mean=self.x[:],
            covariance=deepcopy(self.P),
            timestamp=self.timestamp,
        )

    def set_process_noise(self, Q: List[List[float]]) -> None:
        """Set process noise covariance Q."""
        self.Q = deepcopy(Q)

    def set_measurement_noise(self, R: List[List[float]]) -> None:
        """Set measurement noise covariance R."""
        self.R = deepcopy(R)

    def compute_innovation(self, measurement: List[float]) -> List[float]:
        """Compute the innovation (measurement residual) without updating."""
        z = measurement[:self.measurement_dim]
        predicted_z = _mat_vec_mul(self.H, self.x)
        return _vec_sub(z, predicted_z)


# ---------------------------------------------------------------------------
# ExtendedKalmanFilter
# ---------------------------------------------------------------------------

class ExtendedKalmanFilter:
    """Extended Kalman Filter for non-linear systems.

    Uses numerical Jacobian computation and user-supplied state-transition
    and measurement functions.
    """

    def __init__(
        self,
        state_dim: int,
        measurement_dim: int,
        process_noise: Optional[List[List[float]]] = None,
        measurement_noise: Optional[List[List[float]]] = None,
        initial_state: Optional[List[float]] = None,
        initial_covariance: Optional[List[List[float]]] = None,
    ) -> None:
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        self.Q = process_noise if process_noise is not None else _mat_identity(state_dim)
        self.R = measurement_noise if measurement_noise is not None else _mat_identity(measurement_dim)

        if initial_state is not None:
            self.x = initial_state[:]
        else:
            self.x = [0.0] * state_dim

        if initial_covariance is not None:
            self.P = deepcopy(initial_covariance)
        else:
            self.P = _mat_identity(state_dim)

        self.timestamp = 0.0

    def predict(
        self,
        dt: float,
        control_input: Optional[List[float]] = None,
        state_transition_fn: Optional[Callable[[List[float], float, Optional[List[float]]], List[float]]] = None,
        process_noise_fn: Optional[Callable[[List[float], float], List[List[float]]]] = None,
    ) -> KalmanState:
        """Predict step with non-linear state transition."""
        if state_transition_fn is not None:
            self.x = state_transition_fn(self.x, dt, control_input)
        elif control_input is not None:
            self.x = _vec_add(self.x, control_input[:self.state_dim])

        # Compute Jacobian of state transition
        if state_transition_fn is not None:
            F = self.compute_jacobian(
                lambda s: state_transition_fn(s, dt, control_input), self.x
            )
        else:
            F = _mat_identity(self.state_dim)

        # Process noise
        if process_noise_fn is not None:
            Q = process_noise_fn(self.x, dt)
        else:
            Q = _mat_scale(self.Q, dt)

        Ft = _mat_transpose(F)
        self.P = _mat_add(_mat_mul(_mat_mul(F, self.P), Ft), Q)

        self.timestamp += dt
        return self.get_state()

    def update(
        self,
        measurement: List[float],
        measurement_fn: Callable[[List[float]], List[float]],
        measurement_noise: Optional[List[List[float]]] = None,
    ) -> KalmanState:
        """Update step with non-linear measurement function."""
        H = self.compute_jacobian(measurement_fn, self.x)
        R = measurement_noise if measurement_noise is not None else self.R

        # Predicted measurement
        z_pred = measurement_fn(self.x)
        z = measurement[:self.measurement_dim]
        y = _vec_sub(z, z_pred)

        # Innovation covariance
        Ht = _mat_transpose(H)
        S = _mat_add(_mat_mul(_mat_mul(H, self.P), Ht), R)

        # Kalman gain
        S_inv = _mat_inverse(S)
        if S_inv is None:
            return self.get_state()
        K = _mat_mul(_mat_mul(self.P, Ht), S_inv)

        # State update
        Ky = _mat_vec_mul(K, y)
        self.x = _vec_add(self.x, Ky)

        # Covariance update
        n = self.state_dim
        KH = _mat_mul(K, H)
        I_KH = _mat_sub(_mat_identity(n), KH)
        self.P = _mat_mul(I_KH, self.P)

        return self.get_state()

    def get_state(self) -> KalmanState:
        """Return current state estimate."""
        return KalmanState(
            mean=self.x[:],
            covariance=deepcopy(self.P),
            timestamp=self.timestamp,
        )

    def compute_jacobian(
        self,
        fn: Callable[[List[float]], List[float]],
        state: List[float],
        delta: float = 1e-5,
    ) -> List[List[float]]:
        """Compute numerical Jacobian of fn at state using central differences."""
        n = len(state)
        f0 = fn(state)
        m = len(f0)
        J = _mat_zeros(m, n)
        for j in range(n):
            sp = state[:]
            sm = state[:]
            sp[j] += delta
            sm[j] -= delta
            fp = fn(sp)
            fm = fn(sm)
            for i in range(m):
                J[i][j] = (fp[i] - fm[i]) / (2.0 * delta)
        return J
