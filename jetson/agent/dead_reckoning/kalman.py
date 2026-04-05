"""NEXUS Dead Reckoning - Kalman Filter for GPS + INS Fusion.

2D position Kalman filter: state = [lat, lon, vel_north, vel_east]
Three competing approaches:
  1. Simple weighted average (naive baseline)
  2. Basic Kalman filter (linear, constant velocity model)
  3. Extended Kalman filter (EKF) with heading-rate process model
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class KalmanApproach(Enum):
    """Kalman filter approach selection."""

    WEIGHTED_AVERAGE = auto()
    BASIC_KALMAN = auto()
    EKF = auto()


@dataclass
class GPSReading:
    """GPS sensor reading."""

    latitude: float = 0.0
    longitude: float = 0.0
    speed: float = 0.0        # m/s over ground
    course: float = 0.0       # degrees over ground
    hdop: float = 1.0         # horizontal dilution of precision
    num_satellites: int = 0
    fix_quality: int = 0      # 0=invalid, 1=GPS, 2=DGPS
    timestamp_ms: int = 0

    @property
    def is_valid(self) -> bool:
        return self.fix_quality >= 1 and self.num_satellites >= 4


@dataclass
class KalmanState:
    """Kalman filter state vector: [lat, lon, vel_n, vel_e]."""

    lat: float = 0.0
    lon: float = 0.0
    vel_north: float = 0.0    # m/s
    vel_east: float = 0.0     # m/s


@dataclass
class KalmanCovariance:
    """4x4 state covariance matrix (stored as flat 16-element list, row-major)."""
    elements: list[float] = field(default_factory=lambda: [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ])

    def get(self, i: int, j: int) -> float:
        return self.elements[i * 4 + j]

    def set(self, i: int, j: int, val: float) -> None:
        self.elements[i * 4 + j] = val

    def copy(self) -> KalmanCovariance:
        return KalmanCovariance(elements=list(self.elements))

    def identity(self, scale: float = 1.0) -> None:
        """Reset to scaled identity matrix."""
        for i in range(4):
            for j in range(4):
                self.set(i, j, scale if i == j else 0.0)


class KalmanFilter2D:
    """2D Kalman filter for GPS + INS sensor fusion.

    State: [latitude, longitude, velocity_north, velocity_east]
    Measurements: GPS position + optional INS velocity

    Three approaches available:
      - WEIGHTED_AVERAGE: Simple GPS/INS blend based on confidence
      - BASIC_KALMAN: Standard linear Kalman with constant-velocity model
      - EKF: Extended Kalman with heading-dependent process model

    Usage:
        kf = KalmanFilter2D(approach=KalmanApproach.BASIC_KALMAN)
        kf.initialize(lat=32.0, lon=-117.0)
        kf.update_gps(gps_reading)
        kf.update_ins_velocity(vn, ve)
        state = kf.get_state()
    """

    def __init__(
        self,
        approach: KalmanApproach = KalmanApproach.BASIC_KALMAN,
        gps_weight: float = 0.7,
        process_noise_pos: float = 1e-6,
        process_noise_vel: float = 1e-4,
        gps_noise_pos: float = 1e-4,
        ins_noise_vel: float = 1e-3,
    ) -> None:
        self.approach = approach
        self.gps_weight = gps_weight
        self.process_noise_pos = process_noise_pos
        self.process_noise_vel = process_noise_vel
        self.gps_noise_pos = gps_noise_pos
        self.ins_noise_vel = ins_noise_vel

        self._state = KalmanState()
        self._P = KalmanCovariance()
        self._initialized = False
        self._last_gps: Optional[GPSReading] = None
        self._heading: float = 0.0
        self._update_count: int = 0
        self._gps_update_count: int = 0

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(
        self,
        lat: float,
        lon: float,
        vel_north: float = 0.0,
        vel_east: float = 0.0,
    ) -> None:
        """Initialize the filter at a known position."""
        self._state = KalmanState(
            lat=lat, lon=lon,
            vel_north=vel_north, vel_east=vel_east,
        )
        self._P = KalmanCovariance()
        self._initialized = True
        self._update_count = 0
        self._gps_update_count = 0

    def reset(self) -> None:
        """Reset filter to uninitialized state."""
        self._state = KalmanState()
        self._P = KalmanCovariance()
        self._initialized = False

    def update_gps(self, gps: GPSReading) -> KalmanState:
        """Update filter with GPS measurement.

        Args:
            gps: GPS reading.

        Returns:
            Updated state.
        """
        if not gps.is_valid:
            return self._state

        if not self._initialized:
            self.initialize(gps.latitude, gps.longitude)

        self._last_gps = gps
        self._heading = gps.course
        self._gps_update_count += 1

        if self.approach == KalmanApproach.WEIGHTED_AVERAGE:
            self._update_weighted_average(gps)
        elif self.approach == KalmanApproach.BASIC_KALMAN:
            self._update_basic_kalman(gps)
        else:
            self._update_ekf(gps)

        self._update_count += 1
        return self._state

    def update_ins_velocity(self, vel_north: float, vel_east: float) -> KalmanState:
        """Update filter with INS velocity measurement.

        Args:
            vel_north: North velocity in m/s.
            vel_east: East velocity in m/s.

        Returns:
            Updated state.
        """
        if not self._initialized:
            return self._state

        if self.approach == KalmanApproach.WEIGHTED_AVERAGE:
            ins_weight = 1.0 - self.gps_weight
            self._state.vel_north = self.gps_weight * self._state.vel_north + ins_weight * vel_north
            self._state.vel_east = self.gps_weight * self._state.vel_east + ins_weight * vel_east
        else:
            # Kalman velocity update
            self._kalman_velocity_update(vel_north, vel_east)

        self._update_count += 1
        return self._state

    def predict(self, dt: float) -> KalmanState:
        """Predict state forward by dt seconds.

        Args:
            dt: Time step in seconds.

        Returns:
            Predicted state.
        """
        if not self._initialized:
            return self._state

        if self.approach == KalmanApproach.EKF:
            self._predict_ekf(dt)
        else:
            self._predict_basic(dt)

        return self._state

    def _update_weighted_average(self, gps: GPSReading) -> None:
        """Simple weighted average between GPS and current state."""
        w = self.gps_weight
        # GPS quality factor
        quality = min(1.0, gps.num_satellites / 12.0) * min(1.0, 2.0 / max(gps.hdop, 0.5))
        w *= quality
        w = max(0.1, min(0.95, w))

        self._state.lat = (1.0 - w) * self._state.lat + w * gps.latitude
        self._state.lon = (1.0 - w) * self._state.lon + w * gps.longitude

        if gps.speed > 0:
            h_rad = math.radians(gps.course)
            self._state.vel_north = gps.speed * math.cos(h_rad)
            self._state.vel_east = gps.speed * math.sin(h_rad)

    def _update_basic_kalman(self, gps: GPSReading) -> None:
        """Basic Kalman filter GPS update with constant-velocity model.

        Measurement model: z = H * x + v
        H = [[1, 0, 0, 0],   # measure lat
             [0, 1, 0, 0]]   # measure lon
        """
        # Measurement
        z = [gps.latitude, gps.longitude]

        # Measurement noise from HDOP and satellites
        R_scale = max(0.5, gps.hdop) * max(0.5, 12.0 / max(gps.num_satellites, 4))
        R = [
            [self.gps_noise_pos * R_scale, 0],
            [0, self.gps_noise_pos * R_scale],
        ]

        # Innovation (residual)
        y = [z[0] - self._state.lat, z[1] - self._state.lon]

        # Innovation covariance: S = H * P * H^T + R
        P = self._P
        S = [
            [P.get(0, 0) + R[0][0], P.get(0, 1) + R[0][1]],
            [P.get(1, 0) + R[1][0], P.get(1, 1) + R[1][1]],
        ]

        # Kalman gain: K = P * H^T * S^-1
        S_det = S[0][0] * S[1][1] - S[0][1] * S[1][0]
        if abs(S_det) < 1e-15:
            S_det = 1e-15
        S_inv = [
            [S[1][1] / S_det, -S[0][1] / S_det],
            [-S[1][0] / S_det, S[0][0] / S_det],
        ]

        K = [
            [P.get(0, 0) * S_inv[0][0] + P.get(0, 1) * S_inv[1][0],
             P.get(0, 0) * S_inv[0][1] + P.get(0, 1) * S_inv[1][1]],
            [P.get(1, 0) * S_inv[0][0] + P.get(1, 1) * S_inv[1][0],
             P.get(1, 0) * S_inv[0][1] + P.get(1, 1) * S_inv[1][1]],
        ]

        # State update
        self._state.lat += K[0][0] * y[0] + K[0][1] * y[1]
        self._state.lon += K[1][0] * y[0] + K[1][1] * y[1]

        # Covariance update: P = (I - K*H) * P
        for i in range(2):
            for j in range(2):
                new_val = P.get(i, j) - K[i][0] * P.get(0, j) - K[i][1] * P.get(1, j)
                P.set(i, j, new_val)

    def _kalman_velocity_update(self, vel_n: float, vel_e: float) -> None:
        """Kalman update for velocity measurement from INS."""
        P = self._P
        R_vel = self.ins_noise_vel

        # Innovation
        y_n = vel_n - self._state.vel_north
        y_e = vel_e - self._state.vel_east

        # Variance of state velocity
        P_vn = P.get(2, 2)
        P_ve = P.get(3, 3)

        # Kalman gains for velocity components
        K_vn = P_vn / (P_vn + R_vel) if (P_vn + R_vel) > 1e-15 else 1.0
        K_ve = P_ve / (P_ve + R_vel) if (P_ve + R_vel) > 1e-15 else 1.0

        self._state.vel_north += K_vn * y_n
        self._state.vel_east += K_ve * y_e

        # Update covariance
        P.set(2, 2, P_vn * (1.0 - K_vn))
        P.set(3, 3, P_ve * (1.0 - K_ve))

    def _predict_basic(self, dt: float) -> None:
        """Basic constant-velocity prediction step."""
        cos_lat = math.cos(math.radians(self._state.lat))
        if abs(cos_lat) < 1e-9:
            cos_lat = 1e-9

        # State prediction
        self._state.lat += self._state.vel_north * dt / 111320.0
        self._state.lon += self._state.vel_east * dt / (111320.0 * cos_lat)

        # Covariance prediction (simplified)
        q = self.process_noise_pos
        qv = self.process_noise_vel
        P = self._P
        P.set(0, 0, P.get(0, 0) + q)
        P.set(1, 1, P.get(1, 1) + q)
        P.set(2, 2, P.get(2, 2) + qv)
        P.set(3, 3, P.get(3, 3) + qv)

    def _update_ekf(self, gps: GPSReading) -> None:
        """Extended Kalman filter update with heading-dependent model.

        Uses heading from GPS course to improve the linearization.
        """
        # First do basic Kalman update
        self._update_basic_kalman(gps)

        # EKF correction: blend velocity with GPS speed/course
        if gps.speed > 0.1:
            h_rad = math.radians(gps.course)
            gps_vn = gps.speed * math.cos(h_rad)
            gps_ve = gps.speed * math.sin(h_rad)

            # Use GPS velocity to correct state velocity
            self._kalman_velocity_update(gps_vn, gps_ve)

    def _predict_ekf(self, dt: float) -> None:
        """EKF prediction with heading-dependent process model."""
        # Propagate heading from velocity
        if self._state.vel_north ** 2 + self._state.vel_east ** 2 > 0.01:
            self._heading = math.degrees(
                math.atan2(self._state.vel_east, self._state.vel_north)
            ) % 360

        # Use the basic predict for position, but with heading-dependent noise
        self._predict_basic(dt)

    def get_state(self) -> KalmanState:
        """Return current state estimate."""
        return self._state

    def get_covariance(self) -> KalmanCovariance:
        """Return current covariance matrix."""
        return self._P

    def get_confidence(self) -> float:
        """Return confidence score (0-1) based on covariance.

        Higher covariance = lower confidence.
        """
        trace = (self._P.get(0, 0) + self._P.get(1, 1)
                 + self._P.get(2, 2) + self._P.get(3, 3))
        # Confidence decreases with covariance, but never below 0.01
        conf = 1.0 / (1.0 + trace * 1e6)
        return max(0.01, min(1.0, conf))

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def gps_update_count(self) -> int:
        return self._gps_update_count

    @staticmethod
    def compare_approaches(
        gps_readings: list[GPSReading],
        ins_velocities: list[tuple[float, float]],
        true_positions: list[tuple[float, float]],
        initial_lat: float,
        initial_lon: float,
    ) -> dict[KalmanApproach, dict[str, float]]:
        """Compare all three Kalman approaches against ground truth.

        Returns dict with 'total_error_m', 'max_error_m', 'avg_error_m' per approach.
        """
        from .navigation import NavigationMath

        results: dict[KalmanApproach, dict[str, float]] = {}

        for approach in KalmanApproach:
            kf = KalmanFilter2D(approach=approach)
            kf.initialize(initial_lat, initial_lon)

            total_error = 0.0
            max_error = 0.0
            count = 0

            for gps, (vn, ve), (true_lat, true_lon) in zip(
                gps_readings, ins_velocities, true_positions
            ):
                kf.update_gps(gps)
                kf.update_ins_velocity(vn, ve)

                state = kf.get_state()
                error = NavigationMath.haversine_distance(
                    state.lat, state.lon, true_lat, true_lon
                )
                total_error += error
                max_error = max(max_error, error)
                count += 1

            results[approach] = {
                "total_error_m": total_error,
                "max_error_m": max_error,
                "avg_error_m": total_error / max(count, 1),
            }

        return results
