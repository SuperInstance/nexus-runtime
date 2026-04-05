"""NEXUS Dead Reckoning - Inertial Navigation System (INS).

Three competing integration methods for accelerometer + gyroscope fusion:
  1. Euler (simple forward integration)
  2. Runge-Kutta 4th order (RK4)
  3. Complementary filter (high-pass accel + low-pass gyro)

Each method is evaluated by drift rate; the PositionEstimator
selects the best performer based on simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto

from .navigation import VelocityVector


class IntegrationMethod(Enum):
    """INS integration method selection."""

    EULER = auto()
    RK4 = auto()
    COMPLEMENTARY = auto()


@dataclass
class INSState:
    """Internal state of the INS integrator."""

    position_lat: float = 0.0
    position_lon: float = 0.0
    velocity_north: float = 0.0  # m/s
    velocity_east: float = 0.0   # m/s
    heading: float = 0.0         # degrees
    timestamp_ms: int = 0

    @property
    def velocity(self) -> VelocityVector:
        return VelocityVector(north=self.velocity_north, east=self.velocity_east)


@dataclass
class IMUReading:
    """Single IMU sensor reading.

    accel: m/s^2 in body frame (x=forward, y=right, z=down)
    gyro: deg/s in body frame (x=yaw, y=pitch, z=roll)
    """
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0
    gyro_x: float = 0.0  # yaw rate (deg/s)
    gyro_y: float = 0.0  # pitch rate
    gyro_z: float = 0.0  # roll rate
    timestamp_ms: int = 0


@dataclass
class DriftMetrics:
    """Drift metrics for comparing integration methods."""

    total_drift_m: float = 0.0
    drift_rate_m_per_s: float = 0.0
    max_drift_m: float = 0.0
    position_error_lat: float = 0.0
    position_error_lon: float = 0.0
    update_count: int = 0


class INSIntegrator:
    """Inertial Navigation System integrator with competing methods.

    Supports three integration approaches:
      - EULER: Simple forward Euler (fast, more drift)
      - RK4: 4th-order Runge-Kutta (more accurate, slower)
      - COMPLEMENTARY: Fuses accelerometer (high-pass) and gyro (low-pass)

    Usage:
        ins = INSIntegrator(method=IntegrationMethod.RK4)
        ins.initialize(lat=32.0, lon=-117.0, heading=45.0)
        ins.update(imu_reading, dt=0.1)
        state = ins.get_state()
    """

    def __init__(
        self,
        method: IntegrationMethod = IntegrationMethod.COMPLEMENTARY,
        complementary_alpha: float = 0.98,
        accel_bias: tuple[float, float, float] = (0.0, 0.0, 0.0),
        gyro_bias: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.method = method
        self.complementary_alpha = complementary_alpha  # gyro weight
        self.accel_bias = accel_bias
        self.gyro_bias = gyro_bias

        self._state = INSState()
        self._initialized = False
        self._last_timestamp_ms: int = 0
        self._drift_metrics = DriftMetrics()

        # For complementary filter: low-passed velocity from accel
        self._accel_velocity_n: float = 0.0
        self._accel_velocity_e: float = 0.0

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(
        self,
        lat: float,
        lon: float,
        heading: float = 0.0,
        velocity_north: float = 0.0,
        velocity_east: float = 0.0,
        timestamp_ms: int = 0,
    ) -> None:
        """Initialize the INS at a known position."""
        self._state = INSState(
            position_lat=lat,
            position_lon=lon,
            velocity_north=velocity_north,
            velocity_east=velocity_east,
            heading=heading,
            timestamp_ms=timestamp_ms,
        )
        self._last_timestamp_ms = timestamp_ms
        self._initialized = True
        self._drift_metrics = DriftMetrics()

    def reset(self) -> None:
        """Reset the integrator to uninitialized state."""
        self._state = INSState()
        self._initialized = False
        self._last_timestamp_ms = 0
        self._drift_metrics = DriftMetrics()
        self._accel_velocity_n = 0.0
        self._accel_velocity_e = 0.0

    def update(self, reading: IMUReading, dt: float | None = None) -> INSState:
        """Process an IMU reading and return updated state.

        Args:
            reading: IMU sensor reading.
            dt: Time step in seconds. If None, computed from timestamps.

        Returns:
            Updated INSState.

        Raises:
            RuntimeError: If INS not initialized.
        """
        if not self._initialized:
            raise RuntimeError("INS not initialized. Call initialize() first.")

        if dt is None:
            if reading.timestamp_ms <= self._last_timestamp_ms:
                dt = 0.01  # fallback 10ms
            else:
                dt = (reading.timestamp_ms - self._last_timestamp_ms) / 1000.0

        dt = max(dt, 0.001)  # minimum 1ms
        self._last_timestamp_ms = reading.timestamp_ms

        # Remove biases
        ax = reading.accel_x - self.accel_bias[0]
        ay = reading.accel_y - self.accel_bias[1]
        az = reading.accel_z - self.accel_bias[2]
        gx = reading.gyro_x - self.gyro_bias[0]
        gy = reading.gyro_y - self.gyro_bias[1]
        gz = reading.gyro_z - self.gyro_bias[2]

        if self.method == IntegrationMethod.EULER:
            self._integrate_euler(ax, ay, gx, dt)
        elif self.method == IntegrationMethod.RK4:
            self._integrate_rk4(ax, ay, gx, dt)
        else:
            self._integrate_complementary(ax, ay, gx, dt)

        self._drift_metrics.update_count += 1
        return self._state

    def _integrate_euler(
        self, ax: float, ay: float, gyro_yaw: float, dt: float
    ) -> None:
        """Euler integration: simple forward step."""
        heading_rad = math.radians(self._state.heading)

        # Rotate body-frame acceleration to navigation frame
        a_north = ax * math.cos(heading_rad) - ay * math.sin(heading_rad)
        a_east = ax * math.sin(heading_rad) + ay * math.cos(heading_rad)

        # Update velocity
        self._state.velocity_north += a_north * dt
        self._state.velocity_east += a_east * dt

        # Apply damping (simple friction model)
        damping = 0.999
        self._state.velocity_north *= damping
        self._state.velocity_east *= damping

        # Update heading from gyro
        self._state.heading = (self._state.heading + gyro_yaw * dt) % 360

        # Update position
        self._state.position_lat += self._state.velocity_north * dt / 111320.0
        self._state.position_lon += (
            self._state.velocity_east * dt
            / (111320.0 * math.cos(math.radians(self._state.position_lat)))
        )

    def _integrate_rk4(
        self, ax: float, ay: float, gyro_yaw: float, dt: float
    ) -> None:
        """Runge-Kutta 4th order integration.

        Evaluates derivatives at 4 intermediate points for higher accuracy.
        """
        h = dt
        heading_rad = math.radians(self._state.heading)

        # State vector: [lat, lon, vn, ve, heading]
        s0_lat = self._state.position_lat
        s0_lon = self._state.position_lon
        s0_vn = self._state.velocity_north
        s0_ve = self._state.velocity_east
        s0_hdg = self._state.heading

        def derivatives(lat, lon, vn, ve, hdg, _ax, _ay, _gyro):
            """Compute state derivatives."""
            hr = math.radians(hdg)
            cos_hr = math.cos(hr)
            sin_hr = math.sin(hr)
            a_n = _ax * cos_hr - _ay * sin_hr
            a_e = _ax * sin_hr + _ay * cos_hr
            cos_lat = math.cos(math.radians(lat))
            if abs(cos_lat) < 1e-9:
                cos_lat = 1e-9

            d_lat = vn / 111320.0
            d_lon = ve / (111320.0 * cos_lat)
            d_vn = a_n
            d_ve = a_e
            d_hdg = _gyro
            return d_lat, d_lon, d_vn, d_ve, d_hdg

        # k1
        k1 = derivatives(s0_lat, s0_lon, s0_vn, s0_ve, s0_hdg, ax, ay, gyro_yaw)

        # k2
        k2 = derivatives(
            s0_lat + h / 2 * k1[0], s0_lon + h / 2 * k1[1],
            s0_vn + h / 2 * k1[2], s0_ve + h / 2 * k1[3],
            s0_hdg + h / 2 * k1[4],
            ax, ay, gyro_yaw,
        )

        # k3
        k3 = derivatives(
            s0_lat + h / 2 * k2[0], s0_lon + h / 2 * k2[1],
            s0_vn + h / 2 * k2[2], s0_ve + h / 2 * k2[3],
            s0_hdg + h / 2 * k2[4],
            ax, ay, gyro_yaw,
        )

        # k4
        k4 = derivatives(
            s0_lat + h * k3[0], s0_lon + h * k3[1],
            s0_vn + h * k3[2], s0_ve + h * k3[3],
            s0_hdg + h * k3[4],
            ax, ay, gyro_yaw,
        )

        # Combine
        self._state.position_lat += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        self._state.position_lon += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        self._state.velocity_north += h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        self._state.velocity_east += h / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        self._state.heading += h / 6 * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4])

        # Apply damping
        damping = 0.999
        self._state.velocity_north *= damping
        self._state.velocity_east *= damping
        self._state.heading %= 360

    def _integrate_complementary(
        self, ax: float, ay: float, gyro_yaw: float, dt: float
    ) -> None:
        """Complementary filter: fuses accelerometer velocity with gyro.

        High-pass filter on accelerometer integration (removes drift).
        Low-pass filter on gyro heading (smooth, low noise).
        Alpha controls the blend: alpha=0.98 means 98% gyro, 2% accel.
        """
        alpha = self.complementary_alpha
        heading_rad = math.radians(self._state.heading)

        # Accelerometer-derived velocity in nav frame
        a_north = ax * math.cos(heading_rad) - ay * math.sin(heading_rad)
        a_east = ax * math.sin(heading_rad) + ay * math.cos(heading_rad)

        # Integrate accelerometer (high-pass component)
        self._accel_velocity_n += a_north * dt
        self._accel_velocity_e += a_east * dt

        # Apply high-pass: remove low-frequency drift from accel
        # (leaky integrator)
        leak = 0.01
        self._accel_velocity_n *= (1.0 - leak)
        self._accel_velocity_e *= (1.0 - leak)

        # Complementary blend for velocity
        self._state.velocity_north = alpha * self._state.velocity_north + (1.0 - alpha) * self._accel_velocity_n
        self._state.velocity_east = alpha * self._state.velocity_east + (1.0 - alpha) * self._accel_velocity_e

        # Update heading from gyro (already smooth/low-noise)
        self._state.heading = (self._state.heading + gyro_yaw * dt) % 360

        # Update position
        cos_lat = math.cos(math.radians(self._state.position_lat))
        if abs(cos_lat) < 1e-9:
            cos_lat = 1e-9

        self._state.position_lat += self._state.velocity_north * dt / 111320.0
        self._state.position_lon += (
            self._state.velocity_east * dt / (111320.0 * cos_lat)
        )

    def correct_position(self, lat: float, lon: float) -> None:
        """Correct INS position (e.g., from GPS fix)."""
        if self._initialized:
            self._drift_metrics.total_drift_m += NavigationMath_haversine(
                self._state.position_lat, self._state.position_lon, lat, lon
            )
            err_lat = abs(self._state.position_lat - lat)
            err_lon = abs(self._state.position_lon - lon)
            self._drift_metrics.position_error_lat = err_lat
            self._drift_metrics.position_error_lon = err_lon
            self._drift_metrics.max_drift_m = max(
                self._drift_metrics.max_drift_m,
                self._drift_metrics.position_error_lat * 111320.0,
            )

        self._state.position_lat = lat
        self._state.position_lon = lon

        # Reset accel integration drift on position correction
        self._accel_velocity_n = self._state.velocity_north
        self._accel_velocity_e = self._state.velocity_east

    def correct_heading(self, heading: float) -> None:
        """Correct INS heading (e.g., from compass)."""
        self._state.heading = heading % 360

    def get_state(self) -> INSState:
        """Return current INS state."""
        return self._state

    def get_drift_metrics(self) -> DriftMetrics:
        """Return accumulated drift metrics."""
        return self._drift_metrics

    @staticmethod
    def compare_methods(
        imu_readings: list[IMUReading],
        true_positions: list[tuple[float, float]],
        initial_lat: float,
        initial_lon: float,
        initial_heading: float = 0.0,
    ) -> dict[IntegrationMethod, DriftMetrics]:
        """Compare all three integration methods against ground truth.

        Args:
            imu_readings: List of IMU readings (must have timestamps).
            true_positions: Ground truth positions (lat, lon) at each reading.
            initial_lat: Starting latitude.
            initial_lon: Starting longitude.
            initial_heading: Starting heading in degrees.

        Returns:
            Dictionary mapping method to drift metrics.
        """
        results: dict[IntegrationMethod, DriftMetrics] = {}

        for method in IntegrationMethod:
            ins = INSIntegrator(method=method)
            ins.initialize(initial_lat, initial_lon, initial_heading)

            total_drift = 0.0
            max_drift = 0.0
            count = 0

            for reading, (true_lat, true_lon) in zip(imu_readings, true_positions):
                ins.update(reading)
                count += 1
                state = ins.get_state()
                from .navigation import NavigationMath
                drift = NavigationMath.haversine_distance(
                    state.position_lat, state.position_lon, true_lat, true_lon
                )
                total_drift += drift
                max_drift = max(max_drift, drift)

            results[method] = DriftMetrics(
                total_drift_m=total_drift,
                drift_rate_m_per_s=total_drift / max(count * 0.01, 1.0),
                max_drift_m=max_drift,
                update_count=count,
            )

        return results


def NavigationMath_haversine(lat1, lon1, lat2, lon2):
    """Standalone haversine to avoid circular import in correct_position."""
    import math
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 6371000.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
