"""NEXUS Dead Reckoning - Full Position Estimator Pipeline.

Combines INS, Kalman filter, and compass into a complete dead
reckoning system with automatic sensor quality tracking, graceful
degradation, and GPS re-acquisition.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .ins import INSIntegrator, INSState, IntegrationMethod, IMUReading
from .kalman import KalmanFilter2D, KalmanApproach, KalmanState, GPSReading
from .compass import CompassFusion, CompassReading, HeadingState
from .navigation import NavigationMath, VelocityVector


class NavigationMode(Enum):
    """Navigation mode based on available sensors."""

    FULL_FUSION = auto()       # GPS + INS + compass
    INS_ONLY = auto()          # GPS lost, dead reckoning
    GPS_ONLY = auto()          # INS failed, GPS only
    DEGRADED = auto()          # Both degraded, low confidence
    INITIALIZING = auto()      # Not yet initialized


@dataclass
class SensorQuality:
    """Quality metrics for each sensor source."""

    gps_quality: float = 0.0        # 0-1
    gps_signal_quality: float = 0.0  # based on satellites, HDOP
    ins_quality: float = 0.0        # based on drift
    compass_quality: float = 0.0    # based on gyro_only_samples
    overall_confidence: float = 0.0 # combined 0-1

    gps_dropout_count: int = 0
    gps_dropout_duration_ms: int = 0
    last_gps_time_ms: int = 0
    ins_drift_rate: float = 0.0     # m/s

    @property
    def gps_available(self) -> bool:
        return self.gps_quality > 0.1

    @property
    def ins_available(self) -> bool:
        return self.ins_quality > 0.1

    @property
    def compass_available(self) -> bool:
        return self.compass_quality > 0.1


@dataclass
class PositionEstimate:
    """Complete position estimate from the dead reckoning pipeline."""

    latitude: float = 0.0
    longitude: float = 0.0
    heading: float = 0.0
    speed: float = 0.0            # m/s
    course_over_ground: float = 0.0
    velocity_north: float = 0.0
    velocity_east: float = 0.0
    confidence: float = 0.0
    mode: NavigationMode = NavigationMode.INITIALIZING
    sensor_quality: SensorQuality = field(default_factory=SensorQuality)
    timestamp_ms: int = 0

    @property
    def velocity(self) -> VelocityVector:
        return VelocityVector(north=self.velocity_north, east=self.velocity_east)


@dataclass
class GPSDropoutEvent:
    """Record of a GPS dropout event."""

    start_time_ms: int = 0
    end_time_ms: int = 0
    duration_ms: int = 0
    max_drift_m: float = 0.0


class PositionEstimator:
    """Full dead reckoning pipeline for marine navigation.

    Fuses GPS, IMU, and compass into a continuous position estimate.
    Handles GPS dropout gracefully (pure INS dead reckoning) and
    re-acquisition (Kalman corrects INS drift).

    Pipeline:
    1. Receive sensor data (GPS, IMU, compass)
    2. Compass fusion → heading estimate
    3. INS integration → velocity + position
    4. Kalman filter → fused position + velocity
    5. Quality assessment → confidence score
    6. Output PositionEstimate

    Usage:
        est = PositionEstimator()
        est.initialize(lat=32.0, lon=-117.0)
        estimate = est.update(
            gps=gps_reading,
            imu=imu_reading,
            compass=compass_reading,
        )
    """

    def __init__(
        self,
        ins_method: IntegrationMethod = IntegrationMethod.COMPLEMENTARY,
        kalman_approach: KalmanApproach = KalmanApproach.BASIC_KALMAN,
        gps_dropout_timeout_ms: int = 5000,
        arrival_radius_m: float = 10.0,
    ) -> None:
        self.ins_method = ins_method
        self.kalman_approach = kalman_approach
        self.gps_dropout_timeout_ms = gps_dropout_timeout_ms

        self._ins = INSIntegrator(method=ins_method)
        self._kalman = KalmanFilter2D(approach=kalman_approach)
        self._compass = CompassFusion()

        self._initialized = False
        self._mode = NavigationMode.INITIALIZING
        self._sensor_quality = SensorQuality()
        self._gps_dropout_active = False
        self._gps_dropout_start_ms: int = 0
        self._gps_dropout_history: list[GPSDropoutEvent] = []
        self._last_gps: Optional[GPSReading] = None
        self._last_estimate: Optional[PositionEstimate] = None
        self._update_count: int = 0
        self._position_at_dropout: tuple[float, float] = (0.0, 0.0)

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def mode(self) -> NavigationMode:
        return self._mode

    @property
    def sensor_quality(self) -> SensorQuality:
        return self._sensor_quality

    def initialize(
        self,
        lat: float,
        lon: float,
        heading: float = 0.0,
        vel_north: float = 0.0,
        vel_east: float = 0.0,
    ) -> None:
        """Initialize the position estimator at a known position."""
        self._ins.initialize(lat, lon, heading, vel_north, vel_east)
        self._kalman.initialize(lat, lon, vel_north, vel_east)
        self._compass.initialize(heading)

        self._initialized = True
        self._mode = NavigationMode.FULL_FUSION
        self._sensor_quality = SensorQuality(
            gps_quality=1.0,
            gps_signal_quality=1.0,
            ins_quality=1.0,
            compass_quality=1.0,
            overall_confidence=1.0,
        )
        self._last_estimate = PositionEstimate(
            latitude=lat, longitude=lon, heading=heading,
            speed=0.0, confidence=1.0,
            mode=NavigationMode.FULL_FUSION,
            velocity_north=vel_north,
            velocity_east=vel_east,
        )

    def reset(self) -> None:
        """Reset to uninitialized state."""
        self._ins.reset()
        self._kalman.reset()
        self._compass.reset()
        self._initialized = False
        self._mode = NavigationMode.INITIALIZING
        self._sensor_quality = SensorQuality()
        self._last_gps = None
        self._last_estimate = None
        self._gps_dropout_active = False

    def update(
        self,
        gps: Optional[GPSReading] = None,
        imu: Optional[IMUReading] = None,
        compass: Optional[CompassReading] = None,
        dt: float | None = None,
    ) -> PositionEstimate:
        """Process sensor data and return position estimate.

        Args:
            gps: GPS reading (None if GPS unavailable).
            imu: IMU reading (None if IMU unavailable).
            compass: Compass reading (None if compass unavailable).
            dt: Time step in seconds (auto-computed if None).

        Returns:
            Current position estimate.
        """
        if not self._initialized:
            # Auto-initialize from GPS if available
            if gps and gps.is_valid:
                self.initialize(gps.latitude, gps.longitude)
            else:
                return PositionEstimate(mode=NavigationMode.INITIALIZING)

        current_time = imu.timestamp_ms if imu else (
            gps.timestamp_ms if gps else
            (compass.timestamp_ms if compass else int(time.time() * 1000))
        )

        if dt is None:
            dt = 0.1  # default 100ms

        # Update compass if available
        if compass is not None:
            heading_state = self._compass.update_compass(compass)
            self._sensor_quality.compass_quality = heading_state.confidence

        # Update INS if available
        if imu is not None:
            ins_state = self._ins.update(imu, dt=dt)
            # Feed INS velocity to Kalman
            self._kalman.update_ins_velocity(
                ins_state.velocity_north, ins_state.velocity_east
            )
            self._sensor_quality.ins_quality = 1.0

        # Update compass from gyro if compass not available but IMU is
        if compass is None and imu is not None:
            self._compass.update_gyro(imu.gyro_x, dt)
            self._sensor_quality.compass_quality = self._compass.confidence

        # Update GPS
        if gps is not None and gps.is_valid:
            self._handle_gps_fix(gps)
        else:
            # GPS unavailable (either None or invalid)
            if self._last_gps is not None or self._sensor_quality.gps_quality > 0.1:
                self._handle_gps_dropout(current_time)

        # Kalman prediction
        self._kalman.predict(dt)

        # Compute final estimate
        kalman_state = self._kalman.get_state()
        heading = self._compass.get_heading()
        speed = math.sqrt(
            kalman_state.vel_north ** 2 + kalman_state.vel_east ** 2
        )
        course = math.degrees(math.atan2(
            kalman_state.vel_east, kalman_state.vel_north
        )) % 360 if speed > 0.01 else heading

        # Update overall confidence
        self._update_confidence()

        self._update_count += 1

        estimate = PositionEstimate(
            latitude=kalman_state.lat,
            longitude=kalman_state.lon,
            heading=heading,
            speed=speed,
            course_over_ground=course,
            velocity_north=kalman_state.vel_north,
            velocity_east=kalman_state.vel_east,
            confidence=self._sensor_quality.overall_confidence,
            mode=self._mode,
            sensor_quality=self._sensor_quality,
            timestamp_ms=current_time,
        )
        self._last_estimate = estimate
        return estimate

    def _handle_gps_fix(self, gps: GPSReading) -> None:
        """Handle a valid GPS fix."""
        # Check if we're recovering from a dropout
        if self._gps_dropout_active:
            self._end_gps_dropout(gps.timestamp_ms)
            self._gps_dropout_active = False

        # Update Kalman with GPS
        self._kalman.update_gps(gps)

        # Correct INS with GPS position
        self._ins.correct_position(gps.latitude, gps.longitude)

        # Correct compass heading with GPS course
        if gps.speed > 0.5:
            self._compass.update_compass(CompassReading(
                heading_raw=gps.course,
                quality=0.7,
                timestamp_ms=gps.timestamp_ms,
            ))

        # Update GPS quality
        quality = min(1.0, gps.num_satellites / 12.0)
        quality *= min(1.0, 2.0 / max(gps.hdop, 0.5))
        self._sensor_quality.gps_quality = quality
        self._sensor_quality.gps_signal_quality = quality
        self._sensor_quality.last_gps_time_ms = gps.timestamp_ms
        self._last_gps = gps

        # Update mode
        self._update_mode()

    def _handle_gps_dropout(self, current_time_ms: int) -> None:
        """Handle GPS signal loss."""
        if not self._gps_dropout_active:
            self._gps_dropout_active = True
            self._gps_dropout_start_ms = current_time_ms
            if self._last_estimate:
                self._position_at_dropout = (
                    self._last_estimate.latitude,
                    self._last_estimate.longitude,
                )

        dropout_duration = current_time_ms - self._gps_dropout_start_ms
        self._sensor_quality.gps_quality = max(
            0.0, 1.0 - dropout_duration / self.gps_dropout_timeout_ms
        )
        self._sensor_quality.gps_dropout_count += 1
        self._sensor_quality.gps_dropout_duration_ms = dropout_duration

        self._update_mode()

    def _end_gps_dropout(self, end_time_ms: int) -> None:
        """End a GPS dropout and record it."""
        duration = end_time_ms - self._gps_dropout_start_ms
        drift = 0.0
        if self._last_estimate:
            drift = NavigationMath.haversine_distance(
                self._position_at_dropout[0], self._position_at_dropout[1],
                self._last_estimate.latitude, self._last_estimate.longitude,
            )

        event = GPSDropoutEvent(
            start_time_ms=self._gps_dropout_start_ms,
            end_time_ms=end_time_ms,
            duration_ms=duration,
            max_drift_m=drift,
        )
        self._gps_dropout_history.append(event)
        # Keep only last 100 events
        if len(self._gps_dropout_history) > 100:
            self._gps_dropout_history = self._gps_dropout_history[-50:]

    def _update_mode(self) -> None:
        """Determine navigation mode based on sensor availability."""
        gps_ok = self._sensor_quality.gps_quality > 0.1
        ins_ok = self._sensor_quality.ins_quality > 0.1
        compass_ok = self._sensor_quality.compass_quality > 0.1

        if gps_ok and ins_ok:
            self._mode = NavigationMode.FULL_FUSION
        elif not gps_ok and ins_ok:
            self._mode = NavigationMode.INS_ONLY
        elif gps_ok and not ins_ok:
            self._mode = NavigationMode.GPS_ONLY
        elif gps_ok:
            self._mode = NavigationMode.DEGRADED
        else:
            self._mode = NavigationMode.INS_ONLY  # INS-only is better than nothing

    def _update_confidence(self) -> None:
        """Update overall confidence based on sensor quality."""
        sq = self._sensor_quality
        if self._mode == NavigationMode.FULL_FUSION:
            sq.overall_confidence = (
                sq.gps_quality * 0.5
                + sq.ins_quality * 0.3
                + sq.compass_quality * 0.2
            )
        elif self._mode == NavigationMode.INS_ONLY:
            # Confidence decreases during GPS dropout
            dropout_factor = max(0.01, 1.0 - sq.gps_dropout_duration_ms / 60000.0)
            sq.overall_confidence = sq.ins_quality * sq.compass_quality * dropout_factor
        elif self._mode == NavigationMode.GPS_ONLY:
            sq.overall_confidence = sq.gps_quality * 0.7
        else:
            sq.overall_confidence = max(0.01, sq.gps_quality * 0.3 + sq.compass_quality * 0.2)

        sq.overall_confidence = max(0.01, min(1.0, sq.overall_confidence))

    def get_estimate(self) -> Optional[PositionEstimate]:
        """Return the last position estimate."""
        return self._last_estimate

    def get_gps_dropout_history(self) -> list[GPSDropoutEvent]:
        """Return GPS dropout event history."""
        return list(self._gps_dropout_history)

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def gps_dropout_active(self) -> bool:
        return self._gps_dropout_active
