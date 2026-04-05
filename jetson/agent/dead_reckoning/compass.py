"""NEXUS Dead Reckoning - Compass Heading Fusion.

Tilt-compensated compass heading with gyro integration
and drift correction from compass measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CompassReading:
    """Raw compass sensor reading."""

    heading_raw: float = 0.0     # degrees, uncorrected
    pitch: float = 0.0           # degrees (tilt forward/back)
    roll: float = 0.0            # degrees (tilt left/right)
    magnetic_declination: float = 0.0  # degrees (local declination)
    quality: float = 1.0         # 0-1 sensor quality
    timestamp_ms: int = 0


@dataclass
class HeadingState:
    """Current heading estimation state."""

    heading: float = 0.0         # degrees [0, 360)
    gyro_rate: float = 0.0       # deg/s
    compass_heading: float = 0.0
    confidence: float = 1.0
    timestamp_ms: int = 0
    gyro_only_samples: int = 0   # samples since last compass fix


class CompassFusion:
    """Compass + gyroscope heading fusion.

    Provides tilt-compensated compass heading and fuses it
    with gyro integration for smooth, drift-resistant heading.

    Algorithm:
    1. Apply tilt compensation to raw compass heading
    2. Apply magnetic declination correction
    3. Fuse with gyro integration using complementary filter
    4. Gyro tracks heading between compass updates
    5. Compass corrects gyro drift on each reading

    Usage:
        fusion = CompassFusion()
        fusion.initialize(heading=45.0)
        fusion.update_compass(compass_reading)
        fusion.update_gyro(gyro_yaw_rate, dt=0.1)
        heading = fusion.get_heading()
    """

    def __init__(
        self,
        gyro_weight: float = 0.95,
        compass_weight: float = 0.05,
        max_gyro_drift_rate: float = 2.0,  # deg/s max drift before warning
    ) -> None:
        self.gyro_weight = gyro_weight
        self.compass_weight = compass_weight
        self.max_gyro_drift_rate = max_gyro_drift_rate

        self._heading: float = 0.0
        self._initialized = False
        self._last_gyro_rate: float = 0.0
        self._last_compass_heading: float = 0.0
        self._compass_heading: float = 0.0
        self._confidence: float = 1.0
        self._gyro_only_samples: int = 0
        self._total_compass_updates: int = 0
        self._total_gyro_updates: int = 0

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self, heading: float) -> None:
        """Initialize the fusion with a known heading."""
        self._heading = heading % 360
        self._compass_heading = self._heading
        self._initialized = True
        self._confidence = 1.0
        self._gyro_only_samples = 0

    def reset(self) -> None:
        """Reset fusion state."""
        self._heading = 0.0
        self._initialized = False
        self._last_gyro_rate = 0.0
        self._compass_heading = 0.0
        self._confidence = 1.0
        self._gyro_only_samples = 0
        self._total_compass_updates = 0
        self._total_gyro_updates = 0

    @staticmethod
    def tilt_compensate(
        heading_raw: float, pitch: float, roll: float
    ) -> float:
        """Apply tilt compensation to compass heading.

        For a 2-axis compass (single magnetometer), uses simple
        pitch/roll correction. For full 3-axis compensation,
        the heading is adjusted based on tilt angles.

        Args:
            heading_raw: Raw compass heading in degrees.
            pitch: Pitch angle in degrees (positive = nose up).
            roll: Roll angle in degrees (positive = right wing down).

        Returns:
            Tilt-compensated heading in degrees.
        """
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)
        heading_rad = math.radians(heading_raw)

        # Simple tilt correction factor
        cos_pitch = math.cos(pitch_rad)
        cos_roll = math.cos(roll_rad)

        # Apply correction (approximation for small angles)
        if cos_pitch > 0.01 and cos_roll > 0.01:
            correction = 1.0 / (cos_pitch * cos_roll)
            correction = min(correction, 2.0)  # limit correction magnitude
            correction = max(correction, 0.5)
        else:
            correction = 1.0

        # The correction shifts the heading slightly based on tilt
        tilt_shift = math.degrees(math.atan2(
            math.sin(roll_rad) * math.sin(pitch_rad), cos_pitch
        ))

        compensated = heading_rad * correction + math.radians(tilt_shift)
        return compensated % (2 * math.pi) * (180.0 / math.pi) % 360

    @staticmethod
    def apply_declination(heading: float, declination: float) -> float:
        """Apply magnetic declination to convert magnetic to true heading.

        Args:
            heading: Magnetic heading in degrees.
            declination: Magnetic declination in degrees (positive = east).

        Returns:
            True heading in degrees.
        """
        return (heading + declination) % 360

    def update_compass(self, reading: CompassReading) -> HeadingState:
        """Update heading with compass reading.

        Applies tilt compensation, declination correction,
        then fuses with current gyro-based heading.

        Args:
            reading: Compass reading with raw heading and tilt.

        Returns:
            Current heading state.
        """
        if not self._initialized:
            self.initialize(reading.heading_raw)

        # Tilt compensate
        compensated = self.tilt_compensate(
            reading.heading_raw, reading.pitch, reading.roll
        )

        # Apply declination
        true_heading = self.apply_declination(
            compensated, reading.magnetic_declination
        )

        # Fuse with current heading using compass weight
        # Higher compass quality = more weight on compass
        effective_compass_weight = self.compass_weight * reading.quality
        effective_gyro_weight = 1.0 - effective_compass_weight

        # Handle wrap-around for blending
        diff = self._angular_diff(true_heading, self._heading)
        fused = (self._heading + effective_compass_weight * diff) % 360

        self._heading = fused
        self._compass_heading = true_heading
        self._last_compass_heading = true_heading
        self._confidence = max(self._confidence * 0.9, reading.quality * 0.5)
        self._confidence = min(1.0, self._confidence + 0.1)
        self._gyro_only_samples = 0
        self._total_compass_updates += 1

        return self.get_state()

    def update_gyro(self, gyro_yaw_rate: float, dt: float) -> HeadingState:
        """Update heading with gyroscope yaw rate.

        Integrates gyro to track heading between compass updates.
        Gyro drift accumulates and is corrected when compass updates arrive.

        Args:
            gyro_yaw_rate: Yaw rate in degrees/second.
            dt: Time step in seconds.

        Returns:
            Current heading state.
        """
        if not self._initialized:
            return self.get_state()

        dt = max(dt, 0.001)
        self._last_gyro_rate = gyro_yaw_rate
        self._heading = (self._heading + gyro_yaw_rate * dt) % 360
        self._gyro_only_samples += 1
        self._total_gyro_updates += 1

        # Reduce confidence during gyro-only mode
        if self._gyro_only_samples > 10:
            drift_factor = min(1.0, self._gyro_only_samples / 100.0)
            self._confidence = max(0.01, self._confidence - drift_factor * 0.01)

        return self.get_state()

    def get_heading(self) -> float:
        """Return current fused heading in degrees [0, 360)."""
        return self._heading

    def get_state(self) -> HeadingState:
        """Return full heading state."""
        return HeadingState(
            heading=self._heading,
            gyro_rate=self._last_gyro_rate,
            compass_heading=self._compass_heading,
            confidence=self._confidence,
            gyro_only_samples=self._gyro_only_samples,
        )

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def gyro_only_samples(self) -> int:
        return self._gyro_only_samples

    @staticmethod
    def _angular_diff(a: float, b: float) -> float:
        """Smallest signed angle from b to a in degrees (-180, 180]."""
        diff = (a - b) % 360
        if diff > 180:
            diff -= 360
        return diff
