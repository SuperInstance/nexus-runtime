"""Sensor simulation for NEXUS marine robotics.

Provides noisy sensor models for GPS, compass, LiDAR, and depth sensing
with configurable noise characteristics and confidence scoring.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .world import Vector3, World, WorldObject


@dataclass
class SensorConfig:
    """Configuration for a simulated sensor."""

    type: str = "generic"
    range: float = 100.0
    noise_stddev: float = 1.0
    update_rate: float = 10.0  # Hz
    fov: float = 360.0  # field of view in degrees


@dataclass
class SensorReading:
    """A reading from a simulated sensor."""

    timestamp: float = 0.0
    sensor_id: str = ""
    data: dict = field(default_factory=dict)
    confidence: float = 1.0


class SensorSimulator:
    """Simulates various marine sensors with configurable noise."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._sensors: Dict[str, SensorConfig] = {}
        self._time: float = 0.0

    def add_sensor(self, config: SensorConfig, sensor_id: str) -> None:
        self._sensors[sensor_id] = config

    def remove_sensor(self, sensor_id: str) -> bool:
        if sensor_id in self._sensors:
            del self._sensors[sensor_id]
            return True
        return False

    def get_sensor(self, sensor_id: str) -> Optional[SensorConfig]:
        return self._sensors.get(sensor_id)

    def _gaussian_noise(self, stddev: float) -> float:
        return self._rng.gauss(0.0, stddev)

    def _gaussian_vector(self, stddev: float) -> Tuple[float, float, float]:
        return (
            self._gaussian_noise(stddev),
            self._gaussian_noise(stddev),
            self._gaussian_noise(stddev),
        )

    def simulate_gps(
        self, position: Vector3, sensor_id: str = "gps"
    ) -> SensorReading:
        config = self._sensors.get(sensor_id, SensorConfig(type="gps", noise_stddev=2.0))
        nx, ny, nz = self._gaussian_vector(config.noise_stddev)
        noisy_pos = Vector3(position.x + nx, position.y + ny, position.z + nz)
        # Confidence decreases with noise magnitude
        noise_mag = math.sqrt(nx * nx + ny * ny + nz * nz)
        denom = config.noise_stddev * 3.0
        confidence = max(0.0, min(1.0, 1.0 - (noise_mag / denom if denom > 0 else 0.0)))
        return SensorReading(
            timestamp=self._time,
            sensor_id=sensor_id,
            data={
                "latitude": noisy_pos.x,
                "longitude": noisy_pos.y,
                "altitude": noisy_pos.z,
                "true_position": (position.x, position.y, position.z),
            },
            confidence=confidence,
        )

    def simulate_compass(
        self, heading: float, sensor_id: str = "compass"
    ) -> SensorReading:
        config = self._sensors.get(
            sensor_id, SensorConfig(type="compass", noise_stddev=2.0)
        )
        deviation = self._gaussian_noise(config.noise_stddev)
        noisy_heading = heading + math.radians(deviation)
        # Wrap to [-pi, pi]
        noisy_heading = math.atan2(math.sin(noisy_heading), math.cos(noisy_heading))
        abs_deviation = abs(deviation)
        denom = config.noise_stddev * 3.0
        confidence = max(0.0, min(1.0, 1.0 - (abs_deviation / denom if denom > 0 else 0.0)))
        return SensorReading(
            timestamp=self._time,
            sensor_id=sensor_id,
            data={
                "heading": noisy_heading,
                "heading_degrees": math.degrees(noisy_heading),
                "true_heading": heading,
                "deviation_degrees": deviation,
            },
            confidence=confidence,
        )

    def simulate_lidar(
        self,
        world: World,
        sensor_id: str,
        position: Vector3,
        direction: Vector3,
        fov: float = 90.0,
        max_range: float = 100.0,
        num_rays: int = 37,
    ) -> SensorReading:
        config = self._sensors.get(sensor_id, SensorConfig(type="lidar"))
        actual_range = min(max_range, config.range)
        fov_rad = math.radians(fov)
        dir_norm = direction.normalize()

        distances = []
        angles = []
        for i in range(num_rays):
            # Distribute rays across the FOV
            if num_rays > 1:
                angle = -fov_rad / 2.0 + fov_rad * i / (num_rays - 1)
            else:
                angle = 0.0

            # Rotate direction by angle (2D rotation in xz plane)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            ray_dir = Vector3(
                dir_norm.x * cos_a - dir_norm.z * sin_a,
                dir_norm.y,
                dir_norm.x * sin_a + dir_norm.z * cos_a,
            )

            hit = world.ray_cast(position, ray_dir, actual_range)
            if hit is not None:
                dist = position.distance_to(hit)
            else:
                dist = actual_range

            # Add noise
            noisy_dist = dist + self._gaussian_noise(config.noise_stddev * 0.1)
            noisy_dist = max(0.0, noisy_dist)
            distances.append(noisy_dist)
            angles.append(math.degrees(angle))

        avg_confidence = max(0.1, 1.0 - config.noise_stddev * 0.05)
        return SensorReading(
            timestamp=self._time,
            sensor_id=sensor_id,
            data={
                "distances": distances,
                "angles": angles,
                "num_rays": num_rays,
                "fov": fov,
                "max_range": actual_range,
            },
            confidence=avg_confidence,
        )

    def simulate_depth(
        self, position: Vector3, seabed_depth: float, sensor_id: str = "depth"
    ) -> SensorReading:
        config = self._sensors.get(sensor_id, SensorConfig(type="depth", noise_stddev=0.1))
        depth_below_surface = seabed_depth - position.z
        noisy_depth = depth_below_surface + self._gaussian_noise(config.noise_stddev)
        noisy_depth = max(0.0, noisy_depth)
        noise_abs = abs(noisy_depth - depth_below_surface)
        confidence = max(0.0, min(1.0, 1.0 - noise_abs / (config.noise_stddev * 3.0)))
        return SensorReading(
            timestamp=self._time,
            sensor_id=sensor_id,
            data={
                "depth": noisy_depth,
                "seabed_depth": seabed_depth,
                "surface_position_z": position.z,
                "true_depth": depth_below_surface,
            },
            confidence=confidence,
        )

    def simulate_speed(
        self,
        velocity: Vector3,
        sensor_id: str = "speed_log",
    ) -> SensorReading:
        config = self._sensors.get(
            sensor_id, SensorConfig(type="speed_log", noise_stddev=0.05)
        )
        speed = velocity.magnitude()
        noisy_speed = max(0.0, speed + self._gaussian_noise(config.noise_stddev))
        return SensorReading(
            timestamp=self._time,
            sensor_id=sensor_id,
            data={
                "speed": noisy_speed,
                "true_speed": speed,
                "velocity": (velocity.x, velocity.y, velocity.z),
            },
            confidence=0.95,
        )

    def update_time(self, dt: float) -> None:
        self._time += dt

    @property
    def time(self) -> float:
        return self._time

    @property
    def sensor_count(self) -> int:
        return len(self._sensors)
