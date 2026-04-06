"""Scenario construction for NEXUS marine robotics simulation.

Provides a builder pattern for creating complex simulation scenarios
including traffic, emergency, and patrol patterns.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .world import Vector3, WorldObject, World, TerrainCell


@dataclass
class ScenarioObject:
    """An object specification for a scenario."""

    type: str = "vessel"
    position: Vector3 = field(default_factory=Vector3)
    velocity: Vector3 = field(default_factory=Vector3)
    behavior: str = "stationary"  # stationary, moving, patrol, evasive


@dataclass
class ScenarioConfig:
    """Configuration for a simulation scenario."""

    name: str = "default"
    world_size: float = 1000.0
    objects: List[ScenarioObject] = field(default_factory=list)
    duration: float = 60.0
    metadata: dict = field(default_factory=dict)


class ScenarioBuilder:
    """Builds simulation scenarios from configurations."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def create_world(self, config: ScenarioConfig) -> World:
        """Create a World from a ScenarioConfig."""
        world = World(width=config.world_size, height=config.world_size)
        for i, spec in enumerate(config.objects):
            obj = WorldObject(
                id=f"{spec.type}_{i}",
                position=Vector3(spec.position.x, spec.position.y, spec.position.z),
                velocity=Vector3(spec.velocity.x, spec.velocity.y, spec.velocity.z),
                shape=self._shape_for_type(spec.type),
                properties={"behavior": spec.behavior, "scenario_type": spec.type},
            )
            if spec.type == "vessel":
                obj.properties["radius"] = 5.0
            elif spec.type == "obstacle":
                obj.properties["radius"] = 3.0
            elif spec.type == "buoy":
                obj.properties["radius"] = 1.0
            else:
                obj.properties["radius"] = 2.0
            world.add_object(obj)
        return world

    def _shape_for_type(self, obj_type: str) -> str:
        shapes = {
            "vessel": "cylinder",
            "obstacle": "sphere",
            "buoy": "sphere",
            "dock": "box",
            "waypoint": "sphere",
            "submarine": "cylinder",
        }
        return shapes.get(obj_type, "sphere")

    def add_vessel(self, config: ScenarioConfig, position: Vector3 = None,
                   velocity: Vector3 = None) -> ScenarioConfig:
        """Add a vessel to a scenario config."""
        size = config.world_size
        pos = position or Vector3(
            self._rng.uniform(-size / 2, size / 2),
            self._rng.uniform(-size / 2, size / 2),
            0.0,
        )
        vel = velocity or Vector3(self._rng.uniform(-2, 2), self._rng.uniform(-2, 2), 0.0)
        obj = ScenarioObject(type="vessel", position=pos, velocity=vel, behavior="moving")
        config.objects.append(obj)
        return config

    def add_obstacle(self, config: ScenarioConfig, position: Vector3,
                     size: float = 3.0) -> ScenarioConfig:
        """Add an obstacle to a scenario config."""
        obj = ScenarioObject(
            type="obstacle",
            position=position,
            velocity=Vector3(0.0, 0.0, 0.0),
            behavior="stationary",
        )
        obj.properties = {"radius": size}
        config.objects.append(obj)
        return config

    def create_traffic_scenario(self, density: int = 10,
                                world_size: float = 1000.0) -> ScenarioConfig:
        """Create a maritime traffic scenario with given vessel density."""
        config = ScenarioConfig(
            name="traffic_scenario",
            world_size=world_size,
            duration=120.0,
        )
        half = world_size / 2.0
        for i in range(density):
            pos = Vector3(
                self._rng.uniform(-half, half),
                self._rng.uniform(-half, half),
                0.0,
            )
            speed = self._rng.uniform(0.5, 5.0)
            heading = self._rng.uniform(0, 2 * math.pi)
            vel = Vector3(
                speed * math.cos(heading),
                speed * math.sin(heading),
                0.0,
            )
            behavior = self._rng.choice(["moving", "stationary", "moving"])
            obj = ScenarioObject(type="vessel", position=pos, velocity=vel, behavior=behavior)
            config.objects.append(obj)

        # Add some buoys
        for i in range(density // 5 + 1):
            pos = Vector3(
                self._rng.uniform(-half, half),
                self._rng.uniform(-half, half),
                0.0,
            )
            obj = ScenarioObject(type="buoy", position=pos, velocity=Vector3(), behavior="stationary")
            config.objects.append(obj)

        return config

    def create_emergency_scenario(self, emergency_type: str = "collision",
                                   world_size: float = 1000.0) -> ScenarioConfig:
        """Create an emergency scenario."""
        config = ScenarioConfig(
            name=f"emergency_{emergency_type}",
            world_size=world_size,
            duration=60.0,
        )

        if emergency_type == "collision":
            # Two vessels on collision course
            v1_pos = Vector3(-100.0, 0.0, 0.0)
            v1_vel = Vector3(5.0, 0.0, 0.0)
            v2_pos = Vector3(100.0, 0.0, 0.0)
            v2_vel = Vector3(-5.0, 0.0, 0.0)
            config.objects.append(ScenarioObject(type="vessel", position=v1_pos, velocity=v1_vel, behavior="moving"))
            config.objects.append(ScenarioObject(type="vessel", position=v2_pos, velocity=v2_vel, behavior="moving"))

        elif emergency_type == "man_overboard":
            # Stationary person in water, vessel nearby
            config.objects.append(ScenarioObject(
                type="vessel", position=Vector3(0, 0, 0),
                velocity=Vector3(0, 0, 0), behavior="stationary"
            ))
            config.objects.append(ScenarioObject(
                type="obstacle", position=Vector3(50, 20, 0),
                velocity=Vector3(0, -0.5, 0), behavior="drifting"
            ))

        elif emergency_type == "fire":
            # Vessel with fire
            config.objects.append(ScenarioObject(
                type="vessel", position=Vector3(0, 0, 0),
                velocity=Vector3(0, 0, 0), behavior="stationary"
            ))
            config.objects.append(ScenarioObject(
                type="obstacle", position=Vector3(30, 0, 0),
                velocity=Vector3(0, 0, 0), behavior="stationary"
            ))

        elif emergency_type == "grounding":
            # Vessel near land/reef
            config.objects.append(ScenarioObject(
                type="vessel", position=Vector3(50, 50, 0),
                velocity=Vector3(1, 1, 0), behavior="moving"
            ))
            config.objects.append(ScenarioObject(
                type="obstacle", position=Vector3(80, 80, 0),
                velocity=Vector3(0, 0, 0), behavior="stationary"
            ))

        else:
            # Generic emergency
            config.objects.append(ScenarioObject(
                type="vessel", position=Vector3(0, 0, 0),
                velocity=Vector3(0, 0, 0), behavior="stationary"
            ))

        # Add a dock
        config.objects.append(ScenarioObject(
            type="dock", position=Vector3(-200, -200, 0),
            velocity=Vector3(0, 0, 0), behavior="stationary"
        ))

        return config

    def create_patrol_scenario(self, waypoints: List[Vector3] = None,
                                world_size: float = 1000.0) -> ScenarioConfig:
        """Create a patrol scenario with waypoints."""
        if waypoints is None:
            waypoints = [
                Vector3(-100, -100, 0),
                Vector3(100, -100, 0),
                Vector3(100, 100, 0),
                Vector3(-100, 100, 0),
            ]

        config = ScenarioConfig(
            name="patrol_scenario",
            world_size=world_size,
            duration=300.0,
            metadata={"waypoints": [(w.x, w.y, w.z) for w in waypoints]},
        )

        # Patrol vessel
        if waypoints:
            start = waypoints[0]
            direction = waypoints[1].sub(start).normalize() if len(waypoints) > 1 else Vector3(1, 0, 0)
        else:
            start = Vector3(0, 0, 0)
            direction = Vector3(1, 0, 0)

        config.objects.append(ScenarioObject(
            type="vessel",
            position=start,
            velocity=direction.scale(3.0),
            behavior="patrol",
        ))

        # Add some targets to patrol around
        for i in range(3):
            pos = Vector3(
                self._rng.uniform(-world_size / 3, world_size / 3),
                self._rng.uniform(-world_size / 3, world_size / 3),
                0.0,
            )
            config.objects.append(ScenarioObject(
                type="obstacle", position=pos,
                velocity=Vector3(0, 0, 0), behavior="stationary"
            ))

        return config

    def export_scenario(self, config: ScenarioConfig) -> dict:
        """Export a scenario config to a serializable dict."""
        objects_data = []
        for obj in config.objects:
            objects_data.append({
                "type": obj.type,
                "position": {
                    "x": obj.position.x,
                    "y": obj.position.y,
                    "z": obj.position.z,
                },
                "velocity": {
                    "x": obj.velocity.x,
                    "y": obj.velocity.y,
                    "z": obj.velocity.z,
                },
                "behavior": obj.behavior,
            })
        return {
            "name": config.name,
            "world_size": config.world_size,
            "objects": objects_data,
            "duration": config.duration,
            "metadata": config.metadata,
        }

    def import_scenario(self, data: dict) -> ScenarioConfig:
        """Import a scenario from a dict."""
        objects = []
        for obj_data in data.get("objects", []):
            pos_data = obj_data.get("position", {})
            vel_data = obj_data.get("velocity", {})
            objects.append(ScenarioObject(
                type=obj_data.get("type", "vessel"),
                position=Vector3(
                    pos_data.get("x", 0.0),
                    pos_data.get("y", 0.0),
                    pos_data.get("z", 0.0),
                ),
                velocity=Vector3(
                    vel_data.get("x", 0.0),
                    vel_data.get("y", 0.0),
                    vel_data.get("z", 0.0),
                ),
                behavior=obj_data.get("behavior", "stationary"),
            ))
        return ScenarioConfig(
            name=data.get("name", "imported"),
            world_size=data.get("world_size", 1000.0),
            objects=objects,
            duration=data.get("duration", 60.0),
            metadata=data.get("metadata", {}),
        )
