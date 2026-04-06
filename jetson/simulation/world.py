"""World/space representation for NEXUS marine robotics simulation.

Provides Vector3 math, WorldObject entities, TerrainCell grids,
and a World class for managing simulation state, queries, and collisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Vector3:
    """3D vector with standard math operations."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def add(self, other: Vector3) -> Vector3:
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def sub(self, other: Vector3) -> Vector3:
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def scale(self, scalar: float) -> Vector3:
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalize(self) -> Vector3:
        m = self.magnitude()
        if m == 0.0:
            return Vector3(0.0, 0.0, 0.0)
        return Vector3(self.x / m, self.y / m, self.z / m)

    def dot(self, other: Vector3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3) -> Vector3:
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def distance_to(self, other: Vector3) -> float:
        return self.sub(other).magnitude()

    def __add__(self, other: object) -> Vector3:
        if isinstance(other, Vector3):
            return self.add(other)
        return NotImplemented

    def __sub__(self, other: object) -> Vector3:
        if isinstance(other, Vector3):
            return self.sub(other)
        return NotImplemented

    def __mul__(self, scalar: float) -> Vector3:
        return self.scale(scalar)

    def __rmul__(self, scalar: float) -> Vector3:
        return self.scale(scalar)

    def __repr__(self) -> str:
        return f"Vector3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"


@dataclass
class WorldObject:
    """An object in the simulation world."""

    id: str
    position: Vector3 = field(default_factory=Vector3)
    velocity: Vector3 = field(default_factory=Vector3)
    orientation: float = 0.0  # heading in radians
    shape: str = "sphere"  # sphere, box, cylinder
    properties: dict = field(default_factory=dict)


@dataclass
class TerrainCell:
    """A cell of terrain in the world grid."""

    position: Vector3 = field(default_factory=Vector3)
    elevation: float = 0.0
    type: str = "water"  # water, land, reef, dock


class World:
    """Simulation world managing objects, terrain, queries, and collisions."""

    def __init__(self, width: float = 1000.0, height: float = 1000.0) -> None:
        self.width = width
        self.height = height
        self._objects: Dict[str, WorldObject] = {}
        self._terrain: List[TerrainCell] = []
        self._time: float = 0.0

    def add_object(self, obj: WorldObject) -> None:
        self._objects[obj.id] = obj

    def remove_object(self, obj_id: str) -> bool:
        if obj_id in self._objects:
            del self._objects[obj_id]
            return True
        return False

    def get_object(self, obj_id: str) -> Optional[WorldObject]:
        return self._objects.get(obj_id)

    def get_objects_in_radius(self, center: Vector3, radius: float) -> List[WorldObject]:
        result = []
        for obj in self._objects.values():
            if obj.position.distance_to(center) <= radius:
                result.append(obj)
        return result

    def update(self, dt: float) -> None:
        self._time += dt
        for obj in self._objects.values():
            obj.position = obj.position.add(obj.velocity.scale(dt))

    def check_collision(self, obj_a: WorldObject, obj_b: WorldObject) -> bool:
        """Sphere-based collision detection using object radius property."""
        radius_a = obj_a.properties.get("radius", 1.0)
        radius_b = obj_b.properties.get("radius", 1.0)
        dist = obj_a.position.distance_to(obj_b.position)
        return dist < (radius_a + radius_b)

    def ray_cast(
        self,
        origin: Vector3,
        direction: Vector3,
        max_range: float = 1000.0,
    ) -> Optional[Vector3]:
        """Cast a ray and return the nearest intersection point with an object."""
        direction_norm = direction.normalize()
        if direction_norm.magnitude() == 0.0:
            return None
        closest_t = max_range
        closest_point = None
        for obj in self._objects.values():
            radius = obj.properties.get("radius", 1.0)
            oc = origin.sub(obj.position)
            a = direction_norm.dot(direction_norm)
            b = 2.0 * oc.dot(direction_norm)
            c = oc.dot(oc) - radius * radius
            discriminant = b * b - 4.0 * a * c
            if discriminant >= 0.0:
                sqrt_disc = math.sqrt(discriminant)
                t1 = (-b - sqrt_disc) / (2.0 * a)
                t2 = (-b + sqrt_disc) / (2.0 * a)
                t = t1 if t1 > 0 else t2
                if 0.0 < t < closest_t:
                    closest_t = t
                    closest_point = direction_norm.scale(t).add(origin)
        return closest_point

    def add_terrain(self, cell: TerrainCell) -> None:
        self._terrain.append(cell)

    def get_terrain_at(self, x: float, y: float) -> Optional[TerrainCell]:
        for cell in self._terrain:
            if cell.position.x == x and cell.position.y == y:
                return cell
        return None

    @property
    def time(self) -> float:
        return self._time

    @property
    def objects(self) -> Dict[str, WorldObject]:
        return dict(self._objects)
