"""Tests for world.py — Vector3, WorldObject, TerrainCell, World."""

import math
import pytest

from jetson.simulation.world import Vector3, WorldObject, TerrainCell, World


# ---------- Vector3 ----------

class TestVector3Creation:
    def test_default_creation(self):
        v = Vector3()
        assert v.x == 0.0 and v.y == 0.0 and v.z == 0.0

    def test_param_creation(self):
        v = Vector3(1.0, 2.0, 3.0)
        assert v.x == 1.0 and v.y == 2.0 and v.z == 3.0

    def test_negative_creation(self):
        v = Vector3(-1.0, -2.0, -3.0)
        assert v.x == -1.0 and v.y == -2.0 and v.z == -3.0

    def test_float_creation(self):
        v = Vector3(0.5, 1.5, 2.5)
        assert v.x == 0.5


class TestVector3Add:
    def test_add_basic(self):
        a = Vector3(1, 2, 3)
        b = Vector3(4, 5, 6)
        r = a.add(b)
        assert r.x == 5.0 and r.y == 7.0 and r.z == 9.0

    def test_add_negative(self):
        a = Vector3(1, 2, 3)
        b = Vector3(-1, -2, -3)
        r = a.add(b)
        assert r.x == 0.0 and r.y == 0.0 and r.z == 0.0

    def test_add_zero(self):
        a = Vector3(1, 2, 3)
        b = Vector3(0, 0, 0)
        r = a.add(b)
        assert r.x == 1.0 and r.y == 2.0 and r.z == 3.0

    def test_operator_add(self):
        a = Vector3(1, 2, 3)
        b = Vector3(4, 5, 6)
        r = a + b
        assert r.x == 5.0 and r.y == 7.0 and r.z == 9.0

    def test_add_does_not_mutate(self):
        a = Vector3(1, 2, 3)
        b = Vector3(4, 5, 6)
        a.add(b)
        assert a.x == 1.0


class TestVector3Sub:
    def test_sub_basic(self):
        a = Vector3(5, 7, 9)
        b = Vector3(1, 2, 3)
        r = a.sub(b)
        assert r.x == 4.0 and r.y == 5.0 and r.z == 6.0

    def test_sub_self(self):
        a = Vector3(1, 2, 3)
        r = a.sub(a)
        assert r.x == 0.0 and r.y == 0.0 and r.z == 0.0

    def test_operator_sub(self):
        a = Vector3(5, 7, 9)
        b = Vector3(1, 2, 3)
        r = a - b
        assert r.x == 4.0 and r.y == 5.0 and r.z == 6.0


class TestVector3Scale:
    def test_scale_positive(self):
        v = Vector3(1, 2, 3)
        r = v.scale(2.0)
        assert r.x == 2.0 and r.y == 4.0 and r.z == 6.0

    def test_scale_zero(self):
        v = Vector3(1, 2, 3)
        r = v.scale(0.0)
        assert r.x == 0.0 and r.y == 0.0 and r.z == 0.0

    def test_scale_negative(self):
        v = Vector3(1, 2, 3)
        r = v.scale(-1.0)
        assert r.x == -1.0 and r.y == -2.0 and r.z == -3.0

    def test_operator_mul(self):
        v = Vector3(1, 2, 3)
        r = v * 3.0
        assert r.x == 3.0 and r.y == 6.0 and r.z == 9.0

    def test_operator_rmul(self):
        v = Vector3(1, 2, 3)
        r = 3.0 * v
        assert r.x == 3.0 and r.y == 6.0 and r.z == 9.0


class TestVector3Magnitude:
    def test_magnitude_unit_x(self):
        v = Vector3(1, 0, 0)
        assert v.magnitude() == pytest.approx(1.0)

    def test_magnitude_unit_y(self):
        v = Vector3(0, 1, 0)
        assert v.magnitude() == pytest.approx(1.0)

    def test_magnitude_345(self):
        v = Vector3(3, 4, 0)
        assert v.magnitude() == pytest.approx(5.0)

    def test_magnitude_zero(self):
        v = Vector3(0, 0, 0)
        assert v.magnitude() == pytest.approx(0.0)


class TestVector3Normalize:
    def test_normalize_unit(self):
        v = Vector3(1, 0, 0)
        n = v.normalize()
        assert n.x == pytest.approx(1.0) and n.y == pytest.approx(0.0)

    def test_normalize_arbitrary(self):
        v = Vector3(3, 4, 0)
        n = v.normalize()
        assert n.x == pytest.approx(0.6) and n.y == pytest.approx(0.8)

    def test_normalize_zero(self):
        v = Vector3(0, 0, 0)
        n = v.normalize()
        assert n.x == 0.0 and n.y == 0.0 and n.z == 0.0


class TestVector3Dot:
    def test_dot_parallel(self):
        a = Vector3(1, 0, 0)
        b = Vector3(2, 0, 0)
        assert a.dot(b) == pytest.approx(2.0)

    def test_dot_perpendicular(self):
        a = Vector3(1, 0, 0)
        b = Vector3(0, 1, 0)
        assert a.dot(b) == pytest.approx(0.0)

    def test_dot_general(self):
        a = Vector3(1, 2, 3)
        b = Vector3(4, 5, 6)
        assert a.dot(b) == pytest.approx(32.0)


class TestVector3Cross:
    def test_cross_unit_x_y(self):
        a = Vector3(1, 0, 0)
        b = Vector3(0, 1, 0)
        c = a.cross(b)
        assert c.x == pytest.approx(0.0) and c.y == pytest.approx(0.0) and c.z == pytest.approx(1.0)

    def test_cross_anti_commutative(self):
        a = Vector3(1, 0, 0)
        b = Vector3(0, 1, 0)
        c1 = a.cross(b)
        c2 = b.cross(a)
        assert c1.x == pytest.approx(-c2.x)
        assert c1.y == pytest.approx(-c2.y)
        assert c1.z == pytest.approx(-c2.z)

    def test_cross_parallel(self):
        a = Vector3(1, 2, 3)
        c = a.cross(a)
        assert c.x == pytest.approx(0.0)


class TestVector3DistanceTo:
    def test_distance_same_point(self):
        a = Vector3(1, 2, 3)
        assert a.distance_to(a) == pytest.approx(0.0)

    def test_distance_unit(self):
        a = Vector3(0, 0, 0)
        b = Vector3(1, 0, 0)
        assert a.distance_to(b) == pytest.approx(1.0)


class TestVector3Repr:
    def test_repr_contains_coords(self):
        v = Vector3(1.5, 2.5, 3.5)
        r = repr(v)
        assert "1.5" in r and "2.5" in r and "3.5" in r


# ---------- WorldObject ----------

class TestWorldObject:
    def test_default_creation(self):
        obj = WorldObject(id="test")
        assert obj.id == "test"
        assert obj.orientation == 0.0
        assert obj.shape == "sphere"
        assert obj.properties == {}

    def test_custom_creation(self):
        pos = Vector3(1, 2, 3)
        vel = Vector3(4, 5, 6)
        obj = WorldObject(id="test", position=pos, velocity=vel, orientation=1.5, shape="cylinder", properties={"radius": 5})
        assert obj.position.x == 1.0
        assert obj.velocity.y == 5.0
        assert obj.orientation == 1.5
        assert obj.shape == "cylinder"
        assert obj.properties["radius"] == 5


# ---------- TerrainCell ----------

class TestTerrainCell:
    def test_default_creation(self):
        cell = TerrainCell()
        assert cell.elevation == 0.0
        assert cell.type == "water"

    def test_land_cell(self):
        cell = TerrainCell(position=Vector3(10, 20, 0), elevation=5.0, type="land")
        assert cell.type == "land"
        assert cell.elevation == 5.0

    def test_reef_type(self):
        cell = TerrainCell(type="reef")
        assert cell.type == "reef"

    def test_dock_type(self):
        cell = TerrainCell(type="dock")
        assert cell.type == "dock"


# ---------- World ----------

class TestWorldCreation:
    def test_default_size(self):
        w = World()
        assert w.width == 1000.0 and w.height == 1000.0

    def test_custom_size(self):
        w = World(500.0, 600.0)
        assert w.width == 500.0 and w.height == 600.0


class TestWorldObjects:
    def test_add_object(self):
        w = World()
        obj = WorldObject(id="v1")
        w.add_object(obj)
        assert w.get_object("v1") is not None

    def test_add_multiple(self):
        w = World()
        w.add_object(WorldObject(id="a"))
        w.add_object(WorldObject(id="b"))
        assert w.get_object("a") is not None
        assert w.get_object("b") is not None

    def test_remove_object(self):
        w = World()
        w.add_object(WorldObject(id="v1"))
        assert w.remove_object("v1") is True
        assert w.get_object("v1") is None

    def test_remove_nonexistent(self):
        w = World()
        assert w.remove_object("nope") is False

    def test_get_nonexistent(self):
        w = World()
        assert w.get_object("nope") is None


class TestWorldQuery:
    def test_objects_in_radius(self):
        w = World()
        w.add_object(WorldObject(id="c1", position=Vector3(0, 0, 0)))
        w.add_object(WorldObject(id="c2", position=Vector3(5, 0, 0)))
        w.add_object(WorldObject(id="c3", position=Vector3(100, 0, 0)))
        result = w.get_objects_in_radius(Vector3(0, 0, 0), 10.0)
        ids = [o.id for o in result]
        assert "c1" in ids
        assert "c2" in ids
        assert "c3" not in ids

    def test_objects_in_radius_empty(self):
        w = World()
        result = w.get_objects_in_radius(Vector3(0, 0, 0), 10.0)
        assert result == []

    def test_objects_in_radius_all(self):
        w = World()
        w.add_object(WorldObject(id="a", position=Vector3(0, 0, 0)))
        result = w.get_objects_in_radius(Vector3(0, 0, 0), 1000.0)
        assert len(result) == 1


class TestWorldUpdate:
    def test_update_advances_time(self):
        w = World()
        w.update(0.1)
        assert w.time == pytest.approx(0.1)

    def test_update_multiple(self):
        w = World()
        w.update(0.1)
        w.update(0.2)
        assert w.time == pytest.approx(0.3)

    def test_update_moves_objects(self):
        w = World()
        obj = WorldObject(id="v1", position=Vector3(0, 0, 0), velocity=Vector3(10, 0, 0))
        w.add_object(obj)
        w.update(1.0)
        assert w.get_object("v1").position.x == pytest.approx(10.0)

    def test_update_zero_dt(self):
        w = World()
        obj = WorldObject(id="v1", position=Vector3(5, 0, 0), velocity=Vector3(10, 0, 0))
        w.add_object(obj)
        w.update(0.0)
        assert w.get_object("v1").position.x == pytest.approx(5.0)


class TestWorldCollision:
    def test_collision_overlapping(self):
        w = World()
        a = WorldObject(id="a", position=Vector3(0, 0, 0), properties={"radius": 5})
        b = WorldObject(id="b", position=Vector3(8, 0, 0), properties={"radius": 5})
        assert w.check_collision(a, b) is True

    def test_collision_non_overlapping(self):
        w = World()
        a = WorldObject(id="a", position=Vector3(0, 0, 0), properties={"radius": 5})
        b = WorldObject(id="b", position=Vector3(20, 0, 0), properties={"radius": 5})
        assert w.check_collision(a, b) is False

    def test_collision_touching(self):
        w = World()
        a = WorldObject(id="a", position=Vector3(0, 0, 0), properties={"radius": 5})
        b = WorldObject(id="b", position=Vector3(10, 0, 0), properties={"radius": 5})
        assert w.check_collision(a, b) is False  # dist == r1+r2, not <

    def test_collision_default_radius(self):
        w = World()
        a = WorldObject(id="a", position=Vector3(0, 0, 0))
        b = WorldObject(id="b", position=Vector3(1.5, 0, 0))
        assert w.check_collision(a, b) is True  # default radius=1.0 each


class TestWorldRayCast:
    def test_ray_hit(self):
        w = World()
        w.add_object(WorldObject(id="target", position=Vector3(10, 0, 0), properties={"radius": 1}))
        hit = w.ray_cast(Vector3(0, 0, 0), Vector3(1, 0, 0), 100)
        assert hit is not None
        assert hit.x == pytest.approx(9.0)

    def test_ray_miss(self):
        w = World()
        w.add_object(WorldObject(id="target", position=Vector3(10, 10, 0), properties={"radius": 1}))
        hit = w.ray_cast(Vector3(0, 0, 0), Vector3(1, 0, 0), 100)
        assert hit is None

    def test_ray_empty_world(self):
        w = World()
        hit = w.ray_cast(Vector3(0, 0, 0), Vector3(1, 0, 0), 100)
        assert hit is None

    def test_ray_zero_direction(self):
        w = World()
        w.add_object(WorldObject(id="t", position=Vector3(5, 0, 0)))
        hit = w.ray_cast(Vector3(0, 0, 0), Vector3(0, 0, 0), 100)
        assert hit is None

    def test_ray_max_range(self):
        w = World()
        w.add_object(WorldObject(id="far", position=Vector3(200, 0, 0), properties={"radius": 1}))
        hit = w.ray_cast(Vector3(0, 0, 0), Vector3(1, 0, 0), 50)
        assert hit is None


class TestWorldTerrain:
    def test_add_and_get_terrain(self):
        w = World()
        cell = TerrainCell(position=Vector3(10, 20, 0), elevation=5.0, type="land")
        w.add_terrain(cell)
        found = w.get_terrain_at(10, 20)
        assert found is not None
        assert found.type == "land"

    def test_get_missing_terrain(self):
        w = World()
        assert w.get_terrain_at(0, 0) is None

    def test_world_objects_property(self):
        w = World()
        w.add_object(WorldObject(id="a"))
        objs = w.objects
        assert "a" in objs
