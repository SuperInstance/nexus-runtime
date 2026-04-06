"""Tests for scenario_builder.py — ScenarioObject, ScenarioConfig, ScenarioBuilder."""

import pytest

from jetson.simulation.world import Vector3, World
from jetson.simulation.scenario_builder import ScenarioObject, ScenarioConfig, ScenarioBuilder


class TestScenarioObject:
    def test_default_creation(self):
        obj = ScenarioObject()
        assert obj.type == "vessel"
        assert obj.behavior == "stationary"
        assert obj.position.x == 0.0

    def test_custom_creation(self):
        obj = ScenarioObject(type="obstacle", position=Vector3(1, 2, 3), velocity=Vector3(4, 5, 6), behavior="moving")
        assert obj.type == "obstacle"
        assert obj.behavior == "moving"
        assert obj.position.x == 1.0


class TestScenarioConfig:
    def test_default_creation(self):
        cfg = ScenarioConfig()
        assert cfg.name == "default"
        assert cfg.world_size == 1000.0
        assert cfg.duration == 60.0
        assert cfg.objects == []

    def test_custom_creation(self):
        cfg = ScenarioConfig(name="test", world_size=500, duration=30.0)
        assert cfg.name == "test"
        assert cfg.world_size == 500.0


class TestScenarioBuilder:
    def test_default_creation(self):
        builder = ScenarioBuilder()
        assert builder is not None

    def test_seeded_creation(self):
        builder = ScenarioBuilder(seed=42)
        assert builder is not None


class TestCreateWorld:
    def test_create_empty_world(self):
        builder = ScenarioBuilder()
        cfg = ScenarioConfig(name="empty")
        world = builder.create_world(cfg)
        assert isinstance(world, World)
        assert len(world.objects) == 0

    def test_create_world_with_objects(self):
        builder = ScenarioBuilder()
        cfg = ScenarioConfig(name="test")
        cfg.objects.append(ScenarioObject(type="vessel", position=Vector3(10, 20, 0)))
        cfg.objects.append(ScenarioObject(type="buoy", position=Vector3(30, 40, 0)))
        world = builder.create_world(cfg)
        assert len(world.objects) == 2

    def test_world_has_correct_size(self):
        builder = ScenarioBuilder()
        cfg = ScenarioConfig(world_size=500)
        world = builder.create_world(cfg)
        assert world.width == 500.0


class TestAddVessel:
    def test_add_vessel(self):
        builder = ScenarioBuilder(seed=42)
        cfg = ScenarioConfig()
        builder.add_vessel(cfg)
        assert len(cfg.objects) == 1

    def test_add_vessel_with_position(self):
        builder = ScenarioBuilder()
        cfg = ScenarioConfig()
        builder.add_vessel(cfg, position=Vector3(100, 200, 0))
        assert len(cfg.objects) == 1
        assert cfg.objects[0].position.x == 100.0

    def test_add_multiple_vessels(self):
        builder = ScenarioBuilder(seed=42)
        cfg = ScenarioConfig()
        builder.add_vessel(cfg)
        builder.add_vessel(cfg)
        assert len(cfg.objects) == 2


class TestAddObstacle:
    def test_add_obstacle(self):
        builder = ScenarioBuilder()
        cfg = ScenarioConfig()
        builder.add_obstacle(cfg, Vector3(50, 50, 0), size=5.0)
        assert len(cfg.objects) == 1
        assert cfg.objects[0].type == "obstacle"
        assert cfg.objects[0].behavior == "stationary"

    def test_add_obstacle_size(self):
        builder = ScenarioBuilder()
        cfg = ScenarioConfig()
        builder.add_obstacle(cfg, Vector3(10, 20, 0), size=10.0)
        assert cfg.objects[0].properties.get("radius") == 10.0


class TestTrafficScenario:
    def test_create_traffic_default(self):
        builder = ScenarioBuilder(seed=42)
        cfg = builder.create_traffic_scenario()
        assert cfg.name == "traffic_scenario"
        assert cfg.duration == 120.0
        assert len(cfg.objects) > 0

    def test_traffic_density(self):
        builder = ScenarioBuilder(seed=42)
        cfg = builder.create_traffic_scenario(density=5)
        # Should have 5 vessels + at least 1 buoy
        vessel_count = sum(1 for o in cfg.objects if o.type == "vessel")
        assert vessel_count == 5

    def test_traffic_has_buoys(self):
        builder = ScenarioBuilder(seed=42)
        cfg = builder.create_traffic_scenario(density=20)
        buoy_count = sum(1 for o in cfg.objects if o.type == "buoy")
        assert buoy_count >= 1

    def test_traffic_custom_world_size(self):
        builder = ScenarioBuilder(seed=42)
        cfg = builder.create_traffic_scenario(world_size=500)
        assert cfg.world_size == 500.0

    def test_traffic_vessels_moving(self):
        builder = ScenarioBuilder(seed=42)
        cfg = builder.create_traffic_scenario(density=10)
        # At least some should be moving
        moving = sum(1 for o in cfg.objects if o.behavior == "moving" and o.type == "vessel")
        assert moving >= 1


class TestEmergencyScenario:
    def test_collision_scenario(self):
        builder = ScenarioBuilder()
        cfg = builder.create_emergency_scenario("collision")
        assert "collision" in cfg.name
        vessels = [o for o in cfg.objects if o.type == "vessel"]
        assert len(vessels) == 2

    def test_man_overboard_scenario(self):
        builder = ScenarioBuilder()
        cfg = builder.create_emergency_scenario("man_overboard")
        assert "man_overboard" in cfg.name
        assert len(cfg.objects) >= 2

    def test_fire_scenario(self):
        builder = ScenarioBuilder()
        cfg = builder.create_emergency_scenario("fire")
        assert "fire" in cfg.name

    def test_grounding_scenario(self):
        builder = ScenarioBuilder()
        cfg = builder.create_emergency_scenario("grounding")
        assert "grounding" in cfg.name

    def test_unknown_emergency_type(self):
        builder = ScenarioBuilder()
        cfg = builder.create_emergency_scenario("alien_invasion")
        # Should still work with default
        assert len(cfg.objects) >= 1

    def test_emergency_has_dock(self):
        builder = ScenarioBuilder()
        cfg = builder.create_emergency_scenario("collision")
        docks = [o for o in cfg.objects if o.type == "dock"]
        assert len(docks) >= 1


class TestPatrolScenario:
    def test_default_patrol(self):
        builder = ScenarioBuilder()
        cfg = builder.create_patrol_scenario()
        assert cfg.name == "patrol_scenario"
        assert cfg.duration == 300.0
        assert len(cfg.objects) >= 1

    def test_patrol_with_waypoints(self):
        builder = ScenarioBuilder()
        wps = [Vector3(i * 100, 0, 0) for i in range(5)]
        cfg = builder.create_patrol_scenario(waypoints=wps)
        assert "waypoints" in cfg.metadata
        assert len(cfg.metadata["waypoints"]) == 5

    def test_patrol_vessel_behavior(self):
        builder = ScenarioBuilder()
        cfg = builder.create_patrol_scenario()
        patrol_vessels = [o for o in cfg.objects if o.type == "vessel"]
        assert len(patrol_vessels) == 1
        assert patrol_vessels[0].behavior == "patrol"

    def test_patrol_custom_world_size(self):
        builder = ScenarioBuilder()
        cfg = builder.create_patrol_scenario(world_size=2000)
        assert cfg.world_size == 2000.0


class TestExportImport:
    def test_export_empty_scenario(self):
        builder = ScenarioBuilder()
        cfg = ScenarioConfig(name="empty")
        data = builder.export_scenario(cfg)
        assert data["name"] == "empty"
        assert data["objects"] == []

    def test_export_with_objects(self):
        builder = ScenarioBuilder()
        cfg = ScenarioConfig(name="test")
        cfg.objects.append(ScenarioObject(type="vessel", position=Vector3(1, 2, 3)))
        data = builder.export_scenario(cfg)
        assert len(data["objects"]) == 1
        assert data["objects"][0]["position"]["x"] == 1.0

    def test_import_scenario(self):
        builder = ScenarioBuilder()
        data = {
            "name": "imported",
            "world_size": 800,
            "objects": [
                {"type": "vessel", "position": {"x": 10, "y": 20, "z": 0},
                 "velocity": {"x": 1, "y": 0, "z": 0}, "behavior": "moving"}
            ],
            "duration": 90.0,
        }
        cfg = builder.import_scenario(data)
        assert cfg.name == "imported"
        assert cfg.world_size == 800.0
        assert len(cfg.objects) == 1
        assert cfg.objects[0].type == "vessel"
        assert cfg.objects[0].position.x == 10.0

    def test_export_import_roundtrip(self):
        builder = ScenarioBuilder()
        cfg1 = ScenarioConfig(name="rt", world_size=600, duration=45.0)
        cfg1.objects.append(ScenarioObject(type="buoy", position=Vector3(5, 5, 0)))
        data = builder.export_scenario(cfg1)
        cfg2 = builder.import_scenario(data)
        assert cfg2.name == cfg1.name
        assert cfg2.world_size == cfg1.world_size
        assert cfg2.duration == cfg1.duration
        assert len(cfg2.objects) == 1
        assert cfg2.objects[0].position.x == 5.0

    def test_import_defaults(self):
        builder = ScenarioBuilder()
        cfg = builder.import_scenario({})
        assert cfg.name == "imported"
        assert cfg.world_size == 1000.0
        assert cfg.objects == []
