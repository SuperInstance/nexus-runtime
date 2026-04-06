"""Tests for sensor_sim.py — SensorConfig, SensorReading, SensorSimulator."""

import math
import pytest

from jetson.simulation.world import Vector3, World, WorldObject
from jetson.simulation.sensor_sim import SensorConfig, SensorReading, SensorSimulator


class TestSensorConfig:
    def test_default_creation(self):
        cfg = SensorConfig()
        assert cfg.type == "generic"
        assert cfg.range == 100.0
        assert cfg.noise_stddev == 1.0
        assert cfg.update_rate == 10.0

    def test_custom_creation(self):
        cfg = SensorConfig(type="lidar", range=50.0, noise_stddev=0.5)
        assert cfg.type == "lidar"
        assert cfg.range == 50.0


class TestSensorReading:
    def test_default_creation(self):
        r = SensorReading()
        assert r.timestamp == 0.0
        assert r.sensor_id == ""
        assert r.data == {}
        assert r.confidence == 1.0

    def test_custom_creation(self):
        r = SensorReading(timestamp=1.0, sensor_id="gps", data={"lat": 1.0}, confidence=0.9)
        assert r.timestamp == 1.0
        assert r.confidence == 0.9


class TestSensorSimulatorCreation:
    def test_default_creation(self):
        sim = SensorSimulator()
        assert sim.sensor_count == 0

    def test_seeded_creation(self):
        sim1 = SensorSimulator(seed=42)
        sim2 = SensorSimulator(seed=42)
        # Same seed should produce same results
        r1 = sim1.simulate_gps(Vector3(0, 0, 0))
        r2 = sim2.simulate_gps(Vector3(0, 0, 0))
        assert r1.data["latitude"] == r2.data["latitude"]


class TestAddRemoveSensor:
    def test_add_sensor(self):
        sim = SensorSimulator()
        sim.add_sensor(SensorConfig(type="gps", noise_stddev=1.0), "gps1")
        assert sim.sensor_count == 1

    def test_remove_sensor(self):
        sim = SensorSimulator()
        sim.add_sensor(SensorConfig(type="gps"), "gps1")
        assert sim.remove_sensor("gps1") is True
        assert sim.sensor_count == 0

    def test_remove_nonexistent(self):
        sim = SensorSimulator()
        assert sim.remove_sensor("nope") is False

    def test_get_sensor(self):
        sim = SensorSimulator()
        cfg = SensorConfig(type="gps", noise_stddev=2.5)
        sim.add_sensor(cfg, "gps1")
        found = sim.get_sensor("gps1")
        assert found is not None
        assert found.noise_stddev == 2.5

    def test_get_nonexistent_sensor(self):
        sim = SensorSimulator()
        assert sim.get_sensor("nope") is None


class TestSimulateGPS:
    def test_gps_returns_reading(self):
        sim = SensorSimulator(seed=42)
        reading = sim.simulate_gps(Vector3(10, 20, 0))
        assert isinstance(reading, SensorReading)
        assert "latitude" in reading.data
        assert "longitude" in reading.data

    def test_gps_has_sensor_id(self):
        sim = SensorSimulator()
        reading = sim.simulate_gps(Vector3(0, 0, 0), "my_gps")
        assert reading.sensor_id == "my_gps"

    def test_gps_uses_config_noise(self):
        sim = SensorSimulator(seed=42)
        sim.add_sensor(SensorConfig(type="gps", noise_stddev=0.0), "precise")
        reading = sim.simulate_gps(Vector3(5, 10, 0), "precise")
        assert reading.data["latitude"] == pytest.approx(5.0)
        assert reading.data["longitude"] == pytest.approx(10.0)

    def test_gps_confidence_in_range(self):
        sim = SensorSimulator(seed=42)
        reading = sim.simulate_gps(Vector3(0, 0, 0))
        assert 0.0 <= reading.confidence <= 1.0

    def test_gps_stores_true_position(self):
        sim = SensorSimulator(seed=42)
        reading = sim.simulate_gps(Vector3(1, 2, 3))
        assert reading.data["true_position"] == (1.0, 2.0, 3.0)


class TestSimulateCompass:
    def test_compass_returns_reading(self):
        sim = SensorSimulator(seed=42)
        reading = sim.simulate_compass(math.pi / 2)
        assert isinstance(reading, SensorReading)
        assert "heading" in reading.data

    def test_compass_sensor_id(self):
        sim = SensorSimulator()
        reading = sim.simulate_compass(0, "cmp")
        assert reading.sensor_id == "cmp"

    def test_compass_heading_degrees(self):
        sim = SensorSimulator(seed=42)
        reading = sim.simulate_compass(0.0)
        assert "heading_degrees" in reading.data

    def test_compass_deviation(self):
        sim = SensorSimulator(seed=42)
        reading = sim.simulate_compass(1.0)
        assert "deviation_degrees" in reading.data
        assert "true_heading" in reading.data

    def test_compass_zero_noise(self):
        sim = SensorSimulator(seed=42)
        sim.add_sensor(SensorConfig(type="compass", noise_stddev=0.0), "perfect")
        reading = sim.simulate_compass(math.pi / 4, "perfect")
        assert reading.data["heading"] == pytest.approx(math.pi / 4)

    def test_compass_confidence(self):
        sim = SensorSimulator(seed=42)
        reading = sim.simulate_compass(0.0)
        assert 0.0 <= reading.confidence <= 1.0


class TestSimulateLiDAR:
    def test_lidar_returns_reading(self):
        sim = SensorSimulator(seed=42)
        world = World()
        world.add_object(WorldObject(id="wall", position=Vector3(10, 0, 0), properties={"radius": 5}))
        reading = sim.simulate_lidar(world, "lidar1", Vector3(0, 0, 0), Vector3(1, 0, 0), fov=90, max_range=100)
        assert isinstance(reading, SensorReading)
        assert "distances" in reading.data
        assert "angles" in reading.data

    def test_lidar_num_rays(self):
        sim = SensorSimulator(seed=42)
        world = World()
        reading = sim.simulate_lidar(world, "l1", Vector3(0, 0, 0), Vector3(1, 0, 0), num_rays=10)
        assert len(reading.data["distances"]) == 10
        assert len(reading.data["angles"]) == 10

    def test_lidar_no_hits(self):
        sim = SensorSimulator(seed=42)
        world = World()
        reading = sim.simulate_lidar(world, "l1", Vector3(0, 0, 0), Vector3(1, 0, 0), max_range=50)
        # All rays should return max_range (no hits)
        assert all(d >= 0 for d in reading.data["distances"])

    def test_lidar_max_range_caps(self):
        sim = SensorSimulator(seed=42)
        sim.add_sensor(SensorConfig(type="lidar", range=10.0), "short")
        world = World()
        reading = sim.simulate_lidar(world, "short", Vector3(0, 0, 0), Vector3(1, 0, 0), max_range=100)
        # Sensor range should cap max_range
        assert reading.data["max_range"] <= 10.0


class TestSimulateDepth:
    def test_depth_returns_reading(self):
        sim = SensorSimulator(seed=42)
        reading = sim.simulate_depth(Vector3(0, 0, 0), 10.0)
        assert isinstance(reading, SensorReading)
        assert "depth" in reading.data

    def test_depth_sensor_id(self):
        sim = SensorSimulator()
        reading = sim.simulate_depth(Vector3(0, 0, 5), 20.0, "d1")
        assert reading.sensor_id == "d1"

    def test_depth_positive(self):
        sim = SensorSimulator(seed=42)
        reading = sim.simulate_depth(Vector3(0, 0, 0), 10.0)
        assert reading.data["depth"] >= 0

    def test_depth_submerged(self):
        sim = SensorSimulator(seed=42)
        reading = sim.simulate_depth(Vector3(0, 0, -5), 10.0)
        # depth = seabed - position.z = 10 - (-5) = 15
        assert reading.data["true_depth"] == pytest.approx(15.0)

    def test_depth_confidence(self):
        sim = SensorSimulator(seed=42)
        reading = sim.simulate_depth(Vector3(0, 0, 0), 10.0)
        assert 0.0 <= reading.confidence <= 1.0


class TestSimulateSpeed:
    def test_speed_returns_reading(self):
        sim = SensorSimulator(seed=42)
        reading = sim.simulate_speed(Vector3(3, 4, 0))
        assert isinstance(reading, SensorReading)
        assert "speed" in reading.data

    def test_speed_value(self):
        sim = SensorSimulator(seed=42)
        reading = sim.simulate_speed(Vector3(3, 4, 0))
        assert reading.data["true_speed"] == pytest.approx(5.0)

    def test_speed_at_rest(self):
        sim = SensorSimulator(seed=42)
        reading = sim.simulate_speed(Vector3(0, 0, 0))
        assert reading.data["true_speed"] == pytest.approx(0.0)

    def test_speed_non_negative(self):
        sim = SensorSimulator(seed=42)
        reading = sim.simulate_speed(Vector3(0, 0, 0))
        assert reading.data["speed"] >= 0.0


class TestTimeManagement:
    def test_initial_time(self):
        sim = SensorSimulator()
        assert sim.time == 0.0

    def test_update_time(self):
        sim = SensorSimulator()
        sim.update_time(0.5)
        assert sim.time == pytest.approx(0.5)

    def test_time_reflected_in_readings(self):
        sim = SensorSimulator()
        sim.update_time(2.5)
        reading = sim.simulate_gps(Vector3(0, 0, 0))
        assert reading.timestamp == pytest.approx(2.5)
