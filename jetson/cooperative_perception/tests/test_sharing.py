"""Tests for perception data sharing protocol."""

import math
import time
import zlib
import json
import pytest

from jetson.cooperative_perception.sharing import (
    PerceivedObject,
    PerceptionMessage,
    PerceptionSharer,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sharer():
    return PerceptionSharer(vessel_id="vessel_01", sensor_type="lidar")


@pytest.fixture
def sample_objects():
    now = time.time()
    return [
        PerceivedObject(
            id="obj_1", type="vessel", position=(10.0, 20.0, 0.0),
            velocity=(2.0, 1.0, 0.0), size=(5.0, 2.0, 3.0),
            confidence=0.9, source_vessel="vessel_01", timestamp=now,
        ),
        PerceivedObject(
            id="obj_2", type="buoy", position=(30.0, 40.0, 0.5),
            velocity=(0.0, 0.0, 0.1), size=(1.0, 1.0, 2.0),
            confidence=0.7, source_vessel="vessel_01", timestamp=now,
        ),
        PerceivedObject(
            id="obj_3", type="vessel", position=(-5.0, 100.0, 0.0),
            velocity=(5.0, -1.0, 0.0), size=(8.0, 3.0, 4.0),
            confidence=0.85, source_vessel="vessel_01", timestamp=now,
        ),
    ]


@pytest.fixture
def vessel_state():
    return {
        "position": (0.0, 0.0, 0.0),
        "confidence": 0.95,
        "fields_of_view": [(0, 0, 100, 100)],
    }


@pytest.fixture
def sample_message(sharer, vessel_state, sample_objects):
    return sharer.create_message(vessel_state, sample_objects)


# ── PerceivedObject tests ────────────────────────────────────────────────────

class TestPerceivedObject:

    def test_creation(self):
        obj = PerceivedObject(
            id="o1", type="buoy", position=(1.0, 2.0, 3.0),
            velocity=(0.1, 0.2, 0.0), size=(1, 1, 2),
            confidence=0.8, source_vessel="v1", timestamp=1000.0,
        )
        assert obj.id == "o1"
        assert obj.type == "buoy"
        assert obj.confidence == 0.8

    def test_distance_to_same_position(self):
        obj = PerceivedObject(
            id="o1", type="buoy", position=(1.0, 2.0, 3.0),
            velocity=(0, 0, 0), size=(1, 1, 1),
            confidence=1.0, source_vessel="v1", timestamp=0,
        )
        assert obj.distance_to((1.0, 2.0, 3.0)) == pytest.approx(0.0)

    def test_distance_to_known(self):
        obj = PerceivedObject(
            id="o1", type="buoy", position=(0.0, 0.0, 0.0),
            velocity=(0, 0, 0), size=(1, 1, 1),
            confidence=1.0, source_vessel="v1", timestamp=0,
        )
        assert obj.distance_to((3.0, 4.0, 0.0)) == pytest.approx(5.0)

    def test_speed_zero(self):
        obj = PerceivedObject(
            id="o1", type="buoy", position=(0, 0, 0),
            velocity=(0, 0, 0), size=(1, 1, 1),
            confidence=1.0, source_vessel="v1", timestamp=0,
        )
        assert obj.speed() == pytest.approx(0.0)

    def test_speed_magnitude(self):
        obj = PerceivedObject(
            id="o1", type="vessel", position=(0, 0, 0),
            velocity=(3.0, 4.0, 0.0), size=(5, 2, 3),
            confidence=0.9, source_vessel="v1", timestamp=0,
        )
        assert obj.speed() == pytest.approx(5.0)

    def test_speed_3d(self):
        obj = PerceivedObject(
            id="o1", type="submarine", position=(0, 0, 0),
            velocity=(1, 2, 2), size=(10, 3, 5),
            confidence=0.8, source_vessel="v1", timestamp=0,
        )
        assert obj.speed() == pytest.approx(3.0)


# ── PerceptionMessage tests ──────────────────────────────────────────────────

class TestPerceptionMessage:

    def test_creation(self, sample_message):
        assert sample_message.sender_id == "vessel_01"
        assert sample_message.sensor_type == "lidar"
        assert len(sample_message.objects) == 3

    def test_age_recent(self, sample_message):
        assert sample_message.age() < 1.0

    def test_age_with_timestamp(self):
        msg = PerceptionMessage(
            sender_id="v1", timestamp=time.time() - 10,
            objects=[], sensor_type="cam", position=(0, 0, 0),
            confidence=1.0,
        )
        assert 9.0 <= msg.age() <= 12.0

    def test_object_count(self, sample_message):
        assert sample_message.object_count() == 3

    def test_object_count_empty(self):
        msg = PerceptionMessage(
            sender_id="v1", timestamp=0, objects=[],
            sensor_type="cam", position=(0, 0, 0), confidence=1.0,
        )
        assert msg.object_count() == 0

    def test_fields_of_view_default(self):
        msg = PerceptionMessage(
            sender_id="v1", timestamp=0, objects=[],
            sensor_type="cam", position=(0, 0, 0), confidence=1.0,
        )
        assert msg.fields_of_view == []

    def test_fields_of_view_set(self, vessel_state, sharer, sample_objects):
        msg = sharer.create_message(vessel_state, sample_objects)
        assert msg.fields_of_view == [(0, 0, 100, 100)]


# ── PerceptionSharer tests ───────────────────────────────────────────────────

class TestPerceptionSharerCreation:

    def test_init_defaults(self):
        s = PerceptionSharer("v1")
        assert s.vessel_id == "v1"
        assert s.sensor_type == "lidar"

    def test_init_custom_sensor(self):
        s = PerceptionSharer("v1", sensor_type="radar")
        assert s.sensor_type == "radar"

    def test_create_message_basic(self, sharer, vessel_state, sample_objects):
        msg = sharer.create_message(vessel_state, sample_objects)
        assert msg.sender_id == "vessel_01"
        assert msg.confidence == 0.95
        assert msg.position == (0.0, 0.0, 0.0)

    def test_create_message_no_confidence(self, sharer, sample_objects):
        state = {"position": (5, 5, 0)}
        msg = sharer.create_message(state, sample_objects)
        assert msg.confidence == 1.0

    def test_create_message_no_position(self, sharer, sample_objects):
        msg = sharer.create_message({}, sample_objects)
        assert msg.position == (0.0, 0.0, 0.0)

    def test_create_message_empty_objects(self, sharer, vessel_state):
        msg = sharer.create_message(vessel_state, [])
        assert msg.object_count() == 0


class TestPerceptionSharerSerialize:

    def test_serialize_roundtrip(self, sharer, sample_message):
        data = sharer.serialize_message(sample_message)
        assert isinstance(data, bytes)
        restored = sharer.deserialize_message(data)
        assert restored.sender_id == sample_message.sender_id
        assert restored.timestamp == pytest.approx(sample_message.timestamp)
        assert restored.sensor_type == sample_message.sensor_type
        assert len(restored.objects) == len(sample_message.objects)

    def test_serialize_object_fields(self, sharer, sample_message):
        data = sharer.serialize_message(sample_message)
        restored = sharer.deserialize_message(data)
        for orig, rstr in zip(sample_message.objects, restored.objects):
            assert orig.id == rstr.id
            assert orig.type == rstr.type
            assert orig.position == rstr.position
            assert orig.velocity == rstr.velocity
            assert orig.size == rstr.size
            assert orig.confidence == pytest.approx(rstr.confidence)
            assert orig.source_vessel == rstr.source_vessel

    def test_serialize_empty_message(self, sharer):
        msg = PerceptionMessage(
            sender_id="v1", timestamp=100.0, objects=[],
            sensor_type="cam", position=(1, 2, 3), confidence=0.5,
            fields_of_view=[(0, 0, 50, 50)],
        )
        data = sharer.serialize_message(msg)
        restored = sharer.deserialize_message(data)
        assert restored.sender_id == "v1"
        assert restored.objects == []
        assert restored.fields_of_view == [(0, 0, 50, 50)]

    def test_deserialize_invalid_json(self, sharer):
        with pytest.raises((json.JSONDecodeError, UnicodeDecodeError, KeyError)):
            sharer.deserialize_message(b"not json at all")


class TestPerceptionSharerCompress:

    def test_compress_decompress_roundtrip(self, sharer, sample_message):
        compressed = sharer.compress_for_bandwidth(sample_message, max_bytes=10000)
        decompressed = sharer.decompress_message(compressed)
        assert decompressed.sender_id == sample_message.sender_id
        assert len(decompressed.objects) == len(sample_message.objects)

    def test_compress_small_max_bytes(self, sharer, sample_message):
        # Force pruning by setting a very small max_bytes
        compressed = sharer.compress_for_bandwidth(sample_message, max_bytes=200)
        assert len(compressed) <= 200
        decompressed = sharer.decompress_message(compressed)
        assert decompressed.sender_id == sample_message.sender_id

    def test_compress_uses_zlib(self, sharer, sample_message):
        compressed = sharer.compress_for_bandwidth(sample_message, max_bytes=10000)
        raw = sharer.serialize_message(sample_message)
        assert len(compressed) < len(raw)

    def test_decompress_invalid(self, sharer):
        with pytest.raises(zlib.error):
            sharer.decompress_message(b"not compressed data!")


class TestPerceptionSharerPriority:

    def test_priority_range(self, sharer, sample_message):
        p = sharer.compute_message_priority(sample_message)
        assert 0.0 <= p <= 1.0

    def test_priority_empty_message(self, sharer):
        msg = PerceptionMessage(
            sender_id="v1", timestamp=time.time(), objects=[],
            sensor_type="cam", position=(0, 0, 0), confidence=1.0,
        )
        p = sharer.compute_message_priority(msg)
        assert 0.0 <= p <= 0.2  # Low priority for empty

    def test_priority_high_speed(self, sharer):
        now = time.time()
        obj = PerceivedObject(
            id="fast", type="vessel", position=(0, 0, 0),
            velocity=(15.0, 10.0, 0.0), size=(5, 2, 3),
            confidence=0.95, source_vessel="v1", timestamp=now,
        )
        msg = PerceptionMessage(
            sender_id="v1", timestamp=now, objects=[obj],
            sensor_type="lidar", position=(0, 0, 0), confidence=0.9,
        )
        p = sharer.compute_message_priority(msg)
        assert p > 0.5  # High speed should boost priority

    def test_priority_stale_message(self, sharer):
        obj = PerceivedObject(
            id="s1", type="buoy", position=(0, 0, 0),
            velocity=(0, 0, 0), size=(1, 1, 1),
            confidence=0.5, source_vessel="v1", timestamp=0,
        )
        msg = PerceptionMessage(
            sender_id="v1", timestamp=time.time() - 120, objects=[obj],
            sensor_type="cam", position=(0, 0, 0), confidence=0.5,
        )
        p = sharer.compute_message_priority(msg)
        assert p < 0.5  # Stale message should have low priority


class TestPerceptionSharerFilter:

    def test_filter_keeps_nearby(self, sharer, sample_message):
        filtered = sharer.filter_by_relevance(
            sample_message, receiver_position=(10.0, 20.0, 0.0), max_range=50.0
        )
        assert len(filtered.objects) >= 1

    def test_filter_removes_far(self, sharer, sample_message):
        filtered = sharer.filter_by_relevance(
            sample_message, receiver_position=(0.0, 0.0, 0.0), max_range=5.0
        )
        # obj_1 at (10,20,0) is >5 from origin
        for obj in filtered.objects:
            assert obj.distance_to((0, 0, 0)) <= 5.0

    def test_filter_strips_fov(self, sharer, sample_message):
        filtered = sharer.filter_by_relevance(
            sample_message, receiver_position=(0, 0, 0), max_range=1000
        )
        assert filtered.fields_of_view == []

    def test_filter_preserves_metadata(self, sharer, sample_message):
        filtered = sharer.filter_by_relevance(
            sample_message, receiver_position=(0, 0, 0), max_range=1000
        )
        assert filtered.sender_id == sample_message.sender_id
        assert filtered.sensor_type == sample_message.sensor_type
        assert filtered.timestamp == sample_message.timestamp

    def test_filter_all_out_of_range(self, sharer, sample_message):
        filtered = sharer.filter_by_relevance(
            sample_message, receiver_position=(500.0, 500.0, 0.0), max_range=10.0
        )
        assert filtered.object_count() == 0

    def test_filter_zero_range(self, sharer, sample_message):
        filtered = sharer.filter_by_relevance(
            sample_message, receiver_position=(10.0, 20.0, 0.0), max_range=0.0
        )
        # obj_1 is exactly at (10,20,0) so distance=0 which is <= 0.0
        assert filtered.object_count() == 1
