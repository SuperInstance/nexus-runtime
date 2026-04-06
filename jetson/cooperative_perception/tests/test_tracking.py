"""Tests for cooperative object tracking."""

import time
import pytest

from jetson.cooperative_perception.tracking import (
    CooperativeTrack,
    TrackAssociation,
    CooperativeTracker,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tracker():
    return CooperativeTracker(association_threshold=12.0)


@pytest.fixture
def base_observation():
    return {
        "position": (10.0, 20.0, 0.0),
        "velocity": (2.0, 1.0, 0.0),
        "confidence": 0.9,
        "timestamp": time.time(),
        "source_vessel": "vessel_A",
        "object_type": "vessel",
        "object_id": "obj_1",
    }


@pytest.fixture
def populated_tracker(tracker, base_observation):
    """Tracker with one track."""
    tid = tracker.create_track(base_observation)
    return tracker, tid


# ── CooperativeTrack dataclass tests ──────────────────────────────────────────

class TestCooperativeTrack:

    def test_creation(self):
        t = CooperativeTrack(
            id="t1", state_history=[{"position": (1, 2, 3)}],
            last_update=time.time(),
        )
        assert t.id == "t1"
        assert t.vessel_count() == 0

    def test_latest_state(self):
        t = CooperativeTrack(
            id="t1",
            state_history=[{"position": (0, 0, 0)}, {"position": (1, 1, 1)}],
        )
        assert t.latest_state()["position"] == (1, 1, 1)

    def test_latest_state_empty(self):
        t = CooperativeTrack(id="t1", state_history=[])
        assert t.latest_state() is None

    def test_vessel_count(self):
        t = CooperativeTrack(
            id="t1", state_history=[],
            contributing_vessels=["A", "B", "A"],
        )
        assert t.vessel_count() == 2

    def test_vessel_count_empty(self):
        t = CooperativeTrack(id="t1", state_history=[])
        assert t.vessel_count() == 0

    def test_age(self):
        t = CooperativeTrack(
            id="t1", state_history=[],
            last_update=time.time() - 5.0,
        )
        assert 4.0 <= t.age() <= 7.0

    def test_age_with_now(self):
        t = CooperativeTrack(
            id="t1", state_history=[],
            last_update=100.0,
        )
        assert t.age(now=110.0) == pytest.approx(10.0)

    def test_default_quality(self):
        t = CooperativeTrack(id="t1", state_history=[])
        assert t.quality == 1.0


class TestTrackAssociation:

    def test_creation(self):
        ta = TrackAssociation(
            local_track_id="lt1", remote_track_id="rt1",
            association_confidence=0.85, vessel_ids=["A", "B"],
        )
        assert ta.local_track_id == "lt1"
        assert ta.association_confidence == 0.85

    def test_default_vessel_ids(self):
        ta = TrackAssociation(
            local_track_id="lt1", remote_track_id="rt1",
            association_confidence=0.5,
        )
        assert ta.vessel_ids == []


# ── CooperativeTracker tests ─────────────────────────────────────────────────

class TestCreateTrack:

    def test_create_returns_id(self, tracker, base_observation):
        tid = tracker.create_track(base_observation)
        assert tid.startswith("ctrack_")

    def test_create_adds_to_tracks(self, tracker, base_observation):
        tid = tracker.create_track(base_observation)
        assert tid in tracker.list_tracks()

    def test_create_unique_ids(self, tracker, base_observation):
        id1 = tracker.create_track(base_observation)
        id2 = tracker.create_track(base_observation)
        assert id1 != id2

    def test_create_track_state(self, tracker, base_observation):
        tid = tracker.create_track(base_observation)
        track = tracker.get_track(tid)
        assert track is not None
        assert len(track.state_history) == 1
        assert track.contributing_vessels == ["vessel_A"]

    def test_create_minimal_observation(self, tracker):
        tid = tracker.create_track({})
        track = tracker.get_track(tid)
        assert track is not None
        assert track.contributing_vessels == ["unknown"]


class TestUpdateTrack:

    def test_update_adds_state(self, populated_tracker):
        tracker, tid = populated_tracker
        obs = {
            "position": (12.0, 21.0, 0.0),
            "velocity": (2.0, 1.0, 0.0),
            "confidence": 0.85,
            "timestamp": time.time(),
            "source_vessel": "vessel_A",
        }
        track = tracker.update_track(tid, obs)
        assert len(track.state_history) == 2

    def test_update_clears_prediction(self, populated_tracker):
        tracker, tid = populated_tracker
        # First predict
        tracker.predict_tracks(1.0)
        assert tracker.get_track(tid).predicted_state is not None
        # Then update
        tracker.update_track(tid, {
            "position": (12, 21, 0), "velocity": (2, 1, 0),
            "confidence": 0.9, "timestamp": time.time(),
            "source_vessel": "A",
        })
        assert tracker.get_track(tid).predicted_state is None

    def test_update_adds_new_vessel(self, populated_tracker):
        tracker, tid = populated_tracker
        tracker.update_track(tid, {
            "position": (12, 21, 0), "velocity": (2, 1, 0),
            "confidence": 0.9, "timestamp": time.time(),
            "source_vessel": "vessel_B",
        })
        track = tracker.get_track(tid)
        assert "vessel_B" in track.contributing_vessels

    def test_update_nonexistent(self, tracker):
        with pytest.raises(KeyError):
            tracker.update_track("nonexistent", {})


class TestPredictTracks:

    def test_predict_returns_dict(self, populated_tracker):
        tracker, tid = populated_tracker
        predictions = tracker.predict_tracks(1.0)
        assert isinstance(predictions, dict)
        assert tid in predictions

    def test_predict_position(self, populated_tracker):
        tracker, tid = populated_tracker
        track = tracker.get_track(tid)
        pos = track.latest_state()["position"]
        vel = track.latest_state()["velocity"]
        predictions = tracker.predict_tracks(2.0)
        pred = predictions[tid]
        expected = (
            pos[0] + vel[0] * 2.0,
            pos[1] + vel[1] * 2.0,
            pos[2] + vel[2] * 2.0,
        )
        assert pred["position"] == pytest.approx(expected, abs=0.01)

    def test_predict_confidence_decay(self, populated_tracker):
        tracker, tid = populated_tracker
        predictions = tracker.predict_tracks(10.0)
        pred_conf = predictions[tid]["confidence"]
        # Should be less than original (0.9)
        assert pred_conf < 0.9

    def test_predict_method(self, populated_tracker):
        tracker, tid = populated_tracker
        predictions = tracker.predict_tracks(1.0)
        assert predictions[tid]["method"] == "linear_prediction"

    def test_predict_empty_tracker(self, tracker):
        predictions = tracker.predict_tracks(1.0)
        assert predictions == {}


class TestAssociateLocalRemote:

    def test_associate_matching(self, tracker, base_observation):
        # Create a local track
        local_id = tracker.create_track(base_observation)
        local_track = tracker.get_track(local_id)

        # Create a matching remote track
        remote_id = tracker.create_track({
            **base_observation,
            "position": (10.5, 20.3, 0.0),  # Close to local
            "source_vessel": "vessel_B",
        })
        remote_track = tracker.get_track(remote_id)

        # Remove from tracker so they don't interfere
        del tracker._tracks[remote_id]

        associations = tracker.associate_local_remote([local_track], [remote_track])
        assert len(associations) == 1
        assert associations[0].local_track_id == local_id
        assert associations[0].association_confidence > 0

    def test_associate_no_match(self, tracker, base_observation):
        local_id = tracker.create_track(base_observation)
        local_track = tracker.get_track(local_id)

        far_id = tracker.create_track({
            **base_observation,
            "position": (200.0, 200.0, 0.0),  # Far away
            "source_vessel": "vessel_B",
        })
        far_track = tracker.get_track(far_id)
        del tracker._tracks[far_id]

        associations = tracker.associate_local_remote([local_track], [far_track])
        assert len(associations) == 0

    def test_associate_vessel_ids(self, tracker, base_observation):
        local_id = tracker.create_track(base_observation)
        local_track = tracker.get_track(local_id)

        remote_id = tracker.create_track({
            **base_observation,
            "source_vessel": "vessel_B",
        })
        remote_track = tracker.get_track(remote_id)
        del tracker._tracks[remote_id]

        associations = tracker.associate_local_remote([local_track], [remote_track])
        assert len(associations) == 1
        assert "vessel_A" in associations[0].vessel_ids
        assert "vessel_B" in associations[0].vessel_ids

    def test_associate_empty(self, tracker):
        assert tracker.associate_local_remote([], []) == []


class TestMergeTracks:

    def test_merge_success(self, tracker):
        obs_a = {"position": (0, 0, 0), "velocity": (1, 0, 0),
                 "confidence": 0.9, "timestamp": 100, "source_vessel": "A"}
        obs_b = {"position": (10, 0, 0), "velocity": (0, 1, 0),
                 "confidence": 0.8, "timestamp": 100, "source_vessel": "B"}

        id_a = tracker.create_track(obs_a)
        id_b = tracker.create_track(obs_b)

        merged = tracker.merge_tracks(id_a, id_b)
        assert merged.id not in (id_a, id_b)
        assert id_a not in tracker.list_tracks()
        assert id_b not in tracker.list_tracks()
        assert merged.id in tracker.list_tracks()

    def test_merge_vessels(self, tracker):
        id_a = tracker.create_track({
            "position": (0, 0, 0), "velocity": (1, 0, 0),
            "confidence": 0.9, "timestamp": 100, "source_vessel": "A",
        })
        id_b = tracker.create_track({
            "position": (10, 0, 0), "velocity": (0, 1, 0),
            "confidence": 0.8, "timestamp": 100, "source_vessel": "B",
        })
        merged = tracker.merge_tracks(id_a, id_b)
        assert "A" in merged.contributing_vessels
        assert "B" in merged.contributing_vessels

    def test_merge_keeps_longer_history(self, tracker):
        id_a = tracker.create_track({
            "position": (0, 0, 0), "velocity": (1, 0, 0),
            "confidence": 0.9, "timestamp": 100, "source_vessel": "A",
        })
        tracker.update_track(id_a, {
            "position": (1, 0, 0), "velocity": (1, 0, 0),
            "confidence": 0.9, "timestamp": 101, "source_vessel": "A",
        })
        id_b = tracker.create_track({
            "position": (10, 0, 0), "velocity": (0, 1, 0),
            "confidence": 0.8, "timestamp": 100, "source_vessel": "B",
        })
        merged = tracker.merge_tracks(id_a, id_b)
        assert len(merged.state_history) == 2

    def test_merge_nonexistent(self, tracker):
        with pytest.raises(KeyError):
            tracker.merge_tracks("fake_a", "fake_b")

    def test_merge_quality_average(self, tracker):
        id_a = tracker.create_track({
            "position": (0, 0, 0), "velocity": (0, 0, 0),
            "confidence": 0.8, "timestamp": 100, "source_vessel": "A",
        })
        id_b = tracker.create_track({
            "position": (10, 0, 0), "velocity": (0, 0, 0),
            "confidence": 0.6, "timestamp": 100, "source_vessel": "B",
        })
        merged = tracker.merge_tracks(id_a, id_b)
        assert merged.quality == pytest.approx(0.7)


class TestDeleteStaleTracks:

    def test_delete_stale(self, tracker):
        # Create a track with old timestamp
        old_id = tracker.create_track({
            "position": (0, 0, 0), "velocity": (0, 0, 0),
            "confidence": 0.9, "timestamp": time.time() - 1000,
            "source_vessel": "A",
        })
        # Create a fresh track
        fresh_id = tracker.create_track({
            "position": (1, 0, 0), "velocity": (0, 0, 0),
            "confidence": 0.9, "timestamp": time.time(),
            "source_vessel": "B",
        })
        deleted = tracker.delete_stale_tracks(max_age=60.0)
        assert deleted == 1
        assert old_id not in tracker.list_tracks()
        assert fresh_id in tracker.list_tracks()

    def test_delete_none_fresh(self, tracker, base_observation):
        tracker.create_track(base_observation)
        deleted = tracker.delete_stale_tracks(max_age=60.0)
        assert deleted == 0

    def test_delete_empty_tracker(self, tracker):
        deleted = tracker.delete_stale_tracks(max_age=60.0)
        assert deleted == 0


class TestTrackerUtilities:

    def test_get_track_found(self, populated_tracker):
        tracker, tid = populated_tracker
        track = tracker.get_track(tid)
        assert track is not None
        assert track.id == tid

    def test_get_track_not_found(self, tracker):
        assert tracker.get_track("nonexistent") is None

    def test_list_tracks_empty(self, tracker):
        assert tracker.list_tracks() == []

    def test_list_tracks_populated(self, populated_tracker):
        tracker, tid = populated_tracker
        assert tid in tracker.list_tracks()
        assert len(tracker.list_tracks()) == 1
