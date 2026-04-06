"""Tests for associative.py — Track, Association, DataAssociator."""

import math
import pytest

from jetson.sensor_fusion.associative import Track, Association, DataAssociator


# ===================================================================
# Track tests
# ===================================================================

class TestTrack:
    def test_creation_minimal(self):
        t = Track(id=1, state=[1.0, 2.0])
        assert t.id == 1
        assert t.state == [1.0, 2.0]
        assert t.last_update == 0.0
        assert t.hits == 1
        assert t.misses == 0

    def test_creation_full(self):
        cov = [[1.0, 0.0], [0.0, 1.0]]
        t = Track(id=5, state=[3.0, 4.0], covariance=cov, last_update=10.0, hits=7, misses=2)
        assert t.hits == 7
        assert t.misses == 2
        assert t.last_update == 10.0

    def test_default_covariance(self):
        t = Track(id=1, state=[0.0])
        assert t.covariance == [[1.0]]

    def test_mutable(self):
        t = Track(id=1, state=[0.0])
        t.misses += 1
        assert t.misses == 1
        t.state[0] = 5.0
        assert t.state[0] == 5.0


# ===================================================================
# Association tests
# ===================================================================

class TestAssociation:
    def test_creation(self):
        a = Association(measurement_id=0, track_id=1, distance=2.5, gated=False)
        assert a.measurement_id == 0
        assert a.track_id == 1
        assert a.distance == pytest.approx(2.5, abs=1e-10)
        assert a.gated is False

    def test_creation_defaults(self):
        a = Association(measurement_id=5, track_id=10)
        assert a.distance == 0.0
        assert a.gated is False


# ===================================================================
# DataAssociator tests
# ===================================================================

class TestDataAssociatorCreation:
    def test_default_gate(self):
        da = DataAssociator()
        assert da.gate_threshold == 9.0

    def test_custom_gate(self):
        da = DataAssociator(gate_threshold=25.0)
        assert da.gate_threshold == 25.0


class TestDataAssociatorCostMatrix:
    def test_single_track_single_measurement(self):
        da = DataAssociator()
        tracks = [Track(id=1, state=[0.0, 0.0])]
        meas = [[3.0, 4.0]]
        cost = da.compute_cost_matrix(tracks, meas)
        assert cost[0][0] == pytest.approx(5.0, abs=1e-10)

    def test_multiple_tracks_measurements(self):
        da = DataAssociator()
        tracks = [Track(id=1, state=[0.0, 0.0]), Track(id=2, state=[10.0, 0.0])]
        meas = [[3.0, 0.0], [0.0, 4.0]]
        cost = da.compute_cost_matrix(tracks, meas)
        assert cost[0][0] == pytest.approx(3.0, abs=1e-10)  # t1->m1
        assert cost[0][1] == pytest.approx(4.0, abs=1e-10)  # t1->m2
        assert cost[1][0] == pytest.approx(7.0, abs=1e-10)  # t2->m1
        assert cost[1][1] == pytest.approx(math.sqrt(116), abs=1e-10)  # t2->m2

    def test_empty_tracks(self):
        da = DataAssociator()
        cost = da.compute_cost_matrix([], [[1.0, 2.0]])
        assert cost == []

    def test_empty_measurements(self):
        da = DataAssociator()
        cost = da.compute_cost_matrix([Track(id=1, state=[0.0, 0.0])], [])
        assert cost == [[]]


class TestDataAssociatorNearestNeighbor:
    def test_single_assignment(self):
        da = DataAssociator()
        tracks = [Track(id=1, state=[0.0, 0.0])]
        meas = [[3.0, 4.0]]
        assocs = da.nearest_neighbor(tracks, meas)
        assert len(assocs) == 1
        assert assocs[0].track_id == 1
        assert assocs[0].measurement_id == 0

    def test_closest_wins(self):
        da = DataAssociator()
        tracks = [Track(id=1, state=[0.0, 0.0]), Track(id=2, state=[10.0, 0.0])]
        meas = [[1.0, 0.0]]
        assocs = da.nearest_neighbor(tracks, meas)
        assert assocs[0].track_id == 1

    def test_no_duplicate_assignment(self):
        da = DataAssociator()
        tracks = [Track(id=1, state=[0.0, 0.0])]
        meas = [[0.1, 0.0], [0.2, 0.0]]
        assocs = da.nearest_neighbor(tracks, meas)
        # Only first measurement should be assigned (closest)
        assigned_tracks = {a.track_id for a in assocs}
        assert len(assocs) == 1

    def test_unassociated_measurements(self):
        da = DataAssociator()
        tracks = [Track(id=1, state=[0.0, 0.0])]
        meas = [[100.0, 100.0]]
        assocs = da.nearest_neighbor(tracks, meas)
        assert len(assocs) == 1  # still assigns the nearest

    def test_gate_flag(self):
        da = DataAssociator(gate_threshold=1.0)  # threshold = sqrt(1) = 1.0
        tracks = [Track(id=1, state=[0.0, 0.0])]
        meas = [[10.0, 0.0]]  # distance = 10
        assocs = da.nearest_neighbor(tracks, meas)
        assert assocs[0].gated is True


class TestDataAssociatorGlobalNearestNeighbor:
    def test_optimal_assignment(self):
        da = DataAssociator()
        tracks = [Track(id=1, state=[0.0, 0.0]), Track(id=2, state=[10.0, 0.0])]
        meas = [[1.0, 0.0], [9.0, 0.0]]
        assocs = da.global_nearest_neighbor(tracks, meas)
        assert len(assocs) == 2
        track_ids = {a.track_id for a in assocs}
        meas_ids = {a.measurement_id for a in assocs}
        assert 1 in track_ids
        assert 2 in track_ids
        assert 0 in meas_ids
        assert 1 in meas_ids

    def test_empty_inputs(self):
        da = DataAssociator()
        assert da.global_nearest_neighbor([], []) == []
        assert da.global_nearest_neighbor([Track(id=1, state=[0.0])], []) == []
        assert da.global_nearest_neighbor([], [[1.0]]) == []

    def test_custom_cost_matrix(self):
        da = DataAssociator()
        tracks = [Track(id=1, state=[0.0, 0.0]), Track(id=2, state=[10.0, 0.0])]
        meas = [[1.0, 0.0], [9.0, 0.0]]
        # Custom cost matrix: swap the costs
        cost = [[100.0, 1.0], [1.0, 100.0]]
        assocs = da.global_nearest_neighbor(tracks, meas, cost_matrix=cost)
        # Greedy will pick the 1.0 entries
        assert len(assocs) == 2

    def test_gate_flag_gnn(self):
        da = DataAssociator(gate_threshold=4.0)  # sqrt(4) = 2
        tracks = [Track(id=1, state=[0.0, 0.0])]
        meas = [[10.0, 0.0]]
        assocs = da.global_nearest_neighbor(tracks, meas)
        assert len(assocs) == 1
        assert assocs[0].gated is True


class TestDataAssociatorGatedAssociation:
    def test_within_gate(self):
        da = DataAssociator(gate_threshold=25.0)  # sqrt(25) = 5
        tracks = [Track(id=1, state=[0.0, 0.0])]
        meas = [[3.0, 0.0]]
        filtered = da.gated_association(tracks, meas)
        assert len(filtered) == 1
        assert filtered[0].gated is False

    def test_outside_gate(self):
        da = DataAssociator(gate_threshold=1.0)
        tracks = [Track(id=1, state=[0.0, 0.0])]
        meas = [[10.0, 0.0]]
        filtered = da.gated_association(tracks, meas)
        assert len(filtered) == 0

    def test_custom_gate_threshold(self):
        da = DataAssociator(gate_threshold=100.0)
        tracks = [Track(id=1, state=[0.0, 0.0])]
        meas = [[5.0, 0.0]]
        filtered = da.gated_association(tracks, meas, gate_threshold=3.0)
        assert len(filtered) == 0  # dist=5 > threshold=3

    def test_empty_inputs(self):
        da = DataAssociator()
        assert da.gated_association([], []) == []
        assert da.gated_association([Track(id=1, state=[0.0])], []) == []

    def test_multiple_tracks_multiple_measurements(self):
        da = DataAssociator(gate_threshold=100.0)
        tracks = [Track(id=1, state=[0.0, 0.0]), Track(id=2, state=[0.0, 0.0])]
        meas = [[1.0, 0.0], [2.0, 0.0]]
        filtered = da.gated_association(tracks, meas)
        assert len(filtered) == 4  # 2 tracks × 2 measurements


class TestDataAssociatorCreateNewTracks:
    def test_single_measurement(self):
        da = DataAssociator()
        new = da.create_new_tracks([[5.0, 6.0]])
        assert len(new) == 1
        assert new[0].state == [5.0, 6.0]
        assert new[0].hits == 1
        assert new[0].misses == 0

    def test_multiple_measurements(self):
        da = DataAssociator()
        new = da.create_new_tracks([[1.0], [2.0], [3.0]])
        assert len(new) == 3
        assert new[0].id != new[1].id != new[2].id

    def test_auto_incrementing_ids(self):
        da = DataAssociator()
        t1 = da.create_new_tracks([[0.0]])
        t2 = da.create_new_tracks([[1.0]])
        assert t2[0].id == t1[0].id + 1

    def test_empty_list(self):
        da = DataAssociator()
        new = da.create_new_tracks([])
        assert new == []

    def test_initial_covariance(self):
        da = DataAssociator()
        new = da.create_new_tracks([[0.0, 0.0]])
        assert new[0].covariance[0][0] == 10.0
        assert new[0].covariance[1][1] == 10.0

    def test_state_copy(self):
        da = DataAssociator()
        m = [1.0, 2.0]
        new = da.create_new_tracks([m])
        m[0] = 999.0
        assert new[0].state[0] == 1.0


class TestDataAssociatorDeleteLostTracks:
    def test_no_deletion(self):
        da = DataAssociator()
        tracks = [
            Track(id=1, state=[0.0], misses=0),
            Track(id=2, state=[0.0], misses=2),
        ]
        kept = da.delete_lost_tracks(tracks, max_misses=3)
        assert len(kept) == 2

    def test_delete_at_threshold(self):
        da = DataAssociator()
        tracks = [
            Track(id=1, state=[0.0], misses=2),
            Track(id=2, state=[0.0], misses=3),
            Track(id=3, state=[0.0], misses=4),
        ]
        kept = da.delete_lost_tracks(tracks, max_misses=3)
        assert len(kept) == 2
        assert kept[0].id == 1
        assert kept[1].id == 2

    def test_delete_all(self):
        da = DataAssociator()
        tracks = [Track(id=i, state=[0.0], misses=10) for i in range(5)]
        kept = da.delete_lost_tracks(tracks, max_misses=5)
        assert len(kept) == 0

    def test_empty_list(self):
        da = DataAssociator()
        kept = da.delete_lost_tracks([], max_misses=3)
        assert kept == []

    def test_default_max_misses(self):
        da = DataAssociator()
        tracks = [
            Track(id=1, state=[0.0], misses=3),
            Track(id=2, state=[0.0], misses=4),
        ]
        kept = da.delete_lost_tracks(tracks)  # default max_misses=3
        assert len(kept) == 1
        assert kept[0].id == 1
