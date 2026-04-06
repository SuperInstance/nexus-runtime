"""Tests for multi-vessel perception fusion."""

import pytest

from jetson.cooperative_perception.fusion import (
    FusedObject,
    FusionResult,
    PerceptionFusion,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fusion():
    return PerceptionFusion(association_threshold=15.0, conflict_threshold=5.0)


@pytest.fixture
def single_vessel_obs():
    """Observations from a single vessel."""
    return {
        "vessel_A": [
            {
                "id": "obj_1", "type": "vessel",
                "position": (10.0, 20.0, 0.0),
                "velocity": (2.0, 1.0, 0.0),
                "confidence": 0.9, "timestamp": 100.0,
            },
            {
                "id": "obj_2", "type": "buoy",
                "position": (30.0, 40.0, 0.5),
                "velocity": (0.0, 0.0, 0.1),
                "confidence": 0.7, "timestamp": 100.0,
            },
        ]
    }


@pytest.fixture
def multi_vessel_agreeing():
    """Two vessels observing the same object with similar data."""
    return {
        "vessel_A": [
            {
                "id": "obj_1", "type": "vessel",
                "position": (10.0, 20.0, 0.0),
                "velocity": (2.0, 1.0, 0.0),
                "confidence": 0.9, "timestamp": 100.0,
            },
        ],
        "vessel_B": [
            {
                "id": "obj_1", "type": "vessel",
                "position": (10.5, 20.3, 0.1),
                "velocity": (2.1, 1.1, 0.0),
                "confidence": 0.85, "timestamp": 100.0,
            },
        ],
    }


@pytest.fixture
def multi_vessel_conflicting():
    """Two vessels with type conflict."""
    return {
        "vessel_A": [
            {
                "id": "obj_x", "type": "vessel",
                "position": (50.0, 60.0, 0.0),
                "velocity": (3.0, 0.0, 0.0),
                "confidence": 0.9, "timestamp": 100.0,
            },
        ],
        "vessel_B": [
            {
                "id": "obj_x", "type": "buoy",
                "position": (50.2, 60.1, 0.0),
                "velocity": (3.1, 0.0, 0.0),
                "confidence": 0.8, "timestamp": 100.0,
            },
        ],
    }


@pytest.fixture
def multi_vessel_position_conflict():
    """Two vessels with large position discrepancy."""
    return {
        "vessel_A": [
            {
                "id": "obj_y", "type": "vessel",
                "position": (0.0, 0.0, 0.0),
                "velocity": (1.0, 0.0, 0.0),
                "confidence": 0.9, "timestamp": 100.0,
            },
        ],
        "vessel_B": [
            {
                "id": "obj_y", "type": "vessel",
                "position": (20.0, 0.0, 0.0),
                "velocity": (1.0, 0.0, 0.0),
                "confidence": 0.9, "timestamp": 100.0,
            },
        ],
    }


# ── FusedObject / FusionResult dataclass tests ────────────────────────────────

class TestFusedObject:

    def test_creation(self):
        fo = FusedObject(
            id="f1", type="vessel", position=(1, 2, 3),
            velocity=(0.5, 0.5, 0), confidence=0.9,
            sources=["v_A", "v_B"], fusion_method="weighted_average",
        )
        assert fo.id == "f1"
        assert len(fo.sources) == 2

    def test_default_fusion_method(self):
        fo = FusedObject(
            id="f1", type="buoy", position=(0, 0, 0),
            velocity=(0, 0, 0), confidence=0.8,
            sources=["v_A"],
        )
        assert fo.fusion_method == "weighted_average"


class TestFusionResult:

    def test_creation(self):
        fr = FusionResult(
            fused_objects=[], conflicts=[], new_objects=[], lost_objects=[],
        )
        assert fr.fused_objects == []
        assert fr.lost_objects == []


# ── PerceptionFusion tests ───────────────────────────────────────────────────

class TestFuseObservations:

    def test_single_vessel(self, fusion, single_vessel_obs):
        result = fusion.fuse_observations(single_vessel_obs)
        assert len(result.fused_objects) == 2
        assert len(result.new_objects) == 2  # First time: all new
        assert len(result.lost_objects) == 0

    def test_multi_vessel_agreeing(self, fusion, multi_vessel_agreeing):
        result = fusion.fuse_observations(multi_vessel_agreeing)
        assert len(result.fused_objects) == 1
        fo = result.fused_objects[0]
        assert len(fo.sources) == 2
        assert fo.fusion_method == "weighted_average"
        assert fo.confidence > 0.85  # Multi-vessel boost

    def test_multi_vessel_type_conflict(self, fusion, multi_vessel_conflicting):
        result = fusion.fuse_observations(multi_vessel_conflicting)
        assert len(result.conflicts) >= 1
        reasons = [c["reason"] for c in result.conflicts]
        assert "type_mismatch" in reasons

    def test_multi_vessel_position_conflict(self, fusion, multi_vessel_position_conflict):
        result = fusion.fuse_observations(multi_vessel_position_conflict)
        assert len(result.conflicts) >= 1
        reasons = [c["reason"] for c in result.conflicts]
        assert "position_conflict" in reasons

    def test_empty_observations(self, fusion):
        result = fusion.fuse_observations({})
        assert result.fused_objects == []
        assert result.new_objects == []

    def test_lost_objects(self, fusion):
        # First round
        obs1 = {"v_A": [
            {"id": "o1", "type": "vessel", "position": (1, 2, 0),
             "velocity": (0, 0, 0), "confidence": 0.9, "timestamp": 100},
        ]}
        fusion.fuse_observations(obs1)
        # Second round: o1 disappears
        obs2 = {"v_A": [
            {"id": "o2", "type": "buoy", "position": (5, 5, 0),
             "velocity": (0, 0, 0), "confidence": 0.8, "timestamp": 101},
        ]}
        result = fusion.fuse_observations(obs2)
        assert "o1" in result.lost_objects

    def test_fused_position_between_sources(self, fusion, multi_vessel_agreeing):
        result = fusion.fuse_observations(multi_vessel_agreeing)
        fo = result.fused_objects[0]
        # Position should be between the two observations
        pa = (10.0, 20.0, 0.0)
        pb = (10.5, 20.3, 0.1)
        for i in range(3):
            assert min(pa[i], pb[i]) - 0.1 <= fo.position[i] <= max(pa[i], pb[i]) + 0.1


class TestAssociateObservations:

    def test_matching_observations(self, fusion):
        obs_a = [
            {"id": "o1", "position": (10.0, 20.0, 0.0)},
            {"id": "o2", "position": (30.0, 40.0, 0.0)},
        ]
        obs_b = [
            {"id": "o1", "position": (10.5, 20.3, 0.1)},
            {"id": "o2", "position": (30.1, 40.2, 0.0)},
        ]
        matches = fusion.associate_observations(obs_a, obs_b)
        assert len(matches) == 2

    def test_no_match(self, fusion):
        obs_a = [{"id": "o1", "position": (0, 0, 0)}]
        obs_b = [{"id": "o2", "position": (100, 100, 0)}]
        matches = fusion.associate_observations(obs_a, obs_b)
        assert len(matches) == 0

    def test_empty_lists(self, fusion):
        assert fusion.associate_observations([], []) == []

    def test_one_to_many_best(self, fusion):
        obs_a = [{"id": "o1", "position": (10, 10, 0)}]
        obs_b = [
            {"id": "o1", "position": (10.1, 10.1, 0)},
            {"id": "o2", "position": (10.2, 10.2, 0)},
        ]
        matches = fusion.associate_observations(obs_a, obs_b)
        assert len(matches) == 1


class TestResolveConflicts:

    def test_type_majority_vote(self, fusion):
        conflicts = [
            {"type": "vessel", "confidence": 0.9, "position": (10, 20, 0),
             "velocity": (1, 0, 0), "source_vessel": "A", "id": "x"},
            {"type": "vessel", "confidence": 0.8, "position": (10.5, 20, 0),
             "velocity": (1.1, 0, 0), "source_vessel": "B", "id": "x"},
            {"type": "buoy", "confidence": 0.5, "position": (11, 20, 0),
             "velocity": (1, 0, 0), "source_vessel": "C", "id": "x"},
        ]
        resolved = fusion.resolve_conflicts(conflicts)
        assert len(resolved) == 1
        assert resolved[0]["type"] == "vessel"  # Majority with higher confidence

    def test_empty_conflicts(self, fusion):
        assert fusion.resolve_conflicts([]) == []

    def test_single_observation(self, fusion):
        obs = [
            {"type": "buoy", "confidence": 0.7, "position": (5, 5, 0),
             "velocity": (0, 0, 0), "source_vessel": "A", "id": "y"},
        ]
        resolved = fusion.resolve_conflicts(obs)
        assert len(resolved) == 1
        assert resolved[0]["type"] == "buoy"

    def test_weighted_position(self, fusion):
        conflicts = [
            {"type": "x", "confidence": 0.9, "position": (0, 0, 0),
             "velocity": (0, 0, 0), "source_vessel": "A", "id": "z"},
            {"type": "x", "confidence": 0.1, "position": (100, 0, 0),
             "velocity": (0, 0, 0), "source_vessel": "B", "id": "z"},
        ]
        resolved = fusion.resolve_conflicts(conflicts)
        # Weighted heavily toward (0,0,0)
        assert resolved[0]["position"][0] < 20


class TestComputeFusedPosition:

    def test_equal_weights(self, fusion):
        positions = [(0, 0, 0), (10, 10, 10)]
        confidences = [1.0, 1.0]
        result = fusion.compute_fused_position(positions, confidences)
        assert result == pytest.approx((5.0, 5.0, 5.0))

    def test_unequal_weights(self, fusion):
        positions = [(0, 0, 0), (10, 10, 10)]
        confidences = [1.0, 0.0]
        result = fusion.compute_fused_position(positions, confidences)
        # Weighted toward (0,0,0) since other has 0 weight; total_w=1.0
        assert result == pytest.approx((0.0, 0.0, 0.0))

    def test_empty_positions(self, fusion):
        result = fusion.compute_fused_position([], [])
        assert result == (0.0, 0.0, 0.0)

    def test_single_position(self, fusion):
        result = fusion.compute_fused_position([(1, 2, 3)], [0.5])
        assert result == pytest.approx((1.0, 2.0, 3.0))

    def test_zero_confidence_fallback(self, fusion):
        positions = [(0, 0, 0), (10, 0, 0)]
        confidences = [0.0, 0.0]
        result = fusion.compute_fused_position(positions, confidences)
        assert result == pytest.approx((5.0, 0.0, 0.0))


class TestComputeFusedVelocity:

    def test_equal_weights(self, fusion):
        velocities = [(2, 0, 0), (4, 0, 0)]
        confidences = [1.0, 1.0]
        result = fusion.compute_fused_velocity(velocities, confidences)
        assert result == pytest.approx((3.0, 0.0, 0.0))

    def test_empty(self, fusion):
        result = fusion.compute_fused_velocity([], [])
        assert result == (0.0, 0.0, 0.0)

    def test_single(self, fusion):
        result = fusion.compute_fused_velocity([(1, 2, 3)], [1.0])
        assert result == pytest.approx((1.0, 2.0, 3.0))

    def test_zero_confidence(self, fusion):
        velocities = [(2, 0, 0), (4, 0, 0)]
        result = fusion.compute_fused_velocity(velocities, [0.0, 0.0])
        assert result == pytest.approx((3.0, 0.0, 0.0))


class TestTrackObjectHistory:

    def test_basic_history(self, fusion):
        observations = [
            {"position": (0, 0, 0), "velocity": (1, 0, 0),
             "confidence": 0.9, "timestamp": 1.0},
            {"position": (1, 0, 0), "velocity": (1, 0, 0),
             "confidence": 0.85, "timestamp": 2.0},
        ]
        history = fusion.track_object_history("obj_1", observations)
        assert len(history) == 2
        assert history[0]["object_id"] == "obj_1"
        assert history[0]["position"] == (0, 0, 0)
        assert history[1]["position"] == (1, 0, 0)

    def test_empty_observations(self, fusion):
        history = fusion.track_object_history("obj_1", [])
        assert history == []

    def test_missing_fields_defaults(self, fusion):
        history = fusion.track_object_history("o1", [{"confidence": 0.5}])
        assert len(history) == 1
        assert history[0]["position"] == (0.0, 0.0, 0.0)
        assert history[0]["velocity"] == (0.0, 0.0, 0.0)
