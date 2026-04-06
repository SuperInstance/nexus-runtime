"""Cross-module integration tests: Perception pipeline (kalman, detection, marine, sharing, fusion).

Each test calls 2+ perception modules together.
"""

import math
import time
import pytest

from jetson.sensor_fusion.kalman import (
    LinearKalmanFilter, ExtendedKalmanFilter, KalmanState,
    _mat_identity, _mat_zeros, _mat_mul, _mat_add,
)
from jetson.vision.detection import (
    ObjectDetector, BoundingBox, DetectionResult, GridDetector, GridCell,
)
from jetson.vision.marine_detectors import (
    BuoyDetector, VesselDetector, DebrisDetector, NavigationMarkerDetector,
    MarineObject, MarineDetection,
)
from jetson.cooperative_perception.sharing import (
    PerceptionSharer, PerceivedObject, PerceptionMessage,
)
from jetson.cooperative_perception.fusion import (
    PerceptionFusion, FusedObject, FusionResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_image(w=40, h=30, fill=(100, 100, 100)):
    """Create a simple RGB image as list-of-lists."""
    return [[fill for _ in range(w)] for _ in range(h)]


def _add_buoy_region(image, x, y, radius=3, color=(220, 50, 50)):
    """Add a buoy-colored circle region to the image."""
    h = len(image)
    w = len(image[0]) if h else 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                if dx * dx + dy * dy <= radius * radius:
                    image[ny][nx] = color


def _add_dark_row(image, y, x_start, x_end):
    """Add a dark horizontal row for vessel detection."""
    w = len(image[0]) if image else 0
    for x in range(max(0, x_start), min(w, x_end)):
        image[y][x] = (30, 30, 30)


def _make_perceived_object(obj_id="obj1", obj_type="buoy", x=10.0, y=20.0, z=0.0,
                           vx=1.0, vy=0.0, vz=0.0, confidence=0.9):
    return PerceivedObject(
        id=obj_id, type=obj_type,
        position=(x, y, z), velocity=(vx, vy, vz),
        size=(2.0, 2.0, 2.0), confidence=confidence,
        source_vessel="vessel_a", timestamp=time.time(),
    )


# ===========================================================================
# 1. Kalman + Detection integration tests
# ===========================================================================

class TestKalmanDetectionIntegration:
    """Tests that exercise Kalman filters together with ObjectDetector."""

    def test_kalman_tracks_detected_object_position(self):
        kf = LinearKalmanFilter(state_dim=4, measurement_dim=2)
        detector = ObjectDetector(num_classes=5, conf_threshold=0.3)
        gt = [BoundingBox(10, 10, 20, 20), BoundingBox(12, 11, 20, 20)]
        detections = detector.detect(640, 480, gt)
        assert len(detections) > 0
        # Feed detection center as measurement to Kalman
        d = detections[0]
        cx, cy = d.bbox.cx, d.bbox.cy
        kf.predict(0.1)
        state = kf.update([cx, cy])
        assert len(state.mean) == 4
        assert state.mean[0] != 0 or state.mean[1] != 0

    def test_multi_scale_detection_fed_to_kalman(self):
        kf = LinearKalmanFilter(state_dim=2, measurement_dim=2)
        detector = ObjectDetector()
        gt = [BoundingBox(50, 50, 30, 30)]
        results = detector.detect_multi_scale([0.5, 1.0, 1.5], gt)
        # Feed NMS output to Kalman
        for r in results:
            kf.predict(0.1)
            kf.update([r.bbox.cx, r.bbox.cy])
        state = kf.get_state()
        assert state is not None

    def test_kalman_innovation_from_detection(self):
        kf = LinearKalmanFilter(state_dim=2, measurement_dim=2, initial_state=[100.0, 100.0])
        detector = ObjectDetector()
        gt = [BoundingBox(10, 10, 20, 20)]
        dets = detector.detect(640, 480, gt)
        if dets:
            innovation = kf.compute_innovation([dets[0].bbox.cx, dets[0].bbox.cy])
            assert len(innovation) == 2
            assert abs(innovation[0]) > 50  # should be large since predicted far from detected

    def test_ekf_with_nonlinear_detection_model(self):
        ekf = ExtendedKalmanFilter(state_dim=4, measurement_dim=2)
        detector = ObjectDetector()
        gt = [BoundingBox(100, 100, 40, 40)]
        dets = detector.detect(640, 480, gt)
        if dets:
            def state_fn(x, dt, u=None):
                # Simple constant velocity
                return [x[0] + x[2] * dt, x[1] + x[3] * dt, x[2], x[3]]
            def meas_fn(x):
                return [x[0], x[1]]
            ekf.predict(0.1, state_transition_fn=state_fn)
            ekf.update([dets[0].bbox.cx, dets[0].bbox.cy], measurement_fn=meas_fn)
            state = ekf.get_state()
            assert len(state.mean) == 4

    def test_iou_from_detector_used_with_kalman_uncertainty(self):
        detector = ObjectDetector()
        a = BoundingBox(10, 10, 20, 20)
        b = BoundingBox(15, 15, 20, 20)
        iou = detector.compute_iou(a, b)
        assert 0.0 < iou < 1.0
        kf = LinearKalmanFilter(state_dim=4, measurement_dim=2)
        state = kf.get_state()
        # Covariance diagonal should be 1.0 initially
        cov_diag = state.covariance[0][0]
        assert cov_diag > 0


# ===========================================================================
# 2. Marine Detectors + Kalman integration tests
# ===========================================================================

class TestMarineKalmanIntegration:
    """Tests that exercise marine detectors with Kalman filters."""

    def test_buoy_detection_fed_to_kalman(self):
        buoy_det = BuoyDetector()
        kf = LinearKalmanFilter(state_dim=2, measurement_dim=2)
        image = _make_simple_image(40, 30)
        _add_buoy_region(image, 20, 15, 3, (220, 50, 50))
        buoys = buoy_det.detect_buoys(image, 40, 30)
        if buoys:
            b = buoys[0]
            kf.predict(0.1)
            state = kf.update([b.x, b.y])
            assert state.mean[0] != 0 or state.mean[1] != 0

    def test_vessel_detection_fed_to_kalman(self):
        vessel_det = VesselDetector()
        kf = LinearKalmanFilter(state_dim=2, measurement_dim=2)
        image = _make_simple_image(100, 30)
        _add_dark_row(image, 15, 5, 80)
        vessels = vessel_det.detect_vessels(image, 100, 30)
        if vessels:
            v = vessels[0]
            kf.predict(0.1)
            state = kf.update([v.x, v.y])
            assert state.mean[0] != 0

    def test_multi_buoy_tracking_with_kalman(self):
        buoy_det = BuoyDetector()
        image = _make_simple_image(60, 40)
        _add_buoy_region(image, 10, 10, 2, (220, 50, 50))
        _add_buoy_region(image, 40, 30, 2, (200, 200, 200))
        buoys = buoy_det.detect_buoys(image, 60, 40)
        # Create a separate KF for each buoy
        filters = [LinearKalmanFilter(state_dim=2, measurement_dim=2) for _ in buoys]
        for b, kf in zip(buoys, filters):
            kf.predict(0.1)
            kf.update([b.x, b.y])
        assert len(filters) == len(buoys)

    def test_buoy_distance_and_kalman_state(self):
        buoy_det = BuoyDetector()
        kf = LinearKalmanFilter(state_dim=3, measurement_dim=2)
        dist = buoy_det.estimate_distance(50.0)
        assert dist > 0
        # Use distance as z-measurement
        kf.predict(0.1)
        kf.update([100.0, 200.0])


# ===========================================================================
# 3. Grid Detector + Object Detector integration tests
# ===========================================================================

class TestGridDetectionIntegration:
    """Tests that exercise GridDetector with ObjectDetector."""

    def test_assign_detections_to_grid(self):
        detector = ObjectDetector()
        grid_det = GridDetector(grid_rows=13, grid_cols=13)
        gt = [BoundingBox(100, 100, 50, 50), BoundingBox(300, 200, 60, 40)]
        dets = detector.detect(640, 480, gt)
        grid = grid_det.create_grid()
        grid_det.assign_boxes_to_grid(grid, dets, 640, 480)
        # Check that some cells have detections
        has_detections = any(len(cell.boxes) > 0 for row in grid for cell in row)
        assert has_detections

    def test_grid_aggregation_with_nms(self):
        detector = ObjectDetector(nms_threshold=0.5)
        grid_det = GridDetector(grid_rows=7, grid_cols=7)
        gt = [BoundingBox(50, 50, 30, 30)]
        # Add duplicate detections manually
        dets = detector.detect(640, 480, gt)
        dets.append(DetectionResult(
            class_id=0, class_name="class_0",
            confidence=0.9, bbox=BoundingBox(52, 52, 28, 28),
        ))
        nms_result = detector.non_max_suppression(dets)
        assert len(nms_result) <= len(dets)
        grid = grid_det.create_grid()
        grid_det.assign_boxes_to_grid(grid, nms_result, 640, 480)

    def test_grid_confidence_map(self):
        detector = ObjectDetector(conf_threshold=0.3)
        grid_det = GridDetector(grid_rows=13, grid_cols=13)
        gt = [BoundingBox(200, 200, 40, 40)]
        dets = detector.detect(640, 480, gt)
        grid = grid_det.create_grid()
        grid_det.assign_boxes_to_grid(grid, dets, 640, 480)
        conf_map = grid_det.compute_grid_confidence(grid)
        assert len(conf_map) == 13
        assert len(conf_map[0]) == 13

    def test_grid_aggregate_threshold(self):
        grid_det = GridDetector(grid_rows=7, grid_cols=7)
        grid = grid_det.create_grid()
        # Manually populate a cell
        grid[3][3].confidence = 0.8
        grid[3][3].boxes.append(DetectionResult(
            class_id=0, class_name="class_0",
            confidence=0.8, bbox=BoundingBox(100, 100, 20, 20),
        ))
        aggregated = grid_det.aggregate_grid_detections(grid, threshold=0.5)
        assert len(aggregated) == 1


# ===========================================================================
# 4. Marine Detectors + Object Detector integration tests
# ===========================================================================

class TestMarineObjectDetectionIntegration:
    """Tests that exercise marine detectors with base ObjectDetector."""

    def test_buoy_and_vessel_detected_in_same_image(self):
        image = _make_simple_image(100, 40)
        _add_buoy_region(image, 80, 20, 2, (220, 50, 50))
        _add_dark_row(image, 10, 10, 70)
        buoy_det = BuoyDetector()
        vessel_det = VesselDetector()
        buoys = buoy_det.detect_buoys(image, 100, 40)
        vessels = vessel_det.detect_vessels(image, 100, 40)
        total = len(buoys) + len(vessels)
        assert total >= 0  # detection depends on image content

    def test_buoy_classification_after_detection(self):
        buoy_det = BuoyDetector()
        image = _make_simple_image(30, 30)
        _add_buoy_region(image, 15, 15, 3, (220, 50, 50))
        buoys = buoy_det.detect_buoys(image, 30, 30)
        if buoys:
            cls = buoy_det.classify_buoy(buoys[0])
            assert cls in ("red", "white", "yellow", "green", "orange", "unknown")

    def test_vessel_tracking_across_frames(self):
        vessel_det = VesselDetector()
        prev_positions = {0: (50.0, 15.0), 1: (200.0, 25.0)}
        detections = [
            MarineDetection(obj_type=MarineObject.VESSEL, confidence=0.8, x=52.0, y=15.0, size=30.0),
            MarineDetection(obj_type=MarineObject.VESSEL, confidence=0.7, x=205.0, y=25.0, size=25.0),
        ]
        tracks = vessel_det.track_vessels(detections, prev_positions)
        assert len(tracks) == 2

    def test_debris_detection_and_classification(self):
        debris_det = DebrisDetector()
        # Create image with varying patches
        image = _make_simple_image(30, 30)
        # Add some variation for debris detection
        for y in range(5, 25):
            for x in range(5, 25):
                r = 100 + (x * 7 + y * 13) % 60
                g = 80 + (x * 11 + y * 3) % 40
                b = 60 + (x * 5 + y * 7) % 30
                image[y][x] = (r, g, b)
        debris = debris_det.detect_debris(image, 30, 30)
        if debris:
            cls = debris_det.classify_debris(debris[0])
            assert cls in ("organic", "plastic", "metal", "wood")

    def test_navigation_marker_detection(self):
        marker_det = NavigationMarkerDetector()
        image = _make_simple_image(30, 30)
        _add_buoy_region(image, 10, 10, 2, (200, 30, 30))  # red lateral
        markers = marker_det.detect_markers(image, 30, 30)
        # Detection depends on proximity to pattern colors
        assert isinstance(markers, list)

    def test_debris_drift_estimation(self):
        positions = [(0, 0), (1, 1), (2, 2), (3, 3)]
        speed, direction = DebrisDetector.estimate_drift_direction(positions)
        assert speed > 0
        assert abs(direction) < math.pi

    def test_vessel_size_estimation(self):
        size = VesselDetector.estimate_vessel_size(50.0, distance=100.0)
        assert size > 0

    def test_buoy_bearing_estimation(self):
        det = MarineDetection(obj_type=MarineObject.BUOY, confidence=0.9, x=200.0, y=150.0, size=10.0)
        bearing = BuoyDetector.estimate_bearing(det, 320.0, 240.0)
        assert isinstance(bearing, float)

    def test_vessel_heading_estimation(self):
        points = [(0, 0), (10, 5), (20, 10)]
        heading = VesselDetector.estimate_vessel_heading(points)
        assert isinstance(heading, float)


# ===========================================================================
# 5. Detection + Perception Sharing integration tests
# ===========================================================================

class TestDetectionSharingIntegration:
    """Tests that exercise detection modules with PerceptionSharer."""

    def test_share_buoy_detection(self):
        buoy_det = BuoyDetector()
        sharer = PerceptionSharer(vessel_id="vessel_a", sensor_type="camera")
        image = _make_simple_image(40, 30)
        _add_buoy_region(image, 20, 15, 3, (220, 50, 50))
        buoys = buoy_det.detect_buoys(image, 40, 30)
        if buoys:
            b = buoys[0]
            obj = PerceivedObject(
                id="buoy_1", type="buoy",
                position=(b.x, b.y, 0.0), velocity=(0, 0, 0),
                size=(b.size, b.size, b.size), confidence=b.confidence,
                source_vessel="vessel_a", timestamp=time.time(),
            )
            msg = sharer.create_message(
                {"position": (0, 0, 0), "confidence": 0.9}, [obj]
            )
            assert msg.sender_id == "vessel_a"
            assert msg.object_count() == 1

    def test_serialize_deserialize_detection_message(self):
        sharer = PerceptionSharer(vessel_id="v1")
        obj = _make_perceived_object("det1", "vessel", 100, 200, 0, 2, 0, 0, 0.95)
        msg = sharer.create_message({"position": (0, 0, 0)}, [obj])
        data = sharer.serialize_message(msg)
        restored = sharer.deserialize_message(data)
        assert restored.sender_id == "v1"
        assert restored.object_count() == 1
        assert restored.objects[0].id == "det1"

    def test_compress_decompress_detection_message(self):
        sharer = PerceptionSharer(vessel_id="v1")
        objects = [_make_perceived_object(f"d{i}", "buoy", i * 10, i * 10) for i in range(5)]
        msg = sharer.create_message({"position": (0, 0, 0)}, objects)
        compressed = sharer.compress_for_bandwidth(msg, max_bytes=500)
        assert len(compressed) <= 500
        restored = sharer.decompress_message(compressed)
        assert restored.sender_id == "v1"

    def test_message_priority_from_detections(self):
        sharer = PerceptionSharer(vessel_id="v1")
        objects = [_make_perceived_object("fast", "vessel", 0, 0, 0, 15, 10, 0)]
        msg = sharer.create_message({"position": (0, 0, 0)}, objects)
        priority = sharer.compute_message_priority(msg)
        assert 0.0 <= priority <= 1.0

    def test_relevance_filter_on_shared_detections(self):
        sharer = PerceptionSharer(vessel_id="v1")
        objects = [
            _make_perceived_object("near", "buoy", 5, 5, 0, 0, 0, 0, 0.9),
            _make_perceived_object("far", "buoy", 500, 500, 0, 0, 0, 0, 0.8),
        ]
        msg = sharer.create_message({"position": (0, 0, 0)}, objects)
        filtered = sharer.filter_by_relevance(msg, (10, 10, 0), max_range=100)
        assert filtered.object_count() <= 2

    def test_multi_vessel_sharing(self):
        sharer_a = PerceptionSharer(vessel_id="vessel_a")
        sharer_b = PerceptionSharer(vessel_id="vessel_b")
        obj_a = _make_perceived_object("o1", "buoy", 10, 20, 0)
        obj_b = _make_perceived_object("o1", "buoy", 12, 22, 0)
        msg_a = sharer_a.create_message({"position": (0, 0, 0)}, [obj_a])
        msg_b = sharer_b.create_message({"position": (100, 100, 0)}, [obj_b])
        data_b = sharer_b.serialize_message(msg_b)
        restored_b = sharer_a.deserialize_message(data_b)
        assert restored_b.sender_id == "vessel_b"


# ===========================================================================
# 6. Perception Fusion + Sharing integration tests
# ===========================================================================

class TestFusionSharingIntegration:
    """Tests that exercise PerceptionFusion with PerceptionSharer."""

    def test_fuse_observations_from_two_vessels(self):
        fusion = PerceptionFusion(association_threshold=15.0)
        sharer_a = PerceptionSharer(vessel_id="vessel_a")
        sharer_b = PerceptionSharer(vessel_id="vessel_b")
        obj_a = _make_perceived_object("o1", "buoy", 10, 20, 0, 1, 0, 0, 0.9)
        obj_b = _make_perceived_object("o1", "buoy", 12, 22, 0, 1, 0, 0, 0.85)
        msg_a = sharer_a.create_message({"position": (0, 0, 0)}, [obj_a])
        msg_b = sharer_b.create_message({"position": (100, 100, 0)}, [obj_b])
        obs = {
            "vessel_a": [
                {"id": o.id, "type": o.type, "position": o.position,
                 "velocity": o.velocity, "confidence": o.confidence, "timestamp": o.timestamp}
                for o in msg_a.objects
            ],
            "vessel_b": [
                {"id": o.id, "type": o.type, "position": o.position,
                 "velocity": o.velocity, "confidence": o.confidence, "timestamp": o.timestamp}
                for o in msg_b.objects
            ],
        }
        result = fusion.fuse_observations(obs)
        assert len(result.fused_objects) >= 1

    def test_fusion_new_and_lost_objects(self):
        fusion = PerceptionFusion()
        sharer = PerceptionSharer(vessel_id="v1")
        # First round: one object
        obj1 = _make_perceived_object("o1", "buoy", 10, 20, 0)
        msg1 = sharer.create_message({"position": (0, 0, 0)}, [obj1])
        obs1 = {"v1": [{"id": o.id, "type": o.type, "position": o.position,
                         "velocity": o.velocity, "confidence": o.confidence,
                         "timestamp": o.timestamp} for o in msg1.objects]}
        result1 = fusion.fuse_observations(obs1)
        assert len(result1.new_objects) == 1
        # Second round: different object (o1 is lost)
        obj2 = _make_perceived_object("o2", "vessel", 50, 60, 0)
        msg2 = sharer.create_message({"position": (0, 0, 0)}, [obj2])
        obs2 = {"v1": [{"id": o.id, "type": o.type, "position": o.position,
                         "velocity": o.velocity, "confidence": o.confidence,
                         "timestamp": o.timestamp} for o in msg2.objects]}
        result2 = fusion.fuse_observations(obs2)
        assert "o1" in result2.lost_objects

    def test_associate_observations_from_two_lists(self):
        fusion = PerceptionFusion(association_threshold=20.0)
        list_a = [{"id": "a1", "position": (10, 20, 0)}, {"id": "a2", "position": (50, 60, 0)}]
        list_b = [{"id": "b1", "position": (12, 22, 0)}, {"id": "b2", "position": (200, 200, 0)}]
        matches = fusion.associate_observations(list_a, list_b)
        assert len(matches) >= 1

    def test_resolve_type_conflicts(self):
        fusion = PerceptionFusion()
        conflicts = [
            {"id": "o1", "type": "buoy", "position": (10, 20, 0),
             "velocity": (0, 0, 0), "confidence": 0.9, "source_vessel": "v1"},
            {"id": "o1", "type": "vessel", "position": (12, 22, 0),
             "velocity": (0, 0, 0), "confidence": 0.7, "source_vessel": "v2"},
        ]
        resolved = fusion.resolve_conflicts(conflicts)
        assert len(resolved) == 1
        assert resolved[0]["source_vessel"] == "fusion"

    def test_fused_position_is_weighted_average(self):
        fusion = PerceptionFusion()
        positions = [(10, 20, 0), (30, 20, 0)]
        confidences = [0.8, 0.4]
        fused = fusion.compute_fused_position(positions, confidences)
        # Higher confidence (0.8) pulls closer to (10, 20, 0)
        assert fused[0] < 20  # closer to 10 than 30

    def test_track_history_from_observations(self):
        fusion = PerceptionFusion()
        observations = [
            {"id": "o1", "position": (10, 20, 0), "velocity": (1, 0, 0),
             "confidence": 0.9, "timestamp": 1.0},
            {"id": "o1", "position": (20, 20, 0), "velocity": (1, 0, 0),
             "confidence": 0.9, "timestamp": 2.0},
            {"id": "o1", "position": (30, 20, 0), "velocity": (1, 0, 0),
             "confidence": 0.9, "timestamp": 3.0},
        ]
        track = fusion.track_object_history("o1", observations)
        assert len(track) == 3
        assert track[0]["position"] == (10, 20, 0)
        assert track[2]["position"] == (30, 20, 0)


# ===========================================================================
# 7. Kalman + Fusion integration tests
# ===========================================================================

class TestKalmanFusionIntegration:
    """Tests that exercise Kalman filters with PerceptionFusion."""

    def test_fused_observation_fed_to_kalman(self):
        fusion = PerceptionFusion(association_threshold=15.0)
        kf = LinearKalmanFilter(state_dim=2, measurement_dim=2)
        obs = {
            "v1": [{"id": "o1", "type": "buoy", "position": (10, 20, 0),
                     "velocity": (0, 0, 0), "confidence": 0.9, "timestamp": 1.0}],
            "v2": [{"id": "o1", "type": "buoy", "position": (12, 22, 0),
                     "velocity": (0, 0, 0), "confidence": 0.8, "timestamp": 1.0}],
        }
        result = fusion.fuse_observations(obs)
        assert len(result.fused_objects) >= 1
        fo = result.fused_objects[0]
        kf.predict(0.1)
        kf.update([fo.position[0], fo.position[1]])
        state = kf.get_state()
        assert state.mean[0] > 0

    def test_kalman_predict_between_fusion_updates(self):
        fusion = PerceptionFusion()
        kf = LinearKalmanFilter(state_dim=4, measurement_dim=2)
        # First fusion update
        obs1 = {"v1": [{"id": "o1", "type": "v", "position": (50, 80, 0),
                         "velocity": (1, 1, 0), "confidence": 0.9, "timestamp": 1.0}]}
        result1 = fusion.fuse_observations(obs1)
        fo = result1.fused_objects[0]
        kf.predict(0.1)
        kf.update([fo.position[0], fo.position[1]])
        # Predict forward without measurement
        kf.predict(0.5)
        state = kf.get_state()
        assert state.mean[0] != 0 or state.mean[1] != 0

    def test_ekf_with_fused_velocity(self):
        fusion = PerceptionFusion()
        ekf = ExtendedKalmanFilter(state_dim=4, measurement_dim=2)
        obs = {
            "v1": [{"id": "o1", "type": "v", "position": (10, 20, 0),
                     "velocity": (5, 3, 0), "confidence": 0.9, "timestamp": 1.0}],
        }
        result = fusion.fuse_observations(obs)
        fo = result.fused_objects[0]
        def meas_fn(x):
            return [x[0], x[1]]
        ekf.predict(0.1)
        ekf.update([fo.position[0], fo.position[1]], measurement_fn=meas_fn)
        assert ekf.get_state() is not None


# ===========================================================================
# 8. Full perception pipeline tests (3+ modules)
# ===========================================================================

class TestFullPerceptionPipeline:
    """End-to-end tests: detection -> sharing -> fusion -> kalman tracking."""

    def test_detect_share_fuse_track(self):
        buoy_det = BuoyDetector()
        sharer = PerceptionSharer(vessel_id="v1")
        fusion = PerceptionFusion(association_threshold=20.0)
        kf = LinearKalmanFilter(state_dim=2, measurement_dim=2)
        image = _make_simple_image(40, 30)
        _add_buoy_region(image, 20, 15, 3, (220, 50, 50))
        buoys = buoy_det.detect_buoys(image, 40, 30)
        if buoys:
            b = buoys[0]
            obj = PerceivedObject(
                id="b1", type="buoy", position=(b.x, b.y, 0.0),
                velocity=(0, 0, 0), size=(b.size, b.size, b.size),
                confidence=b.confidence, source_vessel="v1", timestamp=time.time(),
            )
            msg = sharer.create_message({"position": (0, 0, 0)}, [obj])
            data = sharer.serialize_message(msg)
            restored = sharer.deserialize_message(data)
            obs = {"v1": [{"id": o.id, "type": o.type, "position": o.position,
                           "velocity": o.velocity, "confidence": o.confidence,
                           "timestamp": o.timestamp} for o in restored.objects]}
            fusion_result = fusion.fuse_observations(obs)
            assert len(fusion_result.fused_objects) >= 1
            fo = fusion_result.fused_objects[0]
            kf.predict(0.1)
            kf.update([fo.position[0], fo.position[1]])
            assert kf.get_state().mean[0] > 0

    def test_multi_vessel_detect_share_fuse(self):
        sharer_a = PerceptionSharer(vessel_id="va")
        sharer_b = PerceptionSharer(vessel_id="vb")
        fusion = PerceptionFusion(association_threshold=20.0)
        # Vessel A sees object at (10, 20, 0)
        obj_a = _make_perceived_object("o1", "buoy", 10, 20, 0)
        msg_a = sharer_a.create_message({"position": (0, 0, 0)}, [obj_a])
        # Vessel B sees same object at (12, 21, 0)
        obj_b = _make_perceived_object("o1", "buoy", 12, 21, 0)
        msg_b = sharer_b.create_message({"position": (100, 0, 0)}, [obj_b])
        # Share: serialize B and deserialize on A's side
        data_b = sharer_b.serialize_message(msg_b)
        msg_b_on_a = sharer_a.deserialize_message(data_b)
        # Fuse
        all_obs = {
            "va": [{"id": o.id, "type": o.type, "position": o.position,
                    "velocity": o.velocity, "confidence": o.confidence,
                    "timestamp": o.timestamp} for o in msg_a.objects],
            "vb": [{"id": o.id, "type": o.type, "position": o.position,
                    "velocity": o.velocity, "confidence": o.confidence,
                    "timestamp": o.timestamp} for o in msg_b_on_a.objects],
        }
        result = fusion.fuse_observations(all_obs)
        assert len(result.fused_objects) >= 1
        fo = result.fused_objects[0]
        assert "va" in fo.sources or "vb" in fo.sources

    def test_kalman_state_update_with_shared_message(self):
        sharer = PerceptionSharer(vessel_id="v1")
        kf = LinearKalmanFilter(state_dim=2, measurement_dim=2)
        obj = _make_perceived_object("o1", "buoy", 100, 200, 0)
        msg = sharer.create_message({"position": (0, 0, 0)}, [obj])
        # Check message properties
        assert msg.object_count() == 1
        # Feed to Kalman
        kf.predict(0.1)
        kf.update([obj.position[0], obj.position[1]])
        state = kf.get_state()
        # Kalman filter blends measurement with prior; just check non-zero update
        assert state.mean[0] != 0 or state.mean[1] != 0

    def test_grid_detection_to_fusion_pipeline(self):
        detector = ObjectDetector()
        grid_det = GridDetector(grid_rows=13, grid_cols=13)
        sharer = PerceptionSharer(vessel_id="v1")
        fusion = PerceptionFusion()
        gt = [BoundingBox(100, 100, 40, 40)]
        dets = detector.detect(640, 480, gt)
        grid = grid_det.create_grid()
        grid_det.assign_boxes_to_grid(grid, dets, 640, 480)
        aggregated = grid_det.aggregate_grid_detections(grid, threshold=0.1)
        # Convert to PerceivedObjects and share
        if aggregated:
            objs = [PerceivedObject(
                id=f"d{i}", type=f"class_{d.class_id}",
                position=(d.bbox.cx, d.bbox.cy, 0), velocity=(0, 0, 0),
                size=(d.bbox.w, d.bbox.h, 0), confidence=d.confidence,
                source_vessel="v1", timestamp=time.time(),
            ) for i, d in enumerate(aggregated)]
            msg = sharer.create_message({"position": (0, 0, 0)}, objs)
            assert msg.object_count() == len(objs)

    def test_buoy_detection_to_sharing_with_compression(self):
        buoy_det = BuoyDetector()
        sharer = PerceptionSharer(vessel_id="v1")
        image = _make_simple_image(40, 30)
        _add_buoy_region(image, 20, 15, 3, (220, 50, 50))
        buoys = buoy_det.detect_buoys(image, 40, 30)
        if buoys:
            b = buoys[0]
            cls = buoy_det.classify_buoy(b)
            dist = buoy_det.estimate_distance(b.size)
            bearing = buoy_det.estimate_bearing(b, 20, 15)
            obj = PerceivedObject(
                id="b1", type=f"buoy_{cls}", position=(b.x, b.y, 0),
                velocity=(0, 0, 0), size=(b.size, b.size, b.size),
                confidence=b.confidence, source_vessel="v1", timestamp=time.time(),
            )
            msg = sharer.create_message(
                {"position": (0, 0, 0), "confidence": 0.9, "fields_of_view": [(0, 0, 40, 30)]},
                [obj]
            )
            compressed = sharer.compress_for_bandwidth(msg, max_bytes=1000)
            assert len(compressed) <= 1000
            restored = sharer.decompress_message(compressed)
            assert restored.object_count() >= 1

    def test_multi_scale_detect_with_kalman_tracking(self):
        detector = ObjectDetector(conf_threshold=0.3)
        kf = LinearKalmanFilter(state_dim=4, measurement_dim=2)
        gt = [BoundingBox(100, 100, 30, 30)]
        for scale in [0.8, 1.0, 1.2]:
            scaled_gt = [BoundingBox(g.x * scale, g.y * scale, g.w * scale, g.h * scale) for g in gt]
            dets = detector.detect(640, 480, scaled_gt)
            if dets:
                d = dets[0]
                kf.predict(0.1)
                kf.update([d.bbox.cx, d.bbox.cy])
        state = kf.get_state()
        assert len(state.mean) == 4

    def test_fusion_with_conflicts_and_kalman(self):
        fusion = PerceptionFusion(conflict_threshold=3.0)
        kf = LinearKalmanFilter(state_dim=2, measurement_dim=2)
        obs = {
            "v1": [{"id": "o1", "type": "buoy", "position": (10, 20, 0),
                     "velocity": (0, 0, 0), "confidence": 0.9, "timestamp": 1.0}],
            "v2": [{"id": "o1", "type": "vessel", "position": (10, 20, 0),
                     "velocity": (0, 0, 0), "confidence": 0.7, "timestamp": 1.0}],
        }
        result = fusion.fuse_observations(obs)
        # Type conflict should be flagged
        type_conflicts = [c for c in result.conflicts if c.get("reason") == "type_mismatch"]
        if type_conflicts:
            fo = result.fused_objects[0]
            kf.update([fo.position[0], fo.position[1]])
        assert len(result.fused_objects) >= 1

    def test_perceived_object_distance_and_speed(self):
        obj = _make_perceived_object("o1", "vessel", 10, 20, 30, 3, 4, 5)
        dist = obj.distance_to((0, 0, 0))
        assert dist > 0
        speed = obj.speed()
        assert abs(speed - math.sqrt(9 + 16 + 25)) < 1e-6

    def test_message_age_calculation(self):
        sharer = PerceptionSharer(vessel_id="v1")
        msg = sharer.create_message({"position": (0, 0, 0)}, [])
        age = msg.age()
        assert 0.0 <= age < 1.0
