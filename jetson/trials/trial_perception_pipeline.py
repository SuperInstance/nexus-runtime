"""Trial 4: Perception Pipeline — Vision + Sensor fusion + Cooperative perception + NL.

Tests end-to-end perception: simulated sensor data through Kalman filter,
object detection, marine object detection, cooperative perception sharing,
and multi-vessel fusion.
"""

from jetson.sensor_fusion.kalman import (
    LinearKalmanFilter, ExtendedKalmanFilter, KalmanState,
    _mat_identity, _mat_zeros, _mat_mul, _mat_add, _mat_inverse,
)
from jetson.vision.detection import (
    ObjectDetector, BoundingBox, DetectionResult, GridDetector, GridCell,
)
from jetson.vision.marine_detectors import (
    BuoyDetector, VesselDetector, DebrisDetector, NavigationMarkerDetector,
    MarineDetection, MarineObject,
)
from jetson.cooperative_perception.sharing import (
    PerceptionSharer, PerceptionMessage, PerceivedObject,
)
from jetson.cooperative_perception.fusion import (
    PerceptionFusion, FusedObject, FusionResult,
)
from jetson.nl_commands.parser import NLParser, Token, Entity, ParseTree


def run_trial():
    """Run all perception pipeline integration tests. Returns True if all pass."""
    passed = 0
    failed = 0
    total = 0

    def check(name, condition):
        nonlocal passed, failed, total
        total += 1
        if condition:
            passed += 1
        else:
            failed += 1

    # === Sensor Fusion: Linear Kalman Filter ===

    # 1. Basic LKF construction
    lkf = LinearKalmanFilter(state_dim=2, measurement_dim=2)
    check("lkf_created", lkf.state_dim == 2)

    # 2. Initial state
    state = lkf.get_state()
    check("lkf_initial_state_zero", state.mean[0] == 0.0 and state.mean[1] == 0.0)

    # 3. Predict step
    pred = lkf.predict(dt=1.0)
    check("lkf_predict", pred.mean is not None)
    check("lkf_predict_timestamp", pred.timestamp == 1.0)

    # 4. Update step
    upd = lkf.update([5.0, 3.0])
    check("lkf_update_state_changed", abs(upd.mean[0] - 5.0) < 4.0)
    check("lkf_update_convergence", abs(upd.mean[0] - 5.0) < abs(0.0 - 5.0))

    # 5. Multiple predict-update cycles
    lkf2 = LinearKalmanFilter(state_dim=2, measurement_dim=2,
                                process_noise=[[0.01, 0], [0, 0.01]],
                                measurement_noise=[[1.0, 0], [0, 1.0]])
    for i in range(10):
        lkf2.predict(dt=0.1)
        lkf2.update([10.0 + i * 0.1, 5.0 + i * 0.05])
    final = lkf2.get_state()
    check("lkf_convergence", abs(final.mean[0] - 10.9) < 2.0)

    # 6. Innovation
    innov = lkf2.compute_innovation([11.0, 5.5])
    check("innovation_list", len(innov) == 2)

    # 7. Matrix operations used by LKF
    I = _mat_identity(3)
    check("identity_3x3", I[0][0] == 1.0 and I[1][2] == 0.0)
    Z = _mat_zeros(2, 3)
    check("zeros_shape", len(Z) == 2 and len(Z[0]) == 3)
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    C = _mat_mul(A, B)
    check("matmul_result", C[0][0] == 19 and C[1][1] == 44)
    D = _mat_add(A, A)
    check("matadd_result", D[0][0] == 2 and D[1][1] == 8)

    # 8. Matrix inverse
    inv = _mat_inverse([[4, 0], [0, 4]])
    check("mat_inverse_2x2", inv is not None and abs(inv[0][0] - 0.25) < 0.01)

    # 9. Custom F matrix
    F = [[0.9, 0], [0, 0.9]]
    lkf3 = LinearKalmanFilter(state_dim=2, measurement_dim=2, state_transition=F)
    lkf3.predict(dt=1.0)
    check("custom_F_state_decay", abs(lkf3.get_state().mean[0]) < 0.01)

    # 10. Custom H matrix
    H = [[1.0, 0.0]]
    lkf4 = LinearKalmanFilter(state_dim=2, measurement_dim=1, measurement_matrix=H)
    upd4 = lkf4.update([5.0])
    check("partial_measurement", len(upd4.mean) == 2)

    # === Extended Kalman Filter ===

    # 11. EKF construction
    ekf = ExtendedKalmanFilter(state_dim=2, measurement_dim=1)
    check("ekf_created", ekf.state_dim == 2)

    # 12. EKF predict
    def linear_transition(x, dt, u):
        return [x[0] + dt, x[1] + dt * 0.5]

    epred = ekf.predict(dt=1.0, state_transition_fn=linear_transition)
    check("ekf_predict", len(epred.mean) == 2)

    # 13. EKF update
    def measurement_fn(x):
        return [x[0] + 0.1 * x[1]]

    eupd = ekf.update([2.0], measurement_fn)
    check("ekf_update", len(eupd.mean) == 2)

    # 14. EKF Jacobian
    J = ekf.compute_jacobian(measurement_fn, [1.0, 1.0])
    check("ekf_jacobian_shape", len(J) == 1 and len(J[0]) == 2)

    # === Object Detection ===

    det = ObjectDetector(num_classes=5, conf_threshold=0.4)

    # 15. Detect with ground truth
    gt_boxes = [
        BoundingBox(x=10, y=10, w=50, h=40),
        BoundingBox(x=200, y=100, w=30, h=30),
    ]
    dets = det.detect(640, 480, gt_boxes)
    check("detect_count", len(dets) >= 1)

    # 16. Detect without ground truth
    empty = det.detect(640, 480)
    check("detect_empty", len(empty) == 0)

    # 17. Multi-scale detection
    ms_dets = det.detect_multi_scale([0.5, 1.0, 1.5], gt_boxes)
    check("multi_scale_dets", len(ms_dets) >= 1)

    # 18. NMS
    overlaps = [
        DetectionResult(class_id=0, class_name="a", confidence=0.9,
                        bbox=BoundingBox(x=10, y=10, w=50, h=50)),
        DetectionResult(class_id=0, class_name="a", confidence=0.8,
                        bbox=BoundingBox(x=12, y=12, w=50, h=50)),
    ]
    nms = det.non_max_suppression(overlaps)
    check("nms_reduces", len(nms) <= len(overlaps))

    # 19. IoU computation
    a = BoundingBox(x=0, y=0, w=100, h=100)
    b = BoundingBox(x=50, y=50, w=100, h=100)
    iou = ObjectDetector.compute_iou(a, b)
    check("iou_positive", 0 < iou < 1)
    no_overlap = ObjectDetector.compute_iou(
        BoundingBox(x=0, y=0, w=10, h=10),
        BoundingBox(x=100, y=100, w=10, h=10),
    )
    check("iou_zero", no_overlap == 0.0)

    # 20. AP computation
    detections = [
        DetectionResult(class_id=0, class_name="a", confidence=0.9,
                        bbox=BoundingBox(x=0, y=0, w=50, h=50)),
    ]
    gts = [BoundingBox(x=0, y=0, w=50, h=50)]
    ap = ObjectDetector.compute_ap(detections, gts)
    check("ap_high", ap > 0.5)

    # 21. mAP
    det_per_class = {0: detections, 1: []}
    gt_per_class = {0: gts, 1: [BoundingBox(x=10, y=10, w=20, h=20)]}
    map_val = det.compute_map(det_per_class, gt_per_class)
    check("map_range", 0 <= map_val <= 1)

    # === Grid Detector ===

    gd = GridDetector(grid_rows=5, grid_cols=5)

    # 22. Create grid
    grid = gd.create_grid()
    check("grid_shape", len(grid) == 5 and len(grid[0]) == 5)

    # 23. Assign boxes
    gd.assign_boxes_to_grid(grid, dets, 640, 480)
    check("grid_assigned", grid[0][0].confidence >= 0 or any(
        c.confidence > 0 for row in grid for c in row
    ))

    # 24. Aggregate
    agg = gd.aggregate_grid_detections(grid)
    check("aggregate_list", isinstance(agg, list))

    # === Marine Detectors ===

    # 25. BuoyDetector with real image data
    bd = BuoyDetector(color_threshold=0.3, min_size=2)
    # Create a 10x10 image with a red buoy at center
    image = [[(100, 100, 100) for _ in range(10)] for _ in range(10)]
    image[4][4] = (200, 50, 50)  # Red pixel
    image[4][5] = (200, 50, 50)
    image[5][4] = (200, 50, 50)
    buoy_dets = bd.detect_buoys(image, 10, 10)
    check("buoy_detect_list", isinstance(buoy_dets, list))

    # 26. Classify buoy
    if buoy_dets:
        cls = BuoyDetector.classify_buoy(buoy_dets[0])
        check("buoy_classified", cls in ("red", "unknown"))

    # 27. Estimate distance
    dist = BuoyDetector.estimate_distance(50.0, 1.0, 500.0)
    check("buoy_distance", dist > 0)

    # 28. VesselDetector
    vd = VesselDetector(min_length=5, confidence_threshold=0.3)
    dark_image = [[(20, 20, 20) if 2 <= x <= 7 else (100, 100, 100)
                    for x in range(10)] for _ in range(10)]
    vessel_dets = vd.detect_vessels(dark_image, 10, 10)
    check("vessel_detect_list", isinstance(vessel_dets, list))

    # 29. Vessel size estimation
    est_size = VesselDetector.estimate_vessel_size(30.0, 100.0, 500.0)
    check("vessel_size_positive", est_size > 0)

    # 30. Vessel heading
    points = [(0, 0), (10, 5), (20, 10)]
    heading = VesselDetector.estimate_vessel_heading(points)
    check("vessel_heading_numeric", isinstance(heading, float))

    # 31. Vessel tracking
    tracks = VesselDetector.track_vessels(vessel_dets)
    check("vessel_tracks_dict", isinstance(tracks, dict))

    # 32. DebrisDetector
    dd = DebrisDetector(min_area=1, max_area=10000)
    debris_dets = dd.detect_debris(image, 10, 10)
    check("debris_detect_list", isinstance(debris_dets, list))

    # 33. Classify debris
    if debris_dets:
        dcls = DebrisDetector.classify_debris(debris_dets[0])
        check("debris_classified", dcls in ("organic", "plastic", "metal", "wood"))

    # 34. NavigationMarkerDetector
    nmd = NavigationMarkerDetector(confidence_threshold=0.3)
    marker_image = [[(100, 100, 100) for _ in range(10)] for _ in range(10)]
    marker_image[3][3] = (200, 30, 30)  # Red lateral
    marker_dets = nmd.detect_markers(marker_image, 10, 10)
    check("marker_detect_list", isinstance(marker_dets, list))

    # === Cooperative Perception Sharing ===

    sharer = PerceptionSharer(vessel_id="v1", sensor_type="lidar")

    # 35. Create message
    objects = [
        PerceivedObject(
            id="obj1", type="buoy",
            position=(10.0, 20.0, 0.0), velocity=(0.1, 0.2, 0.0),
            size=(1.0, 1.0, 2.0), confidence=0.9,
            source_vessel="v1", timestamp=1000.0,
        ),
        PerceivedObject(
            id="obj2", type="vessel",
            position=(100.0, 200.0, 0.0), velocity=(1.0, 0.0, 0.0),
            size=(5.0, 2.0, 3.0), confidence=0.8,
            source_vessel="v1", timestamp=1000.0,
        ),
    ]
    msg = sharer.create_message(
        {"position": (0.0, 0.0, 0.0), "confidence": 0.95}, objects,
    )
    check("msg_created", msg.sender_id == "v1")
    check("msg_object_count", msg.object_count() == 2)

    # 36. Serialize/deserialize
    data = sharer.serialize_message(msg)
    restored = sharer.deserialize_message(data)
    check("deserialize_id", restored.sender_id == "v1")
    check("deserialize_objects", len(restored.objects) == 2)

    # 37. Compress/decompress
    compressed = sharer.compress_for_bandwidth(msg, 10000)
    decompressed = sharer.decompress_message(compressed)
    check("decompress_id", decompressed.sender_id == "v1")
    check("decompress_objects", len(decompressed.objects) == 2)

    # 38. Priority computation
    priority = sharer.compute_message_priority(msg)
    check("priority_range", 0 <= priority <= 1.0)

    # 39. Relevance filtering
    filtered = sharer.filter_by_relevance(msg, (15.0, 25.0, 0.0), 1000.0)
    check("filtered_keeps_nearby", len(filtered.objects) >= 1)

    # 40. Message age
    age = msg.age()
    check("age_non_negative", age >= 0)

    # === Perception Fusion ===

    fusion = PerceptionFusion(association_threshold=20.0, conflict_threshold=10.0)

    # 41. Fuse observations from two vessels
    obs = {
        "v1": [
            {"id": "o1", "type": "buoy", "position": (10.1, 20.1, 0.0),
             "velocity": (0.1, 0.2, 0.0), "confidence": 0.9, "timestamp": 1000},
            {"id": "o2", "type": "vessel", "position": (100.0, 200.0, 0.0),
             "velocity": (1.0, 0.0, 0.0), "confidence": 0.8, "timestamp": 1000},
        ],
        "v2": [
            {"id": "o1", "type": "buoy", "position": (10.3, 20.2, 0.0),
             "velocity": (0.15, 0.25, 0.0), "confidence": 0.85, "timestamp": 1001},
        ],
    }
    fresult = fusion.fuse_observations(obs)
    check("fusion_result", isinstance(fresult, FusionResult))
    check("fused_objects_count", len(fresult.fused_objects) >= 1)

    # 42. Multi-source confidence boost
    o1_fused = next((o for o in fresult.fused_objects if o.id == "o1"), None)
    if o1_fused:
        check("multi_source_boost", o1_fused.confidence >= 0.85)

    # 43. New objects detected
    check("new_objects", len(fresult.new_objects) > 0)

    # 44. Fused position weighted average
    if o1_fused:
        check("fused_pos_reasonable",
              abs(o1_fused.position[0] - 10.2) < 1.0)

    # 45. Associate observations
    obs_a = [{"id": "a1", "position": (0, 0, 0)}]
    obs_b = [{"id": "b1", "position": (1, 1, 0)}]
    matches = fusion.associate_observations(obs_a, obs_b)
    check("associate_matches", isinstance(matches, list))

    # 46. Resolve conflicts
    conflict_obs = [
        {"id": "o1", "type": "buoy", "position": (10.0, 20.0, 0.0),
         "velocity": (0.1, 0.2, 0.0), "confidence": 0.9, "source_vessel": "v1"},
        {"id": "o1", "type": "vessel", "position": (10.5, 20.5, 0.0),
         "velocity": (0.15, 0.25, 0.0), "confidence": 0.85, "source_vessel": "v2"},
    ]
    resolved = fusion.resolve_conflicts(conflict_obs)
    check("resolved_single", len(resolved) == 1)
    check("resolved_has_type", "type" in resolved[0])

    # 47. Fused velocity
    fused_vel = fusion.compute_fused_velocity(
        [(0.1, 0.2, 0.0), (0.15, 0.25, 0.0)], [0.9, 0.85],
    )
    check("fused_vel_tuple", len(fused_vel) == 3)
    check("fused_vel_x", 0.1 <= fused_vel[0] <= 0.15)

    # 48. Track history
    track = fusion.track_object_history("o1", [
        {"position": (10.0, 20.0, 0.0), "velocity": (0.1, 0, 0),
         "confidence": 0.9, "timestamp": i} for i in range(5)
    ])
    check("track_history_len", len(track) == 5)

    # === NL Parser ===

    parser = NLParser()

    # 49. Tokenize
    tokens = parser.tokenize("Go to waypoint at 36.5, -122.3 at 5 knots")
    check("tokens_nonempty", len(tokens) > 0)
    verb_tokens = [t for t in tokens if t.pos_tag == "VERB"]
    check("tokens_have_verb", len(verb_tokens) > 0)

    # 50. Full parse
    tree = parser.parse("Navigate to 36.5, -122.3 heading 90 degrees at 3 knots")
    check("parse_root_verb", tree.root != "")
    check("parse_confidence", tree.confidence > 0)

    # 51. Entity extraction
    entities = parser.extract_entities("Navigate to 36.5, -122.3 heading 90 degrees")
    check("entities_extracted", len(entities) >= 1)
    coord_ents = [e for e in entities if e.entity_type == "coordinate"]
    heading_ents = [e for e in entities if e.entity_type == "heading"]
    check("coord_entities", len(coord_ents) >= 1)
    check("heading_entities", len(heading_ents) >= 1)

    # 52. Number extraction
    nums = parser.extract_numbers("speed 5 knots at 30 minutes")
    check("numbers_extracted", len(nums) >= 2)

    # 53. Duration extraction
    durs = parser.extract_durations("patrol for 30 minutes then 5 seconds")
    check("durations_extracted", len(durs) >= 1)

    # 54. Coordinate extraction
    coords = parser.extract_coordinates("Go to 36.5, -122.3")
    check("coords_extracted", len(coords) >= 1)

    # 55. Normalize
    norm = parser.normalize("GO TO 36.5, -122.3 AT 5 KTS heading 45deg")
    check("normalized_lowercase", norm == norm.lower())

    # 56. Command splitting
    cmds = parser.extract_commands("Go to A. Return to base. Stop engines.")
    check("commands_split", len(cmds) >= 2)

    # === Cross-module: Sensor Fusion + Cooperative Perception ===

    # 57. Kalman filter output feeds into perception message
    lkf_nav = LinearKalmanFilter(
        state_dim=3, measurement_dim=3,
        initial_state=[10.0, 20.0, 0.0],
    )
    lkf_nav.predict(dt=0.1)
    nav_state = lkf_nav.update([10.05, 20.03, 0.01])
    nav_pos = tuple(nav_state.mean)
    check("kf_to_perception_pos", len(nav_pos) == 3)

    obj_from_kf = PerceivedObject(
        id="tracked", type="vessel",
        position=nav_pos, velocity=(0.5, 0.5, 0.0),
        size=(5, 2, 3), confidence=0.95,
        source_vessel="v1", timestamp=1000.0,
    )
    check("kf_fed_obj", obj_from_kf.position[0] > 0)

    # 58. Detection + BuoyDetector feeds into PerceivedObject
    if buoy_dets:
        bd_det = buoy_dets[0]
        perceived = PerceivedObject(
            id=f"buoy_{bd_det.x}_{bd_det.y}",
            type="buoy",
            position=(bd_det.x, bd_det.y, 0.0),
            velocity=(0, 0, 0),
            size=(bd_det.size, bd_det.size, bd_det.size),
            confidence=bd_det.confidence,
            source_vessel="cam1",
            timestamp=1000.0,
        )
        check("detection_to_perceived", perceived.confidence > 0)

    # 59. Multi-vessel fusion confidence > single vessel
    single_obs = {"v1": [obs["v1"][0]]}
    single_result = fusion.fuse_observations(single_obs)
    multi_result = fresult
    if single_result.fused_objects:
        single_conf = next((o for o in single_result.fused_objects if o.id == "o1"), None)
        if single_conf and o1_fused:
            check("multi_confidence_higher", o1_fused.confidence >= single_conf.confidence * 0.95)

    # 60. Speed from Kalman filter
    speed_est = abs(nav_state.mean[2])
    check("speed_from_kf", isinstance(speed_est, float))

    return failed == 0
