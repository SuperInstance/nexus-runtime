"""Tests for detection module — BoundingBox, ObjectDetector, GridDetector."""

import pytest
from jetson.vision.detection import (
    BoundingBox, DetectionResult, AnchorBox, GridCell,
    ObjectDetector, GridDetector,
)


# ---- BoundingBox -----------------------------------------------------------

class TestBoundingBox:
    def test_area(self):
        bb = BoundingBox(0, 0, 10, 20)
        assert bb.area == 200.0

    def test_x2_y2(self):
        bb = BoundingBox(5, 10, 3, 4)
        assert bb.x2 == 8.0
        assert bb.y2 == 14.0

    def test_centroid(self):
        bb = BoundingBox(0, 0, 10, 10)
        assert bb.cx == 5.0
        assert bb.cy == 5.0

    def test_intersection_overlapping(self):
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(5, 5, 10, 10)
        inter = a.intersection(b)
        assert inter.x == 5.0
        assert inter.y == 5.0
        assert inter.w == 5.0
        assert inter.h == 5.0
        assert inter.area == 25.0

    def test_intersection_no_overlap(self):
        a = BoundingBox(0, 0, 5, 5)
        b = BoundingBox(10, 10, 5, 5)
        inter = a.intersection(b)
        assert inter.area == 0.0

    def test_intersection_contained(self):
        a = BoundingBox(0, 0, 20, 20)
        b = BoundingBox(5, 5, 5, 5)
        inter = a.intersection(b)
        assert inter.area == 25.0

    def test_aspect_ratio_anchor(self):
        ab = AnchorBox(100, 50)
        assert ab.aspect_ratio == 2.0


# ---- DetectionResult -------------------------------------------------------

class TestDetectionResult:
    def test_comparison(self):
        d1 = DetectionResult(0, "a", 0.9, BoundingBox(0, 0, 1, 1))
        d2 = DetectionResult(0, "b", 0.5, BoundingBox(0, 0, 1, 1))
        assert d2 < d1

    def test_fields(self):
        dr = DetectionResult(1, "cat", 0.8, BoundingBox(10, 20, 30, 40))
        assert dr.class_id == 1
        assert dr.class_name == "cat"
        assert dr.confidence == 0.8


# ---- GridCell --------------------------------------------------------------

class TestGridCell:
    def test_defaults(self):
        gc = GridCell(2, 3)
        assert gc.row == 2
        assert gc.col == 3
        assert gc.confidence == 0.0
        assert gc.boxes == []
        assert gc.class_probs == {}


# ---- ObjectDetector --------------------------------------------------------

class TestObjectDetector:
    def _make_detector(self):
        return ObjectDetector(num_classes=5, conf_threshold=0.5, nms_threshold=0.45)

    def test_detect_no_gt(self):
        det = self._make_detector()
        results = det.detect(640, 480)
        assert results == []

    def test_detect_with_gt(self):
        det = self._make_detector()
        gt = [BoundingBox(10, 20, 50, 60), BoundingBox(100, 200, 30, 40)]
        results = det.detect(640, 480, ground_truth=gt)
        assert len(results) == 2
        assert all(r.confidence >= 0.5 for r in results)

    def test_detect_multi_scale(self):
        det = self._make_detector()
        gt = [BoundingBox(50, 50, 100, 100)]
        results = det.detect_multi_scale([0.5, 1.0, 1.5], ground_truth=gt)
        assert len(results) >= 1

    def test_detect_multi_scale_empty(self):
        det = self._make_detector()
        results = det.detect_multi_scale([1.0])
        assert results == []

    def test_compute_iou_identical(self):
        bb = BoundingBox(0, 0, 10, 10)
        assert ObjectDetector.compute_iou(bb, bb) == 1.0

    def test_compute_iou_no_overlap(self):
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(20, 20, 10, 10)
        assert ObjectDetector.compute_iou(a, b) == 0.0

    def test_compute_iou_partial(self):
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(5, 5, 10, 10)
        iou = ObjectDetector.compute_iou(a, b)
        assert 0.0 < iou < 1.0

    def test_nms_empty(self):
        det = self._make_detector()
        assert det.non_max_suppression([]) == []

    def test_nms_single(self):
        det = self._make_detector()
        d = DetectionResult(0, "a", 0.9, BoundingBox(0, 0, 10, 10))
        assert len(det.non_max_suppression([d])) == 1

    def test_nms_suppresses(self):
        det = ObjectDetector(nms_threshold=0.3)
        d1 = DetectionResult(0, "a", 0.95, BoundingBox(0, 0, 10, 10))
        d2 = DetectionResult(0, "a", 0.8, BoundingBox(1, 1, 10, 10))
        out = det.non_max_suppression([d1, d2])
        assert len(out) == 1
        assert out[0].confidence == 0.95

    def test_nms_keeps_different(self):
        det = ObjectDetector(nms_threshold=0.3)
        d1 = DetectionResult(0, "a", 0.9, BoundingBox(0, 0, 10, 10))
        d2 = DetectionResult(0, "b", 0.8, BoundingBox(50, 50, 10, 10))
        out = det.non_max_suppression([d1, d2])
        assert len(out) == 2

    def test_compute_ap_no_gt(self):
        det = self._make_detector()
        ap = ObjectDetector.compute_ap([], [])
        assert ap == 1.0  # no GT, no false positives

    def test_compute_ap_no_detections(self):
        det = self._make_detector()
        gt = [BoundingBox(0, 0, 10, 10)]
        ap = ObjectDetector.compute_ap([], gt)
        assert ap == 0.0

    def test_compute_ap_perfect(self):
        gt = [BoundingBox(0, 0, 10, 10)]
        det = DetectionResult(0, "a", 0.99, BoundingBox(0, 0, 10, 10))
        ap = ObjectDetector.compute_ap([det], gt)
        assert ap > 0.8

    def test_compute_ap_bad(self):
        gt = [BoundingBox(0, 0, 10, 10)]
        det = DetectionResult(0, "a", 0.99, BoundingBox(100, 100, 10, 10))
        ap = ObjectDetector.compute_ap([det], gt)
        assert ap < 0.5

    def test_compute_map_single_class(self):
        det = self._make_detector()
        gt = [BoundingBox(0, 0, 10, 10)]
        d = DetectionResult(0, "a", 0.95, BoundingBox(0, 0, 10, 10))
        mAP = det.compute_map({0: [d]}, {0: gt})
        assert mAP > 0.5

    def test_compute_map_empty(self):
        det = self._make_detector()
        mAP = det.compute_map({}, {})
        assert mAP == 0.0

    def test_filter_by_threshold(self):
        det = ObjectDetector(conf_threshold=0.7)
        d1 = DetectionResult(0, "a", 0.9, BoundingBox(0, 0, 1, 1))
        d2 = DetectionResult(0, "b", 0.3, BoundingBox(0, 0, 1, 1))
        result = det._filter([d1, d2])
        assert len(result) == 1


# ---- GridDetector ----------------------------------------------------------

class TestGridDetector:
    def test_create_grid(self):
        gd = GridDetector(7, 9)
        grid = gd.create_grid()
        assert len(grid) == 7
        assert len(grid[0]) == 9

    def test_assign_boxes_to_grid(self):
        gd = GridDetector(10, 10)
        grid = gd.create_grid()
        dets = [
            DetectionResult(0, "a", 0.9, BoundingBox(5, 5, 10, 10)),
        ]
        gd.assign_boxes_to_grid(grid, dets, 100, 100)
        assert grid[1][1].confidence > 0

    def test_assign_empty(self):
        gd = GridDetector(5, 5)
        grid = gd.create_grid()
        gd.assign_boxes_to_grid(grid, [], 100, 100)
        assert all(cell.confidence == 0.0 for row in grid for cell in row)

    def test_compute_grid_confidence(self):
        gd = GridDetector(3, 3)
        grid = gd.create_grid()
        grid[0][0].confidence = 0.9
        conf = gd.compute_grid_confidence(grid)
        assert conf[0][0] == 0.9
        assert conf[1][1] == 0.0

    def test_aggregate_grid_detections(self):
        gd = GridDetector(5, 5)
        grid = gd.create_grid()
        d1 = DetectionResult(0, "a", 0.9, BoundingBox(10, 10, 5, 5))
        d2 = DetectionResult(1, "b", 0.3, BoundingBox(50, 50, 5, 5))
        grid[1][1].confidence = 0.9
        grid[1][1].boxes = [d1]
        grid[3][3].confidence = 0.3
        grid[3][3].boxes = [d2]
        agg = gd.aggregate_grid_detections(grid, threshold=0.5)
        assert len(agg) == 1
        assert agg[0].confidence == 0.9

    def test_aggregate_empty(self):
        gd = GridDetector(3, 3)
        grid = gd.create_grid()
        assert gd.aggregate_grid_detections(grid) == []
