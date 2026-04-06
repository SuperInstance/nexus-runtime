"""Object detection abstraction — pure Python.

Provides bounding boxes, NMS, grid-based detection, and mAP computation.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


# ---- dataclasses -----------------------------------------------------------

@dataclass
class BoundingBox:
    x: float  # top-left x
    y: float  # top-left y
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    def intersection(self, other: "BoundingBox") -> "BoundingBox":
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        return BoundingBox(x1, y1, w, h)


@dataclass
class DetectionResult:
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox

    def __lt__(self, other: "DetectionResult") -> bool:
        return self.confidence < other.confidence


@dataclass
class AnchorBox:
    w: float
    h: float

    @property
    def aspect_ratio(self) -> float:
        return self.w / max(self.h, 1e-9)


@dataclass
class GridCell:
    row: int
    col: int
    confidence: float = 0.0
    boxes: List[DetectionResult] = field(default_factory=list)
    class_probs: Dict[int, float] = field(default_factory=dict)


# ---- ObjectDetector --------------------------------------------------------

class ObjectDetector:
    """Base object detector with NMS and mAP utilities."""

    def __init__(self, num_classes: int = 20, conf_threshold: float = 0.5,
                 nms_threshold: float = 0.45):
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

    # --- simulation helpers -------------------------------------------------

    def detect(self, image_width: int, image_height: int,
               ground_truth: Optional[List[BoundingBox]] = None) -> List[DetectionResult]:
        """Simulate detection on a blank image.  When ground_truth is provided
        the simulated detector returns detections that overlap with them."""
        results: List[DetectionResult] = []
        if ground_truth:
            for i, gt in enumerate(ground_truth):
                # add noise to simulate imperfect detection
                rng = random.Random(i * 7 + 3)
                off_x = rng.uniform(-gt.w * 0.05, gt.w * 0.05)
                off_y = rng.uniform(-gt.h * 0.05, gt.h * 0.05)
                conf = rng.uniform(0.55, 0.98)
                bbox = BoundingBox(
                    max(0, gt.x + off_x), max(0, gt.y + off_y),
                    gt.w * rng.uniform(0.9, 1.1), gt.h * rng.uniform(0.9, 1.1),
                )
                results.append(DetectionResult(
                    class_id=i % self.num_classes,
                    class_name=f"class_{i % self.num_classes}",
                    confidence=conf,
                    bbox=bbox,
                ))
        return self._filter(results)

    def detect_multi_scale(self, scales: List[float],
                           ground_truth: Optional[List[BoundingBox]] = None) -> List[DetectionResult]:
        """Simulate multi-scale detection."""
        all_det: List[DetectionResult] = []
        for s in scales:
            scaled_gt = None
            if ground_truth:
                scaled_gt = [BoundingBox(g.x * s, g.y * s, g.w * s, g.h * s)
                             for g in ground_truth]
            all_det.extend(self.detect(640, 480, scaled_gt))
        return self.non_max_suppression(all_det)

    # --- NMS & IoU ----------------------------------------------------------

    @staticmethod
    def compute_iou(a: BoundingBox, b: BoundingBox) -> float:
        inter = a.intersection(b)
        inter_area = inter.area
        union_area = a.area + b.area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def non_max_suppression(self, detections: List[DetectionResult],
                            threshold: Optional[float] = None) -> List[DetectionResult]:
        thr = threshold if threshold is not None else self.nms_threshold
        if not detections:
            return []
        sorted_d = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep: List[DetectionResult] = []
        while sorted_d:
            best = sorted_d.pop(0)
            keep.append(best)
            sorted_d = [d for d in sorted_d
                        if self.compute_iou(best.bbox, d.bbox) < thr]
        return keep

    # --- AP / mAP -----------------------------------------------------------

    @staticmethod
    def compute_ap(detections: List[DetectionResult],
                   ground_truths: List[BoundingBox],
                   iou_threshold: float = 0.5) -> float:
        """Average Precision for a single class."""
        if not ground_truths:
            return 0.0 if detections else 1.0
        if not detections:
            return 0.0
        det_sorted = sorted(detections, key=lambda d: d.confidence, reverse=True)
        tp = [0] * len(det_sorted)
        fp = [0] * len(det_sorted)
        matched_gt: List[bool] = [False] * len(ground_truths)
        for i, det in enumerate(det_sorted):
            best_iou = 0.0
            best_idx = -1
            for j, gt in enumerate(ground_truths):
                iou = ObjectDetector.compute_iou(det.bbox, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_iou >= iou_threshold and not matched_gt[best_idx]:
                tp[i] = 1
                matched_gt[best_idx] = True
            else:
                fp[i] = 1
        cum_tp = []
        cum_fp = []
        run_tp = 0
        run_fp = 0
        for i in range(len(det_sorted)):
            run_tp += tp[i]
            run_fp += fp[i]
            cum_tp.append(run_tp)
            cum_fp.append(run_fp)
        precision = [ct / (ct + cf) if (ct + cf) > 0 else 0.0
                     for ct, cf in zip(cum_tp, cum_fp)]
        recall = [ct / len(ground_truths) for ct in cum_tp]
        # 11-point interpolation
        ap = 0.0
        for t in [i / 10.0 for i in range(11)]:
            p_max = 0.0
            for p, r in zip(precision, recall):
                if r >= t:
                    p_max = max(p_max, p)
            ap += p_max / 11.0
        return ap

    def compute_map(self, detections_per_class: Dict[int, List[DetectionResult]],
                    ground_truths_per_class: Dict[int, List[BoundingBox]]) -> float:
        """Mean Average Precision across classes."""
        aps = []
        all_classes = set(detections_per_class) | set(ground_truths_per_class)
        for cid in all_classes:
            aps.append(self.compute_ap(
                detections_per_class.get(cid, []),
                ground_truths_per_class.get(cid, []),
            ))
        return sum(aps) / len(aps) if aps else 0.0

    # --- internal -----------------------------------------------------------

    def _filter(self, dets: List[DetectionResult]) -> List[DetectionResult]:
        return [d for d in dets if d.confidence >= self.conf_threshold]


# ---- GridDetector ----------------------------------------------------------

class GridDetector:
    """Grid-based detection aggregation (like YOLO grid cells)."""

    def __init__(self, grid_rows: int = 13, grid_cols: int = 13):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

    def create_grid(self) -> List[List[GridCell]]:
        return [
            [GridCell(r, c) for c in range(self.grid_cols)]
            for r in range(self.grid_rows)
        ]

    def assign_boxes_to_grid(self, grid: List[List[GridCell]],
                             detections: List[DetectionResult],
                             image_width: int, image_height: int) -> None:
        cell_w = image_width / self.grid_cols
        cell_h = image_height / self.grid_rows
        for det in detections:
            cx = det.bbox.cx
            cy = det.bbox.cy
            col = int(cx / cell_w) if cell_w > 0 else 0
            row = int(cy / cell_h) if cell_h > 0 else 0
            col = min(col, self.grid_cols - 1)
            row = min(row, self.grid_rows - 1)
            grid[row][col].confidence = max(grid[row][col].confidence, det.confidence)
            grid[row][col].boxes.append(det)
            grid[row][col].class_probs[det.class_id] = (
                max(grid[row][col].class_probs.get(det.class_id, 0.0), det.confidence)
            )

    def compute_grid_confidence(self, grid: List[List[GridCell]]) -> List[List[float]]:
        return [
            [cell.confidence for cell in row]
            for row in grid
        ]

    def aggregate_grid_detections(self, grid: List[List[GridCell]],
                                  threshold: float = 0.25) -> List[DetectionResult]:
        results: List[DetectionResult] = []
        for row in grid:
            for cell in row:
                if cell.confidence >= threshold:
                    # pick best box per cell
                    best = max(cell.boxes, key=lambda d: d.confidence) if cell.boxes else None
                    if best:
                        results.append(best)
        return results
