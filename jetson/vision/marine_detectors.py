"""Marine-specific object detectors — pure Python simulation.

Provides detectors for buoys, vessels, debris, and navigation markers.
"""

from __future__ import annotations
import math
import random
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

try:
    from statistics import mean, stdev
except ImportError:
    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0
    def stdev(xs):
        m = mean(xs)
        var = sum((x - m) ** 2 for x in xs) / len(xs) if xs else 0.0
        return math.sqrt(var)


# ---- enum ------------------------------------------------------------------

class MarineObject(Enum):
    BUOY = "buoy"
    VESSEL = "vessel"
    DEBRIS = "debris"
    NAVIGATION_MARKER = "navigation_marker"
    LAND = "land"
    PERSON = "person"
    DOCK = "dock"


@dataclass
class MarineDetection:
    obj_type: MarineObject
    confidence: float
    x: float
    y: float
    size: float = 0.0
    heading: float = 0.0
    extra: Dict = field(default_factory=dict)


# ---- helpers ---------------------------------------------------------------

def _dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _bearing(x1: float, y1: float, x2: float, y2: float) -> float:
    """Bearing from (x1,y1) to (x2,y2) in radians."""
    return math.atan2(x2 - x1, y2 - y1)


# ---- BuoyDetector ----------------------------------------------------------

class BuoyDetector:
    def __init__(self, color_threshold: float = 0.3, min_size: int = 4):
        self.color_threshold = color_threshold
        self.min_size = min_size

    def detect_buoys(self, image_data, image_width: int, image_height: int
                     ) -> List[MarineDetection]:
        """Scan image (list-of-lists of (r,g,b)) for buoy-like blobs."""
        h = len(image_data)
        w = len(image_data[0]) if h else 0
        candidates: List[MarineDetection] = []
        visited = [[False] * w for _ in range(h)]
        for y in range(0, h, 2):
            for x in range(0, w, 2):
                if visited[y][x]:
                    continue
                r, g, b = image_data[y][x]
                # buoys are typically red or orange or yellow
                is_buoy_color = (r > 150 and g < 120 and b < 80) or \
                                (r > 180 and g > 100 and b < 80) or \
                                (r > 200 and g > 180 and b < 80)
                if not is_buoy_color:
                    continue
                # flood-fill to find blob size
                region = self._flood_fill(image_data, visited, x, y, w, h)
                if len(region) >= self.min_size:
                    avg_r = mean([p[0] for p in region])
                    avg_g = mean([p[1] for p in region])
                    avg_b = mean([p[2] for p in region])
                    cx = mean([p[3] for p in region])
                    cy = mean([p[4] for p in region])
                    size = math.sqrt(len(region)) * 2
                    candidates.append(MarineDetection(
                        obj_type=MarineObject.BUOY,
                        confidence=min(1.0, len(region) / 50.0),
                        x=cx, y=cy, size=size,
                        extra={"avg_color": (avg_r, avg_g, avg_b)}
                    ))
        return candidates

    @staticmethod
    def _flood_fill(data, visited, sx, sy, w, h, max_pixels=200):
        region = []
        stack = [(sx, sy)]
        while stack and len(region) < max_pixels:
            x, y = stack.pop()
            if x < 0 or x >= w or y < 0 or y >= h or visited[y][x]:
                continue
            visited[y][x] = True
            p = data[y][x]
            region.append((p[0], p[1], p[2], float(x), float(y)))
            stack.append((x + 1, y))
            stack.append((x - 1, y))
            stack.append((x, y + 1))
            stack.append((x, y - 1))
        return region

    @staticmethod
    def classify_buoy(detection: MarineDetection) -> str:
        color = detection.extra.get("avg_color", (128, 128, 128))
        r, g, b = color
        if r > 180 and g < 100:
            return "red"
        if r > 200 and g > 200 and b > 200:
            return "white"
        if r > 200 and g > 180 and b < 180:
            return "yellow"
        if r < 80 and g > 150 and b > 150:
            return "green"
        if r > 180 and g > 100 and b < 80:
            return "orange"
        return "unknown"

    @staticmethod
    def estimate_distance(pixel_size: float, real_size: float = 1.0,
                          focal_length: float = 500.0) -> float:
        """Estimate distance in metres from pixel size of buoy."""
        if pixel_size <= 0:
            return float("inf")
        return real_size * focal_length / pixel_size

    @staticmethod
    def estimate_bearing(detection: MarineDetection, image_cx: float,
                         image_cy: float, fov_h: float = 60.0,
                         fov_v: float = 45.0) -> float:
        """Bearing in radians relative to camera centre."""
        dx = (detection.x - image_cx) / max(image_cx, 1)
        dy = (detection.y - image_cy) / max(image_cy, 1)
        azimuth = math.atan2(dx, 1.0) * (fov_h / 2.0)
        elevation = math.atan2(-dy, 1.0) * (fov_v / 2.0)
        return math.sqrt(azimuth ** 2 + elevation ** 2)


# ---- VesselDetector --------------------------------------------------------

class VesselDetector:
    def __init__(self, min_length: int = 20, confidence_threshold: float = 0.4):
        self.min_length = min_length
        self.conf_threshold = confidence_threshold

    def detect_vessels(self, image_data, image_width: int, image_height: int
                       ) -> List[MarineDetection]:
        """Simulate vessel detection — looks for large dark horizontal regions."""
        h = len(image_data)
        w = len(image_data[0]) if h else 0
        detections: List[MarineDetection] = []
        # scan rows for contiguous dark runs
        for y in range(0, h, 3):
            run_start = None
            for x in range(w):
                r, g, b = image_data[y][x]
                dark = (r + g + b) / 3 < 80
                if dark and run_start is None:
                    run_start = x
                elif (not dark or x == w - 1) and run_start is not None:
                    length = x - run_start if not dark else x - run_start + 1
                    if length >= self.min_length:
                        conf = min(0.95, length / (self.min_length * 3))
                        detections.append(MarineDetection(
                            obj_type=MarineObject.VESSEL,
                            confidence=conf,
                            x=run_start + length / 2.0,
                            y=float(y),
                            size=float(length),
                        ))
                    run_start = None
        # merge nearby detections (same-ish y)
        merged = self._merge_nearby(detections, max_dy=6)
        return [d for d in merged if d.confidence >= self.conf_threshold]

    @staticmethod
    def _merge_nearby(dets: List[MarineDetection], max_dy: float = 5.0) -> List[MarineDetection]:
        if not dets:
            return []
        dets.sort(key=lambda d: d.y)
        merged: List[MarineDetection] = [dets[0]]
        for d in dets[1:]:
            last = merged[-1]
            if abs(d.y - last.y) < max_dy:
                last.x = (last.x + d.x) / 2
                last.y = (last.y + d.y) / 2
                last.size = max(last.size, d.size)
                last.confidence = max(last.confidence, d.confidence)
            else:
                merged.append(d)
        return merged

    @staticmethod
    def estimate_vessel_size(pixel_length: float, distance: float = 100.0,
                             focal_length: float = 500.0) -> float:
        if distance <= 0:
            return 0.0
        return pixel_length * distance / focal_length

    @staticmethod
    def estimate_vessel_heading(points: List[Tuple[float, float]]) -> float:
        if len(points) < 2:
            return 0.0
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        # principal axis via least-squares
        mx, my = mean(xs), mean(ys)
        numerator = sum((x - mx) * (y - my) for x, y in points)
        denominator = sum((x - mx) ** 2 for x, y in points) + 1e-12
        slope = numerator / denominator
        return math.atan(slope)

    @staticmethod
    def track_vessels(detections: List[MarineDetection],
                      prev_positions: Optional[Dict[int, Tuple[float, float]]] = None,
                      max_distance: float = 50.0) -> Dict[int, int]:
        """Simple nearest-neighbour tracker.  Returns {det_idx: track_id}."""
        if prev_positions is None or not prev_positions or not detections:
            return {i: i for i in range(len(detections))}
        tracks = sorted(prev_positions.items(), key=lambda kv: kv[0])
        assignments: Dict[int, int] = {}
        used_tracks: set = set()
        for di, det in enumerate(detections):
            best_tid = -1
            best_dist = max_distance
            for tid, (tx, ty) in tracks:
                if tid in used_tracks:
                    continue
                d = _dist(det.x, det.y, tx, ty)
                if d < best_dist:
                    best_dist = d
                    best_tid = tid
            if best_tid >= 0:
                assignments[di] = best_tid
                used_tracks.add(best_tid)
            else:
                assignments[di] = max(prev_positions.keys(), default=-1) + 1
        return assignments


# ---- DebrisDetector --------------------------------------------------------

class DebrisDetector:
    def __init__(self, min_area: int = 6, max_area: int = 500):
        self.min_area = min_area
        self.max_area = max_area

    def detect_debris(self, image_data, image_width: int, image_height: int
                      ) -> List[MarineDetection]:
        h = len(image_data)
        w = len(image_data[0]) if h else 0
        detections: List[MarineDetection] = []
        # look for small non-uniform patches
        for y in range(2, h - 2, 4):
            for x in range(2, w - 2, 4):
                patch = []
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        patch.append(image_data[y + dy][x + dx])
                avg_r = mean([p[0] for p in patch])
                avg_g = mean([p[1] for p in patch])
                avg_b = mean([p[2] for p in patch])
                variation = stdev([p[0] for p in patch]) + \
                            stdev([p[1] for p in patch]) + \
                            stdev([p[2] for p in patch])
                brightness = (avg_r + avg_g + avg_b) / 3
                # debris: moderate brightness, moderate variation, brownish/greyish
                if 40 < brightness < 200 and variation > 15 and variation < 80:
                    area = 25  # patch size
                    if self.min_area <= area <= self.max_area:
                        detections.append(MarineDetection(
                            obj_type=MarineObject.DEBRIS,
                            confidence=min(1.0, variation / 100.0),
                            x=float(x), y=float(y), size=float(area),
                            extra={"variation": variation, "brightness": brightness}
                        ))
        return detections

    @staticmethod
    def classify_debris(detection: MarineDetection) -> str:
        variation = detection.extra.get("variation", 0)
        brightness = detection.extra.get("brightness", 128)
        if brightness < 80:
            return "organic"
        if variation > 50:
            return "plastic"
        if brightness > 160:
            return "metal"
        return "wood"

    @staticmethod
    def estimate_drift_direction(historical_positions: List[Tuple[float, float]]
                                 ) -> Tuple[float, float]:
        """Returns (drift_speed, drift_direction_rad)."""
        if len(historical_positions) < 2:
            return (0.0, 0.0)
        dx = historical_positions[-1][0] - historical_positions[0][0]
        dy = historical_positions[-1][1] - historical_positions[0][1]
        dt = len(historical_positions)  # assume unit time steps
        speed = math.sqrt(dx ** 2 + dy ** 2) / dt
        direction = math.atan2(dy, dx)
        return (speed, direction)


# ---- NavigationMarkerDetector ----------------------------------------------

class NavigationMarkerDetector:
    MARKER_PATTERNS: Dict[str, Tuple[int, int, int]] = {
        "red_lateral": (200, 30, 30),
        "green_lateral": (30, 180, 30),
        "safe_water": (200, 200, 200),
        "isolated_danger": (0, 0, 0),
        "special": (200, 200, 0),
    }

    def __init__(self, confidence_threshold: float = 0.5):
        self.conf_threshold = confidence_threshold

    def detect_markers(self, image_data, image_width: int, image_height: int
                       ) -> List[MarineDetection]:
        h = len(image_data)
        w = len(image_data[0]) if h else 0
        detections: List[MarineDetection] = []
        for y in range(1, h - 1, 3):
            for x in range(1, w - 1, 3):
                r, g, b = image_data[y][x]
                for mtype, (mr, mg, mb) in self.MARKER_PATTERNS.items():
                    dist = math.sqrt((r - mr) ** 2 + (g - mg) ** 2 + (b - mb) ** 2)
                    if dist < 60:
                        conf = max(0.0, 1.0 - dist / 60.0)
                        if conf >= self.conf_threshold:
                            detections.append(MarineDetection(
                                obj_type=MarineObject.NAVIGATION_MARKER,
                                confidence=conf,
                                x=float(x), y=float(y), size=9.0,
                                extra={"marker_type": mtype}
                            ))
                            break  # one marker per location
        return detections

    @staticmethod
    def read_marker_type(detection: MarineDetection) -> str:
        return detection.extra.get("marker_type", "unknown")

    @staticmethod
    def compute_position_fix(marker_detections: List[MarineDetection],
                             known_marker_positions: Dict[str, Tuple[float, float]],
                             image_width: int, image_height: int
                             ) -> Optional[Tuple[float, float]]:
        """Estimate vessel position from bearing cross-fix."""
        if not marker_detections or not known_marker_positions:
            return None
        cx, cy = image_width / 2.0, image_height / 2.0
        intersections: List[Tuple[float, float]] = []
        for det in marker_detections:
            mtype = det.extra.get("marker_type", "")
            if mtype not in known_marker_positions:
                continue
            mx, my = known_marker_positions[mtype]
            bearing_h = math.atan2(det.x - cx, 1.0) * (math.pi / 4)
            bearing_v = math.atan2(det.y - cy, 1.0) * (math.pi / 4)
            # simplified fix: midpoint between marker and estimated line
            est_x = mx - math.sin(bearing_h) * 100
            est_y = my - math.cos(bearing_v) * 100
            intersections.append((est_x, est_y))
        if not intersections:
            return None
        fx = mean([p[0] for p in intersections])
        fy = mean([p[1] for p in intersections])
        return (fx, fy)
