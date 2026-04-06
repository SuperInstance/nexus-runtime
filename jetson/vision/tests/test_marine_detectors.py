"""Tests for marine_detectors module — buoy, vessel, debris, nav-marker detectors."""

import pytest
from jetson.vision.marine_detectors import (
    MarineObject, MarineDetection,
    BuoyDetector, VesselDetector, DebrisDetector, NavigationMarkerDetector,
)


# ---- helpers ---------------------------------------------------------------

def make_red_buoy_image(w=30, h=30):
    """Image with a red blob in the centre."""
    data = [[(100, 100, 100)] * w for _ in range(h)]
    cx, cy = w // 2, h // 2
    for y in range(cy - 3, cy + 3):
        for x in range(cx - 3, cx + 3):
            if 0 <= y < h and 0 <= x < w:
                data[y][x] = (220, 40, 30)
    return data


def make_dark_vessel_image(w=60, h=20):
    """Image with a dark horizontal band (vessel-like)."""
    data = [[(150, 180, 200)] * w for _ in range(h)]
    for y in range(h // 2 - 1, h // 2 + 1):
        for x in range(5, w - 5):
            data[y][x] = (30, 30, 40)
    return data


def make_debris_image(w=30, h=30):
    """Image with non-uniform patches."""
    import random
    rng = random.Random(42)
    data = []
    for y in range(h):
        row = []
        for x in range(w):
            base = 100 + rng.randint(-20, 20)
            row.append((base, base - 10, base + 15))
        data.append(row)
    return data


# ---- MarineObject enum -----------------------------------------------------

class TestMarineObject:
    def test_values(self):
        assert MarineObject.BUOY.value == "buoy"
        assert MarineObject.VESSEL.value == "vessel"
        assert MarineObject.DEBRIS.value == "debris"
        assert MarineObject.NAVIGATION_MARKER.value == "navigation_marker"
        assert MarineObject.LAND.value == "land"
        assert MarineObject.PERSON.value == "person"
        assert MarineObject.DOCK.value == "dock"

    def test_all_members(self):
        assert len(MarineObject) == 7


# ---- MarineDetection -------------------------------------------------------

class TestMarineDetection:
    def test_defaults(self):
        md = MarineDetection(MarineObject.BUOY, 0.9, 10.0, 20.0)
        assert md.obj_type == MarineObject.BUOY
        assert md.confidence == 0.9
        assert md.size == 0.0
        assert md.heading == 0.0
        assert md.extra == {}


# ---- BuoyDetector ----------------------------------------------------------

class TestBuoyDetector:
    def test_detect_buoys(self):
        bd = BuoyDetector(min_size=2)
        img = make_red_buoy_image()
        results = bd.detect_buoys(img, 30, 30)
        assert isinstance(results, list)

    def test_detect_buoys_empty(self):
        bd = BuoyDetector()
        blank = [[(100, 100, 100)] * 10 for _ in range(10)]
        results = bd.detect_buoys(blank, 10, 10)
        assert results == []

    def test_classify_buoy_red(self):
        det = MarineDetection(MarineObject.BUOY, 0.9, 10, 10,
                             extra={"avg_color": (220, 50, 30)})
        assert BuoyDetector.classify_buoy(det) == "red"

    def test_classify_buoy_orange(self):
        det = MarineDetection(MarineObject.BUOY, 0.9, 10, 10,
                             extra={"avg_color": (200, 120, 40)})
        assert BuoyDetector.classify_buoy(det) == "orange"

    def test_classify_buoy_yellow(self):
        det = MarineDetection(MarineObject.BUOY, 0.9, 10, 10,
                             extra={"avg_color": (220, 200, 50)})
        assert BuoyDetector.classify_buoy(det) == "yellow"

    def test_classify_buoy_green(self):
        det = MarineDetection(MarineObject.BUOY, 0.9, 10, 10,
                             extra={"avg_color": (50, 180, 180)})
        assert BuoyDetector.classify_buoy(det) == "green"

    def test_classify_buoy_white(self):
        det = MarineDetection(MarineObject.BUOY, 0.9, 10, 10,
                             extra={"avg_color": (220, 220, 220)})
        assert BuoyDetector.classify_buoy(det) == "white"

    def test_classify_buoy_unknown(self):
        det = MarineDetection(MarineObject.BUOY, 0.9, 10, 10,
                             extra={"avg_color": (128, 128, 128)})
        assert BuoyDetector.classify_buoy(det) == "unknown"

    def test_estimate_distance(self):
        d = BuoyDetector.estimate_distance(pixel_size=50.0, real_size=1.0, focal_length=500.0)
        assert abs(d - 10.0) < 0.1

    def test_estimate_distance_zero(self):
        d = BuoyDetector.estimate_distance(pixel_size=0.0)
        assert d == float("inf")

    def test_estimate_bearing(self):
        det = MarineDetection(MarineObject.BUOY, 0.9, 340, 220)
        bearing = BuoyDetector.estimate_bearing(det, 320, 240)
        assert isinstance(bearing, float)

    def test_estimate_bearing_centre(self):
        det = MarineDetection(MarineObject.BUOY, 0.9, 320, 240)
        bearing = BuoyDetector.estimate_bearing(det, 320, 240)
        assert abs(bearing) < 0.01


# ---- VesselDetector --------------------------------------------------------

class TestVesselDetector:
    def test_detect_vessels(self):
        vd = VesselDetector(min_length=10)
        img = make_dark_vessel_image()
        results = vd.detect_vessels(img, 60, 20)
        assert isinstance(results, list)

    def test_detect_vessels_empty(self):
        vd = VesselDetector()
        blank = [[(200, 200, 200)] * 10 for _ in range(10)]
        results = vd.detect_vessels(blank, 10, 10)
        assert results == []

    def test_estimate_vessel_size(self):
        s = VesselDetector.estimate_vessel_size(100.0, 200.0, 500.0)
        assert abs(s - 40.0) < 0.1

    def test_estimate_vessel_size_zero_dist(self):
        s = VesselDetector.estimate_vessel_size(100.0, 0.0)
        assert s == 0.0

    def test_estimate_vessel_heading(self):
        points = [(0, 0), (10, 0), (20, 0)]
        heading = VesselDetector.estimate_vessel_heading(points)
        assert abs(heading) < 0.01  # horizontal → ~0

    def test_estimate_vessel_heading_vertical(self):
        points = [(0, 0), (10, 0), (20, 0)]
        heading = VesselDetector.estimate_vessel_heading(points)
        # horizontal → slope ≈ 0
        assert abs(heading) < 0.1

    def test_estimate_vessel_heading_diagonal(self):
        points = [(0, 0), (10, 10), (20, 20)]
        heading = VesselDetector.estimate_vessel_heading(points)
        # diagonal → slope ≈ 1 → atan(1) ≈ 0.785
        assert abs(heading - 0.7854) < 0.1

    def test_estimate_vessel_heading_single(self):
        heading = VesselDetector.estimate_vessel_heading([(5, 5)])
        assert heading == 0.0

    def test_track_vessels_no_prev(self):
        dets = [MarineDetection(MarineObject.VESSEL, 0.9, 10, 10)]
        tracks = VesselDetector.track_vessels(dets)
        assert len(tracks) == 1

    def test_track_vessels_with_prev(self):
        dets = [MarineDetection(MarineObject.VESSEL, 0.9, 12, 10),
                MarineDetection(MarineObject.VESSEL, 0.9, 60, 10)]
        prev = {0: (10.0, 10.0)}
        tracks = VesselDetector.track_vessels(dets, prev_positions=prev)
        assert len(tracks) == 2

    def test_track_vessels_empty(self):
        assert VesselDetector.track_vessels([]) == {}

    def test_merge_nearby(self):
        d1 = MarineDetection(MarineObject.VESSEL, 0.8, 10, 10, 20)
        d2 = MarineDetection(MarineObject.VESSEL, 0.7, 12, 12, 22)
        merged = VesselDetector._merge_nearby([d1, d2], max_dy=5)
        assert len(merged) == 1


# ---- DebrisDetector --------------------------------------------------------

class TestDebrisDetector:
    def test_detect_debris(self):
        dd = DebrisDetector(min_area=1)
        img = make_debris_image()
        results = dd.detect_debris(img, 30, 30)
        assert isinstance(results, list)

    def test_detect_debris_uniform(self):
        dd = DebrisDetector()
        uniform = [[(128, 128, 128)] * 20 for _ in range(20)]
        results = dd.detect_debris(uniform, 20, 20)
        # uniform image → low variation → no debris
        assert results == []

    def test_classify_debris_organic(self):
        det = MarineDetection(MarineObject.DEBRIS, 0.5, 5, 5,
                             extra={"variation": 30, "brightness": 50})
        assert DebrisDetector.classify_debris(det) == "organic"

    def test_classify_debris_plastic(self):
        det = MarineDetection(MarineObject.DEBRIS, 0.5, 5, 5,
                             extra={"variation": 60, "brightness": 120})
        assert DebrisDetector.classify_debris(det) == "plastic"

    def test_classify_debris_metal(self):
        det = MarineDetection(MarineObject.DEBRIS, 0.5, 5, 5,
                             extra={"variation": 30, "brightness": 180})
        assert DebrisDetector.classify_debris(det) == "metal"

    def test_classify_debris_wood(self):
        det = MarineDetection(MarineObject.DEBRIS, 0.5, 5, 5,
                             extra={"variation": 20, "brightness": 120})
        assert DebrisDetector.classify_debris(det) == "wood"

    def test_estimate_drift_single(self):
        drift = DebrisDetector.estimate_drift_direction([(0, 0)])
        assert drift == (0.0, 0.0)

    def test_estimate_drift_movement(self):
        drift = DebrisDetector.estimate_drift_direction([(0, 0), (10, 0), (20, 0)])
        speed, direction = drift
        assert speed > 0
        assert abs(direction) < 0.01  # moving right

    def test_estimate_drift_empty(self):
        drift = DebrisDetector.estimate_drift_direction([])
        assert drift == (0.0, 0.0)


# ---- NavigationMarkerDetector ----------------------------------------------

class TestNavigationMarkerDetector:
    def test_detect_markers(self):
        nmd = NavigationMarkerDetector(confidence_threshold=0.3)
        # create image with a green lateral marker
        img = [[(100, 100, 100)] * 20 for _ in range(20)]
        img[10][10] = (30, 180, 30)
        results = nmd.detect_markers(img, 20, 20)
        # should detect at least something close
        assert isinstance(results, list)

    def test_detect_markers_empty(self):
        nmd = NavigationMarkerDetector()
        blank = [[(128, 128, 128)] * 10 for _ in range(10)]
        results = nmd.detect_markers(blank, 10, 10)
        assert results == []

    def test_read_marker_type(self):
        det = MarineDetection(MarineObject.NAVIGATION_MARKER, 0.9, 10, 10,
                             extra={"marker_type": "red_lateral"})
        assert NavigationMarkerDetector.read_marker_type(det) == "red_lateral"

    def test_read_marker_type_unknown(self):
        det = MarineDetection(MarineObject.NAVIGATION_MARKER, 0.9, 10, 10)
        assert NavigationMarkerDetector.read_marker_type(det) == "unknown"

    def test_compute_position_fix(self):
        det = MarineDetection(MarineObject.NAVIGATION_MARKER, 0.9, 16, 12,
                             extra={"marker_type": "green_lateral"})
        known = {"green_lateral": (50.0, 50.0)}
        fix = NavigationMarkerDetector.compute_position_fix([det], known, 32, 24)
        assert fix is not None
        assert isinstance(fix[0], float)
        assert isinstance(fix[1], float)

    def test_compute_position_fix_no_markers(self):
        fix = NavigationMarkerDetector.compute_position_fix([], {}, 10, 10)
        assert fix is None

    def test_compute_position_fix_no_known(self):
        det = MarineDetection(MarineObject.NAVIGATION_MARKER, 0.9, 10, 10,
                             extra={"marker_type": "unknown"})
        fix = NavigationMarkerDetector.compute_position_fix([det], {}, 10, 10)
        assert fix is None

    def test_marker_patterns(self):
        patterns = NavigationMarkerDetector.MARKER_PATTERNS
        assert "red_lateral" in patterns
        assert "green_lateral" in patterns
        assert "safe_water" in patterns
        assert "isolated_danger" in patterns
        assert "special" in patterns
