"""Tests for sonar module — SonarPing, SonarScan, preprocessing, segmentation, mapping."""

import pytest
from jetson.vision.sonar import (
    SonarPing, SonarScan,
    SonarPreprocessor, SonarSegmenter, SonarMapper, SonarSegment,
)


# ---- helpers ---------------------------------------------------------------

def make_scan(n_pings=36, max_range=50.0) -> SonarScan:
    """Create a synthetic sonar scan."""
    import math
    pings = []
    for i in range(n_pings):
        angle = i * (360.0 / n_pings)
        for r in range(1, int(max_range)):
            intensity = max(0, 200 - r * 3)  # decreasing with range
            if r == 25:  # add a return at 25m
                intensity = 250
            pings.append(SonarPing(range_m=float(r), angle_deg=angle,
                                   intensity=float(intensity)))
    return SonarScan(pings=pings, timestamp=1.0, heading_deg=0.0, resolution=1.0)


def make_noise_scan(n=10) -> SonarScan:
    import random
    rng = random.Random(42)
    pings = [SonarPing(range_m=float(i), angle_deg=0.0,
                       intensity=rng.uniform(0, 255)) for i in range(n)]
    return SonarScan(pings=pings)


# ---- SonarPing -------------------------------------------------------------

class TestSonarPing:
    def test_fields(self):
        p = SonarPing(range_m=10.0, angle_deg=45.0, intensity=128.0)
        assert p.range_m == 10.0
        assert p.angle_deg == 45.0
        assert p.intensity == 128.0
        assert p.timestamp == 0.0

    def test_timestamp(self):
        p = SonarPing(5.0, 0.0, 50.0, timestamp=2.5)
        assert p.timestamp == 2.5


# ---- SonarScan -------------------------------------------------------------

class TestSonarScan:
    def test_empty(self):
        s = SonarScan()
        assert s.max_range == 0.0
        assert s.min_range == 0.0

    def test_max_range(self):
        pings = [SonarPing(10, 0, 100), SonarPing(50, 10, 200)]
        s = SonarScan(pings=pings)
        assert s.max_range == 50.0
        assert s.min_range == 10.0

    def test_properties(self):
        s = SonarScan(pings=[], timestamp=1.0, heading_deg=45.0, resolution=0.5)
        assert s.timestamp == 1.0
        assert s.heading_deg == 45.0
        assert s.resolution == 0.5


# ---- SonarPreprocessor -----------------------------------------------------

class TestSonarPreprocessor:
    def test_remove_noise(self):
        sp = SonarPreprocessor(noise_threshold=5.0)
        scan = make_noise_scan()
        result = sp.remove_noise(scan)
        assert isinstance(result, SonarScan)
        assert len(result.pings) == len(scan.pings)

    def test_remove_noise_empty(self):
        sp = SonarPreprocessor()
        result = sp.remove_noise(SonarScan(pings=[]))
        assert len(result.pings) == 0

    def test_remove_noise_properties(self):
        sp = SonarPreprocessor()
        scan = make_noise_scan()
        result = sp.remove_noise(scan)
        assert result.timestamp == scan.timestamp
        assert result.heading_deg == scan.heading_deg

    def test_apply_gain(self):
        sp = SonarPreprocessor(gain=2.0)
        scan = make_noise_scan()
        result = sp.apply_gain(scan)
        assert len(result.pings) == len(scan.pings)
        # intensities should be increased (capped at 255)
        assert all(p.intensity <= 255.0 for p in result.pings)

    def test_apply_gain_empty(self):
        sp = SonarPreprocessor()
        result = sp.apply_gain(SonarScan(pings=[]))
        assert len(result.pings) == 0

    def test_detect_returns(self):
        sp = SonarPreprocessor()
        pings = [SonarPing(10, 0, 5), SonarPing(20, 0, 50), SonarPing(30, 0, 100)]
        scan = SonarScan(pings=pings)
        returns = sp.detect_returns(scan, threshold=20)
        assert len(returns) == 2

    def test_detect_returns_empty(self):
        sp = SonarPreprocessor()
        assert sp.detect_returns(SonarScan(), threshold=20) == []

    def test_detect_returns_none_above(self):
        sp = SonarPreprocessor()
        pings = [SonarPing(10, 0, 5), SonarPing(20, 0, 10)]
        scan = SonarScan(pings=pings)
        assert sp.detect_returns(scan, threshold=100) == []

    def test_sidelobe_suppression(self):
        sp = SonarPreprocessor(sidelobe_threshold=0.3)
        pings = [SonarPing(10, 0, 200), SonarPing(20, 0, 10)]
        scan = SonarScan(pings=pings)
        result = sp.sidelobe_suppression(scan)
        assert result.pings[1].intensity == 0.0

    def test_sidelobe_suppression_empty(self):
        sp = SonarPreprocessor()
        result = sp.sidelobe_suppression(SonarScan())
        assert len(result.pings) == 0

    def test_sidelobe_suppression_all_weak(self):
        sp = SonarPreprocessor(sidelobe_threshold=0.9)
        pings = [SonarPing(10, 0, 50), SonarPing(20, 0, 50)]
        scan = SonarScan(pings=pings)
        result = sp.sidelobe_suppression(scan)
        # max=50, each ratio=1.0 >= 0.9 → none suppressed
        assert all(p.intensity == 50.0 for p in result.pings)

    def test_sidelobe_suppression_zero_max(self):
        sp = SonarPreprocessor()
        scan = SonarScan(pings=[SonarPing(10, 0, 0), SonarPing(20, 0, 0)])
        result = sp.sidelobe_suppression(scan)
        assert all(p.intensity == 0.0 for p in result.pings)


# ---- SonarSegmenter --------------------------------------------------------

class TestSonarSegmenter:
    def test_segment_scan(self):
        ss = SonarSegmenter(min_segment_size=3)
        pings = [SonarPing(10, 0, 100)] * 10 + [SonarPing(10, 0, 0)] * 5 + \
                [SonarPing(10, 0, 80)] * 8
        scan = SonarScan(pings=pings)
        segments = ss.segment_scan(scan)
        assert len(segments) >= 2

    def test_segment_scan_empty(self):
        ss = SonarSegmenter()
        assert ss.segment_scan(SonarScan()) == []

    def test_segment_scan_no_strong(self):
        ss = SonarSegmenter(min_segment_size=5)
        pings = [SonarPing(10, 0, 0)] * 20
        scan = SonarScan(pings=pings)
        assert ss.segment_scan(scan) == []

    def test_merge_segments(self):
        ss = SonarSegmenter()
        seg1 = SonarSegment(pings=[SonarPing(10, 0, 100), SonarPing(10, 1, 100)])
        seg2 = SonarSegment(pings=[SonarPing(10, 2, 100), SonarPing(10, 3, 100)])
        merged = ss.merge_segments([seg1, seg2], gap_threshold=5)
        assert len(merged) == 1

    def test_merge_segments_far(self):
        ss = SonarSegmenter()
        seg1 = SonarSegment(pings=[SonarPing(10, 0, 100)])
        seg2 = SonarSegment(pings=[SonarPing(10, 50, 100)])
        merged = ss.merge_segments([seg1, seg2], gap_threshold=5)
        assert len(merged) == 2

    def test_merge_segments_empty(self):
        ss = SonarSegmenter()
        assert ss.merge_segments([]) == []
        assert ss.merge_segments([SonarSegment()]) == [SonarSegment()]

    def test_classify_segment_bottom(self):
        ss = SonarSegmenter()
        pings = [SonarPing(5, 0, 200)] * 10
        seg = SonarSegment(pings=pings)
        label, conf = ss.classify_segment(seg)
        assert label == "bottom"

    def test_classify_segment_empty(self):
        ss = SonarSegmenter()
        label, conf = ss.classify_segment(SonarSegment())
        assert label == "unknown"
        assert conf == 0.0

    def test_compute_segment_features(self):
        ss = SonarSegmenter()
        pings = [SonarPing(10, 0, 100), SonarPing(15, 5, 150), SonarPing(20, 10, 200)]
        seg = SonarSegment(pings=pings)
        feats = ss.compute_segment_features(seg)
        assert "mean_intensity" in feats
        assert "max_intensity" in feats
        assert "mean_range" in feats
        assert "size" in feats
        assert feats["size"] == 3

    def test_compute_segment_features_empty(self):
        ss = SonarSegmenter()
        assert ss.compute_segment_features(SonarSegment()) == {}


# ---- SonarMapper -----------------------------------------------------------

class TestSonarMapper:
    def test_init(self):
        sm = SonarMapper(grid_size=50, cell_size=1.0)
        assert len(sm.grid) == 50
        assert sm.grid[25][25] == 0.5

    def test_integrate_scan(self):
        sm = SonarMapper(grid_size=100, cell_size=0.5)
        scan = make_scan(n_pings=36, max_range=30.0)
        sm.integrate_scan(scan)
        # grid should have some non-0.5 values
        changed = sum(1 for row in sm.grid for v in row if abs(v - 0.5) > 0.01)
        assert changed > 0

    def test_integrate_scan_empty(self):
        sm = SonarMapper()
        sm.integrate_scan(SonarScan())
        assert sm.grid[50][50] == 0.5

    def test_occupancy_grid_from_scans(self):
        sm = SonarMapper(grid_size=50, cell_size=1.0)
        scan = make_scan(n_pings=12, max_range=20.0)
        grid = sm.occupancy_grid_from_scans([scan])
        assert len(grid) == 50
        assert len(grid[0]) == 50

    def test_occupancy_grid_empty(self):
        sm = SonarMapper()
        grid = sm.occupancy_grid_from_scans([])
        assert all(v == 0.5 for row in grid for v in row)

    def test_detect_changes(self):
        sm = SonarMapper(grid_size=20, cell_size=1.0)
        old = [[0.5] * 20 for _ in range(20)]
        new = [row[:] for row in old]
        new[5][5] = 0.9
        new[5][6] = 0.1
        changes = sm.detect_changes(old, new, threshold=0.3)
        assert len(changes) == 2

    def test_detect_changes_no_change(self):
        sm = SonarMapper(grid_size=10, cell_size=1.0)
        grid = [[0.5] * 10 for _ in range(10)]
        assert sm.detect_changes(grid, grid) == []

    def test_detect_changes_empty(self):
        sm = SonarMapper()
        assert sm.detect_changes([], []) == []

    def test_estimate_object_position(self):
        seg = SonarSegment(pings=[SonarPing(25, 0, 200)])
        pos = SonarMapper.estimate_object_position(seg, (0, 0), 0)
        assert pos != (0, 0)

    def test_estimate_object_position_empty(self):
        seg = SonarSegment(pings=[])
        pos = SonarMapper.estimate_object_position(seg, (10, 10), 45)
        assert pos == (10, 10)

    def test_scan_count(self):
        sm = SonarMapper()
        sm.integrate_scan(SonarScan(pings=[SonarPing(10, 0, 100)]))
        sm.integrate_scan(SonarScan(pings=[SonarPing(10, 0, 100)]))
        assert sm._scan_count == 2
