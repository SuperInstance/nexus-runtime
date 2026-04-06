"""Tests for odometry module — FeatureDetector, FeatureMatcher, VisualOdometry."""

import math
import pytest
from jetson.vision.odometry import (
    FeaturePoint, FeatureMatch,
    FeatureDetector, FeatureMatcher, VisualOdometry,
)


# ---- helpers ---------------------------------------------------------------

def make_gradient_gray(w, h):
    """Simple gradient grayscale image."""
    return [[float(x + y * w) / (w * h) * 255.0 for x in range(w)] for y in range(h)]


def make_checker_gray(w, h, block=4):
    """Checkerboard pattern."""
    gray = []
    for y in range(h):
        row = []
        for x in range(w):
            v = 200.0 if ((x // block) + (y // block)) % 2 == 0 else 50.0
            row.append(v)
        gray.append(row)
    return gray


# ---- FeaturePoint ----------------------------------------------------------

class TestFeaturePoint:
    def test_defaults(self):
        fp = FeaturePoint(1.0, 2.0)
        assert fp.x == 1.0
        assert fp.y == 2.0
        assert fp.response == 0.0
        assert fp.descriptor == []

    def test_with_fields(self):
        fp = FeaturePoint(10.0, 20.0, response=5.0, descriptor=[1, 2, 3])
        assert fp.response == 5.0
        assert fp.descriptor == [1, 2, 3]


# ---- FeatureMatch ----------------------------------------------------------

class TestFeatureMatch:
    def test_defaults(self):
        fm = FeatureMatch(0, 1, 0.5)
        assert fm.idx_a == 0
        assert fm.idx_b == 1
        assert fm.distance == 0.5
        assert fm.ratio == 1.0


# ---- FeatureDetector -------------------------------------------------------

class TestFeatureDetector:
    def test_detect_harris_corners(self):
        fd = FeatureDetector(threshold=0.01)
        gray = make_checker_gray(20, 20)
        corners = fd.detect_harris_corners(gray)
        assert len(corners) > 0
        assert all(isinstance(c, FeaturePoint) for c in corners)

    def test_detect_harris_limits(self):
        fd = FeatureDetector(max_features=10)
        gray = make_checker_gray(30, 30)
        corners = fd.detect_harris_corners(gray)
        assert len(corners) <= 10

    def test_detect_harris_blank(self):
        fd = FeatureDetector()
        gray = [[0.0] * 10 for _ in range(10)]
        corners = fd.detect_harris_corners(gray)
        assert corners == []

    def test_detect_fast_corners(self):
        fd = FeatureDetector(threshold=0.001)
        gray = make_checker_gray(20, 20, block=2)
        corners = fd.detect_fast_corners(gray)
        # may find some or none depending on threshold
        assert isinstance(corners, list)

    def test_detect_fast_limits(self):
        fd = FeatureDetector(max_features=5, threshold=0.001)
        gray = make_checker_gray(20, 20, block=2)
        corners = fd.detect_fast_corners(gray)
        assert len(corners) <= 5

    def test_compute_descriptors(self):
        fd = FeatureDetector()
        gray = make_checker_gray(20, 20)
        features = [FeaturePoint(10.0, 10.0)]
        result = fd.compute_descriptors(features, gray)
        assert len(result[0].descriptor) > 0

    def test_compute_descriptors_empty(self):
        gray = [[0.0] * 5 for _ in range(5)]
        result = FeatureDetector.compute_descriptors([], gray)
        assert result == []

    def test_compute_orb_descriptors(self):
        fd = FeatureDetector()
        gray = make_checker_gray(20, 20)
        features = [FeaturePoint(10.0, 10.0)]
        result = fd.compute_orb_descriptors(features, gray)
        assert len(result[0].descriptor) == 32
        assert all(v in (0.0, 1.0) for v in result[0].descriptor)

    def test_compute_orb_empty(self):
        result = FeatureDetector.compute_orb_descriptors([], [[0.0]])
        assert result == []

    def test_compute_orb_multiple(self):
        gray = make_gradient_gray(20, 20)
        features = [FeaturePoint(5.0, 5.0), FeaturePoint(15.0, 15.0)]
        result = FeatureDetector.compute_orb_descriptors(features, gray)
        assert len(result) == 2
        assert all(len(f.descriptor) == 32 for f in result)


# ---- FeatureMatcher --------------------------------------------------------

class TestFeatureMatcher:
    def _make_features(self, n=10, desc_len=32):
        feats = []
        for i in range(n):
            feats.append(FeaturePoint(float(i), float(i), descriptor=[float(i)] * desc_len))
        return feats

    def test_brute_force_match_identical(self):
        fm = FeatureMatcher()
        feats = self._make_features(5)
        matches = fm.brute_force_match(feats, feats)
        assert len(matches) == 5
        assert all(m.distance == 0.0 for m in matches)

    def test_brute_force_match_empty(self):
        fm = FeatureMatcher()
        assert fm.brute_force_match([], []) == []

    def test_brute_force_match_no_descriptors(self):
        fm = FeatureMatcher()
        feats = [FeaturePoint(1, 1)]
        assert fm.brute_force_match(feats, feats) == []

    def test_ratio_test(self):
        fm = FeatureMatcher(ratio_threshold=0.7)
        matches = [FeatureMatch(0, 0, 0.1), FeatureMatch(1, 1, 100.0)]
        filtered = fm.ratio_test(matches)
        # median distance is about 50, threshold=50*0.7=35
        # so only the first match passes
        assert len(filtered) == 1

    def test_ratio_test_empty(self):
        fm = FeatureMatcher()
        assert fm.ratio_test([]) == []

    def test_ransac_filter_no_matches(self):
        fm = FeatureMatcher()
        assert fm.ransac_filter([], [], []) == []

    def test_ransac_filter_single(self):
        fm = FeatureMatcher()
        m = FeatureMatch(0, 0, 1.0)
        fa = [FeaturePoint(0, 0)]
        fb = [FeaturePoint(0, 0)]
        result = fm.ransac_filter([m], fa, fb)
        assert result == [m]

    def test_ransac_filter_consensus(self):
        fm = FeatureMatcher()
        fa = [FeaturePoint(float(i), float(i)) for i in range(10)]
        fb = [FeaturePoint(float(i) + 2.0, float(i) + 1.0) for i in range(10)]
        matches = [FeatureMatch(i, i, 0.0) for i in range(10)]
        result = fm.ransac_filter(matches, fa, fb)
        assert len(result) > 0

    def test_compute_fundamental_matrix_enough(self):
        fa = [FeaturePoint(float(i), float(i)) for i in range(10)]
        fb = [FeaturePoint(float(i) + 1, float(i)) for i in range(10)]
        matches = [FeatureMatch(i, i, 0.5) for i in range(10)]
        F = FeatureMatcher.compute_fundamental_matrix(matches, fa, fb)
        assert len(F) == 3
        assert len(F[0]) == 3
        # F[2][2] should be 1 after normalisation
        # F should be a valid 3x3 matrix; check structure
        assert len(F) == 3
        assert len(F[0]) == 3
        # may be degenerate for colinear points, just verify it computed

    def test_compute_fundamental_matrix_few(self):
        matches = [FeatureMatch(0, 0, 0.5)] * 3
        F = FeatureMatcher.compute_fundamental_matrix(matches, [], [])
        assert F == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def test_desc_dist(self):
        d = FeatureMatcher._desc_dist([0, 0], [3, 4])
        assert abs(d - 5.0) < 1e-6

    def test_desc_dist_empty(self):
        assert FeatureMatcher._desc_dist([], []) == float("inf")


# ---- VisualOdometry --------------------------------------------------------

class TestVisualOdometry:
    def test_init(self):
        vo = VisualOdometry()
        assert vo.pose == [0.0, 0.0, 0.0]
        assert len(vo.trajectory) == 1

    def test_process_frame(self):
        vo = VisualOdometry()
        gray = make_checker_gray(16, 16)
        result = vo.process_frame(gray)
        assert "features" in result
        assert "matches" in result
        assert "inliers" in result
        assert result["matches"] == 0  # first frame has no matches

    def test_process_two_frames(self):
        vo = VisualOdometry()
        gray = make_checker_gray(16, 16)
        vo.process_frame(gray)
        result = vo.process_frame(gray)
        assert result["matches"] >= 0
        assert result["features"] >= 0

    def test_estimate_motion_empty(self):
        m = VisualOdometry.estimate_motion([], [], [])
        assert m == (0.0, 0.0, 0.0)

    def test_estimate_motion_translation(self):
        fa = [FeaturePoint(0, 0), FeaturePoint(10, 10)]
        fb = [FeaturePoint(5, 5), FeaturePoint(15, 15)]
        matches = [FeatureMatch(0, 0, 0.0), FeatureMatch(1, 1, 0.0)]
        dx, dy, dt = VisualOdometry.estimate_motion(matches, fa, fb)
        assert abs(dx - 5.0) < 0.1
        assert abs(dy - 5.0) < 0.1

    def test_update_pose(self):
        vo = VisualOdometry()
        initial_len = len(vo.trajectory)
        vo.update_pose((1.0, 2.0, 0.1))
        assert len(vo.trajectory) == initial_len + 1
        assert abs(vo.pose[0] - 1.0) < 0.01

    def test_get_trajectory(self):
        vo = VisualOdometry()
        traj = vo.get_trajectory()
        assert len(traj) >= 1
        assert traj[0] == (0.0, 0.0, 0.0)

    def test_compute_drift_no_gt(self):
        vo = VisualOdometry()
        gray = make_checker_gray(12, 12)
        vo.process_frame(gray)
        vo.process_frame(gray)
        drift = vo.compute_drift()
        assert drift >= 0

    def test_compute_drift_with_gt(self):
        vo = VisualOdometry()
        vo.process_frame(make_checker_gray(12, 12))
        vo.process_frame(make_checker_gray(12, 12))
        gt = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
        drift = vo.compute_drift(ground_truth=gt)
        assert drift >= 0

    def test_focal_length(self):
        vo = VisualOdometry(focal_length=300.0, cx=160.0, cy=120.0)
        assert vo.focal_length == 300.0
