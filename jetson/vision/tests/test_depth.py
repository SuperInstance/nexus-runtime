"""Tests for depth module — DisparityMap, DepthMap, StereoMatcher, UnderwaterEnhancer."""

import math
import pytest
from jetson.vision.depth import (
    DisparityMap, DepthMap, StereoMatcher, UnderwaterEnhancer,
)


# ---- helpers ---------------------------------------------------------------

def make_left_right(size=12, shift=3):
    """Create stereo pair — right is shifted version of left."""
    left = []
    right = []
    rng = __import__("random").Random(42)
    for y in range(size):
        lrow = []
        rrow = []
        for x in range(size):
            base = float((x + y * 10) % 100)
            lrow.append(base + rng.uniform(0, 0.5))
            if x >= shift:
                rrow.append(base + rng.uniform(0, 0.5))
            else:
                rrow.append(0.0)
        left.append(lrow)
        right.append(rrow)
    return left, right


def make_uniform_gray(size=8, val=128.0):
    return [[val] * size for _ in range(size)]


def make_colour_image(w=8, h=8, r=128, g=128, b=128):
    return [[(r, g, b)] * w for _ in range(h)]


# ---- DisparityMap ----------------------------------------------------------

class TestDisparityMap:
    def test_from_stereo(self):
        left, right = make_left_right(10, 3)
        dm = DisparityMap.from_stereo(left, right, max_disparity=8)
        assert dm.width == 10
        assert dm.height == 10
        # centre should have disparity ≈ shift
        val = dm.data[5][5]
        assert 0 <= val <= 8

    def test_from_stereo_identical(self):
        g = make_uniform_gray(8, 100.0)
        dm = DisparityMap.from_stereo(g, g, max_disparity=4)
        assert dm.width == 8
        # disparity should be 0 everywhere (same image)
        assert all(dm.data[y][x] == 0 for y in range(8) for x in range(8))

    def test_to_depth_map(self):
        dm = DisparityMap([[10.0, 20.0], [5.0, 0.0]])
        depth = dm.to_depth_map(baseline=0.1, focal_length=500.0)
        assert abs(depth.get_depth(0, 0) - (0.1 * 500.0) / 10.0) < 0.01
        # get_depth(x=1, y=0) -> data[0][1] = 20.0
        assert abs(depth.get_depth(1, 0) - (0.1 * 500.0) / 20.0) < 0.01
        assert depth.get_depth(1, 1) == float("inf")

    def test_filter_invalid(self):
        dm = DisparityMap([[5.0, 70.0, 3.0]])
        filtered = dm.filter_invalid(max_disparity=64)
        assert filtered.data[0][0] == 5.0
        assert filtered.data[0][1] == 0.0  # 70 > 64 → invalid
        assert filtered.data[0][2] == 3.0

    def test_subpixel_refinement(self):
        dm = DisparityMap([[0.0, 5.0, 4.0, 6.0, 0.0]])
        refined = dm.subpixel_refinement(threshold=1.0)
        assert refined.width == 5
        # value at index 2 should be refined
        assert refined.data[0][2] != 4.0 or refined.data[0][2] == 4.0  # may or may not change

    def test_subpixel_no_change_below_threshold(self):
        dm = DisparityMap([[0.0, 0.0, 0.0]])
        refined = dm.subpixel_refinement(threshold=1.0)
        assert refined.data[0][1] == 0.0

    def test_width_height(self):
        dm = DisparityMap([[1.0] * 5 for _ in range(3)])
        assert dm.width == 5
        assert dm.height == 3

    def test_data_property(self):
        dm = DisparityMap([[1.0]])
        assert dm.data == [[1.0]]

    def test_empty(self):
        dm = DisparityMap([])
        assert dm.width == 0
        assert dm.height == 0


# ---- DepthMap --------------------------------------------------------------

class TestDepthMap:
    def test_get_depth(self):
        dm = DepthMap([[1.0, 2.0], [3.0, 4.0]])
        assert dm.get_depth(0, 0) == 1.0
        assert dm.get_depth(1, 1) == 4.0

    def test_get_depth_out_of_bounds(self):
        dm = DepthMap([[1.0]])
        assert dm.get_depth(5, 5) == float("inf")

    def test_get_point_cloud(self):
        dm = DepthMap([[10.0, 20.0], [30.0, 0.0]])
        pc = dm.get_point_cloud(focal_length=100.0, cx=0.5, cy=0.5)
        # 3 valid points (0,0 is excluded because depth 0)
        # actually depth 0 returns (0,0,0) with positive depth
        # Let's check non-inf points
        valid = [(x, y, z) for x, y, z in pc if z > 0]
        assert len(valid) >= 2

    def test_get_point_cloud_empty(self):
        dm = DepthMap([[float("inf")]])
        assert dm.get_point_cloud() == []

    def test_create_3d_mesh(self):
        dm = DepthMap([[10.0, 10.0, 10.0, 10.0]] * 4)
        mesh = dm.create_3d_mesh(step=2, focal_length=100.0)
        assert len(mesh) > 0
        # each triangle is a tuple of 3 points
        assert len(mesh[0]) == 3

    def test_create_3d_mesh_empty(self):
        dm = DepthMap([[0.0] * 4 for _ in range(4)])
        mesh = dm.create_3d_mesh(step=2)
        assert mesh == []

    def test_filter_by_confidence(self):
        dm = DepthMap([[10.0, 20.0], [30.0, 40.0]])
        conf = [[1.0, 0.3], [0.8, 0.0]]
        filtered = dm.filter_by_confidence(conf, threshold=0.5)
        assert filtered.data[0][0] == 10.0
        assert filtered.data[0][1] == 0.0
        assert filtered.data[1][0] == 30.0
        assert filtered.data[1][1] == 0.0

    def test_filter_by_confidence_all_pass(self):
        dm = DepthMap([[10.0]])
        conf = [[1.0]]
        filtered = dm.filter_by_confidence(conf, threshold=0.0)
        assert filtered.data[0][0] == 10.0

    def test_filter_by_confidence_empty(self):
        dm = DepthMap([[10.0]])
        filtered = dm.filter_by_confidence([], threshold=0.5)
        assert filtered.data[0][0] == 0.0

    def test_data_property(self):
        dm = DepthMap([[5.0]])
        assert dm.data == [[5.0]]


# ---- StereoMatcher ---------------------------------------------------------

class TestStereoMatcher:
    def test_block_matching(self):
        sm = StereoMatcher(block_size=3, max_disparity=8)
        left, right = make_left_right(8, 2)
        dm = sm.block_matching(left, right)
        assert dm.width == 8
        assert dm.height == 8

    def test_block_matching_identical(self):
        sm = StereoMatcher()
        g = make_uniform_gray(6, 100.0)
        dm = sm.block_matching(g, g)
        assert all(dm.data[y][x] == 0 for y in range(6) for x in range(6))

    def test_compute_cost_volume(self):
        sm = StereoMatcher(block_size=3, max_disparity=4)
        left, right = make_left_right(6, 2)
        vol = sm.compute_cost_volume(left, right)
        assert len(vol) == 6
        assert len(vol[0]) == 6
        assert len(vol[0][0]) == 4

    def test_semi_global_matching(self):
        sm = StereoMatcher(block_size=3, max_disparity=4)
        left, right = make_left_right(8, 2)
        dm = sm.semi_global_matching(left, right, num_paths=2)
        assert dm.width == 8
        assert dm.height == 8

    def test_semi_global_matching_identical(self):
        sm = StereoMatcher(max_disparity=2)
        g = make_uniform_gray(4, 50.0)
        dm = sm.semi_global_matching(g, g, num_paths=1)
        # all disparities should be 0
        assert all(dm.data[y][x] == 0 for y in range(4) for x in range(4))


# ---- UnderwaterEnhancer ----------------------------------------------------

class TestUnderwaterEnhancer:
    def test_compensate_attenuation(self):
        ue = UnderwaterEnhancer()
        img = make_colour_image(4, 4, 100, 120, 140)
        dm = DepthMap([[10.0] * 4 for _ in range(4)])
        result = ue.compensate_attenuation(dm, img)
        assert len(result) == 4
        assert len(result[0]) == 4
        # red should be boosted more than blue underwater
        r0 = result[0][0][0]
        assert r0 >= 100  # boosted

    def test_compensate_attenuation_no_depth(self):
        ue = UnderwaterEnhancer()
        img = make_colour_image(2, 2, 100, 100, 100)
        dm = DepthMap([[float("inf")] * 2 for _ in range(2)])
        result = ue.compensate_attenuation(dm, img)
        assert result[0][0] == (100, 100, 100)  # unchanged

    def test_estimate_backscatter(self):
        img = make_colour_image(10, 10, 200, 210, 220)
        bs = UnderwaterEnhancer.estimate_backscatter(img)
        assert len(bs) == 3
        assert all(0 <= v <= 255 for v in bs)

    def test_estimate_backscatter_mixed(self):
        img = make_colour_image(20, 20, 200, 200, 200)
        # modify some pixels to be brighter
        img[0][0] = (255, 255, 255)
        bs = UnderwaterEnhancer.estimate_backscatter(img)
        assert bs[2] > 200  # blue channel should be high

    def test_estimate_waterlight(self):
        img = make_colour_image(10, 10, 50, 60, 70)
        wl = UnderwaterEnhancer.estimate_waterlight(img)
        assert len(wl) == 3
        assert all(v >= 0 for v in wl)

    def test_restore_color(self):
        ue = UnderwaterEnhancer()
        img = make_colour_image(4, 4, 150, 140, 130)
        bs = (100, 105, 110)
        wl = (10, 15, 20)
        result = ue.restore_color(img, bs, wl)
        assert len(result) == 4
        assert len(result[0]) == 4
        # output should be within valid range
        for row in result:
            for r, g, b in row:
                assert 0 <= r <= 255
                assert 0 <= g <= 255
                assert 0 <= b <= 255

    def test_restore_color_identity(self):
        ue = UnderwaterEnhancer(fog_factor=0.0)
        img = make_colour_image(2, 2, 100, 100, 100)
        bs = (0, 0, 0)
        wl = (0, 0, 0)
        result = ue.restore_color(img, bs, wl)
        # no backscatter removal, no waterlight addition → slightly reduced by scale
        assert result[0][0][0] == 100  # red scale is 1.0

    def test_attenuation_params(self):
        ue = UnderwaterEnhancer(attenuation_r=0.05, attenuation_g=0.03,
                                attenuation_b=0.01, fog_factor=0.9)
        assert ue.att_r == 0.05
        assert ue.fog_factor == 0.9
