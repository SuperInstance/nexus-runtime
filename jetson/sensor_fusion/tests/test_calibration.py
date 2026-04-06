"""Tests for calibration.py — CalibrationParams, SensorCalibrator."""

import math
import pytest

from jetson.sensor_fusion.calibration import CalibrationParams, SensorCalibrator


# ===================================================================
# CalibrationParams tests
# ===================================================================

class TestCalibrationParams:
    def test_creation_minimal(self):
        cp = CalibrationParams(bias=[0.1], scale=[1.0])
        assert cp.bias == [0.1]
        assert cp.scale == [1.0]

    def test_creation_full(self):
        rot = [[1.0, 0.0], [0.0, 1.0]]
        cp = CalibrationParams(bias=[0.0, 0.0], scale=[1.0, 1.0], rotation_matrix=rot, timestamp=42.0)
        assert cp.timestamp == 42.0
        assert cp.rotation_matrix == rot

    def test_default_rotation(self):
        cp = CalibrationParams(bias=[], scale=[])
        assert cp.rotation_matrix == [[1.0]]

    def test_default_timestamp(self):
        cp = CalibrationParams(bias=[0.0], scale=[1.0])
        assert cp.timestamp == 0.0

    def test_mutable(self):
        cp = CalibrationParams(bias=[0.0], scale=[1.0])
        cp.bias[0] = 5.0
        assert cp.bias[0] == 5.0


# ===================================================================
# SensorCalibrator tests
# ===================================================================

class TestComputeBias:
    def test_zero_bias(self):
        sc = SensorCalibrator()
        gt = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        ms = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        bias = sc.compute_bias(gt, ms)
        assert bias == pytest.approx([0.0, 0.0], abs=1e-10)

    def test_constant_bias(self):
        sc = SensorCalibrator()
        gt = [[0.0], [0.0], [0.0]]
        ms = [[1.0], [1.0], [1.0]]
        bias = sc.compute_bias(gt, ms)
        assert bias == pytest.approx([1.0], abs=1e-10)

    def test_different_bias_per_dim(self):
        sc = SensorCalibrator()
        gt = [[0.0, 0.0], [0.0, 0.0]]
        ms = [[2.0, -1.0], [2.0, -1.0]]
        bias = sc.compute_bias(gt, ms)
        assert bias[0] == pytest.approx(2.0, abs=1e-10)
        assert bias[1] == pytest.approx(-1.0, abs=1e-10)

    def test_empty_raises(self):
        sc = SensorCalibrator()
        with pytest.raises(ValueError):
            sc.compute_bias([], [])

    def test_length_mismatch_raises(self):
        sc = SensorCalibrator()
        with pytest.raises(ValueError):
            sc.compute_bias([[1.0]], [[1.0], [2.0]])


class TestComputeScale:
    def test_identity_scale(self):
        sc = SensorCalibrator()
        gt = [[1.0, 2.0], [3.0, 4.0]]
        ms = [[1.0, 2.0], [3.0, 4.0]]
        scale = sc.compute_scale(gt, ms)
        assert scale[0] == pytest.approx(1.0, abs=1e-10)
        assert scale[1] == pytest.approx(1.0, abs=1e-10)

    def test_doubled_scale(self):
        sc = SensorCalibrator()
        gt = [[2.0, 4.0], [6.0, 8.0]]
        ms = [[1.0, 2.0], [3.0, 4.0]]
        scale = sc.compute_scale(gt, ms)
        assert scale[0] == pytest.approx(2.0, abs=1e-10)
        assert scale[1] == pytest.approx(2.0, abs=1e-10)

    def test_zero_measurements_returns_one(self):
        sc = SensorCalibrator()
        gt = [[0.0]]
        ms = [[0.0]]
        scale = sc.compute_scale(gt, ms)
        # denominator is 0, so returns default 1.0
        assert scale == [1.0]

    def test_empty_raises(self):
        sc = SensorCalibrator()
        with pytest.raises(ValueError):
            sc.compute_scale([], [])

    def test_length_mismatch_raises(self):
        sc = SensorCalibrator()
        with pytest.raises(ValueError):
            sc.compute_scale([[1.0]], [[1.0], [2.0]])


class TestCalibrate:
    def test_identity_calibration(self):
        sc = SensorCalibrator()
        params = CalibrationParams(bias=[0.0, 0.0], scale=[1.0, 1.0])
        assert sc.calibrate([5.0, 3.0], params) == [5.0, 3.0]

    def test_bias_correction(self):
        sc = SensorCalibrator()
        params = CalibrationParams(bias=[2.0, 1.0], scale=[1.0, 1.0])
        assert sc.calibrate([5.0, 5.0], params) == [3.0, 4.0]

    def test_scale_correction(self):
        sc = SensorCalibrator()
        params = CalibrationParams(bias=[0.0, 0.0], scale=[2.0, 0.5])
        assert sc.calibrate([3.0, 4.0], params) == [6.0, 2.0]

    def test_combined_bias_and_scale(self):
        sc = SensorCalibrator()
        params = CalibrationParams(bias=[1.0], scale=[2.0])
        assert sc.calibrate([3.0], params) == [4.0]  # (3 - 1) * 2

    def test_dimension_mismatch_uses_defaults(self):
        sc = SensorCalibrator()
        params = CalibrationParams(bias=[1.0], scale=[1.0])
        result = sc.calibrate([5.0, 10.0], params)
        assert result[0] == 4.0
        assert result[1] == 10.0  # no bias/scale for dim 1


class TestComputeResiduals:
    def test_zero_residuals(self):
        sc = SensorCalibrator()
        est = [[1.0, 2.0], [3.0, 4.0]]
        gt = [[1.0, 2.0], [3.0, 4.0]]
        res = sc.compute_residuals(est, gt)
        assert res == [[0.0, 0.0], [0.0, 0.0]]

    def test_nonzero_residuals(self):
        sc = SensorCalibrator()
        est = [[2.0, 3.0]]
        gt = [[1.0, 1.0]]
        res = sc.compute_residuals(est, gt)
        assert res == [[1.0, 2.0]]

    def test_length_mismatch_raises(self):
        sc = SensorCalibrator()
        with pytest.raises(ValueError):
            sc.compute_residuals([[1.0]], [[1.0], [2.0]])

    def test_empty_lists(self):
        sc = SensorCalibrator()
        res = sc.compute_residuals([], [])
        assert res == []


class TestComputeRMSE:
    def test_zero_rmse(self):
        sc = SensorCalibrator()
        gt = [[0.0, 0.0], [0.0, 0.0]]
        est = [[0.0, 0.0], [0.0, 0.0]]
        assert sc.compute_rmse(est, gt) == pytest.approx(0.0, abs=1e-10)

    def test_known_rmse(self):
        sc = SensorCalibrator()
        gt = [[0.0], [0.0]]
        est = [[1.0], [3.0]]
        # residuals: [1, 3], mse = (1+9)/2 = 5, rmse = sqrt(5)
        rmse = sc.compute_rmse(est, gt)
        assert rmse == pytest.approx(math.sqrt(5.0), abs=1e-10)

    def test_multi_dim_rmse(self):
        sc = SensorCalibrator()
        gt = [[0.0, 0.0]]
        est = [[3.0, 4.0]]
        # residuals: [3, 4], mse = (9+16)/2 = 12.5, rmse = sqrt(12.5)
        rmse = sc.compute_rmse(est, gt)
        assert rmse == pytest.approx(math.sqrt(12.5), abs=1e-10)

    def test_empty_rmse(self):
        sc = SensorCalibrator()
        assert sc.compute_rmse([], []) == 0.0


class TestLeastSquaresFit:
    def test_exact_solution(self):
        sc = SensorCalibrator()
        # 2x = 4 => x = 2
        A = [[2.0]]
        b = [4.0]
        x = sc.least_squares_fit(A, b)
        assert x is not None
        assert x[0] == pytest.approx(2.0, abs=1e-10)

    def test_overdetermined_system(self):
        sc = SensorCalibrator()
        # y = 2x + 1: (1,3), (2,5), (3,7)
        A = [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
        b = [3.0, 5.0, 7.0]
        x = sc.least_squares_fit(A, b)
        assert x is not None
        assert x[0] == pytest.approx(2.0, abs=1e-6)
        assert x[1] == pytest.approx(1.0, abs=1e-6)

    def test_identity_system(self):
        sc = SensorCalibrator()
        A = [[1.0, 0.0], [0.0, 1.0]]
        b = [5.0, 7.0]
        x = sc.least_squares_fit(A, b)
        assert x == pytest.approx([5.0, 7.0], abs=1e-10)

    def test_singular_system_returns_none(self):
        sc = SensorCalibrator()
        A = [[1.0, 2.0], [2.0, 4.0]]  # singular
        b = [3.0, 6.0]
        x = sc.least_squares_fit(A, b)
        assert x is None

    def test_empty_A_returns_none(self):
        sc = SensorCalibrator()
        assert sc.least_squares_fit([], []) is None

    def test_dimension_mismatch_returns_none(self):
        sc = SensorCalibrator()
        A = [[1.0, 2.0]]
        b = [1.0, 2.0, 3.0]
        assert sc.least_squares_fit(A, b) is None


class TestAssessCalibrationQuality:
    def test_perfect_calibration(self):
        sc = SensorCalibrator()
        residuals = [[0.0, 0.0], [0.0, 0.0]]
        score = sc.assess_calibration_quality(residuals)
        assert score == pytest.approx(1.0, abs=1e-10)

    def test_poor_calibration(self):
        sc = SensorCalibrator()
        residuals = [[100.0, 100.0]]
        score = sc.assess_calibration_quality(residuals)
        assert score < 0.1

    def test_better_calibration_higher_score(self):
        sc = SensorCalibrator()
        good = [[0.1], [0.2]]
        bad = [[10.0], [20.0]]
        assert sc.assess_calibration_quality(good) > sc.assess_calibration_quality(bad)

    def test_empty_residuals(self):
        sc = SensorCalibrator()
        score = sc.assess_calibration_quality([])
        assert score == 0.0

    def test_score_between_zero_and_one(self):
        sc = SensorCalibrator()
        residuals = [[1.0, 2.0, 3.0]]
        score = sc.assess_calibration_quality(residuals)
        assert 0.0 < score < 1.0

    def test_negative_residuals_handled(self):
        sc = SensorCalibrator()
        residuals = [[-1.0, -2.0]]
        score = sc.assess_calibration_quality(residuals)
        assert 0.0 < score < 1.0


class TestMatrixHelpers:
    def test_transpose(self):
        assert SensorCalibrator._transpose([[1.0, 2.0], [3.0, 4.0]]) == [[1.0, 3.0], [2.0, 4.0]]

    def test_mat_mul(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[5.0, 6.0], [7.0, 8.0]]
        C = SensorCalibrator._mat_mul(A, B)
        assert C == [[19.0, 22.0], [43.0, 50.0]]

    def test_mat_vec_mul(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        v = [5.0, 6.0]
        assert SensorCalibrator._mat_vec_mul(A, v) == [17.0, 39.0]

    def test_mat_invert_2x2(self):
        M = [[4.0, 7.0], [2.0, 6.0]]
        inv = SensorCalibrator._mat_invert(M)
        assert inv is not None
        # M * inv ≈ I
        prod = SensorCalibrator._mat_mul(M, inv)
        for i in range(2):
            for j in range(2):
                assert abs(prod[i][j] - (1.0 if i == j else 0.0)) < 1e-10

    def test_mat_invert_singular(self):
        M = [[1.0, 2.0], [2.0, 4.0]]
        assert SensorCalibrator._mat_invert(M) is None
