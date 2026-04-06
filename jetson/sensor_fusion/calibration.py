"""Sensor calibration utilities.

Pure Python — no external dependencies.
Implements bias/scale computation, least-squares fitting, RMSE, and
calibration quality assessment.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CalibrationParams:
    """Parameters describing a sensor calibration."""
    bias: List[float]
    scale: List[float]
    rotation_matrix: List[List[float]] = field(default_factory=lambda: [[1.0]])
    timestamp: float = 0.0


class SensorCalibrator:
    """Sensor calibration engine.

    Computes bias and scale factors, applies calibration, and assesses
    calibration quality through RMSE and residual analysis.
    """

    def compute_bias(
        self,
        ground_truth: List[List[float]],
        measurements: List[List[float]],
    ) -> List[float]:
        """Compute mean bias vector: E[measurement - ground_truth]."""
        if len(ground_truth) == 0 or len(ground_truth) != len(measurements):
            raise ValueError("ground_truth and measurements must have the same non-zero length")
        dim = len(ground_truth[0])
        bias = [0.0] * dim
        for gt, ms in zip(ground_truth, measurements):
            for j in range(dim):
                bias[j] += (ms[j] - gt[j])
        n = len(ground_truth)
        return [b / n for b in bias]

    def compute_scale(
        self,
        ground_truth: List[List[float]],
        measurements: List[List[float]],
    ) -> List[float]:
        """Compute scale factors using least-squares on per-dimension data.

        For each dimension j: scale_j = sum(gt_j * ms_j) / sum(ms_j^2)
        """
        if len(ground_truth) == 0 or len(ground_truth) != len(measurements):
            raise ValueError("ground_truth and measurements must have the same non-zero length")
        dim = len(ground_truth[0])
        scale = [1.0] * dim
        for j in range(dim):
            num = 0.0
            den = 0.0
            for gt, ms in zip(ground_truth, measurements):
                num += gt[j] * ms[j]
                den += ms[j] * ms[j]
            if abs(den) > 1e-12:
                scale[j] = num / den
        return scale

    def calibrate(
        self,
        raw_reading: List[float],
        params: CalibrationParams,
    ) -> List[float]:
        """Apply calibration: calibrated = (raw - bias) * scale."""
        dim = len(raw_reading)
        result = [0.0] * dim
        for j in range(dim):
            b = params.bias[j] if j < len(params.bias) else 0.0
            s = params.scale[j] if j < len(params.scale) else 1.0
            result[j] = (raw_reading[j] - b) * s
        return result

    def compute_residuals(
        self,
        estimated: List[List[float]],
        ground_truth: List[List[float]],
    ) -> List[List[float]]:
        """Compute per-sample residuals: estimated - ground_truth."""
        if len(estimated) != len(ground_truth):
            raise ValueError("estimated and ground_truth must have the same length")
        residuals = []
        for est, gt in zip(estimated, ground_truth):
            residuals.append([e - g for e, g in zip(est, gt)])
        return residuals

    def compute_rmse(
        self,
        estimated: List[List[float]],
        ground_truth: List[List[float]],
    ) -> float:
        """Compute root mean squared error across all dimensions and samples."""
        residuals = self.compute_residuals(estimated, ground_truth)
        if not residuals:
            return 0.0
        sum_sq = 0.0
        count = 0
        for r in residuals:
            for v in r:
                sum_sq += v * v
                count += 1
        if count == 0:
            return 0.0
        return math.sqrt(sum_sq / count)

    def least_squares_fit(
        self,
        A: List[List[float]],
        b: List[float],
    ) -> Optional[List[float]]:
        """Solve Ax = b via the normal equations: x = (A^T A)^{-1} A^T b.

        Returns None if the system is singular.
        """
        n_rows = len(A)
        if n_rows == 0:
            return None
        n_cols = len(A[0])
        if len(b) != n_rows:
            return None

        # A^T * A
        At = self._transpose(A)
        AtA = self._mat_mul(At, A)
        # A^T * b
        Atb = self._mat_vec_mul(At, b)
        # Invert AtA
        AtA_inv = self._mat_invert(AtA)
        if AtA_inv is None:
            return None
        return self._mat_vec_mul(AtA_inv, Atb)

    def assess_calibration_quality(
        self,
        residuals: List[List[float]],
    ) -> float:
        """Assess calibration quality as a score in [0, 1].

        Score = 1 / (1 + mean(|residual|)).
        Perfect calibration (zero residuals) -> score = 1.
        """
        if not residuals:
            return 0.0
        total_abs = 0.0
        count = 0
        for r in residuals:
            for v in r:
                total_abs += abs(v)
                count += 1
        if count == 0:
            return 0.0
        mean_abs = total_abs / count
        return 1.0 / (1.0 + mean_abs)

    # -- matrix helpers (minimal, no numpy) --------------------------------

    @staticmethod
    def _transpose(A: List[List[float]]) -> List[List[float]]:
        rows = len(A)
        cols = len(A[0])
        return [[A[i][j] for i in range(rows)] for j in range(cols)]

    @staticmethod
    def _mat_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        ra, ca = len(A), len(A[0])
        cb = len(B[0])
        C = [[0.0] * cb for _ in range(ra)]
        for i in range(ra):
            for j in range(cb):
                s = 0.0
                for k in range(ca):
                    s += A[i][k] * B[k][j]
                C[i][j] = s
        return C

    @staticmethod
    def _mat_vec_mul(A: List[List[float]], v: List[float]) -> List[float]:
        n = len(A)
        m = len(v)
        return [sum(A[i][j] * v[j] for j in range(m)) for i in range(n)]

    @staticmethod
    def _mat_invert(M: List[List[float]]) -> Optional[List[List[float]]]:
        """Gauss-Jordan inversion."""
        n = len(M)
        aug = [M[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        for col in range(n):
            max_val = abs(aug[col][col])
            max_row = col
            for row in range(col + 1, n):
                if abs(aug[row][col]) > max_val:
                    max_val = abs(aug[row][col])
                    max_row = row
            aug[col], aug[max_row] = aug[max_row], aug[col]
            if abs(aug[col][col]) < 1e-12:
                return None
            pivot = aug[col][col]
            for j in range(2 * n):
                aug[col][j] /= pivot
            for row in range(n):
                if row != col:
                    factor = aug[row][col]
                    for j in range(2 * n):
                        aug[row][j] -= factor * aug[col][j]
        return [row[n:] for row in aug]
