"""Visual odometry — feature detection, matching, pose estimation.

Pure-Python simulation of a typical visual-odometry pipeline.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

try:
    from statistics import mean, stdev
except ImportError:
    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0
    def stdev(xs):
        m = mean(xs)
        var = sum((x - m) ** 2 for x in xs) / len(xs) if xs else 0.0
        return math.sqrt(var)


# ---- dataclasses -----------------------------------------------------------

@dataclass
class FeaturePoint:
    x: float
    y: float
    response: float = 0.0
    descriptor: List[float] = field(default_factory=list)
    octave: int = 0


@dataclass
class FeatureMatch:
    idx_a: int
    idx_b: int
    distance: float
    ratio: float = 1.0


# ---- FeatureDetector -------------------------------------------------------

class FeatureDetector:
    """Simulated corner/feature detector."""

    def __init__(self, threshold: float = 0.01, max_features: int = 500):
        self.threshold = threshold
        self.max_features = max_features

    def detect_harris_corners(self, gray: List[List[float]]) -> List[FeaturePoint]:
        """Simulated Harris corner detection using intensity gradients."""
        h = len(gray)
        w = len(gray[0]) if h else 0
        corners: List[FeaturePoint] = []
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                # Sobel-like gradient approximations
                gx = (gray[y][x + 1] - gray[y][x - 1]) / 2.0
                gy = (gray[y + 1][x] - gray[y - 1][x]) / 2.0
                gxx = abs(gray[y][x + 1] - 2 * gray[y][x] + gray[y][x - 1])
                gyy = abs(gray[y + 1][x] - 2 * gray[y][x] + gray[y - 1][x])
                gxy = abs(
                    (gray[y + 1][x + 1] - gray[y + 1][x - 1]
                     - gray[y - 1][x + 1] + gray[y - 1][x - 1]) / 4.0
                )
                det_h = gxx * gyy - gxy * gxy
                trace_h = gxx + gyy
                r = det_h - 0.04 * trace_h * trace_h
                if r > self.threshold:
                    corners.append(FeaturePoint(x=float(x), y=float(y), response=r))
        corners.sort(key=lambda c: c.response, reverse=True)
        return corners[: self.max_features]

    def detect_fast_corners(self, gray: List[List[float]]) -> List[FeaturePoint]:
        """Simulated FAST corner detection (Bresenham circle 16 pixels)."""
        h = len(gray)
        w = len(gray[0]) if h else 0
        corners: List[FeaturePoint] = []
        intensity_threshold = self.threshold * 255 * 10  # scale for FAST
        offsets = [(0, -3), (1, -3), (2, -2), (3, -1),
                   (3, 0), (3, 1), (2, 2), (1, 3),
                   (0, 3), (-1, 3), (-2, 2), (-3, 1),
                   (-3, 0), (-3, -1), (-2, -2), (-1, -3)]
        for y in range(3, h - 3):
            for x in range(3, w - 3):
                center = gray[y][x]
                bright_count = 0
                dark_count = 0
                for dx, dy in offsets:
                    val = gray[y + dy][x + dx]
                    if val > center + intensity_threshold:
                        bright_count += 1
                    elif val < center - intensity_threshold:
                        dark_count += 1
                if bright_count >= 9 or dark_count >= 9:
                    resp = max(bright_count, dark_count) / 16.0
                    corners.append(FeaturePoint(float(x), float(y), response=resp))
        corners.sort(key=lambda c: c.response, reverse=True)
        return corners[: self.max_features]

    @staticmethod
    def compute_descriptors(features: List[FeaturePoint],
                            gray: List[List[float]],
                            patch_size: int = 5) -> List[FeaturePoint]:
        """Compute a simple patch descriptor for each feature."""
        h = len(gray)
        w = len(gray[0]) if h else 0
        ps = patch_size // 2
        for fp in features:
            ix, iy = int(round(fp.x)), int(round(fp.y))
            patch_vals: List[float] = []
            for dy in range(-ps, ps + 1):
                for dx in range(-ps, ps + 1):
                    nx, ny = ix + dx, iy + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        patch_vals.append(gray[ny][nx])
                    else:
                        patch_vals.append(0.0)
            # normalize
            m = mean(patch_vals)
            s = stdev(patch_vals)
            fp.descriptor = [(v - m) / s if s > 1e-9 else 0.0 for v in patch_vals]
        return features

    @staticmethod
    def compute_orb_descriptors(features: List[FeaturePoint],
                                gray: List[List[float]]) -> List[FeaturePoint]:
        """Simulated ORB binary descriptors (stored as 0/1 float list)."""
        h = len(gray)
        w = len(gray[0]) if h else 0
        # pre-defined 32 comparison pairs (angle, radius offsets)
        pairs = []
        rng = random.Random(42)
        for _ in range(32):
            a1, r1 = rng.uniform(0, 2 * math.pi), rng.uniform(1, 8)
            a2, r2 = rng.uniform(0, 2 * math.pi), rng.uniform(1, 8)
            pairs.append((a1, r1, a2, r2))
        for fp in features:
            ix, iy = int(round(fp.x)), int(round(fp.y))
            desc: List[float] = []
            for a1, r1, a2, r2 in pairs:
                x1 = ix + int(round(r1 * math.cos(a1)))
                y1 = iy + int(round(r1 * math.sin(a1)))
                x2 = ix + int(round(r2 * math.cos(a2)))
                y2 = iy + int(round(r2 * math.sin(a2)))
                v1 = gray[max(0, min(y1, h - 1))][max(0, min(x1, w - 1))]
                v2 = gray[max(0, min(y2, h - 1))][max(0, min(x2, w - 1))]
                desc.append(1.0 if v1 < v2 else 0.0)
            fp.descriptor = desc
        return features


# ---- FeatureMatcher --------------------------------------------------------

class FeatureMatcher:
    """Match feature descriptors between two frames."""

    def __init__(self, ratio_threshold: float = 0.75):
        self.ratio_threshold = ratio_threshold

    @staticmethod
    def _desc_dist(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return float("inf")
        return math.sqrt(sum((va - vb) ** 2 for va, vb in zip(a, b)))

    def brute_force_match(self, feat_a: List[FeaturePoint],
                          feat_b: List[FeaturePoint]) -> List[FeatureMatch]:
        matches: List[FeatureMatch] = []
        for i, fa in enumerate(feat_a):
            if not fa.descriptor:
                continue
            best_dist = float("inf")
            best_j = -1
            for j, fb in enumerate(feat_b):
                if not fb.descriptor:
                    continue
                d = self._desc_dist(fa.descriptor, fb.descriptor)
                if d < best_dist:
                    best_dist = d
                    best_j = j
            if best_j >= 0:
                matches.append(FeatureMatch(idx_a=i, idx_b=best_j,
                                            distance=best_dist))
        return matches

    def ratio_test(self, matches: List[FeatureMatch]) -> List[FeatureMatch]:
        """Lowe's ratio test — returns matches where the best is much better
        than the second best.  Since brute_force_match only returns best,
        we simulate the second best by looking at all distances."""
        # The matches come sorted by distance ascending (worst case they aren't)
        # For simulation we keep matches whose distance is small enough
        if not matches:
            return []
        distances = sorted(m.distance for m in matches)
        median_dist = distances[len(distances) // 2]
        return [m for m in matches if m.distance < median_dist * self.ratio_threshold]

    def ransac_filter(self, matches: List[FeatureMatch],
                      feat_a: List[FeaturePoint],
                      feat_b: List[FeaturePoint],
                      inlier_threshold: float = 3.0,
                      max_iterations: int = 200) -> List[FeatureMatch]:
        """RANSAC — find consensus set under a translational model."""
        if len(matches) < 2:
            return list(matches)
        best_inliers: List[FeatureMatch] = []
        rng = random.Random(0)
        for _ in range(max_iterations):
            m1, m2 = rng.sample(matches, 2)
            fa1 = feat_a[m1.idx_a]
            fa2 = feat_a[m2.idx_a]
            fb1 = feat_b[m1.idx_b]
            fb2 = feat_b[m2.idx_b]
            dx = fb1.x - fa1.x
            dy = fb1.y - fa1.y
            inliers: List[FeatureMatch] = []
            for m in matches:
                fa = feat_a[m.idx_a]
                fb = feat_b[m.idx_b]
                err = math.sqrt((fb.x - fa.x - dx) ** 2 + (fb.y - fa.y - dy) ** 2)
                if err < inlier_threshold:
                    inliers.append(m)
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
        return best_inliers

    @staticmethod
    def compute_fundamental_matrix(matches: List[FeatureMatch],
                                   feat_a: List[FeaturePoint],
                                   feat_b: List[FeaturePoint]) -> List[List[float]]:
        """Compute a simplified 3×3 fundamental matrix from matches using DLT."""
        if len(matches) < 8:
            return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        # Normalise points
        ax = [feat_a[m.idx_a].x for m in matches]
        ay = [feat_a[m.idx_a].y for m in matches]
        bx = [feat_b[m.idx_b].x for m in matches]
        by = [feat_b[m.idx_b].y for m in matches]
        ma, mb = mean(ax), mean(bx)
        sa = max(stdev(ax), stdev(ay), 1e-9)
        sb = max(stdev(bx), stdev(by), 1e-9)
        # Build constraint matrix A  (each match → one row)
        A: List[List[float]] = []
        for i in range(len(matches)):
            x1 = (ax[i] - ma) / sa
            y1 = (ay[i] - ma) / sa
            x2 = (bx[i] - mb) / sb
            y2 = (by[i] - mb) / sb
            A.append([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1.0])
        # Solve via simplified pseudo-inverse (use normal equations)
        # A^T A  (9x9)
        rows = len(A)
        cols = 9
        ATA: List[List[float]] = [[0.0] * cols for _ in range(cols)]
        for k in range(rows):
            for i in range(cols):
                for j in range(cols):
                    ATA[i][j] += A[k][i] * A[k][j]
        # Identity right-hand side
        RHS: List[List[float]] = [[1.0 if i == j else 0.0 for j in range(cols)] for i in range(cols)]
        # Gaussian elimination
        for i in range(cols):
            # pivot
            max_val = abs(ATA[i][i])
            max_row = i
            for r in range(i + 1, cols):
                if abs(ATA[r][i]) > max_val:
                    max_val = abs(ATA[r][i])
                    max_row = r
            ATA[i], ATA[max_row] = ATA[max_row], ATA[i]
            RHS[i], RHS[max_row] = RHS[max_row], RHS[i]
            pivot = ATA[i][i]
            if abs(pivot) < 1e-12:
                continue
            for j in range(cols):
                ATA[i][j] /= pivot
                RHS[i][j] /= pivot
            for r in range(cols):
                if r == i:
                    continue
                factor = ATA[r][i]
                for j in range(cols):
                    ATA[r][j] -= factor * ATA[i][j]
                    RHS[r][j] -= factor * RHS[i][j]
        # Last column of RHS is the solution
        f_flat = [RHS[i][8] for i in range(9)]
        # Reshape to 3x3
        F = [[f_flat[i * 3 + j] for j in range(3)] for i in range(3)]
        # Normalise so that F[2][2] == 1
        scale = F[2][2] if abs(F[2][2]) > 1e-12 else 1.0
        for i in range(3):
            for j in range(3):
                F[i][j] /= scale
        return F


# ---- VisualOdometry --------------------------------------------------------

class VisualOdometry:
    """Simulated monocular visual odometry."""

    def __init__(self, focal_length: float = 500.0,
                 cx: float = 320.0, cy: float = 240.0):
        self.focal_length = focal_length
        self.cx = cx
        self.cy = cy
        self.pose: List[float] = [0.0, 0.0, 0.0]  # x, y, theta
        self.trajectory: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
        self._prev_features: Optional[List[FeaturePoint]] = None
        self._frame_idx = 0
        self._total_drift: float = 0.0

    def process_frame(self, gray: List[List[float]]) -> Dict:
        """Process one grayscale frame.  Returns match info dict."""
        detector = FeatureDetector()
        features = detector.detect_harris_corners(gray)
        if features:
            FeatureDetector.compute_descriptors(features, gray)

        result: Dict = {"features": len(features), "matches": 0, "inliers": 0}

        if self._prev_features is not None and features:
            matcher = FeatureMatcher()
            matches = matcher.brute_force_match(self._prev_features, features)
            result["matches"] = len(matches)
            filtered = matcher.ransac_filter(matches, self._prev_features, features)
            result["inliers"] = len(filtered)
            motion = self.estimate_motion(filtered, self._prev_features, features)
            self.update_pose(motion)

        self._prev_features = features
        self._frame_idx += 1
        return result

    @staticmethod
    def estimate_motion(matches: List[FeatureMatch],
                        feat_a: List[FeaturePoint],
                        feat_b: List[FeaturePoint]) -> Tuple[float, float, float]:
        """Estimate (dx, dy, dtheta) from matches."""
        if not matches:
            return (0.0, 0.0, 0.0)
        dxs = [feat_b[m.idx_b].x - feat_a[m.idx_a].x for m in matches]
        dys = [feat_b[m.idx_b].y - feat_a[m.idx_a].y for m in matches]
        avg_dx = mean(dxs)
        avg_dy = mean(dys)
        # dtheta from average rotation of displacement vectors
        dthetas = []
        for m in matches:
            dx = feat_b[m.idx_b].x - feat_a[m.idx_a].x
            dy = feat_b[m.idx_b].y - feat_a[m.idx_a].y
            angle = math.atan2(dy, dx)
            dthetas.append(angle)
        avg_dtheta = mean(dthetas) if dthetas else 0.0
        return (avg_dx, avg_dy, avg_dtheta)

    def update_pose(self, motion: Tuple[float, float, float]) -> None:
        dx, dy, dtheta = motion
        self._total_drift += math.sqrt(dx * dx + dy * dy)
        self.pose[0] += dx
        self.pose[1] += dy
        self.pose[2] += dtheta
        self.trajectory.append((self.pose[0], self.pose[1], self.pose[2]))

    def get_trajectory(self) -> List[Tuple[float, float, float]]:
        return list(self.trajectory)

    def compute_drift(self, ground_truth: Optional[List[Tuple[float, float, float]]] = None) -> float:
        if ground_truth is None:
            return self._total_drift
        # RMS error
        n = min(len(self.trajectory), len(ground_truth))
        if n == 0:
            return 0.0
        total = 0.0
        for i in range(n):
            px, py, _ = self.trajectory[i]
            gx, gy, _ = ground_truth[i]
            total += (px - gx) ** 2 + (py - gy) ** 2
        return math.sqrt(total / n)
