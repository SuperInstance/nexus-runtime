"""Depth estimation — stereo matching, disparity, depth maps, underwater enhancement.

Pure-Python simulation.
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


# ---- DisparityMap ----------------------------------------------------------

class DisparityMap:
    """2-D disparity values from stereo matching."""

    def __init__(self, data: List[List[float]]):
        self._data = data

    @staticmethod
    def from_stereo(left_gray: List[List[float]],
                    right_gray: List[List[float]],
                    max_disparity: int = 64) -> "DisparityMap":
        """Simple block-matching stereo from two grayscale images."""
        h = len(left_gray)
        w = len(left_gray[0]) if h else 0
        disp: List[List[float]] = [[0.0] * w for _ in range(h)]
        block = 3
        half = block // 2
        for y in range(half, h - half):
            for x in range(half, w - half):
                best_d = 0
                best_cost = float("inf")
                for d in range(max_disparity):
                    rx = x - d
                    if rx < half:
                        break
                    cost = 0.0
                    for by in range(-half, half + 1):
                        for bx in range(-half, half + 1):
                            cost += abs(left_gray[y + by][x + bx]
                                        - right_gray[y + by][rx + bx])
                    if cost < best_cost:
                        best_cost = cost
                        best_d = d
                disp[y][x] = float(best_d)
        return DisparityMap(disp)

    def to_depth_map(self, baseline: float = 0.1, focal_length: float = 500.0) -> "DepthMap":
        h = len(self._data)
        w = len(self._data[0]) if h else 0
        depth_data: List[List[float]] = [[0.0] * w for _ in range(h)]
        for y in range(h):
            for x in range(w):
                d = self._data[y][x]
                if d > 0:
                    depth_data[y][x] = (baseline * focal_length) / d
                else:
                    depth_data[y][x] = float("inf")
        return DepthMap(depth_data)

    def filter_invalid(self, max_disparity: int = 64) -> "DisparityMap":
        h = len(self._data)
        w = len(self._data[0]) if h else 0
        filtered = [
            [self._data[y][x] if 0 <= self._data[y][x] <= max_disparity else 0.0
             for x in range(w)]
            for y in range(h)
        ]
        return DisparityMap(filtered)

    def subpixel_refinement(self, threshold: float = 0.0) -> "DisparityMap":
        """Apply sub-pixel refinement using parabola fitting."""
        h = len(self._data)
        w = len(self._data[0]) if h else 0
        refined = [row[:] for row in self._data]
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                d = self._data[y][x]
                if d <= threshold:
                    continue
                left_d = self._data[y][x - 1]
                right_d = self._data[y][x + 1]
                denom = 2.0 * (2 * d - left_d - right_d)
                if abs(denom) > 1e-9:
                    offset = (left_d - right_d) / denom
                    refined[y][x] = d + offset
        return DisparityMap(refined)

    @property
    def data(self) -> List[List[float]]:
        return self._data

    @property
    def width(self) -> int:
        return len(self._data[0]) if self._data else 0

    @property
    def height(self) -> int:
        return len(self._data)


# ---- DepthMap --------------------------------------------------------------

class DepthMap:
    """2-D depth (distance) values."""

    def __init__(self, data: List[List[float]]):
        self._data = data

    def get_depth(self, x: int, y: int) -> float:
        if 0 <= y < len(self._data) and 0 <= x < len(self._data[0]):
            return self._data[y][x]
        return float("inf")

    def get_point_cloud(self, focal_length: float = 500.0,
                        cx: float = 0.0, cy: float = 0.0
                        ) -> List[Tuple[float, float, float]]:
        """Convert depth map to 3-D point cloud (x, y, z)."""
        points: List[Tuple[float, float, float]] = []
        h = len(self._data)
        w = len(self._data[0]) if h else 0
        for y in range(h):
            for x in range(w):
                z = self._data[y][x]
                if z != float("inf") and z > 0:
                    px = (x - cx) * z / focal_length
                    py = (y - cy) * z / focal_length
                    points.append((px, py, z))
        return points

    def create_3d_mesh(self, step: int = 4,
                       focal_length: float = 500.0,
                       cx: float = 0.0, cy: float = 0.0
                       ) -> List[Tuple[Tuple[float, float, float],
                                       Tuple[float, float, float],
                                       Tuple[float, float, float]]]:
        """Create triangle mesh from depth map."""
        h = len(self._data)
        w = len(self._data[0]) if h else 0
        triangles: List[Tuple[Tuple[float, float, float],
                              Tuple[float, float, float],
                              Tuple[float, float, float]]] = []
        for y in range(0, h - step, step):
            for x in range(0, w - step, step):
                def _p(px, py):
                    z = self._data[py][px]
                    if z == float("inf") or z <= 0:
                        return None
                    rx = (px - cx) * z / focal_length
                    ry = (py - cy) * z / focal_length
                    return (rx, ry, z)
                p00 = _p(x, y)
                p10 = _p(x + step, y)
                p01 = _p(x, y + step)
                p11 = _p(x + step, y + step)
                if p00 and p10 and p01:
                    triangles.append((p00, p10, p01))
                if p10 and p11 and p01:
                    triangles.append((p10, p11, p01))
        return triangles

    def filter_by_confidence(self, confidence: List[List[float]],
                             threshold: float = 0.5) -> "DepthMap":
        """Zero-out low-confidence depth values."""
        h = len(self._data)
        w = len(self._data[0]) if h else 0
        ch = len(confidence)
        cw = len(confidence[0]) if ch else 0
        filtered: List[List[float]] = [[0.0] * w for _ in range(h)]
        for y in range(min(h, ch)):
            for x in range(min(w, cw)):
                if confidence[y][x] >= threshold:
                    filtered[y][x] = self._data[y][x]
                else:
                    filtered[y][x] = 0.0
        return DepthMap(filtered)

    @property
    def data(self) -> List[List[float]]:
        return self._data


# ---- StereoMatcher ---------------------------------------------------------

class StereoMatcher:
    """Stereo matching algorithms."""

    def __init__(self, block_size: int = 5, max_disparity: int = 64):
        self.block_size = block_size
        self.max_disparity = max_disparity

    def block_matching(self, left_gray: List[List[float]],
                       right_gray: List[List[float]]) -> DisparityMap:
        return DisparityMap.from_stereo(left_gray, right_gray,
                                        max_disparity=self.max_disparity)

    def compute_cost_volume(self, left_gray: List[List[float]],
                            right_gray: List[List[float]]
                            ) -> List[List[List[float]]]:
        """3-D cost volume: [y][x][disparity]."""
        h = len(left_gray)
        w = len(left_gray[0]) if h else 0
        half = self.block_size // 2
        volume: List[List[List[float]]] = [
            [[0.0] * self.max_disparity for _ in range(w)]
            for _ in range(h)
        ]
        for y in range(half, h - half):
            for x in range(half, w - half):
                for d in range(self.max_disparity):
                    rx = x - d
                    if rx < half:
                        volume[y][x][d] = float("inf")
                        continue
                    cost = 0.0
                    for by in range(-half, half + 1):
                        for bx in range(-half, half + 1):
                            cost += abs(left_gray[y + by][x + bx]
                                        - right_gray[y + by][rx + bx])
                    volume[y][x][d] = cost
        return volume

    def semi_global_matching(self, left_gray: List[List[float]],
                             right_gray: List[List[float]],
                             num_paths: int = 4) -> DisparityMap:
        """Simplified SGM: aggregate costs along multiple directions."""
        cost_volume = self.compute_cost_volume(left_gray, right_gray)
        h = len(left_gray)
        w = len(left_gray[0]) if h else 0
        # directions: right, down, left, up
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dirs = directions[:min(num_paths, 4)]
        # aggregate
        aggregated = [row[:] for row in cost_volume]
        for dy, dx in dirs:
            for y in range(h):
                for x in range(w):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        for d in range(self.max_disparity):
                            penalty = min(aggregated[y][x][d],
                                          aggregated[ny][nx][d] + 1.0)
                            aggregated[y][x][d] = (
                                cost_volume[y][x][d] + penalty
                            )
        # pick best disparity
        disp: List[List[float]] = [[0.0] * w for _ in range(h)]
        for y in range(h):
            for x in range(w):
                best_d = 0
                best_cost = float("inf")
                for d in range(self.max_disparity):
                    if aggregated[y][x][d] < best_cost:
                        best_cost = aggregated[y][x][d]
                        best_d = d
                disp[y][x] = float(best_d)
        return DisparityMap(disp)


# ---- UnderwaterEnhancer ----------------------------------------------------

class UnderwaterEnhancer:
    """Enhance underwater images by compensating for optical effects."""

    def __init__(self, attenuation_r: float = 0.04,
                 attenuation_g: float = 0.02,
                 attenuation_b: float = 0.01,
                 fog_factor: float = 0.8):
        self.att_r = attenuation_r
        self.att_g = attenuation_g
        self.att_b = attenuation_b
        self.fog_factor = fog_factor

    def compensate_attenuation(self, depth_map: DepthMap,
                               image_data: List[List[Tuple[int, int, int]]]
                               ) -> List[List[Tuple[int, int, int]]]:
        """Compensate colour attenuation with distance."""
        h = len(image_data)
        w = len(image_data[0]) if h else 0
        result: List[List[Tuple[int, int, int]]] = [
            [(0, 0, 0)] * w for _ in range(h)
        ]
        for y in range(h):
            for x in range(w):
                r, g, b = image_data[y][x]
                d = depth_map.get_depth(x, y)
                if d == float("inf") or d <= 0:
                    result[y][x] = (r, g, b)
                    continue
                # inverse attenuation
                cr = min(255, int(r / math.exp(-self.att_r * d)))
                cg = min(255, int(g / math.exp(-self.att_g * d)))
                cb = min(255, int(b / math.exp(-self.att_b * d)))
                result[y][x] = (cr, cg, cb)
        return result

    @staticmethod
    def estimate_backscatter(image_data: List[List[Tuple[int, int, int]]]
                            ) -> Tuple[float, float, float]:
        """Estimate backscatter as the average of the brightest 10% pixels."""
        h = len(image_data)
        w = len(image_data[0]) if h else 0
        brightnesses: List[float] = []
        for y in range(h):
            for x in range(w):
                r, g, b = image_data[y][x]
                brightnesses.append((r + g + b) / 3.0)
        brightnesses.sort(reverse=True)
        top_n = max(1, len(brightnesses) // 10)
        top = brightnesses[:top_n]
        avg_b = mean(top)
        # backscatter is typically blueish
        return (avg_b * 0.8, avg_b * 0.85, avg_b)

    @staticmethod
    def estimate_waterlight(image_data: List[List[Tuple[int, int, int]]]
                           ) -> Tuple[float, float, float]:
        """Estimate the ambient water light from the darkest pixels."""
        h = len(image_data)
        w = len(image_data[0]) if h else 0
        if h == 0 or w == 0:
            return (0.0, 0.0, 0.0)
        brightnesses: List[Tuple[float, float, float]] = []
        for y in range(h):
            for x in range(w):
                brightnesses.append(image_data[y][x])
        brightnesses.sort(key=lambda p: p[0] + p[1] + p[2])
        bottom_n = max(1, len(brightnesses) // 10)
        bottom = brightnesses[:bottom_n]
        avg_r = mean([p[0] for p in bottom])
        avg_g = mean([p[1] for p in bottom])
        avg_b = mean([p[2] for p in bottom])
        return (avg_r, avg_g, avg_b)

    def restore_color(self, image_data: List[List[Tuple[int, int, int]]],
                      backscatter: Tuple[float, float, float],
                      waterlight: Tuple[float, float, float]
                      ) -> List[List[Tuple[int, int, int]]]:
        """Simple underwater colour restoration (dark channel prior inspired)."""
        h = len(image_data)
        w = len(image_data[0]) if h else 0
        result: List[List[Tuple[int, int, int]]] = [
            [(0, 0, 0)] * w for _ in range(h)
        ]
        bs_r, bs_g, bs_b = backscatter
        wl_r, wl_g, wl_b = waterlight
        for y in range(h):
            for x in range(w):
                r, g, b = image_data[y][x]
                # remove backscatter
                r2 = max(0, r - bs_r * self.fog_factor) + wl_r * 0.1
                g2 = max(0, g - bs_g * self.fog_factor) + wl_g * 0.1
                b2 = max(0, b - bs_b * self.fog_factor) + wl_b * 0.1
                # white balance (boost red relative to blue)
                scale_r = 1.0
                scale_g = 0.95
                scale_b = 0.85
                r3 = min(255, int(r2 * scale_r))
                g3 = min(255, int(g2 * scale_g))
                b3 = min(255, int(b2 * scale_b))
                result[y][x] = (r3, g3, b3)
        return result
