"""Sonar data processing — pure Python simulation.

Provides sonar ping/scan types, preprocessing, segmentation, and mapping.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set

try:
    from statistics import mean, stdev, median
except ImportError:
    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0
    def stdev(xs):
        m = mean(xs)
        var = sum((x - m) ** 2 for x in xs) / len(xs) if xs else 0.0
        return math.sqrt(var)
    def median(xs):
        if not xs:
            return 0.0
        s = sorted(xs)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


# ---- dataclasses -----------------------------------------------------------

@dataclass
class SonarPing:
    """Single sonar return sample."""
    range_m: float
    angle_deg: float
    intensity: float
    timestamp: float = 0.0


@dataclass
class SonarScan:
    """A full 360° sonar scan composed of pings."""
    pings: List[SonarPing] = field(default_factory=list)
    timestamp: float = 0.0
    heading_deg: float = 0.0
    resolution: float = 1.0  # degrees per ping

    @property
    def max_range(self) -> float:
        return max((p.range_m for p in self.pings), default=0.0)

    @property
    def min_range(self) -> float:
        return min((p.range_m for p in self.pings), default=0.0)


# ---- SonarPreprocessor -----------------------------------------------------

class SonarPreprocessor:
    def __init__(self, noise_threshold: float = 5.0, gain: float = 1.0,
                 sidelobe_threshold: float = 0.3):
        self.noise_threshold = noise_threshold
        self.gain = gain
        self.sidelobe_threshold = sidelobe_threshold

    def remove_noise(self, scan: SonarScan, window_size: int = 3) -> SonarScan:
        """Median-filter pings to remove noise spikes."""
        if not scan.pings:
            return SonarScan(pings=[], timestamp=scan.timestamp,
                             heading_deg=scan.heading_deg, resolution=scan.resolution)
        filtered: List[SonarPing] = []
        n = len(scan.pings)
        half = window_size // 2
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            window = [scan.pings[j].intensity for j in range(lo, hi)]
            med = median(window)
            filtered.append(SonarPing(
                range_m=scan.pings[i].range_m,
                angle_deg=scan.pings[i].angle_deg,
                intensity=med,
                timestamp=scan.pings[i].timestamp,
            ))
        return SonarScan(pings=filtered, timestamp=scan.timestamp,
                         heading_deg=scan.heading_deg, resolution=scan.resolution)

    def apply_gain(self, scan: SonarScan) -> SonarScan:
        """Apply time-varying gain compensation (TVG)."""
        max_r = scan.max_range if scan.max_range > 0 else 1.0
        adjusted: List[SonarPing] = []
        for p in scan.pings:
            tvg = 1.0 + (p.range_m / max_r) * 0.5  # increases with range
            new_intensity = p.intensity * self.gain * tvg
            adjusted.append(SonarPing(
                range_m=p.range_m,
                angle_deg=p.angle_deg,
                intensity=min(255.0, new_intensity),
                timestamp=p.timestamp,
            ))
        return SonarScan(pings=adjusted, timestamp=scan.timestamp,
                         heading_deg=scan.heading_deg, resolution=scan.resolution)

    def detect_returns(self, scan: SonarScan, threshold: float = 20.0) -> List[SonarPing]:
        """Extract significant returns above threshold."""
        return [p for p in scan.pings if p.intensity > threshold]

    def sidelobe_suppression(self, scan: SonarScan) -> SonarScan:
        """Suppress weak returns that are likely sidelobes."""
        if not scan.pings:
            return scan
        max_intensity = max(p.intensity for p in scan.pings)
        if max_intensity == 0:
            return scan
        suppressed: List[SonarPing] = []
        for p in scan.pings:
            if p.intensity / max_intensity < self.sidelobe_threshold:
                suppressed.append(SonarPing(
                    range_m=p.range_m, angle_deg=p.angle_deg,
                    intensity=0.0, timestamp=p.timestamp))
            else:
                suppressed.append(p)
        return SonarScan(pings=suppressed, timestamp=scan.timestamp,
                         heading_deg=scan.heading_deg, resolution=scan.resolution)


# ---- SonarSegmenter --------------------------------------------------------

@dataclass
class SonarSegment:
    pings: List[SonarPing] = field(default_factory=list)
    label: str = "unknown"
    confidence: float = 0.0


class SonarSegmenter:
    def __init__(self, min_segment_size: int = 5, merge_threshold: float = 10.0):
        self.min_segment_size = min_segment_size
        self.merge_threshold = merge_threshold

    def segment_scan(self, scan: SonarScan) -> List[SonarSegment]:
        """Segment a scan into contiguous groups of strong returns."""
        if not scan.pings:
            return []
        segments: List[SonarSegment] = []
        current: List[SonarPing] = []
        for p in scan.pings:
            if p.intensity > self.merge_threshold:
                current.append(p)
            else:
                if len(current) >= self.min_segment_size:
                    segments.append(SonarSegment(pings=list(current)))
                current = []
        if len(current) >= self.min_segment_size:
            segments.append(SonarSegment(pings=list(current)))
        return segments

    def merge_segments(self, segments: List[SonarSegment],
                       gap_threshold: float = 5.0) -> List[SonarSegment]:
        """Merge segments that are close in angle."""
        if len(segments) <= 1:
            return segments
        merged: List[SonarSegment] = [segments[0]]
        for seg in segments[1:]:
            last = merged[-1]
            last_angle = last.pings[-1].angle_deg if last.pings else 0
            first_angle = seg.pings[0].angle_deg if seg.pings else 0
            if abs(first_angle - last_angle) < gap_threshold:
                last.pings.extend(seg.pings)
            else:
                merged.append(seg)
        return merged

    def classify_segment(self, segment: SonarSegment) -> Tuple[str, float]:
        """Classify a segment as a marine object type."""
        if not segment.pings:
            return ("unknown", 0.0)
        intensities = [p.intensity for p in segment.pings]
        ranges = [p.range_m for p in segment.pings]
        mean_int = mean(intensities)
        mean_range = mean(ranges)
        int_var = stdev(intensities) if len(intensities) > 1 else 0.0
        range_var = stdev(ranges) if len(ranges) > 1 else 0.0
        # heuristics
        if mean_range < 10 and mean_int > 100:
            return ("bottom", 0.8)
        if range_var < 2 and mean_int > 80:
            return ("wall", 0.7)
        if mean_int > 120 and int_var < 20:
            return ("vessel", 0.75)
        if len(segment.pings) > 20 and mean_int < 60:
            return ("school_of_fish", 0.6)
        if range_var > 5 and mean_int > 50:
            return ("rock", 0.5)
        return ("unknown", 0.3)

    def compute_segment_features(self, segment: SonarSegment) -> Dict:
        if not segment.pings:
            return {}
        intensities = [p.intensity for p in segment.pings]
        ranges = [p.range_m for p in segment.pings]
        angles = [p.angle_deg for p in segment.pings]
        return {
            "mean_intensity": mean(intensities),
            "std_intensity": stdev(intensities) if len(intensities) > 1 else 0.0,
            "max_intensity": max(intensities),
            "min_intensity": min(intensities),
            "mean_range": mean(ranges),
            "std_range": stdev(ranges) if len(ranges) > 1 else 0.0,
            "angle_span": max(angles) - min(angles) if len(angles) > 1 else 0.0,
            "size": len(segment.pings),
            "energy": sum(i ** 2 for i in intensities),
        }


# ---- SonarMapper -----------------------------------------------------------

class SonarMapper:
    """Integrate sonar scans into an occupancy grid."""

    def __init__(self, grid_size: int = 200, cell_size: float = 0.5):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.grid: List[List[float]] = [[0.5] * grid_size for _ in range(grid_size)]
        self._scan_count = 0

    def integrate_scan(self, scan: SonarScan, position: Tuple[float, float] = (0, 0),
                       heading: float = 0.0) -> None:
        """Integrate a scan into the occupancy grid using Bayesian updates."""
        self._scan_count += 1
        cx = self.grid_size // 2
        cy = self.grid_size // 2
        for p in scan.pings:
            rad = math.radians(p.angle_deg + heading)
            # compute grid cell
            gx = int(round(cx + p.range_m * math.sin(rad) / self.cell_size))
            gy = int(round(cy + p.range_m * math.cos(rad) / self.cell_size))
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                # Bayesian update: occupied prob
                prob_occ = p.intensity / 255.0
                prior = self.grid[gy][gx]
                # simple log-odds update
                log_prior = math.log(prior / (1 - prior + 1e-12))
                log_occ = math.log(prob_occ / (1 - prob_occ + 1e-12) + 1e-12)
                log_posterior = log_prior + log_occ
                posterior = 1.0 / (1.0 + math.exp(-log_posterior))
                self.grid[gy][gx] = max(0.01, min(0.99, posterior))

    def occupancy_grid_from_scans(self, scans: List[SonarScan]) -> List[List[float]]:
        """Build fresh grid from a list of scans."""
        self.grid = [[0.5] * self.grid_size for _ in range(self.grid_size)]
        for scan in scans:
            self.integrate_scan(scan)
        return self.grid

    def detect_changes(self, old_grid: List[List[float]],
                       new_grid: List[List[float]],
                       threshold: float = 0.3) -> List[Tuple[int, int]]:
        """Detect cells that changed significantly."""
        changes: List[Tuple[int, int]] = []
        rows = min(len(old_grid), len(new_grid))
        if rows == 0:
            return changes
        cols = min(len(old_grid[0]), len(new_grid[0]))
        for y in range(rows):
            for x in range(cols):
                if abs(new_grid[y][x] - old_grid[y][x]) > threshold:
                    changes.append((x, y))
        return changes

    @staticmethod
    def estimate_object_position(segment: SonarSegment, vessel_pos: Tuple[float, float],
                                 vessel_heading: float) -> Tuple[float, float]:
        """Estimate world position of a sonar segment."""
        if not segment.pings:
            return vessel_pos
        mean_range = mean(p.range_m for p in segment.pings)
        mean_angle = mean(p.angle_deg for p in segment.pings)
        rad = math.radians(mean_angle + vessel_heading)
        wx = vessel_pos[0] + mean_range * math.sin(rad)
        wy = vessel_pos[1] + mean_range * math.cos(rad)
        return (wx, wy)
