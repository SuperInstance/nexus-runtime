"""
Traffic Pattern Analysis Module.

Detects traffic patterns, classifies density, predicts congestion,
identifies hotspots, computes flow rates, and analyzes seasonal trends.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class DensityLevel(Enum):
    """Traffic density classification."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"


class PatternType(Enum):
    """Types of traffic patterns."""
    SHIPPING_LANE = "shipping_lane"
    FERRY_ROUTE = "ferry_route"
    FISHING_GROUND = "fishing_ground"
    ANCHORAGE_APPROACH = "anchorage_approach"
    PILOTAGE_ROUTE = "pilotage_route"
    RANDOM = "random"


@dataclass
class TrafficPattern:
    """Represents a detected traffic pattern."""
    pattern_type: str
    corridor_center: Tuple[float, float, float, float]  # (lat1, lon1, lat2, lon2)
    corridor_width: float = 0.0      # nautical miles
    typical_speed: float = 0.0       # knots
    direction: float = 0.0           # degrees true
    confidence: float = 0.0          # 0.0 - 1.0


@dataclass
class VesselTrack:
    """A vessel's track history for traffic analysis."""
    mmsi: int
    positions: List[Tuple[float, float, float]] = field(default_factory=list)  # (lat, lon, timestamp)
    speeds: List[float] = field(default_factory=list)
    headings: List[float] = field(default_factory=list)


@dataclass
class HotspotArea:
    """Identified traffic hotspot area."""
    center: Tuple[float, float]
    radius: float              # nautical miles
    vessel_count: int
    avg_speed: float
    time_window: Tuple[float, float]  # (start, end) timestamps


@dataclass
class CongestionForecast:
    """Predicted congestion conditions."""
    expected_density: DensityLevel
    confidence: float
    peak_time: Optional[float] = None
    affected_area: Optional[Tuple[float, float, float, float]] = None  # (lat1, lon1, lat2, lon2)
    recommendation: str = ""


@dataclass
class SeasonalTrend:
    """Seasonal traffic trend data."""
    month: int
    avg_vessel_count: float
    peak_hour: int
    common_patterns: List[str]
    density_trend: str  # 'increasing', 'decreasing', 'stable'


class TrafficAnalyzer:
    """Analyzes maritime traffic patterns and density."""

    def __init__(self) -> None:
        self._patterns: List[TrafficPattern] = []

    def detect_patterns(self, vessel_tracks: List[VesselTrack]) -> List[TrafficPattern]:
        """
        Detect traffic patterns from vessel track data.
        Clusters tracks by similarity of direction and position to identify
        shipping lanes, ferry routes, etc.
        """
        if not vessel_tracks:
            return []

        # Analyze direction clusters
        direction_groups: Dict[float, List[VesselTrack]] = defaultdict(list)
        direction_bin = 15.0  # 15-degree bins

        for track in vessel_tracks:
            if track.headings and track.positions:
                avg_heading = self._circular_mean(track.headings)
                binned = round(avg_heading / direction_bin) * direction_bin % 360
                direction_groups[binned].append(track)

        patterns = []
        for direction, tracks in direction_groups.items():
            if len(tracks) < 2:
                continue

            # Compute corridor center line from track positions
            all_lats = []
            all_lons = []
            all_speeds = []
            for t in tracks:
                for pos in t.positions:
                    all_lats.append(pos[0])
                    all_lons.append(pos[1])
                all_speeds.extend(t.speeds)

            if not all_lats:
                continue

            min_lat, max_lat = min(all_lats), max(all_lats)
            min_lon, max_lon = min(all_lons), max(all_lons)
            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2

            # Estimate corridor width from track spread
            spread = max(max_lat - min_lat, max_lon - min_lon)
            width = spread * 60.0 / 2.0  # rough nm conversion

            avg_speed = sum(all_speeds) / len(all_speeds) if all_speeds else 0.0
            confidence = min(len(tracks) / 5.0, 1.0)

            # Determine pattern type
            if len(tracks) >= 10 and width < 5.0:
                ptype = PatternType.SHIPPING_LANE.value
            elif len(tracks) >= 5 and avg_speed > 15.0:
                ptype = PatternType.FERRY_ROUTE.value
            elif avg_speed < 5.0:
                ptype = PatternType.FISHING_GROUND.value
            elif width < 2.0:
                ptype = PatternType.PILOTAGE_ROUTE.value
            else:
                ptype = PatternType.RANDOM.value

            pattern = TrafficPattern(
                pattern_type=ptype,
                corridor_center=(min_lat, min_lon, max_lat, max_lon),
                corridor_width=round(width, 2),
                typical_speed=round(avg_speed, 1),
                direction=round(direction, 1),
                confidence=round(confidence, 2),
            )
            patterns.append(pattern)

        self._patterns = patterns
        return patterns

    def classify_traffic_density(
        self,
        area: Tuple[float, float, float, float],
        vessels: List[Tuple[float, float, float]],
    ) -> DensityLevel:
        """
        Classify traffic density within a rectangular area.
        area: (lat1, lon1, lat2, lon2)
        vessels: list of (lat, lon, speed_knots)
        Returns DensityLevel enum.
        """
        lat_min = min(area[0], area[2])
        lat_max = max(area[0], area[2])
        lon_min = min(area[1], area[3])
        lon_max = max(area[1], area[3])

        count = 0
        for v in vessels:
            if lat_min <= v[0] <= lat_max and lon_min <= v[1] <= lon_max:
                count += 1

        # Area in square nautical miles (approximate)
        area_width = (lat_max - lat_min) * 60.0
        area_height = (lon_max - lon_min) * 60.0 * math.cos(math.radians((lat_min + lat_max) / 2))
        area_sq_nm = area_width * area_height if area_width > 0 and area_height > 0 else 1.0
        density_per_sq_nm = count / area_sq_nm

        if density_per_sq_nm < 0.1:
            return DensityLevel.VERY_LOW
        elif density_per_sq_nm < 0.5:
            return DensityLevel.LOW
        elif density_per_sq_nm < 1.0:
            return DensityLevel.MODERATE
        elif density_per_sq_nm < 2.0:
            return DensityLevel.HIGH
        elif density_per_sq_nm < 5.0:
            return DensityLevel.VERY_HIGH
        else:
            return DensityLevel.CRITICAL

    def predict_congestion(
        self,
        current_traffic: List[Tuple[float, float, float, float]],
        trends: List[Tuple[float, float]],
    ) -> CongestionForecast:
        """
        Predict congestion based on current traffic and trends.
        current_traffic: list of (lat, lon, speed, timestamp)
        trends: list of (time_offset_hours, vessel_count_multiplier)
        Returns CongestionForecast.
        """
        if not current_traffic:
            return CongestionForecast(
                expected_density=DensityLevel.VERY_LOW,
                confidence=0.5,
                recommendation="No traffic data available."
            )

        vessel_count = len(current_traffic)
        avg_lat = sum(t[0] for t in current_traffic) / vessel_count
        avg_lon = sum(t[1] for t in current_traffic) / vessel_count

        # Use latest trend multiplier
        multiplier = 1.0
        if trends:
            multiplier = trends[-1][1]

        projected_count = vessel_count * multiplier
        density = min(projected_count / 10.0, 10.0)

        if density < 0.5:
            expected = DensityLevel.LOW
        elif density < 1.0:
            expected = DensityLevel.MODERATE
        elif density < 2.0:
            expected = DensityLevel.HIGH
        elif density < 5.0:
            expected = DensityLevel.VERY_HIGH
        else:
            expected = DensityLevel.CRITICAL

        confidence = min(len(trends) / 3.0, 0.95) if trends else 0.3

        recommendation = ""
        if expected in (DensityLevel.HIGH, DensityLevel.VERY_HIGH, DensityLevel.CRITICAL):
            recommendation = "Consider alternate routing to avoid congestion."
        elif expected == DensityLevel.MODERATE:
            recommendation = "Monitor traffic conditions."
        else:
            recommendation = "Normal operations."

        return CongestionForecast(
            expected_density=expected,
            confidence=round(confidence, 2),
            peak_time=current_traffic[-1][3] + 3600 if trends else None,
            affected_area=(avg_lat - 0.1, avg_lon - 0.1, avg_lat + 0.1, avg_lon + 0.1),
            recommendation=recommendation,
        )

    def identify_hotspots(
        self,
        vessels: List[Tuple[float, float, float, float]],
        time_window: Tuple[float, float] = (0.0, float('inf')),
    ) -> List[HotspotArea]:
        """
        Identify traffic hotspot areas where vessels cluster.
        vessels: list of (lat, lon, speed, timestamp)
        time_window: (start_time, end_time) to filter
        """
        filtered = [
            v for v in vessels
            if time_window[0] <= v[3] <= time_window[1]
        ]

        if len(filtered) < 3:
            return []

        # Simple grid-based clustering
        grid_size = 0.05  # ~3nm grid
        grid: Dict[Tuple[int, int], List] = defaultdict(list)
        for v in filtered:
            key = (int(v[0] / grid_size), int(v[1] / grid_size))
            grid[key].append(v)

        hotspots = []
        for key, points in grid.items():
            if len(points) >= 3:
                avg_lat = sum(p[0] for p in points) / len(points)
                avg_lon = sum(p[1] for p in points) / len(points)
                avg_speed = sum(p[2] for p in points) / len(points)
                radius = grid_size * 60.0 / 2.0  # nm

                hotspots.append(HotspotArea(
                    center=(avg_lat, avg_lon),
                    radius=round(radius, 2),
                    vessel_count=len(points),
                    avg_speed=round(avg_speed, 1),
                    time_window=time_window,
                ))

        hotspots.sort(key=lambda h: h.vessel_count, reverse=True)
        return hotspots

    def compute_flow_rate(
        self,
        boundary: Tuple[float, float, float, float],
        vessels: List[Tuple[float, float, float, float, float]],
        time_window: Tuple[float, float],
    ) -> float:
        """
        Compute vessel flow rate across a boundary line.
        boundary: (lat1, lon1, lat2, lon2) defining a line
        vessels: list of (lat, lon, speed, heading, timestamp)
        time_window: (start, end)
        Returns vessels per hour.
        """
        # Filter by time
        filtered = [
            v for v in vessels
            if time_window[0] <= v[4] <= time_window[1]
        ]

        if not filtered:
            return 0.0

        # Count vessels crossing the boundary line
        crossings = 0
        lat1, lon1, lat2, lon2 = boundary

        for v in filtered:
            # Simple crossing detection: vessel within corridor along boundary
            dist_to_line = self._point_line_distance(
                v[0], v[1], lat1, lon1, lat2, lon2
            )
            if dist_to_line < 0.02:  # ~1.2 nm corridor
                crossings += 1

        duration_hours = (time_window[1] - time_window[0]) / 3600.0
        if duration_hours <= 0:
            return 0.0

        flow_rate = crossings / duration_hours
        return round(flow_rate, 2)

    def analyze_seasonal_patterns(
        self,
        historical_data: Dict[int, List[Tuple[float, float, float]]],
    ) -> List[SeasonalTrend]:
        """
        Analyze seasonal traffic patterns from historical data.
        historical_data: {month: [(timestamp, vessel_count, avg_speed), ...]}
        Returns list of SeasonalTrend sorted by month.
        """
        trends = []
        all_months_data: Dict[int, List[float]] = defaultdict(list)
        all_months_speeds: Dict[int, List[float]] = defaultdict(list)

        for month, data_points in historical_data.items():
            counts = [d[1] for d in data_points]
            speeds = [d[2] for d in data_points]
            all_months_data[month].extend(counts)
            all_months_speeds[month].extend(speeds)

        prev_count = 0
        for month in range(1, 13):
            counts = all_months_data.get(month, [])
            speeds = all_months_speeds.get(month, [])

            if counts:
                avg_count = sum(counts) / len(counts)
                avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
                # Determine peak hour from timestamp data (simplified: hour of day)
                hours = [int(d[0]) % 24 for d in historical_data.get(month, [])]
                peak_hour = max(set(hours), key=hours.count) if hours else 12
            else:
                avg_count = 0.0
                avg_speed = 0.0
                peak_hour = 12

            # Trend direction
            if prev_count == 0:
                trend_dir = "stable"
            elif avg_count > prev_count * 1.1:
                trend_dir = "increasing"
            elif avg_count < prev_count * 0.9:
                trend_dir = "decreasing"
            else:
                trend_dir = "stable"

            common_patterns = []
            if avg_count > 50:
                common_patterns.append("heavy_traffic")
            if avg_speed < 5:
                common_patterns.append("fishing_activity")
            if avg_speed > 15:
                common_patterns.append("fast_vessel_corridor")
            if not common_patterns:
                common_patterns.append("normal")

            trends.append(SeasonalTrend(
                month=month,
                avg_vessel_count=round(avg_count, 1),
                peak_hour=peak_hour,
                common_patterns=common_patterns,
                density_trend=trend_dir,
            ))
            prev_count = avg_count

        return trends

    @staticmethod
    def _circular_mean(angles: List[float]) -> float:
        """Compute circular mean of angles in degrees."""
        if not angles:
            return 0.0
        sin_sum = sum(math.sin(math.radians(a)) for a in angles)
        cos_sum = sum(math.cos(math.radians(a)) for a in angles)
        return math.degrees(math.atan2(sin_sum, cos_sum)) % 360

    @staticmethod
    def _point_line_distance(
        lat: float, lon: float,
        lat1: float, lon1: float,
        lat2: float, lon2: float,
    ) -> float:
        """Compute approximate distance from point to line segment in degrees."""
        dx = lon2 - lon1
        dy = lat2 - lat1
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-12:
            return math.sqrt((lon - lon1) ** 2 + (lat - lat1) ** 2)

        t = max(0, min(1, ((lon - lon1) * dx + (lat - lat1) * dy) / length_sq))
        proj_lon = lon1 + t * dx
        proj_lat = lat1 + t * dy
        return math.sqrt((lon - proj_lon) ** 2 + (lat - proj_lat) ** 2)
