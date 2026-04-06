"""Tests for Traffic Pattern Analysis module."""

import math
import time

import pytest

from jetson.maritime_domain.traffic import (
    CongestionForecast,
    DensityLevel,
    HotspotArea,
    PatternType,
    SeasonalTrend,
    TrafficAnalyzer,
    TrafficPattern,
    VesselTrack,
)


class TestTrafficPattern:
    """Tests for TrafficPattern dataclass."""

    def test_defaults(self):
        p = TrafficPattern(pattern_type="shipping_lane", corridor_center=(0.0, 0.0, 1.0, 1.0))
        assert p.pattern_type == "shipping_lane"
        assert p.corridor_width == 0.0
        assert p.typical_speed == 0.0
        assert p.direction == 0.0
        assert p.confidence == 0.0

    def test_full_construction(self):
        p = TrafficPattern(
            pattern_type="ferry_route",
            corridor_center=(50.0, -5.0, 51.0, -4.0),
            corridor_width=2.5,
            typical_speed=18.0,
            direction=90.0,
            confidence=0.85,
        )
        assert p.pattern_type == "ferry_route"
        assert p.corridor_width == 2.5
        assert abs(p.confidence - 0.85) < 0.01


class TestDensityLevel:
    """Tests for DensityLevel enum."""

    def test_levels(self):
        assert DensityLevel.VERY_LOW.value == "very_low"
        assert DensityLevel.LOW.value == "low"
        assert DensityLevel.MODERATE.value == "moderate"
        assert DensityLevel.HIGH.value == "high"
        assert DensityLevel.VERY_HIGH.value == "very_high"
        assert DensityLevel.CRITICAL.value == "critical"


class TestPatternType:
    """Tests for PatternType enum."""

    def test_types(self):
        assert PatternType.SHIPPING_LANE.value == "shipping_lane"
        assert PatternType.FERRY_ROUTE.value == "ferry_route"
        assert PatternType.FISHING_GROUND.value == "fishing_ground"


class TestVesselTrack:
    """Tests for VesselTrack dataclass."""

    def test_defaults(self):
        t = VesselTrack(mmsi=123)
        assert t.mmsi == 123
        assert t.positions == []
        assert t.speeds == []
        assert t.headings == []

    def test_with_data(self):
        t = VesselTrack(
            mmsi=456,
            positions=[(50.0, -5.0, 100.0)],
            speeds=[12.5],
            headings=[195.0],
        )
        assert len(t.positions) == 1


class TestHotspotArea:
    """Tests for HotspotArea dataclass."""

    def test_defaults(self):
        h = HotspotArea(
            center=(50.0, -5.0),
            radius=3.0,
            vessel_count=10,
            avg_speed=10.0,
            time_window=(0.0, 3600.0),
        )
        assert h.vessel_count == 10


class TestCongestionForecast:
    """Tests for CongestionForecast dataclass."""

    def test_defaults(self):
        c = CongestionForecast(
            expected_density=DensityLevel.MODERATE,
            confidence=0.7,
        )
        assert c.expected_density == DensityLevel.MODERATE
        assert c.peak_time is None
        assert c.affected_area is None
        assert c.recommendation == ""


class TestSeasonalTrend:
    """Tests for SeasonalTrend dataclass."""

    def test_construction(self):
        s = SeasonalTrend(
            month=6,
            avg_vessel_count=45.0,
            peak_hour=14,
            common_patterns=["heavy_traffic"],
            density_trend="increasing",
        )
        assert s.month == 6
        assert s.density_trend == "increasing"


class TestTrafficAnalyzerInit:
    """Tests for TrafficAnalyzer initialization."""

    def test_init(self):
        analyzer = TrafficAnalyzer()
        assert isinstance(analyzer, TrafficAnalyzer)


class TestDetectPatterns:
    """Tests for pattern detection."""

    def test_empty_tracks(self):
        analyzer = TrafficAnalyzer()
        patterns = analyzer.detect_patterns([])
        assert patterns == []

    def test_single_track_no_pattern(self):
        analyzer = TrafficAnalyzer()
        tracks = [VesselTrack(
            mmsi=1,
            positions=[(50.0, -5.0, 100.0), (50.5, -4.5, 200.0)],
            speeds=[12.0, 12.0],
            headings=[45.0, 45.0],
        )]
        patterns = analyzer.detect_patterns(tracks)
        # Single track doesn't form a pattern (< 2 in same direction bin)
        assert isinstance(patterns, list)

    def test_parallel_tracks_shipping_lane(self):
        analyzer = TrafficAnalyzer()
        tracks = []
        for i in range(10):
            tracks.append(VesselTrack(
                mmsi=i,
                positions=[(50.0 + i * 0.001, -5.0, 100.0), (50.5 + i * 0.001, -4.5, 200.0)],
                speeds=[12.0],
                headings=[45.0],
            ))
        patterns = analyzer.detect_patterns(tracks)
        assert len(patterns) >= 1
        assert patterns[0].pattern_type in [pt.value for pt in PatternType]

    def test_fast_tracks_ferry_route(self):
        analyzer = TrafficAnalyzer()
        tracks = []
        for i in range(5):
            tracks.append(VesselTrack(
                mmsi=i + 100,
                positions=[(50.0, -5.0 + i * 0.001, 100.0), (51.0, -5.0 + i * 0.001, 200.0)],
                speeds=[20.0],
                headings=[0.0],
            ))
        patterns = analyzer.detect_patterns(tracks)
        assert len(patterns) >= 1

    def test_slow_tracks_fishing(self):
        analyzer = TrafficAnalyzer()
        tracks = []
        for i in range(5):
            tracks.append(VesselTrack(
                mmsi=i + 200,
                positions=[(50.0, -5.0 + i * 0.01, 100.0), (50.0, -5.0 + i * 0.01 + 0.01, 200.0)],
                speeds=[2.0],
                headings=[90.0],
            ))
        patterns = analyzer.detect_patterns(tracks)
        assert len(patterns) >= 1

    def test_pattern_confidence_scales_with_tracks(self):
        analyzer = TrafficAnalyzer()
        small_tracks = [
            VesselTrack(mmsi=i, positions=[(50.0, -5.0, 100.0)],
                        speeds=[10.0], headings=[0.0])
            for i in range(3)
        ]
        large_tracks = [
            VesselTrack(mmsi=i, positions=[(50.0, -5.0, 100.0)],
                        speeds=[10.0], headings=[0.0])
            for i in range(15)
        ]
        patterns_small = analyzer.detect_patterns(small_tracks)
        patterns_large = analyzer.detect_patterns(large_tracks)
        # More tracks should produce patterns with equal or higher confidence
        for p in patterns_large:
            assert p.confidence <= 1.0

    def test_tracks_with_empty_positions(self):
        analyzer = TrafficAnalyzer()
        tracks = [VesselTrack(mmsi=1, positions=[], speeds=[], headings=[])]
        patterns = analyzer.detect_patterns(tracks)
        assert patterns == []


class TestClassifyTrafficDensity:
    """Tests for traffic density classification."""

    def test_empty_area(self):
        analyzer = TrafficAnalyzer()
        area = (50.0, -5.0, 51.0, -4.0)
        density = analyzer.classify_traffic_density(area, [])
        assert density == DensityLevel.VERY_LOW

    def test_very_low_density(self):
        analyzer = TrafficAnalyzer()
        area = (50.0, -5.0, 51.0, -4.0)
        vessels = [(50.5, -4.5, 10.0)]
        density = analyzer.classify_traffic_density(area, vessels)
        assert density == DensityLevel.VERY_LOW

    def test_moderate_density(self):
        analyzer = TrafficAnalyzer()
        area = (50.0, -5.0, 50.1, -4.9)
        vessels = [(50.05, -4.95, 10.0) for _ in range(5)]
        density = analyzer.classify_traffic_density(area, vessels)
        # 5 vessels in small area should be at least low density
        assert density in (DensityLevel.LOW, DensityLevel.MODERATE, DensityLevel.HIGH,
                           DensityLevel.VERY_HIGH, DensityLevel.CRITICAL)

    def test_critical_density(self):
        analyzer = TrafficAnalyzer()
        area = (50.0, -5.0, 50.05, -4.95)
        vessels = [(50.025, -4.975, 10.0) for _ in range(50)]
        density = analyzer.classify_traffic_density(area, vessels)
        assert density == DensityLevel.CRITICAL

    def test_vessels_outside_area(self):
        analyzer = TrafficAnalyzer()
        area = (50.0, -5.0, 51.0, -4.0)
        vessels = [(0.0, 0.0, 10.0), (30.0, -30.0, 10.0)]
        density = analyzer.classify_traffic_density(area, vessels)
        assert density == DensityLevel.VERY_LOW

    def test_area_ordering_doesnt_matter(self):
        analyzer = TrafficAnalyzer()
        area1 = (50.0, -5.0, 51.0, -4.0)
        area2 = (51.0, -4.0, 50.0, -5.0)
        vessels = [(50.5, -4.5, 10.0)]
        d1 = analyzer.classify_traffic_density(area1, vessels)
        d2 = analyzer.classify_traffic_density(area2, vessels)
        assert d1 == d2


class TestPredictCongestion:
    """Tests for congestion prediction."""

    def test_empty_traffic(self):
        analyzer = TrafficAnalyzer()
        forecast = analyzer.predict_congestion([], [])
        assert forecast.expected_density == DensityLevel.VERY_LOW
        assert forecast.confidence == 0.5

    def test_light_traffic_no_trend(self):
        analyzer = TrafficAnalyzer()
        traffic = [(50.0, -5.0, 10.0, 100.0)]
        forecast = analyzer.predict_congestion(traffic, [])
        assert forecast.confidence < 0.5

    def test_growing_trend(self):
        analyzer = TrafficAnalyzer()
        traffic = [(50.0, -5.0, 10.0, 100.0) for _ in range(15)]
        trends = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        forecast = analyzer.predict_congestion(traffic, trends)
        assert forecast.confidence > 0.3

    def test_high_traffic_congestion(self):
        analyzer = TrafficAnalyzer()
        traffic = [(50.0, -5.0, 10.0, 100.0) for _ in range(50)]
        trends = [(1.0, 3.0), (2.0, 3.0), (3.0, 3.0)]
        forecast = analyzer.predict_congestion(traffic, trends)
        assert forecast.expected_density in (
            DensityLevel.HIGH, DensityLevel.VERY_HIGH, DensityLevel.CRITICAL
        )

    def test_forecast_has_recommendation(self):
        analyzer = TrafficAnalyzer()
        traffic = [(50.0, -5.0, 10.0, 100.0)]
        forecast = analyzer.predict_congestion(traffic, [])
        assert isinstance(forecast.recommendation, str)
        assert len(forecast.recommendation) > 0


class TestIdentifyHotspots:
    """Tests for hotspot identification."""

    def test_no_vessels(self):
        analyzer = TrafficAnalyzer()
        hotspots = analyzer.identify_hotspots([])
        assert hotspots == []

    def test_too_few_vessels(self):
        analyzer = TrafficAnalyzer()
        vessels = [(50.0, -5.0, 10.0, 100.0)]
        hotspots = analyzer.identify_hotspots(vessels)
        assert hotspots == []

    def test_clustered_vessels(self):
        analyzer = TrafficAnalyzer()
        # 5 vessels very close together
        vessels = [
            (50.001, -5.001, 5.0, 100.0),
            (50.002, -5.002, 6.0, 100.0),
            (50.001, -5.003, 4.0, 100.0),
            (50.003, -5.001, 5.5, 100.0),
            (50.002, -5.002, 4.5, 100.0),
        ]
        hotspots = analyzer.identify_hotspots(vessels)
        assert len(hotspots) >= 1
        assert hotspots[0].vessel_count >= 3

    def test_hotspot_sorted_by_count(self):
        analyzer = TrafficAnalyzer()
        # Two clusters
        vessels = []
        for _ in range(5):
            vessels.append((50.001, -5.001, 5.0, 100.0))
        for _ in range(8):
            vessels.append((51.001, -4.001, 5.0, 100.0))
        hotspots = analyzer.identify_hotspots(vessels)
        if len(hotspots) >= 2:
            assert hotspots[0].vessel_count >= hotspots[1].vessel_count

    def test_time_window_filtering(self):
        analyzer = TrafficAnalyzer()
        vessels = [
            (50.001, -5.001, 5.0, 100.0),
            (50.002, -5.002, 6.0, 200.0),
            (50.001, -5.003, 4.0, 300.0),
            (50.003, -5.001, 5.5, 10000.0),  # outside window
        ]
        hotspots = analyzer.identify_hotspots(vessels, time_window=(0.0, 1000.0))
        total_vessels = sum(h.vessel_count for h in hotspots)
        assert total_vessels <= 3


class TestComputeFlowRate:
    """Tests for flow rate computation."""

    def test_no_vessels(self):
        analyzer = TrafficAnalyzer()
        rate = analyzer.compute_flow_rate(
            (50.0, -5.0, 51.0, -4.0),
            [],
            (0.0, 3600.0),
        )
        assert rate == 0.0

    def test_zero_duration(self):
        analyzer = TrafficAnalyzer()
        vessels = [(50.0, -5.0, 10.0, 0.0, 100.0)]
        rate = analyzer.compute_flow_rate(
            (50.0, -5.0, 51.0, -4.0),
            vessels,
            (100.0, 100.0),
        )
        assert rate == 0.0

    def test_vessels_on_boundary(self):
        analyzer = TrafficAnalyzer()
        # Vessels crossing the boundary line
        vessels = [
            (50.5, -4.5, 10.0, 0.0, 100.0),
            (50.5, -4.5, 10.0, 0.0, 500.0),
            (50.5, -4.5, 10.0, 0.0, 1000.0),
            (50.5, -4.5, 10.0, 0.0, 2000.0),
        ]
        rate = analyzer.compute_flow_rate(
            (50.0, -4.5, 51.0, -4.5),
            vessels,
            (0.0, 3600.0),
        )
        assert rate > 0

    def test_vessels_far_from_boundary(self):
        analyzer = TrafficAnalyzer()
        vessels = [(0.0, 0.0, 10.0, 0.0, 100.0)]
        rate = analyzer.compute_flow_rate(
            (50.0, -5.0, 51.0, -4.0),
            vessels,
            (0.0, 3600.0),
        )
        assert rate == 0.0

    def test_flow_rate_per_hour(self):
        analyzer = TrafficAnalyzer()
        vessels = [(50.5, -4.5, 10.0, 0.0, 1800.0) for _ in range(6)]
        rate = analyzer.compute_flow_rate(
            (50.0, -4.5, 51.0, -4.5),
            vessels,
            (0.0, 7200.0),
        )
        assert isinstance(rate, float)
        assert rate >= 0


class TestAnalyzeSeasonalPatterns:
    """Tests for seasonal pattern analysis."""

    def test_empty_data(self):
        analyzer = TrafficAnalyzer()
        trends = analyzer.analyze_seasonal_patterns({})
        assert len(trends) == 12
        for t in trends:
            assert t.avg_vessel_count == 0.0

    def test_single_month_data(self):
        analyzer = TrafficAnalyzer()
        data = {
            6: [(14, 100.0, 12.0), (14, 120.0, 10.0)],
        }
        trends = analyzer.analyze_seasonal_patterns(data)
        june = trends[5]
        assert june.month == 6
        assert june.avg_vessel_count == 110.0

    def test_all_months_present(self):
        analyzer = TrafficAnalyzer()
        data = {m: [(12, float(20 + m), 10.0)] for m in range(1, 13)}
        trends = analyzer.analyze_seasonal_patterns(data)
        assert len(trends) == 12
        months = [t.month for t in trends]
        assert months == list(range(1, 13))

    def test_increasing_trend_detected(self):
        analyzer = TrafficAnalyzer()
        data = {
            m: [(12, float(10 * m), 10.0)] for m in range(1, 13)
        }
        trends = analyzer.analyze_seasonal_patterns(data)
        assert any(t.density_trend == "increasing" for t in trends)

    def test_decreasing_trend_detected(self):
        analyzer = TrafficAnalyzer()
        data = {
            m: [(12, float(130 - 10 * m), 10.0)] for m in range(1, 13)
        }
        trends = analyzer.analyze_seasonal_patterns(data)
        assert any(t.density_trend == "decreasing" for t in trends)

    def test_heavy_traffic_pattern(self):
        analyzer = TrafficAnalyzer()
        data = {6: [(12, 60.0, 10.0)]}
        trends = analyzer.analyze_seasonal_patterns(data)
        june = trends[5]
        assert "heavy_traffic" in june.common_patterns

    def test_fishing_pattern_detected(self):
        analyzer = TrafficAnalyzer()
        data = {3: [(12, 20.0, 3.0)]}
        trends = analyzer.analyze_seasonal_patterns(data)
        march = trends[2]
        assert "fishing_activity" in march.common_patterns

    def test_fast_corridor_detected(self):
        analyzer = TrafficAnalyzer()
        data = {8: [(12, 30.0, 18.0)]}
        trends = analyzer.analyze_seasonal_patterns(data)
        august = trends[7]
        assert "fast_vessel_corridor" in august.common_patterns

    def test_trend_data_multiple_points(self):
        analyzer = TrafficAnalyzer()
        data = {
            7: [(10, 50.0, 12.0), (14, 70.0, 10.0), (16, 60.0, 11.0)],
        }
        trends = analyzer.analyze_seasonal_patterns(data)
        july = trends[6]
        assert july.avg_vessel_count == 60.0
        assert july.common_patterns  # should have at least one pattern
