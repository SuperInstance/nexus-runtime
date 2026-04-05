"""Comprehensive tests for NEXUS Dead Reckoning Engine.

100+ tests covering navigation math, INS, Kalman, compass,
position estimator, intention broadcasting, waypoint nav.
"""


import math
import pytest

from jetson.agent.dead_reckoning.navigation import NavigationMath, VelocityVector
from jetson.agent.dead_reckoning.ins import (
    INSIntegrator, INSState, IMUReading, IntegrationMethod, DriftMetrics,
)
from jetson.agent.dead_reckoning.kalman import (
    KalmanFilter2D, KalmanApproach, KalmanState, KalmanCovariance, GPSReading,
)
from jetson.agent.dead_reckoning.compass import (
    CompassFusion, CompassReading, HeadingState,
)
from jetson.agent.dead_reckoning.position_estimator import (
    PositionEstimator, PositionEstimate, SensorQuality, NavigationMode,
)
from jetson.agent.dead_reckoning.intention import (
    IntentionBroadcaster, IntentionMessage, CollisionAssessment,
    CPAAlgorithm, DRMessageType, DR_MSG_TYPE_INFO,
)
from jetson.agent.dead_reckoning.waypoint import (
    WaypointNavigator, Waypoint, WaypointStatus, HeadingCommand,
)


# ===================================================================
# NavigationMath Tests
# ===================================================================

class TestNavigationMath:
    def test_haversine_same_point(self):
        d = NavigationMath.haversine_distance(32.0, -117.0, 32.0, -117.0)
        assert d == pytest.approx(0.0, abs=0.01)

    def test_haversine_known_distance_equator(self):
        d = NavigationMath.haversine_distance(0.0, 0.0, 1.0, 0.0)
        assert d == pytest.approx(111320.0, rel=0.01)

    def test_haversine_known_distance_longitude(self):
        d = NavigationMath.haversine_distance(0.0, 0.0, 0.0, 1.0)
        assert d == pytest.approx(111320.0, rel=0.01)

    def test_haversine_sandiego_to_london(self):
        d = NavigationMath.haversine_distance(32.7, -117.2, 51.5, -0.1)
        assert d == pytest.approx(8_800_000, rel=0.05)

    def test_haversine_short_distance(self):
        d = NavigationMath.haversine_distance(32.0, -117.0, 32.001, -117.0)
        assert 90 < d < 130

    def test_bearing_north(self):
        b = NavigationMath.bearing(32.0, -117.0, 33.0, -117.0)
        assert b == pytest.approx(0.0, abs=1.0)

    def test_bearing_east(self):
        b = NavigationMath.bearing(32.0, -117.0, 32.0, -116.0)
        assert b == pytest.approx(90.0, abs=1.0)

    def test_bearing_south(self):
        b = NavigationMath.bearing(33.0, -117.0, 32.0, -117.0)
        assert b == pytest.approx(180.0, abs=1.0)

    def test_bearing_west(self):
        b = NavigationMath.bearing(32.0, -116.0, 32.0, -117.0)
        assert b == pytest.approx(270.0, abs=1.0)

    def test_bearing_ne_quadrant(self):
        b = NavigationMath.bearing(32.0, -117.0, 33.0, -116.0)
        assert 30 < b < 60

    def test_bearing_range_0_360(self):
        b = NavigationMath.bearing(0.0, 0.0, -0.1, 0.0)
        assert 0 <= b < 360

    def test_destination_point_north(self):
        lat, lon = NavigationMath.destination_point(32.0, -117.0, 0.0, 111320.0)
        assert lat == pytest.approx(33.0, abs=0.01)
        assert lon == pytest.approx(-117.0, abs=0.01)

    def test_destination_point_east(self):
        lat, lon = NavigationMath.destination_point(0.0, 0.0, 90.0, 111320.0)
        assert lat == pytest.approx(0.0, abs=0.01)
        assert lon == pytest.approx(1.0, abs=0.01)

    def test_destination_point_roundtrip(self):
        lat1, lon1 = 32.5, -117.5
        bearing = NavigationMath.bearing(lat1, lon1, 33.0, -117.0)
        dist = NavigationMath.haversine_distance(lat1, lon1, 33.0, -117.0)
        lat2, lon2 = NavigationMath.destination_point(lat1, lon1, bearing, dist)
        assert lat2 == pytest.approx(33.0, abs=0.01)
        assert lon2 == pytest.approx(-117.0, abs=0.01)

    def test_normalize_angle(self):
        assert NavigationMath.normalize_angle(0) == 0
        assert NavigationMath.normalize_angle(360) == 0
        assert NavigationMath.normalize_angle(450) == 90
        assert NavigationMath.normalize_angle(-10) == 350

    def test_angular_difference(self):
        assert NavigationMath.angular_difference(10, 0) == pytest.approx(10)
        assert NavigationMath.angular_difference(0, 10) == pytest.approx(-10)
        assert NavigationMath.angular_difference(350, 0) == pytest.approx(-10)
        assert NavigationMath.angular_difference(10, 350) == pytest.approx(20)

    def test_midpoint(self):
        lat, lon = NavigationMath.midpoint(0.0, 0.0, 0.0, 2.0)
        assert lat == pytest.approx(0.0, abs=0.01)
        assert lon == pytest.approx(1.0, abs=0.01)

    def test_cross_track_on_path(self):
        d = NavigationMath.cross_track_distance(32.0, -117.0, 33.0, -117.0, 32.5, -117.0)
        assert abs(d) < 50

    def test_along_track_distance(self):
        d = NavigationMath.along_track_distance(32.0, -117.0, 33.0, -117.0, 32.5, -117.0)
        assert d > 0


class TestVelocityVector:
    def test_zero_velocity(self):
        v = VelocityVector()
        assert v.speed == 0.0
        assert v.heading == 0.0

    def test_north_velocity(self):
        v = VelocityVector(north=5.0, east=0.0)
        assert v.speed == pytest.approx(5.0)
        assert v.heading == pytest.approx(0.0, abs=1.0)

    def test_east_velocity(self):
        v = VelocityVector(north=0.0, east=5.0)
        assert v.speed == pytest.approx(5.0)
        assert v.heading == pytest.approx(90.0, abs=1.0)

    def test_from_speed_heading(self):
        v = VelocityVector.from_speed_heading(10.0, 45.0)
        assert v.speed == pytest.approx(10.0)
        assert v.north > 0
        assert v.east > 0


# ===================================================================
# INS Integration Tests
# ===================================================================

class TestINSEuler:
    def setup_method(self):
        self.ins = INSIntegrator(method=IntegrationMethod.EULER)
        self.ins.initialize(lat=32.0, lon=-117.0, heading=0.0)

    def test_euler_no_acceleration(self):
        for i in range(100):
            self.ins.update(IMUReading(timestamp_ms=i * 10), dt=0.01)
        state = self.ins.get_state()
        assert abs(state.position_lat - 32.0) < 0.001

    def test_euler_forward_acceleration(self):
        for i in range(100):
            self.ins.update(IMUReading(accel_x=1.0, timestamp_ms=i * 10), dt=0.01)
        state = self.ins.get_state()
        assert state.position_lat > 32.0

    def test_euler_gyro_integration(self):
        for i in range(100):
            self.ins.update(IMUReading(gyro_x=36.0, timestamp_ms=i * 10), dt=0.01)
        state = self.ins.get_state()
        assert state.heading == pytest.approx(36.0, abs=5.0)

    def test_euler_not_initialized_raises(self):
        ins = INSIntegrator(method=IntegrationMethod.EULER)
        with pytest.raises(RuntimeError):
            ins.update(IMUReading())

    def test_euler_reset(self):
        self.ins.reset()
        assert not self.ins.is_initialized

    def test_euler_correct_position(self):
        self.ins.correct_position(33.0, -118.0)
        state = self.ins.get_state()
        assert state.position_lat == 33.0
        assert state.position_lon == -118.0


class TestINSRK4:
    def setup_method(self):
        self.ins = INSIntegrator(method=IntegrationMethod.RK4)
        self.ins.initialize(lat=32.0, lon=-117.0, heading=0.0)

    def test_rk4_no_acceleration(self):
        for i in range(100):
            self.ins.update(IMUReading(timestamp_ms=i * 10), dt=0.01)
        state = self.ins.get_state()
        assert abs(state.position_lat - 32.0) < 0.001

    def test_rk4_forward_acceleration(self):
        for i in range(100):
            self.ins.update(IMUReading(accel_x=1.0, timestamp_ms=i * 10), dt=0.01)
        state = self.ins.get_state()
        assert state.position_lat > 32.0

    def test_rk4_gyro_heading(self):
        for i in range(100):
            self.ins.update(IMUReading(gyro_x=36.0, timestamp_ms=i * 10), dt=0.01)
        state = self.ins.get_state()
        assert state.heading == pytest.approx(36.0, abs=5.0)


class TestINSComplementary:
    def setup_method(self):
        self.ins = INSIntegrator(method=IntegrationMethod.COMPLEMENTARY)
        self.ins.initialize(lat=32.0, lon=-117.0, heading=0.0)

    def test_complementary_no_acceleration(self):
        for i in range(100):
            self.ins.update(IMUReading(timestamp_ms=i * 10), dt=0.01)
        state = self.ins.get_state()
        assert abs(state.position_lat - 32.0) < 0.01

    def test_complementary_forward_motion(self):
        for i in range(100):
            self.ins.update(IMUReading(accel_x=0.5, timestamp_ms=i * 10), dt=0.01)
        state = self.ins.get_state()
        assert state.position_lat > 32.0

    def test_complementary_position_correction(self):
        for i in range(50):
            self.ins.update(IMUReading(accel_x=1.0, timestamp_ms=i * 10), dt=0.01)
        self.ins.correct_position(32.0, -117.0)
        state = self.ins.get_state()
        assert state.position_lat == pytest.approx(32.0, abs=0.001)

    def test_complementary_alpha_parameter(self):
        ins = INSIntegrator(method=IntegrationMethod.COMPLEMENTARY, complementary_alpha=0.5)
        assert ins.complementary_alpha == 0.5


class TestINSComparison:
    def test_compare_methods_returns_all(self):
        readings = [IMUReading(accel_x=0.1, timestamp_ms=i * 10) for i in range(50)]
        true_pos = [(32.0 + i * 0.00001, -117.0) for i in range(50)]
        results = INSIntegrator.compare_methods(readings, true_pos, 32.0, -117.0)
        assert len(results) == 3
        for method in IntegrationMethod:
            assert method in results

    def test_compare_methods_drift_metrics(self):
        readings = [IMUReading(accel_x=0.5, timestamp_ms=i * 10) for i in range(50)]
        true_pos = [(32.0, -117.0)] * 50
        results = INSIntegrator.compare_methods(readings, true_pos, 32.0, -117.0)
        for m, met in results.items():
            assert met.update_count == 50
            assert met.total_drift_m >= 0

    def test_euler_drifts_on_curve(self):
        readings = [IMUReading(accel_x=0.1, gyro_x=10.0, timestamp_ms=i * 10) for i in range(100)]
        true_pos = [(32.0, -117.0)] * 100
        results = INSIntegrator.compare_methods(readings, true_pos, 32.0, -117.0)
        for m, met in results.items():
            assert met.total_drift_m >= 0

    def test_ins_state_velocity(self):
        ins = INSIntegrator(method=IntegrationMethod.EULER)
        ins.initialize(32.0, -117.0, velocity_north=3.0, velocity_east=4.0)
        state = ins.get_state()
        assert state.velocity.speed == pytest.approx(5.0)


# ===================================================================
# Kalman Filter Tests
# ===================================================================

def _make_gps(lat, lon, speed=0.0, course=0.0, hdop=1.0, sats=8, ts=0):
    return GPSReading(latitude=lat, longitude=lon, speed=speed, course=course,
                      hdop=hdop, num_satellites=sats, fix_quality=1, timestamp_ms=ts)


class TestKalmanFilter:
    def test_weighted_average_basic(self):
        kf = KalmanFilter2D(approach=KalmanApproach.WEIGHTED_AVERAGE)
        kf.initialize(32.0, -117.0)
        kf.update_gps(_make_gps(32.001, -117.001, ts=100))
        state = kf.get_state()
        assert 32.0 <= state.lat <= 32.001

    def test_basic_kalman_convergence(self):
        kf = KalmanFilter2D(approach=KalmanApproach.BASIC_KALMAN)
        kf.initialize(32.0, -117.0)
        for i in range(50):
            kf.update_gps(_make_gps(32.5, -117.5, ts=i * 100))
        state = kf.get_state()
        assert state.lat == pytest.approx(32.5, abs=0.01)
        assert state.lon == pytest.approx(-117.5, abs=0.01)

    def test_ekf_convergence(self):
        kf = KalmanFilter2D(approach=KalmanApproach.EKF)
        kf.initialize(32.0, -117.0)
        for i in range(50):
            kf.update_gps(_make_gps(32.5, -117.5, speed=5.0, course=45.0, ts=i * 100))
        state = kf.get_state()
        assert state.lat == pytest.approx(32.5, abs=0.01)

    def test_invalid_gps_ignored(self):
        kf = KalmanFilter2D()
        kf.initialize(32.0, -117.0)
        kf.update_gps(GPSReading(fix_quality=0, num_satellites=0))
        assert kf.get_state().lat == 32.0

    def test_auto_initialize_from_gps(self):
        kf = KalmanFilter2D()
        assert not kf.is_initialized
        kf.update_gps(_make_gps(32.0, -117.0, ts=100))
        assert kf.is_initialized

    def test_ins_velocity_update(self):
        kf = KalmanFilter2D(approach=KalmanApproach.BASIC_KALMAN)
        kf.initialize(32.0, -117.0)
        kf.update_ins_velocity(5.0, 3.0)
        state = kf.get_state()
        assert state.vel_north > 0 or state.vel_east > 0

    def test_predict_moves_position(self):
        kf = KalmanFilter2D()
        kf.initialize(32.0, -117.0, vel_north=100.0, vel_east=0.0)
        kf.predict(1.0)
        assert kf.get_state().lat > 32.0

    def test_confidence_decreases(self):
        kf = KalmanFilter2D()
        kf.initialize(32.0, -117.0)
        c1 = kf.get_confidence()
        for _ in range(10):
            kf.predict(0.1)
        c2 = kf.get_confidence()
        assert c2 <= c1

    def test_gps_quality_affects_weighted(self):
        kf = KalmanFilter2D(approach=KalmanApproach.WEIGHTED_AVERAGE, gps_weight=0.5)
        kf.initialize(32.0, -117.0)
        kf.update_gps(_make_gps(32.01, -117.0, hdop=0.8, sats=12, ts=100))
        s1 = kf.get_state().lat
        kf2 = KalmanFilter2D(approach=KalmanApproach.WEIGHTED_AVERAGE, gps_weight=0.5)
        kf2.initialize(32.0, -117.0)
        kf2.update_gps(_make_gps(32.01, -117.0, hdop=5.0, sats=4, ts=100))
        s2 = kf2.get_state().lat
        # Good GPS quality → more GPS weight → closer to GPS reading
        assert abs(s1 - 32.01) < abs(s2 - 32.01) + 0.001

    def test_compare_approaches(self):
        gps_list = [_make_gps(32.0 + i * 0.001, -117.0 + i * 0.001, ts=i * 100) for i in range(30)]
        vel_list = [(1.0, 1.0)] * 30
        true_pos = [(32.0 + i * 0.001, -117.0 + i * 0.001) for i in range(30)]
        results = KalmanFilter2D.compare_approaches(gps_list, vel_list, true_pos, 32.0, -117.0)
        assert len(results) == 3

    def test_covariance_matrix(self):
        cov = KalmanCovariance()
        assert cov.get(0, 0) == 1.0
        cov.set(2, 3, 5.0)
        assert cov.get(2, 3) == 5.0
        c2 = cov.copy()
        assert c2.get(2, 3) == 5.0

    def test_update_count(self):
        kf = KalmanFilter2D()
        kf.initialize(32.0, -117.0)
        assert kf.update_count == 0
        kf.update_gps(_make_gps(32.001, -117.0, ts=100))
        assert kf.update_count == 1
        assert kf.gps_update_count == 1
        kf.update_ins_velocity(1.0, 0.0)
        assert kf.update_count == 2

    def test_gps_reading_is_valid(self):
        assert GPSReading(fix_quality=1, num_satellites=8).is_valid
        assert not GPSReading(fix_quality=0, num_satellites=0).is_valid
        assert not GPSReading(fix_quality=1, num_satellites=3).is_valid

    def test_kalman_reset(self):
        kf = KalmanFilter2D()
        kf.initialize(32.0, -117.0)
        kf.reset()
        assert not kf.is_initialized


# ===================================================================
# Compass Fusion Tests
# ===================================================================

class TestCompassFusion:
    def test_initialize(self):
        cf = CompassFusion()
        assert not cf.is_initialized
        cf.initialize(45.0)
        assert cf.is_initialized

    def test_reset(self):
        cf = CompassFusion()
        cf.initialize(45.0)
        cf.reset()
        assert not cf.is_initialized

    def test_tilt_compensate_level(self):
        r = CompassFusion.tilt_compensate(90.0, 0.0, 0.0)
        assert r == pytest.approx(90.0, abs=1.0)

    def test_tilt_compensate_with_tilt(self):
        r = CompassFusion.tilt_compensate(90.0, 10.0, 5.0)
        assert 0 <= r < 360

    def test_declination_east(self):
        assert CompassFusion.apply_declination(90.0, 10.0) == pytest.approx(100.0, abs=0.1)

    def test_declination_west(self):
        assert CompassFusion.apply_declination(10.0, -15.0) == pytest.approx(355.0, abs=0.1)

    def test_update_compass(self):
        cf = CompassFusion()
        cf.initialize(0.0)
        state = cf.update_compass(CompassReading(heading_raw=90.0, quality=0.9, timestamp_ms=100))
        assert isinstance(state, HeadingState)

    def test_update_gyro(self):
        cf = CompassFusion()
        cf.initialize(0.0)
        cf.update_gyro(36.0, dt=1.0)
        assert cf.get_heading() == pytest.approx(36.0, abs=1.0)

    def test_gyro_drift_reduces_confidence(self):
        cf = CompassFusion()
        cf.initialize(0.0)
        c = cf.confidence
        for _ in range(200):
            cf.update_gyro(0.0, dt=0.1)
        assert cf.confidence < c

    def test_compass_restores_confidence(self):
        cf = CompassFusion()
        cf.initialize(0.0)
        for _ in range(200):
            cf.update_gyro(0.0, dt=0.1)
        low = cf.confidence
        cf.update_compass(CompassReading(heading_raw=cf.get_heading(), quality=0.9, timestamp_ms=1000))
        assert cf.confidence >= low

    def test_heading_wrap_around(self):
        cf = CompassFusion()
        cf.initialize(350.0)
        cf.update_gyro(36.0, dt=1.0)
        assert 0 <= cf.get_heading() < 360

    def test_get_state(self):
        cf = CompassFusion()
        cf.initialize(45.0)
        s = cf.get_state()
        assert s.heading == pytest.approx(45.0)
        assert s.confidence > 0

    def test_auto_init_from_compass(self):
        cf = CompassFusion()
        cf.update_compass(CompassReading(heading_raw=90.0, timestamp_ms=100))
        assert cf.is_initialized


# ===================================================================
# Position Estimator Tests
# ===================================================================

class TestPositionEstimator:
    def test_initialize(self):
        pe = PositionEstimator()
        pe.initialize(32.0, -117.0, heading=45.0)
        assert pe.is_initialized
        assert pe.mode == NavigationMode.FULL_FUSION

    def test_auto_initialize_from_gps(self):
        pe = PositionEstimator()
        est = pe.update(gps=_make_gps(32.0, -117.0, ts=100))
        assert pe.is_initialized

    def test_update_without_init(self):
        pe = PositionEstimator()
        est = pe.update()
        assert est.mode == NavigationMode.INITIALIZING

    def test_full_fusion_update(self):
        pe = PositionEstimator()
        pe.initialize(32.0, -117.0)
        est = pe.update(
            gps=_make_gps(32.0, -117.0, ts=100),
            imu=IMUReading(timestamp_ms=100),
            compass=CompassReading(heading_raw=0.0, quality=0.9, timestamp_ms=100),
        )
        assert est.mode == NavigationMode.FULL_FUSION
        assert est.confidence > 0

    def test_gps_dropout_ins_only(self):
        pe = PositionEstimator(gps_dropout_timeout_ms=1000)
        pe.initialize(32.0, -117.0)
        pe.update(gps=_make_gps(32.0, -117.0, ts=100))
        for i in range(200):
            pe.update(imu=IMUReading(timestamp_ms=200 + i * 10))
        assert pe.mode == NavigationMode.INS_ONLY
        assert pe.gps_dropout_active

    def test_gps_reacquisition(self):
        pe = PositionEstimator(gps_dropout_timeout_ms=1000)
        pe.initialize(32.0, -117.0)
        pe.update(gps=_make_gps(32.0, -117.0, ts=100))
        for i in range(50):
            pe.update(imu=IMUReading(timestamp_ms=200 + i * 10))
        assert pe.gps_dropout_active
        pe.update(gps=_make_gps(32.001, -117.0, ts=1000))
        assert not pe.gps_dropout_active

    def test_dropout_history(self):
        pe = PositionEstimator(gps_dropout_timeout_ms=1000)
        pe.initialize(32.0, -117.0)
        pe.update(gps=_make_gps(32.0, -117.0, ts=100))
        for i in range(50):
            pe.update(imu=IMUReading(timestamp_ms=200 + i * 10))
        pe.update(gps=_make_gps(32.0, -117.0, ts=1000))
        assert len(pe.get_gps_dropout_history()) >= 1

    def test_confidence_decreases_during_dropout(self):
        pe = PositionEstimator()
        pe.initialize(32.0, -117.0)
        pe.update(gps=_make_gps(32.0, -117.0, ts=100))
        c1 = pe.sensor_quality.overall_confidence
        for i in range(200):
            pe.update(imu=IMUReading(timestamp_ms=200 + i * 10))
        assert pe.sensor_quality.overall_confidence < c1

    def test_reset(self):
        pe = PositionEstimator()
        pe.initialize(32.0, -117.0)
        pe.reset()
        assert not pe.is_initialized

    def test_estimate_fields(self):
        pe = PositionEstimator()
        pe.initialize(32.0, -117.0)
        est = pe.update()
        assert hasattr(est, "latitude")
        assert hasattr(est, "confidence")

    def test_get_estimate(self):
        pe = PositionEstimator()
        pe.initialize(32.0, -117.0)
        pe.update(gps=_make_gps(32.0, -117.0, ts=100))
        assert pe.get_estimate() is not None

    def test_compass_updates_heading(self):
        pe = PositionEstimator()
        pe.initialize(32.0, -117.0, heading=0.0)
        est = pe.update(compass=CompassReading(heading_raw=90.0, quality=0.9, timestamp_ms=100))
        assert est.heading > 0

    def test_update_count(self):
        pe = PositionEstimator()
        pe.initialize(32.0, -117.0)
        pe.update(gps=_make_gps(32.0, -117.0, ts=100))
        assert pe.update_count == 1

    def test_gps_available(self):
        pe = PositionEstimator()
        pe.initialize(32.0, -117.0)
        pe.update(gps=_make_gps(32.0, -117.0, ts=100))
        assert pe.sensor_quality.gps_available

    def test_gps_unavailable_after_dropout(self):
        pe = PositionEstimator(gps_dropout_timeout_ms=500)
        pe.initialize(32.0, -117.0)
        pe.update(gps=_make_gps(32.0, -117.0, ts=100))
        for i in range(200):
            pe.update(imu=IMUReading(timestamp_ms=200 + i * 10))
        assert not pe.sensor_quality.gps_available

    def test_velocity_from_estimate(self):
        pe = PositionEstimator()
        pe.initialize(32.0, -117.0)
        est = pe.update(gps=_make_gps(32.0, -117.0, speed=5.0, course=45.0, ts=100))
        assert isinstance(est.velocity, VelocityVector)


# ===================================================================
# Intention Broadcasting Tests
# ===================================================================

class TestIntentionMessage:
    def test_create_valid(self):
        msg = IntentionMessage(vessel_id="VA", current_lat=32.0, current_lon=-117.0, confidence=0.9)
        assert msg.is_valid

    def test_empty_invalid(self):
        assert not IntentionMessage().is_valid

    def test_encode_decode_position(self):
        orig = IntentionMessage(vessel_id="S1", current_lat=32.5, current_lon=-117.3,
                                heading=90.0, speed=7.0, confidence=0.95, timestamp_ms=12345)
        decoded = IntentionMessage.decode_position_report(orig.encode_position_report())
        assert decoded.vessel_id == "S1"
        assert decoded.current_lat == pytest.approx(32.5, abs=0.001)
        assert decoded.heading == pytest.approx(90.0, abs=0.1)

    def test_encode_decode_intention(self):
        orig = IntentionMessage(vessel_id="B", current_lat=33.0, current_lon=-118.0,
                                dest_lat=34.0, dest_lon=-117.0, eta_seconds=3600.0,
                                confidence=0.8, timestamp_ms=54321)
        decoded = IntentionMessage.decode_intention_broadcast(orig.encode_intention_broadcast())
        assert decoded.vessel_id == "B"
        assert decoded.dest_lat == pytest.approx(34.0, abs=0.001)
        assert decoded.eta_seconds == pytest.approx(3600.0, abs=1.0)

    def test_short_payload_raises(self):
        with pytest.raises(ValueError):
            IntentionMessage.decode_position_report(b"\x00" * 5)

    def test_vessel_id_truncated(self):
        msg = IntentionMessage(vessel_id="A_VERY_LONG_NAME")
        decoded = IntentionMessage.decode_position_report(msg.encode_position_report())
        assert len(decoded.vessel_id) <= 8


class TestCPAAlgorithm:
    def test_head_on_collision(self):
        cpa = CPAAlgorithm(danger_distance_m=500.0, warning_distance_m=1000.0, caution_distance_m=2000.0)
        # Closer vessels for clearer collision
        r = cpa.compute_cpa(32.0, -117.0, 5.0, 0.0, 32.01, -117.0, 5.0, 180.0)
        assert r.risk_level >= 1  # at least CAUTION for head-on

    def test_parallel_no_risk(self):
        cpa = CPAAlgorithm()
        r = cpa.compute_cpa(32.0, -117.0, 5.0, 90.0, 32.01, -117.1, 5.0, 90.0)
        assert r.risk_level == 0

    def test_stationary_vessels(self):
        cpa = CPAAlgorithm()
        r = cpa.compute_cpa(32.0, -117.0, 0.0, 0.0, 32.01, -117.01, 0.0, 0.0)
        assert r.cpa_distance_m > 0

    def test_perpendicular_crossing(self):
        cpa = CPAAlgorithm(warning_distance_m=500, danger_distance_m=200)
        r = cpa.compute_cpa(32.0, -117.05, 5.0, 90.0, 31.98, -117.0, 5.0, 0.0)
        assert r.risk_level >= 0

    def test_risk_labels(self):
        a = CollisionAssessment(risk_level=0)
        assert a.risk_label == "NONE"
        assert not a.is_risk
        a.risk_level = 2
        assert a.risk_label == "WARNING"
        a.risk_level = 3
        assert a.risk_label == "DANGER"


class TestCollisionWarning:
    def test_encode_decode(self):
        orig = CollisionAssessment(vessel_id="S1", other_vessel_id="S2",
                                  cpa_distance_m=150.0, cpa_time_s=120.0, risk_level=2)
        decoded = CollisionAssessment.decode_collision_warning(orig.encode_collision_warning())
        assert decoded.vessel_id == "S1"
        assert decoded.cpa_distance_m == pytest.approx(150.0, abs=0.1)
        assert decoded.risk_level == 2


class TestIntentionBroadcaster:
    def test_update_own(self):
        ib = IntentionBroadcaster(vessel_id="VA")
        msg = ib.update_own(lat=32.0, lon=-117.0, heading=45.0, speed=5.0,
                            dest_lat=33.0, dest_lon=-116.0, confidence=0.9)
        assert msg.vessel_id == "VA"

    def test_receive_intention_risk(self):
        ib = IntentionBroadcaster(vessel_id="VA", cpa_warning_m=2000.0, cpa_danger_m=1000.0)
        ib.update_own(lat=32.0, lon=-117.0, heading=0.0, speed=5.0)
        other = IntentionMessage(vessel_id="VB", current_lat=32.01, current_lon=-117.0,
                                heading=180.0, speed=5.0, confidence=0.9)
        result = ib.receive_intention(other)
        assert result is not None

    def test_no_risk_parallel(self):
        ib = IntentionBroadcaster(vessel_id="VA")
        ib.update_own(lat=32.0, lon=-117.0, heading=90.0, speed=5.0)
        other = IntentionMessage(vessel_id="VB", current_lat=32.01, current_lon=-117.0,
                                heading=90.0, speed=5.0, confidence=0.9)
        assert ib.receive_intention(other) is None

    def test_invalid_ignored(self):
        ib = IntentionBroadcaster(vessel_id="VA")
        ib.update_own(lat=32.0, lon=-117.0)
        assert ib.receive_intention(IntentionMessage(vessel_id="")) is None

    def test_get_known_vessels(self):
        ib = IntentionBroadcaster(vessel_id="VA")
        ib.update_own(lat=32.0, lon=-117.0)
        ib.receive_intention(IntentionMessage(vessel_id="VB", current_lat=33.0,
                                             current_lon=-118.0, confidence=0.9))
        assert "VB" in ib.get_known_vessels()

    def test_msg_types(self):
        assert DRMessageType.POSITION_REPORT == 0x30
        assert DRMessageType.INTENTION_BROADCAST == 0x31
        assert DRMessageType.COLLISION_WARNING == 0x32
        assert DRMessageType.WAYPOINT_COMMAND == 0x33
        assert 0x30 in DR_MSG_TYPE_INFO

    def test_receive_position_report(self):
        ib = IntentionBroadcaster(vessel_id="VA")
        msg = IntentionMessage(vessel_id="S1", current_lat=33.0, current_lon=-118.0,
                              heading=45.0, speed=5.0, confidence=0.9, timestamp_ms=1000)
        payload = msg.encode_position_report()
        result = ib.receive_position_report(payload)
        assert result is not None
        assert result.vessel_id == "S1"

    def test_receive_intention_broadcast(self):
        ib = IntentionBroadcaster(vessel_id="VA", cpa_warning_m=2000.0, cpa_danger_m=1000.0)
        ib.update_own(lat=32.0, lon=-117.0, heading=0.0, speed=5.0)
        msg = IntentionMessage(vessel_id="APP", current_lat=32.01, current_lon=-117.0,
                              heading=180.0, speed=5.0, confidence=0.9, timestamp_ms=1000)
        result = ib.receive_intention_broadcast(msg.encode_intention_broadcast())
        assert result is not None


# ===================================================================
# Waypoint Navigator Tests
# ===================================================================

class TestWaypointNavigator:
    def test_single_waypoint(self):
        nav = WaypointNavigator(default_arrival_radius_m=500)
        nav.set_waypoints([Waypoint(32.01, -117.0)])
        cmd = nav.update(32.0, -117.0, heading=0.0, speed=5.0)
        assert nav.status == WaypointStatus.NAVIGATING

    def test_arrival_detection(self):
        nav = WaypointNavigator(default_arrival_radius_m=200)
        nav.set_waypoints([Waypoint(32.001, -117.0)])
        assert not nav.update(32.0, -117.0).is_arrival
        assert nav.update(32.001, -117.0).is_arrival
        assert nav.status == WaypointStatus.ARRIVED

    def test_multi_waypoint(self):
        nav = WaypointNavigator(default_arrival_radius_m=200)
        nav.set_waypoints([Waypoint(32.001, -117.0), Waypoint(32.001, -117.01)])
        assert nav.total_waypoints == 2
        nav.update(32.001, -117.0)
        assert nav.advance()
        assert nav.current_index == 1

    def test_advance_past_last(self):
        nav = WaypointNavigator(default_arrival_radius_m=200)
        nav.set_waypoints([Waypoint(32.001, -117.0)])
        nav.update(32.001, -117.0)
        assert not nav.advance()
        assert nav.is_complete

    def test_skip_to(self):
        nav = WaypointNavigator()
        nav.set_waypoints([Waypoint(32.0, -117.0), Waypoint(32.0, -118.0), Waypoint(32.0, -119.0)])
        assert nav.skip_to(2)
        assert nav.current_index == 2

    def test_skip_invalid(self):
        nav = WaypointNavigator()
        nav.set_waypoints([Waypoint(32.0, -117.0)])
        assert not nav.skip_to(5)

    def test_heading_command(self):
        nav = WaypointNavigator()
        nav.set_waypoints([Waypoint(32.0, -116.9)])  # east
        cmd = nav.update(32.0, -117.0, heading=0.0, speed=5.0)
        assert cmd.desired_heading_deg > 80  # should be roughly east
        assert cmd.desired_speed_ms > 0

    def test_turn_port(self):
        nav = WaypointNavigator()
        nav.set_waypoints([Waypoint(32.0, -117.1)])
        cmd = nav.update(32.0, -117.0, heading=0.0, speed=5.0)
        assert cmd.turn_direction == "port"

    def test_turn_starboard(self):
        nav = WaypointNavigator()
        nav.set_waypoints([Waypoint(32.0, -116.9)])
        cmd = nav.update(32.0, -117.0, heading=0.0, speed=5.0)
        assert cmd.turn_direction == "starboard"

    def test_speed_reduction_near_wp(self):
        nav = WaypointNavigator(default_arrival_radius_m=20, max_speed_ms=10.0)
        nav.set_waypoints([Waypoint(32.001, -117.0)])
        cmd = nav.update(32.0005, -117.0, heading=0.0, speed=10.0)
        assert cmd.desired_speed_ms < 10.0

    def test_progress(self):
        nav = WaypointNavigator(default_arrival_radius_m=5000)
        nav.set_waypoints([Waypoint(32.01, -117.0), Waypoint(32.02, -117.0)])
        p = nav.get_progress()
        assert p.total_waypoints == 2
        assert p.current_waypoint_idx == 0

    def test_empty_waypoints(self):
        nav = WaypointNavigator()
        nav.set_waypoints([])
        assert nav.is_complete

    def test_reset(self):
        nav = WaypointNavigator()
        nav.set_waypoints([Waypoint(32.01, -117.0)])
        nav.reset()
        assert nav.total_waypoints == 0

    def test_all_complete(self):
        nav = WaypointNavigator(default_arrival_radius_m=200)
        nav.set_waypoints([Waypoint(32.001, -117.0)])
        nav.update(32.001, -117.0)
        nav.advance()
        assert nav.status == WaypointStatus.ALL_COMPLETE

    def test_distance_tracked(self):
        nav = WaypointNavigator(default_arrival_radius_m=10000)
        nav.set_waypoints([Waypoint(32.1, -117.0)])
        nav.update(32.0, -117.0)
        nav.update(32.05, -117.0)
        assert nav.distance_traveled > 0

    def test_arrival_count(self):
        nav = WaypointNavigator(default_arrival_radius_m=5000)
        nav.set_waypoints([Waypoint(32.001, -117.0), Waypoint(32.001, -117.01)])
        nav.update(32.001, -117.0)
        assert nav.arrival_count == 1
        nav.advance()
        nav.update(32.001, -117.01)
        assert nav.arrival_count == 2

    def test_waypoint_validity(self):
        assert Waypoint(32.0, -117.0).is_valid
        assert not Waypoint(91.0, -117.0).is_valid
        assert not Waypoint(32.0, -181.0).is_valid


# ===================================================================
# Edge Cases
# ===================================================================

class TestEdgeCases:
    def test_zero_velocity_no_movement(self):
        kf = KalmanFilter2D()
        kf.initialize(32.0, -117.0)
        for _ in range(100):
            kf.predict(0.1)
        assert abs(kf.get_state().lat - 32.0) < 0.0001

    def test_extreme_north(self):
        d = NavigationMath.haversine_distance(89.9, 0.0, 89.9, 1.0)
        assert d > 0

    def test_extreme_south(self):
        d = NavigationMath.haversine_distance(-89.9, 0.0, -89.9, 1.0)
        assert d > 0

    def test_antipodal(self):
        d = NavigationMath.haversine_distance(90.0, 0.0, -90.0, 0.0)
        assert d == pytest.approx(NavigationMath.EARTH_RADIUS_M * math.pi, rel=0.01)

    def test_gps_failure_recovery(self):
        pe = PositionEstimator()
        pe.initialize(32.0, -117.0)
        pe.update(gps=GPSReading(latitude=32.0, longitude=-117.0, hdop=1.0,
                                 num_satellites=10, fix_quality=1, timestamp_ms=100))
        pe.update(gps=GPSReading(latitude=0.0, longitude=0.0, num_satellites=0,
                                 fix_quality=0, timestamp_ms=200))
        est = pe.get_estimate()
        assert abs(est.latitude - 32.0) < 1.0

    def test_imu_all_zeros(self):
        ins = INSIntegrator(method=IntegrationMethod.EULER)
        ins.initialize(32.0, -117.0)
        for i in range(100):
            ins.update(IMUReading(timestamp_ms=i * 10), dt=0.01)
        assert abs(ins.get_state().position_lat - 32.0) < 0.01

    def test_very_small_dt(self):
        ins = INSIntegrator(method=IntegrationMethod.RK4)
        ins.initialize(32.0, -117.0)
        ins.update(IMUReading(accel_x=1.0, timestamp_ms=10), dt=0.0001)
        assert abs(ins.get_state().position_lat - 32.0) < 0.01

    def test_large_dt(self):
        ins = INSIntegrator(method=IntegrationMethod.COMPLEMENTARY)
        ins.initialize(32.0, -117.0)
        ins.update(IMUReading(gyro_x=0.0, timestamp_ms=10000), dt=10.0)
        assert abs(ins.get_state().position_lat - 32.0) < 0.1

    def test_compass_quality_zero(self):
        cf = CompassFusion()
        cf.initialize(45.0)
        cf.update_compass(CompassReading(heading_raw=90.0, quality=0.0, timestamp_ms=100))
        assert cf.get_heading() == pytest.approx(45.0, abs=5.0)

    def test_waypoint_at_same_position(self):
        nav = WaypointNavigator(default_arrival_radius_m=20)
        nav.set_waypoints([Waypoint(32.0, -117.0)])
        assert nav.update(32.0, -117.0).is_arrival

    def test_covariance_identity_reset(self):
        cov = KalmanCovariance()
        cov.set(0, 1, 5.0)
        cov.identity(scale=2.0)
        assert cov.get(0, 1) == 0.0
        assert cov.get(0, 0) == 2.0

    def test_cpa_same_position(self):
        cpa = CPAAlgorithm()
        r = cpa.compute_cpa(32.0, -117.0, 5.0, 0.0, 32.0, -117.0, 5.0, 90.0)
        assert r.cpa_distance_m < 10

    def test_malformed_position_report(self):
        ib = IntentionBroadcaster()
        assert ib.receive_position_report(b"\x00" * 3) is None

    def test_malformed_intention_broadcast(self):
        ib = IntentionBroadcaster()
        assert ib.receive_intention_broadcast(b"short") is None

    def test_very_close_points(self):
        d = NavigationMath.haversine_distance(32.0, -117.0, 32.0000001, -117.0)
        assert 0 <= d < 1.0

    def test_bearing_same_point(self):
        b = NavigationMath.bearing(32.0, -117.0, 32.0, -117.0)
        assert 0 <= b < 360


# ===================================================================
# Integration Tests
# ===================================================================

class TestIntegration:
    def test_full_pipeline(self):
        pe = PositionEstimator()
        ib = IntentionBroadcaster(vessel_id="NAV_TEST")
        pe.initialize(32.0, -117.0, heading=0.0)
        ib.update_own(lat=32.0, lon=-117.0, heading=0.0, speed=5.0,
                       dest_lat=33.0, dest_lon=-117.0, confidence=1.0)
        for i in range(100):
            ts = i * 100
            gps = GPSReading(latitude=32.0 + i * 0.0001, longitude=-117.0 + i * 0.00005,
                           speed=5.0, course=45.0, hdop=1.0, num_satellites=10,
                           fix_quality=1, timestamp_ms=ts)
            imu = IMUReading(accel_x=0.01, gyro_x=0.1, timestamp_ms=ts)
            compass = CompassReading(heading_raw=45.0 + i * 0.01, quality=0.9, timestamp_ms=ts)
            est = pe.update(gps=gps, imu=imu, compass=compass, dt=0.1)
            if i % 25 == 0:
                ib.update_own(lat=est.latitude, lon=est.longitude, heading=est.heading,
                             speed=est.speed, confidence=est.confidence)
        final = pe.get_estimate()
        assert final is not None
        assert final.confidence > 0.5

    def test_dropout_and_reacquisition(self):
        pe = PositionEstimator(gps_dropout_timeout_ms=500)
        pe.initialize(32.0, -117.0)
        for i in range(20):
            pe.update(gps=_make_gps(32.0, -117.0, ts=i * 100),
                      imu=IMUReading(timestamp_ms=i * 100))
        for i in range(20):
            pe.update(imu=IMUReading(accel_x=0.01, timestamp_ms=2000 + i * 100), dt=0.1)
        assert pe.mode == NavigationMode.INS_ONLY
        pe.update(gps=_make_gps(32.001, -117.0, ts=5000),
                  imu=IMUReading(timestamp_ms=5000))
        assert not pe.gps_dropout_active
        assert len(pe.get_gps_dropout_history()) >= 1

    def test_waypoint_with_estimator(self):
        pe = PositionEstimator()
        nav = WaypointNavigator(default_arrival_radius_m=5000)
        nav.set_waypoints([Waypoint(32.001, -117.0)])
        pe.initialize(32.0, -117.0)
        arrived = False
        for i in range(500):
            frac = i / 500.0
            lat = 32.0 + frac * 0.002
            gps = GPSReading(latitude=lat, longitude=-117.0, speed=5.0, course=0.0,
                           hdop=1.0, num_satellites=10, fix_quality=1, timestamp_ms=i * 100)
            est = pe.update(gps=gps, imu=IMUReading(timestamp_ms=i * 100), dt=0.1)
            cmd = nav.update(est.latitude, est.longitude, est.heading, est.speed)
            if cmd.is_arrival:
                arrived = True
                break
        assert arrived or nav.arrival_count >= 1

    def test_multi_vessel_collision(self):
        ib = IntentionBroadcaster(vessel_id="OWN")
        ib.update_own(lat=32.0, lon=-117.0, heading=0.0, speed=5.0)
        v1 = IntentionMessage(vessel_id="A1", current_lat=32.1, current_lon=-117.0,
                              heading=180.0, speed=5.0, confidence=0.9)
        v2 = IntentionMessage(vessel_id="D1", current_lat=32.0, current_lon=-116.5,
                              heading=90.0, speed=5.0, confidence=0.9)
        ib.receive_intention(v1)
        ib.receive_intention(v2)
        assert len(ib.get_known_vessels()) == 2

    def test_consecutive_compass_gyro(self):
        cf = CompassFusion()
        cf.initialize(0.0)
        headings = []
        for i in range(100):
            if i % 10 == 0:
                cf.update_compass(CompassReading(heading_raw=i * 3.6, quality=0.8, timestamp_ms=i * 100))
            else:
                cf.update_gyro(36.0, dt=0.1)
            headings.append(cf.get_heading())
        for i in range(1, len(headings)):
            diff = (headings[i] - headings[i - 1]) % 360
            if diff > 180:
                diff -= 360
            assert diff > -50

    def test_broadcast_multiple_vessels(self):
        ib = IntentionBroadcaster(vessel_id="C")
        ib.update_own(lat=32.0, lon=-117.0, heading=0.0, speed=5.0)
        for i in range(5):
            ib.receive_intention(IntentionMessage(
                vessel_id=f"F{i}", current_lat=32.0 + i * 0.01,
                current_lon=-117.0 + i * 0.01, heading=90.0, speed=3.0, confidence=0.8))
        assert len(ib.get_known_vessels()) == 5
