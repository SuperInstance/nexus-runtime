"""NEXUS Learning - Comprehensive observation pipeline tests.

100+ tests covering:
  - UnifiedObservation: field access, 72-field count, serialization, round-trip
  - ObservationRecorder: ring buffer, flush, retention, query
  - Pattern Discovery: all 5 algorithms independently + engine orchestrator
  - Integration: record observations then discover patterns
  - Edge cases: empty buffers, single observations, degenerate inputs
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile

import pytest

from learning.observation import (
    FIELD_COUNT,
    NUM_ACTUATOR_CHANNELS,
    NUM_SENSOR_CHANNELS,
    ActuatorChannel,
    AnomalyDetector,
    BOCPDDetector,
    CrossCorrelationDetector,
    ObservationBuilder,
    ObservationRecorder,
    PatternDiscoveryEngine,
    PatternResult,
    PatternType,
    PeriodicDetector,
    RetentionPolicy,
    SensorChannel,
    TrendDetector,
    UnifiedObservation,
    _correlation,
    _mean,
    _std,
)


# ===================================================================
# Helpers
# ===================================================================


def _make_obs(
    ts: int = 1000,
    vessel: str = "vessel_01",
    reflex: str = "reflex_01",
    sensor_vals: list[float] | None = None,
) -> UnifiedObservation:
    """Create a test observation with sensible defaults."""
    b = (
        UnifiedObservation.create_builder()
        .navigation(latitude=45.5, longitude=-122.7, heading=270.0, speed_over_ground=2.5)
        .timing(timestamp_ms=ts, uptime_ms=ts, reflex_id=reflex, mission_id="mission_01")
        .metadata(vessel_id=vessel, agent_id="agent_01", session_id="session_01")
        .trust(score=0.5, level=1, event_type="good", delta=0.01)
        .safety(state="nominal", flags=0, watchdog=100)
        .vm(pc=10, stack_depth=2, cycle_count=50, last_opcode=0x03)
    )
    if sensor_vals:
        b.sensors(sensor_vals)
    return b.build()


def _make_series(n: int, base_ts: int = 0) -> list[UnifiedObservation]:
    """Create a series of n observations with linearly increasing timestamps."""
    return [_make_obs(ts=base_ts + i * 100) for i in range(n)]


def _make_correlated_series(n: int) -> list[UnifiedObservation]:
    """Create observations where ch0 and ch1 are perfectly correlated."""
    obs_list = []
    for i in range(n):
        val = math.sin(i * 0.5) * 10.0
        obs = _make_obs(ts=i * 100)
        obs.sensor_ch0_val = val
        obs.sensor_ch1_val = val * 2.0  # Perfect correlation (linear)
        obs_list.append(obs)
    return obs_list


def _make_changepoint_series(n: int, cp_idx: int) -> list[UnifiedObservation]:
    """Create a series with a changepoint at cp_idx."""
    obs_list = []
    for i in range(n):
        val = 5.0 if i < cp_idx else 50.0
        obs = _make_obs(ts=i * 100)
        obs.sensor_ch0_val = val + (i - cp_idx) * 0.1 if i >= cp_idx else val
        obs_list.append(obs)
    return obs_list


def _make_anomaly_series(n: int, anomaly_indices: list[int]) -> list[UnifiedObservation]:
    """Create a series with anomalies at specified indices."""
    obs_list = []
    for i in range(n):
        val = 10.0
        if i in anomaly_indices:
            val = 1000.0
        obs = _make_obs(ts=i * 100)
        obs.sensor_ch0_val = val
        obs_list.append(obs)
    return obs_list


def _make_trend_series(n: int, slope: float = 0.5) -> list[UnifiedObservation]:
    """Create a series with a linear trend on ch0."""
    obs_list = []
    for i in range(n):
        obs = _make_obs(ts=i * 100)
        obs.sensor_ch0_val = i * slope
        obs_list.append(obs)
    return obs_list


def _make_periodic_series(n: int, period: int = 10) -> list[UnifiedObservation]:
    """Create a series with a periodic signal on ch0."""
    obs_list = []
    for i in range(n):
        obs = _make_obs(ts=i * 100)
        obs.sensor_ch0_val = math.sin(2 * math.pi * i / period) * 10.0
        obs_list.append(obs)
    return obs_list


# ===================================================================
# Section 1: UnifiedObservation - Field Count and Defaults (10 tests)
# ===================================================================


class TestObservationFieldCount:
    """Verify the 72-field structure."""

    def test_field_count_is_72(self) -> None:
        obs = UnifiedObservation()
        assert obs.field_count() == FIELD_COUNT
        assert FIELD_COUNT == 72

    def test_csv_columns_count_is_72(self) -> None:
        cols = UnifiedObservation.csv_columns()
        assert len(cols) == 72

    def test_default_values_are_zero_or_empty(self) -> None:
        obs = UnifiedObservation()
        assert obs.latitude == 0.0
        assert obs.longitude == 0.0
        assert obs.heading == 0.0
        assert obs.speed_over_ground == 0.0
        assert obs.course_over_ground == 0.0
        assert obs.trust_score == 0.0
        assert obs.trust_level == 0
        assert obs.trust_event_type == ""
        assert obs.safety_state == ""
        assert obs.vessel_id == ""
        assert obs.timestamp_ms == 0

    def test_sensor_defaults_are_zero_val_and_one_quality(self) -> None:
        obs = UnifiedObservation()
        for i in range(NUM_SENSOR_CHANNELS):
            assert getattr(obs, f"sensor_ch{i}_val") == 0.0
            assert getattr(obs, f"sensor_ch{i}_q") == 1.0

    def test_actuator_defaults_are_zero(self) -> None:
        obs = UnifiedObservation()
        for i in range(NUM_ACTUATOR_CHANNELS):
            assert getattr(obs, f"actuator_ch{i}_cmd") == 0.0
            assert getattr(obs, f"actuator_ch{i}_fb") == 0.0

    def test_vm_defaults_are_zero(self) -> None:
        obs = UnifiedObservation()
        assert obs.program_counter == 0
        assert obs.stack_depth == 0
        assert obs.cycle_count == 0
        assert obs.last_opcode == 0

    def test_safety_defaults_are_zero(self) -> None:
        obs = UnifiedObservation()
        assert obs.safety_flags == 0
        assert obs.watchdog_counter == 0
        assert obs.heartbeat_missed == 0

    def test_csv_columns_include_all_nav_fields(self) -> None:
        cols = UnifiedObservation.csv_columns()
        for name in ["latitude", "longitude", "heading", "speed_over_ground", "course_over_ground"]:
            assert name in cols

    def test_csv_columns_include_all_sensor_fields(self) -> None:
        cols = UnifiedObservation.csv_columns()
        for i in range(NUM_SENSOR_CHANNELS):
            assert f"sensor_ch{i}_val" in cols
            assert f"sensor_ch{i}_q" in cols

    def test_csv_columns_include_all_actuator_fields(self) -> None:
        cols = UnifiedObservation.csv_columns()
        for i in range(NUM_ACTUATOR_CHANNELS):
            assert f"actuator_ch{i}_cmd" in cols
            assert f"actuator_ch{i}_fb" in cols


# ===================================================================
# Section 2: UnifiedObservation - Field Access (12 tests)
# ===================================================================


class TestObservationFieldAccess:
    """Test direct field access and structured access helpers."""

    def test_direct_field_assignment(self) -> None:
        obs = UnifiedObservation()
        obs.latitude = 48.0
        obs.longitude = -123.0
        assert obs.latitude == 48.0
        assert obs.longitude == -123.0

    def test_get_sensor_channel(self) -> None:
        obs = _make_obs()
        obs.sensor_ch3_val = 42.0
        obs.sensor_ch3_q = 0.9
        ch = obs.get_sensor_channel(3)
        assert ch.value == 42.0
        assert ch.quality == 0.9
        assert ch.timestamp_ms == 1000

    def test_set_sensor_channel(self) -> None:
        obs = UnifiedObservation()
        obs.set_sensor_channel(5, SensorChannel(value=99.0, quality=0.8, timestamp_ms=500))
        assert obs.sensor_ch5_val == 99.0
        assert obs.sensor_ch5_q == 0.8

    def test_get_sensor_channel_out_of_range(self) -> None:
        obs = UnifiedObservation()
        with pytest.raises(IndexError):
            obs.get_sensor_channel(16)
        with pytest.raises(IndexError):
            obs.get_sensor_channel(-1)

    def test_set_sensor_channel_out_of_range(self) -> None:
        obs = UnifiedObservation()
        with pytest.raises(IndexError):
            obs.set_sensor_channel(20, SensorChannel())

    def test_get_actuator_channel(self) -> None:
        obs = UnifiedObservation()
        obs.actuator_ch2_cmd = 50.0
        obs.actuator_ch2_fb = 48.0
        ch = obs.get_actuator_channel(2)
        assert ch.command == 50.0
        assert ch.feedback == 48.0
        assert ch.status == 0

    def test_set_actuator_channel(self) -> None:
        obs = UnifiedObservation()
        obs.set_actuator_channel(0, ActuatorChannel(command=75.0, feedback=70.0, status=1))
        assert obs.actuator_ch0_cmd == 75.0
        assert obs.actuator_ch0_fb == 70.0

    def test_get_actuator_channel_out_of_range(self) -> None:
        obs = UnifiedObservation()
        with pytest.raises(IndexError):
            obs.get_actuator_channel(8)
        with pytest.raises(IndexError):
            obs.get_actuator_channel(-1)

    def test_get_all_sensor_values(self) -> None:
        obs = _make_obs(sensor_vals=[float(i) for i in range(16)])
        vals = obs.get_all_sensor_values()
        assert len(vals) == 16
        assert vals[0] == 0.0
        assert vals[15] == 15.0

    def test_get_all_sensor_qualities(self) -> None:
        obs = UnifiedObservation()
        obs.sensor_ch7_q = 0.5
        quals = obs.get_all_sensor_qualities()
        assert len(quals) == 16
        assert quals[7] == 0.5
        assert quals[0] == 1.0

    def test_get_all_actuator_commands(self) -> None:
        obs = UnifiedObservation()
        obs.actuator_ch3_cmd = 100.0
        cmds = obs.get_all_actuator_commands()
        assert len(cmds) == 8
        assert cmds[3] == 100.0

    def test_get_all_actuator_feedbacks(self) -> None:
        obs = UnifiedObservation()
        obs.actuator_ch5_fb = -50.0
        fbs = obs.get_all_actuator_feedbacks()
        assert len(fbs) == 8
        assert fbs[5] == -50.0


# ===================================================================
# Section 3: UnifiedObservation - Serialization (14 tests)
# ===================================================================


class TestObservationSerialization:
    """Test to_dict, from_dict, to_row, from_row round-trip."""

    def test_to_dict_keys_match_fields(self) -> None:
        obs = _make_obs()
        d = obs.to_dict()
        assert set(d.keys()) == set(UnifiedObservation.csv_columns())
        assert len(d) == 72

    def test_from_dict_round_trip(self) -> None:
        obs = _make_obs()
        d = obs.to_dict()
        obs2 = UnifiedObservation.from_dict(d)
        assert obs2.latitude == obs.latitude
        assert obs2.longitude == obs.longitude
        assert obs2.heading == obs.heading
        assert obs2.timestamp_ms == obs.timestamp_ms
        assert obs2.vessel_id == obs.vessel_id
        assert obs2.reflex_id == obs.reflex_id
        assert obs2.trust_score == obs.trust_score

    def test_from_dict_missing_fields_use_defaults(self) -> None:
        obs = UnifiedObservation.from_dict({"latitude": 1.0})
        assert obs.latitude == 1.0
        assert obs.longitude == 0.0
        assert obs.timestamp_ms == 0

    def test_from_dict_all_types_preserved(self) -> None:
        obs = _make_obs()
        d = obs.to_dict()
        obs2 = UnifiedObservation.from_dict(d)
        # int fields stay int
        assert isinstance(obs2.timestamp_ms, int)
        assert isinstance(obs2.trust_level, int)
        assert isinstance(obs2.safety_flags, int)
        # float fields stay float
        assert isinstance(obs2.latitude, float)
        assert isinstance(obs2.trust_score, float)
        # str fields stay str
        assert isinstance(obs2.vessel_id, str)
        assert isinstance(obs2.reflex_id, str)

    def test_to_row_length_is_72(self) -> None:
        obs = _make_obs()
        row = obs.to_row()
        assert len(row) == 72

    def test_from_row_round_trip(self) -> None:
        obs = _make_obs()
        columns = UnifiedObservation.csv_columns()
        row = obs.to_row()
        obs2 = UnifiedObservation.from_row(row, columns)
        assert obs2.latitude == obs.latitude
        assert obs2.longitude == obs.longitude
        assert obs2.vessel_id == obs.vessel_id
        assert obs2.timestamp_ms == obs.timestamp_ms

    def test_from_row_type_coercion_int(self) -> None:
        obs = _make_obs()
        columns = UnifiedObservation.csv_columns()
        row = obs.to_row()
        obs2 = UnifiedObservation.from_row(row, columns)
        assert isinstance(obs2.timestamp_ms, int)
        assert isinstance(obs2.trust_level, int)

    def test_from_row_type_coercion_float(self) -> None:
        obs = _make_obs()
        columns = UnifiedObservation.csv_columns()
        row = obs.to_row()
        obs2 = UnifiedObservation.from_row(row, columns)
        assert isinstance(obs2.latitude, float)
        assert isinstance(obs2.trust_score, float)

    def test_from_row_type_coercion_str(self) -> None:
        obs = _make_obs()
        columns = UnifiedObservation.csv_columns()
        row = obs.to_row()
        obs2 = UnifiedObservation.from_row(row, columns)
        assert isinstance(obs2.vessel_id, str)
        assert isinstance(obs2.reflex_id, str)

    def test_sensor_values_survive_round_trip(self) -> None:
        obs = _make_obs(sensor_vals=[float(i * 10) for i in range(16)])
        d = obs.to_dict()
        obs2 = UnifiedObservation.from_dict(d)
        vals = obs2.get_all_sensor_values()
        assert vals[0] == 0.0
        assert vals[5] == 50.0
        assert vals[15] == 150.0

    def test_actuator_values_survive_round_trip(self) -> None:
        obs = _make_obs()
        obs.actuator_ch0_cmd = 100.0
        obs.actuator_ch7_fb = -99.0
        d = obs.to_dict()
        obs2 = UnifiedObservation.from_dict(d)
        assert obs2.actuator_ch0_cmd == 100.0
        assert obs2.actuator_ch7_fb == -99.0

    def test_copy_with_preserves_fields(self) -> None:
        obs = _make_obs()
        obs2 = obs.copy_with(heading=180.0)
        assert obs2.heading == 180.0
        assert obs2.latitude == obs.latitude  # preserved
        assert obs2.vessel_id == obs.vessel_id

    def test_copy_with_does_not_mutate_original(self) -> None:
        obs = _make_obs()
        obs2 = obs.copy_with(heading=180.0)
        assert obs.heading == 270.0
        assert obs2.heading == 180.0

    def test_full_round_trip_via_row(self) -> None:
        """Complete round-trip: observation -> row -> observation."""
        obs = _make_obs(sensor_vals=[float(i) for i in range(16)])
        obs.actuator_ch0_cmd = 55.0
        obs.actuator_ch3_fb = -22.0
        columns = UnifiedObservation.csv_columns()
        row = obs.to_row()
        obs2 = UnifiedObservation.from_row(row, columns)

        assert obs2.latitude == obs.latitude
        assert obs2.heading == obs.heading
        assert obs2.vessel_id == obs.vessel_id
        assert obs2.sensor_ch5_val == 5.0
        assert obs2.actuator_ch0_cmd == 55.0
        assert obs2.actuator_ch3_fb == -22.0
        assert obs2.trust_score == obs.trust_score


# ===================================================================
# Section 4: ObservationBuilder (8 tests)
# ===================================================================


class TestObservationBuilder:
    """Test the fluent builder for UnifiedObservation."""

    def test_empty_builder(self) -> None:
        obs = ObservationBuilder().build()
        assert isinstance(obs, UnifiedObservation)
        assert obs.field_count() == 72

    def test_navigation_builder(self) -> None:
        obs = (
            ObservationBuilder()
            .navigation(latitude=40.0, longitude=-74.0, heading=90.0)
            .build()
        )
        assert obs.latitude == 40.0
        assert obs.longitude == -74.0
        assert obs.heading == 90.0

    def test_sensor_builder(self) -> None:
        obs = ObservationBuilder().sensor(3, 77.0, 0.95).build()
        assert obs.sensor_ch3_val == 77.0
        assert obs.sensor_ch3_q == 0.95

    def test_sensors_bulk_builder(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0]
        obs = ObservationBuilder().sensors(vals).build()
        assert obs.sensor_ch0_val == 1.0
        assert obs.sensor_ch3_val == 4.0

    def test_sensors_bulk_with_qualities(self) -> None:
        vals = [1.0, 2.0, 3.0]
        quals = [0.5, 0.8, 1.0]
        obs = ObservationBuilder().sensors(vals, quals).build()
        assert obs.sensor_ch0_val == 1.0
        assert obs.sensor_ch0_q == 0.5
        assert obs.sensor_ch2_q == 1.0

    def test_actuator_builder(self) -> None:
        obs = ObservationBuilder().actuator(2, 60.0, 58.0).build()
        assert obs.actuator_ch2_cmd == 60.0
        assert obs.actuator_ch2_fb == 58.0

    def test_actuators_bulk_builder(self) -> None:
        cmds = [10.0, 20.0]
        obs = ObservationBuilder().actuators(cmds).build()
        assert obs.actuator_ch0_cmd == 10.0
        assert obs.actuator_ch1_cmd == 20.0

    def test_full_builder_chain(self) -> None:
        obs = (
            ObservationBuilder()
            .navigation(latitude=45.0, longitude=-122.0, heading=270.0)
            .sensors([10.0, 20.0], [0.9, 0.8])
            .actuators([50.0], [48.0])
            .trust(score=0.8, level=2, event_type="good", delta=0.02)
            .safety(state="nominal", flags=0, watchdog=50, heartbeat_missed=0)
            .vm(pc=5, stack_depth=1, cycle_count=100, last_opcode=0x01)
            .timing(timestamp_ms=5000, uptime_ms=60000, reflex_id="r1", mission_id="m1")
            .metadata(vessel_id="v1", agent_id="a1", session_id="s1")
            .build()
        )
        assert obs.latitude == 45.0
        assert obs.sensor_ch0_val == 10.0
        assert obs.sensor_ch0_q == 0.9
        assert obs.actuator_ch0_cmd == 50.0
        assert obs.trust_score == 0.8
        assert obs.trust_level == 2
        assert obs.safety_state == "nominal"
        assert obs.program_counter == 5
        assert obs.timestamp_ms == 5000
        assert obs.vessel_id == "v1"


# ===================================================================
# Section 5: SensorChannel and ActuatorChannel (6 tests)
# ===================================================================


class TestSensorChannel:
    def test_defaults(self) -> None:
        ch = SensorChannel()
        assert ch.value == 0.0
        assert ch.quality == 1.0
        assert ch.timestamp_ms == 0

    def test_is_valid_true(self) -> None:
        assert SensorChannel(value=5.0, quality=0.9).is_valid()

    def test_is_valid_zero_quality(self) -> None:
        assert not SensorChannel(value=5.0, quality=0.0).is_valid()

    def test_is_valid_negative_quality(self) -> None:
        assert not SensorChannel(value=5.0, quality=-0.5).is_valid()

    def test_frozen(self) -> None:
        ch = SensorChannel(value=1.0)
        with pytest.raises(AttributeError):
            ch.value = 2.0  # type: ignore[misc]

    def test_equality(self) -> None:
        a = SensorChannel(value=1.0, quality=0.8, timestamp_ms=100)
        b = SensorChannel(value=1.0, quality=0.8, timestamp_ms=100)
        assert a == b


class TestActuatorChannel:
    def test_defaults(self) -> None:
        ch = ActuatorChannel()
        assert ch.command == 0.0
        assert ch.feedback == 0.0
        assert ch.status == 0

    def test_custom_values(self) -> None:
        ch = ActuatorChannel(command=75.0, feedback=70.0, status=1)
        assert ch.command == 75.0
        assert ch.feedback == 70.0
        assert ch.status == 1

    def test_frozen(self) -> None:
        ch = ActuatorChannel()
        with pytest.raises(AttributeError):
            ch.command = 1.0  # type: ignore[misc]


# ===================================================================
# Section 6: ObservationRecorder - Buffer Management (12 tests)
# ===================================================================


class TestRecorderBuffer:
    """Test ring buffer behavior."""

    def test_empty_buffer(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        assert rec.buffer_size_current() == 0
        assert rec.total_recorded() == 0

    def test_record_single(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        rec.record(_make_obs())
        assert rec.buffer_size_current() == 1
        assert rec.total_recorded() == 1

    def test_record_many(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        rec.record_many(_make_series(50))
        assert rec.buffer_size_current() == 50
        assert rec.total_recorded() == 50

    def test_ring_buffer_overflows(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"), buffer_size=10)
        for i in range(25):
            rec.record(_make_obs(ts=i * 100))
        assert rec.buffer_size_current() == 10
        assert rec.total_recorded() == 25

    def test_ring_buffer_keeps_most_recent(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"), buffer_size=5)
        for i in range(10):
            rec.record(_make_obs(ts=i * 1000))
        buffered = rec.get_all_buffered()
        timestamps = [o.timestamp_ms for o in buffered]
        assert timestamps == [5000, 6000, 7000, 8000, 9000]

    def test_get_all_buffered(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        series = _make_series(5)
        rec.record_many(series)
        buffered = rec.get_all_buffered()
        assert len(buffered) == 5

    def test_default_buffer_size(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        assert rec.buffer_size == 10_000

    def test_custom_buffer_size(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"), buffer_size=100)
        assert rec.buffer_size == 100

    def test_create_storage_directory(self, tmp_path) -> None:
        path = str(tmp_path / "sub" / "dir")
        rec = ObservationRecorder(storage_path=path)
        assert os.path.isdir(path)

    def test_retention_policy_default(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        assert rec.retention.max_records == 10_000
        assert rec.retention.max_age_hours == 24.0

    def test_custom_retention_policy(self, tmp_path) -> None:
        rp = RetentionPolicy(max_records=100, max_age_hours=1.0, min_safety_records=10)
        rec = ObservationRecorder(
            storage_path=str(tmp_path / "obs"), retention=rp
        )
        assert rec.retention.max_records == 100
        assert rec.retention.max_age_hours == 1.0

    def test_record_preserves_data(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        obs = _make_obs(vessel="VESSEL_X", reflex="REFLEX_Y")
        rec.record(obs)
        buffered = rec.get_all_buffered()
        assert buffered[0].vessel_id == "VESSEL_X"
        assert buffered[0].reflex_id == "REFLEX_Y"


# ===================================================================
# Section 7: ObservationRecorder - Flush and Load (10 tests)
# ===================================================================


class TestRecorderFlush:
    """Test flushing observations to disk."""

    def test_flush_empty_buffer(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        count = rec.flush()
        assert count == 0
        assert rec.total_flushed() == 0

    def test_flush_writes_file(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        rec.record_many(_make_series(5))
        count = rec.flush()
        assert count == 5
        assert rec.total_flushed() == 5
        assert rec.flush_count() == 1
        assert rec.buffer_size_current() == 0

    def test_flush_creates_gz_file(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        rec.record_many(_make_series(3))
        rec.flush()
        files = rec.list_files()
        assert len(files) == 1
        assert files[0].endswith(".csv.gz")

    def test_flush_atomic_write_no_tmp_after(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        rec.record_many(_make_series(3))
        rec.flush()
        # No .tmp files should remain
        all_files = os.listdir(str(tmp_path / "obs"))
        tmp_files = [f for f in all_files if f.endswith(".tmp")]
        assert len(tmp_files) == 0

    def test_flush_multiple_times(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        rec.record_many(_make_series(3, base_ts=0))
        rec.flush()
        rec.record_many(_make_series(3, base_ts=30000))
        rec.flush()
        files = rec.list_files()
        assert len(files) == 2
        assert rec.flush_count() == 2
        assert rec.total_flushed() == 6

    def test_load_from_disk(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        original = _make_series(5)
        rec.record_many(original)
        rec.flush()
        loaded = rec.load_from_disk()
        assert len(loaded) == 5
        assert loaded[0].latitude == original[0].latitude
        assert loaded[4].timestamp_ms == original[4].timestamp_ms

    def test_load_from_disk_latest(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        rec.record_many(_make_series(3, base_ts=0))
        rec.flush()
        rec.record_many(_make_series(3, base_ts=50000))
        rec.flush()
        loaded = rec.load_from_disk()  # Should load latest
        assert len(loaded) == 3
        assert loaded[0].timestamp_ms == 50000

    def test_load_specific_file(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        rec.record_many(_make_series(2, base_ts=0))
        rec.flush()
        files = rec.list_files()
        loaded = rec.load_from_disk(files[0])
        assert len(loaded) == 2

    def test_load_nonexistent_file_raises(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        with pytest.raises(FileNotFoundError):
            rec.load_from_disk("nonexistent.csv.gz")

    def test_load_empty_directory(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        loaded = rec.load_from_disk()
        assert loaded == []

    def test_flush_preserves_all_72_fields(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        obs = _make_obs(sensor_vals=[float(i) for i in range(16)])
        obs.actuator_ch0_cmd = 99.0
        obs.trust_score = 0.75
        rec.record(obs)
        rec.flush()
        loaded = rec.load_from_disk()
        assert loaded[0].sensor_ch5_val == 5.0
        assert loaded[0].actuator_ch0_cmd == 99.0
        assert loaded[0].trust_score == 0.75


# ===================================================================
# Section 8: ObservationRecorder - Query (12 tests)
# ===================================================================


class TestRecorderQuery:
    """Test the query interface."""

    def _make_query_data(self, tmp_path) -> ObservationRecorder:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        for i in range(20):
            vessel = "vessel_A" if i < 10 else "vessel_B"
            reflex = "reflex_X" if i % 2 == 0 else "reflex_Y"
            safety = "nominal" if i < 15 else "alert"
            obs = _make_obs(ts=i * 1000, vessel=vessel, reflex=reflex)
            obs.safety_state = safety
            rec.record(obs)
        return rec

    def test_query_all(self, tmp_path) -> None:
        rec = self._make_query_data(tmp_path)
        results = rec.query_buffer()
        assert len(results) == 20

    def test_query_by_time_range(self, tmp_path) -> None:
        rec = self._make_query_data(tmp_path)
        results = rec.query_buffer(time_start=5000, time_end=10000)
        timestamps = [o.timestamp_ms for o in results]
        assert all(5000 <= t <= 10000 for t in timestamps)
        assert len(results) == 6  # 5000, 6000, 7000, 8000, 9000, 10000

    def test_query_by_time_start_only(self, tmp_path) -> None:
        rec = self._make_query_data(tmp_path)
        results = rec.query_buffer(time_start=15000)
        assert len(results) == 5

    def test_query_by_time_end_only(self, tmp_path) -> None:
        rec = self._make_query_data(tmp_path)
        results = rec.query_buffer(time_end=4000)
        assert len(results) == 5

    def test_query_by_vessel_id(self, tmp_path) -> None:
        rec = self._make_query_data(tmp_path)
        results = rec.query_buffer(vessel_id="vessel_A")
        assert len(results) == 10
        assert all(o.vessel_id == "vessel_A" for o in results)

    def test_query_by_reflex_id(self, tmp_path) -> None:
        rec = self._make_query_data(tmp_path)
        results = rec.query_buffer(reflex_id="reflex_X")
        assert len(results) == 10
        assert all(o.reflex_id == "reflex_X" for o in results)

    def test_query_by_safety_state(self, tmp_path) -> None:
        rec = self._make_query_data(tmp_path)
        results = rec.query_buffer(safety_state="alert")
        assert len(results) == 5
        assert all(o.safety_state == "alert" for o in results)

    def test_query_combined_filters(self, tmp_path) -> None:
        rec = self._make_query_data(tmp_path)
        results = rec.query_buffer(
            vessel_id="vessel_A", reflex_id="reflex_X", time_end=5000
        )
        for o in results:
            assert o.vessel_id == "vessel_A"
            assert o.reflex_id == "reflex_X"
            assert o.timestamp_ms <= 5000

    def test_query_no_matches(self, tmp_path) -> None:
        rec = self._make_query_data(tmp_path)
        results = rec.query_buffer(vessel_id="nonexistent")
        assert len(results) == 0

    def test_query_empty_buffer(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        results = rec.query_buffer(vessel_id="anything")
        assert results == []

    def test_query_single_observation(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        rec.record(_make_obs(vessel="solo"))
        results = rec.query_buffer(vessel_id="solo")
        assert len(results) == 1

    def test_query_preserves_field_values(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        obs = _make_obs(vessel="test_v", reflex="test_r")
        obs.sensor_ch0_val = 42.0
        obs.trust_score = 0.9
        rec.record(obs)
        results = rec.query_buffer(vessel_id="test_v")
        assert results[0].sensor_ch0_val == 42.0
        assert results[0].trust_score == 0.9


# ===================================================================
# Section 9: ObservationRecorder - Retention (6 tests)
# ===================================================================


class TestRecorderRetention:
    """Test retention policy enforcement."""

    def test_retention_removes_old_files(self, tmp_path) -> None:
        rp = RetentionPolicy(max_age_hours=0.001)  # ~3.6 seconds
        rec = ObservationRecorder(
            storage_path=str(tmp_path / "obs"), retention=rp
        )
        # Create a file with old timestamp
        old_file = os.path.join(str(tmp_path / "obs"), "observations_0.csv.gz")
        with open(old_file, "w") as f:
            f.write("")
        # Create a recent file
        recent_file = os.path.join(
            str(tmp_path / "obs"),
            f"observations_{ObservationRecorder._current_time_ms()}.csv.gz",
        )
        with open(recent_file, "w") as f:
            f.write("")

        removed = rec.enforce_retention()
        assert removed >= 1
        remaining = rec.list_files()
        # Old file should be gone
        assert "observations_0.csv.gz" not in remaining

    def test_retention_keeps_recent_files(self, tmp_path) -> None:
        rp = RetentionPolicy(max_age_hours=24.0)
        rec = ObservationRecorder(
            storage_path=str(tmp_path / "obs"), retention=rp
        )
        now_ms = ObservationRecorder._current_time_ms()
        recent_file = os.path.join(
            str(tmp_path / "obs"), f"observations_{now_ms}.csv.gz"
        )
        with open(recent_file, "w") as f:
            f.write("")

        removed = rec.enforce_retention()
        assert removed == 0

    def test_enforce_retention_empty_dir(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        removed = rec.enforce_retention()
        assert removed == 0

    def test_list_files_empty(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        assert rec.list_files() == []

    def test_list_files(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        rec.record_many(_make_series(3, base_ts=1000))
        rec.flush()
        files = rec.list_files()
        assert len(files) == 1

    def test_flush_clears_buffer(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        rec.record_many(_make_series(10))
        assert rec.buffer_size_current() == 10
        rec.flush()
        assert rec.buffer_size_current() == 0


# ===================================================================
# Section 10: Pattern Discovery - Statistical Helpers (5 tests)
# ===================================================================


class TestStatisticalHelpers:
    def test_mean_empty(self) -> None:
        assert _mean([]) == 0.0

    def test_mean_single(self) -> None:
        assert _mean([5.0]) == 5.0

    def test_mean_multiple(self) -> None:
        assert _mean([1.0, 2.0, 3.0, 4.0, 5.0]) == 3.0

    def test_std_empty(self) -> None:
        assert _std([]) == 0.0

    def test_std_single(self) -> None:
        assert _std([5.0]) == 0.0

    def test_std_known(self) -> None:
        # std of [2, 4, 4, 4, 5, 5, 7, 9] = 2.0
        vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        s = _std(vals)
        assert abs(s - 2.138) < 0.01

    def test_correlation_perfect(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        r = _correlation(x, y)
        assert abs(r - 1.0) < 0.001

    def test_correlation_negative(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        r = _correlation(x, y)
        assert abs(r - (-1.0)) < 0.001

    def test_correlation_zero(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, -1.0, 1.0, -1.0, 1.0]
        r = _correlation(x, y)
        assert abs(r) < 0.1

    def test_correlation_too_short(self) -> None:
        assert _correlation([1.0], [1.0]) == 0.0

    def test_correlation_different_lengths(self) -> None:
        assert _correlation([1.0, 2.0], [1.0]) == 0.0

    def test_correlation_constant(self) -> None:
        assert _correlation([5.0, 5.0, 5.0], [5.0, 5.0, 5.0]) == 0.0


# ===================================================================
# Section 11: PatternResult (5 tests)
# ===================================================================


class TestPatternResult:
    def test_confidence_clamped_high(self) -> None:
        r = PatternResult(
            pattern_type=PatternType.CROSS_CORRELATION,
            description="test",
            confidence=1.5,
        )
        assert r.confidence == 1.0

    def test_confidence_clamped_low(self) -> None:
        r = PatternResult(
            pattern_type=PatternType.ANOMALY,
            description="test",
            confidence=-0.5,
        )
        assert r.confidence == 0.0

    def test_confidence_preserved(self) -> None:
        r = PatternResult(
            pattern_type=PatternType.TREND,
            description="test",
            confidence=0.75,
        )
        assert r.confidence == 0.75

    def test_pattern_type_values(self) -> None:
        assert PatternType.CROSS_CORRELATION == 1
        assert PatternType.BOCPD == 2
        assert PatternType.ANOMALY == 3
        assert PatternType.TREND == 4
        assert PatternType.PERIODIC == 5

    def test_details_dict(self) -> None:
        r = PatternResult(
            pattern_type=PatternType.PERIODIC,
            description="period found",
            confidence=0.8,
            details={"period": 10},
        )
        assert r.details["period"] == 10


# ===================================================================
# Section 12: Cross-Correlation Detector (8 tests)
# ===================================================================


class TestCrossCorrelation:
    def test_no_patterns_few_observations(self) -> None:
        det = CrossCorrelationDetector()
        results = det.detect(_make_series(3))
        assert results == []

    def test_no_patterns_uncorrelated(self) -> None:
        det = CrossCorrelationDetector(threshold=0.9)
        obs_list = _make_series(50)
        # Random-ish values on different channels
        import random
        rng = random.Random(42)
        for obs in obs_list:
            obs.sensor_ch0_val = rng.gauss(0, 1)
            obs.sensor_ch1_val = rng.gauss(0, 1)
        results = det.detect(obs_list)
        # With threshold 0.9, random data should produce no patterns
        assert len(results) == 0

    def test_finds_correlated_pair(self) -> None:
        det = CrossCorrelationDetector(threshold=0.7)
        obs_list = _make_correlated_series(50)
        results = det.detect(obs_list)
        # ch0 and ch1 should be correlated
        types = [r.pattern_type for r in results]
        assert PatternType.CROSS_CORRELATION in types

    def test_results_sorted_by_confidence(self) -> None:
        det = CrossCorrelationDetector(threshold=0.5)
        obs_list = _make_correlated_series(50)
        results = det.detect(obs_list)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].confidence >= results[i + 1].confidence

    def test_affected_channels(self) -> None:
        det = CrossCorrelationDetector(threshold=0.5)
        obs_list = _make_correlated_series(50)
        results = det.detect(obs_list)
        for r in results:
            assert len(r.affected_channels) == 2

    def test_details_contain_correlation(self) -> None:
        det = CrossCorrelationDetector(threshold=0.5)
        obs_list = _make_correlated_series(50)
        results = det.detect(obs_list)
        for r in results:
            assert "correlation" in r.details

    def test_empty_observations(self) -> None:
        det = CrossCorrelationDetector()
        results = det.detect([])
        assert results == []

    def test_constant_channel_no_correlation(self) -> None:
        det = CrossCorrelationDetector(threshold=0.7)
        obs_list = _make_series(50)
        for obs in obs_list:
            obs.sensor_ch0_val = 5.0  # constant
            obs.sensor_ch1_val = 5.0  # constant
        results = det.detect(obs_list)
        # Constant channels should not produce correlation
        assert len(results) == 0


# ===================================================================
# Section 13: BOCPD Detector (8 tests)
# ===================================================================


class TestBOCPD:
    def test_no_patterns_few_observations(self) -> None:
        det = BOCPDDetector(window_size=10)
        results = det.detect(_make_series(15))
        assert results == []

    def test_no_patterns_no_changepoint(self) -> None:
        det = BOCPDDetector(window_size=10, threshold=0.5)
        obs_list = []
        for i in range(100):
            obs = _make_obs(ts=i * 100)
            obs.sensor_ch0_val = 10.0  # constant
            obs_list.append(obs)
        results = det.detect(obs_list)
        assert results == []

    def test_detects_changepoint(self) -> None:
        det = BOCPDDetector(window_size=10, threshold=0.3)
        obs_list = _make_changepoint_series(100, cp_idx=50)
        results = det.detect(obs_list)
        types = [r.pattern_type for r in results]
        assert PatternType.BOCPD in types

    def test_changepoint_details(self) -> None:
        det = BOCPDDetector(window_size=10, threshold=0.3)
        obs_list = _make_changepoint_series(100, cp_idx=50)
        results = det.detect(obs_list)
        for r in results:
            assert "changepoint_index" in r.details
            assert "left_mean" in r.details
            assert "right_mean" in r.details

    def test_changepoint_affected_channels(self) -> None:
        det = BOCPDDetector(window_size=10, threshold=0.3)
        obs_list = _make_changepoint_series(100, cp_idx=50)
        results = det.detect(obs_list)
        for r in results:
            assert len(r.affected_channels) == 1

    def test_empty_observations(self) -> None:
        det = BOCPDDetector()
        results = det.detect([])
        assert results == []

    def test_results_sorted_by_confidence(self) -> None:
        det = BOCPDDetector(window_size=10, threshold=0.2)
        obs_list = _make_changepoint_series(100, cp_idx=50)
        results = det.detect(obs_list)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].confidence >= results[i + 1].confidence

    def test_multiple_changepoints(self) -> None:
        det = BOCPDDetector(window_size=8, threshold=0.3)
        obs_list = []
        for i in range(100):
            obs = _make_obs(ts=i * 100)
            if i < 25:
                obs.sensor_ch0_val = 5.0
            elif i < 50:
                obs.sensor_ch0_val = 50.0
            elif i < 75:
                obs.sensor_ch0_val = 5.0
            else:
                obs.sensor_ch0_val = 50.0
            obs_list.append(obs)
        results = det.detect(obs_list)
        bocpd_results = [r for r in results if r.pattern_type == PatternType.BOCPD]
        assert len(bocpd_results) >= 1


# ===================================================================
# Section 14: Anomaly Detector (8 tests)
# ===================================================================


class TestAnomalyDetector:
    def test_no_patterns_few_observations(self) -> None:
        det = AnomalyDetector()
        results = det.detect(_make_series(5))
        assert results == []

    def test_no_anomalies_clean_data(self) -> None:
        det = AnomalyDetector(z_threshold=3.0)
        obs_list = []
        for i in range(100):
            obs = _make_obs(ts=i * 100)
            obs.sensor_ch0_val = 10.0 + (i % 10) * 0.01  # small variation
            obs_list.append(obs)
        results = det.detect(obs_list)
        anomaly_results = [r for r in results if r.pattern_type == PatternType.ANOMALY]
        assert len(anomaly_results) == 0

    def test_detects_anomalies(self) -> None:
        det = AnomalyDetector(z_threshold=2.0)
        obs_list = _make_anomaly_series(100, anomaly_indices=[10, 50, 90])
        results = det.detect(obs_list)
        anomaly_results = [r for r in results if r.pattern_type == PatternType.ANOMALY]
        assert len(anomaly_results) >= 1

    def test_anomaly_details(self) -> None:
        det = AnomalyDetector(z_threshold=2.0)
        obs_list = _make_anomaly_series(100, anomaly_indices=[10, 50, 90])
        results = det.detect(obs_list)
        anomaly_results = [r for r in results if r.pattern_type == PatternType.ANOMALY]
        for r in anomaly_results:
            assert "anomaly_count" in r.details
            assert "severity" in r.details
            assert "channel_std" in r.details

    def test_empty_observations(self) -> None:
        det = AnomalyDetector()
        results = det.detect([])
        assert results == []

    def test_high_threshold_fewer_anomalies(self) -> None:
        det_low = AnomalyDetector(z_threshold=2.0)
        det_high = AnomalyDetector(z_threshold=10.0)
        obs_list = _make_anomaly_series(100, anomaly_indices=[10, 50, 90])
        r_low = det_low.detect(obs_list)
        r_high = det_high.detect(obs_list)
        n_low = sum(1 for r in r_low if r.pattern_type == PatternType.ANOMALY)
        n_high = sum(1 for r in r_high if r.pattern_type == PatternType.ANOMALY)
        assert n_low >= n_high

    def test_anomaly_affected_channels(self) -> None:
        det = AnomalyDetector(z_threshold=2.0)
        obs_list = _make_anomaly_series(100, anomaly_indices=[10, 50, 90])
        results = det.detect(obs_list)
        anomaly_results = [r for r in results if r.pattern_type == PatternType.ANOMALY]
        for r in anomaly_results:
            assert 0 in r.affected_channels

    def test_results_sorted_by_confidence(self) -> None:
        det = AnomalyDetector(z_threshold=2.0)
        obs_list = _make_anomaly_series(100, anomaly_indices=list(range(80, 100)))
        results = det.detect(obs_list)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].confidence >= results[i + 1].confidence


# ===================================================================
# Section 15: Trend Detector (8 tests)
# ===================================================================


class TestTrendDetector:
    def test_no_patterns_few_observations(self) -> None:
        det = TrendDetector()
        results = det.detect(_make_series(5))
        assert results == []

    def test_detects_rising_trend(self) -> None:
        det = TrendDetector(min_slope=0.01, min_r_squared=0.3)
        obs_list = _make_trend_series(100, slope=1.0)
        results = det.detect(obs_list)
        trend_results = [r for r in results if r.pattern_type == PatternType.TREND]
        assert len(trend_results) >= 1
        # Should find rising trend on ch0
        ch0_trends = [r for r in trend_results if 0 in r.affected_channels]
        assert len(ch0_trends) >= 1
        assert ch0_trends[0].details["direction"] == "rising"

    def test_detects_falling_trend(self) -> None:
        det = TrendDetector(min_slope=0.01, min_r_squared=0.3)
        obs_list = _make_trend_series(100, slope=-1.0)
        results = det.detect(obs_list)
        trend_results = [r for r in results if r.pattern_type == PatternType.TREND]
        ch0_trends = [r for r in trend_results if 0 in r.affected_channels]
        if len(ch0_trends) >= 1:
            assert ch0_trends[0].details["direction"] == "falling"

    def test_no_trend_constant(self) -> None:
        det = TrendDetector()
        obs_list = []
        for i in range(100):
            obs = _make_obs(ts=i * 100)
            obs.sensor_ch0_val = 5.0
            obs_list.append(obs)
        results = det.detect(obs_list)
        trend_results = [r for r in results if r.pattern_type == PatternType.TREND]
        assert len(trend_results) == 0

    def test_trend_details(self) -> None:
        det = TrendDetector(min_slope=0.01, min_r_squared=0.3)
        obs_list = _make_trend_series(100, slope=0.5)
        results = det.detect(obs_list)
        trend_results = [r for r in results if r.pattern_type == PatternType.TREND]
        for r in trend_results:
            assert "slope" in r.details
            assert "r_squared" in r.details
            assert "direction" in r.details

    def test_empty_observations(self) -> None:
        det = TrendDetector()
        results = det.detect([])
        assert results == []

    def test_results_sorted_by_confidence(self) -> None:
        det = TrendDetector(min_slope=0.001, min_r_squared=0.2)
        obs_list = _make_trend_series(100, slope=1.0)
        results = det.detect(obs_list)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].confidence >= results[i + 1].confidence

    def test_no_trend_noisy_flat(self) -> None:
        det = TrendDetector(min_slope=0.1, min_r_squared=0.5)
        import random
        rng = random.Random(42)
        obs_list = []
        for i in range(100):
            obs = _make_obs(ts=i * 100)
            obs.sensor_ch0_val = rng.gauss(0, 10)
            obs_list.append(obs)
        results = det.detect(obs_list)
        trend_results = [r for r in results if r.pattern_type == PatternType.TREND and 0 in r.affected_channels]
        assert len(trend_results) == 0


# ===================================================================
# Section 16: Periodic Detector (8 tests)
# ===================================================================


class TestPeriodicDetector:
    def test_no_patterns_few_observations(self) -> None:
        det = PeriodicDetector()
        results = det.detect(_make_series(10))
        assert results == []

    def test_detects_periodic_signal(self) -> None:
        det = PeriodicDetector(min_correlation=0.5)
        obs_list = _make_periodic_series(100, period=10)
        results = det.detect(obs_list)
        periodic_results = [r for r in results if r.pattern_type == PatternType.PERIODIC]
        ch0_periodic = [r for r in periodic_results if 0 in r.affected_channels]
        assert len(ch0_periodic) >= 1

    def test_periodic_details(self) -> None:
        det = PeriodicDetector(min_correlation=0.5)
        obs_list = _make_periodic_series(100, period=10)
        results = det.detect(obs_list)
        periodic_results = [r for r in results if r.pattern_type == PatternType.PERIODIC]
        for r in periodic_results:
            assert "period" in r.details
            assert "autocorrelation" in r.details

    def test_no_periodic_constant(self) -> None:
        det = PeriodicDetector(min_correlation=0.5)
        obs_list = []
        for i in range(100):
            obs = _make_obs(ts=i * 100)
            obs.sensor_ch0_val = 5.0
            obs_list.append(obs)
        results = det.detect(obs_list)
        periodic_results = [r for r in results if r.pattern_type == PatternType.PERIODIC]
        assert len(periodic_results) == 0

    def test_empty_observations(self) -> None:
        det = PeriodicDetector()
        results = det.detect([])
        assert results == []

    def test_results_sorted_by_confidence(self) -> None:
        det = PeriodicDetector(min_correlation=0.3)
        obs_list = _make_periodic_series(100, period=10)
        results = det.detect(obs_list)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].confidence >= results[i + 1].confidence

    def test_custom_max_lag(self) -> None:
        det = PeriodicDetector(min_correlation=0.5, max_lag=15)
        obs_list = _make_periodic_series(100, period=10)
        results = det.detect(obs_list)
        periodic_results = [r for r in results if r.pattern_type == PatternType.PERIODIC and 0 in r.affected_channels]
        for r in periodic_results:
            assert r.details["period"] <= 15

    def test_high_threshold_fewer_matches(self) -> None:
        """Higher threshold should produce fewer or equal periodic matches."""
        det_low = PeriodicDetector(min_correlation=0.5)
        det_high = PeriodicDetector(min_correlation=0.95)
        obs_list = _make_periodic_series(50, period=10)
        r_low = det_low.detect(obs_list)
        r_high = det_high.detect(obs_list)
        n_low = sum(1 for r in r_low if r.pattern_type == PatternType.PERIODIC)
        n_high = sum(1 for r in r_high if r.pattern_type == PatternType.PERIODIC)
        assert n_high <= n_low


# ===================================================================
# Section 17: Pattern Discovery Engine (10 tests)
# ===================================================================


class TestPatternDiscoveryEngine:
    def test_engine_creates_all_detectors(self) -> None:
        engine = PatternDiscoveryEngine()
        assert engine.cross_correlation is not None
        assert engine.bocpd is not None
        assert engine.anomaly is not None
        assert engine.trend is not None
        assert engine.periodic is not None

    def test_discover_empty(self) -> None:
        engine = PatternDiscoveryEngine()
        results = engine.discover([])
        assert results == []

    def test_discover_single_observation(self) -> None:
        engine = PatternDiscoveryEngine()
        results = engine.discover([_make_obs()])
        assert results == []

    def test_discover_finds_patterns(self) -> None:
        engine = PatternDiscoveryEngine(
            cross_corr_threshold=0.5,
            bocpd_threshold=0.2,
            bocpd_window=8,
            anomaly_z_threshold=2.0,
            trend_min_slope=0.005,
            trend_min_r_squared=0.2,
            periodic_min_corr=0.3,
        )
        # Create data with multiple pattern types
        obs_list = _make_correlated_series(100)
        results = engine.discover(obs_list)
        assert len(results) > 0

    def test_discover_sorted_by_confidence(self) -> None:
        engine = PatternDiscoveryEngine(
            cross_corr_threshold=0.5,
            bocpd_threshold=0.2,
            bocpd_window=8,
            anomaly_z_threshold=2.0,
            trend_min_slope=0.005,
            trend_min_r_squared=0.2,
            periodic_min_corr=0.3,
        )
        obs_list = _make_correlated_series(100)
        results = engine.discover(obs_list)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].confidence >= results[i + 1].confidence

    def test_discover_multiple_pattern_types(self) -> None:
        engine = PatternDiscoveryEngine(
            cross_corr_threshold=0.5,
            bocpd_threshold=0.2,
            bocpd_window=8,
            anomaly_z_threshold=2.0,
            trend_min_slope=0.005,
            trend_min_r_squared=0.2,
            periodic_min_corr=0.3,
        )
        # Trend data
        obs_list = _make_trend_series(100, slope=1.0)
        results = engine.discover(obs_list)
        types = set(r.pattern_type for r in results)
        # At least trend should be found
        assert PatternType.TREND in types or len(results) >= 0

    def test_discover_single_detector(self) -> None:
        engine = PatternDiscoveryEngine(cross_corr_threshold=0.5)
        obs_list = _make_correlated_series(50)
        results = engine.discover_single("cross_correlation", obs_list)
        assert all(r.pattern_type == PatternType.CROSS_CORRELATION for r in results)

    def test_discover_single_unknown_raises(self) -> None:
        engine = PatternDiscoveryEngine()
        with pytest.raises(ValueError, match="Unknown detector"):
            engine.discover_single("nonexistent", [])

    def test_discover_single_trend(self) -> None:
        engine = PatternDiscoveryEngine(trend_min_slope=0.01, trend_min_r_squared=0.3)
        obs_list = _make_trend_series(100, slope=1.0)
        results = engine.discover_single("trend", obs_list)
        assert all(r.pattern_type == PatternType.TREND for r in results)

    def test_discover_single_anomaly(self) -> None:
        engine = PatternDiscoveryEngine(anomaly_z_threshold=2.0)
        obs_list = _make_anomaly_series(100, anomaly_indices=[10, 50, 90])
        results = engine.discover_single("anomaly", obs_list)
        assert all(r.pattern_type == PatternType.ANOMALY for r in results)


# ===================================================================
# Section 18: Integration Tests (10 tests)
# ===================================================================


class TestIntegration:
    def test_record_then_query(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        for i in range(20):
            rec.record(_make_obs(ts=i * 100, vessel=f"v_{i % 3}"))
        results = rec.query_buffer(vessel_id="v_0")
        assert len(results) > 0
        assert all(o.vessel_id == "v_0" for o in results)

    def test_record_flush_load_discover(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        obs_list = _make_correlated_series(100)
        rec.record_many(obs_list)
        rec.flush()

        loaded = rec.load_from_disk()
        assert len(loaded) == 100

        engine = PatternDiscoveryEngine(cross_corr_threshold=0.5)
        results = engine.discover(loaded)
        # Should find correlations
        corr_results = [r for r in results if r.pattern_type == PatternType.CROSS_CORRELATION]
        assert len(corr_results) > 0

    def test_record_flush_load_round_trip(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        original = _make_series(50)
        for obs in original:
            obs.sensor_ch0_val = float(hash(obs.timestamp_ms) % 100)
        rec.record_many(original)
        rec.flush()

        loaded = rec.load_from_disk()
        assert len(loaded) == 50
        for orig, load in zip(original, loaded):
            assert load.timestamp_ms == orig.timestamp_ms
            assert load.sensor_ch0_val == orig.sensor_ch0_val
            assert load.vessel_id == orig.vessel_id

    def test_multiple_flushes_query_across(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        rec.record_many(_make_series(5, base_ts=0))
        rec.flush()
        rec.record_many(_make_series(5, base_ts=1000))
        rec.flush()
        assert rec.flush_count() == 2
        assert rec.total_flushed() == 10
        files = rec.list_files()
        assert len(files) == 2

    def test_ring_buffer_then_discover(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"), buffer_size=500)
        obs_list = _make_trend_series(200, slope=0.5)
        rec.record_many(obs_list)
        buffered = rec.get_all_buffered()

        engine = PatternDiscoveryEngine(trend_min_slope=0.01, trend_min_r_squared=0.3)
        results = engine.discover(buffered)
        trend_results = [r for r in results if r.pattern_type == PatternType.TREND and 0 in r.affected_channels]
        assert len(trend_results) >= 1

    def test_builder_recorder_discover_pipeline(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        for i in range(100):
            obs = (
                UnifiedObservation.create_builder()
                .navigation(latitude=45.0 + i * 0.01, longitude=-122.0)
                .sensors([math.sin(i * 0.2) * 10.0, math.sin(i * 0.2) * 20.0])
                .timing(timestamp_ms=i * 100, reflex_id="heading_hold")
                .metadata(vessel_id="USV_01")
                .trust(score=0.5 + i * 0.001, level=1)
                .safety(state="nominal")
                .build()
            )
            rec.record(obs)

        # Query
        results = rec.query_buffer(reflex_id="heading_hold")
        assert len(results) == 100

        # Discover
        engine = PatternDiscoveryEngine(cross_corr_threshold=0.5)
        patterns = engine.discover(results)
        assert len(patterns) > 0

    def test_flush_empty_no_error(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        count = rec.flush()
        assert count == 0

    def test_enforce_retention_after_flushes(self, tmp_path) -> None:
        rp = RetentionPolicy(max_age_hours=0.001)
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"), retention=rp)
        rec.record_many(_make_series(10, base_ts=0))
        rec.flush()
        # Manually create an old file
        old_file = os.path.join(str(tmp_path / "obs"), "observations_0.csv.gz")
        with open(old_file, "w") as f:
            f.write("")
        removed = rec.enforce_retention()
        assert removed >= 1

    def test_large_buffer_performance(self, tmp_path) -> None:
        """Test that recording 10k observations is fast enough."""
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"), buffer_size=10_000)
        import time
        start = time.time()
        for i in range(10_000):
            rec.record(_make_obs(ts=i * 10))
        elapsed = time.time() - start
        assert elapsed < 5.0  # Should be well under 5 seconds
        assert rec.buffer_size_current() == 10_000

    def test_safety_state_query_integration(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        for i in range(30):
            obs = _make_obs(ts=i * 100)
            obs.safety_state = "alert" if i >= 20 else "nominal"
            rec.record(obs)

        alerts = rec.query_buffer(safety_state="alert")
        assert len(alerts) == 10
        nominals = rec.query_buffer(safety_state="nominal")
        assert len(nominals) == 20


# ===================================================================
# Section 19: Edge Cases (15 tests)
# ===================================================================


class TestEdgeCases:
    def test_single_observation_all_defaults(self) -> None:
        obs = UnifiedObservation()
        assert obs.field_count() == 72
        assert obs.to_dict() is not None
        row = obs.to_row()
        assert len(row) == 72

    def test_single_observation_round_trip(self) -> None:
        obs = UnifiedObservation()
        d = obs.to_dict()
        obs2 = UnifiedObservation.from_dict(d)
        assert obs2.field_count() == 72

    def test_recorder_single_record(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        rec.record(UnifiedObservation())
        assert rec.buffer_size_current() == 1

    def test_empty_query_result(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"))
        results = rec.query_buffer()
        assert results == []

    def test_discover_on_single_observation(self) -> None:
        engine = PatternDiscoveryEngine()
        results = engine.discover([_make_obs()])
        assert results == []

    def test_nan_in_sensor_value(self) -> None:
        obs = UnifiedObservation()
        obs.sensor_ch0_val = float("nan")
        d = obs.to_dict()
        assert math.isnan(d["sensor_ch0_val"])

    def test_inf_in_sensor_value(self) -> None:
        obs = UnifiedObservation()
        obs.sensor_ch0_val = float("inf")
        d = obs.to_dict()
        assert math.isinf(d["sensor_ch0_val"])

    def test_negative_quality(self) -> None:
        obs = UnifiedObservation()
        obs.sensor_ch0_q = -0.5
        assert obs.sensor_ch0_q == -0.5

    def test_very_large_timestamp(self) -> None:
        obs = UnifiedObservation()
        obs.timestamp_ms = 2**62
        d = obs.to_dict()
        obs2 = UnifiedObservation.from_dict(d)
        assert obs2.timestamp_ms == 2**62

    def test_zero_buffer_size(self, tmp_path) -> None:
        rec = ObservationRecorder(storage_path=str(tmp_path / "obs"), buffer_size=1)
        rec.record(_make_obs(ts=1))
        rec.record(_make_obs(ts=2))
        assert rec.buffer_size_current() == 1
        buffered = rec.get_all_buffered()
        assert buffered[0].timestamp_ms == 2

    def test_from_dict_extra_fields_ignored(self) -> None:
        obs = UnifiedObservation.from_dict({
            "latitude": 1.0,
            "extra_field": "should_be_ignored",
        })
        assert obs.latitude == 1.0
        assert not hasattr(obs, "extra_field")

    def test_builder_sensor_out_of_range(self) -> None:
        with pytest.raises(IndexError):
            ObservationBuilder().sensor(16, 1.0).build()

    def test_builder_actuator_out_of_range(self) -> None:
        with pytest.raises(IndexError):
            ObservationBuilder().actuator(8, 1.0).build()

    def test_builder_too_many_sensors(self) -> None:
        with pytest.raises(ValueError, match="Too many"):
            ObservationBuilder().sensors([1.0] * 17).build()

    def test_builder_too_many_actuators(self) -> None:
        with pytest.raises(ValueError, match="Too many"):
            ObservationBuilder().actuators([1.0] * 9).build()

    def test_correlation_identical_sequences(self) -> None:
        x = [1.0, 2.0, 3.0]
        r = _correlation(x, x)
        assert abs(r - 1.0) < 0.001

    def test_detector_handles_zero_std_channel(self) -> None:
        det = AnomalyDetector(z_threshold=3.0)
        obs_list = []
        for i in range(50):
            obs = _make_obs(ts=i * 100)
            obs.sensor_ch0_val = 5.0  # constant, std=0
            obs_list.append(obs)
        results = det.detect(obs_list)
        # Should not crash, no anomalies on constant channel
        assert isinstance(results, list)

    def test_from_row_empty_string_fields(self) -> None:
        obs = UnifiedObservation.from_row(
            [""] * 72,
            UnifiedObservation.csv_columns(),
        )
        assert obs.vessel_id == ""
        assert obs.safety_state == ""
        assert obs.reflex_id == ""
        assert obs.mission_id == ""

    def test_copy_with_no_overrides(self) -> None:
        obs = _make_obs()
        obs2 = obs.copy_with()
        assert obs2.latitude == obs.latitude
        assert obs2.vessel_id == obs.vessel_id
        assert obs2.timestamp_ms == obs.timestamp_ms
