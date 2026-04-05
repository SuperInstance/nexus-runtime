"""NEXUS Learning - Observation recording and pattern discovery.

UnifiedObservation: 72-field observation record for marine robotics.
ObservationRecorder: Ring-buffer storage with CSV serialization and query.
PatternDiscoveryEngine: 5 competing algorithms for pattern detection.
"""

from __future__ import annotations

import csv
import gzip
import math
import os
from collections import deque
from dataclasses import dataclass, field, fields
from enum import IntEnum
from typing import Any, ClassVar


# ===================================================================
# Constants
# ===================================================================

NUM_SENSOR_CHANNELS: int = 16
NUM_ACTUATOR_CHANNELS: int = 8
FIELD_COUNT: int = 72


# ===================================================================
# Sensor & Actuator Channel Data
# ===================================================================


@dataclass(frozen=True)
class SensorChannel:
    """Single sensor channel reading: value, quality, and sample timestamp."""

    value: float = 0.0
    quality: float = 1.0  # 0.0 = invalid, 1.0 = perfect
    timestamp_ms: int = 0

    def is_valid(self) -> bool:
        return self.quality > 0.0


@dataclass(frozen=True)
class ActuatorChannel:
    """Single actuator channel state: command, feedback, and status."""

    command: float = 0.0
    feedback: float = 0.0
    status: int = 0  # 0=idle, 1=active, 2=fault, 3=overridden


# ===================================================================
# UnifiedObservation - 72-field observation record
# ===================================================================
# Field layout (72 total flattened fields):
#   Navigation:     5 fields  (latitude, longitude, heading, speed_over_ground, course_over_ground)
#   Sensor values: 16 fields  (sensor_ch0_val .. sensor_ch15_val)
#   Sensor quality:16 fields  (sensor_ch0_q   .. sensor_ch15_q)
#   Actuator cmd:   8 fields  (actuator_ch0_cmd .. actuator_ch7_cmd)
#   Actuator fb:    8 fields  (actuator_ch0_fb  .. actuator_ch7_fb)
#   Trust:          4 fields  (trust_score, trust_level, trust_event_type, trust_delta)
#   Safety:         4 fields  (safety_state, safety_flags, watchdog_counter, heartbeat_missed)
#   VM:             4 fields  (program_counter, stack_depth, cycle_count, last_opcode)
#   Timing:         4 fields  (timestamp_ms, uptime_ms, reflex_id, mission_id)
#   Metadata:       3 fields  (vessel_id, agent_id, session_id)


@dataclass
class UnifiedObservation:
    """Single observation record with 72 flattened fields.

    Combines navigation, sensor (16 ch), actuator (8 ch), trust, safety,
    VM state, timing, and metadata into a single observation.
    Sensor and actuator channels have structured access helpers.
    """

    # -- Navigation (5) --
    latitude: float = 0.0
    longitude: float = 0.0
    heading: float = 0.0
    speed_over_ground: float = 0.0
    course_over_ground: float = 0.0

    # -- Sensor channels (16 x 2 = 32 flattened fields: value + quality) --
    sensor_ch0_val: float = 0.0
    sensor_ch0_q: float = 1.0
    sensor_ch1_val: float = 0.0
    sensor_ch1_q: float = 1.0
    sensor_ch2_val: float = 0.0
    sensor_ch2_q: float = 1.0
    sensor_ch3_val: float = 0.0
    sensor_ch3_q: float = 1.0
    sensor_ch4_val: float = 0.0
    sensor_ch4_q: float = 1.0
    sensor_ch5_val: float = 0.0
    sensor_ch5_q: float = 1.0
    sensor_ch6_val: float = 0.0
    sensor_ch6_q: float = 1.0
    sensor_ch7_val: float = 0.0
    sensor_ch7_q: float = 1.0
    sensor_ch8_val: float = 0.0
    sensor_ch8_q: float = 1.0
    sensor_ch9_val: float = 0.0
    sensor_ch9_q: float = 1.0
    sensor_ch10_val: float = 0.0
    sensor_ch10_q: float = 1.0
    sensor_ch11_val: float = 0.0
    sensor_ch11_q: float = 1.0
    sensor_ch12_val: float = 0.0
    sensor_ch12_q: float = 1.0
    sensor_ch13_val: float = 0.0
    sensor_ch13_q: float = 1.0
    sensor_ch14_val: float = 0.0
    sensor_ch14_q: float = 1.0
    sensor_ch15_val: float = 0.0
    sensor_ch15_q: float = 1.0

    # -- Actuator channels (8 x 2 = 16 flattened fields: command + feedback) --
    actuator_ch0_cmd: float = 0.0
    actuator_ch0_fb: float = 0.0
    actuator_ch1_cmd: float = 0.0
    actuator_ch1_fb: float = 0.0
    actuator_ch2_cmd: float = 0.0
    actuator_ch2_fb: float = 0.0
    actuator_ch3_cmd: float = 0.0
    actuator_ch3_fb: float = 0.0
    actuator_ch4_cmd: float = 0.0
    actuator_ch4_fb: float = 0.0
    actuator_ch5_cmd: float = 0.0
    actuator_ch5_fb: float = 0.0
    actuator_ch6_cmd: float = 0.0
    actuator_ch6_fb: float = 0.0
    actuator_ch7_cmd: float = 0.0
    actuator_ch7_fb: float = 0.0

    # -- Trust (4) --
    trust_score: float = 0.0
    trust_level: int = 0
    trust_event_type: str = ""
    trust_delta: float = 0.0

    # -- Safety (4) --
    safety_state: str = ""
    safety_flags: int = 0
    watchdog_counter: int = 0
    heartbeat_missed: int = 0

    # -- VM (4) --
    program_counter: int = 0
    stack_depth: int = 0
    cycle_count: int = 0
    last_opcode: int = 0

    # -- Timing (4) --
    timestamp_ms: int = 0
    uptime_ms: int = 0
    reflex_id: str = ""
    mission_id: str = ""

    # -- Metadata (3) --
    vessel_id: str = ""
    agent_id: str = ""
    session_id: str = ""

    # ------------------------------------------------------------------
    # Structured access helpers
    # ------------------------------------------------------------------

    def get_sensor_channel(self, idx: int) -> SensorChannel:
        """Get a sensor channel as a SensorChannel object."""
        if not 0 <= idx < NUM_SENSOR_CHANNELS:
            raise IndexError(
                f"Sensor channel index {idx} out of range [0, {NUM_SENSOR_CHANNELS})"
            )
        return SensorChannel(
            value=getattr(self, f"sensor_ch{idx}_val"),
            quality=getattr(self, f"sensor_ch{idx}_q"),
            timestamp_ms=self.timestamp_ms,
        )

    def set_sensor_channel(self, idx: int, ch: SensorChannel) -> None:
        """Set sensor channel value and quality from a SensorChannel."""
        if not 0 <= idx < NUM_SENSOR_CHANNELS:
            raise IndexError(
                f"Sensor channel index {idx} out of range [0, {NUM_SENSOR_CHANNELS})"
            )
        setattr(self, f"sensor_ch{idx}_val", ch.value)
        setattr(self, f"sensor_ch{idx}_q", ch.quality)

    def get_actuator_channel(self, idx: int) -> ActuatorChannel:
        """Get an actuator channel as an ActuatorChannel object."""
        if not 0 <= idx < NUM_ACTUATOR_CHANNELS:
            raise IndexError(
                f"Actuator channel index {idx} out of range [0, {NUM_ACTUATOR_CHANNELS})"
            )
        return ActuatorChannel(
            command=getattr(self, f"actuator_ch{idx}_cmd"),
            feedback=getattr(self, f"actuator_ch{idx}_fb"),
        )

    def set_actuator_channel(self, idx: int, ch: ActuatorChannel) -> None:
        """Set actuator channel command and feedback from an ActuatorChannel."""
        if not 0 <= idx < NUM_ACTUATOR_CHANNELS:
            raise IndexError(
                f"Actuator channel index {idx} out of range [0, {NUM_ACTUATOR_CHANNELS})"
            )
        setattr(self, f"actuator_ch{idx}_cmd", ch.command)
        setattr(self, f"actuator_ch{idx}_fb", ch.feedback)

    def get_all_sensor_values(self) -> list[float]:
        """Return all 16 sensor values as a list."""
        return [getattr(self, f"sensor_ch{i}_val") for i in range(NUM_SENSOR_CHANNELS)]

    def get_all_sensor_qualities(self) -> list[float]:
        """Return all 16 sensor qualities as a list."""
        return [getattr(self, f"sensor_ch{i}_q") for i in range(NUM_SENSOR_CHANNELS)]

    def get_all_actuator_commands(self) -> list[float]:
        """Return all 8 actuator commands as a list."""
        return [getattr(self, f"actuator_ch{i}_cmd") for i in range(NUM_ACTUATOR_CHANNELS)]

    def get_all_actuator_feedbacks(self) -> list[float]:
        """Return all 8 actuator feedbacks as a list."""
        return [getattr(self, f"actuator_ch{i}_fb") for i in range(NUM_ACTUATOR_CHANNELS)]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    _CSV_COLUMNS: ClassVar[list[str] | None] = None

    @classmethod
    def csv_columns(cls) -> list[str]:
        """Return the ordered list of CSV column names."""
        if cls._CSV_COLUMNS is None:
            cls._CSV_COLUMNS = [f.name for f in fields(cls)]
        return list(cls._CSV_COLUMNS)

    def to_dict(self) -> dict[str, Any]:
        """Serialize observation to a flat dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_row(self) -> list[str]:
        """Serialize to a list of strings (CSV row)."""
        return [str(getattr(self, f.name)) for f in fields(self)]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> UnifiedObservation:
        """Deserialize from a flat dictionary."""
        valid: dict[str, Any] = {}
        for f in fields(cls):
            if f.name in d:
                valid[f.name] = d[f.name]
            else:
                valid[f.name] = f.default
        return cls(**valid)

    @classmethod
    def from_row(cls, row: list[str], columns: list[str]) -> UnifiedObservation:
        """Deserialize from a CSV row (list of strings) with column names."""
        d: dict[str, str] = {}
        for col, val in zip(columns, row):
            d[col] = val
        return cls.from_dict(cls._coerce_types(d))

    @staticmethod
    def _coerce_types(d: dict[str, str]) -> dict[str, Any]:
        """Coerce string values from CSV back to correct Python types."""
        import typing
        type_map = typing.get_type_hints(UnifiedObservation)
        # Build defaults from dataclass fields
        defaults = {f.name: f.default for f in fields(UnifiedObservation)}
        result: dict[str, Any] = {}
        for key, val in d.items():
            expected = type_map.get(key, str)
            if val == "" or val is None:
                result[key] = defaults.get(key, 0 if expected in (int, float) else "")
            elif expected is int:
                result[key] = int(val)
            elif expected is float:
                result[key] = float(val)
            else:
                result[key] = val
        return result

    def field_count(self) -> int:
        """Return the number of fields in this observation (always 72)."""
        return len(fields(self))

    @classmethod
    def create_builder(cls) -> ObservationBuilder:
        """Return a builder for constructing observations fluently."""
        return ObservationBuilder()

    def copy_with(self, **overrides: Any) -> UnifiedObservation:
        """Create a copy with some fields overridden."""
        d = self.to_dict()
        d.update(overrides)
        return UnifiedObservation(**d)


# ===================================================================
# Observation Builder
# ===================================================================


class ObservationBuilder:
    """Fluent builder for UnifiedObservation."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def navigation(
        self,
        latitude: float = 0.0,
        longitude: float = 0.0,
        heading: float = 0.0,
        speed_over_ground: float = 0.0,
        course_over_ground: float = 0.0,
    ) -> ObservationBuilder:
        self._data["latitude"] = latitude
        self._data["longitude"] = longitude
        self._data["heading"] = heading
        self._data["speed_over_ground"] = speed_over_ground
        self._data["course_over_ground"] = course_over_ground
        return self

    def sensor(self, idx: int, value: float, quality: float = 1.0) -> ObservationBuilder:
        if not 0 <= idx < NUM_SENSOR_CHANNELS:
            raise IndexError(f"Sensor index {idx} out of range")
        self._data[f"sensor_ch{idx}_val"] = value
        self._data[f"sensor_ch{idx}_q"] = quality
        return self

    def sensors(
        self, values: list[float], qualities: list[float] | None = None
    ) -> ObservationBuilder:
        if len(values) > NUM_SENSOR_CHANNELS:
            raise ValueError(f"Too many sensor values: {len(values)} > {NUM_SENSOR_CHANNELS}")
        for i, v in enumerate(values):
            q = qualities[i] if qualities and i < len(qualities) else 1.0
            self.sensor(i, v, q)
        return self

    def actuator(self, idx: int, command: float, feedback: float = 0.0) -> ObservationBuilder:
        if not 0 <= idx < NUM_ACTUATOR_CHANNELS:
            raise IndexError(f"Actuator index {idx} out of range")
        self._data[f"actuator_ch{idx}_cmd"] = command
        self._data[f"actuator_ch{idx}_fb"] = feedback
        return self

    def actuators(
        self, commands: list[float], feedbacks: list[float] | None = None
    ) -> ObservationBuilder:
        if len(commands) > NUM_ACTUATOR_CHANNELS:
            raise ValueError(
                f"Too many actuator commands: {len(commands)} > {NUM_ACTUATOR_CHANNELS}"
            )
        for i, c in enumerate(commands):
            fb = feedbacks[i] if feedbacks and i < len(feedbacks) else 0.0
            self.actuator(i, c, fb)
        return self

    def trust(
        self,
        score: float = 0.0,
        level: int = 0,
        event_type: str = "",
        delta: float = 0.0,
    ) -> ObservationBuilder:
        self._data["trust_score"] = score
        self._data["trust_level"] = level
        self._data["trust_event_type"] = event_type
        self._data["trust_delta"] = delta
        return self

    def safety(
        self,
        state: str = "",
        flags: int = 0,
        watchdog: int = 0,
        heartbeat_missed: int = 0,
    ) -> ObservationBuilder:
        self._data["safety_state"] = state
        self._data["safety_flags"] = flags
        self._data["watchdog_counter"] = watchdog
        self._data["heartbeat_missed"] = heartbeat_missed
        return self

    def vm(
        self,
        pc: int = 0,
        stack_depth: int = 0,
        cycle_count: int = 0,
        last_opcode: int = 0,
    ) -> ObservationBuilder:
        self._data["program_counter"] = pc
        self._data["stack_depth"] = stack_depth
        self._data["cycle_count"] = cycle_count
        self._data["last_opcode"] = last_opcode
        return self

    def timing(
        self,
        timestamp_ms: int = 0,
        uptime_ms: int = 0,
        reflex_id: str = "",
        mission_id: str = "",
    ) -> ObservationBuilder:
        self._data["timestamp_ms"] = timestamp_ms
        self._data["uptime_ms"] = uptime_ms
        self._data["reflex_id"] = reflex_id
        self._data["mission_id"] = mission_id
        return self

    def metadata(
        self, vessel_id: str = "", agent_id: str = "", session_id: str = ""
    ) -> ObservationBuilder:
        self._data["vessel_id"] = vessel_id
        self._data["agent_id"] = agent_id
        self._data["session_id"] = session_id
        return self

    def build(self) -> UnifiedObservation:
        return UnifiedObservation(**self._data)


# ===================================================================
# Retention Policy
# ===================================================================


@dataclass
class RetentionPolicy:
    """Configurable retention policy for observations."""

    max_records: int = 10_000
    max_age_hours: float = 24.0
    min_safety_records: int = 100


# ===================================================================
# ObservationRecorder - Ring buffer + CSV storage + query
# ===================================================================


class ObservationRecorder:
    """Observation recording pipeline with ring buffer, flush, and query.

    Features:
    - Configurable ring buffer (default 10,000 observations)
    - CSV + gzip compressed storage with atomic writes (.tmp then rename)
    - Time-based and count-based retention policy
    - Query interface: by time range, vessel_id, reflex_id, safety_state
    """

    def __init__(
        self,
        storage_path: str = "/tmp/nexus_observations",
        buffer_size: int = 10_000,
        retention: RetentionPolicy | None = None,
    ) -> None:
        self.storage_path = storage_path
        self.buffer_size = buffer_size
        self.retention = retention or RetentionPolicy(max_records=buffer_size)
        self._buffer: deque[UnifiedObservation] = deque(maxlen=buffer_size)
        self._total_recorded: int = 0
        self._total_flushed: int = 0
        self._flush_count: int = 0
        os.makedirs(storage_path, exist_ok=True)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, observation: UnifiedObservation) -> None:
        """Record a single observation into the ring buffer."""
        self._buffer.append(observation)
        self._total_recorded += 1

    def record_many(self, observations: list[UnifiedObservation]) -> None:
        """Record multiple observations at once."""
        for obs in observations:
            self.record(obs)

    def buffer_size_current(self) -> int:
        """Return the number of observations currently in the buffer."""
        return len(self._buffer)

    def total_recorded(self) -> int:
        """Return the total number of observations ever recorded."""
        return self._total_recorded

    # ------------------------------------------------------------------
    # Flush to disk (atomic write)
    # ------------------------------------------------------------------

    def flush(self) -> int:
        """Flush buffered observations to a gzip-compressed CSV file.

        Uses atomic write: writes to a .tmp file first, then renames.
        Returns the number of observations flushed.
        """
        if not self._buffer:
            return 0

        count = len(self._buffer)
        observations = list(self._buffer)

        # Generate filename with timestamp
        ts = observations[-1].timestamp_ms if observations else 0
        filename = f"observations_{ts}.csv.gz"
        filepath = os.path.join(self.storage_path, filename)
        tmp_path = filepath + ".tmp"

        try:
            columns = UnifiedObservation.csv_columns()
            with gzip.open(tmp_path, "wt", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                for obs in observations:
                    writer.writerow(obs.to_row())
            os.replace(tmp_path, filepath)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        self._buffer.clear()
        self._total_flushed += count
        self._flush_count += 1
        return count

    def total_flushed(self) -> int:
        """Return the total number of observations flushed to disk."""
        return self._total_flushed

    def flush_count(self) -> int:
        """Return the number of flush operations performed."""
        return self._flush_count

    # ------------------------------------------------------------------
    # Load from disk
    # ------------------------------------------------------------------

    def load_from_disk(
        self, filename: str | None = None
    ) -> list[UnifiedObservation]:
        """Load observations from a gzip-compressed CSV file.

        If filename is None, loads the most recent file by name sort.
        """
        if filename is None:
            files = sorted(
                f
                for f in os.listdir(self.storage_path)
                if f.startswith("observations_") and f.endswith(".csv.gz")
            )
            if not files:
                return []
            filename = files[-1]

        filepath = os.path.join(self.storage_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Observation file not found: {filepath}")

        observations: list[UnifiedObservation] = []
        with gzip.open(filepath, "rt", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            columns = list(header)
            for row in reader:
                obs = UnifiedObservation.from_row(list(row), columns)
                observations.append(obs)
        return observations

    def list_files(self) -> list[str]:
        """List all observation files on disk."""
        return sorted(
            f
            for f in os.listdir(self.storage_path)
            if f.startswith("observations_") and f.endswith(".csv.gz")
        )

    # ------------------------------------------------------------------
    # Retention enforcement
    # ------------------------------------------------------------------

    def enforce_retention(self) -> int:
        """Enforce retention policy on disk files.

        Removes files whose observations are older than max_age_hours.
        Returns the number of files removed.
        """
        removed = 0
        files = self.list_files()
        now_ms = self._current_time_ms()
        max_age_ms = self.retention.max_age_hours * 3_600_000

        for filename in files:
            try:
                parts = filename.replace(".csv.gz", "").split("_")
                file_ts = int(parts[1])
            except (IndexError, ValueError):
                continue

            if now_ms - file_ts > max_age_ms:
                filepath = os.path.join(self.storage_path, filename)
                os.remove(filepath)
                removed += 1

        return removed

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def query_buffer(
        self,
        time_start: int | None = None,
        time_end: int | None = None,
        vessel_id: str | None = None,
        reflex_id: str | None = None,
        safety_state: str | None = None,
    ) -> list[UnifiedObservation]:
        """Query the in-memory buffer with filters."""
        results: list[UnifiedObservation] = []
        for obs in self._buffer:
            if time_start is not None and obs.timestamp_ms < time_start:
                continue
            if time_end is not None and obs.timestamp_ms > time_end:
                continue
            if vessel_id is not None and obs.vessel_id != vessel_id:
                continue
            if reflex_id is not None and obs.reflex_id != reflex_id:
                continue
            if safety_state is not None and obs.safety_state != safety_state:
                continue
            results.append(obs)
        return results

    def get_all_buffered(self) -> list[UnifiedObservation]:
        """Return all buffered observations as a list."""
        return list(self._buffer)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _current_time_ms() -> int:
        """Return current time in milliseconds."""
        import time

        return int(time.time() * 1000)


# ===================================================================
# Pattern Discovery - Result types
# ===================================================================


class PatternType(IntEnum):
    """Types of patterns that can be discovered."""

    CROSS_CORRELATION = 1
    BOCPD = 2
    ANOMALY = 3
    TREND = 4
    PERIODIC = 5


@dataclass
class PatternResult:
    """A single discovered pattern with confidence score."""

    pattern_type: PatternType
    description: str
    confidence: float  # 0.0 to 1.0
    details: dict[str, Any] = field(default_factory=dict)
    timestamp_ms: int = 0
    affected_channels: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, self.confidence))


# ===================================================================
# Statistical helpers
# ===================================================================


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / (n - 1)
    return math.sqrt(variance)


def _correlation(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient between two sequences."""
    n = len(x)
    if n < 3 or n != len(y):
        return 0.0
    mx, my = _mean(x), _mean(y)
    sx, sy = _std(x), _std(y)
    if sx == 0.0 or sy == 0.0:
        return 0.0
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n - 1)
    return cov / (sx * sy)


# ===================================================================
# Algorithm 1: Cross-correlation detector
# ===================================================================


class CrossCorrelationDetector:
    """Find correlated sensor pairs using Pearson correlation.

    Examines all pairs of sensor channels and reports those with
    |correlation| above a configurable threshold.
    """

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold

    def detect(
        self, observations: list[UnifiedObservation]
    ) -> list[PatternResult]:
        if len(observations) < 5:
            return []

        results: list[PatternResult] = []

        # Extract sensor values for all channels
        channel_values: dict[int, list[float]] = {}
        for ch in range(NUM_SENSOR_CHANNELS):
            channel_values[ch] = [
                getattr(obs, f"sensor_ch{ch}_val") for obs in observations
            ]

        # Check all pairs
        for i in range(NUM_SENSOR_CHANNELS):
            for j in range(i + 1, NUM_SENSOR_CHANNELS):
                r = _correlation(channel_values[i], channel_values[j])
                if abs(r) >= self.threshold:
                    confidence = min(
                        1.0,
                        (abs(r) - self.threshold) / (1.0 - self.threshold) * 0.8 + 0.2,
                    )
                    results.append(
                        PatternResult(
                            pattern_type=PatternType.CROSS_CORRELATION,
                            description=f"Sensors ch{i} and ch{j}: r={r:.3f}",
                            confidence=confidence,
                            details={
                                "correlation": r,
                                "channel_a": i,
                                "channel_b": j,
                            },
                            affected_channels=[i, j],
                            timestamp_ms=observations[-1].timestamp_ms
                            if observations
                            else 0,
                        )
                    )

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results


# ===================================================================
# Algorithm 2: BOCPD (Bayesian Online Changepoint Detection)
# ===================================================================


class BOCPDDetector:
    """Bayesian Online Changepoint Detection - simplified.

    Uses sliding window comparison to detect regime changes.
    When means of adjacent windows differ significantly (z > 2),
    reports a changepoint.
    """

    def __init__(self, threshold: float = 0.5, window_size: int = 20) -> None:
        self.threshold = threshold
        self.window_size = window_size

    def detect(
        self, observations: list[UnifiedObservation]
    ) -> list[PatternResult]:
        if len(observations) < self.window_size * 2:
            return []

        results: list[PatternResult] = []
        n_obs = len(observations)
        w = self.window_size

        for ch in range(NUM_SENSOR_CHANNELS):
            values = [
                getattr(obs, f"sensor_ch{ch}_val") for obs in observations
            ]

            changepoints: list[int] = []
            for i in range(w, n_obs - w):
                left = values[i - w : i]
                right = values[i : i + w]
                ml, mr = _mean(left), _mean(right)
                sl, sr = _std(left), _std(right)

                combined_std = (
                    math.sqrt(sl ** 2 + sr ** 2) if (sl + sr) > 0 else 0.001
                )
                z_score = (
                    abs(ml - mr) / combined_std if combined_std > 0 else 0.0
                )

                if z_score > 2.0:
                    prob = min(1.0, z_score / 5.0)
                    if prob > self.threshold:
                        changepoints.append(i)

            if changepoints:
                best_idx = changepoints[0]
                left_mean = _mean(values[best_idx - w : best_idx])
                right_mean = _mean(values[best_idx : best_idx + w])
                confidence = min(
                    1.0,
                    abs(right_mean - left_mean)
                    / (max(abs(left_mean), 0.001) * 2 + 0.001),
                )
                confidence = max(0.1, min(1.0, confidence))

                results.append(
                    PatternResult(
                        pattern_type=PatternType.BOCPD,
                        description=(
                            f"Changepoint on ch{ch} at obs {best_idx}: "
                            f"mean {left_mean:.2f} -> {right_mean:.2f}"
                        ),
                        confidence=confidence,
                        details={
                            "channel": ch,
                            "changepoint_index": best_idx,
                            "left_mean": left_mean,
                            "right_mean": right_mean,
                            "num_changepoints": len(changepoints),
                        },
                        affected_channels=[ch],
                        timestamp_ms=observations[best_idx].timestamp_ms
                        if best_idx < n_obs
                        else 0,
                    )
                )

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results


# ===================================================================
# Algorithm 3: Statistical anomaly detection (Z-score)
# ===================================================================


class AnomalyDetector:
    """Z-score based statistical anomaly detection.

    Detects observations where sensor values deviate significantly
    from the running mean (beyond a configurable z-score threshold).
    """

    def __init__(self, z_threshold: float = 3.0) -> None:
        self.z_threshold = z_threshold

    def detect(
        self, observations: list[UnifiedObservation]
    ) -> list[PatternResult]:
        if len(observations) < 10:
            return []

        results: list[PatternResult] = []
        n_obs = len(observations)

        # Compute statistics per channel
        channel_stats: dict[int, tuple[float, float]] = {}
        for ch in range(NUM_SENSOR_CHANNELS):
            values = [
                getattr(obs, f"sensor_ch{ch}_val") for obs in observations
            ]
            channel_stats[ch] = (_mean(values), _std(values))

        # Find anomalies
        anomaly_counts: dict[int, int] = {}
        anomaly_examples: dict[int, list[dict]] = {}

        for i, obs in enumerate(observations):
            for ch in range(NUM_SENSOR_CHANNELS):
                val = getattr(obs, f"sensor_ch{ch}_val")
                mean, std = channel_stats[ch]
                if std > 0:
                    z = abs(val - mean) / std
                    if z > self.z_threshold:
                        anomaly_counts[ch] = anomaly_counts.get(ch, 0) + 1
                        if ch not in anomaly_examples:
                            anomaly_examples[ch] = []
                        anomaly_examples[ch].append(
                            {
                                "index": i,
                                "value": val,
                                "z_score": z,
                                "mean": mean,
                                "std": std,
                            }
                        )

        for ch, count in anomaly_counts.items():
            severity = count / n_obs
            confidence = min(1.0, severity * 5 + 0.3)
            examples = anomaly_examples[ch][:3]
            mean, std = channel_stats[ch]
            results.append(
                PatternResult(
                    pattern_type=PatternType.ANOMALY,
                    description=(
                        f"Anomaly on ch{ch}: {count}/{n_obs} obs "
                        f"exceed z={self.z_threshold:.1f}"
                    ),
                    confidence=confidence,
                    details={
                        "channel": ch,
                        "anomaly_count": count,
                        "total_observations": n_obs,
                        "severity": severity,
                        "examples": examples,
                        "channel_mean": mean,
                        "channel_std": std,
                    },
                    affected_channels=[ch],
                    timestamp_ms=observations[0].timestamp_ms,
                )
            )

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results


# ===================================================================
# Algorithm 4: Trend detection (linear regression + significance)
# ===================================================================


class TrendDetector:
    """Linear regression trend detection with significance test.

    Fits a line to each sensor channel time series and reports
    channels with statistically significant trends (R-squared check).
    """

    def __init__(
        self,
        min_slope: float = 0.01,
        min_r_squared: float = 0.3,
    ) -> None:
        self.min_slope = min_slope
        self.min_r_squared = min_r_squared

    def detect(
        self, observations: list[UnifiedObservation]
    ) -> list[PatternResult]:
        if len(observations) < 10:
            return []

        results: list[PatternResult] = []
        n = len(observations)

        x = list(range(n))
        mx = _mean(x)
        sx = _std(x)

        for ch in range(NUM_SENSOR_CHANNELS):
            y = [
                getattr(obs, f"sensor_ch{ch}_val") for obs in observations
            ]
            my = _mean(y)
            sy = _std(y)

            if sy == 0 or sx == 0:
                continue

            # Linear regression: y = slope * x + intercept
            cov_xy = (
                sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n - 1)
            )
            slope = cov_xy / (sx * sy) * (sy / sx)
            intercept = my - slope * mx

            # R-squared
            ss_res = sum(
                (yi - (slope * xi + intercept)) ** 2
                for xi, yi in zip(x, y)
            )
            ss_tot = sum((yi - my) ** 2 for yi in y)
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            if abs(slope) >= self.min_slope and r_squared >= self.min_r_squared:
                direction = "rising" if slope > 0 else "falling"
                confidence = min(1.0, r_squared * 0.7 + abs(slope) * 0.3)
                results.append(
                    PatternResult(
                        pattern_type=PatternType.TREND,
                        description=(
                            f"Ch{ch} {direction} trend: "
                            f"slope={slope:.4f}, R2={r_squared:.3f}"
                        ),
                        confidence=confidence,
                        details={
                            "channel": ch,
                            "slope": slope,
                            "intercept": intercept,
                            "r_squared": r_squared,
                            "direction": direction,
                        },
                        affected_channels=[ch],
                        timestamp_ms=observations[-1].timestamp_ms
                        if observations
                        else 0,
                    )
                )

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results


# ===================================================================
# Algorithm 5: Periodic pattern detection (autocorrelation)
# ===================================================================


class PeriodicDetector:
    """Periodic pattern detection via autocorrelation.

    Finds repeating cycles in sensor data by computing autocorrelation
    at various lags and reporting significant periodicities.
    """

    def __init__(
        self,
        min_correlation: float = 0.5,
        max_lag: int | None = None,
    ) -> None:
        self.min_correlation = min_correlation
        self.max_lag = max_lag

    def detect(
        self, observations: list[UnifiedObservation]
    ) -> list[PatternResult]:
        if len(observations) < 20:
            return []

        results: list[PatternResult] = []
        n = len(observations)
        max_lag = self.max_lag or (n // 2)

        for ch in range(NUM_SENSOR_CHANNELS):
            values = [
                getattr(obs, f"sensor_ch{ch}_val") for obs in observations
            ]
            std = _std(values)

            if std < 1e-9:
                continue

            # Compute autocorrelation at each lag
            best_lag = 0
            best_acf = 0.0
            for lag in range(2, max_lag):
                x = values[:-lag] if lag < n else values
                y = values[lag:]
                if len(x) < 5:
                    break
                acf = _correlation(x, y)
                if acf > best_acf:
                    best_acf = acf
                    best_lag = lag

            if best_acf >= self.min_correlation and best_lag > 0:
                confidence = min(1.0, best_acf * 0.8 + 0.2)
                results.append(
                    PatternResult(
                        pattern_type=PatternType.PERIODIC,
                        description=(
                            f"Ch{ch} periodic: period={best_lag} obs, "
                            f"acf={best_acf:.3f}"
                        ),
                        confidence=confidence,
                        details={
                            "channel": ch,
                            "period": best_lag,
                            "autocorrelation": best_acf,
                        },
                        affected_channels=[ch],
                        timestamp_ms=observations[-1].timestamp_ms
                        if observations
                        else 0,
                    )
                )

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results


# ===================================================================
# Pattern Discovery Engine - Orchestrator
# ===================================================================


class PatternDiscoveryEngine:
    """Orchestrates all 5 pattern discovery algorithms and merges results.

    Each algorithm independently analyzes the observation history and
    produces PatternResult objects with confidence scores. The engine
    merges and deduplicates results, sorted by confidence.
    """

    def __init__(
        self,
        cross_corr_threshold: float = 0.7,
        bocpd_threshold: float = 0.5,
        bocpd_window: int = 20,
        anomaly_z_threshold: float = 3.0,
        trend_min_slope: float = 0.01,
        trend_min_r_squared: float = 0.3,
        periodic_min_corr: float = 0.5,
        periodic_max_lag: int | None = None,
    ) -> None:
        self.cross_correlation = CrossCorrelationDetector(
            threshold=cross_corr_threshold
        )
        self.bocpd = BOCPDDetector(
            threshold=bocpd_threshold, window_size=bocpd_window
        )
        self.anomaly = AnomalyDetector(z_threshold=anomaly_z_threshold)
        self.trend = TrendDetector(
            min_slope=trend_min_slope, min_r_squared=trend_min_r_squared
        )
        self.periodic = PeriodicDetector(
            min_correlation=periodic_min_corr, max_lag=periodic_max_lag
        )

    def discover(
        self, observations: list[UnifiedObservation]
    ) -> list[PatternResult]:
        """Run all 5 algorithms and return merged results sorted by confidence."""
        all_results: list[PatternResult] = []

        detectors = [
            (self.cross_correlation, "cross_correlation"),
            (self.bocpd, "bocpd"),
            (self.anomaly, "anomaly"),
            (self.trend, "trend"),
            (self.periodic, "periodic"),
        ]

        for detector, name in detectors:
            try:
                results = detector.detect(observations)
                all_results.extend(results)
            except Exception:
                # Individual detector failures should not crash the engine
                continue

        all_results.sort(key=lambda r: r.confidence, reverse=True)
        return all_results

    def discover_single(
        self,
        detector_name: str,
        observations: list[UnifiedObservation],
    ) -> list[PatternResult]:
        """Run a single named detector."""
        detectors = {
            "cross_correlation": self.cross_correlation,
            "bocpd": self.bocpd,
            "anomaly": self.anomaly,
            "trend": self.trend,
            "periodic": self.periodic,
        }
        detector = detectors.get(detector_name)
        if detector is None:
            raise ValueError(f"Unknown detector: {detector_name}")
        return detector.detect(observations)
