"""NEXUS git-agent bridge — Telemetry ingestion.

Batches sensor/actuator readings and commits them to git at intervals.
Data is stored as structured JSON in .agent/telemetry/ directory.

Batching strategy:
  - Readings accumulate in memory
  - flush() commits the batch and creates a git commit
  - Auto-flush after configurable batch size or time window
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

try:
    from git import Repo
except ImportError:
    Repo = Any  # type: ignore[assignment,misc]


# ── Constants ──────────────────────────────────────────────────────

DEFAULT_BATCH_SIZE = 100  # readings per batch
DEFAULT_WINDOW_SECONDS = 300  # 5-minute window
TELEMETRY_DIR = ".agent/telemetry"


# ── Data types ─────────────────────────────────────────────────────

@dataclass
class SensorReading:
    """A single sensor or actuator reading."""
    sensor_id: int
    value: float
    timestamp: float
    sensor_type: str = "analog"  # analog, digital, pwm, i2c, spi, can
    unit: str = ""


@dataclass
class TelemetryBatch:
    """A batch of sensor readings ready for commit."""
    readings: list[dict] = field(default_factory=list)
    batch_start: float = 0.0
    batch_end: float = 0.0
    vessel_id: str = ""
    reading_count: int = 0

    def to_dict(self) -> dict:
        return {
            "vessel_id": self.vessel_id,
            "batch_start": self.batch_start,
            "batch_end": self.batch_end,
            "batch_start_iso": datetime.fromtimestamp(
                self.batch_start, tz=timezone.utc
            ).isoformat() if self.batch_start else "",
            "batch_end_iso": datetime.fromtimestamp(
                self.batch_end, tz=timezone.utc
            ).isoformat() if self.batch_end else "",
            "reading_count": self.reading_count,
            "readings": self.readings,
        }


@dataclass
class TelemetryResult:
    """Result of a telemetry ingestion operation."""
    committed: bool
    commit_hash: str = ""
    readings_count: int = 0
    batch_file: str = ""


# ── Telemetry Ingester ────────────────────────────────────────────

class TelemetryIngester:
    """Batches sensor data and commits to git at intervals.

    Usage:
        ingester = TelemetryIngester(vessel_id="vessel-001")
        ingester.add_reading(sensor_id=1, value=23.5, timestamp=time.time())
        ingester.add_reading(sensor_id=2, value=1013.25, timestamp=time.time())
        result = ingester.flush(repo)
    """

    def __init__(
        self,
        vessel_id: str = "unknown",
        batch_size: int = DEFAULT_BATCH_SIZE,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
    ) -> None:
        self.vessel_id = vessel_id
        self.batch_size = batch_size
        self.window_seconds = window_seconds
        self._readings: list[dict] = []
        self._batch_start: float = 0.0
        self._last_flush: float = 0.0

    def add_reading(
        self,
        sensor_id: int,
        value: float,
        timestamp: float,
        sensor_type: str = "analog",
        unit: str = "",
    ) -> bool:
        """Add a sensor reading to the current batch.

        Args:
            sensor_id: Numeric sensor identifier.
            value: Sensor value (float).
            timestamp: Unix timestamp of the reading.
            sensor_type: Type of sensor (analog, digital, pwm, etc.).
            unit: Optional unit string (e.g. "C", "hPa", "m/s").

        Returns:
            True if batch is full and should be flushed.
        """
        if not self._batch_start:
            self._batch_start = timestamp

        reading = {
            "sensor_id": sensor_id,
            "value": value,
            "timestamp": timestamp,
            "timestamp_iso": datetime.fromtimestamp(
                timestamp, tz=timezone.utc
            ).isoformat(),
            "sensor_type": sensor_type,
            "unit": unit,
        }
        self._readings.append(reading)
        return len(self._readings) >= self.batch_size

    def should_flush(self, current_time: float) -> bool:
        """Check if batch should be flushed based on time window.

        Args:
            current_time: Current Unix timestamp.

        Returns:
            True if window has elapsed and there are readings.
        """
        if not self._readings:
            return False
        if not self._batch_start:
            return False
        elapsed = current_time - self._batch_start
        return elapsed >= self.window_seconds

    def flush(self, repo: Any) -> TelemetryResult:
        """Commit accumulated batch to git.

        Creates a JSON file in .agent/telemetry/ with batch data
        and creates a git commit.

        Args:
            repo: gitpython Repo object (or path string).

        Returns:
            TelemetryResult with commit hash and stats.
        """
        if isinstance(repo, str):
            repo = Repo(repo)

        if not self._readings:
            return TelemetryResult(
                committed=False, readings_count=0
            )

        now = time.time()
        batch = TelemetryBatch(
            readings=self._readings,
            batch_start=self._batch_start,
            batch_end=now,
            vessel_id=self.vessel_id,
            reading_count=len(self._readings),
        )

        # Create telemetry directory
        abs_dir = os.path.join(repo.working_dir, TELEMETRY_DIR)
        os.makedirs(abs_dir, exist_ok=True)

        # Write batch file
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%z")
        filename = f"{timestamp}_batch_{len(self._readings)}.json"
        filepath = os.path.join(abs_dir, filename)
        with open(filepath, "w") as f:
            json.dump(batch.to_dict(), f, indent=2)

        # Stage and commit
        repo.index.add([filepath])
        commit_msg = (
            f"TELEMETRY: {len(self._readings)} readings | "
            f"window={now - self._batch_start:.1f}s | "
            f"vessel={self.vessel_id}"
        )
        commit = repo.index.commit(commit_msg)

        result = TelemetryResult(
            committed=True,
            commit_hash=commit.hexsha,
            readings_count=len(self._readings),
            batch_file=filename,
        )

        # Reset batch
        self._readings = []
        self._batch_start = 0.0
        self._last_flush = now

        return result

    @property
    def pending_count(self) -> int:
        """Number of readings waiting to be flushed."""
        return len(self._readings)

    @property
    def batch_age_seconds(self) -> float:
        """Age of current batch in seconds (0 if empty)."""
        if not self._batch_start:
            return 0.0
        return time.time() - self._batch_start
