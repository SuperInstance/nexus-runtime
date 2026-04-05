"""Emergency Detectors — Specific anomaly detection modules.

Each detector focuses on a single category of emergency:
    - SensorFailureDetector:   Sensor hardware/data issues
    - TrustCollapseDetector:   Rapid trust degradation
    - CommunicationLossDetector:  Loss of heartbeat/telemetry
    - SafetyViolationDetector: E-Stop, safety state machine faults
    - MissionTimeoutDetector:  Mission duration overruns
"""

from __future__ import annotations

import time
from typing import Any

from .protocol import Incident, generate_incident_id


class SensorFailureDetector:
    """Detect sensor failures: no readings, stale data, out-of-range values.

    Checks sensor_status dict for:
    - Missing or None readings (sensor offline)
    - Stale data (timestamp older than threshold)
    - Out-of-range values (outside defined min/max)
    - Quality flags indicating failure
    """

    def detect(self, sensor_readings: dict, config: dict) -> list[Incident]:
        """Detect sensor failure incidents.

        Args:
            sensor_readings: Dict with sensor data. Expected keys per sensor:
                - sensors: list of dicts, each with:
                    - id: sensor identifier
                    - value: current reading (None if offline)
                    - timestamp: unix timestamp of reading
                    - min_value: minimum valid value (optional)
                    - max_value: maximum valid value (optional)
                    - quality: quality flag (0.0-1.0, optional)
            config: Thresholds dict with at least:
                - sensor_stale_seconds: max age before stale

        Returns:
            List of Incident objects for any detected failures.
        """
        incidents: list[Incident] = []
        now = time.time()
        stale_threshold = config.get("sensor_stale_seconds", 30.0)

        sensors = sensor_readings.get("sensors", [])
        if not sensors:
            # No sensor data at all — could be a systemic issue
            # But we don't raise YELLOW here as it might be intentional
            return incidents

        for sensor in sensors:
            sensor_id = sensor.get("id", "unknown")
            value = sensor.get("value")
            ts = sensor.get("timestamp", 0)
            min_val = sensor.get("min_value")
            max_val = sensor.get("max_value")
            quality = sensor.get("quality", 1.0)

            # Check for offline sensor (None value)
            if value is None:
                incidents.append(Incident(
                    id=generate_incident_id(),
                    level="ORANGE",
                    category="SENSOR",
                    description=f"Sensor '{sensor_id}' is offline (no reading)",
                    timestamp=now,
                ))
                continue

            # Check for stale data
            if (now - ts) > stale_threshold:
                incidents.append(Incident(
                    id=generate_incident_id(),
                    level="YELLOW",
                    category="SENSOR",
                    description=(
                        f"Sensor '{sensor_id}' data stale: "
                        f"{now - ts:.1f}s old (threshold: {stale_threshold}s)"
                    ),
                    timestamp=now,
                ))

            # Check for out-of-range values
            if min_val is not None and value < min_val:
                incidents.append(Incident(
                    id=generate_incident_id(),
                    level="YELLOW",
                    category="SENSOR",
                    description=(
                        f"Sensor '{sensor_id}' below range: "
                        f"{value} < {min_val}"
                    ),
                    timestamp=now,
                ))
            elif max_val is not None and value > max_val:
                incidents.append(Incident(
                    id=generate_incident_id(),
                    level="YELLOW",
                    category="SENSOR",
                    description=(
                        f"Sensor '{sensor_id}' above range: "
                        f"{value} > {max_val}"
                    ),
                    timestamp=now,
                ))

            # Check for low quality
            if quality < 0.3:
                incidents.append(Incident(
                    id=generate_incident_id(),
                    level="YELLOW",
                    category="SENSOR",
                    description=(
                        f"Sensor '{sensor_id}' quality degraded: {quality:.2f}"
                    ),
                    timestamp=now,
                ))

        return incidents


class TrustCollapseDetector:
    """Detect trust collapse: rapid trust loss, multiple subsystem degradation.

    Examines current trust scores and recent history to identify:
    - Individual subsystem trust below thresholds
    - Multiple subsystems simultaneously degraded
    - Rapid trust loss rate
    """

    def detect(self, trust_scores: dict, trust_history: list[dict]) -> list[Incident]:
        """Detect trust collapse incidents.

        Args:
            trust_scores: Dict mapping subsystem names to current trust scores (0.0-1.0).
            trust_history: List of historical trust snapshots (dicts of subsystem→score).

        Returns:
            List of Incident objects for any detected trust issues.
        """
        incidents: list[Incident] = []
        now = time.time()

        if not trust_scores:
            return incidents

        # Default thresholds
        trust_yellow = 0.60
        trust_orange = 0.35
        trust_red = 0.15

        degraded_subsystems: list[str] = []
        critical_subsystems: list[str] = []

        for subsystem, score in trust_scores.items():
            if not isinstance(score, (int, float)):
                continue

            if score <= trust_red:
                incidents.append(Incident(
                    id=generate_incident_id(),
                    level="RED",
                    category="TRUST",
                    description=(
                        f"Trust collapse on '{subsystem}': "
                        f"score {score:.3f} <= {trust_red}"
                    ),
                    timestamp=now,
                ))
                critical_subsystems.append(subsystem)
            elif score <= trust_orange:
                incidents.append(Incident(
                    id=generate_incident_id(),
                    level="ORANGE",
                    category="TRUST",
                    description=(
                        f"Critical trust degradation on '{subsystem}': "
                        f"score {score:.3f} <= {trust_orange}"
                    ),
                    timestamp=now,
                ))
                critical_subsystems.append(subsystem)
            elif score <= trust_yellow:
                incidents.append(Incident(
                    id=generate_incident_id(),
                    level="YELLOW",
                    category="TRUST",
                    description=(
                        f"Trust degradation on '{subsystem}': "
                        f"score {score:.3f} <= {trust_yellow}"
                    ),
                    timestamp=now,
                ))
                degraded_subsystems.append(subsystem)

        # Multiple degraded subsystems → bump to ORANGE
        if len(degraded_subsystems) >= 3 and not critical_subsystems:
            incidents.append(Incident(
                id=generate_incident_id(),
                level="ORANGE",
                category="TRUST",
                description=(
                    f"Multiple subsystems degraded: "
                    f"{', '.join(degraded_subsystems)}"
                ),
                timestamp=now,
            ))

        return incidents


class CommunicationLossDetector:
    """Detect communication loss: no heartbeat, no telemetry, no commands.

    Checks the time elapsed since the last successful communication
    with the vessel or fleet coordinator.
    """

    def detect(self, last_comm_time: float, config: dict) -> list[Incident]:
        """Detect communication loss incidents.

        Args:
            last_comm_time: Unix timestamp of last successful communication.
            config: Thresholds dict with:
                - comm_timeout_seconds: time before ORANGE (default 120)
                - comm_dead_seconds: time before RED (default 300)

        Returns:
            List of Incident objects for any detected communication issues.
        """
        incidents: list[Incident] = []
        now = time.time()
        elapsed = now - last_comm_time

        comm_timeout = config.get("comm_timeout_seconds", 120.0)
        comm_dead = config.get("comm_dead_seconds", 300.0)

        if elapsed >= comm_dead:
            incidents.append(Incident(
                id=generate_incident_id(),
                level="RED",
                category="COMMUNICATION",
                description=(
                    f"Communication dead: no contact for {elapsed:.1f}s "
                    f"(threshold: {comm_dead}s)"
                ),
                timestamp=now,
            ))
        elif elapsed >= comm_timeout:
            incidents.append(Incident(
                id=generate_incident_id(),
                level="ORANGE",
                category="COMMUNICATION",
                description=(
                    f"Communication timeout: no contact for {elapsed:.1f}s "
                    f"(threshold: {comm_timeout}s)"
                ),
                timestamp=now,
            ))

        return incidents


class SafetyViolationDetector:
    """Detect safety violations: E-Stop triggered, safety state machine entered FAULT.

    Monitors the vessel's safety state machine for transitions into
    error or fault states.
    """

    def detect(self, safety_state: dict) -> list[Incident]:
        """Detect safety violation incidents.

        Args:
            safety_state: Dict with safety state machine info:
                - state: current state name (e.g., "NOMINAL", "WARNING", "FAULT", "E_STOP")
                - e_stop: bool, whether E-Stop is active
                - watchdog_triggered: bool, whether watchdog has fired
                - violation_type: string describing the violation (optional)

        Returns:
            List of Incident objects for any detected safety violations.
        """
        incidents: list[Incident] = []
        now = time.time()

        if not safety_state:
            return incidents

        state = safety_state.get("state", "NOMINAL")

        # E-Stop is always RED
        if safety_state.get("e_stop", False):
            incidents.append(Incident(
                id=generate_incident_id(),
                level="RED",
                category="SAFETY",
                description="E-Stop triggered: all autonomous operations halted",
                timestamp=now,
            ))

        # FAULT state is RED
        if state in ("FAULT", "E_STOP", "CRITICAL"):
            # Avoid double-counting if e_stop already triggered for E_STOP state
            if not safety_state.get("e_stop", False):
                incidents.append(Incident(
                    id=generate_incident_id(),
                    level="RED",
                    category="SAFETY",
                    description=f"Safety state machine entered {state}",
                    timestamp=now,
                ))

        # WARNING state is ORANGE
        if state == "WARNING":
            violation_type = safety_state.get("violation_type", "unknown")
            incidents.append(Incident(
                id=generate_incident_id(),
                level="ORANGE",
                category="SAFETY",
                description=(
                    f"Safety WARNING: {violation_type}"
                ),
                timestamp=now,
            ))

        # Watchdog triggered → RED (system unresponsive)
        if safety_state.get("watchdog_triggered", False):
            incidents.append(Incident(
                id=generate_incident_id(),
                level="RED",
                category="SAFETY",
                description="Watchdog triggered: system unresponsive",
                timestamp=now,
            ))

        return incidents


class MissionTimeoutDetector:
    """Detect mission timeout: mission exceeding expected duration.

    Compares mission elapsed time against expected duration to detect
    missions that are running significantly over their allocated time.
    """

    def detect(self, mission_start: float, expected_duration: float) -> list[Incident]:
        """Detect mission timeout incidents.

        Args:
            mission_start: Unix timestamp when the mission started.
            expected_duration: Expected mission duration in seconds.
                Use float('inf') for no expected duration.

        Returns:
            List of Incident objects for any detected timeouts.
        """
        incidents: list[Incident] = []
        now = time.time()

        if expected_duration == float("inf") or expected_duration <= 0:
            return incidents

        elapsed = now - mission_start

        # Allow 1.5x expected duration before ORANGE
        overrun_fraction = 1.5
        if elapsed > expected_duration * overrun_fraction:
            overrun_pct = ((elapsed - expected_duration) / expected_duration) * 100
            incidents.append(Incident(
                id=generate_incident_id(),
                level="ORANGE",
                category="MISSION",
                description=(
                    f"Mission timeout: {elapsed:.0f}s elapsed "
                    f"vs {expected_duration:.0f}s expected "
                    f"({overrun_pct:.0f}% overrun)"
                ),
                timestamp=now,
            ))

        return incidents
