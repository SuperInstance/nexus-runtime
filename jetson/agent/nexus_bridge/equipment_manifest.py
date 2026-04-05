"""NEXUS git-agent bridge — Equipment Manifest.

Declares NEXUS HAL sensor/actuator capabilities in git-agent format.

The EquipmentManifest provides capability discovery and attestation,
enabling vessels to advertise their hardware to the fleet.

JSON schema:
    {
        "vessel_id": "vessel-001",
        "manifest_version": "1.0",
        "generated_at": "2025-04-05T12:00:00Z",
        "sensors": [
            {
                "id": 1,
                "type": "analog",
                "pin": 34,
                "unit": "V",
                "range": [0.0, 3.3],
                "description": "Battery voltage monitor"
            }
        ],
        "actuators": [
            {
                "id": 1,
                "type": "pwm",
                "pin": 25,
                "range": [0, 255],
                "description": "Main thruster ESC"
            }
        ],
        "capabilities": ["navigation", "payload", "communication"],
        "compute": {
            "platform": "Jetson Nano",
            "vm_max_cycles": 1000,
            "vm_max_stack": 16
        }
    }
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ── Constants ──────────────────────────────────────────────────────

MANIFEST_VERSION = "1.0"
VALID_SENSOR_TYPES = {
    "analog", "digital", "pwm", "i2c", "spi", "can", "uart", "gps",
    "imu", "sonar", "lidar", "camera", "temperature", "pressure",
    "humidity", "current", "voltage",
}
VALID_ACTUATOR_TYPES = {
    "pwm", "digital", "servo", "stepper", "can", "dac", "relay",
    "motor", "thruster", "rudder", "winch",
}


# ── Data types ─────────────────────────────────────────────────────

@dataclass
class SensorEntry:
    """A sensor capability entry."""
    id: int
    type: str
    pin: int
    unit: str = ""
    range_min: float = 0.0
    range_max: float = 0.0
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "pin": self.pin,
            "unit": self.unit,
            "range": [self.range_min, self.range_max],
            "description": self.description,
        }


@dataclass
class ActuatorEntry:
    """An actuator capability entry."""
    id: int
    type: str
    pin: int
    range_min: float = 0.0
    range_max: float = 0.0
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "pin": self.pin,
            "range": [self.range_min, self.range_max],
            "description": self.description,
        }


# ── Equipment Manifest ────────────────────────────────────────────

class EquipmentManifest:
    """Declares NEXUS HAL sensor/actuator capabilities in git-agent format.

    Generates standardized capability manifests from NEXUS node_role_config
    and provides validation/serialization utilities.
    """

    def from_hal_config(self, node_config: dict[str, Any]) -> dict:
        """Generate EquipmentManifest from NEXUS node_role_config.

        The node_role_config follows the JSON schema in schemas/node_role_config.json.

        Args:
            node_config: NEXUS node role configuration dict. Expected keys:
                - vessel_id: Vessel identifier
                - node_role: Role name (e.g. "usv-navigation")
                - sensors: List of sensor configs
                - actuators: List of actuator configs

        Returns:
            Complete manifest dict ready for serialization.
        """
        vessel_id = node_config.get("vessel_id", "unknown")
        sensors = node_config.get("sensors", [])
        actuators = node_config.get("actuators", [])

        # Parse sensor entries
        sensor_entries = []
        for s in sensors:
            entry = SensorEntry(
                id=s.get("id", 0),
                type=s.get("type", "analog"),
                pin=s.get("pin", 0),
                unit=s.get("unit", ""),
                range_min=s.get("range", [0.0, 0.0])[0] if isinstance(s.get("range"), list) else 0.0,
                range_max=s.get("range", [0.0, 0.0])[1] if isinstance(s.get("range"), list) else 0.0,
                description=s.get("description", ""),
            )
            sensor_entries.append(entry.to_dict())

        # Parse actuator entries
        actuator_entries = []
        for a in actuators:
            entry = ActuatorEntry(
                id=a.get("id", 0),
                type=a.get("type", "pwm"),
                pin=a.get("pin", 0),
                range_min=a.get("range", [0.0, 0.0])[0] if isinstance(a.get("range"), list) else 0.0,
                range_max=a.get("range", [0.0, 0.0])[1] if isinstance(a.get("range"), list) else 0.0,
                description=a.get("description", ""),
            )
            actuator_entries.append(entry.to_dict())

        # Infer capabilities from sensors/actuators
        capabilities = self._infer_capabilities(node_config)

        manifest = {
            "vessel_id": vessel_id,
            "manifest_version": MANIFEST_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "node_role": node_config.get("node_role", "unknown"),
            "sensors": sensor_entries,
            "actuators": actuator_entries,
            "capabilities": capabilities,
            "compute": {
                "platform": node_config.get("platform", "Jetson Nano"),
                "vm_max_cycles": node_config.get("vm_max_cycles", 1000),
                "vm_max_stack": node_config.get("vm_max_stack", 16),
            },
        }

        return manifest

    def to_json(self, manifest: dict) -> str:
        """Serialize manifest to JSON.

        Args:
            manifest: Manifest dict from from_hal_config().

        Returns:
            Pretty-printed JSON string.
        """
        return json.dumps(manifest, indent=2, default=str)

    def validate(self, manifest: dict) -> tuple[bool, list[str]]:
        """Validate manifest structure.

        Checks:
          - Required fields present
          - Valid manifest version
          - Valid sensor/actuator types
          - Valid pin ranges
          - Valid capability names

        Args:
            manifest: Manifest dict.

        Returns:
            (is_valid, list_of_errors).
        """
        errors: list[str] = []

        # Required fields
        required = ["vessel_id", "manifest_version", "generated_at", "sensors", "actuators"]
        for field_name in required:
            if field_name not in manifest:
                errors.append(f"Missing required field: {field_name}")

        # Version check
        version = manifest.get("manifest_version", "")
        if version != MANIFEST_VERSION:
            errors.append(
                f"Unsupported manifest version: {version} "
                f"(expected {MANIFEST_VERSION})"
            )

        # Validate sensors
        sensors = manifest.get("sensors", [])
        for i, s in enumerate(sensors):
            if not isinstance(s, dict):
                errors.append(f"Sensor[{i}]: not a dict")
                continue
            if "type" in s and s["type"] not in VALID_SENSOR_TYPES:
                errors.append(f"Sensor[{i}]: invalid type '{s['type']}'")
            if "pin" in s and not isinstance(s["pin"], int):
                errors.append(f"Sensor[{i}]: pin must be an integer")

        # Validate actuators
        actuators = manifest.get("actuators", [])
        for i, a in enumerate(actuators):
            if not isinstance(a, dict):
                errors.append(f"Actuator[{i}]: not a dict")
                continue
            if "type" in a and a["type"] not in VALID_ACTUATOR_TYPES:
                errors.append(f"Actuator[{i}]: invalid type '{a['type']}'")
            if "pin" in a and not isinstance(a["pin"], int):
                errors.append(f"Actuator[{i}]: pin must be an integer")

        # Validate capabilities
        caps = manifest.get("capabilities", [])
        if not isinstance(caps, list):
            errors.append("capabilities must be a list")
        else:
            valid_caps = {
                "navigation", "steering", "engine", "payload",
                "communication", "safety", "surveillance", "fishing",
                "research", "rescue", "patrol", "transport",
            }
            for cap in caps:
                if cap not in valid_caps:
                    errors.append(f"Unknown capability: {cap}")

        return (len(errors) == 0, errors)

    def _infer_capabilities(self, node_config: dict) -> list[str]:
        """Infer vessel capabilities from sensor/actuator configuration."""
        capabilities: list[str] = []
        sensors = node_config.get("sensors", [])
        actuators = node_config.get("actuators", [])
        node_role = node_config.get("node_role", "")

        # Infer from sensors
        sensor_types = {s.get("type", "") for s in sensors}
        if "gps" in sensor_types or "imu" in sensor_types:
            capabilities.append("navigation")
        if "sonar" in sensor_types or "lidar" in sensor_types:
            capabilities.append("navigation")
        if "camera" in sensor_types:
            capabilities.append("surveillance")
        if "current" in sensor_types or "voltage" in sensor_types:
            capabilities.append("safety")

        # Infer from actuators
        actuator_types = {a.get("type", "") for a in actuators}
        if "thruster" in actuator_types or "motor" in actuator_types or "pwm" in actuator_types:
            capabilities.append("engine")
        if "rudder" in actuator_types or "servo" in actuator_types:
            capabilities.append("steering")
        if "winch" in actuator_types:
            capabilities.append("payload")

        # Infer from role
        role_lower = node_role.lower()
        if "comm" in role_lower or "radio" in role_lower:
            if "communication" not in capabilities:
                capabilities.append("communication")

        # Deduplicate while preserving order
        seen: set[str] = set()
        result: list[str] = []
        for cap in capabilities:
            if cap not in seen:
                seen.add(cap)
                result.append(cap)

        return result
