"""
NEXUS BLE GATT Profiles for Marine Sensor Data

Defines custom BLE services, characteristics, and descriptors for the
NEXUS distributed intelligence platform. All UUIDs use the NEXUS base
UUID pattern: 6E787873-XXXX-4000-8000-001122334455.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
import struct


# ---------------------------------------------------------------------------
# NEXUS UUID Constants
# ---------------------------------------------------------------------------

# Base UUID: 6E787873-XXXX-4000-8000-001122334455
# The "6E787873" bytes decode to "nx" + "x" + "s" (NEXUS ASCII shorthand)
NEXUS_BASE_UUID = "6E787873-{:04x}-4000-8000-001122334455"

# NEXUS Service UUIDs (16-bit short values mapped into base UUID)
NEXUS_SERVICE_UUID = NEXUS_BASE_UUID.format(0x0001)    # Marine Sensor Service
NEXUS_DEVICE_INFO_UUID = NEXUS_BASE_UUID.format(0x0002) # Device Info Service
NEXUS_COMMAND_UUID = NEXUS_BASE_UUID.format(0x0003)     # Command Service
NEXUS_TELEMETRY_UUID = NEXUS_BASE_UUID.format(0x0004)   # Telemetry Stream Service

# Standard Bluetooth UUIDs
BLE_UUID_BATTERY_SERVICE = "0000180f-0000-1000-8000-00805f9b34fb"
BLE_UUID_BATTERY_LEVEL = "00002a19-0000-1000-8000-00805f9b34fb"
BLE_UUID_DEVICE_NAME = "00002a00-0000-1000-8000-00805f9b34fb"
BLE_UUID_MANUFACTURER = "00002a29-0000-1000-8000-00805f9b34fb"


# ---------------------------------------------------------------------------
# Characteristic UUIDs for NEXUS Marine Sensor Service (0x0001)
# ---------------------------------------------------------------------------

class NEXUSCharacteristicUUID:
    """Characteristic UUIDs within the NEXUS Marine Sensor Service."""
    TEMPERATURE = NEXUS_BASE_UUID.format(0x0101)
    DEPTH = NEXUS_BASE_UUID.format(0x0102)
    PRESSURE = NEXUS_BASE_UUID.format(0x0103)
    CONDUCTIVITY = NEXUS_BASE_UUID.format(0x0104)
    PH = NEXUS_BASE_UUID.format(0x0105)
    DISSOLVED_OXYGEN = NEXUS_BASE_UUID.format(0x0106)
    TURBIDITY = NEXUS_BASE_UUID.format(0x0107)
    GPS_POSITION = NEXUS_BASE_UUID.format(0x0108)
    HEADING = NEXUS_BASE_UUID.format(0x0109)
    WATER_SPEED = NEXUS_BASE_UUID.format(0x010A)
    SALINITY = NEXUS_BASE_UUID.format(0x010B)
    SONAR_DISTANCE = NEXUS_BASE_UUID.format(0x010C)
    BATTERY_VOLTAGE = NEXUS_BASE_UUID.format(0x010D)
    LEAK_DETECTION = NEXUS_BASE_UUID.format(0x010E)
    HULL_INTEGRITY = NEXUS_BASE_UUID.format(0x010F)
    SENSOR_STATUS = NEXUS_BASE_UUID.format(0x0110)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CharProperty(IntEnum):
    """BLE characteristic property flags."""
    BROADCAST = 0x01
    READ = 0x02
    WRITE_NO_RESPONSE = 0x04
    WRITE = 0x08
    NOTIFY = 0x10
    INDICATE = 0x20
    AUTHENTICATED_WRITE = 0x40
    EXTENDED_PROPERTIES = 0x80


class CharFormat(IntEnum):
    """BLE characteristic format types (Bluetooth SIG)."""
    BOOLEAN = 0x01
    UINT8 = 0x04
    UINT16 = 0x06
    UINT32 = 0x08
    SINT8 = 0x0B
    SINT16 = 0x0D
    SINT32 = 0x0F
    FLOAT32 = 0x32
    SFLOAT = 0x34
    UTF8S = 0x25


class GATTError(IntEnum):
    """Standard BLE GATT error codes used by NEXUS."""
    SUCCESS = 0x00
    INVALID_HANDLE = 0x01
    READ_NOT_PERMITTED = 0x02
    WRITE_NOT_PERMITTED = 0x03
    INVALID_PDU = 0x04
    INSUFF_AUTHENTICATION = 0x05
    REQUEST_NOT_SUPPORTED = 0x06
    INVALID_OFFSET = 0x07
    INSUFF_AUTHORIZATION = 0x08
    PREPARE_QUEUE_FULL = 0x09
    ATTRIBUTE_NOT_FOUND = 0x0A
    INVALID_ATTRIBUTE_LENGTH = 0x0D
    INSUFFICIENT_RESOURCES = 0x11


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class CharacteristicDescriptor:
    """BLE characteristic descriptor (e.g., CCCD for notifications)."""
    uuid: str
    handle: int = 0
    value: bytes = b""
    permissions: int = CharProperty.READ | CharProperty.WRITE
    description: str = ""

    @property
    def short_uuid(self) -> str:
        """Extract the 16-bit short UUID if it's a standard BLE UUID."""
        parts = self.uuid.split("-")
        return parts[0][-4:] if len(parts) == 5 else self.uuid


@dataclass
class CharacteristicMetadata:
    """Metadata describing a BLE characteristic's format and units."""
    format: CharFormat = CharFormat.FLOAT32
    exponent: int = 0               # 10^exponent multiplier
    unit_uuid: str = "2700"         # Bluetooth SIG unit (0x2700 = "unitless")
    namespace: int = 1              # Bluetooth SIG namespace
    description: str = ""

    def encode_presentation_format(self) -> bytes:
        """Encode to BLE Characteristic Presentation Format descriptor value."""
        return struct.pack(
            "<BbH16s",
            self.format,
            self.exponent,
            int(self.unit_uuid, 16),
            bytes(16),  # namespace description (empty)
        )


@dataclass
class BLECharacteristic:
    """
    Represents a single BLE characteristic in the NEXUS GATT profile.

    Each characteristic has a UUID, properties, value encoding,
    optional metadata, and a list of descriptors.
    """
    uuid: str
    name: str
    properties: int = CharProperty.READ | CharProperty.NOTIFY
    handle: int = 0
    value_handle: int = 0
    max_data_length: int = 20       # Default ATT MTU - 3
    value: bytes = b""
    metadata: Optional[CharacteristicMetadata] = None
    descriptors: List[CharacteristicDescriptor] = field(default_factory=list)

    def can_read(self) -> bool:
        return bool(self.properties & CharProperty.READ)

    def can_write(self) -> bool:
        return bool(self.properties & (CharProperty.WRITE | CharProperty.WRITE_NO_RESPONSE))

    def can_notify(self) -> bool:
        return bool(self.properties & CharProperty.NOTIFY)

    def can_indicate(self) -> bool:
        return bool(self.properties & CharProperty.INDICATE)

    def add_cccd(self) -> CharacteristicDescriptor:
        """Add Client Characteristic Configuration Descriptor for notify/indicate."""
        cccd = CharacteristicDescriptor(
            uuid="00002902-0000-1000-8000-00805f9b34fb",
            permissions=CharProperty.READ | CharProperty.WRITE,
            description="CCCD",
        )
        self.descriptors.append(cccd)
        return cccd

    def set_value_float(self, value: float) -> bytes:
        """Encode a 32-bit float value."""
        self.value = struct.pack("<f", value)
        return self.value

    def set_value_uint16(self, value: int) -> bytes:
        """Encode a 16-bit unsigned integer value."""
        self.value = struct.pack("<H", value & 0xFFFF)
        return self.value

    def set_value_int16(self, value: int) -> bytes:
        """Encode a 16-bit signed integer value."""
        self.value = struct.pack("<h", value)
        return self.value

    def set_value_string(self, text: str, max_len: int = 20) -> bytes:
        """Encode a UTF-8 string value."""
        encoded = text.encode("utf-8")[:max_len]
        self.value = encoded
        return self.value

    def set_value_gps(self, lat: float, lon: float, alt: float = 0.0) -> bytes:
        """Encode GPS position as 3 x float32 (lat, lon, altitude)."""
        self.value = struct.pack("<fff", lat, lon, alt)
        return self.value

    def decode_float(self) -> Optional[float]:
        if len(self.value) < 4:
            return None
        return struct.unpack("<f", self.value[:4])[0]

    def decode_uint16(self) -> Optional[int]:
        if len(self.value) < 2:
            return None
        return struct.unpack("<H", self.value[:2])[0]

    def decode_string(self) -> str:
        return self.value.decode("utf-8", errors="replace")

    def decode_gps(self) -> Optional[Tuple[float, float, float]]:
        if len(self.value) < 12:
            return None
        return struct.unpack("<fff", self.value[:12])

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.uuid:
            errors.append("Characteristic UUID cannot be empty.")
        if not self.name:
            errors.append("Characteristic name cannot be empty.")
        if self.max_data_length < 1:
            errors.append("Max data length must be >= 1.")
        if self.can_read() and not self.can_notify() and not self.can_indicate():
            if not self.value and self.properties & CharProperty.READ:
                pass  # Empty value is OK for readable characteristics
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


@dataclass
class BLEService:
    """
    Represents a BLE GATT service containing characteristics.

    Services can be primary or secondary and contain multiple
    characteristics with optional descriptors.
    """
    uuid: str
    name: str
    is_primary: bool = True
    handle: int = 0
    characteristics: List[BLECharacteristic] = field(default_factory=list)

    def add_characteristic(self, char: BLECharacteristic) -> BLECharacteristic:
        self.characteristics.append(char)
        return char

    def get_characteristic(self, uuid: str) -> Optional[BLECharacteristic]:
        for c in self.characteristics:
            if c.uuid == uuid:
                return c
        return None

    def characteristic_count(self) -> int:
        return len(self.characteristics)

    def total_handles(self) -> int:
        """Calculate total GATT handles needed: 1 (service) + 2 per char + descriptors."""
        handles = 1  # Service declaration
        for c in self.characteristics:
            handles += 2  # Characteristic declaration + value
            handles += len(c.descriptors)
        return handles

    def build_gatt_table(self) -> List[dict]:
        """Build a GATT table representation for firmware integration."""
        table = []
        handle = self.handle or 1
        # Service declaration
        table.append({
            "handle": handle,
            "type": "service",
            "uuid": self.uuid,
            "name": self.name,
            "primary": self.is_primary,
        })
        handle += 1
        for char in self.characteristics:
            char.handle = handle
            handle += 1
            char.value_handle = handle
            table.append({
                "handle": char.handle,
                "type": "characteristic_declaration",
                "value_handle": char.value_handle,
                "uuid": char.uuid,
                "name": char.name,
                "properties": char.properties,
            })
            table.append({
                "handle": char.value_handle,
                "type": "characteristic_value",
                "uuid": char.uuid,
                "name": char.name,
                "value": char.value.hex() if char.value else "",
            })
            handle += 1
            for desc in char.descriptors:
                desc.handle = handle
                table.append({
                    "handle": handle,
                    "type": "descriptor",
                    "uuid": desc.uuid,
                    "name": desc.description,
                })
                handle += 1
        return table

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.uuid:
            errors.append("Service UUID cannot be empty.")
        if not self.name:
            errors.append("Service name cannot be empty.")
        uuids = [c.uuid for c in self.characteristics]
        if len(uuids) != len(set(uuids)):
            errors.append("Duplicate characteristic UUIDs detected.")
        for c in self.characteristics:
            errors.extend(c.validate())
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


# ---------------------------------------------------------------------------
# NEXUS Marine Sensor Service Factory
# ---------------------------------------------------------------------------

class NEXUSMarineService(BLEService):
    """
    Pre-configured NEXUS Marine Sensor BLE service.

    Provides factory methods for adding standard marine sensor
    characteristics (temperature, depth, GPS, etc.) with proper
    UUIDs, properties, and metadata.
    """

    def __init__(self):
        super().__init__(
            uuid=NEXUS_SERVICE_UUID,
            name="NEXUS Marine Sensor",
            is_primary=True,
        )

    def add_temperature_characteristic(self) -> BLECharacteristic:
        char = BLECharacteristic(
            uuid=NEXUSCharacteristicUUID.TEMPERATURE,
            name="Temperature",
            properties=CharProperty.READ | CharProperty.NOTIFY,
            metadata=CharacteristicMetadata(
                format=CharFormat.FLOAT32,
                unit_uuid="272F",  # Degree Celsius
                description="Water temperature in Celsius",
            ),
        )
        char.add_cccd()
        return self.add_characteristic(char)

    def add_depth_characteristic(self) -> BLECharacteristic:
        char = BLECharacteristic(
            uuid=NEXUSCharacteristicUUID.DEPTH,
            name="Depth",
            properties=CharProperty.READ | CharProperty.NOTIFY,
            metadata=CharacteristicMetadata(
                format=CharFormat.FLOAT32,
                unit_uuid="2701",  # Metre
                description="Water depth in meters",
            ),
        )
        char.add_cccd()
        return self.add_characteristic(char)

    def add_pressure_characteristic(self) -> BLECharacteristic:
        char = BLECharacteristic(
            uuid=NEXUSCharacteristicUUID.PRESSURE,
            name="Pressure",
            properties=CharProperty.READ | CharProperty.NOTIFY,
            metadata=CharacteristicMetadata(
                format=CharFormat.FLOAT32,
                unit_uuid="2714",  # Pascal
                description="Barometric pressure in Pascals",
            ),
        )
        char.add_cccd()
        return self.add_characteristic(char)

    def add_gps_characteristic(self) -> BLECharacteristic:
        char = BLECharacteristic(
            uuid=NEXUSCharacteristicUUID.GPS_POSITION,
            name="GPS Position",
            properties=CharProperty.READ | CharProperty.NOTIFY,
            max_data_length=12,  # 3 x float32
            metadata=CharacteristicMetadata(
                format=CharFormat.UTF8S,  # Structured data
                description="GPS position (lat, lon, alt) as 3x float32",
            ),
        )
        char.add_cccd()
        return self.add_characteristic(char)

    def add_conductivity_characteristic(self) -> BLECharacteristic:
        char = BLECharacteristic(
            uuid=NEXUSCharacteristicUUID.CONDUCTIVITY,
            name="Conductivity",
            properties=CharProperty.READ | CharProperty.NOTIFY,
            metadata=CharacteristicMetadata(
                format=CharFormat.FLOAT32,
                unit_uuid="2727",  # Siemens per metre
                description="Water conductivity in S/m",
            ),
        )
        char.add_cccd()
        return self.add_characteristic(char)

    def add_dissolved_oxygen_characteristic(self) -> BLECharacteristic:
        char = BLECharacteristic(
            uuid=NEXUSCharacteristicUUID.DISSOLVED_OXYGEN,
            name="Dissolved Oxygen",
            properties=CharProperty.READ | CharProperty.NOTIFY,
            metadata=CharacteristicMetadata(
                format=CharFormat.FLOAT32,
                unit_uuid="2794",  # Percent
                description="Dissolved oxygen saturation %",
            ),
        )
        char.add_cccd()
        return self.add_characteristic(char)

    def add_heading_characteristic(self) -> BLECharacteristic:
        char = BLECharacteristic(
            uuid=NEXUSCharacteristicUUID.HEADING,
            name="Heading",
            properties=CharProperty.READ | CharProperty.NOTIFY,
            metadata=CharacteristicMetadata(
                format=CharFormat.FLOAT32,
                unit_uuid="2763",  # Degree (angle)
                description="Compass heading in degrees",
            ),
        )
        char.add_cccd()
        return self.add_characteristic(char)

    def add_battery_voltage_characteristic(self) -> BLECharacteristic:
        char = BLECharacteristic(
            uuid=NEXUSCharacteristicUUID.BATTERY_VOLTAGE,
            name="Battery Voltage",
            properties=CharProperty.READ | CharProperty.NOTIFY,
            metadata=CharacteristicMetadata(
                format=CharFormat.FLOAT32,
                unit_uuid="271D",  # Volt
                description="Battery voltage in volts",
            ),
        )
        char.add_cccd()
        return self.add_characteristic(char)

    def add_leak_detection_characteristic(self) -> BLECharacteristic:
        char = BLECharacteristic(
            uuid=NEXUSCharacteristicUUID.LEAK_DETECTION,
            name="Leak Detection",
            properties=CharProperty.READ | CharProperty.NOTIFY | CharProperty.INDICATE,
            metadata=CharacteristicMetadata(
                format=CharFormat.UINT8,
                description="Leak detection status: 0=dry, 1=warning, 2=critical",
            ),
        )
        char.add_cccd()
        return self.add_characteristic(char)

    def add_sensor_status_characteristic(self) -> BLECharacteristic:
        char = BLECharacteristic(
            uuid=NEXUSCharacteristicUUID.SENSOR_STATUS,
            name="Sensor Status",
            properties=CharProperty.READ | CharProperty.NOTIFY,
            max_data_length=16,
            metadata=CharacteristicMetadata(
                format=CharFormat.UINT32,
                description="Bitfield of sensor health status flags",
            ),
        )
        char.add_cccd()
        return self.add_characteristic(char)

    def add_all_marine_characteristics(self) -> List[BLECharacteristic]:
        """Add all standard marine sensor characteristics and return them."""
        chars = []
        chars.append(self.add_temperature_characteristic())
        chars.append(self.add_depth_characteristic())
        chars.append(self.add_pressure_characteristic())
        chars.append(self.add_gps_characteristic())
        chars.append(self.add_conductivity_characteristic())
        chars.append(self.add_dissolved_oxygen_characteristic())
        chars.append(self.add_heading_characteristic())
        chars.append(self.add_battery_voltage_characteristic())
        chars.append(self.add_leak_detection_characteristic())
        chars.append(self.add_sensor_status_characteristic())
        return chars


# ---------------------------------------------------------------------------
# GATT Server Configuration
# ---------------------------------------------------------------------------

@dataclass
class GATTServerConfig:
    """
    Configuration for the BLE GATT server on nRF52840.
    """
    max_services: int = 10
    max_characteristics_per_service: int = 20
    max_descriptors_per_char: int = 5
    att_mtu: int = 247
    max_connected_clients: int = 20
    service_changed_indication: bool = True
    max_write_without_response_queue: int = 8

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.att_mtu < 23 or self.att_mtu > 517:
            errors.append(f"ATT MTU must be 23-517, got {self.att_mtu}.")
        if self.max_connected_clients < 1 or self.max_connected_clients > 20:
            errors.append(f"Max connections must be 1-20, got {self.max_connected_clients}.")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0
