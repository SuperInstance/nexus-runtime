"""
Tests for NEXUS BLE GATT Profiles.
"""

import struct
import pytest
from hardware.nrf52.ble_profiles import (
    NEXUS_BASE_UUID,
    NEXUS_SERVICE_UUID,
    NEXUS_DEVICE_INFO_UUID,
    NEXUS_COMMAND_UUID,
    NEXUS_TELEMETRY_UUID,
    NEXUSCharacteristicUUID,
    BLE_UUID_BATTERY_SERVICE,
    BLE_UUID_BATTERY_LEVEL,
    BLE_UUID_DEVICE_NAME,
    BLE_UUID_MANUFACTURER,
    CharProperty,
    CharFormat,
    GATTError,
    CharacteristicDescriptor,
    CharacteristicMetadata,
    BLECharacteristic,
    BLEService,
    NEXUSMarineService,
    GATTServerConfig,
)


class TestUUIDConstants:
    def test_base_uuid_format(self):
        assert "6E787873" in NEXUS_BASE_UUID
        assert "{:04x}" in NEXUS_BASE_UUID

    def test_service_uuid(self):
        assert NEXUS_SERVICE_UUID == NEXUS_BASE_UUID.format(0x0001)

    def test_device_info_uuid(self):
        assert NEXUS_DEVICE_INFO_UUID == NEXUS_BASE_UUID.format(0x0002)

    def test_command_uuid(self):
        assert NEXUS_COMMAND_UUID == NEXUS_BASE_UUID.format(0x0003)

    def test_telemetry_uuid(self):
        assert NEXUS_TELEMETRY_UUID == NEXUS_BASE_UUID.format(0x0004)

    def test_characteristic_uuids_unique(self):
        uuids = [
            NEXUSCharacteristicUUID.TEMPERATURE,
            NEXUSCharacteristicUUID.DEPTH,
            NEXUSCharacteristicUUID.PRESSURE,
            NEXUSCharacteristicUUID.GPS_POSITION,
            NEXUSCharacteristicUUID.HEADING,
        ]
        assert len(uuids) == len(set(uuids))

    def test_standard_uuids(self):
        assert BLE_UUID_BATTERY_SERVICE == "0000180f-0000-1000-8000-00805f9b34fb"
        assert BLE_UUID_BATTERY_LEVEL == "00002a19-0000-1000-8000-00805f9b34fb"


class TestCharMetadata:
    def test_encode_presentation_format(self):
        meta = CharacteristicMetadata(format=CharFormat.FLOAT32, unit_uuid="272F")
        encoded = meta.encode_presentation_format()
        assert len(encoded) == 20  # 1 + 1 + 2 + 16 bytes
        fmt_byte = struct.unpack("<B", encoded[:1])[0]
        assert fmt_byte == CharFormat.FLOAT32

    def test_exponent(self):
        meta = CharacteristicMetadata(exponent=-1)
        encoded = meta.encode_presentation_format()
        exp = struct.unpack("<b", encoded[1:2])[0]
        assert exp == -1


class TestBLECharacteristic:
    def test_properties(self):
        c = BLECharacteristic(
            uuid="test-uuid",
            name="Test",
            properties=CharProperty.READ | CharProperty.NOTIFY,
        )
        assert c.can_read() is True
        assert c.can_write() is False
        assert c.can_notify() is True
        assert c.can_indicate() is False

    def test_write_property(self):
        c = BLECharacteristic(
            uuid="test",
            name="T",
            properties=CharProperty.READ | CharProperty.WRITE,
        )
        assert c.can_write() is True

    def test_indicate_property(self):
        c = BLECharacteristic(
            uuid="test", name="T",
            properties=CharProperty.READ | CharProperty.INDICATE,
        )
        assert c.can_indicate() is True

    def test_add_cccd(self):
        c = BLECharacteristic(uuid="test", name="T", properties=CharProperty.NOTIFY)
        c.add_cccd()
        assert len(c.descriptors) == 1
        assert "2902" in c.descriptors[0].uuid

    def test_set_value_float(self):
        c = BLECharacteristic(uuid="t", name="T")
        val = c.set_value_float(23.5)
        decoded = struct.unpack("<f", val)[0]
        assert abs(decoded - 23.5) < 0.01

    def test_set_value_uint16(self):
        c = BLECharacteristic(uuid="t", name="T")
        val = c.set_value_uint16(65535)
        assert len(val) == 2
        assert c.decode_uint16() == 65535

    def test_set_value_int16(self):
        c = BLECharacteristic(uuid="t", name="T")
        c.set_value_int16(-100)
        assert c.decode_uint16() == 65436  # unsigned interpretation

    def test_set_value_string(self):
        c = BLECharacteristic(uuid="t", name="T")
        c.set_value_string("NEXUS")
        assert c.decode_string() == "NEXUS"

    def test_set_value_string_truncated(self):
        c = BLECharacteristic(uuid="t", name="T")
        c.set_value_string("A" * 30, max_len=10)
        assert len(c.value) == 10

    def test_set_value_gps(self):
        c = BLECharacteristic(uuid="t", name="T")
        c.set_value_gps(37.7749, -122.4194, 10.0)
        result = c.decode_gps()
        assert result is not None
        lat, lon, alt = result
        assert abs(lat - 37.7749) < 0.01
        assert abs(lon - (-122.4194)) < 0.01
        assert abs(alt - 10.0) < 0.01

    def test_decode_float_short(self):
        c = BLECharacteristic(uuid="t", name="T", value=b"\x01")
        assert c.decode_float() is None

    def test_decode_uint16_short(self):
        c = BLECharacteristic(uuid="t", name="T", value=b"\x01")
        assert c.decode_uint16() is None

    def test_validate_ok(self):
        c = BLECharacteristic(uuid="test-uuid", name="TestChar")
        assert c.is_valid() is True

    def test_validate_empty_uuid(self):
        c = BLECharacteristic(uuid="", name="Test")
        errors = c.validate()
        assert any("UUID" in e for e in errors)

    def test_validate_empty_name(self):
        c = BLECharacteristic(uuid="test", name="")
        errors = c.validate()
        assert any("name" in e for e in errors)


class TestBLEService:
    def test_create_service(self):
        s = BLEService(uuid="svc-uuid", name="TestService")
        assert s.is_primary is True
        assert s.characteristic_count() == 0

    def test_add_and_get_characteristic(self):
        s = BLEService(uuid="svc", name="Svc")
        c = BLECharacteristic(uuid="char-uuid", name="Char")
        s.add_characteristic(c)
        assert s.characteristic_count() == 1
        assert s.get_characteristic("char-uuid") is c

    def test_get_missing_characteristic(self):
        s = BLEService(uuid="svc", name="Svc")
        assert s.get_characteristic("nonexistent") is None

    def test_total_handles_no_chars(self):
        s = BLEService(uuid="svc", name="Svc")
        assert s.total_handles() == 1

    def test_total_handles_with_chars(self):
        s = BLEService(uuid="svc", name="Svc")
        c = BLECharacteristic(uuid="c1", name="C1")
        c.add_cccd()
        s.add_characteristic(c)
        # 1 (service) + 2 (char decl + value) + 1 (cccd) = 4
        assert s.total_handles() == 4

    def test_validate_duplicate_uuids(self):
        s = BLEService(uuid="svc", name="Svc")
        s.add_characteristic(BLECharacteristic(uuid="dup", name="C1"))
        s.add_characteristic(BLECharacteristic(uuid="dup", name="C2"))
        errors = s.validate()
        assert any("Duplicate" in e for e in errors)

    def test_build_gatt_table(self):
        s = BLEService(uuid="svc-uuid", name="MyService")
        c = BLECharacteristic(uuid="char-uuid", name="MyChar", properties=CharProperty.READ)
        s.add_characteristic(c)
        table = s.build_gatt_table()
        assert len(table) >= 2
        assert table[0]["type"] == "service"
        assert table[1]["type"] == "characteristic_declaration"

    def test_build_gatt_table_handles_increment(self):
        s = BLEService(uuid="svc", name="Svc")
        c = BLECharacteristic(uuid="ch", name="Ch", properties=CharProperty.READ)
        c.add_cccd()
        s.add_characteristic(c)
        table = s.build_gatt_table()
        handles = [e["handle"] for e in table]
        assert handles == sorted(handles)
        assert all(handles[i] + 1 == handles[i + 1] for i in range(len(handles) - 1))


class TestNEXUSMarineService:
    def setup_method(self):
        self.svc = NEXUSMarineService()

    def test_service_uuid(self):
        assert self.svc.uuid == NEXUS_SERVICE_UUID

    def test_add_temperature(self):
        c = self.svc.add_temperature_characteristic()
        assert c.name == "Temperature"
        assert c.can_notify() is True
        assert len(c.descriptors) == 1  # CCCD

    def test_add_depth(self):
        c = self.svc.add_depth_characteristic()
        assert c.name == "Depth"
        assert c.metadata is not None

    def test_add_gps(self):
        c = self.svc.add_gps_characteristic()
        assert c.name == "GPS Position"
        assert c.max_data_length == 12

    def test_add_conductivity(self):
        c = self.svc.add_conductivity_characteristic()
        assert c.name == "Conductivity"

    def test_add_leak_detection(self):
        c = self.svc.add_leak_detection_characteristic()
        assert c.name == "Leak Detection"
        assert c.can_indicate() is True

    def test_add_all_marine_characteristics(self):
        chars = self.svc.add_all_marine_characteristics()
        assert len(chars) == 10
        assert self.svc.characteristic_count() == 10

    def test_service_valid(self):
        self.svc.add_all_marine_characteristics()
        assert self.svc.is_valid() is True

    def test_build_gatt_table_full(self):
        self.svc.add_all_marine_characteristics()
        table = self.svc.build_gatt_table()
        assert len(table) > 0
        # Each char = 3 handles (decl + value + cccd), plus 1 for service
        expected = 1 + (3 * 10)
        assert len(table) == expected

    def test_gps_value_roundtrip(self):
        c = self.svc.add_gps_characteristic()
        c.set_value_gps(45.0, -90.0, 5.5)
        lat, lon, alt = c.decode_gps()
        assert abs(lat - 45.0) < 0.01
        assert abs(lon - (-90.0)) < 0.01

    def test_sensor_status_metadata(self):
        c = self.svc.add_sensor_status_characteristic()
        assert c.metadata.format == CharFormat.UINT32
        assert c.max_data_length == 16


class TestGATTServerConfig:
    def test_default_valid(self):
        cfg = GATTServerConfig()
        assert cfg.is_valid() is True

    def test_invalid_mtu(self):
        cfg = GATTServerConfig(att_mtu=10)
        errors = cfg.validate()
        assert any("MTU" in e for e in errors)

    def test_invalid_max_conn(self):
        cfg = GATTServerConfig(max_connected_clients=50)
        errors = cfg.validate()
        assert any("connections" in e for e in errors)

    def test_max_conn_boundary_low(self):
        cfg = GATTServerConfig(max_connected_clients=0)
        errors = cfg.validate()
        assert any("connections" in e for e in errors)


class TestDescriptor:
    def test_short_uuid_standard(self):
        d = CharacteristicDescriptor(uuid="00002902-0000-1000-8000-00805f9b34fb")
        assert d.short_uuid == "2902"

    def test_permissions(self):
        d = CharacteristicDescriptor(
            uuid="test",
            permissions=CharProperty.READ | CharProperty.WRITE,
        )
        assert d.permissions & CharProperty.READ
        assert d.permissions & CharProperty.WRITE
