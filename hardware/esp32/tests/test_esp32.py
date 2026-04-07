"""Comprehensive tests for NEXUS ESP32 hardware configuration modules.

Covers config_esp32, config_esp32_s3, config_esp32_c6, and wifi_mesh
with 65 tests spanning default values, overrides, type checks, enums,
frozen dataclasses, factory functions, and edge cases.
"""

from __future__ import annotations

import sys
import os
import types

# Ensure the project root is on sys.path so imports work when running
# pytest from any working directory.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest
from dataclasses import fields, is_dataclass

from hardware.esp32.config_esp32 import (
    ESP32PinMap,
    SerialConfig,
    WireProtocolConfig,
    CommsConfig,
    ESP32BoardConfig,
    create_esp32_config,
)
from hardware.esp32.config_esp32_s3 import (
    ESP32S3PinMap,
    ESP32S3BoardConfig,
    create_esp32_s3_config,
)
from hardware.esp32.config_esp32_c6 import (
    ESP32C6PinMap,
    ESP32C6BoardConfig,
    create_esp32_c6_config,
)
from hardware.esp32.wifi_mesh import (
    MeshType,
    NodeRole,
    MeshConfig,
    MeshTopology,
    DataRateConfig,
    MeshNodeInfo,
)


# ===================================================================
# ESP32PinMap tests
# ===================================================================

class TestESP32PinMap:
    """Tests for the ESP32 (classic) pin mapping."""

    def test_default_gps_tx(self):
        pm = ESP32PinMap()
        assert pm.gps_tx == 9

    def test_default_gps_rx(self):
        pm = ESP32PinMap()
        assert pm.gps_rx == 10

    def test_default_imu_sda(self):
        pm = ESP32PinMap()
        assert pm.imu_sda == 21

    def test_default_imu_scl(self):
        pm = ESP32PinMap()
        assert pm.imu_scl == 22

    def test_default_sonar_trig(self):
        pm = ESP32PinMap()
        assert pm.sonar_trig == 5

    def test_default_sonar_echo(self):
        pm = ESP32PinMap()
        assert pm.sonar_echo == 18

    def test_default_servo_1(self):
        pm = ESP32PinMap()
        assert pm.servo_1 == 13

    def test_default_servo_2(self):
        pm = ESP32PinMap()
        assert pm.servo_2 == 14

    def test_default_led(self):
        pm = ESP32PinMap()
        assert pm.led == 2

    def test_default_temp_pin(self):
        pm = ESP32PinMap()
        assert pm.temp_pin == 34

    def test_default_pressure_pins(self):
        pm = ESP32PinMap()
        assert pm.pressure_sda == 21
        assert pm.pressure_scl == 22

    def test_frozen(self):
        pm = ESP32PinMap()
        with pytest.raises(AttributeError):
            pm.gps_tx = 99  # type: ignore[misc]

    def test_custom_pins(self):
        pm = ESP32PinMap(gps_tx=7, gps_rx=8)
        assert pm.gps_tx == 7
        assert pm.gps_rx == 8


# ===================================================================
# SerialConfig tests
# ===================================================================

class TestSerialConfig:
    """Tests for serial port configuration."""

    def test_default_baud_rate(self):
        sc = SerialConfig()
        assert sc.baud_rate == 115200

    def test_default_data_bits(self):
        sc = SerialConfig()
        assert sc.data_bits == 8

    def test_default_parity(self):
        sc = SerialConfig()
        assert sc.parity == "N"

    def test_default_stop_bits(self):
        sc = SerialConfig()
        assert sc.stop_bits == 1

    def test_custom_baud(self):
        sc = SerialConfig(baud_rate=9600)
        assert sc.baud_rate == 9600

    def test_frozen(self):
        sc = SerialConfig()
        with pytest.raises(AttributeError):
            sc.baud_rate = 57600  # type: ignore[misc]


# ===================================================================
# WireProtocolConfig tests
# ===================================================================

class TestWireProtocolConfig:
    """Tests for wire protocol framing configuration."""

    def test_default_preamble(self):
        wp = WireProtocolConfig()
        assert wp.frame_preamble == b"\xaa\x55"

    def test_default_max_frame_size(self):
        wp = WireProtocolConfig()
        assert wp.max_frame_size == 1024

    def test_default_heartbeat_ms(self):
        wp = WireProtocolConfig()
        assert wp.heartbeat_ms == 1000

    def test_frozen(self):
        wp = WireProtocolConfig()
        with pytest.raises(AttributeError):
            wp.max_frame_size = 2048  # type: ignore[misc]


# ===================================================================
# CommsConfig tests
# ===================================================================

class TestCommsConfig:
    """Tests for communications configuration."""

    def test_default_wifi_ssid_empty(self):
        cc = CommsConfig()
        assert cc.wifi_ssid == ""

    def test_default_wifi_password_empty(self):
        cc = CommsConfig()
        assert cc.wifi_password == ""

    def test_default_mqtt_broker_empty(self):
        cc = CommsConfig()
        assert cc.mqtt_broker == ""

    def test_default_ble_enabled(self):
        cc = CommsConfig()
        assert cc.ble_enabled is True

    def test_default_wifi_channel(self):
        cc = CommsConfig()
        assert cc.wifi_channel == 6

    def test_default_ap_mode(self):
        cc = CommsConfig()
        assert cc.ap_mode is False

    def test_mutable(self):
        cc = CommsConfig()
        cc.wifi_ssid = "MARINE-WIFI"
        assert cc.wifi_ssid == "MARINE-WIFI"


# ===================================================================
# ESP32BoardConfig tests
# ===================================================================

class TestESP32BoardConfig:
    """Tests for the top-level ESP32 board configuration."""

    def test_default_board_name(self):
        cfg = ESP32BoardConfig()
        assert cfg.board_name == "ESP32"

    def test_default_cpu(self):
        cfg = ESP32BoardConfig()
        assert cfg.cpu == "Xtensa LX6"

    def test_default_clock_mhz(self):
        cfg = ESP32BoardConfig()
        assert cfg.clock_mhz == 240

    def test_default_cores(self):
        cfg = ESP32BoardConfig()
        assert cfg.cores == 2

    def test_default_flash_mb(self):
        cfg = ESP32BoardConfig()
        assert cfg.flash_mb == 4

    def test_default_sram_kb(self):
        cfg = ESP32BoardConfig()
        assert cfg.sram_kb == 520

    def test_default_gpio_count(self):
        cfg = ESP32BoardConfig()
        assert cfg.gpio_count == 34

    def test_default_adc_channels(self):
        cfg = ESP32BoardConfig()
        assert cfg.adc_channels == 18

    def test_default_pwm_channels(self):
        cfg = ESP32BoardConfig()
        assert cfg.pwm_channels == 16

    def test_default_uart_count(self):
        cfg = ESP32BoardConfig()
        assert cfg.uart_count == 3

    def test_default_spi_count(self):
        cfg = ESP32BoardConfig()
        assert cfg.spi_count == 2

    def test_default_i2c_count(self):
        cfg = ESP32BoardConfig()
        assert cfg.i2c_count == 2

    def test_default_wifi(self):
        cfg = ESP32BoardConfig()
        assert cfg.wifi is True

    def test_default_ble(self):
        cfg = ESP32BoardConfig()
        assert cfg.ble is True

    def test_nested_pin_map_is_esp32_pin_map(self):
        cfg = ESP32BoardConfig()
        assert isinstance(cfg.pin_map, ESP32PinMap)

    def test_nested_serial_is_serial_config(self):
        cfg = ESP32BoardConfig()
        assert isinstance(cfg.serial, SerialConfig)

    def test_nested_protocol_is_wire_protocol_config(self):
        cfg = ESP32BoardConfig()
        assert isinstance(cfg.protocol, WireProtocolConfig)

    def test_nested_comms_is_comms_config(self):
        cfg = ESP32BoardConfig()
        assert isinstance(cfg.comms, CommsConfig)

    def test_is_dataclass(self):
        assert is_dataclass(ESP32BoardConfig)


# ===================================================================
# create_esp32_config factory tests
# ===================================================================

class TestCreateESP32Config:
    """Tests for the ESP32 board config factory function."""

    def test_returns_esp32_board_config(self):
        cfg = create_esp32_config()
        assert isinstance(cfg, ESP32BoardConfig)

    def test_no_overrides_gives_defaults(self):
        cfg = create_esp32_config()
        assert cfg.board_name == "ESP32"
        assert cfg.clock_mhz == 240

    def test_override_board_name(self):
        cfg = create_esp32_config(board_name="ESP32-NAV")
        assert cfg.board_name == "ESP32-NAV"

    def test_override_clock(self):
        cfg = create_esp32_config(clock_mhz=160)
        assert cfg.clock_mhz == 160

    def test_override_nested_pin_map_dict(self):
        cfg = create_esp32_config(pin_map={"gps_tx": 7, "gps_rx": 8})
        assert cfg.pin_map.gps_tx == 7
        assert cfg.pin_map.gps_rx == 8

    def test_override_nested_serial_dict(self):
        cfg = create_esp32_config(serial={"baud_rate": 9600})
        assert cfg.serial.baud_rate == 9600

    def test_override_nested_comms_dict(self):
        cfg = create_esp32_config(comms={"wifi_ssid": "TEST", "ble_enabled": False})
        assert cfg.comms.wifi_ssid == "TEST"
        assert cfg.comms.ble_enabled is False

    def test_unknown_override_raises_type_error(self):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            create_esp32_config(nonexistent_param=42)


# ===================================================================
# ESP32-S3 tests
# ===================================================================

class TestESP32S3:
    """Tests for ESP32-S3 board configuration."""

    def test_default_board_name(self):
        cfg = ESP32S3BoardConfig()
        assert cfg.board_name == "ESP32-S3"

    def test_cpu_is_lx7(self):
        cfg = ESP32S3BoardConfig()
        assert cfg.cpu == "Xtensa LX7"

    def test_clock_240(self):
        cfg = ESP32S3BoardConfig()
        assert cfg.clock_mhz == 240

    def test_dual_core(self):
        cfg = ESP32S3BoardConfig()
        assert cfg.cores == 2

    def test_flash_8mb(self):
        cfg = ESP32S3BoardConfig()
        assert cfg.flash_mb == 8

    def test_sram_512kb(self):
        cfg = ESP32S3BoardConfig()
        assert cfg.sram_kb == 512

    def test_psram_8mb(self):
        cfg = ESP32S3BoardConfig()
        assert cfg.psram_mb == 8

    def test_gpio_45(self):
        cfg = ESP32S3BoardConfig()
        assert cfg.gpio_count == 45

    def test_wifi_6(self):
        cfg = ESP32S3BoardConfig()
        assert cfg.wifi_6 is True

    def test_ble_5_0(self):
        cfg = ESP32S3BoardConfig()
        assert cfg.ble_version == "5.0"

    def test_usb_otg(self):
        cfg = ESP32S3BoardConfig()
        assert cfg.usb_otg is True

    def test_camera_dvp(self):
        cfg = ESP32S3BoardConfig()
        assert cfg.camera_dvp is True

    def test_pin_map_is_frozen(self):
        pm = ESP32S3PinMap()
        with pytest.raises(AttributeError):
            pm.gps_tx = 0  # type: ignore[misc]

    def test_factory_no_overrides(self):
        cfg = create_esp32_s3_config()
        assert isinstance(cfg, ESP32S3BoardConfig)
        assert cfg.board_name == "ESP32-S3"

    def test_factory_with_overrides(self):
        cfg = create_esp32_s3_config(board_name="S3-PRIMARY", psram_mb=16)
        assert cfg.board_name == "S3-PRIMARY"
        assert cfg.psram_mb == 16

    def test_factory_nested_pin_map_override(self):
        cfg = create_esp32_s3_config(pin_map={"led": 99})
        assert cfg.pin_map.led == 99

    def test_factory_unknown_param_raises(self):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            create_esp32_s3_config(bogus=1)


# ===================================================================
# ESP32-C6 tests
# ===================================================================

class TestESP32C6:
    """Tests for ESP32-C6 board configuration."""

    def test_default_board_name(self):
        cfg = ESP32C6BoardConfig()
        assert cfg.board_name == "ESP32-C6"

    def test_cpu_is_riscv(self):
        cfg = ESP32C6BoardConfig()
        assert cfg.cpu == "RISC-V"

    def test_clock_160(self):
        cfg = ESP32C6BoardConfig()
        assert cfg.clock_mhz == 160

    def test_single_core(self):
        cfg = ESP32C6BoardConfig()
        assert cfg.cores == 1

    def test_flash_4mb(self):
        cfg = ESP32C6BoardConfig()
        assert cfg.flash_mb == 4

    def test_sram_512kb(self):
        cfg = ESP32C6BoardConfig()
        assert cfg.sram_kb == 512

    def test_no_psram(self):
        cfg = ESP32C6BoardConfig()
        assert cfg.psram_mb == 0

    def test_wifi_6(self):
        cfg = ESP32C6BoardConfig()
        assert cfg.wifi_6 is True

    def test_zigbee(self):
        cfg = ESP32C6BoardConfig()
        assert cfg.zigbee is True

    def test_thread(self):
        cfg = ESP32C6BoardConfig()
        assert cfg.thread is True

    def test_matter(self):
        cfg = ESP32C6BoardConfig()
        assert cfg.matter is True

    def test_pin_map_is_frozen(self):
        pm = ESP32C6PinMap()
        with pytest.raises(AttributeError):
            pm.led = 0  # type: ignore[misc]

    def test_factory_no_overrides(self):
        cfg = create_esp32_c6_config()
        assert isinstance(cfg, ESP32C6BoardConfig)
        assert cfg.board_name == "ESP32-C6"

    def test_factory_with_overrides(self):
        cfg = create_esp32_c6_config(board_name="C6-SENSOR-NODE")
        assert cfg.board_name == "C6-SENSOR-NODE"

    def test_factory_nested_pin_map_override(self):
        cfg = create_esp32_c6_config(pin_map={"gps_tx": 1})
        assert cfg.pin_map.gps_tx == 1

    def test_factory_unknown_param_raises(self):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            create_esp32_c6_config(invalid=True)


# ===================================================================
# WiFi Mesh tests
# ===================================================================

class TestMeshEnums:
    """Tests for mesh enumerations."""

    def test_mesh_type_star(self):
        assert MeshType.STAR == "star"

    def test_mesh_type_tree(self):
        assert MeshType.TREE == "tree"

    def test_mesh_type_chain(self):
        assert MeshType.CHAIN == "chain"

    def test_mesh_type_values(self):
        assert set(MeshType) == {MeshType.STAR, MeshType.TREE, MeshType.CHAIN}

    def test_node_role_coordinator(self):
        assert NodeRole.COORDINATOR == "coordinator"

    def test_node_role_router(self):
        assert NodeRole.ROUTER == "router"

    def test_node_role_end_device(self):
        assert NodeRole.END_DEVICE == "end_device"

    def test_node_role_values(self):
        assert set(NodeRole) == {NodeRole.COORDINATOR, NodeRole.ROUTER, NodeRole.END_DEVICE}


class TestMeshConfig:
    """Tests for mesh network configuration."""

    def test_default_mesh_type(self):
        mc = MeshConfig()
        assert mc.mesh_type == MeshType.TREE

    def test_default_max_nodes(self):
        mc = MeshConfig()
        assert mc.max_nodes == 20

    def test_default_channel(self):
        mc = MeshConfig()
        assert mc.channel == 6

    def test_default_encryption(self):
        mc = MeshConfig()
        assert mc.encryption is True

    def test_default_ssid_prefix(self):
        mc = MeshConfig()
        assert mc.ssid_prefix == "NEXUS-MESH"

    def test_default_password_empty(self):
        mc = MeshConfig()
        assert mc.password == ""

    def test_custom_values(self):
        mc = MeshConfig(mesh_type=MeshType.STAR, max_nodes=10, channel=1)
        assert mc.mesh_type == MeshType.STAR
        assert mc.max_nodes == 10
        assert mc.channel == 1


class TestMeshTopology:
    """Tests for mesh topology parameters."""

    def test_default_auto_heal(self):
        mt = MeshTopology()
        assert mt.auto_heal is True

    def test_default_max_hops(self):
        mt = MeshTopology()
        assert mt.max_hops == 5

    def test_default_parent_timeout(self):
        mt = MeshTopology()
        assert mt.parent_timeout_s == 60

    def test_default_retry_count(self):
        mt = MeshTopology()
        assert mt.retry_count == 3


class TestDataRateConfig:
    """Tests for data-rate configuration."""

    def test_default_sensor_hz(self):
        dr = DataRateConfig()
        assert dr.sensor_hz == 10

    def test_default_cmd_latency_ms(self):
        dr = DataRateConfig()
        assert dr.cmd_latency_ms == 100

    def test_default_heartbeat_s(self):
        dr = DataRateConfig()
        assert dr.heartbeat_s == 5

    def test_default_max_payload_bytes(self):
        dr = DataRateConfig()
        assert dr.max_payload_bytes == 256


class TestMeshNodeInfo:
    """Tests for mesh node runtime info."""

    def test_default_node_id_empty(self):
        ni = MeshNodeInfo()
        assert ni.node_id == ""

    def test_default_role(self):
        ni = MeshNodeInfo()
        assert ni.role == NodeRole.END_DEVICE

    def test_default_parent_id_empty(self):
        ni = MeshNodeInfo()
        assert ni.parent_id == ""

    def test_default_rssi(self):
        ni = MeshNodeInfo()
        assert ni.rssi_dbm == 0

    def test_default_uptime(self):
        ni = MeshNodeInfo()
        assert ni.uptime_s == 0

    def test_custom_values(self):
        ni = MeshNodeInfo(
            node_id="NEXUS-001",
            role=NodeRole.COORDINATOR,
            rssi_dbm=-45,
            uptime_s=3600,
        )
        assert ni.node_id == "NEXUS-001"
        assert ni.role == NodeRole.COORDINATOR
        assert ni.rssi_dbm == -45
        assert ni.uptime_s == 3600
        assert ni.parent_id == ""  # coordinator has no parent

    def test_router_with_parent(self):
        ni = MeshNodeInfo(
            node_id="NEXUS-005",
            role=NodeRole.ROUTER,
            parent_id="NEXUS-001",
            rssi_dbm=-60,
        )
        assert ni.role == NodeRole.ROUTER
        assert ni.parent_id == "NEXUS-001"


# ===================================================================
# Cross-module / integration tests
# ===================================================================

class TestCrossModule:
    """Integration-style tests spanning multiple modules."""

    def test_all_configs_are_dataclasses(self):
        assert is_dataclass(ESP32BoardConfig)
        assert is_dataclass(ESP32S3BoardConfig)
        assert is_dataclass(ESP32C6BoardConfig)
        assert is_dataclass(MeshConfig)
        assert is_dataclass(MeshTopology)
        assert is_dataclass(DataRateConfig)
        assert is_dataclass(MeshNodeInfo)

    def test_esp32_has_more_uart_than_c6(self):
        esp32 = ESP32BoardConfig()
        c6 = ESP32C6BoardConfig()
        assert esp32.uart_count > c6.uart_count

    def test_s3_has_more_gpio_than_esp32(self):
        esp32 = ESP32BoardConfig()
        s3 = ESP32S3BoardConfig()
        assert s3.gpio_count > esp32.gpio_count

    def test_s3_has_more_flash_than_c6(self):
        s3 = ESP32S3BoardConfig()
        c6 = ESP32C6BoardConfig()
        assert s3.flash_mb > c6.flash_mb

    def test_c6_is_single_core(self):
        esp32 = ESP32BoardConfig()
        s3 = ESP32S3BoardConfig()
        c6 = ESP32C6BoardConfig()
        assert esp32.cores == 2
        assert s3.cores == 2
        assert c6.cores == 1

    def test_s3_and_c6_have_wifi_6(self):
        s3 = ESP32S3BoardConfig()
        c6 = ESP32C6BoardConfig()
        assert s3.wifi_6 is True
        assert c6.wifi_6 is True

    def test_c6_has_protocol_features_others_dont(self):
        c6 = ESP32C6BoardConfig()
        assert c6.zigbee is True
        assert c6.thread is True
        assert c6.matter is True

    def test_s3_has_usb_otg(self):
        s3 = ESP32S3BoardConfig()
        assert s3.usb_otg is True

    def test_s3_has_camera_dvp(self):
        s3 = ESP32S3BoardConfig()
        assert s3.camera_dvp is True

    def test_pin_maps_are_frozen(self):
        assert isinstance(ESP32PinMap(), type(ESP32PinMap()))
        # All three pin maps should be frozen dataclasses
        for pm_cls in (ESP32PinMap, ESP32S3PinMap, ESP32C6PinMap):
            pm = pm_cls()
            with pytest.raises(AttributeError):
                pm.led = 999  # type: ignore[misc]

    def test_package_imports(self):
        """Verify the package-level imports all resolve."""
        from hardware.esp32 import (
            ESP32PinMap,
            SerialConfig,
            WireProtocolConfig,
            CommsConfig,
            ESP32BoardConfig,
            create_esp32_config,
            ESP32S3PinMap,
            ESP32S3BoardConfig,
            create_esp32_s3_config,
            ESP32C6PinMap,
            ESP32C6BoardConfig,
            create_esp32_c6_config,
            MeshType,
            NodeRole,
            MeshConfig,
            MeshTopology,
            DataRateConfig,
            MeshNodeInfo,
        )
        # Just verify they imported without error
        assert ESP32BoardConfig is not None
