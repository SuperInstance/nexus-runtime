"""Comprehensive tests for NEXUS ESP32-H2 hardware configuration module."""

from __future__ import annotations

import sys
import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest
from dataclasses import is_dataclass

from hardware.esp32.config_esp32_h2 import (
    ESP32H2PinMap,
    IoTGatewayConfig,
    ESP32H2BoardConfig,
    create_esp32_h2_config,
)


# ===================================================================
# ESP32H2PinMap tests
# ===================================================================

class TestESP32H2PinMap:
    def test_default_gps_tx(self):
        pm = ESP32H2PinMap()
        assert pm.gps_tx == 4

    def test_default_gps_rx(self):
        pm = ESP32H2PinMap()
        assert pm.gps_rx == 5

    def test_default_imu_sda(self):
        pm = ESP32H2PinMap()
        assert pm.imu_sda == 6

    def test_default_imu_scl(self):
        pm = ESP32H2PinMap()
        assert pm.imu_scl == 7

    def test_default_sonar_trig(self):
        pm = ESP32H2PinMap()
        assert pm.sonar_trig == 2

    def test_default_sonar_echo(self):
        pm = ESP32H2PinMap()
        assert pm.sonar_echo == 3

    def test_default_servo_1(self):
        pm = ESP32H2PinMap()
        assert pm.servo_1 == 8

    def test_default_servo_2(self):
        pm = ESP32H2PinMap()
        assert pm.servo_2 == 9

    def test_default_led(self):
        pm = ESP32H2PinMap()
        assert pm.led == 10

    def test_default_temp_pin(self):
        pm = ESP32H2PinMap()
        assert pm.temp_pin == 1

    def test_default_pressure_pins(self):
        pm = ESP32H2PinMap()
        assert pm.pressure_sda == 6
        assert pm.pressure_scl == 7

    def test_frozen(self):
        pm = ESP32H2PinMap()
        with pytest.raises(AttributeError):
            pm.led = 99

    def test_custom_pins(self):
        pm = ESP32H2PinMap(gps_tx=1, led=5)
        assert pm.gps_tx == 1
        assert pm.led == 5


# ===================================================================
# IoTGatewayConfig tests
# ===================================================================

class TestIoTGatewayConfig:
    def test_gateway_enabled(self):
        cfg = IoTGatewayConfig()
        assert cfg.gateway_enabled is True

    def test_thread_enabled(self):
        cfg = IoTGatewayConfig()
        assert cfg.thread_enabled is True

    def test_thread_network_name(self):
        cfg = IoTGatewayConfig()
        assert cfg.thread_network_name == "NEXUS-Thread"

    def test_thread_pan_id(self):
        cfg = IoTGatewayConfig()
        assert cfg.thread_pan_id == 0xDEAD

    def test_thread_channel(self):
        cfg = IoTGatewayConfig()
        assert cfg.thread_channel == 15

    def test_matter_fabric(self):
        cfg = IoTGatewayConfig()
        assert cfg.matter_fabric is True

    def test_zigbee_coordinator(self):
        cfg = IoTGatewayConfig()
        assert cfg.zigbee_coordinator is True

    def test_max_thread_nodes(self):
        cfg = IoTGatewayConfig()
        assert cfg.max_thread_nodes == 32

    def test_max_zigbee_nodes(self):
        cfg = IoTGatewayConfig()
        assert cfg.max_zigbee_nodes == 64

    def test_max_matter_devices(self):
        cfg = IoTGatewayConfig()
        assert cfg.max_matter_devices == 32

    def test_telemetry_bridging(self):
        cfg = IoTGatewayConfig()
        assert cfg.telemetry_bridging is True

    def test_command_forwarding(self):
        cfg = IoTGatewayConfig()
        assert cfg.command_forwarding is True

    def test_ota_update_enabled(self):
        cfg = IoTGatewayConfig()
        assert cfg.ota_update_enabled is True

    def test_serial_baud_rate(self):
        cfg = IoTGatewayConfig()
        assert cfg.serial_baud_rate == 115200

    def test_backbone_interface_default(self):
        cfg = IoTGatewayConfig()
        assert cfg.backbone_interface == "serial"

    def test_mutable(self):
        cfg = IoTGatewayConfig()
        cfg.gateway_id = "H2-GW-001"
        assert cfg.gateway_id == "H2-GW-001"


# ===================================================================
# ESP32H2BoardConfig tests
# ===================================================================

class TestESP32H2BoardConfig:
    def test_default_board_name(self):
        cfg = ESP32H2BoardConfig()
        assert cfg.board_name == "ESP32-H2"

    def test_cpu_is_riscv(self):
        cfg = ESP32H2BoardConfig()
        assert cfg.cpu == "RISC-V"

    def test_clock_96(self):
        cfg = ESP32H2BoardConfig()
        assert cfg.clock_mhz == 96

    def test_single_core(self):
        cfg = ESP32H2BoardConfig()
        assert cfg.cores == 1

    def test_flash_4mb(self):
        cfg = ESP32H2BoardConfig()
        assert cfg.flash_mb == 4

    def test_sram_256kb(self):
        cfg = ESP32H2BoardConfig()
        assert cfg.sram_kb == 256

    def test_no_psram(self):
        cfg = ESP32H2BoardConfig()
        assert cfg.psram_mb == 0

    def test_gpio_count(self):
        cfg = ESP32H2BoardConfig()
        assert cfg.gpio_count == 19

    def test_no_wifi(self):
        cfg = ESP32H2BoardConfig()
        assert cfg.wifi is False

    def test_no_wifi_6(self):
        cfg = ESP32H2BoardConfig()
        assert cfg.wifi_6 is False

    def test_ble_5(self):
        cfg = ESP32H2BoardConfig()
        assert cfg.ble is True
        assert cfg.ble_version == "5.0"

    def test_zigbee(self):
        cfg = ESP32H2BoardConfig()
        assert cfg.zigbee is True

    def test_thread(self):
        cfg = ESP32H2BoardConfig()
        assert cfg.thread is True

    def test_matter(self):
        cfg = ESP32H2BoardConfig()
        assert cfg.matter is True

    def test_ieee_802_15_4(self):
        cfg = ESP32H2BoardConfig()
        assert cfg.ieee_802_15_4 is True

    def test_deep_sleep(self):
        cfg = ESP32H2BoardConfig()
        assert cfg.deep_sleep is True

    def test_nested_pin_map(self):
        cfg = ESP32H2BoardConfig()
        assert isinstance(cfg.pin_map, ESP32H2PinMap)

    def test_nested_iot_gateway(self):
        cfg = ESP32H2BoardConfig()
        assert isinstance(cfg.iot_gateway, IoTGatewayConfig)

    def test_is_dataclass(self):
        assert is_dataclass(ESP32H2BoardConfig)


# ===================================================================
# create_esp32_h2_config factory tests
# ===================================================================

class TestCreateESP32H2Config:
    def test_returns_esp32_h2_config(self):
        cfg = create_esp32_h2_config()
        assert isinstance(cfg, ESP32H2BoardConfig)

    def test_no_overrides_gives_defaults(self):
        cfg = create_esp32_h2_config()
        assert cfg.board_name == "ESP32-H2"
        assert cfg.clock_mhz == 96

    def test_override_board_name(self):
        cfg = create_esp32_h2_config(board_name="H2-GATEWAY-01")
        assert cfg.board_name == "H2-GATEWAY-01"

    def test_override_nested_pin_map_dict(self):
        cfg = create_esp32_h2_config(pin_map={"led": 99})
        assert cfg.pin_map.led == 99

    def test_override_nested_iot_gateway_dict(self):
        cfg = create_esp32_h2_config(iot_gateway={"gateway_id": "GW-001"})
        assert cfg.iot_gateway.gateway_id == "GW-001"

    def test_unknown_override_raises_type_error(self):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            create_esp32_h2_config(invalid=True)


# ===================================================================
# Cross-module tests
# ===================================================================

class TestESP32H2CrossModule:
    def test_h2_no_wifi_unlike_esp32(self):
        from hardware.esp32.config_esp32 import ESP32BoardConfig
        esp32 = ESP32BoardConfig()
        h2 = ESP32H2BoardConfig()
        assert esp32.wifi is True
        assert h2.wifi is False

    def test_h2_has_802_15_4(self):
        cfg = ESP32H2BoardConfig()
        assert cfg.ieee_802_15_4 is True

    def test_h2_lower_clock_than_esp32(self):
        from hardware.esp32.config_esp32 import ESP32BoardConfig
        esp32 = ESP32BoardConfig()
        h2 = ESP32H2BoardConfig()
        assert h2.clock_mhz < esp32.clock_mhz

    def test_h2_lower_sram_than_c3(self):
        from hardware.esp32.config_esp32_c3 import ESP32C3BoardConfig
        c3 = ESP32C3BoardConfig()
        h2 = ESP32H2BoardConfig()
        assert h2.sram_kb < c3.sram_kb

    def test_h2_fewer_gpio_than_c3(self):
        from hardware.esp32.config_esp32_c3 import ESP32C3BoardConfig
        c3 = ESP32C3BoardConfig()
        h2 = ESP32H2BoardConfig()
        assert h2.gpio_count < c3.gpio_count

    def test_h2_lower_clock_than_c3(self):
        from hardware.esp32.config_esp32_c3 import ESP32C3BoardConfig
        c3 = ESP32C3BoardConfig()
        h2 = ESP32H2BoardConfig()
        assert h2.clock_mhz < c3.clock_mhz

    def test_h2_has_iot_gateway_config(self):
        cfg = ESP32H2BoardConfig()
        assert hasattr(cfg, 'iot_gateway')
        assert isinstance(cfg.iot_gateway, IoTGatewayConfig)

    def test_package_imports(self):
        from hardware.esp32 import ESP32H2BoardConfig as Imported
        assert Imported is not None
