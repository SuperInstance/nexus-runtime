"""Comprehensive tests for NEXUS ESP32-C3 hardware configuration module."""

from __future__ import annotations

import sys
import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest
from dataclasses import is_dataclass

from hardware.esp32.config_esp32_c3 import (
    ESP32C3PinMap,
    NexusNodeConfig,
    ESP32C3BoardConfig,
    create_esp32_c3_config,
)


# ===================================================================
# ESP32C3PinMap tests
# ===================================================================

class TestESP32C3PinMap:
    def test_default_gps_tx(self):
        pm = ESP32C3PinMap()
        assert pm.gps_tx == 4

    def test_default_gps_rx(self):
        pm = ESP32C3PinMap()
        assert pm.gps_rx == 5

    def test_default_imu_sda(self):
        pm = ESP32C3PinMap()
        assert pm.imu_sda == 8

    def test_default_imu_scl(self):
        pm = ESP32C3PinMap()
        assert pm.imu_scl == 9

    def test_default_sonar_trig(self):
        pm = ESP32C3PinMap()
        assert pm.sonar_trig == 2

    def test_default_sonar_echo(self):
        pm = ESP32C3PinMap()
        assert pm.sonar_echo == 3

    def test_default_servo_1(self):
        pm = ESP32C3PinMap()
        assert pm.servo_1 == 6

    def test_default_servo_2(self):
        pm = ESP32C3PinMap()
        assert pm.servo_2 == 7

    def test_default_led(self):
        pm = ESP32C3PinMap()
        assert pm.led == 10

    def test_default_temp_pin(self):
        pm = ESP32C3PinMap()
        assert pm.temp_pin == 1

    def test_default_pressure_pins(self):
        pm = ESP32C3PinMap()
        assert pm.pressure_sda == 8
        assert pm.pressure_scl == 9

    def test_frozen(self):
        pm = ESP32C3PinMap()
        with pytest.raises(AttributeError):
            pm.led = 99

    def test_custom_pins(self):
        pm = ESP32C3PinMap(gps_tx=1, led=5)
        assert pm.gps_tx == 1
        assert pm.led == 5


# ===================================================================
# NexusNodeConfig tests
# ===================================================================

class TestNexusNodeConfig:
    def test_deep_sleep_enabled(self):
        cfg = NexusNodeConfig()
        assert cfg.deep_sleep_enabled is True

    def test_deep_sleep_us(self):
        cfg = NexusNodeConfig()
        assert cfg.deep_sleep_us == 5_000_000

    def test_wake_on_gpio(self):
        cfg = NexusNodeConfig()
        assert cfg.wake_on_gpio is True

    def test_low_power_clock(self):
        cfg = NexusNodeConfig()
        assert cfg.low_power_clock_mhz == 32

    def test_sensor_burst_mode(self):
        cfg = NexusNodeConfig()
        assert cfg.sensor_burst_mode is True

    def test_max_sensor_channels(self):
        cfg = NexusNodeConfig()
        assert cfg.max_sensor_channels == 4

    def test_telemetry_compression(self):
        cfg = NexusNodeConfig()
        assert cfg.telemetry_compression is True

    def test_battery_monitor_adc(self):
        cfg = NexusNodeConfig()
        assert cfg.battery_monitor_adc == 0

    def test_low_battery_threshold(self):
        cfg = NexusNodeConfig()
        assert cfg.low_battery_threshold_mv == 3000

    def test_node_role(self):
        cfg = NexusNodeConfig()
        assert cfg.node_role == "end_device"


# ===================================================================
# ESP32C3BoardConfig tests
# ===================================================================

class TestESP32C3BoardConfig:
    def test_default_board_name(self):
        cfg = ESP32C3BoardConfig()
        assert cfg.board_name == "ESP32-C3"

    def test_cpu_is_riscv(self):
        cfg = ESP32C3BoardConfig()
        assert cfg.cpu == "RISC-V"

    def test_clock_160(self):
        cfg = ESP32C3BoardConfig()
        assert cfg.clock_mhz == 160

    def test_single_core(self):
        cfg = ESP32C3BoardConfig()
        assert cfg.cores == 1

    def test_flash_4mb(self):
        cfg = ESP32C3BoardConfig()
        assert cfg.flash_mb == 4

    def test_sram_400kb(self):
        cfg = ESP32C3BoardConfig()
        assert cfg.sram_kb == 400

    def test_no_psram(self):
        cfg = ESP32C3BoardConfig()
        assert cfg.psram_mb == 0

    def test_gpio_count(self):
        cfg = ESP32C3BoardConfig()
        assert cfg.gpio_count == 22

    def test_wifi_enabled(self):
        cfg = ESP32C3BoardConfig()
        assert cfg.wifi is True

    def test_no_wifi_6(self):
        cfg = ESP32C3BoardConfig()
        assert cfg.wifi_6 is False

    def test_ble_5(self):
        cfg = ESP32C3BoardConfig()
        assert cfg.ble is True
        assert cfg.ble_version == "5.0"

    def test_zigbee(self):
        cfg = ESP32C3BoardConfig()
        assert cfg.zigbee is True

    def test_thread(self):
        cfg = ESP32C3BoardConfig()
        assert cfg.thread is True

    def test_matter(self):
        cfg = ESP32C3BoardConfig()
        assert cfg.matter is True

    def test_deep_sleep(self):
        cfg = ESP32C3BoardConfig()
        assert cfg.deep_sleep is True

    def test_nested_pin_map(self):
        cfg = ESP32C3BoardConfig()
        assert isinstance(cfg.pin_map, ESP32C3PinMap)

    def test_nested_nexus_node(self):
        cfg = ESP32C3BoardConfig()
        assert isinstance(cfg.nexus_node, NexusNodeConfig)

    def test_is_dataclass(self):
        assert is_dataclass(ESP32C3BoardConfig)


# ===================================================================
# create_esp32_c3_config factory tests
# ===================================================================

class TestCreateESP32C3Config:
    def test_returns_esp32_c3_config(self):
        cfg = create_esp32_c3_config()
        assert isinstance(cfg, ESP32C3BoardConfig)

    def test_no_overrides_gives_defaults(self):
        cfg = create_esp32_c3_config()
        assert cfg.board_name == "ESP32-C3"
        assert cfg.clock_mhz == 160

    def test_override_board_name(self):
        cfg = create_esp32_c3_config(board_name="C3-SENSOR-NODE")
        assert cfg.board_name == "C3-SENSOR-NODE"

    def test_override_nested_pin_map_dict(self):
        cfg = create_esp32_c3_config(pin_map={"gps_tx": 1})
        assert cfg.pin_map.gps_tx == 1

    def test_override_nested_nexus_node_dict(self):
        cfg = create_esp32_c3_config(nexus_node={"deep_sleep_us": 10_000_000})
        assert cfg.nexus_node.deep_sleep_us == 10_000_000

    def test_unknown_override_raises_type_error(self):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            create_esp32_c3_config(invalid=True)


# ===================================================================
# Cross-module tests
# ===================================================================

class TestESP32C3CrossModule:
    def test_c3_is_single_core(self):
        from hardware.esp32.config_esp32 import ESP32BoardConfig
        esp32 = ESP32BoardConfig()
        c3 = ESP32C3BoardConfig()
        assert esp32.cores == 2
        assert c3.cores == 1

    def test_c3_lower_clock_than_esp32(self):
        from hardware.esp32.config_esp32 import ESP32BoardConfig
        esp32 = ESP32BoardConfig()
        c3 = ESP32C3BoardConfig()
        assert c3.clock_mhz < esp32.clock_mhz

    def test_c3_has_deep_sleep(self):
        from hardware.esp32.config_esp32 import ESP32BoardConfig
        esp32 = ESP32BoardConfig()
        c3 = ESP32C3BoardConfig()
        assert c3.deep_sleep is True
        assert not hasattr(esp32, 'deep_sleep')
