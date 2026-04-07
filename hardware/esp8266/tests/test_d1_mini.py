"""Tests for NEXUS Wemos D1 Mini hardware configuration module."""

from __future__ import annotations

import sys
import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest
from dataclasses import is_dataclass

from hardware.esp8266.config_d1_mini import (
    D1MiniPinMap,
    D1MiniBoardConfig,
    create_d1_mini_config,
)


# ===================================================================
# D1MiniPinMap tests
# ===================================================================

class TestD1MiniPinMap:
    def test_d0_pin(self):
        pm = D1MiniPinMap()
        assert pm.D0 == 16

    def test_d1_pin(self):
        pm = D1MiniPinMap()
        assert pm.D1 == 5

    def test_d2_pin(self):
        pm = D1MiniPinMap()
        assert pm.D2 == 4

    def test_d3_pin(self):
        pm = D1MiniPinMap()
        assert pm.D3 == 0

    def test_d4_pin(self):
        pm = D1MiniPinMap()
        assert pm.D4 == 2

    def test_d5_pin(self):
        pm = D1MiniPinMap()
        assert pm.D5 == 14

    def test_d6_pin(self):
        pm = D1MiniPinMap()
        assert pm.D6 == 12

    def test_d7_pin(self):
        pm = D1MiniPinMap()
        assert pm.D7 == 13

    def test_d8_pin(self):
        pm = D1MiniPinMap()
        assert pm.D8 == 15

    def test_a0_pin(self):
        pm = D1MiniPinMap()
        assert pm.A0 == 17

    def test_default_gps_tx(self):
        pm = D1MiniPinMap()
        assert pm.gps_tx == 1

    def test_default_gps_rx(self):
        pm = D1MiniPinMap()
        assert pm.gps_rx == 3

    def test_default_imu_pins(self):
        pm = D1MiniPinMap()
        assert pm.imu_sda == 4
        assert pm.imu_scl == 5

    def test_default_sonar_pins(self):
        pm = D1MiniPinMap()
        assert pm.sonar_trig == 14
        assert pm.sonar_echo == 12

    def test_default_servo(self):
        pm = D1MiniPinMap()
        assert pm.servo_1 == 0

    def test_default_led(self):
        pm = D1MiniPinMap()
        assert pm.led == 2

    def test_default_temp_pin(self):
        pm = D1MiniPinMap()
        assert pm.temp_pin == 17

    def test_spi_pins(self):
        pm = D1MiniPinMap()
        assert pm.spi_sck == 14
        assert pm.spi_miso == 12
        assert pm.spi_mosi == 13
        assert pm.spi_ss == 15

    def test_frozen(self):
        pm = D1MiniPinMap()
        with pytest.raises(AttributeError):
            pm.led = 99

    def test_custom_pins(self):
        pm = D1MiniPinMap(gps_tx=7, led=5)
        assert pm.gps_tx == 7
        assert pm.led == 5


# ===================================================================
# D1MiniBoardConfig tests
# ===================================================================

class TestD1MiniBoardConfig:
    def test_default_board_name(self):
        cfg = D1MiniBoardConfig()
        assert cfg.board_name == "Wemos D1 Mini"

    def test_default_cpu(self):
        cfg = D1MiniBoardConfig()
        assert cfg.cpu == "Xtensa L106"

    def test_default_clock_mhz(self):
        cfg = D1MiniBoardConfig()
        assert cfg.clock_mhz == 80

    def test_default_flash_mb(self):
        cfg = D1MiniBoardConfig()
        assert cfg.flash_mb == 4

    def test_default_sram_kb(self):
        cfg = D1MiniBoardConfig()
        assert cfg.sram_kb == 80

    def test_gpio_count(self):
        cfg = D1MiniBoardConfig()
        assert cfg.gpio_count == 11

    def test_single_adc(self):
        cfg = D1MiniBoardConfig()
        assert cfg.adc_channels == 1

    def test_wifi_enabled(self):
        cfg = D1MiniBoardConfig()
        assert cfg.wifi is True

    def test_no_ble(self):
        cfg = D1MiniBoardConfig()
        assert cfg.ble is False

    def test_nested_pin_map(self):
        cfg = D1MiniBoardConfig()
        assert isinstance(cfg.pin_map, D1MiniPinMap)

    def test_is_dataclass(self):
        assert is_dataclass(D1MiniBoardConfig)


# ===================================================================
# create_d1_mini_config factory tests
# ===================================================================

class TestCreateD1MiniConfig:
    def test_returns_d1_mini_config(self):
        cfg = create_d1_mini_config()
        assert isinstance(cfg, D1MiniBoardConfig)

    def test_no_overrides_gives_defaults(self):
        cfg = create_d1_mini_config()
        assert cfg.board_name == "Wemos D1 Mini"
        assert cfg.clock_mhz == 80

    def test_override_board_name(self):
        cfg = create_d1_mini_config(board_name="D1-TEMP-NODE")
        assert cfg.board_name == "D1-TEMP-NODE"

    def test_override_clock(self):
        cfg = create_d1_mini_config(clock_mhz=160)
        assert cfg.clock_mhz == 160

    def test_override_nested_pin_map_dict(self):
        cfg = create_d1_mini_config(pin_map={"led": 99})
        assert cfg.pin_map.led == 99

    def test_unknown_override_raises_type_error(self):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            create_d1_mini_config(nonexistent_param=42)


# ===================================================================
# Cross-module tests
# ===================================================================

class TestD1MiniCrossModule:
    def test_uses_same_soc_as_esp8266(self):
        from hardware.esp8266.config_esp8266 import ESP8266BoardConfig
        d1 = D1MiniBoardConfig()
        nodemcu = ESP8266BoardConfig()
        assert d1.cpu == nodemcu.cpu
        assert d1.flash_mb == nodemcu.flash_mb
        assert d1.sram_kb == nodemcu.sram_kb

    def test_d1_has_fewer_gpio_than_nodemcu(self):
        from hardware.esp8266.config_esp8266 import ESP8266BoardConfig
        d1 = D1MiniBoardConfig()
        nodemcu = ESP8266BoardConfig()
        assert d1.gpio_count < nodemcu.gpio_count

    def test_package_imports(self):
        from hardware.esp8266 import D1MiniBoardConfig as Imported
        assert Imported is not None
