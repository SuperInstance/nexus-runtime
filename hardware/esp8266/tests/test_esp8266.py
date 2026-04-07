"""Comprehensive tests for NEXUS ESP8266 hardware configuration modules.

Covers config_esp8266 and wifi_manager with tests spanning default values,
overrides, type checks, enums, frozen dataclasses, factory functions,
WiFi connection state machine, and edge cases.
"""

from __future__ import annotations

import sys
import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest
from dataclasses import is_dataclass

from hardware.esp8266.config_esp8266 import (
    ESP8266PinMap,
    SerialConfig,
    WireProtocolConfig,
    CommsConfig,
    ESP8266BoardConfig,
    create_esp8266_config,
)
from hardware.esp8266.wifi_manager import (
    WiFiConnectionState,
    WiFiCredentials,
    WiFiManagerConfig,
    WiFiManager,
)


# ===================================================================
# ESP8266PinMap tests
# ===================================================================

class TestESP8266PinMap:
    def test_default_gps_tx(self):
        pm = ESP8266PinMap()
        assert pm.gps_tx == 1

    def test_default_gps_rx(self):
        pm = ESP8266PinMap()
        assert pm.gps_rx == 3

    def test_default_imu_sda(self):
        pm = ESP8266PinMap()
        assert pm.imu_sda == 4

    def test_default_imu_scl(self):
        pm = ESP8266PinMap()
        assert pm.imu_scl == 5

    def test_default_sonar_trig(self):
        pm = ESP8266PinMap()
        assert pm.sonar_trig == 14

    def test_default_sonar_echo(self):
        pm = ESP8266PinMap()
        assert pm.sonar_echo == 12

    def test_default_servo_1(self):
        pm = ESP8266PinMap()
        assert pm.servo_1 == 0

    def test_default_servo_2(self):
        pm = ESP8266PinMap()
        assert pm.servo_2 == 15

    def test_default_led(self):
        pm = ESP8266PinMap()
        assert pm.led == 2

    def test_default_temp_pin(self):
        pm = ESP8266PinMap()
        assert pm.temp_pin == 17

    def test_default_pressure_pins(self):
        pm = ESP8266PinMap()
        assert pm.pressure_sda == 4
        assert pm.pressure_scl == 5

    def test_default_relay(self):
        pm = ESP8266PinMap()
        assert pm.relay == 16

    def test_nexus_serial_pins(self):
        pm = ESP8266PinMap()
        assert pm.nexus_tx == 15
        assert pm.nexus_rx == 13

    def test_frozen(self):
        pm = ESP8266PinMap()
        with pytest.raises(AttributeError):
            pm.gps_tx = 99

    def test_custom_pins(self):
        pm = ESP8266PinMap(gps_tx=7, gps_rx=8)
        assert pm.gps_tx == 7
        assert pm.gps_rx == 8


# ===================================================================
# SerialConfig tests
# ===================================================================

class TestESP8266SerialConfig:
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
            sc.baud_rate = 57600


# ===================================================================
# WireProtocolConfig tests
# ===================================================================

class TestESP8266WireProtocolConfig:
    def test_default_preamble(self):
        wp = WireProtocolConfig()
        assert wp.frame_preamble == b"\xaa\x55"

    def test_default_max_frame_size(self):
        wp = WireProtocolConfig()
        assert wp.max_frame_size == 512

    def test_default_heartbeat_ms(self):
        wp = WireProtocolConfig()
        assert wp.heartbeat_ms == 1000

    def test_frozen(self):
        wp = WireProtocolConfig()
        with pytest.raises(AttributeError):
            wp.max_frame_size = 2048


# ===================================================================
# CommsConfig tests
# ===================================================================

class TestESP8266CommsConfig:
    def test_default_wifi_ssid_empty(self):
        cc = CommsConfig()
        assert cc.wifi_ssid == ""

    def test_default_wifi_password_empty(self):
        cc = CommsConfig()
        assert cc.wifi_password == ""

    def test_default_ble_disabled(self):
        cc = CommsConfig()
        assert cc.ble_enabled is False

    def test_default_ap_mode(self):
        cc = CommsConfig()
        assert cc.ap_mode is False

    def test_default_ap_ssid(self):
        cc = CommsConfig()
        assert cc.ap_ssid == "NEXUS-ESP8266"

    def test_default_mdns_enabled(self):
        cc = CommsConfig()
        assert cc.mdns_enabled is True

    def test_mutable(self):
        cc = CommsConfig()
        cc.wifi_ssid = "MARINE-WIFI"
        assert cc.wifi_ssid == "MARINE-WIFI"


# ===================================================================
# ESP8266BoardConfig tests
# ===================================================================

class TestESP8266BoardConfig:
    def test_default_board_name(self):
        cfg = ESP8266BoardConfig()
        assert "ESP8266" in cfg.board_name
        assert "NodeMCU" in cfg.board_name

    def test_default_cpu(self):
        cfg = ESP8266BoardConfig()
        assert cfg.cpu == "Xtensa L106"

    def test_default_clock_mhz(self):
        cfg = ESP8266BoardConfig()
        assert cfg.clock_mhz == 80

    def test_default_flash_mb(self):
        cfg = ESP8266BoardConfig()
        assert cfg.flash_mb == 4

    def test_default_sram_kb(self):
        cfg = ESP8266BoardConfig()
        assert cfg.sram_kb == 80

    def test_default_gpio_count(self):
        cfg = ESP8266BoardConfig()
        assert cfg.gpio_count == 17

    def test_single_adc(self):
        cfg = ESP8266BoardConfig()
        assert cfg.adc_channels == 1

    def test_single_uart(self):
        cfg = ESP8266BoardConfig()
        assert cfg.uart_count == 1

    def test_wifi_enabled(self):
        cfg = ESP8266BoardConfig()
        assert cfg.wifi is True

    def test_no_ble(self):
        cfg = ESP8266BoardConfig()
        assert cfg.ble is False

    def test_nested_pin_map(self):
        cfg = ESP8266BoardConfig()
        assert isinstance(cfg.pin_map, ESP8266PinMap)

    def test_nested_serial(self):
        cfg = ESP8266BoardConfig()
        assert isinstance(cfg.serial, SerialConfig)

    def test_nested_protocol(self):
        cfg = ESP8266BoardConfig()
        assert isinstance(cfg.protocol, WireProtocolConfig)

    def test_nested_comms(self):
        cfg = ESP8266BoardConfig()
        assert isinstance(cfg.comms, CommsConfig)

    def test_is_dataclass(self):
        assert is_dataclass(ESP8266BoardConfig)


# ===================================================================
# create_esp8266_config factory tests
# ===================================================================

class TestCreateESP8266Config:
    def test_returns_esp8266_board_config(self):
        cfg = create_esp8266_config()
        assert isinstance(cfg, ESP8266BoardConfig)

    def test_no_overrides_gives_defaults(self):
        cfg = create_esp8266_config()
        assert cfg.clock_mhz == 80

    def test_override_board_name(self):
        cfg = create_esp8266_config(board_name="ESP8266-SENSOR")
        assert cfg.board_name == "ESP8266-SENSOR"

    def test_override_clock(self):
        cfg = create_esp8266_config(clock_mhz=160)
        assert cfg.clock_mhz == 160

    def test_override_nested_pin_map_dict(self):
        cfg = create_esp8266_config(pin_map={"led": 99})
        assert cfg.pin_map.led == 99

    def test_override_nested_comms_dict(self):
        cfg = create_esp8266_config(comms={"wifi_ssid": "TEST"})
        assert cfg.comms.wifi_ssid == "TEST"

    def test_unknown_override_raises_type_error(self):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            create_esp8266_config(nonexistent_param=42)


# ===================================================================
# WiFiConnectionState tests
# ===================================================================

class TestWiFiConnectionState:
    def test_disconnected(self):
        assert WiFiConnectionState.DISCONNECTED == "disconnected"

    def test_connected(self):
        assert WiFiConnectionState.CONNECTED == "connected"

    def test_reconnecting(self):
        assert WiFiConnectionState.RECONNECTING == "reconnecting"

    def test_ap_mode(self):
        assert WiFiConnectionState.AP_MODE == "ap_mode"

    def test_ap_fallback(self):
        assert WiFiConnectionState.AP_FALLBACK == "ap_fallback"

    def test_all_states(self):
        expected = {
            "disconnected", "scanning", "connecting", "connected",
            "reconnecting", "ap_mode", "connection_failed", "ap_fallback",
        }
        assert {s.value for s in WiFiConnectionState} == expected


# ===================================================================
# WiFiCredentials tests
# ===================================================================

class TestWiFiCredentials:
    def test_empty_credentials_invalid(self):
        cred = WiFiCredentials()
        assert cred.is_valid() is False

    def test_ssid_only_valid(self):
        cred = WiFiCredentials(ssid="MARINE-WIFI")
        assert cred.is_valid() is True

    def test_full_credentials_valid(self):
        cred = WiFiCredentials(ssid="MARINE", password="secret", bssid="AA:BB:CC:DD:EE:FF")
        assert cred.is_valid() is True
        assert cred.bssid == "AA:BB:CC:DD:EE:FF"


# ===================================================================
# WiFiManagerConfig tests
# ===================================================================

class TestWiFiManagerConfig:
    def test_auto_connect_default(self):
        cfg = WiFiManagerConfig()
        assert cfg.auto_connect is True

    def test_auto_reconnect_default(self):
        cfg = WiFiManagerConfig()
        assert cfg.auto_reconnect is True

    def test_max_reconnect_attempts(self):
        cfg = WiFiManagerConfig()
        assert cfg.max_reconnect_attempts == 10

    def test_ap_fallback_default(self):
        cfg = WiFiManagerConfig()
        assert cfg.ap_fallback is True

    def test_ap_ssid_default(self):
        cfg = WiFiManagerConfig()
        assert cfg.ap_ssid == "NEXUS-ESP8266"

    def test_hostname_default(self):
        cfg = WiFiManagerConfig()
        assert cfg.hostname == "nexus-esp8266"

    def test_power_save_default(self):
        cfg = WiFiManagerConfig()
        assert cfg.power_save is False


# ===================================================================
# WiFiManager tests
# ===================================================================

class TestWiFiManager:
    def test_default_state_disconnected(self):
        mgr = WiFiManager()
        assert mgr.state == WiFiConnectionState.DISCONNECTED

    def test_set_credentials(self):
        mgr = WiFiManager()
        mgr.set_credentials("MARINE-WIFI", "password123")
        assert mgr.credentials.ssid == "MARINE-WIFI"
        assert mgr.credentials.password == "password123"
        assert mgr.reconnect_count == 0

    def test_is_connected_false(self):
        mgr = WiFiManager()
        assert mgr.is_connected is False

    def test_is_connected_true(self):
        mgr = WiFiManager()
        mgr.state = WiFiConnectionState.CONNECTED
        assert mgr.is_connected is True

    def test_should_reconnect_no_credentials(self):
        mgr = WiFiManager()
        assert mgr.should_reconnect is False

    def test_should_reconnect_with_credentials(self):
        mgr = WiFiManager()
        mgr.set_credentials("MARINE", "pass")
        assert mgr.should_reconnect is True

    def test_should_reconnect_exceeds_attempts(self):
        mgr = WiFiManager()
        mgr.set_credentials("MARINE", "pass")
        mgr.reconnect_count = 10
        assert mgr.should_reconnect is False

    def test_should_fallback_to_ap(self):
        mgr = WiFiManager()
        mgr.reconnect_count = 10
        assert mgr.should_fallback_to_ap is True

    def test_should_not_fallback_when_active(self):
        mgr = WiFiManager()
        mgr.ap_mode_active = True
        mgr.reconnect_count = 10
        assert mgr.should_fallback_to_ap is False

    def test_signal_quality_excellent(self):
        mgr = WiFiManager()
        mgr.rssi_dbm = -30
        assert mgr.signal_quality_percent() == 100

    def test_signal_quality_poor(self):
        mgr = WiFiManager()
        mgr.rssi_dbm = -80
        assert mgr.signal_quality_percent() == 0

    def test_signal_quality_medium(self):
        mgr = WiFiManager()
        mgr.rssi_dbm = -55
        quality = mgr.signal_quality_percent()
        assert 0 < quality < 100

    def test_get_connection_info_keys(self):
        mgr = WiFiManager()
        info = mgr.get_connection_info()
        assert "state" in info
        assert "connected_ssid" in info
        assert "rssi_dbm" in info
        assert "config" in info


# ===================================================================
# Cross-module / integration tests
# ===================================================================

class TestCrossModule:
    def test_all_configs_are_dataclasses(self):
        assert is_dataclass(ESP8266BoardConfig)
        assert is_dataclass(WiFiManagerConfig)
        assert is_dataclass(WiFiCredentials)

    def test_esp8266_no_ble(self):
        cfg = ESP8266BoardConfig()
        assert cfg.ble is False

    def test_esp8266_single_adc(self):
        cfg = ESP8266BoardConfig()
        assert cfg.adc_channels == 1

    def test_package_imports(self):
        from hardware.esp8266 import (
            ESP8266PinMap,
            SerialConfig,
            WireProtocolConfig,
            CommsConfig,
            ESP8266BoardConfig,
            create_esp8266_config,
            WiFiConnectionState,
            WiFiCredentials,
            WiFiManagerConfig,
            WiFiManager,
        )
        assert ESP8266BoardConfig is not None
