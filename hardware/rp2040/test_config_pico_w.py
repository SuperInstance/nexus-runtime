"""
Tests for NEXUS Pico W Configuration.
"""

import pytest
from hardware.rp2040.config_pico_w import (
    PicoWConfig,
    WiFiConfig,
    BLEConfig,
    MQTTConfig,
    WiFiSecurity,
    BLERole,
    SRAM_TOTAL,
)


class TestWiFiConfig:
    def test_default(self):
        wf = WiFiConfig()
        assert wf.ssid == ""
        assert wf.security == WiFiSecurity.WPA2
        assert wf.timeout_ms == 15000

    def test_open_complete(self):
        wf = WiFiConfig(ssid="OpenNet", security=WiFiSecurity.OPEN)
        assert wf.is_complete() is True

    def test_wpa2_incomplete(self):
        wf = WiFiConfig(ssid="SecureNet", security=WiFiSecurity.WPA2)
        assert wf.is_complete() is False

    def test_wpa2_complete(self):
        wf = WiFiConfig(ssid="SecureNet", password="pass123", security=WiFiSecurity.WPA2)
        assert wf.is_complete() is True

    def test_to_dict(self):
        wf = WiFiConfig(ssid="Test", password="secret")
        d = wf.to_dict()
        assert d["ssid"] == "Test"
        assert d["security"] == "WPA2"
        assert d["password"] == "secret"


class TestBLEConfig:
    def test_default_valid(self):
        ble = BLEConfig()
        assert ble.is_valid() is True

    def test_invalid_mtu_low(self):
        ble = BLEConfig(mtu=10)
        errors = ble.validate()
        assert any("MTU" in e for e in errors)

    def test_invalid_mtu_high(self):
        ble = BLEConfig(mtu=600)
        errors = ble.validate()
        assert any("MTU" in e for e in errors)

    def test_invalid_tx_power(self):
        ble = BLEConfig(tx_power_dbm=20)
        errors = ble.validate()
        assert any("TX power" in e for e in errors)

    def test_invalid_adv_interval(self):
        ble = BLEConfig(advertising_interval_ms=5)
        errors = ble.validate()
        assert any("Advertising interval" in e for e in errors)

    def test_min_gt_max_interval(self):
        ble = BLEConfig(min_connection_interval_ms=100, max_connection_interval_ms=50)
        errors = ble.validate()
        assert any("Min connection" in e for e in errors)

    def test_empty_name(self):
        ble = BLEConfig(device_name="")
        errors = ble.validate()
        assert any("device name" in e.lower() for e in errors)


class TestMQTTConfig:
    def test_default_invalid(self):
        mqtt = MQTTConfig()
        assert mqtt.is_valid() is False

    def test_valid_config(self):
        mqtt = MQTTConfig(broker="192.168.1.1", port=1883)
        assert mqtt.is_valid() is True

    def test_invalid_port(self):
        mqtt = MQTTConfig(broker="test", port=99999)
        errors = mqtt.validate()
        assert any("Invalid port" in e for e in errors)

    def test_invalid_qos(self):
        mqtt = MQTTConfig(broker="test", qos=5)
        errors = mqtt.validate()
        assert any("Invalid QoS" in e for e in errors)

    def test_empty_broker(self):
        mqtt = MQTTConfig()
        errors = mqtt.validate()
        assert any("broker" in e.lower() for e in errors)


class TestPicoWConfig:
    def setup_method(self):
        self.config = PicoWConfig()
        self.config.configure_marine_sensors()

    def test_inherits_from_rp2040(self):
        assert hasattr(self.config, "PIN_SONAR_TRIG")
        assert hasattr(self.config, "clock")

    def test_wireless_constants(self):
        assert "2.4 GHz" in self.config.WIFI_BANDS
        assert "802.11 n" in self.config.WIFI_PROTOCOLS

    def test_configure_wifi(self):
        self.config.configure_wifi("NEXUS_NET", "pass", WiFiSecurity.WPA2)
        assert self.config.wifi.ssid == "NEXUS_NET"
        assert self.config.wifi.is_complete() is True

    def test_configure_ble(self):
        self.config.configure_ble("NEXUS-Boat", BLERole.CENTRAL, tx_power_dbm=3)
        assert self.config.ble.device_name == "NEXUS-Boat"
        assert self.config.ble.role == BLERole.CENTRAL
        assert self.config.ble.tx_power_dbm == 3

    def test_configure_mqtt(self):
        self.config.configure_mqtt("mqtt.nexus.io", 8883)
        assert self.config.mqtt.broker == "mqtt.nexus.io"
        assert self.config.mqtt.port == 8883

    def test_set_static_ip(self):
        self.config.set_static_ip("10.0.0.100", "10.0.0.1", "8.8.8.8")
        assert self.config.wifi.static_ip == "10.0.0.100"
        assert self.config.wifi.static_gateway == "10.0.0.1"

    def test_validate_wireless_clean(self):
        errors = self.config.validate_wireless()
        assert len(errors) == 0

    def test_validate_memory_overflow(self):
        # Deliberately add too much memory
        self.config.memory.add_region("overflow", 0x20000000, SRAM_TOTAL + 100000)
        errors = self.config.validate_wireless()
        assert any("overflow" in e.lower() for e in errors)

    def test_summary_includes_wireless(self):
        s = self.config.summary()
        assert s["variant"] == "Pico W"
        assert s["wireless_chip"] == "CYW43439"
        assert "wifi_ssid" in s
        assert "ble_device_name" in s

    def test_wireless_memory_reserved(self):
        lwip = self.config.memory.region_by_name("lwip_heap")
        assert lwip is not None
        assert lwip.size_bytes == 32768

    def test_default_ble_valid(self):
        assert self.config.ble.is_valid()
