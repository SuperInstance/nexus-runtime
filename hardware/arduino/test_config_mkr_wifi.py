"""
Unit tests for hardware.arduino.config_mkr_wifi
"""

import pytest

from hardware.arduino.config_mkr_wifi import (
    BoardConfig,
    SerialConfig,
    WireProtocolConfig,
    WiFiConfig,
    NexusBridgeConfig,
    PinMapping,
    InterfacePins,
    MKRWiFiConfig,
    get_mkr_wifi_config,
)


# ---------------------------------------------------------------------------
# BoardConfig
# ---------------------------------------------------------------------------

class TestMKRBoardConfig:
    def test_board_name(self):
        cfg = BoardConfig()
        assert cfg.board_name == "Arduino MKR WiFi 1010"

    def test_cpu(self):
        cfg = BoardConfig()
        assert cfg.cpu == "SAMD21G18A"

    def test_cpu_arch(self):
        cfg = BoardConfig()
        assert cfg.cpu_arch == "ARM Cortex-M0+"

    def test_clock_hz(self):
        cfg = BoardConfig()
        assert cfg.clock_hz == 48_000_000

    def test_memory(self):
        cfg = BoardConfig()
        assert cfg.flash_kb == 256
        assert cfg.ram_kb == 32
        assert cfg.eeprom_kb == 0

    def test_peripheral_counts(self):
        cfg = BoardConfig()
        assert cfg.uart_count == 1
        assert cfg.spi_count == 1
        assert cfg.i2c_count == 1

    def test_gpio_counts(self):
        cfg = BoardConfig()
        assert cfg.gpio_count == 22
        assert cfg.adc_count == 7
        assert cfg.pwm_pins == 12

    def test_adc_resolution(self):
        cfg = BoardConfig()
        assert cfg.adc_resolution_bits == 12

    def test_operating_voltage(self):
        cfg = BoardConfig()
        assert cfg.operating_voltage_v == 3.3

    def test_is_frozen(self):
        cfg = BoardConfig()
        with pytest.raises(AttributeError):
            cfg.board_name = "modified"


# ---------------------------------------------------------------------------
# SerialConfig
# ---------------------------------------------------------------------------

class TestMKRSerialConfig:
    def test_defaults(self):
        cfg = SerialConfig()
        assert cfg.baud_rate == 115_200
        assert cfg.data_bits == 8
        assert cfg.parity == "N"
        assert cfg.stop_bits == 1

    def test_default_port_is_serial1(self):
        cfg = SerialConfig()
        assert cfg.serial_port == "Serial1"

    def test_uart_pins(self):
        cfg = SerialConfig()
        assert cfg.UART_TX == 14
        assert cfg.UART_RX == 13

    def test_custom_port(self):
        cfg = SerialConfig(serial_port="Serial")
        assert cfg.serial_port == "Serial"

    def test_is_frozen(self):
        cfg = SerialConfig()
        with pytest.raises(AttributeError):
            cfg.baud_rate = 0


# ---------------------------------------------------------------------------
# WireProtocolConfig
# ---------------------------------------------------------------------------

class TestMKRWireProtocolConfig:
    def test_preamble(self):
        cfg = WireProtocolConfig()
        assert cfg.frame_preamble == b"\xaa\x55"

    def test_max_frame_size_larger_than_uno(self):
        cfg = WireProtocolConfig()
        assert cfg.max_frame_size == 512

    def test_heartbeat_interval(self):
        cfg = WireProtocolConfig()
        assert cfg.heartbeat_interval_ms == 500


# ---------------------------------------------------------------------------
# WiFiConfig
# ---------------------------------------------------------------------------

class TestWiFiConfig:
    def test_wifi_enabled(self):
        cfg = WiFiConfig()
        assert cfg.wifi_enabled is True

    def test_ble_enabled(self):
        cfg = WiFiConfig()
        assert cfg.ble_enabled is True

    def test_default_ssid_empty(self):
        cfg = WiFiConfig()
        assert cfg.wifi_ssid == ""

    def test_default_mqtt_port(self):
        cfg = WiFiConfig()
        assert cfg.mqtt_port == 1883

    def test_default_ap_mode_off(self):
        cfg = WiFiConfig()
        assert cfg.ap_mode is False

    def test_ap_ssid_default(self):
        cfg = WiFiConfig()
        assert cfg.ap_ssid == "NEXUS-MKR"

    def test_is_frozen(self):
        cfg = WiFiConfig()
        with pytest.raises(AttributeError):
            cfg.wifi_ssid = "test"


# ---------------------------------------------------------------------------
# NexusBridgeConfig
# ---------------------------------------------------------------------------

class TestNexusBridgeConfig:
    def test_bridge_enabled(self):
        cfg = NexusBridgeConfig()
        assert cfg.bridge_enabled is True

    def test_bridge_protocol_mqtt(self):
        cfg = NexusBridgeConfig()
        assert cfg.bridge_protocol == "MQTT"

    def test_telemetry_topic(self):
        cfg = NexusBridgeConfig()
        assert cfg.telemetry_topic == "nexus/telemetry"

    def test_command_topic(self):
        cfg = NexusBridgeConfig()
        assert cfg.command_topic == "nexus/command"

    def test_reconnect_settings(self):
        cfg = NexusBridgeConfig()
        assert cfg.max_reconnect_attempts == 5
        assert cfg.reconnect_interval_ms == 5000

    def test_is_frozen(self):
        cfg = NexusBridgeConfig()
        with pytest.raises(AttributeError):
            cfg.bridge_protocol = "WS"


# ---------------------------------------------------------------------------
# PinMapping
# ---------------------------------------------------------------------------

class TestMKRPinMapping:
    def test_gps_on_serial1(self):
        pm = PinMapping()
        assert pm.GPS_TX == 13
        assert pm.GPS_RX == 14

    def test_imu_on_i2c(self):
        pm = PinMapping()
        assert pm.IMU_SDA == 11
        assert pm.IMU_SCL == 12

    def test_sonar_pins(self):
        pm = PinMapping()
        assert pm.SONAR_TRIG == 6
        assert pm.SONAR_ECHO == 7

    def test_temp_pressure_pins(self):
        pm = PinMapping()
        assert pm.TEMP_PIN == 0
        assert pm.PRESSURE_PIN == 1

    def test_servo_pins(self):
        pm = PinMapping()
        assert pm.SERVO_PINS == (4, 5)

    def test_thruster_pin(self):
        pm = PinMapping()
        assert pm.THRUSTER_PWM == 4

    def test_led_pin(self):
        pm = PinMapping()
        assert pm.LED_PIN == 6

    def test_spi_pins(self):
        pm = PinMapping()
        assert pm.SPI_SS == 5
        assert pm.SPI_MOSI == 8
        assert pm.SPI_MISO == 10
        assert pm.SPI_SCK == 9

    def test_is_frozen(self):
        pm = PinMapping()
        with pytest.raises(AttributeError):
            pm.GPS_TX = 0


# ---------------------------------------------------------------------------
# InterfacePins
# ---------------------------------------------------------------------------

class TestMKRInterfacePins:
    def test_uart_pins(self):
        ip = InterfacePins()
        assert ip.UART_TX == 14
        assert ip.UART_RX == 13

    def test_spi_pins(self):
        ip = InterfacePins()
        assert ip.SPI_SS == 5
        assert ip.SPI_MOSI == 8
        assert ip.SPI_MISO == 10
        assert ip.SPI_SCK == 9

    def test_i2c_pins(self):
        ip = InterfacePins()
        assert ip.I2C_SDA == 11
        assert ip.I2C_SCL == 12

    def test_pwm_pins_count(self):
        ip = InterfacePins()
        assert len(ip.PWM_PINS) == 12

    def test_is_frozen(self):
        ip = InterfacePins()
        with pytest.raises(AttributeError):
            ip.UART_TX = 0


# ---------------------------------------------------------------------------
# MKRWiFiConfig composite
# ---------------------------------------------------------------------------

class TestMKRWiFiConfig:
    def test_composite_defaults(self):
        cfg = MKRWiFiConfig()
        assert isinstance(cfg.board_config, BoardConfig)
        assert isinstance(cfg.serial_config, SerialConfig)
        assert isinstance(cfg.wire_protocol, WireProtocolConfig)
        assert isinstance(cfg.wifi_config, WiFiConfig)
        assert isinstance(cfg.nexus_bridge, NexusBridgeConfig)
        assert isinstance(cfg.pin_mapping, PinMapping)
        assert isinstance(cfg.interface_pins, InterfacePins)

    def test_nested_access(self):
        cfg = MKRWiFiConfig()
        assert cfg.board_config.clock_hz == 48_000_000
        assert cfg.serial_config.serial_port == "Serial1"
        assert cfg.wifi_config.wifi_enabled is True
        assert cfg.nexus_bridge.bridge_protocol == "MQTT"


# ---------------------------------------------------------------------------
# get_mkr_wifi_config factory
# ---------------------------------------------------------------------------

class TestGetMKRWiFiConfig:
    def test_returns_mkr_wifi_config(self):
        cfg = get_mkr_wifi_config()
        assert isinstance(cfg, MKRWiFiConfig)

    def test_baud_rate_override(self):
        cfg = get_mkr_wifi_config(baud_rate=57600)
        assert cfg.serial_config.baud_rate == 57600

    def test_wifi_ssid_override(self):
        cfg = get_mkr_wifi_config(wifi_ssid="MARINE-WIFI")
        assert cfg.wifi_config.wifi_ssid == "MARINE-WIFI"

    def test_pin_override(self):
        cfg = get_mkr_wifi_config(GPS_TX=10, SONAR_TRIG=3)
        assert cfg.pin_mapping.GPS_TX == 10
        assert cfg.pin_mapping.SONAR_TRIG == 3

    def test_bridge_protocol_override(self):
        cfg = get_mkr_wifi_config(bridge_protocol="WebSocket")
        assert cfg.nexus_bridge.bridge_protocol == "WebSocket"

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown config keys"):
            get_mkr_wifi_config(fake_param=123)
