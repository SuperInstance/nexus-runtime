"""
Unit tests for hardware.arduino.config_nano33_iot
"""

import pytest

from hardware.arduino.config_nano33_iot import (
    BoardConfig,
    SerialConfig,
    WireProtocolConfig,
    WiFiConfig,
    NexusMeshConfig,
    PinMapping,
    InterfacePins,
    Nano33IoTConfig,
    get_nano33_iot_config,
)


# ---------------------------------------------------------------------------
# BoardConfig
# ---------------------------------------------------------------------------

class TestNano33IoTBoardConfig:
    def test_board_name(self):
        cfg = BoardConfig()
        assert cfg.board_name == "Arduino Nano 33 IoT"

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
        assert cfg.gpio_count == 20
        assert cfg.adc_count == 8
        assert cfg.pwm_pins == 4

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

class TestNano33IoTSerialConfig:
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
        assert cfg.UART_TX == 1
        assert cfg.UART_RX == 0

    def test_is_frozen(self):
        cfg = SerialConfig()
        with pytest.raises(AttributeError):
            cfg.baud_rate = 0


# ---------------------------------------------------------------------------
# WireProtocolConfig
# ---------------------------------------------------------------------------

class TestNano33IoTWireProtocolConfig:
    def test_preamble(self):
        cfg = WireProtocolConfig()
        assert cfg.frame_preamble == b"\xaa\x55"

    def test_max_frame_size(self):
        cfg = WireProtocolConfig()
        assert cfg.max_frame_size == 256

    def test_heartbeat_interval(self):
        cfg = WireProtocolConfig()
        assert cfg.heartbeat_interval_ms == 500


# ---------------------------------------------------------------------------
# WiFiConfig
# ---------------------------------------------------------------------------

class TestNano33IoTWiFiConfig:
    def test_wifi_enabled(self):
        cfg = WiFiConfig()
        assert cfg.wifi_enabled is True

    def test_ble_enabled(self):
        cfg = WiFiConfig()
        assert cfg.ble_enabled is True

    def test_default_ssid_empty(self):
        cfg = WiFiConfig()
        assert cfg.wifi_ssid == ""

    def test_low_power_mode_default(self):
        cfg = WiFiConfig()
        assert cfg.low_power_mode is True

    def test_wifi_timeout(self):
        cfg = WiFiConfig()
        assert cfg.wifi_timeout_ms == 15000

    def test_max_reconnect(self):
        cfg = WiFiConfig()
        assert cfg.max_reconnect_attempts == 10

    def test_ap_ssid_default(self):
        cfg = WiFiConfig()
        assert cfg.ap_ssid == "NEXUS-NANO33"

    def test_is_frozen(self):
        cfg = WiFiConfig()
        with pytest.raises(AttributeError):
            cfg.wifi_ssid = "test"


# ---------------------------------------------------------------------------
# NexusMeshConfig
# ---------------------------------------------------------------------------

class TestNexusMeshConfig:
    def test_mesh_enabled(self):
        cfg = NexusMeshConfig()
        assert cfg.mesh_enabled is True

    def test_default_role_is_end_device(self):
        cfg = NexusMeshConfig()
        assert cfg.node_role == "end_device"

    def test_default_node_id_empty(self):
        cfg = NexusMeshConfig()
        assert cfg.node_id == ""

    def test_telemetry_topic(self):
        cfg = NexusMeshConfig()
        assert cfg.telemetry_topic == "nexus/telemetry"

    def test_command_topic(self):
        cfg = NexusMeshConfig()
        assert cfg.command_topic == "nexus/command"

    def test_heartbeat_topic(self):
        cfg = NexusMeshConfig()
        assert cfg.heartbeat_topic == "nexus/heartbeat"

    def test_telemetry_interval(self):
        cfg = NexusMeshConfig()
        assert cfg.telemetry_interval_ms == 1000

    def test_status_report_interval(self):
        cfg = NexusMeshConfig()
        assert cfg.status_report_interval_s == 60

    def test_is_frozen(self):
        cfg = NexusMeshConfig()
        with pytest.raises(AttributeError):
            cfg.node_role = "coordinator"


# ---------------------------------------------------------------------------
# PinMapping
# ---------------------------------------------------------------------------

class TestNano33IoTPinMapping:
    def test_gps_pins(self):
        pm = PinMapping()
        assert pm.GPS_TX == 0
        assert pm.GPS_RX == 1

    def test_imu_pins(self):
        pm = PinMapping()
        assert pm.IMU_SDA == 18
        assert pm.IMU_SCL == 19

    def test_sonar_pins(self):
        pm = PinMapping()
        assert pm.SONAR_TRIG == 7
        assert pm.SONAR_ECHO == 8

    def test_temp_pressure_pins(self):
        pm = PinMapping()
        assert pm.TEMP_PIN == 0
        assert pm.PRESSURE_PIN == 1

    def test_extra_adc_pins(self):
        pm = PinMapping()
        assert pm.EXTRA_ADC_1 == 2
        assert pm.EXTRA_ADC_2 == 3

    def test_servo_pin(self):
        pm = PinMapping()
        assert pm.SERVO_PINS == (9,)

    def test_thruster_pin(self):
        pm = PinMapping()
        assert pm.THRUSTER_PWM == 3

    def test_led_pin(self):
        pm = PinMapping()
        assert pm.LED_PIN == 13
        assert pm.LED_BUILTIN == 13

    def test_rgb_led_pins(self):
        pm = PinMapping()
        assert pm.LED_R == 26
        assert pm.LED_G == 25
        assert pm.LED_B == 27

    def test_builtin_imu_pins(self):
        pm = PinMapping()
        assert pm.BUILTIN_IMU_SDA == 18
        assert pm.BUILTIN_IMU_SCL == 19

    def test_is_frozen(self):
        pm = PinMapping()
        with pytest.raises(AttributeError):
            pm.GPS_TX = 99


# ---------------------------------------------------------------------------
# InterfacePins
# ---------------------------------------------------------------------------

class TestNano33IoTInterfacePins:
    def test_uart_pins(self):
        ip = InterfacePins()
        assert ip.UART_TX == 1
        assert ip.UART_RX == 0

    def test_spi_pins(self):
        ip = InterfacePins()
        assert ip.SPI_SS == 10
        assert ip.SPI_MOSI == 11
        assert ip.SPI_MISO == 12
        assert ip.SPI_SCK == 13

    def test_i2c_pins(self):
        ip = InterfacePins()
        assert ip.I2C_SDA == 18
        assert ip.I2C_SCL == 19

    def test_pwm_pins_count(self):
        ip = InterfacePins()
        assert ip.PWM_PINS == (2, 3, 5, 9)

    def test_is_frozen(self):
        ip = InterfacePins()
        with pytest.raises(AttributeError):
            ip.UART_TX = 99


# ---------------------------------------------------------------------------
# Nano33IoTConfig composite
# ---------------------------------------------------------------------------

class TestNano33IoTConfig:
    def test_composite_defaults(self):
        cfg = Nano33IoTConfig()
        assert isinstance(cfg.board_config, BoardConfig)
        assert isinstance(cfg.serial_config, SerialConfig)
        assert isinstance(cfg.wire_protocol, WireProtocolConfig)
        assert isinstance(cfg.wifi_config, WiFiConfig)
        assert isinstance(cfg.nexus_mesh, NexusMeshConfig)
        assert isinstance(cfg.pin_mapping, PinMapping)
        assert isinstance(cfg.interface_pins, InterfacePins)

    def test_nested_access(self):
        cfg = Nano33IoTConfig()
        assert cfg.board_config.clock_hz == 48_000_000
        assert cfg.serial_config.serial_port == "Serial1"
        assert cfg.wifi_config.low_power_mode is True
        assert cfg.nexus_mesh.mesh_enabled is True


# ---------------------------------------------------------------------------
# get_nano33_iot_config factory
# ---------------------------------------------------------------------------

class TestGetNano33IoTConfig:
    def test_returns_nano33_iot_config(self):
        cfg = get_nano33_iot_config()
        assert isinstance(cfg, Nano33IoTConfig)

    def test_baud_rate_override(self):
        cfg = get_nano33_iot_config(baud_rate=57600)
        assert cfg.serial_config.baud_rate == 57600

    def test_wifi_ssid_override(self):
        cfg = get_nano33_iot_config(wifi_ssid="MARINE-WIFI")
        assert cfg.wifi_config.wifi_ssid == "MARINE-WIFI"

    def test_mesh_role_override(self):
        cfg = get_nano33_iot_config(node_role="router")
        assert cfg.nexus_mesh.node_role == "router"

    def test_pin_override(self):
        cfg = get_nano33_iot_config(GPS_TX=4, SONAR_TRIG=5)
        assert cfg.pin_mapping.GPS_TX == 4
        assert cfg.pin_mapping.SONAR_TRIG == 5

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown config keys"):
            get_nano33_iot_config(nonexistent_key=42)
