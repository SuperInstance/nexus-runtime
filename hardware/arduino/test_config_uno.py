"""
Unit tests for hardware.arduino.config_uno
"""

import pytest

from hardware.arduino.config_uno import (
    BoardConfig,
    SerialConfig,
    WireProtocolConfig,
    PinMapping,
    UnoConfig,
    get_uno_config,
)


# ---------------------------------------------------------------------------
# BoardConfig
# ---------------------------------------------------------------------------

class TestBoardConfig:
    def test_board_name(self):
        cfg = BoardConfig()
        assert cfg.board_name == "Arduino Uno R3"

    def test_cpu(self):
        cfg = BoardConfig()
        assert cfg.cpu == "ATmega328P"

    def test_clock_hz(self):
        cfg = BoardConfig()
        assert cfg.clock_hz == 16_000_000

    def test_memory(self):
        cfg = BoardConfig()
        assert cfg.flash_kb == 32
        assert cfg.ram_kb == 2
        assert cfg.eeprom_kb == 1

    def test_peripheral_counts(self):
        cfg = BoardConfig()
        assert cfg.uart_count == 1
        assert cfg.spi_count == 1
        assert cfg.i2c_count == 1

    def test_gpio_counts(self):
        cfg = BoardConfig()
        assert cfg.gpio_count == 14
        assert cfg.adc_count == 6
        assert cfg.pwm_pins == 6

    def test_is_frozen(self):
        cfg = BoardConfig()
        with pytest.raises(AttributeError):
            cfg.board_name = "modified"

    def test_operating_voltage(self):
        cfg = BoardConfig()
        assert cfg.operating_voltage_v == 5.0


# ---------------------------------------------------------------------------
# SerialConfig
# ---------------------------------------------------------------------------

class TestSerialConfig:
    def test_defaults_115200_8n1(self):
        cfg = SerialConfig()
        assert cfg.baud_rate == 115_200
        assert cfg.data_bits == 8
        assert cfg.parity == "N"
        assert cfg.stop_bits == 1

    def test_custom_baud(self):
        cfg = SerialConfig(baud_rate=9600)
        assert cfg.baud_rate == 9600

    def test_is_frozen(self):
        cfg = SerialConfig()
        with pytest.raises(AttributeError):
            cfg.baud_rate = 0


# ---------------------------------------------------------------------------
# WireProtocolConfig
# ---------------------------------------------------------------------------

class TestWireProtocolConfig:
    def test_preamble(self):
        cfg = WireProtocolConfig()
        assert cfg.frame_preamble == b"\xaa\x55"

    def test_max_frame_size(self):
        cfg = WireProtocolConfig()
        assert cfg.max_frame_size == 256

    def test_heartbeat_interval(self):
        cfg = WireProtocolConfig()
        assert cfg.heartbeat_interval_ms == 500

    def test_offsets(self):
        cfg = WireProtocolConfig()
        assert cfg.message_id_offset == 2
        assert cfg.payload_offset == 3


# ---------------------------------------------------------------------------
# PinMapping
# ---------------------------------------------------------------------------

class TestPinMapping:
    def test_gps_pins(self):
        pm = PinMapping()
        assert pm.GPS_TX == 0
        assert pm.GPS_RX == 1

    def test_imu_pins(self):
        pm = PinMapping()
        assert pm.IMU_SDA == 18  # A4
        assert pm.IMU_SCL == 19  # A5

    def test_sonar_pins(self):
        pm = PinMapping()
        assert pm.SONAR_TRIG == 7
        assert pm.SONAR_ECHO == 8

    def test_temp_pressure_pins(self):
        pm = PinMapping()
        assert pm.TEMP_PIN == 0      # A0
        assert pm.PRESSURE_PIN == 1  # A1

    def test_servo_and_thruster_pins_are_pwm_capable(self):
        pm = PinMapping()
        # PWM pins on Uno: 3, 5, 6, 9, 10, 11
        assert pm.SERVO_PINS == (9,)
        assert pm.THRUSTER_PWM == 10

    def test_led_pin(self):
        pm = PinMapping()
        assert pm.LED_PIN == 13

    def test_is_frozen(self):
        pm = PinMapping()
        with pytest.raises(AttributeError):
            pm.GPS_TX = 99


# ---------------------------------------------------------------------------
# UnoConfig composite
# ---------------------------------------------------------------------------

class TestUnoConfig:
    def test_composite_defaults(self):
        cfg = UnoConfig()
        assert isinstance(cfg.board_config, BoardConfig)
        assert isinstance(cfg.serial_config, SerialConfig)
        assert isinstance(cfg.wire_protocol, WireProtocolConfig)
        assert isinstance(cfg.pin_mapping, PinMapping)

    def test_nested_access(self):
        cfg = UnoConfig()
        assert cfg.board_config.clock_hz == 16_000_000
        assert cfg.serial_config.baud_rate == 115_200
        assert cfg.pin_mapping.SONAR_TRIG == 7


# ---------------------------------------------------------------------------
# get_uno_config factory
# ---------------------------------------------------------------------------

class TestGetUnoConfig:
    def test_returns_uno_config(self):
        cfg = get_uno_config()
        assert isinstance(cfg, UnoConfig)

    def test_baud_rate_override(self):
        cfg = get_uno_config(baud_rate=9600)
        assert cfg.serial_config.baud_rate == 9600

    def test_pin_override(self):
        cfg = get_uno_config(GPS_TX=4, SONAR_TRIG=5)
        assert cfg.pin_mapping.GPS_TX == 4
        assert cfg.pin_mapping.SONAR_TRIG == 5

    def test_board_override(self):
        cfg = get_uno_config(board_name="Custom Uno")
        assert cfg.board_config.board_name == "Custom Uno"

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown config keys"):
            get_uno_config(nonexistent_key=42)

    def test_heartbeat_override(self):
        cfg = get_uno_config(heartbeat_interval_ms=250)
        assert cfg.wire_protocol.heartbeat_interval_ms == 250
