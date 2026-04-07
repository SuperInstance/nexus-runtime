"""
Unit tests for hardware.arduino.config_mega
"""

import pytest

from hardware.arduino.config_mega import (
    BoardConfig,
    SerialConfig,
    WireProtocolConfig,
    PinMapping,
    MegaConfig,
    get_mega_config,
)


# ---------------------------------------------------------------------------
# BoardConfig
# ---------------------------------------------------------------------------

class TestMegaBoardConfig:
    def test_board_name(self):
        cfg = BoardConfig()
        assert cfg.board_name == "Arduino Mega 2560"

    def test_cpu(self):
        cfg = BoardConfig()
        assert cfg.cpu == "ATmega2560"

    def test_memory(self):
        cfg = BoardConfig()
        assert cfg.flash_kb == 256
        assert cfg.ram_kb == 8
        assert cfg.eeprom_kb == 4

    def test_gpio_counts(self):
        cfg = BoardConfig()
        assert cfg.gpio_count == 54
        assert cfg.adc_count == 16
        assert cfg.pwm_pins == 15

    def test_four_uarts(self):
        cfg = BoardConfig()
        assert cfg.uart_count == 4

    def test_is_frozen(self):
        cfg = BoardConfig()
        with pytest.raises(AttributeError):
            cfg.gpio_count = 0

    def test_clock_and_voltage(self):
        cfg = BoardConfig()
        assert cfg.clock_hz == 16_000_000
        assert cfg.operating_voltage_v == 5.0


# ---------------------------------------------------------------------------
# SerialConfig
# ---------------------------------------------------------------------------

class TestMegaSerialConfig:
    def test_defaults(self):
        cfg = SerialConfig()
        assert cfg.baud_rate == 115_200
        assert cfg.data_bits == 8
        assert cfg.parity == "N"
        assert cfg.stop_bits == 1

    def test_default_port_is_serial1(self):
        cfg = SerialConfig()
        assert cfg.serial_port == "Serial1"

    def test_uart_pin_assignments(self):
        cfg = SerialConfig()
        assert cfg.UART1_PINS == (19, 18)
        assert cfg.UART2_PINS == (17, 16)
        assert cfg.UART3_PINS == (15, 14)
        assert cfg.UART0_PINS == (0, 1)

    def test_custom_port(self):
        cfg = SerialConfig(serial_port="Serial3")
        assert cfg.serial_port == "Serial3"

    def test_is_frozen(self):
        cfg = SerialConfig()
        with pytest.raises(AttributeError):
            cfg.baud_rate = 0


# ---------------------------------------------------------------------------
# PinMapping
# ---------------------------------------------------------------------------

class TestMegaPinMapping:
    def test_gps_on_serial1(self):
        pm = PinMapping()
        assert pm.GPS_TX == 19   # RX1
        assert pm.GPS_RX == 18   # TX1

    def test_imu_on_i2c(self):
        pm = PinMapping()
        assert pm.IMU_SDA == 20
        assert pm.IMU_SCL == 21

    def test_triple_sonar_array(self):
        pm = PinMapping()
        assert pm.SONAR1_TRIG == 22
        assert pm.SONAR1_ECHO == 23
        assert pm.SONAR2_TRIG == 24
        assert pm.SONAR2_ECHO == 25
        assert pm.SONAR3_TRIG == 26
        assert pm.SONAR3_ECHO == 27

    def test_multiple_temp_sensors(self):
        pm = PinMapping()
        assert pm.TEMP_PIN_1 == 0
        assert pm.TEMP_PIN_2 == 1
        assert pm.TEMP_PIN_3 == 2

    def test_pressure_pin(self):
        pm = PinMapping()
        assert pm.PRESSURE_PIN == 3

    def test_servo_pins_tuple(self):
        pm = PinMapping()
        assert pm.SERVO_PINS == (44, 45, 46)

    def test_thruster_pins(self):
        pm = PinMapping()
        assert pm.THRUSTER_PORT == 44
        assert pm.THRUSTER_STARBOARD == 45
        assert pm.THRUSTER_VERTICAL == 46

    def test_companion_link_on_serial2(self):
        pm = PinMapping()
        assert pm.COMPANION_TX == 16  # TX2
        assert pm.COMPANION_RX == 17  # RX2

    def test_aux_on_serial3(self):
        pm = PinMapping()
        assert pm.AUX_TX == 14
        assert pm.AUX_RX == 15

    def test_is_frozen(self):
        pm = PinMapping()
        with pytest.raises(AttributeError):
            pm.GPS_TX = 0


# ---------------------------------------------------------------------------
# MegaConfig composite
# ---------------------------------------------------------------------------

class TestMegaConfig:
    def test_composite_defaults(self):
        cfg = MegaConfig()
        assert isinstance(cfg.board_config, BoardConfig)
        assert isinstance(cfg.serial_config, SerialConfig)
        assert isinstance(cfg.wire_protocol, WireProtocolConfig)
        assert isinstance(cfg.pin_mapping, PinMapping)

    def test_nested_access(self):
        cfg = MegaConfig()
        assert cfg.board_config.gpio_count == 54
        assert cfg.serial_config.serial_port == "Serial1"
        assert cfg.pin_mapping.SONAR2_TRIG == 24


# ---------------------------------------------------------------------------
# get_mega_config factory
# ---------------------------------------------------------------------------

class TestGetMegaConfig:
    def test_returns_mega_config(self):
        cfg = get_mega_config()
        assert isinstance(cfg, MegaConfig)

    def test_baud_rate_override(self):
        cfg = get_mega_config(baud_rate=57600)
        assert cfg.serial_config.baud_rate == 57600

    def test_pin_override(self):
        cfg = get_mega_config(GPS_TX=10, SONAR1_TRIG=50)
        assert cfg.pin_mapping.GPS_TX == 10
        assert cfg.pin_mapping.SONAR1_TRIG == 50

    def test_serial_port_override(self):
        cfg = get_mega_config(serial_port="Serial2")
        assert cfg.serial_config.serial_port == "Serial2"

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown config keys"):
            get_mega_config(fake_param=123)
