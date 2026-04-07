"""
Unit tests for hardware.arduino.config_due
"""

import pytest

from hardware.arduino.config_due import (
    BoardConfig,
    SerialConfig,
    WireProtocolConfig,
    NexusEdgeConfig,
    PinMapping,
    InterfacePins,
    DueConfig,
    get_due_config,
)


# ---------------------------------------------------------------------------
# BoardConfig
# ---------------------------------------------------------------------------

class TestDueBoardConfig:
    def test_board_name(self):
        cfg = BoardConfig()
        assert cfg.board_name == "Arduino Due"

    def test_cpu(self):
        cfg = BoardConfig()
        assert cfg.cpu == "AT91SAM3X8E"

    def test_cpu_arch(self):
        cfg = BoardConfig()
        assert cfg.cpu_arch == "ARM Cortex-M3"

    def test_clock_hz(self):
        cfg = BoardConfig()
        assert cfg.clock_hz == 84_000_000

    def test_memory(self):
        cfg = BoardConfig()
        assert cfg.flash_kb == 512
        assert cfg.ram_kb == 96
        assert cfg.eeprom_kb == 0

    def test_peripheral_counts(self):
        cfg = BoardConfig()
        assert cfg.uart_count == 4
        assert cfg.spi_count == 2
        assert cfg.i2c_count == 2

    def test_gpio_counts(self):
        cfg = BoardConfig()
        assert cfg.gpio_count == 54
        assert cfg.adc_count == 12
        assert cfg.pwm_pins == 12

    def test_adc_resolution(self):
        cfg = BoardConfig()
        assert cfg.adc_resolution_bits == 12

    def test_has_dac(self):
        cfg = BoardConfig()
        assert cfg.dac_count == 2

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

class TestDueSerialConfig:
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
        assert cfg.UART0_PINS == (0, 1)
        assert cfg.UART1_PINS == (19, 18)
        assert cfg.UART2_PINS == (17, 16)
        assert cfg.UART3_PINS == (15, 14)

    def test_custom_port(self):
        cfg = SerialConfig(serial_port="Serial3")
        assert cfg.serial_port == "Serial3"

    def test_is_frozen(self):
        cfg = SerialConfig()
        with pytest.raises(AttributeError):
            cfg.baud_rate = 0


# ---------------------------------------------------------------------------
# WireProtocolConfig
# ---------------------------------------------------------------------------

class TestDueWireProtocolConfig:
    def test_preamble(self):
        cfg = WireProtocolConfig()
        assert cfg.frame_preamble == b"\xaa\x55"

    def test_max_frame_size(self):
        cfg = WireProtocolConfig()
        assert cfg.max_frame_size == 512

    def test_heartbeat_faster_than_uno(self):
        cfg = WireProtocolConfig()
        assert cfg.heartbeat_interval_ms == 250


# ---------------------------------------------------------------------------
# NexusEdgeConfig
# ---------------------------------------------------------------------------

class TestNexusEdgeConfig:
    def test_edge_processing_enabled(self):
        cfg = NexusEdgeConfig()
        assert cfg.edge_processing_enabled is True

    def test_sensor_fusion_rate(self):
        cfg = NexusEdgeConfig()
        assert cfg.sensor_fusion_rate_hz == 50.0

    def test_pid_loop_rate(self):
        cfg = NexusEdgeConfig()
        assert cfg.pid_loop_rate_hz == 100.0

    def test_max_sensor_streams(self):
        cfg = NexusEdgeConfig()
        assert cfg.max_sensor_streams == 8

    def test_watchdog_timeout(self):
        cfg = NexusEdgeConfig()
        assert cfg.watchdog_timeout_ms == 1000

    def test_buffer_size(self):
        cfg = NexusEdgeConfig()
        assert cfg.buffer_size_frames == 64

    def test_is_frozen(self):
        cfg = NexusEdgeConfig()
        with pytest.raises(AttributeError):
            cfg.sensor_fusion_rate_hz = 200


# ---------------------------------------------------------------------------
# PinMapping
# ---------------------------------------------------------------------------

class TestDuePinMapping:
    def test_gps_on_serial1(self):
        pm = PinMapping()
        assert pm.GPS_TX == 19
        assert pm.GPS_RX == 18

    def test_dual_imu(self):
        pm = PinMapping()
        assert pm.IMU_SDA == 20
        assert pm.IMU_SCL == 21
        assert pm.IMU2_SDA == 70
        assert pm.IMU2_SCL == 71

    def test_quad_sonar_array(self):
        pm = PinMapping()
        assert pm.SONAR1_TRIG == 22
        assert pm.SONAR1_ECHO == 23
        assert pm.SONAR2_TRIG == 24
        assert pm.SONAR2_ECHO == 25
        assert pm.SONAR3_TRIG == 26
        assert pm.SONAR3_ECHO == 27
        assert pm.SONAR4_TRIG == 28
        assert pm.SONAR4_ECHO == 29

    def test_four_temp_sensors(self):
        pm = PinMapping()
        assert pm.TEMP_PIN_1 == 0
        assert pm.TEMP_PIN_2 == 1
        assert pm.TEMP_PIN_3 == 2
        assert pm.TEMP_PIN_4 == 3

    def test_pressure_pin(self):
        pm = PinMapping()
        assert pm.PRESSURE_PIN == 4

    def test_dac_pins(self):
        pm = PinMapping()
        assert pm.DAC0_PIN == 66
        assert pm.DAC1_PIN == 67

    def test_four_thrusters(self):
        pm = PinMapping()
        assert pm.THRUSTER_PORT == 34
        assert pm.THRUSTER_STARBOARD == 35
        assert pm.THRUSTER_VERTICAL == 36
        assert pm.THRUSTER_LATERAL == 37

    def test_companion_link_on_serial2(self):
        pm = PinMapping()
        assert pm.COMPANION_TX == 16
        assert pm.COMPANION_RX == 17

    def test_aux_on_serial3(self):
        pm = PinMapping()
        assert pm.AUX_TX == 14
        assert pm.AUX_RX == 15

    def test_relay_pins(self):
        pm = PinMapping()
        assert pm.RELAY_PIN_1 == 30
        assert pm.RELAY_PIN_2 == 31

    def test_water_quality_pin(self):
        pm = PinMapping()
        assert pm.WATER_QUALITY_PIN == 5

    def test_is_frozen(self):
        pm = PinMapping()
        with pytest.raises(AttributeError):
            pm.GPS_TX = 0


# ---------------------------------------------------------------------------
# InterfacePins
# ---------------------------------------------------------------------------

class TestDueInterfacePins:
    def test_all_four_uarts(self):
        ip = InterfacePins()
        assert ip.UART0_TX == 1
        assert ip.UART0_RX == 0
        assert ip.UART1_TX == 18
        assert ip.UART1_RX == 19
        assert ip.UART2_TX == 16
        assert ip.UART2_RX == 17
        assert ip.UART3_TX == 14
        assert ip.UART3_RX == 15

    def test_dual_spi(self):
        ip = InterfacePins()
        assert ip.SPI_SS == 10
        assert ip.SPI1_SS == 52

    def test_dual_i2c(self):
        ip = InterfacePins()
        assert ip.I2C_SDA == 20
        assert ip.I2C1_SDA == 70

    def test_pwm_pins_count(self):
        ip = InterfacePins()
        assert len(ip.PWM_PINS) == 12

    def test_is_frozen(self):
        ip = InterfacePins()
        with pytest.raises(AttributeError):
            ip.UART0_TX = 99


# ---------------------------------------------------------------------------
# DueConfig composite
# ---------------------------------------------------------------------------

class TestDueConfig:
    def test_composite_defaults(self):
        cfg = DueConfig()
        assert isinstance(cfg.board_config, BoardConfig)
        assert isinstance(cfg.serial_config, SerialConfig)
        assert isinstance(cfg.wire_protocol, WireProtocolConfig)
        assert isinstance(cfg.nexus_edge, NexusEdgeConfig)
        assert isinstance(cfg.pin_mapping, PinMapping)
        assert isinstance(cfg.interface_pins, InterfacePins)

    def test_nested_access(self):
        cfg = DueConfig()
        assert cfg.board_config.gpio_count == 54
        assert cfg.serial_config.serial_port == "Serial1"
        assert cfg.nexus_edge.edge_processing_enabled is True
        assert cfg.pin_mapping.SONAR4_TRIG == 28


# ---------------------------------------------------------------------------
# get_due_config factory
# ---------------------------------------------------------------------------

class TestGetDueConfig:
    def test_returns_due_config(self):
        cfg = get_due_config()
        assert isinstance(cfg, DueConfig)

    def test_baud_rate_override(self):
        cfg = get_due_config(baud_rate=57600)
        assert cfg.serial_config.baud_rate == 57600

    def test_pin_override(self):
        cfg = get_due_config(GPS_TX=10, SONAR1_TRIG=50)
        assert cfg.pin_mapping.GPS_TX == 10
        assert cfg.pin_mapping.SONAR1_TRIG == 50

    def test_serial_port_override(self):
        cfg = get_due_config(serial_port="Serial2")
        assert cfg.serial_config.serial_port == "Serial2"

    def test_edge_config_override(self):
        cfg = get_due_config(sensor_fusion_rate_hz=100.0)
        assert cfg.nexus_edge.sensor_fusion_rate_hz == 100.0

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown config keys"):
            get_due_config(fake_param=123)
