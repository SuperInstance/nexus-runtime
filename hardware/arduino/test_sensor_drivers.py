"""
Unit tests for hardware.arduino.sensor_drivers
"""

import pytest

from hardware.arduino.sensor_drivers import (
    GPSSensorConfig,
    IMUSensorConfig,
    SonarConfig,
    PressureSensorConfig,
    TemperatureSensorConfig,
    ServoConfig,
    MotorControllerConfig,
    Interface,
)


# ---------------------------------------------------------------------------
# GPSSensorConfig
# ---------------------------------------------------------------------------

class TestGPSSensorConfig:
    def test_defaults(self):
        cfg = GPSSensorConfig()
        assert cfg.baud_rate == 9600
        assert cfg.update_rate_hz == 5.0
        assert cfg.protocol == "NMEA"

    def test_serial_params_dict(self):
        cfg = GPSSensorConfig(baud_rate=4800)
        params = cfg.serial_params()
        assert params["baudrate"] == 4800
        assert params["timeout"] == 1.0

    def test_invalid_baud_raises(self):
        with pytest.raises(ValueError, match="baud_rate"):
            GPSSensorConfig(baud_rate=-1)

    def test_invalid_update_rate_raises(self):
        with pytest.raises(ValueError, match="update_rate_hz"):
            GPSSensorConfig(update_rate_hz=100.0)

    def test_nmea_filter(self):
        cfg = GPSSensorConfig()
        assert "GGA" in cfg.nmea_filter
        assert "RMC" in cfg.nmea_filter


# ---------------------------------------------------------------------------
# IMUSensorConfig
# ---------------------------------------------------------------------------

class TestIMUSensorConfig:
    def test_defaults_i2c(self):
        cfg = IMUSensorConfig()
        assert cfg.protocol == "I2C"
        assert cfg.interface == Interface.I2C

    def test_spi_mode(self):
        cfg = IMUSensorConfig(protocol="SPI", spi_cs_pin=10)
        assert cfg.interface == Interface.SPI
        assert cfg.spi_cs_pin == 10

    def test_invalid_protocol_raises(self):
        with pytest.raises(ValueError, match="protocol"):
            IMUSensorConfig(protocol="UART")

    def test_negative_accel_raises(self):
        with pytest.raises(ValueError, match="accel_range_g"):
            IMUSensorConfig(accel_range_g=-1)

    def test_axis_flags(self):
        cfg = IMUSensorConfig(enable_mag=False)
        assert cfg.enable_accel is True
        assert cfg.enable_gyro is True
        assert cfg.enable_mag is False


# ---------------------------------------------------------------------------
# SonarConfig
# ---------------------------------------------------------------------------

class TestSonarConfig:
    def test_defaults(self):
        cfg = SonarConfig()
        assert cfg.max_range_cm == 400.0
        assert cfg.trigger_pulse_us == 10

    def test_distance_calculation_air(self):
        cfg = SonarConfig(speed_of_sound_cm_us=0.034)
        # echo_us = 1176 µs → distance ≈ (1176 * 0.034) / 2 = 20 cm
        dist = cfg.distance_cm(1176.0)
        assert abs(dist - 20.0) < 0.1

    def test_distance_calculation_water(self):
        cfg = SonarConfig(speed_of_sound_cm_us=0.015, medium="water")
        dist = cfg.distance_cm(2666.0)
        assert abs(dist - 20.0) < 0.1

    def test_effective_timeout_ms(self):
        cfg = SonarConfig(echo_timeout_us=30_000)
        assert cfg.effective_timeout_ms == pytest.approx(30.0)

    def test_max_less_than_min_raises(self):
        with pytest.raises(ValueError, match="max_range_cm"):
            SonarConfig(max_range_cm=1.0, min_range_cm=10.0)

    def test_zero_trigger_pulse_raises(self):
        with pytest.raises(ValueError, match="trigger_pulse_us"):
            SonarConfig(trigger_pulse_us=0)


# ---------------------------------------------------------------------------
# PressureSensorConfig
# ---------------------------------------------------------------------------

class TestPressureSensorConfig:
    def test_defaults(self):
        cfg = PressureSensorConfig()
        assert cfg.range_kpa == 300.0
        assert cfg.resolution_bits == 24
        assert cfg.protocol == "I2C"

    def test_depth_conversion(self):
        cfg = PressureSensorConfig()
        # 101.325 kPa ≈ 1 atm ≈ 10.06 m depth in seawater
        depth = cfg.pressure_to_depth_m(101.325)
        assert abs(depth - 10.06) < 0.1

    def test_invalid_resolution_raises(self):
        with pytest.raises(ValueError, match="resolution_bits"):
            PressureSensorConfig(resolution_bits=7)

    def test_invalid_protocol_raises(self):
        with pytest.raises(ValueError, match="protocol"):
            PressureSensorConfig(protocol="UART")


# ---------------------------------------------------------------------------
# TemperatureSensorConfig
# ---------------------------------------------------------------------------

class TestTemperatureSensorConfig:
    def test_defaults_1wire(self):
        cfg = TemperatureSensorConfig()
        assert cfg.protocol == "1WIRE"
        assert cfg.range_celsius == (-40.0, 85.0)

    def test_i2c_mode(self):
        cfg = TemperatureSensorConfig(protocol="I2C")
        assert cfg.protocol == "I2C"

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError, match="range_celsius"):
            TemperatureSensorConfig(range_celsius=(100.0, 0.0))

    def test_invalid_resolution_raises(self):
        with pytest.raises(ValueError, match="resolution_bits"):
            TemperatureSensorConfig(resolution_bits=13)

    def test_invalid_protocol_raises(self):
        with pytest.raises(ValueError, match="protocol"):
            TemperatureSensorConfig(protocol="SPI")


# ---------------------------------------------------------------------------
# ServoConfig
# ---------------------------------------------------------------------------

class TestServoConfig:
    def test_defaults(self):
        cfg = ServoConfig()
        assert cfg.min_pulse_us == 1000
        assert cfg.max_pulse_us == 2000
        assert cfg.frequency_hz == 50

    def test_pulse_range(self):
        cfg = ServoConfig()
        assert cfg.pulse_range_us == 1000

    def test_angle_to_pulse(self):
        cfg = ServoConfig()
        pulse_0 = cfg.angle_to_pulse(0.0)
        pulse_90 = cfg.angle_to_pulse(90.0)
        pulse_180 = cfg.angle_to_pulse(180.0)
        assert pulse_0 == 1000
        assert pulse_90 == 1500
        assert pulse_180 == 2000

    def test_angle_to_pulse_inverted(self):
        cfg = ServoConfig(invert=True)
        pulse_0 = cfg.angle_to_pulse(0.0)
        pulse_180 = cfg.angle_to_pulse(180.0)
        assert pulse_0 == 2000
        assert pulse_180 == 1000

    def test_invalid_angle_raises(self):
        cfg = ServoConfig()
        with pytest.raises(ValueError, match="angle_deg"):
            cfg.angle_to_pulse(-1.0)

    def test_invalid_frequency_raises(self):
        with pytest.raises(ValueError, match="frequency_hz"):
            ServoConfig(frequency_hz=0)


# ---------------------------------------------------------------------------
# MotorControllerConfig
# ---------------------------------------------------------------------------

class TestMotorControllerConfig:
    def test_defaults(self):
        cfg = MotorControllerConfig()
        assert cfg.max_current_a == 30.0
        assert cfg.channels == 1
        assert cfg.pins == (10,)

    def test_throttle_range(self):
        cfg = MotorControllerConfig()
        assert cfg.throttle_range_us == 1000

    def test_throttle_to_pulse(self):
        cfg = MotorControllerConfig()
        assert cfg.throttle_to_pulse(0.0) == 1000
        assert cfg.throttle_to_pulse(0.5) == 1500
        assert cfg.throttle_to_pulse(1.0) == 2000

    def test_multi_channel(self):
        cfg = MotorControllerConfig(channels=3, pins=(44, 45, 46))
        assert cfg.channels == 3
        assert len(cfg.pins) == 3

    def test_channel_pin_mismatch_raises(self):
        with pytest.raises(ValueError, match="len\\(pins\\)"):
            MotorControllerConfig(channels=2, pins=(10,))

    def test_invalid_throttle_fraction_raises(self):
        cfg = MotorControllerConfig()
        with pytest.raises(ValueError, match="fraction"):
            cfg.throttle_to_pulse(1.5)

    def test_bidirectional_flag(self):
        cfg = MotorControllerConfig(bidirectional=True)
        assert cfg.bidirectional is True
