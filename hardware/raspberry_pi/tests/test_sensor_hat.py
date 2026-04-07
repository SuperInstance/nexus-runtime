"""Tests for NEXUS Marine Sensor HAT configuration (sensor_hat.py).

Covers: sensor registration, bus conflict detection, I2C multiplexer,
power rail management, HAT compatibility, config hashing, and
sensor catalog lookups.
"""

import pytest
from hardware.raspberry_pi.sensor_hat import (
    MarineSensorHAT,
    SensorRegistration,
    SensorInterface,
    SensorCategory,
    I2CMultiplexer,
    PowerRail,
    PowerRailConfig,
    HATHardware,
    HAT_DEFS,
    DEFAULT_SENSOR_CATALOG,
)


class TestHATConstruction:
    """Verify HAT construction and hardware definitions."""

    def test_default_construction(self, sensor_hat):
        assert sensor_hat.board == "pi4b"
        assert sensor_hat.revision == "v2"
        assert sensor_hat.sensor_count() == 0

    def test_v1_hat(self, sensor_hat_v1):
        assert sensor_hat_v1.revision == "v1"
        assert sensor_hat_v1.hardware.has_adc is False
        assert sensor_hat_v1.hardware.has_can is False

    def test_v2_hat_has_features(self, sensor_hat):
        assert sensor_hat.hardware.has_power_control is True
        assert sensor_hat.hardware.has_i2c_mux is True
        assert sensor_hat.hardware.has_adc is True
        assert sensor_hat.hardware.has_can is True

    def test_custom_hat_id(self):
        hat = MarineSensorHAT(board="pi5", revision="custom", hat_id="my-custom-hat")
        assert hat.hat_id == "my-custom-hat"
        assert hat.board == "pi5"

    def test_known_hat_defs(self):
        assert "nexus-marine-v1" in HAT_DEFS
        assert "nexus-marine-v2" in HAT_DEFS
        assert "nexus-marine-v3" in HAT_DEFS

    def test_hat_compatibility(self, sensor_hat):
        assert sensor_hat.check_compatibility("pi4b") is True
        assert sensor_hat.check_compatibility("pi5") is True
        assert sensor_hat.check_compatibility("pizerow") is False


class TestSensorRegistration:
    """Verify sensor registration and validation."""

    def test_register_i2c_sensor(self, sensor_hat):
        reg = sensor_hat.register_sensor(
            "ms5837", bus="i2c", address=0x76,
        )
        assert reg.sensor_id == "ms5837"
        assert reg.interface == SensorInterface.I2C
        assert reg.address == 0x76

    def test_register_spi_sensor(self, sensor_hat):
        reg = sensor_hat.register_sensor(
            "mcp2515", bus="spi", address=0,
        )
        assert reg.interface == SensorInterface.SPI
        assert reg.address == 0

    def test_register_uart_sensor(self, sensor_hat):
        reg = sensor_hat.register_sensor(
            "gps_ublox", bus="uart", config={"uart_num": 0},
        )
        assert reg.interface == SensorInterface.UART

    def test_register_gpio_sensor(self, sensor_hat):
        reg = sensor_hat.register_sensor(
            "leak_sensor", bus="gpio", pins={"pin": 17},
        )
        assert reg.interface == SensorInterface.GPIO
        assert reg.pins["pin"] == 17

    def test_duplicate_registration_raises(self, sensor_hat):
        sensor_hat.register_sensor("ms5837", bus="i2c", address=0x76)
        with pytest.raises(ValueError, match="already registered"):
            sensor_hat.register_sensor("ms5837", bus="i2c", address=0x77)

    def test_unknown_bus_raises(self, sensor_hat):
        with pytest.raises(ValueError, match="Unknown bus interface"):
            sensor_hat.register_sensor("foo", bus="canbus")

    def test_unregister_sensor(self, sensor_hat):
        sensor_hat.register_sensor("ms5837", bus="i2c", address=0x76)
        sensor_hat.unregister_sensor("ms5837")
        assert sensor_hat.sensor_count() == 0

    def test_unregister_nonexistent_raises(self, sensor_hat):
        with pytest.raises(KeyError):
            sensor_hat.unregister_sensor("nonexistent")


class TestBusConflictDetection:
    """Verify bus conflict detection during registration."""

    def test_i2c_address_conflict(self, sensor_hat):
        sensor_hat.register_sensor("ms5837", bus="i2c", address=0x76)
        with pytest.raises(ValueError, match="Bus conflict"):
            sensor_hat.register_sensor("ms5837_dup", bus="i2c", address=0x76)

    def test_i2c_different_address_ok(self, sensor_hat):
        sensor_hat.register_sensor("ms5837", bus="i2c", address=0x76)
        # Same I2C bus but different address — should work (shared bus)
        reg = sensor_hat.register_sensor("bno055", bus="i2c", address=0x28)
        assert reg.sensor_id == "bno055"

    def test_spi_different_cs_ok(self, sensor_hat):
        sensor_hat.register_sensor("adc1", bus="spi", address=0)
        reg = sensor_hat.register_sensor("adc2", bus="spi", address=1)
        assert reg.sensor_id == "adc2"

    def test_spi_same_cs_conflict(self, sensor_hat):
        sensor_hat.register_sensor("adc1", bus="spi", address=0)
        with pytest.raises(ValueError, match="Bus conflict"):
            sensor_hat.register_sensor("adc2", bus="spi", address=0)


class TestSensorQuery:
    """Verify sensor query and listing methods."""

    def test_list_sensors_empty(self, sensor_hat):
        assert sensor_hat.list_sensors() == []

    def test_list_sensors_after_registration(self, sensor_hat):
        sensor_hat.register_sensor("ms5837", bus="i2c", address=0x76)
        sensor_hat.register_sensor("bno055", bus="i2c", address=0x28)
        sensors = sensor_hat.list_sensors()
        assert len(sensors) == 2
        assert sensors == ["bno055", "ms5837"]  # Sorted

    def test_list_by_category(self, sensor_hat):
        sensor_hat.register_sensor("ms5837", bus="i2c", address=0x76)
        sensor_hat.register_sensor("gps_ublox", bus="uart")
        nav = sensor_hat.list_sensors_by_category(SensorCategory.NAVIGATION)
        assert len(nav) >= 1

    def test_list_by_interface(self, sensor_hat):
        sensor_hat.register_sensor("ms5837", bus="i2c", address=0x76)
        sensor_hat.register_sensor("gps_ublox", bus="uart")
        i2c = sensor_hat.list_sensors_by_interface(SensorInterface.I2C)
        assert i2c == ["ms5837"]

    def test_get_sensor(self, sensor_hat):
        reg = sensor_hat.register_sensor("ms5837", bus="i2c", address=0x76)
        fetched = sensor_hat.get_sensor("ms5837")
        assert fetched is reg

    def test_get_nonexistent_sensor(self, sensor_hat):
        assert sensor_hat.get_sensor("nonexistent") is None


class TestSensorReadInterface:
    """Verify simulated sensor read interface."""

    def test_read_all_empty(self, sensor_hat):
        result = sensor_hat.read_all()
        assert result == {}

    def test_read_all_returns_registered(self, sensor_hat):
        sensor_hat.register_sensor("ms5837", bus="i2c", address=0x76)
        sensor_hat.register_sensor("bno055", bus="i2c", address=0x28)
        result = sensor_hat.read_all()
        assert "ms5837" in result
        assert "bno055" in result
        assert result["ms5837"]["status"] == "simulated"

    def test_read_single_sensor(self, sensor_hat):
        sensor_hat.register_sensor("ms5837", bus="i2c", address=0x76)
        result = sensor_hat.read_sensor("ms5837")
        assert result["sensor_name"] != ""
        assert result["interface"] == "i2c"

    def test_read_nonexistent_raises(self, sensor_hat):
        with pytest.raises(KeyError):
            sensor_hat.read_sensor("nonexistent")


class TestI2CMultiplexer:
    """Verify I2C multiplexer configuration."""

    def test_enable_mux(self, sensor_hat):
        sensor_hat.enable_i2c_mux(mux_address=0x71)
        assert sensor_hat._i2c_mux is not None
        assert sensor_hat._i2c_mux.mux_address == 0x71

    def test_register_mux_channel(self, sensor_hat):
        sensor_hat.enable_i2c_mux()
        sensor_hat.register_mux_channel("pressure", 0)
        assert sensor_hat._i2c_mux.channel_map["pressure"] == 0

    def test_register_mux_without_enabling_raises(self, sensor_hat):
        with pytest.raises(RuntimeError, match="not enabled"):
            sensor_hat.register_mux_channel("test", 0)

    def test_mux_channel_invalid_raises(self, sensor_hat):
        sensor_hat.enable_i2c_mux()
        with pytest.raises(ValueError, match="not in active channels"):
            sensor_hat.register_mux_channel("test", 8)


class TestPowerRailManagement:
    """Verify power rail configuration and budgeting."""

    def test_default_power_rails(self, sensor_hat):
        budget = sensor_hat.get_power_budget_ma()
        assert "3.3V" in budget
        assert "5.0V" in budget

    def test_add_custom_rail(self, sensor_hat):
        rail_cfg = PowerRailConfig(
            rail=PowerRail.RAIL_12V, max_current_ma=5000
        )
        sensor_hat.add_power_rail(rail_cfg)
        budget = sensor_hat.get_power_budget_ma()
        assert "12.0V" in budget
        assert budget["12.0V"] == 5000

    def test_power_budget_values(self, sensor_hat):
        budget = sensor_hat.get_power_budget_ma()
        assert budget["3.3V"] == 800
        assert budget["5.0V"] == 2000


class TestSensorCatalog:
    """Verify default sensor catalog."""

    def test_catalog_has_common_sensors(self):
        assert "bme280" in DEFAULT_SENSOR_CATALOG
        assert "ms5837" in DEFAULT_SENSOR_CATALOG
        assert "bno055" in DEFAULT_SENSOR_CATALOG
        assert "ina219" in DEFAULT_SENSOR_CATALOG
        assert "leak_sensor" in DEFAULT_SENSOR_CATALOG
        assert "gps_ublox" in DEFAULT_SENSOR_CATALOG

    def test_catalog_ms5837_is_navigation(self):
        assert DEFAULT_SENSOR_CATALOG["ms5837"]["category"] == SensorCategory.NAVIGATION

    def test_catalog_leak_is_safety(self):
        assert DEFAULT_SENSOR_CATALOG["leak_sensor"]["category"] == SensorCategory.SAFETY

    def test_catalog_bme280_is_environmental(self):
        assert DEFAULT_SENSOR_CATALOG["bme280"]["category"] == SensorCategory.ENVIRONMENTAL


class TestHATConfigHash:
    """Verify HAT configuration hashing."""

    def test_hash_empty_hat(self, sensor_hat):
        h = sensor_hat.get_config_hash()
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_changes_with_sensors(self, sensor_hat):
        h1 = sensor_hat.get_config_hash()
        sensor_hat.register_sensor("ms5837", bus="i2c", address=0x76)
        h2 = sensor_hat.get_config_hash()
        assert h1 != h2


class TestHATSummary:
    """Verify HAT summary output."""

    def test_summary_keys(self, sensor_hat):
        sensor_hat.register_sensor("ms5837", bus="i2c", address=0x76)
        s = sensor_hat.summary()
        assert "hat_id" in s
        assert "hat_name" in s
        assert "revision" in s
        assert "sensor_count" in s
        assert "sensors" in s
        assert "power_rails" in s

    def test_summary_sensor_count(self, sensor_hat):
        sensor_hat.register_sensor("ms5837", bus="i2c", address=0x76)
        sensor_hat.register_sensor("bno055", bus="i2c", address=0x28)
        s = sensor_hat.summary()
        assert s["sensor_count"] == 2

    def test_repr(self, sensor_hat):
        sensor_hat.register_sensor("ms5837", bus="i2c", address=0x76)
        r = repr(sensor_hat)
        assert "nexus-marine" in r
        assert "sensors=1" in r


class TestSensorRegistrationBusKey:
    """Verify SensorRegistration bus_key generation."""

    def test_i2c_bus_key(self):
        reg = SensorRegistration(
            sensor_id="test", sensor_name="Test",
            interface=SensorInterface.I2C,
            category=SensorCategory.ENVIRONMENTAL,
            address=0x76, bus_number=1,
        )
        assert reg.bus_key() == "i2c-1-0x76"

    def test_spi_bus_key(self):
        reg = SensorRegistration(
            sensor_id="test", sensor_name="Test",
            interface=SensorInterface.SPI,
            category=SensorCategory.ENVIRONMENTAL,
            address=0, bus_number=0,
        )
        assert reg.bus_key() == "spi-0-cs0"

    def test_uart_bus_key(self):
        reg = SensorRegistration(
            sensor_id="test", sensor_name="Test",
            interface=SensorInterface.UART,
            category=SensorCategory.NAVIGATION,
            bus_number=0,
        )
        assert reg.bus_key() == "uart-0"

    def test_gpio_bus_key(self):
        reg = SensorRegistration(
            sensor_id="test", sensor_name="Test",
            interface=SensorInterface.GPIO,
            category=SensorCategory.SAFETY,
            pins={"pin": 17},
        )
        assert reg.bus_key() == "gpio-17"
