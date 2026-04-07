"""Tests for NEXUS Raspberry Pi 3B+ configuration."""

import pytest
from hardware.raspberry_pi.config_pi3b import (
    Pi3BConfig,
    GPIOPin,
    PinMapping,
    MARINE_SENSOR_MAPPINGS,
    PINOUT,
    ThermalProfile,
    PowerProfile,
    MarineSensorPinAssignment,
)


@pytest.fixture
def config():
    return Pi3BConfig()


class TestPi3BSoCIdentity:
    def test_soc_is_bcm2837b0(self, config):
        assert config.soc == "BCM2837B0"

    def test_cpu_architecture(self, config):
        assert config.cpu_arch == "cortex-a53"

    def test_cpu_core_count(self, config):
        assert config.cpu_cores == 4

    def test_cpu_max_clock(self, config):
        assert config.cpu_clock_max_mhz == 1400

    def test_gpu_is_videocore_iv(self, config):
        assert "VideoCore IV" in config.gpu

    def test_ram_type(self, config):
        assert config.ram_type == "LPDDR2-900"

    def test_ram_options(self, config):
        assert config.ram_options_gb == (1,)

    def test_default_ram(self, config):
        assert config.default_ram_gb == 1


class TestPi3BGPIO:
    def test_total_gpio_count(self, config):
        assert config.total_gpio == 28

    def test_gpio_pin_enum_count(self, config):
        assert len(config.gpio_pins) == 28

    def test_pinout_has_40_entries(self, config):
        assert len(config.pinout) == 40

    def test_gpio0_present(self, config):
        assert GPIOPin.GPIO0 in config.gpio_pins

    def test_gpio27_present(self, config):
        assert GPIOPin.GPIO27 in config.gpio_pins

    def test_pin_lookup_by_bcm(self, config):
        pin = config.get_pin(17)
        assert pin is not None
        assert pin.bcm_gpio == 17

    def test_pin_lookup_by_physical(self, config):
        pin = config.get_pin_by_physical(11)
        assert pin is not None
        assert pin.bcm_gpio == 17

    def test_pin_lookup_invalid(self, config):
        assert config.get_pin(99) is None
        assert config.get_pin_by_physical(50) is None

    def test_validate_gpio_valid(self, config):
        assert config.validate_gpio_assignment(17) is True

    def test_validate_gpio_invalid(self, config):
        assert config.validate_gpio_assignment(28) is False


class TestPi3BMarineSensors:
    def test_ctd_exists(self, config):
        ctd = config.get_sensor_mapping("ctd")
        assert ctd is not None
        assert ctd.interface == "I2C-1"

    def test_gps_uses_uart0(self, config):
        gps = config.get_sensor_mapping("gps")
        assert gps is not None
        assert gps.pins["tx"] == 14
        assert gps.pins["rx"] == 15

    def test_leak_sensor_gpio(self, config):
        leak = config.get_sensor_mapping("leak_sensor")
        assert leak is not None
        assert leak.pins["alarm"] == 17

    def test_esc_uses_pwm(self, config):
        esc = config.get_sensor_mapping("esc_thrust")
        assert esc is not None
        assert esc.config_params["frequency_hz"] == 50

    def test_list_mappings(self, config):
        keys = config.list_sensor_mappings()
        assert len(keys) >= 10
        assert "ctd" in keys
        assert "gps" in keys


class TestPi3BThermalAndPower:
    def test_thermal_throttle(self, config):
        assert config.thermal.throttle_start_c == 80.0

    def test_thermal_critical(self, config):
        assert config.thermal.critical_c == 85.0

    def test_power_idle(self, config):
        assert config.power.idle_w == 2.2

    def test_power_max(self, config):
        assert config.power.max_load_w == 6.7

    def test_power_budget_base(self, config):
        budget = config.compute_power_budget([])
        assert budget["base_w"] == 2.2
        assert budget["total_w"] == 2.2

    def test_power_budget_with_peripherals(self, config):
        budget = config.compute_power_budget(["csi", "wifi", "ethernet"])
        assert budget["total_w"] > budget["base_w"]


class TestPi3BPeripherals:
    def test_bluetooth_42(self, config):
        assert config.peripherals["bluetooth"]["version"] == "4.2"

    def test_no_usb3(self, config):
        assert config.peripherals["usb"]["usb3_ports"] == 0

    def test_usb2_count(self, config):
        assert config.peripherals["usb"]["usb2_ports"] == 4

    def test_single_csi(self, config):
        assert config.peripherals["csi"]["count"] == 1

    def test_wifi_5ghz(self, config):
        assert "5GHz" in config.peripherals["wifi"]["bands"]

    def test_ethernet_usb_bottleneck(self, config):
        assert config.peripherals["ethernet"]["speed_mbps"] == 300


class TestPi3BConfigHash:
    def test_hash_is_sha256(self, config):
        h = config.get_config_hash()
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_deterministic(self, config):
        assert config.get_config_hash() == config.get_config_hash()


class TestPi3BSummary:
    def test_summary_keys(self, config):
        s = config.summary()
        assert "board" in s
        assert "soc" in s
        assert "cpu" in s
        assert "gpio_count" in s

    def test_summary_values(self, config):
        s = config.summary()
        assert s["soc"] == "BCM2837B0"
        assert s["gpio_count"] == 28

    def test_repr(self, config):
        r = repr(config)
        assert "BCM2837B0" in r
        assert "Pi3BConfig" in r
