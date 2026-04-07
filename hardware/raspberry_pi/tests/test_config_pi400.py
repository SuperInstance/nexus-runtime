"""Tests for NEXUS Raspberry Pi 400 configuration."""

import pytest
from hardware.raspberry_pi.config_pi400 import (
    Pi400Config,
    GPIOPin,
    PINOUT,
    ThermalProfile,
    PowerProfile,
    MarineSensorPinAssignment,
)


@pytest.fixture
def config():
    return Pi400Config()


class TestPi400SoCIdentity:
    def test_soc_is_bcm2711(self, config):
        assert config.soc == "BCM2711"

    def test_cpu_architecture(self, config):
        assert config.cpu_arch == "cortex-a72"

    def test_cpu_core_count(self, config):
        assert config.cpu_cores == 4

    def test_cpu_max_clock(self, config):
        assert config.cpu_clock_max_mhz == 1500

    def test_ram_type(self, config):
        assert config.ram_type == "LPDDR4-3200"

    def test_ram_options(self, config):
        assert config.ram_options_gb == (4,)

    def test_default_ram(self, config):
        assert config.default_ram_gb == 4


class TestPi400GPIO:
    def test_total_gpio_count(self, config):
        assert config.total_gpio == 28

    def test_pinout_40_entries(self, config):
        assert len(config.pinout) == 40

    def test_pin_lookup_by_bcm(self, config):
        pin = config.get_pin(2)
        assert pin is not None
        assert pin.bcm_gpio == 2

    def test_pin_lookup_by_physical(self, config):
        pin = config.get_pin_by_physical(3)
        assert pin is not None
        assert pin.bcm_gpio == 2

    def test_validate_gpio(self, config):
        assert config.validate_gpio_assignment(17) is True
        assert config.validate_gpio_assignment(99) is False


class TestPi400MarineSensors:
    def test_ctd_exists(self, config):
        assert config.get_sensor_mapping("ctd") is not None

    def test_gps_exists(self, config):
        assert config.get_sensor_mapping("gps") is not None

    def test_list_mappings(self, config):
        keys = config.list_sensor_mappings()
        assert len(keys) >= 6


class TestPi400ThermalAndPower:
    def test_idle_power(self, config):
        assert config.power.idle_w == 3.5

    def test_max_power(self, config):
        assert config.power.max_load_w == 8.5

    def test_power_budget(self, config):
        budget = config.compute_power_budget(["wifi", "ethernet"])
        assert budget["total_w"] > budget["base_w"]


class TestPi400Peripherals:
    def test_single_csi(self, config):
        assert config.peripherals["csi"]["count"] == 1

    def test_three_usb3(self, config):
        assert config.peripherals["usb"]["usb3_ports"] == 3

    def test_gigabit_ethernet(self, config):
        assert config.peripherals["ethernet"]["speed_mbps"] == 1000

    def test_bluetooth_5(self, config):
        assert config.peripherals["bluetooth"]["version"] == "5.0"

    def test_wifi_5ghz(self, config):
        assert "5GHz" in config.peripherals["wifi"]["bands"]


class TestPi400NexusRole:
    def test_role_is_surface_station(self, config):
        assert config.nexus_role == "surface_station"

    def test_form_factor_is_keyboard(self, config):
        assert "keyboard" in config.form_factor.lower()


class TestPi400ConfigHash:
    def test_hash_is_sha256(self, config):
        h = config.get_config_hash()
        assert len(h) == 64

    def test_hash_deterministic(self, config):
        assert config.get_config_hash() == config.get_config_hash()


class TestPi400Summary:
    def test_summary_keys(self, config):
        s = config.summary()
        assert "board" in s
        assert "form_factor" in s

    def test_repr(self, config):
        assert "BCM2711" in repr(config)
