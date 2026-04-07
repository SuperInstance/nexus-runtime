"""Tests for NEXUS Raspberry Pi 5 configuration (config_pi5.py).

Covers: SoC identity, RP1 I/O controller, PCIe config, CSI 4K@60,
GPIO pin mapping, thermal/power profiles, sensor mappings, and
conflict detection.
"""

import pytest
from hardware.raspberry_pi.config_pi5 import (
    Pi5Config,
    Pi5GPIOPin,
    Pi5PinMapping,
    Pi5ThermalProfile,
    Pi5PowerProfile,
    PCIeConfig,
    PI5_PINOUT,
    PI5_MARINE_SENSORS,
)
from hardware.raspberry_pi import BoardModel


class TestPi5SoCIdentity:
    """Verify SoC and CPU specifications for Pi 5."""

    def test_soc_is_bcm2712(self, pi5_config):
        assert pi5_config.soc == "BCM2712"

    def test_io_controller_is_rp1(self, pi5_config):
        assert pi5_config.io_controller == "RP1"

    def test_cpu_architecture(self, pi5_config):
        assert pi5_config.cpu_arch == "cortex-a76"

    def test_cpu_core_count(self, pi5_config):
        assert pi5_config.cpu_cores == 4

    def test_cpu_max_clock(self, pi5_config):
        assert pi5_config.cpu_clock_max_mhz == 2400

    def test_cpu_clock_higher_than_pi4(self, pi5_config):
        """Pi 5 should clock higher than Pi 4B (1500 vs 2400 MHz)."""
        assert pi5_config.cpu_clock_max_mhz > 1500

    def test_gpu_is_videocore_vii(self, pi5_config):
        assert "VideoCore VII" in pi5_config.gpu

    def test_ram_type_lpddr4x(self, pi5_config):
        assert pi5_config.ram_type == "LPDDR4X-4267"

    def test_ram_options(self, pi5_config):
        assert pi5_config.ram_options_gb == (4, 8)

    def test_has_onboard_rtc(self, pi5_config):
        assert pi5_config.peripherals["rtc"] is True


class TestPi5GPIO:
    """Verify GPIO pin definitions via RP1."""

    def test_total_gpio_count(self, pi5_config):
        assert pi5_config.total_gpio == 28

    def test_has_extra_rp1_gpio(self, pi5_config):
        assert pi5_config.extra_rp1_gpio == 16

    def test_pinout_has_40_entries(self, pi5_config):
        assert len(pi5_config.pinout) == 40

    def test_gpio0_present(self, pi5_config):
        assert Pi5GPIOPin.GPIO0 in pi5_config.gpio_pins

    def test_gpio27_present(self, pi5_config):
        assert Pi5GPIOPin.GPIO27 in pi5_config.gpio_pins

    def test_pin_lookup_by_rp1_gpio(self, pi5_config):
        pin = pi5_config.get_pin(14)
        assert pin is not None
        assert pin.rp1_gpio == 14
        assert pin.physical_pin == 8

    def test_pin_lookup_by_physical(self, pi5_config):
        pin = pi5_config.get_pin_by_physical(12)
        assert pin is not None
        assert pin.rp1_gpio == 18

    def test_validate_gpio_valid(self, pi5_config):
        assert pi5_config.validate_gpio_assignment(12) is True

    def test_validate_gpio_invalid(self, pi5_config):
        assert pi5_config.validate_gpio_assignment(28) is False


class TestPi5PCIe:
    """Verify PCIe 2.0 x1 configuration."""

    def test_pcie_version(self, pi5_config):
        assert pi5_config.pcie.version == "2.0"

    def test_pcie_lanes(self, pi5_config):
        assert pi5_config.pcie.lanes == 1

    def test_pcie_in_peripherals(self, pi5_config):
        assert "pcie" in pi5_config.peripherals
        assert pi5_config.peripherals["pcie"]["version"] == "2.0"

    def test_pcie_supported_devices(self, pi5_config):
        devices = pi5_config.pcie.supported_devices
        assert len(devices) >= 3
        assert any("NVMe" in d for d in devices)
        assert any("AI" in d for d in devices)


class TestPi5CSI4K:
    """Verify 4K@60 CSI camera capability."""

    def test_csi_max_resolution_4k60(self, pi5_config):
        max_res = pi5_config.peripherals["csi"]["max_resolution"]
        assert "4K" in max_res
        assert "60" in max_res

    def test_csi_has_4_lanes(self, pi5_config):
        assert pi5_config.peripherals["csi"]["lanes_per_port"] == 4


class TestPi5MarineSensors:
    """Verify marine sensor mappings for Pi 5."""

    def test_ctd_exists(self, pi5_config):
        ctd = pi5_config.get_sensor_mapping("ctd")
        assert ctd is not None
        assert ctd.interface == "I2C-1"

    def test_gps_exists(self, pi5_config):
        gps = pi5_config.get_sensor_mapping("gps")
        assert gps is not None

    def test_sonar_4k_on_csi(self, pi5_config):
        sonar = pi5_config.get_sensor_mapping("sonar")
        assert sonar is not None
        assert sonar.interface == "CSI-0"
        assert "4K" in sonar.config_params["resolution"]

    def test_leak_sensor_is_gpio(self, pi5_config):
        leak = pi5_config.get_sensor_mapping("leak_sensor")
        assert leak is not None
        assert leak.interface == "GPIO"

    def test_list_sensors(self, pi5_config):
        keys = pi5_config.list_sensor_mappings()
        assert len(keys) >= 5
        assert "ctd" in keys
        assert "sonar" in keys


class TestPi5ThermalAndPower:
    """Verify thermal and power profiles."""

    def test_thermal_requires_active_cooling(self, pi5_config):
        assert "REQUIRED" in pi5_config.thermal.enclosure_note

    def test_thermal_throttle_at_80c(self, pi5_config):
        assert pi5_config.thermal.throttle_start_c == 80.0

    def test_power_idle_higher_than_pi4(self, pi5_config):
        """Pi 5 idle power should exceed Pi 4B idle (3.0W)."""
        assert pi5_config.power.idle_w > 3.0

    def test_power_max_is_12w(self, pi5_config):
        assert pi5_config.power.max_load_w == 12.0

    def test_has_pcie_power_budget(self, pi5_config):
        assert pi5_config.power.pcie_budget_w > 0

    def test_power_budget_with_pcie(self, pi5_config):
        budget = pi5_config.compute_power_budget(["pcie", "csi", "fan"])
        assert "pcie_w" in budget
        assert budget["total_w"] > pi5_config.power.idle_w


class TestPi5ConfigHash:
    """Verify configuration integrity hashing."""

    def test_hash_is_sha256(self, pi5_config):
        h = pi5_config.get_config_hash()
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_deterministic(self, pi5_config):
        assert pi5_config.get_config_hash() == pi5_config.get_config_hash()

    def test_hash_differs_from_pi4(self, pi5_config, pi4_config):
        """Pi 5 and Pi 4B should have different config hashes."""
        h5 = pi5_config.get_config_hash()
        h4 = pi4_config.get_config_hash()
        assert h5 != h4


class TestPi5Summary:
    """Verify summary output."""

    def test_summary_keys(self, pi5_config):
        s = pi5_config.summary()
        assert "board" in s
        assert "io_controller" in s
        assert "pcie" in s
        assert "csi_max" in s

    def test_summary_values(self, pi5_config):
        s = pi5_config.summary()
        assert s["soc"] == "BCM2712"
        assert s["io_controller"] == "RP1"

    def test_repr(self, pi5_config):
        r = repr(pi5_config)
        assert "BCM2712" in r
        assert "RP1" in r
        assert "pcie" in r


class TestPi5PeripheralCounts:
    """Verify peripheral counts exceed Pi 4B."""

    def test_more_i2c_buses(self, pi5_config, pi4_config):
        """Pi 5 has more I2C buses via RP1 (6 vs 2)."""
        assert pi5_config.peripherals["i2c"]["count"] > pi4_config.peripherals["i2c"]["count"]

    def test_more_uart(self, pi5_config, pi4_config):
        assert pi5_config.peripherals["uart"]["count"] >= pi4_config.peripherals["uart"]["count"]

    def test_more_pwm_channels(self, pi5_config, pi4_config):
        assert pi5_config.peripherals["pwm"]["channels"] > pi4_config.peripherals["pwm"]["channels"]
