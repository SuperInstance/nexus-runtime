"""Tests for NEXUS Raspberry Pi Zero 2 W configuration (config_pizerow.py).

Covers: SoC identity, GPIO pin mapping, sensor presets, thermal/power
profiles, battery life estimation, and pin lookups.
"""

import pytest
from hardware.raspberry_pi.config_pizerow import (
    PiZero2WConfig,
    Zero2WGPIOPin,
    Zero2WPinMapping,
    Zero2WThermalProfile,
    Zero2WPowerProfile,
    Zero2WSensorPreset,
    ZERO2W_PINOUT,
    ZERO2W_SENSOR_PRESETS,
)
from hardware.raspberry_pi import BoardModel


class TestZero2WSoCIdentity:
    """Verify SoC and CPU specifications for Pi Zero 2W."""

    def test_soc_is_bcm2710(self, pizerow_config):
        assert pizerow_config.soc == "BCM2710"

    def test_cpu_architecture(self, pizerow_config):
        assert pizerow_config.cpu_arch == "cortex-a53"

    def test_cpu_core_count(self, pizerow_config):
        assert pizerow_config.cpu_cores == 4

    def test_cpu_max_clock(self, pizerow_config):
        assert pizerow_config.cpu_clock_max_mhz == 1000

    def test_ram_is_512mb(self, pizerow_config):
        assert pizerow_config.default_ram_gb == 0.5

    def test_ram_single_option(self, pizerow_config):
        assert pizerow_config.ram_options_gb == (0.5,)

    def test_gpu_is_videocore_iv(self, pizerow_config):
        assert "VideoCore IV" in pizerow_config.gpu

    def test_nexus_role_is_sensor_node(self, pizerow_config):
        assert pizerow_config.nexus_role == "sensor_node"


class TestZero2WGPIO:
    """Verify GPIO pin definitions."""

    def test_total_gpio_count(self, pizerow_config):
        assert pizerow_config.total_gpio == 28

    def test_pinout_40_entries(self, pizerow_config):
        assert len(pizerow_config.pinout) == 40

    def test_gpio2_present(self, pizerow_config):
        assert Zero2WGPIOPin.GPIO2 in pizerow_config.gpio_pins

    def test_gpio27_present(self, pizerow_config):
        assert Zero2WGPIOPin.GPIO27 in pizerow_config.gpio_pins

    def test_pin_lookup_by_bcm(self, pizerow_config):
        pin = pizerow_config.get_pin(17)
        assert pin is not None
        assert pin.bcm_gpio == 17

    def test_pin_lookup_by_physical(self, pizerow_config):
        pin = pizerow_config.get_pin_by_physical(3)
        assert pin is not None
        assert pin.bcm_gpio == 2

    def test_validate_gpio_valid(self, pizerow_config):
        assert pizerow_config.validate_gpio_assignment(12) is True

    def test_validate_gpio_invalid(self, pizerow_config):
        assert pizerow_config.validate_gpio_assignment(28) is False


class TestZero2WSensorPresets:
    """Verify marine sensor node presets."""

    def test_has_buoy_ctd_preset(self, pizerow_config):
        preset = pizerow_config.get_sensor_preset("buoy_ctd")
        assert preset is not None
        assert "CTD" in preset.description

    def test_has_rov_depth_preset(self, pizerow_config):
        preset = pizerow_config.get_sensor_preset("rov_depth")
        assert preset is not None
        assert "Depth" in preset.description

    def test_has_telemetry_beacon(self, pizerow_config):
        preset = pizerow_config.get_sensor_preset("telemetry_beacon")
        assert preset is not None
        assert "BLE" in preset.description

    def test_list_presets(self, pizerow_config):
        presets = pizerow_config.list_sensor_presets()
        assert len(presets) == 3
        assert "buoy_ctd" in presets
        assert "rov_depth" in presets
        assert "telemetry_beacon" in presets

    def test_buoy_preset_power_estimate(self, pizerow_config):
        preset = pizerow_config.get_sensor_preset("buoy_ctd")
        assert preset.power_estimate_mw > 0
        assert preset.sample_rate_hz > 0

    def test_preset_frozen(self, pizerow_config):
        preset = pizerow_config.get_sensor_preset("buoy_ctd")
        with pytest.raises(AttributeError):
            preset.preset_name = "Modified"


class TestZero2WThermalAndPower:
    """Verify thermal and power profiles."""

    def test_power_idle_under_1w(self, pizerow_config):
        assert pizerow_config.power.idle_w < 1.0

    def test_power_max_is_1_5w(self, pizerow_config):
        assert pizerow_config.power.max_load_w == 1.5

    def test_power_lower_than_pi4(self, pizerow_config, pi4_config):
        """Zero 2W should draw significantly less power than Pi 4B."""
        assert pizerow_config.power.idle_w < pi4_config.power.idle_w
        assert pizerow_config.power.max_load_w < pi4_config.power.max_load_w

    def test_solar_compatible(self, pizerow_config):
        assert pizerow_config.power.solar_compatible is True

    def test_recommended_battery(self, pizerow_config):
        assert pizerow_config.power.recommended_battery_mah == 5000

    def test_thermal_idle_lower(self, pizerow_config, pi4_config):
        """Zero 2W should run cooler than Pi 4B at idle."""
        assert pizerow_config.thermal.idle_temp_c < pi4_config.thermal.idle_temp_c


class TestZero2WBatteryLife:
    """Verify battery life estimation."""

    def test_battery_life_5000mah(self, pizerow_config):
        life = pizerow_config.estimate_battery_life(5000, load_factor=0.5)
        assert life > 0
        assert life < 100  # Should be reasonable hours

    def test_battery_life_increases_with_capacity(self, pizerow_config):
        life_small = pizerow_config.estimate_battery_life(2000)
        life_large = pizerow_config.estimate_battery_life(10000)
        assert life_large > life_small

    def test_battery_life_decreases_with_load(self, pizerow_config):
        life_low = pizerow_config.estimate_battery_life(5000, load_factor=0.3)
        life_high = pizerow_config.estimate_battery_life(5000, load_factor=1.0)
        assert life_low > life_high

    def test_battery_life_100mah(self, pizerow_config):
        life = pizerow_config.estimate_battery_life(100)
        assert life < 5  # Very small battery, short life


class TestZero2WPeripherals:
    """Verify peripheral specifications."""

    def test_single_csi_port(self, pizerow_config):
        assert pizerow_config.peripherals["csi"]["count"] == 1

    def test_csi_2_lanes(self, pizerow_config):
        assert pizerow_config.peripherals["csi"]["lanes_per_port"] == 2

    def test_single_usb_otg(self, pizerow_config):
        assert pizerow_config.peripherals["usb"]["usb2_ports"] == 1
        assert pizerow_config.peripherals["usb"]["otg"] is True

    def test_wifi_2_4ghz_only(self, pizerow_config):
        assert pizerow_config.peripherals["wifi"]["bands"] == ["2.4GHz"]

    def test_bluetooth_4_2(self, pizerow_config):
        assert pizerow_config.peripherals["bluetooth"]["version"] == "4.2"


class TestZero2WConfigHash:
    """Verify configuration integrity hashing."""

    def test_hash_is_sha256(self, pizerow_config):
        h = pizerow_config.get_config_hash()
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_deterministic(self, pizerow_config):
        assert pizerow_config.get_config_hash() == pizerow_config.get_config_hash()

    def test_hash_differs_from_pi4(self, pizerow_config, pi4_config):
        h_z = pizerow_config.get_config_hash()
        h_4 = pi4_config.get_config_hash()
        assert h_z != h_4


class TestZero2WSummary:
    """Verify summary output."""

    def test_summary_keys(self, pizerow_config):
        s = pizerow_config.summary()
        assert "board" in s
        assert "soc" in s
        assert "form_factor" in s
        assert "sensor_presets" in s
        assert "solar_compatible" in s

    def test_summary_form_factor(self, pizerow_config):
        s = pizerow_config.summary()
        assert "65" in s["form_factor"]

    def test_repr(self, pizerow_config):
        r = repr(pizerow_config)
        assert "BCM2710" in r
        assert "sensor_node" in r
