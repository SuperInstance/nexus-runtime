"""Tests for NEXUS Compute Module 4 configuration."""

import pytest
from hardware.raspberry_pi.config_cm4 import (
    CM4Config,
    CM4Variant,
    CarrierBoardSpec,
    CARRIER_BOARDS,
    GPIOPin,
    PINOUT,
    ThermalProfile,
    PowerProfile,
)


@pytest.fixture
def config():
    return CM4Config()


@pytest.fixture
def wireless_config():
    return CM4Config(variant=CM4Variant(wireless=True))


class TestCM4SoCIdentity:
    def test_soc_is_bcm2711(self, config):
        assert config.soc == "BCM2711"

    def test_cpu_architecture(self, config):
        assert config.cpu_arch == "cortex-a72"

    def test_cpu_core_count(self, config):
        assert config.cpu_cores == 4

    def test_cpu_max_clock(self, config):
        assert config.cpu_clock_max_mhz == 1500

    def test_ram_options(self, config):
        assert config.ram_options_gb == (1, 2, 4, 8)

    def test_emmc_options(self, config):
        assert config.emmc_options_gb == (0, 8, 16, 32)


class TestCM4GPIO:
    def test_total_gpio_count(self, config):
        assert config.total_gpio == 28

    def test_pinout_entries(self, config):
        assert len(config.pinout) == 40

    def test_validate_gpio(self, config):
        assert config.validate_gpio_assignment(17) is True
        assert config.validate_gpio_assignment(99) is False


class TestCM4CarrierBoard:
    def test_default_carrier(self, config):
        assert config.carrier_name == "nexus-io-board"
        assert config.carrier.name == "NEXUS CM4 IO Board"

    def test_carrier_has_pcie(self, config):
        assert config.carrier.pcie_slot is True

    def test_carrier_has_nvme(self, config):
        assert config.carrier.nvme_slot is True

    def test_carrier_industrial_temp(self, config):
        assert config.carrier.industrial_temp is True

    def test_custom_carrier(self):
        cfg = CM4Config(carrier="custom-board")
        assert cfg.carrier.name == "custom-board"
        assert cfg.carrier.manufacturer == "Custom"

    def test_predefined_carriers(self):
        assert "nexus-io-board" in CARRIER_BOARDS
        assert "raspberry-pi-io" in CARRIER_BOARDS


class TestCM4Variant:
    def test_default_variant_no_wireless(self, config):
        assert config.variant.wireless is False

    def test_wireless_variant(self, wireless_config):
        assert wireless_config.variant.wireless is True

    def test_wireless_wifi_bands(self, wireless_config):
        assert "2.4GHz" in wireless_config.peripherals["wifi"]["bands"]

    def test_wireless_bt5(self, wireless_config):
        assert wireless_config.peripherals["bluetooth"]["version"] == "5.0"

    def test_no_wireless_wifi_none(self, config):
        assert config.peripherals["wifi"]["standard"] == "none"

    def test_repr_with_wireless(self, wireless_config):
        assert "wireless" in repr(wireless_config)


class TestCM4Peripherals:
    def test_pcie_20(self, config):
        assert config.peripherals["pcie"]["version"] == "2.0"

    def test_two_csi(self, config):
        assert config.peripherals["csi"]["count"] == 2

    def test_four_lane_csi(self, config):
        assert config.peripherals["csi"]["lanes_per_port"] == 4


class TestCM4ThermalAndPower:
    def test_idle_power(self, config):
        assert config.power.idle_w == 2.5

    def test_max_power(self, config):
        assert config.power.max_load_w == 7.0

    def test_power_budget_with_pcie(self, config):
        budget = config.compute_power_budget(["pcie", "csi"])
        assert "pcie_w" in budget
        assert budget["total_w"] > budget["base_w"]


class TestCM4NexusRole:
    def test_role_is_embedded(self, config):
        assert config.nexus_role == "embedded_compute"


class TestCM4ConfigHash:
    def test_hash_is_sha256(self, config):
        assert len(config.get_config_hash()) == 64

    def test_hash_differs_with_carrier(self, config):
        cfg2 = CM4Config(carrier="raspberry-pi-io")
        assert config.get_config_hash() != cfg2.get_config_hash()

    def test_hash_differs_with_wireless(self, config):
        cfg_w = CM4Config(variant=CM4Variant(wireless=True))
        assert config.get_config_hash() != cfg_w.get_config_hash()


class TestCM4Summary:
    def test_summary_keys(self, config):
        s = config.summary()
        assert "board" in s
        assert "carrier" in s
        assert "emmc_options_gb" in s
        assert "pcie" in s
        assert "wireless" in s

    def test_repr(self, config):
        assert "BCM2711" in repr(config)
