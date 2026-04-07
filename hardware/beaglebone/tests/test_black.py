"""
Test suite for BeagleBone Black hardware configuration.
"""

import pytest
from dataclasses import fields, FrozenInstanceError

from hardware.beaglebone.config_black import (
    BlackCPUConfig,
    BlackMemoryConfig,
    BlackStorageConfig,
    BlackPowerConfig,
    BlackGPIOConfig,
    BeagleBoneBlackConfig,
    create_beaglebone_black_config,
)


class TestBlackCPUConfig:
    def test_arch(self):
        assert BlackCPUConfig().arch == "ARM"

    def test_cores(self):
        assert BlackCPUConfig().cores == 1

    def test_core_type(self):
        assert BlackCPUConfig().core_type == "Cortex-A8"

    def test_clock_ghz(self):
        assert BlackCPUConfig().clock_ghz == 1.0

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            BlackCPUConfig().cores = 4

    def test_equality(self):
        assert BlackCPUConfig() == BlackCPUConfig()

    def test_field_count(self):
        assert len(fields(BlackCPUConfig)) == 4


class TestBlackMemoryConfig:
    def test_ram_gb(self):
        assert BlackMemoryConfig().ram_gb == 0.5

    def test_bandwidth(self):
        assert BlackMemoryConfig().bandwidth_gbps == 6.4

    def test_type(self):
        assert BlackMemoryConfig().type == "DDR3"

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            BlackMemoryConfig().ram_gb = 2

    def test_equality(self):
        assert BlackMemoryConfig() == BlackMemoryConfig()


class TestBlackStorageConfig:
    def test_boot(self):
        assert BlackStorageConfig().boot == "microSD / eMMC 4GB"

    def test_usb2_count(self):
        assert BlackStorageConfig().usb2_count == 1

    def test_usb3_count(self):
        assert BlackStorageConfig().usb3_count == 0

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            BlackStorageConfig().usb2_count = 2


class TestBlackPowerConfig:
    def test_max_watts(self):
        assert BlackPowerConfig().max_watts == 5

    def test_thermal_throttle(self):
        assert BlackPowerConfig().thermal_throttle_c == 85

    def test_input_voltage(self):
        assert BlackPowerConfig().input_voltage == "5V DC"

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            BlackPowerConfig().max_watts = 10


class TestBlackGPIOConfig:
    def test_gps_tx(self):
        assert BlackGPIOConfig().GPS_TX == "P9.21"

    def test_gps_rx(self):
        assert BlackGPIOConfig().GPS_RX == "P9.22"

    def test_thruster_pwm_0(self):
        assert BlackGPIOConfig().THRUSTER_PWM_0 == "P9.14"

    def test_led_count(self):
        """Verify 4 USR LEDs defined."""
        gpio = BlackGPIOConfig()
        assert hasattr(gpio, "LED_USR0")
        assert hasattr(gpio, "LED_USR3")

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            BlackGPIOConfig().GPS_TX = "P8.1"

    def test_equality(self):
        assert BlackGPIOConfig() == BlackGPIOConfig()


class TestBeagleBoneBlackConfig:
    def test_board_name(self):
        assert BeagleBoneBlackConfig().board_name == "BeagleBone Black"

    def test_cpu_type(self):
        assert isinstance(BeagleBoneBlackConfig().cpu, BlackCPUConfig)

    def test_memory_type(self):
        assert isinstance(BeagleBoneBlackConfig().memory, BlackMemoryConfig)

    def test_gpio_type(self):
        assert isinstance(BeagleBoneBlackConfig().gpio, BlackGPIOConfig)

    def test_pru_count(self):
        assert BeagleBoneBlackConfig().pru_count == 2

    def test_field_count(self):
        assert len(fields(BeagleBoneBlackConfig)) == 7


class TestCreateBeagleBoneBlackConfig:
    def test_factory_returns(self):
        cfg = create_beaglebone_black_config()
        assert isinstance(cfg, BeagleBoneBlackConfig)

    def test_defaults(self):
        cfg = create_beaglebone_black_config()
        assert cfg.board_name == "BeagleBone Black"
        assert cfg.pru_count == 2

    def test_nested_cpu_override(self):
        cfg = create_beaglebone_black_config(cpu={"clock_ghz": 0.8})
        assert cfg.cpu.clock_ghz == 0.8

    def test_board_name_override(self):
        cfg = create_beaglebone_black_config(board_name="BBB Custom")
        assert cfg.board_name == "BBB Custom"

    def test_pru_count_override(self):
        cfg = create_beaglebone_black_config(pru_count=4)
        assert cfg.pru_count == 4


class TestPackageInitImport:
    def test_import_black_config(self):
        from hardware.beaglebone import BeagleBoneBlackConfig as BBC
        assert BBC is BeagleBoneBlackConfig

    def test_import_black_cpu(self):
        from hardware.beaglebone import BlackCPUConfig as BCC
        assert BCC is BlackCPUConfig

    def test_registry_contains_black(self):
        from hardware.beaglebone import BOARD_REGISTRY
        assert "beaglebone-black" in BOARD_REGISTRY

    def test_list_supported_boards(self):
        from hardware.beaglebone import list_supported_boards
        boards = list_supported_boards()
        assert "beaglebone-black" in boards

    def test_get_board_info_black(self):
        from hardware.beaglebone import get_board_info
        info = get_board_info("beaglebone-black")
        assert info["config_class"] == "BeagleBoneBlackConfig"

    def test_get_board_info_unknown_raises(self):
        from hardware.beaglebone import get_board_info
        with pytest.raises(ValueError):
            get_board_info("unknown-board")
