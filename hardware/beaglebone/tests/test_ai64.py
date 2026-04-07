"""
Test suite for BeagleBone AI-64 hardware configuration.
"""

import pytest
from dataclasses import fields, FrozenInstanceError

from hardware.beaglebone.config_ai64 import (
    AI64CPUConfig,
    AI64DSPConfig,
    AI64MemoryConfig,
    AI64StorageConfig,
    AI64PowerConfig,
    AI64GPIOConfig,
    BeagleBoneAI64Config,
    create_beaglebone_ai64_config,
)


class TestAI64CPUConfig:
    def test_arch(self):
        assert AI64CPUConfig().arch == "ARM"

    def test_cores(self):
        assert AI64CPUConfig().cores == 2

    def test_core_type(self):
        assert AI64CPUConfig().core_type == "Cortex-A15"

    def test_clock_ghz(self):
        assert AI64CPUConfig().clock_ghz == 1.5

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            AI64CPUConfig().cores = 4

    def test_equality(self):
        assert AI64CPUConfig() == AI64CPUConfig()

    def test_field_count(self):
        assert len(fields(AI64CPUConfig)) == 4


class TestAI64DSPConfig:
    def test_dsp_type(self):
        assert AI64DSPConfig().dsp_type == "C66x"

    def test_count(self):
        assert AI64DSPConfig().count == 2

    def test_clock_ghz(self):
        assert AI64DSPConfig().clock_ghz == 1.0

    def test_eve_count(self):
        assert AI64DSPConfig().eve_count == 4

    def test_eve_type(self):
        assert AI64DSPConfig().eve_type == "EVE (Vision Accelerator)"

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            AI64DSPConfig().count = 4

    def test_equality(self):
        assert AI64DSPConfig() == AI64DSPConfig()

    def test_field_count(self):
        assert len(fields(AI64DSPConfig)) == 5


class TestAI64MemoryConfig:
    def test_ram_gb(self):
        assert AI64MemoryConfig().ram_gb == 2

    def test_bandwidth(self):
        assert AI64MemoryConfig().bandwidth_gbps == 12.8

    def test_type(self):
        assert AI64MemoryConfig().type == "DDR3"

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            AI64MemoryConfig().ram_gb = 4


class TestAI64StorageConfig:
    def test_boot(self):
        assert AI64StorageConfig().boot == "microSD / eMMC 8GB"

    def test_usb3_count(self):
        assert AI64StorageConfig().usb3_count == 1

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            AI64StorageConfig().boot = "NVMe"


class TestAI64PowerConfig:
    def test_max_watts(self):
        assert AI64PowerConfig().max_watts == 10

    def test_input_voltage(self):
        assert AI64PowerConfig().input_voltage == "12V DC / 5V USB-C"

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            AI64PowerConfig().max_watts = 15


class TestAI64GPIOConfig:
    def test_gps_tx(self):
        assert AI64GPIOConfig().GPS_TX == "P9.21"

    def test_thruster_channels(self):
        gpio = AI64GPIOConfig()
        assert hasattr(gpio, "THRUSTER_PWM_0")
        assert hasattr(gpio, "THRUSTER_PWM_3")

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            AI64GPIOConfig().GPS_TX = "P8.1"

    def test_equality(self):
        assert AI64GPIOConfig() == AI64GPIOConfig()


class TestBeagleBoneAI64Config:
    def test_board_name(self):
        assert BeagleBoneAI64Config().board_name == "BeagleBone AI-64"

    def test_cpu_type(self):
        assert isinstance(BeagleBoneAI64Config().cpu, AI64CPUConfig)

    def test_dsp_type(self):
        assert isinstance(BeagleBoneAI64Config().dsp, AI64DSPConfig)

    def test_memory_type(self):
        assert isinstance(BeagleBoneAI64Config().memory, AI64MemoryConfig)

    def test_pru_count(self):
        assert BeagleBoneAI64Config().pru_count == 4

    def test_field_count(self):
        assert len(fields(BeagleBoneAI64Config)) == 8

    def test_has_dsp_field(self):
        assert hasattr(BeagleBoneAI64Config(), "dsp")


class TestCreateBeagleBoneAI64Config:
    def test_factory_returns(self):
        cfg = create_beaglebone_ai64_config()
        assert isinstance(cfg, BeagleBoneAI64Config)

    def test_defaults(self):
        cfg = create_beaglebone_ai64_config()
        assert cfg.board_name == "BeagleBone AI-64"
        assert cfg.pru_count == 4

    def test_nested_cpu_override(self):
        cfg = create_beaglebone_ai64_config(cpu={"cores": 1})
        assert cfg.cpu.cores == 1

    def test_nested_dsp_override(self):
        cfg = create_beaglebone_ai64_config(dsp={"eve_count": 2})
        assert cfg.dsp.eve_count == 2

    def test_board_name_override(self):
        cfg = create_beaglebone_ai64_config(board_name="AI64 Custom")
        assert cfg.board_name == "AI64 Custom"


class TestPackageInitImport:
    def test_import_ai64_config(self):
        from hardware.beaglebone import BeagleBoneAI64Config as AC
        assert AC is BeagleBoneAI64Config

    def test_registry_contains_ai64(self):
        from hardware.beaglebone import BOARD_REGISTRY
        assert "beaglebone-ai64" in BOARD_REGISTRY

    def test_get_board_info_ai64(self):
        from hardware.beaglebone import get_board_info
        info = get_board_info("beaglebone-ai64")
        assert info["config_class"] == "BeagleBoneAI64Config"
