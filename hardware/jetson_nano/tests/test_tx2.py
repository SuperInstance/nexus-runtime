"""
Test suite for Jetson TX2 hardware configuration.

Covers config_tx2 module with defaults, overrides, types, factories,
power modes, and marine CV optimization flags.
"""

import pytest
from dataclasses import fields, FrozenInstanceError

from hardware.jetson_nano.config_tx2 import (
    TX2CPUConfig,
    TX2GPUConfig,
    TX2MemoryConfig,
    TX2StorageConfig,
    TX2PowerConfig,
    TX2AIConfig,
    TX2PinMapping,
    JetsonTX2Config,
    create_jetson_tx2_config,
)


class TestTX2CPUConfig:
    def test_arch(self):
        assert TX2CPUConfig().arch == "ARM"

    def test_cores(self):
        assert TX2CPUConfig().cores == 6

    def test_core_type(self):
        assert TX2CPUConfig().core_type == "Cortex-A57"

    def test_clock_ghz(self):
        assert TX2CPUConfig().clock_ghz == 2.0

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            TX2CPUConfig().cores = 8

    def test_equality(self):
        assert TX2CPUConfig() == TX2CPUConfig()

    def test_field_count(self):
        assert len(fields(TX2CPUConfig)) == 4


class TestTX2GPUConfig:
    def test_name(self):
        assert TX2GPUConfig().name == "Pascal"

    def test_cuda_cores(self):
        assert TX2GPUConfig().cuda_cores == 256

    def test_tflops(self):
        assert TX2GPUConfig().tflops == 1.3

    def test_tensor_cores(self):
        assert TX2GPUConfig().tensor_cores == 0

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            TX2GPUConfig().cuda_cores = 512

    def test_equality(self):
        assert TX2GPUConfig() == TX2GPUConfig()

    def test_field_count(self):
        assert len(fields(TX2GPUConfig)) == 4


class TestTX2MemoryConfig:
    def test_ram_gb(self):
        assert TX2MemoryConfig().ram_gb == 8

    def test_bandwidth(self):
        assert TX2MemoryConfig().bandwidth_gbps == 51.2

    def test_type(self):
        assert TX2MemoryConfig().type == "LPDDR4"

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            TX2MemoryConfig().ram_gb = 16

    def test_equality(self):
        assert TX2MemoryConfig() == TX2MemoryConfig()


class TestTX2StorageConfig:
    def test_boot(self):
        assert TX2StorageConfig().boot == "eMMC 32GB"

    def test_usb3_count(self):
        assert TX2StorageConfig().usb3_count == 3

    def test_sata_support(self):
        assert TX2StorageConfig().sata_support is True

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            TX2StorageConfig().boot = "NVMe"


class TestTX2PowerConfig:
    def test_max_watts(self):
        assert TX2PowerConfig().max_watts == 15

    def test_thermal_throttle(self):
        assert TX2PowerConfig().thermal_throttle_c == 90

    def test_fan_curve(self):
        curve = TX2PowerConfig().fan_curve
        assert len(curve) == 4

    def test_power_modes_count(self):
        assert len(TX2PowerConfig().power_modes) == 4

    def test_power_mode_maxn(self):
        assert TX2PowerConfig().power_modes["MAXN"] == 15

    def test_power_mode_maxq(self):
        assert TX2PowerConfig().power_modes["MAXQ"] == 7

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            TX2PowerConfig().max_watts = 30


class TestTX2AIConfig:
    def test_precision(self):
        assert TX2AIConfig().tensorrt_precision == "FP16"

    def test_max_batch(self):
        assert TX2AIConfig().max_batch == 8

    def test_inference_threads(self):
        assert TX2AIConfig().inference_threads == 6

    def test_max_resolution(self):
        assert TX2AIConfig().max_resolution == 2560

    def test_marine_cv_optimized(self):
        assert TX2AIConfig().marine_cv_optimized is True

    def test_target_fps(self):
        assert TX2AIConfig().target_fps == 30

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            TX2AIConfig().max_batch = 16


class TestTX2PinMapping:
    def test_gps_tx(self):
        assert TX2PinMapping().GPS_TX == 14

    def test_gps_rx(self):
        assert TX2PinMapping().GPS_RX == 15

    def test_sonar_trig(self):
        assert TX2PinMapping().SONAR_TRIG == 18

    def test_water_temp(self):
        assert TX2PinMapping().WATER_TEMP == 19

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            TX2PinMapping().GPS_TX = 99

    def test_equality(self):
        assert TX2PinMapping() == TX2PinMapping()

    def test_field_count(self):
        assert len(fields(TX2PinMapping)) == 11


class TestJetsonTX2Config:
    def test_board_name(self):
        assert JetsonTX2Config().board_name == "Jetson TX2"

    def test_jetpack_version(self):
        assert JetsonTX2Config().jetpack_version == "4.6"

    def test_cpu_type(self):
        assert isinstance(JetsonTX2Config().cpu, TX2CPUConfig)

    def test_gpu_type(self):
        assert isinstance(JetsonTX2Config().gpu, TX2GPUConfig)

    def test_memory_type(self):
        assert isinstance(JetsonTX2Config().memory, TX2MemoryConfig)

    def test_storage_type(self):
        assert isinstance(JetsonTX2Config().storage, TX2StorageConfig)

    def test_power_type(self):
        assert isinstance(JetsonTX2Config().power, TX2PowerConfig)

    def test_ai_type(self):
        assert isinstance(JetsonTX2Config().ai, TX2AIConfig)

    def test_pin_map_type(self):
        assert isinstance(JetsonTX2Config().pin_map, TX2PinMapping)

    def test_field_count(self):
        assert len(fields(JetsonTX2Config)) == 9


class TestCreateJetsonTX2Config:
    def test_factory_returns(self):
        cfg = create_jetson_tx2_config()
        assert isinstance(cfg, JetsonTX2Config)

    def test_defaults(self):
        cfg = create_jetson_tx2_config()
        assert cfg.board_name == "Jetson TX2"
        assert cfg.jetpack_version == "4.6"

    def test_nested_cpu_override(self):
        cfg = create_jetson_tx2_config(cpu={"cores": 4})
        assert cfg.cpu.cores == 4

    def test_nested_gpu_override(self):
        cfg = create_jetson_tx2_config(gpu={"cuda_cores": 512})
        assert cfg.gpu.cuda_cores == 512

    def test_nested_power_override(self):
        cfg = create_jetson_tx2_config(power={"max_watts": 10})
        assert cfg.power.max_watts == 10

    def test_board_name_override(self):
        cfg = create_jetson_tx2_config(board_name="TX2 Custom")
        assert cfg.board_name == "TX2 Custom"

    def test_jetpack_override(self):
        cfg = create_jetson_tx2_config(jetpack_version="4.4")
        assert cfg.jetpack_version == "4.4"

    def test_multiple_overrides(self):
        cfg = create_jetson_tx2_config(
            board_name="Test", jetpack_version="4.5", cpu={"cores": 2}
        )
        assert cfg.board_name == "Test"
        assert cfg.jetpack_version == "4.5"
        assert cfg.cpu.cores == 2
