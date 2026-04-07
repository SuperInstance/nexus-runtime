"""
Test suite for Jetson Orin NX hardware configuration.

Covers config_orin_nx module with defaults, overrides, types, factories,
Ampere GPU specs, and fleet commander configuration.
"""

import pytest
from dataclasses import fields, FrozenInstanceError

from hardware.jetson_nano.config_orin_nx import (
    OrinNXCPUConfig,
    OrinNXGPUConfig,
    OrinNXMemoryConfig,
    OrinNXStorageConfig,
    OrinNXPowerConfig,
    OrinNXAIConfig,
    OrinNXPinMapping,
    JetsonOrinNXConfig,
    create_jetson_orin_nx_config,
)


class TestOrinNXCPUConfig:
    def test_arch(self):
        assert OrinNXCPUConfig().arch == "ARM"

    def test_cores(self):
        assert OrinNXCPUConfig().cores == 8

    def test_core_type(self):
        assert OrinNXCPUConfig().core_type == "Carmel"

    def test_clock_ghz(self):
        assert OrinNXCPUConfig().clock_ghz == 2.0

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            OrinNXCPUConfig().cores = 12

    def test_equality(self):
        assert OrinNXCPUConfig() == OrinNXCPUConfig()

    def test_field_count(self):
        assert len(fields(OrinNXCPUConfig)) == 4


class TestOrinNXGPUConfig:
    def test_name(self):
        assert OrinNXGPUConfig().name == "Ampere"

    def test_cuda_cores(self):
        assert OrinNXGPUConfig().cuda_cores == 1024

    def test_tflops(self):
        assert OrinNXGPUConfig().tflops == 5.0

    def test_tensor_cores(self):
        assert OrinNXGPUConfig().tensor_cores == 32

    def test_tops(self):
        assert OrinNXGPUConfig().tops == 100

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            OrinNXGPUConfig().cuda_cores = 2048

    def test_equality(self):
        assert OrinNXGPUConfig() == OrinNXGPUConfig()

    def test_field_count(self):
        assert len(fields(OrinNXGPUConfig)) == 5


class TestOrinNXMemoryConfig:
    def test_ram_gb(self):
        assert OrinNXMemoryConfig().ram_gb == 16

    def test_bandwidth(self):
        assert OrinNXMemoryConfig().bandwidth_gbps == 102.4

    def test_type(self):
        assert OrinNXMemoryConfig().type == "LPDDR5"

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            OrinNXMemoryConfig().ram_gb = 32

    def test_equality(self):
        assert OrinNXMemoryConfig() == OrinNXMemoryConfig()


class TestOrinNXStorageConfig:
    def test_boot(self):
        assert OrinNXStorageConfig().boot == "NVMe SSD"

    def test_usb3_count(self):
        assert OrinNXStorageConfig().usb3_count == 4

    def test_pcie_gen(self):
        assert OrinNXStorageConfig().pcie_gen == 4

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            OrinNXStorageConfig().pcie_gen = 5


class TestOrinNXPowerConfig:
    def test_max_watts(self):
        assert OrinNXPowerConfig().max_watts == 25

    def test_thermal_throttle(self):
        assert OrinNXPowerConfig().thermal_throttle_c == 100

    def test_fan_curve_count(self):
        assert len(OrinNXPowerConfig().fan_curve) == 5

    def test_power_modes_count(self):
        assert len(OrinNXPowerConfig().power_modes) == 3

    def test_power_mode_25w(self):
        assert OrinNXPowerConfig().power_modes["25W"] == 25

    def test_fan_curve_start_at_35(self):
        curve = OrinNXPowerConfig().fan_curve
        assert 35 in curve
        assert curve[35] == 0

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            OrinNXPowerConfig().max_watts = 30


class TestOrinNXAIConfig:
    def test_precision(self):
        assert OrinNXAIConfig().tensorrt_precision == "FP16"

    def test_max_batch(self):
        assert OrinNXAIConfig().max_batch == 32

    def test_inference_threads(self):
        assert OrinNXAIConfig().inference_threads == 8

    def test_max_resolution(self):
        assert OrinNXAIConfig().max_resolution == 7680

    def test_dlas_supported(self):
        assert OrinNXAIConfig().dlas_supported is True

    def test_triton_supported(self):
        assert OrinNXAIConfig().triton_supported is True

    def test_dl_model_cache_mb(self):
        assert OrinNXAIConfig().dl_model_cache_mb == 2048

    def test_fleet_commander_mode(self):
        assert OrinNXAIConfig().fleet_commander_mode is True

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            OrinNXAIConfig().max_batch = 64


class TestOrinNXPinMapping:
    def test_gps_tx(self):
        assert OrinNXPinMapping().GPS_TX == 14

    def test_has_cam1(self):
        assert hasattr(OrinNXPinMapping(), "CAM1_CLK")

    def test_has_cam2(self):
        assert hasattr(OrinNXPinMapping(), "CAM2_CLK")

    def test_cam2_clk(self):
        assert OrinNXPinMapping().CAM2_CLK == 30

    def test_cam2_data0(self):
        assert OrinNXPinMapping().CAM2_DATA0 == 58

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            OrinNXPinMapping().GPS_TX = 99

    def test_equality(self):
        assert OrinNXPinMapping() == OrinNXPinMapping()

    def test_field_count(self):
        assert len(fields(OrinNXPinMapping)) == 14


class TestJetsonOrinNXConfig:
    def test_board_name(self):
        assert JetsonOrinNXConfig().board_name == "Jetson Orin NX 16GB"

    def test_jetpack_version(self):
        assert JetsonOrinNXConfig().jetpack_version == "5.1"

    def test_cpu_type(self):
        assert isinstance(JetsonOrinNXConfig().cpu, OrinNXCPUConfig)

    def test_gpu_type(self):
        assert isinstance(JetsonOrinNXConfig().gpu, OrinNXGPUConfig)

    def test_memory_type(self):
        assert isinstance(JetsonOrinNXConfig().memory, OrinNXMemoryConfig)

    def test_power_type(self):
        assert isinstance(JetsonOrinNXConfig().power, OrinNXPowerConfig)

    def test_ai_type(self):
        assert isinstance(JetsonOrinNXConfig().ai, OrinNXAIConfig)

    def test_field_count(self):
        assert len(fields(JetsonOrinNXConfig)) == 9


class TestCreateJetsonOrinNXConfig:
    def test_factory_returns(self):
        cfg = create_jetson_orin_nx_config()
        assert isinstance(cfg, JetsonOrinNXConfig)

    def test_defaults(self):
        cfg = create_jetson_orin_nx_config()
        assert cfg.board_name == "Jetson Orin NX 16GB"
        assert cfg.jetpack_version == "5.1"

    def test_nested_cpu_override(self):
        cfg = create_jetson_orin_nx_config(cpu={"cores": 6})
        assert cfg.cpu.cores == 6

    def test_nested_gpu_override(self):
        cfg = create_jetson_orin_nx_config(gpu={"cuda_cores": 2048})
        assert cfg.gpu.cuda_cores == 2048

    def test_nested_ai_override(self):
        cfg = create_jetson_orin_nx_config(ai={"max_batch": 64})
        assert cfg.ai.max_batch == 64

    def test_board_name_override(self):
        cfg = create_jetson_orin_nx_config(board_name="Orin NX Custom")
        assert cfg.board_name == "Orin NX Custom"

    def test_fleet_commander_override(self):
        cfg = create_jetson_orin_nx_config(ai={"fleet_commander_mode": False})
        assert cfg.ai.fleet_commander_mode is False
