"""
Test suite for Jetson Xavier NX hardware configuration.

Covers config_xavier_nx module with defaults, overrides, types, factories,
Volta GPU specs, and deep learning inference configuration.
"""

import pytest
from dataclasses import fields, FrozenInstanceError

from hardware.jetson_nano.config_xavier_nx import (
    XavierNXCPUConfig,
    XavierNXGPUConfig,
    XavierNXMemoryConfig,
    XavierNXStorageConfig,
    XavierNXPowerConfig,
    XavierNXAIConfig,
    XavierNXPinMapping,
    JetsonXavierNXConfig,
    create_jetson_xavier_nx_config,
)


class TestXavierNXCPUConfig:
    def test_arch(self):
        assert XavierNXCPUConfig().arch == "ARM"

    def test_cores(self):
        assert XavierNXCPUConfig().cores == 6

    def test_core_type(self):
        assert XavierNXCPUConfig().core_type == "Carmel"

    def test_clock_ghz(self):
        assert XavierNXCPUConfig().clock_ghz == 1.9

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            XavierNXCPUConfig().cores = 8

    def test_equality(self):
        assert XavierNXCPUConfig() == XavierNXCPUConfig()

    def test_field_count(self):
        assert len(fields(XavierNXCPUConfig)) == 4


class TestXavierNXGPUConfig:
    def test_name(self):
        assert XavierNXGPUConfig().name == "Volta"

    def test_cuda_cores(self):
        assert XavierNXGPUConfig().cuda_cores == 384

    def test_tflops(self):
        assert XavierNXGPUConfig().tflops == 1.93

    def test_tensor_cores(self):
        assert XavierNXGPUConfig().tensor_cores == 48

    def test_tops(self):
        assert XavierNXGPUConfig().tops == 21

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            XavierNXGPUConfig().cuda_cores = 512

    def test_equality(self):
        assert XavierNXGPUConfig() == XavierNXGPUConfig()

    def test_field_count(self):
        assert len(fields(XavierNXGPUConfig)) == 5

    def test_has_tensor_cores(self):
        assert XavierNXGPUConfig().tensor_cores > 0


class TestXavierNXMemoryConfig:
    def test_ram_gb(self):
        assert XavierNXMemoryConfig().ram_gb == 8

    def test_bandwidth(self):
        assert XavierNXMemoryConfig().bandwidth_gbps == 59.7

    def test_type(self):
        assert XavierNXMemoryConfig().type == "LPDDR4x"

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            XavierNXMemoryConfig().ram_gb = 16

    def test_equality(self):
        assert XavierNXMemoryConfig() == XavierNXMemoryConfig()


class TestXavierNXStorageConfig:
    def test_boot(self):
        assert XavierNXStorageConfig().boot == "microSD / NVMe"

    def test_usb3_count(self):
        assert XavierNXStorageConfig().usb3_count == 4

    def test_pcie_gen(self):
        assert XavierNXStorageConfig().pcie_gen == 3

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            XavierNXStorageConfig().pcie_gen = 4


class TestXavierNXPowerConfig:
    def test_max_watts(self):
        assert XavierNXPowerConfig().max_watts == 20

    def test_thermal_throttle(self):
        assert XavierNXPowerConfig().thermal_throttle_c == 95

    def test_fan_curve_count(self):
        assert len(XavierNXPowerConfig().fan_curve) == 5

    def test_power_modes_count(self):
        assert len(XavierNXPowerConfig().power_modes) == 3

    def test_power_mode_20w(self):
        assert XavierNXPowerConfig().power_modes["20W"] == 20

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            XavierNXPowerConfig().max_watts = 30


class TestXavierNXAIConfig:
    def test_precision(self):
        assert XavierNXAIConfig().tensorrt_precision == "FP16"

    def test_max_batch(self):
        assert XavierNXAIConfig().max_batch == 16

    def test_inference_threads(self):
        assert XavierNXAIConfig().inference_threads == 6

    def test_max_resolution(self):
        assert XavierNXAIConfig().max_resolution == 3840

    def test_dlas_supported(self):
        assert XavierNXAIConfig().dlas_supported is True

    def test_triton_not_supported(self):
        assert XavierNXAIConfig().triton_supported is False

    def test_dl_model_cache_mb(self):
        assert XavierNXAIConfig().dl_model_cache_mb == 512

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            XavierNXAIConfig().max_batch = 32


class TestXavierNXPinMapping:
    def test_gps_tx(self):
        assert XavierNXPinMapping().GPS_TX == 14

    def test_has_cam1(self):
        assert hasattr(XavierNXPinMapping(), "CAM1_CLK")

    def test_cam1_clk(self):
        assert XavierNXPinMapping().CAM1_CLK == 29

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            XavierNXPinMapping().GPS_TX = 99

    def test_equality(self):
        assert XavierNXPinMapping() == XavierNXPinMapping()

    def test_field_count(self):
        assert len(fields(XavierNXPinMapping)) == 12


class TestJetsonXavierNXConfig:
    def test_board_name(self):
        assert JetsonXavierNXConfig().board_name == "Jetson Xavier NX"

    def test_jetpack_version(self):
        assert JetsonXavierNXConfig().jetpack_version == "5.0"

    def test_cpu_type(self):
        assert isinstance(JetsonXavierNXConfig().cpu, XavierNXCPUConfig)

    def test_gpu_type(self):
        assert isinstance(JetsonXavierNXConfig().gpu, XavierNXGPUConfig)

    def test_memory_type(self):
        assert isinstance(JetsonXavierNXConfig().memory, XavierNXMemoryConfig)

    def test_power_type(self):
        assert isinstance(JetsonXavierNXConfig().power, XavierNXPowerConfig)

    def test_ai_type(self):
        assert isinstance(JetsonXavierNXConfig().ai, XavierNXAIConfig)

    def test_field_count(self):
        assert len(fields(JetsonXavierNXConfig)) == 9


class TestCreateJetsonXavierNXConfig:
    def test_factory_returns(self):
        cfg = create_jetson_xavier_nx_config()
        assert isinstance(cfg, JetsonXavierNXConfig)

    def test_defaults(self):
        cfg = create_jetson_xavier_nx_config()
        assert cfg.board_name == "Jetson Xavier NX"
        assert cfg.jetpack_version == "5.0"

    def test_nested_cpu_override(self):
        cfg = create_jetson_xavier_nx_config(cpu={"cores": 4})
        assert cfg.cpu.cores == 4

    def test_nested_gpu_override(self):
        cfg = create_jetson_xavier_nx_config(gpu={"tensor_cores": 64})
        assert cfg.gpu.tensor_cores == 64

    def test_nested_ai_override(self):
        cfg = create_jetson_xavier_nx_config(ai={"triton_supported": True})
        assert cfg.ai.triton_supported is True

    def test_board_name_override(self):
        cfg = create_jetson_xavier_nx_config(board_name="Xavier Custom")
        assert cfg.board_name == "Xavier Custom"
