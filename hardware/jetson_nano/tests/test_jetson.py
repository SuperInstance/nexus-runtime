"""
Comprehensive test suite for NEXUS Jetson hardware configurations.

Covers config_nano, config_orin_nano, config_agx_orin, and ai_pipeline modules
with 100+ tests covering defaults, overrides, types, enums, factories, and
pipeline profiles.
"""

import pytest
from dataclasses import fields, FrozenInstanceError, replace

# ── config_nano ────────────────────────────────────────────────────────────
from hardware.jetson_nano.config_nano import (
    CPUConfig,
    GPUConfig,
    MemoryConfig,
    StorageConfig,
    PowerConfig,
    AIConfig,
    PinMapping,
    JetsonConfig,
    create_jetson_nano_config,
)

# ── config_orin_nano ───────────────────────────────────────────────────────
from hardware.jetson_nano.config_orin_nano import (
    OrinNanoCPUConfig,
    OrinNanoGPUConfig,
    OrinNanoMemoryConfig,
    OrinNanoStorageConfig,
    OrinNanoPowerConfig,
    OrinNanoAIConfig,
    OrinNanoPinMapping,
    JetsonOrinNanoConfig,
    create_jetson_orin_nano_config,
)

# ── config_agx_orin ────────────────────────────────────────────────────────
from hardware.jetson_nano.config_agx_orin import (
    AGXCPUConfig,
    AGXGPUConfig,
    AGXMemoryConfig,
    AGXStorageConfig,
    AGXPowerConfig,
    AGXAIConfig,
    AGXPinMapping,
    JetsonAGXOrinConfig,
    create_jetson_agx_orin_config,
)

# ── ai_pipeline ────────────────────────────────────────────────────────────
from hardware.jetson_nano.ai_pipeline import (
    ObjectDetector,
    CameraConfig,
    ModelConfig,
    PerceptionPipeline,
    get_pipeline_profile,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. CPUConfig (Jetson Nano 4GB)
# ═══════════════════════════════════════════════════════════════════════════

class TestCPUConfig:
    def test_default_arch(self):
        assert CPUConfig().arch == "ARM"

    def test_default_cores(self):
        assert CPUConfig().cores == 4

    def test_default_core_type(self):
        assert CPUConfig().core_type == "Cortex-A57"

    def test_default_clock_ghz(self):
        assert CPUConfig().clock_ghz == 1.43

    def test_custom_arch(self):
        cfg = CPUConfig(arch="ARM64")
        assert cfg.arch == "ARM64"

    def test_frozen_immutability(self):
        cfg = CPUConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.cores = 8

    def test_equality(self):
        a = CPUConfig()
        b = CPUConfig()
        assert a == b

    def test_field_count(self):
        assert len(fields(CPUConfig)) == 4

    def test_clock_ghz_is_float(self):
        assert isinstance(CPUConfig().clock_ghz, float)

    def test_cores_is_int(self):
        assert isinstance(CPUConfig().cores, int)


# ═══════════════════════════════════════════════════════════════════════════
# 2. GPUConfig (Jetson Nano 4GB)
# ═══════════════════════════════════════════════════════════════════════════

class TestGPUConfig:
    def test_default_name(self):
        assert GPUConfig().name == "Maxwell"

    def test_default_cuda_cores(self):
        assert GPUConfig().cuda_cores == 128

    def test_default_tflops(self):
        assert GPUConfig().tflops == 0.47

    def test_default_tensor_cores(self):
        assert GPUConfig().tensor_cores == 0

    def test_frozen(self):
        cfg = GPUConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.cuda_cores = 256

    def test_equality(self):
        assert GPUConfig() == GPUConfig()

    def test_custom_tensor_cores(self):
        cfg = GPUConfig(tensor_cores=8)
        assert cfg.tensor_cores == 8

    def test_field_count(self):
        assert len(fields(GPUConfig)) == 4

    def test_tflops_is_float(self):
        assert isinstance(GPUConfig().tflops, float)

    def test_tensor_cores_zero(self):
        assert GPUConfig().tensor_cores == 0


# ═══════════════════════════════════════════════════════════════════════════
# 3. MemoryConfig (Jetson Nano 4GB)
# ═══════════════════════════════════════════════════════════════════════════

class TestMemoryConfig:
    def test_default_ram_gb(self):
        assert MemoryConfig().ram_gb == 4

    def test_default_bandwidth_gbps(self):
        assert MemoryConfig().bandwidth_gbps == 25.6

    def test_default_type(self):
        assert MemoryConfig().type == "LPDDR4"

    def test_frozen(self):
        cfg = MemoryConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.ram_gb = 8

    def test_equality(self):
        assert MemoryConfig() == MemoryConfig()

    def test_field_count(self):
        assert len(fields(MemoryConfig)) == 3

    def test_bandwidth_is_float(self):
        assert isinstance(MemoryConfig().bandwidth_gbps, float)

    def test_ram_is_int(self):
        assert isinstance(MemoryConfig().ram_gb, int)

    def test_custom_ram(self):
        assert MemoryConfig(ram_gb=2).ram_gb == 2


# ═══════════════════════════════════════════════════════════════════════════
# 4. StorageConfig (Jetson Nano 4GB)
# ═══════════════════════════════════════════════════════════════════════════

class TestStorageConfig:
    def test_default_boot(self):
        assert StorageConfig().boot == "microSD"

    def test_default_usb3_count(self):
        assert StorageConfig().usb3_count == 4

    def test_frozen(self):
        cfg = StorageConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.boot = "NVMe"

    def test_equality(self):
        assert StorageConfig() == StorageConfig()

    def test_field_count(self):
        assert len(fields(StorageConfig)) == 2

    def test_custom_usb3(self):
        assert StorageConfig(usb3_count=2).usb3_count == 2

    def test_usb3_is_int(self):
        assert isinstance(StorageConfig().usb3_count, int)


# ═══════════════════════════════════════════════════════════════════════════
# 5. PowerConfig (Jetson Nano 4GB)
# ═══════════════════════════════════════════════════════════════════════════

class TestPowerConfig:
    def test_default_max_watts(self):
        assert PowerConfig().max_watts == 20

    def test_default_thermal_throttle(self):
        assert PowerConfig().thermal_throttle_c == 85

    def test_default_fan_curve(self):
        curve = PowerConfig().fan_curve
        assert curve == {40: 0, 60: 50, 80: 100}

    def test_fan_curve_keys(self):
        assert set(PowerConfig().fan_curve.keys()) == {40, 60, 80}

    def test_fan_curve_at_80(self):
        assert PowerConfig().fan_curve[80] == 100

    def test_frozen(self):
        cfg = PowerConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.max_watts = 30

    def test_field_count(self):
        assert len(fields(PowerConfig)) == 3

    def test_equality(self):
        assert PowerConfig() == PowerConfig()

    def test_custom_max_watts(self):
        assert PowerConfig(max_watts=15).max_watts == 15


# ═══════════════════════════════════════════════════════════════════════════
# 6. AIConfig (Jetson Nano 4GB)
# ═══════════════════════════════════════════════════════════════════════════

class TestAIConfig:
    def test_default_precision(self):
        assert AIConfig().tensorrt_precision == "FP16"

    def test_default_max_batch(self):
        assert AIConfig().max_batch == 8

    def test_default_inference_threads(self):
        assert AIConfig().inference_threads == 4

    def test_default_max_resolution(self):
        assert AIConfig().max_resolution == 1920

    def test_frozen(self):
        cfg = AIConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.max_batch = 16

    def test_equality(self):
        assert AIConfig() == AIConfig()

    def test_field_count(self):
        assert len(fields(AIConfig)) == 4

    def test_custom_precision(self):
        assert AIConfig(tensorrt_precision="INT8").tensorrt_precision == "INT8"

    def test_resolution_is_int(self):
        assert isinstance(AIConfig().max_resolution, int)


# ═══════════════════════════════════════════════════════════════════════════
# 7. PinMapping (Jetson Nano 4GB)
# ═══════════════════════════════════════════════════════════════════════════

class TestPinMapping:
    def test_gps_tx(self):
        assert PinMapping().GPS_TX == 14

    def test_gps_rx(self):
        assert PinMapping().GPS_RX == 15

    def test_imu_sda(self):
        assert PinMapping().IMU_SDA == 1

    def test_imu_scl(self):
        assert PinMapping().IMU_SCL == 0

    def test_sonar_trig(self):
        assert PinMapping().SONAR_TRIG == 18

    def test_sonar_echo(self):
        assert PinMapping().SONAR_ECHO == 17

    def test_cam0_clk(self):
        assert PinMapping().CAM0_CLK == 28

    def test_cam0_data0(self):
        assert PinMapping().CAM0_DATA0 == 56

    def test_thruster_i2c(self):
        assert PinMapping().THRUSTER_I2C == 1

    def test_led(self):
        assert PinMapping().LED == 33

    def test_frozen(self):
        cfg = PinMapping()
        with pytest.raises(FrozenInstanceError):
            cfg.GPS_TX = 99

    def test_equality(self):
        assert PinMapping() == PinMapping()

    def test_field_count(self):
        assert len(fields(PinMapping)) == 10


# ═══════════════════════════════════════════════════════════════════════════
# 8. JetsonConfig & create_jetson_nano_config
# ═══════════════════════════════════════════════════════════════════════════

class TestJetsonConfig:
    def test_default_board_name(self):
        assert JetsonConfig().board_name == "Jetson Nano 4GB"

    def test_default_jetpack(self):
        assert JetsonConfig().jetpack_version == "4.6"

    def test_cpu_is_cpuconfig(self):
        assert isinstance(JetsonConfig().cpu, CPUConfig)

    def test_gpu_is_gpuconfig(self):
        assert isinstance(JetsonConfig().gpu, GPUConfig)

    def test_memory_is_memoryconfig(self):
        assert isinstance(JetsonConfig().memory, MemoryConfig)

    def test_storage_is_storageconfig(self):
        assert isinstance(JetsonConfig().storage, StorageConfig)

    def test_power_is_powerconfig(self):
        assert isinstance(JetsonConfig().power, PowerConfig)

    def test_ai_is_aiconfig(self):
        assert isinstance(JetsonConfig().ai, AIConfig)

    def test_pin_map_is_pinmapping(self):
        assert isinstance(JetsonConfig().pin_map, PinMapping)

    def test_field_count(self):
        assert len(fields(JetsonConfig)) == 9


class TestCreateJetsonNanoConfig:
    def test_factory_returns_jetson_config(self):
        cfg = create_jetson_nano_config()
        assert isinstance(cfg, JetsonConfig)

    def test_factory_defaults(self):
        cfg = create_jetson_nano_config()
        assert cfg.board_name == "Jetson Nano 4GB"
        assert cfg.jetpack_version == "4.6"

    def test_override_board_name(self):
        cfg = create_jetson_nano_config(board_name="Custom Board")
        assert cfg.board_name == "Custom Board"

    def test_override_jetpack(self):
        cfg = create_jetson_nano_config(jetpack_version="4.5")
        assert cfg.jetpack_version == "4.5"

    def test_override_nested_cpu(self):
        cfg = create_jetson_nano_config(cpu={"cores": 2})
        assert cfg.cpu.cores == 2

    def test_override_nested_gpu(self):
        cfg = create_jetson_nano_config(gpu={"cuda_cores": 256})
        assert cfg.gpu.cuda_cores == 256

    def test_override_nested_memory(self):
        cfg = create_jetson_nano_config(memory={"ram_gb": 8})
        assert cfg.memory.ram_gb == 8

    def test_override_nested_power(self):
        cfg = create_jetson_nano_config(power={"max_watts": 15})
        assert cfg.power.max_watts == 15

    def test_multiple_overrides(self):
        cfg = create_jetson_nano_config(
            board_name="Test", jetpack_version="4.4", cpu={"cores": 1}
        )
        assert cfg.board_name == "Test"
        assert cfg.jetpack_version == "4.4"
        assert cfg.cpu.cores == 1


# ═══════════════════════════════════════════════════════════════════════════
# 9. Orin Nano Config
# ═══════════════════════════════════════════════════════════════════════════

class TestOrinNanoCPUConfig:
    def test_cores(self):
        assert OrinNanoCPUConfig().cores == 6

    def test_core_type(self):
        assert OrinNanoCPUConfig().core_type == "Cortex-A78AE"

    def test_clock(self):
        assert OrinNanoCPUConfig().clock_ghz == 1.5

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            OrinNanoCPUConfig().cores = 8

class TestOrinNanoGPUConfig:
    def test_name(self):
        assert OrinNanoGPUConfig().name == "Ampere"

    def test_cuda_cores(self):
        assert OrinNanoGPUConfig().cuda_cores == 1024

    def test_tops(self):
        assert OrinNanoGPUConfig().tops == 40

    def test_tensor_cores(self):
        assert OrinNanoGPUConfig().tensor_cores == 32

class TestOrinNanoMemoryConfig:
    def test_ram_gb(self):
        assert OrinNanoMemoryConfig().ram_gb == 8

    def test_bandwidth(self):
        assert OrinNanoMemoryConfig().bandwidth_gbps == 68.0

    def test_type(self):
        assert OrinNanoMemoryConfig().type == "LPDDR5"

class TestOrinNanoPowerConfig:
    def test_max_watts(self):
        assert OrinNanoPowerConfig().max_watts == 25

    def test_thermal_throttle(self):
        assert OrinNanoPowerConfig().thermal_throttle_c == 90

    def test_fan_curve_4_points(self):
        curve = OrinNanoPowerConfig().fan_curve
        assert len(curve) == 4

    def test_power_modes(self):
        modes = OrinNanoPowerConfig().power_modes
        assert "15W" in modes
        assert "25W" in modes

class TestOrinNanoAIConfig:
    def test_max_batch(self):
        assert OrinNanoAIConfig().max_batch == 16

    def test_max_resolution(self):
        assert OrinNanoAIConfig().max_resolution == 3840

    def test_dlas_supported(self):
        assert OrinNanoAIConfig().dlas_supported is True

class TestJetsonOrinNanoConfig:
    def test_board_name(self):
        assert JetsonOrinNanoConfig().board_name == "Jetson Orin Nano 8GB"

    def test_jetpack(self):
        assert JetsonOrinNanoConfig().jetpack_version == "5.1"

    def test_cpu_type(self):
        assert isinstance(JetsonOrinNanoConfig().cpu, OrinNanoCPUConfig)

    def test_gpu_type(self):
        assert isinstance(JetsonOrinNanoConfig().gpu, OrinNanoGPUConfig)

class TestCreateOrinNanoConfig:
    def test_factory_returns(self):
        cfg = create_jetson_orin_nano_config()
        assert isinstance(cfg, JetsonOrinNanoConfig)

    def test_nested_override(self):
        cfg = create_jetson_orin_nano_config(cpu={"cores": 4})
        assert cfg.cpu.cores == 4

    def test_board_name_override(self):
        cfg = create_jetson_orin_nano_config(board_name="Orin Custom")
        assert cfg.board_name == "Orin Custom"


# ═══════════════════════════════════════════════════════════════════════════
# 10. AGX Orin Config
# ═══════════════════════════════════════════════════════════════════════════

class TestAGXCPUConfig:
    def test_cores(self):
        assert AGXCPUConfig().cores == 12

    def test_clock(self):
        assert AGXCPUConfig().clock_ghz == 2.2

    def test_core_type(self):
        assert AGXCPUConfig().core_type == "Cortex-A78AE"

class TestAGXGPUConfig:
    def test_cuda_cores(self):
        assert AGXGPUConfig().cuda_cores == 2048

    def test_tops(self):
        assert AGXGPUConfig().tops == 275

    def test_tensor_cores(self):
        assert AGXGPUConfig().tensor_cores == 64

class TestAGXMemoryConfig:
    def test_ram_gb(self):
        assert AGXMemoryConfig().ram_gb == 64

    def test_bandwidth(self):
        assert AGXMemoryConfig().bandwidth_gbps == 204.8

    def test_type(self):
        assert AGXMemoryConfig().type == "LPDDR5"

class TestAGXPowerConfig:
    def test_max_watts(self):
        assert AGXPowerConfig().max_watts == 60

    def test_thermal_throttle(self):
        assert AGXPowerConfig().thermal_throttle_c == 100

    def test_power_modes_count(self):
        assert len(AGXPowerConfig().power_modes) == 4

    def test_fan_curve_5_points(self):
        assert len(AGXPowerConfig().fan_curve) == 5

class TestAGXAIConfig:
    def test_max_batch(self):
        assert AGXAIConfig().max_batch == 32

    def test_max_resolution(self):
        assert AGXAIConfig().max_resolution == 7680

    def test_triton_supported(self):
        assert AGXAIConfig().triton_supported is True

    def test_dlas_supported(self):
        assert AGXAIConfig().dlas_supported is True

class TestAGXPinMapping:
    def test_has_cam1(self):
        assert hasattr(AGXPinMapping(), "CAM1_CLK")

    def test_cam1_clk(self):
        assert AGXPinMapping().CAM1_CLK == 29

    def test_cam1_data0(self):
        assert AGXPinMapping().CAM1_DATA0 == 57

class TestJetsonAGXOrinConfig:
    def test_board_name(self):
        assert JetsonAGXOrinConfig().board_name == "Jetson AGX Orin 64GB"

    def test_jetpack(self):
        assert JetsonAGXOrinConfig().jetpack_version == "5.1"

    def test_all_subconfigs(self):
        cfg = JetsonAGXOrinConfig()
        assert isinstance(cfg.cpu, AGXCPUConfig)
        assert isinstance(cfg.gpu, AGXGPUConfig)
        assert isinstance(cfg.memory, AGXMemoryConfig)
        assert isinstance(cfg.power, AGXPowerConfig)
        assert isinstance(cfg.ai, AGXAIConfig)

class TestCreateAGXOrinConfig:
    def test_factory_returns(self):
        cfg = create_jetson_agx_orin_config()
        assert isinstance(cfg, JetsonAGXOrinConfig)

    def test_nested_override(self):
        cfg = create_jetson_agx_orin_config(cpu={"cores": 8})
        assert cfg.cpu.cores == 8

    def test_board_name_override(self):
        cfg = create_jetson_agx_orin_config(board_name="AGX Custom")
        assert cfg.board_name == "AGX Custom"


# ═══════════════════════════════════════════════════════════════════════════
# 11. Cross-platform comparison tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossPlatform:
    def test_nano_less_cores_than_orin_nano(self):
        nano = create_jetson_nano_config()
        orin = create_jetson_orin_nano_config()
        assert nano.cpu.cores < orin.cpu.cores

    def test_orin_nano_less_cores_than_agx(self):
        orin = create_jetson_orin_nano_config()
        agx = create_jetson_agx_orin_config()
        assert orin.cpu.cores < agx.cpu.cores

    def test_gpu_tops_scaling(self):
        nano = create_jetson_nano_config()
        orin = create_jetson_orin_nano_config()
        agx = create_jetson_agx_orin_config()
        assert nano.gpu.cuda_cores < orin.gpu.cuda_cores < agx.gpu.cuda_cores

    def test_ram_scaling(self):
        configs = [
            create_jetson_nano_config(),
            create_jetson_orin_nano_config(),
            create_jetson_agx_orin_config(),
        ]
        rams = [c.memory.ram_gb for c in configs]
        assert rams == sorted(rams)

    def test_bandwidth_scaling(self):
        configs = [
            create_jetson_nano_config(),
            create_jetson_orin_nano_config(),
            create_jetson_agx_orin_config(),
        ]
        bw = [c.memory.bandwidth_gbps for c in configs]
        assert bw == sorted(bw)

    def test_power_scaling(self):
        configs = [
            create_jetson_nano_config(),
            create_jetson_orin_nano_config(),
            create_jetson_agx_orin_config(),
        ]
        watts = [c.power.max_watts for c in configs]
        assert watts == sorted(watts)

    def test_nano_no_tensor_cores(self):
        assert create_jetson_nano_config().gpu.tensor_cores == 0

    def test_orin_nano_has_tensor_cores(self):
        assert create_jetson_orin_nano_config().gpu.tensor_cores > 0

    def test_agx_has_most_tensor_cores(self):
        nano = create_jetson_nano_config()
        orin = create_jetson_orin_nano_config()
        agx = create_jetson_agx_orin_config()
        assert agx.gpu.tensor_cores > orin.gpu.tensor_cores > nano.gpu.tensor_cores


# ═══════════════════════════════════════════════════════════════════════════
# 12. ObjectDetector enum
# ═══════════════════════════════════════════════════════════════════════════

class TestObjectDetector:
    def test_yolov5n(self):
        assert ObjectDetector.YOLOV5N == "yolov5n"

    def test_yolov8s(self):
        assert ObjectDetector.YOLOV8S == "yolov8s"

    def test_yolov8m(self):
        assert ObjectDetector.YOLOV8M == "yolov8m"

    def test_yolov8l(self):
        assert ObjectDetector.YOLOV8L == "yolov8l"

    def test_count(self):
        assert len(ObjectDetector) == 4

    def test_is_str(self):
        assert isinstance(ObjectDetector.YOLOV8S, str)

    def test_iteration(self):
        members = list(ObjectDetector)
        assert len(members) == 4

    def test_value_uniqueness(self):
        values = [m.value for m in ObjectDetector]
        assert len(values) == len(set(values))


# ═══════════════════════════════════════════════════════════════════════════
# 13. CameraConfig
# ═══════════════════════════════════════════════════════════════════════════

class TestCameraConfig:
    def test_default_resolution(self):
        assert CameraConfig().resolution == (1280, 720)

    def test_default_fps(self):
        assert CameraConfig().fps == 15

    def test_default_codec(self):
        assert CameraConfig().codec == "H264"

    def test_underwater_mode(self):
        assert CameraConfig().underwater_mode is True

    def test_iso_range(self):
        assert CameraConfig().iso_range == (100, 1600)

    def test_exposure_ms(self):
        assert CameraConfig().exposure_ms == (1, 33)

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            CameraConfig().fps = 30

    def test_equality(self):
        assert CameraConfig() == CameraConfig()

    def test_field_count(self):
        assert len(fields(CameraConfig)) == 6

    def test_resolution_is_tuple(self):
        assert isinstance(CameraConfig().resolution, tuple)

    def test_custom_fps(self):
        assert CameraConfig(fps=30).fps == 30

    def test_underwater_false(self):
        assert CameraConfig(underwater_mode=False).underwater_mode is False


# ═══════════════════════════════════════════════════════════════════════════
# 14. ModelConfig
# ═══════════════════════════════════════════════════════════════════════════

class TestModelConfig:
    def test_default_model_name(self):
        assert ModelConfig().model_name == "yolov8s"

    def test_default_input_size(self):
        assert ModelConfig().input_size == 640

    def test_default_confidence(self):
        assert ModelConfig().confidence == 0.5

    def test_default_nms(self):
        assert ModelConfig().nms_threshold == 0.45

    def test_default_max_detections(self):
        assert ModelConfig().max_detections == 100

    def test_field_count(self):
        assert len(fields(ModelConfig)) == 5

    def test_equality(self):
        assert ModelConfig() == ModelConfig()

    def test_mutable(self):
        cfg = ModelConfig()
        cfg.confidence = 0.7
        assert cfg.confidence == 0.7

    def test_custom_model(self):
        assert ModelConfig(model_name="yolov5n").model_name == "yolov5n"

    def test_custom_input_size(self):
        assert ModelConfig(input_size=1280).input_size == 1280


# ═══════════════════════════════════════════════════════════════════════════
# 15. PerceptionPipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestPerceptionPipeline:
    def test_default_stages(self):
        assert PerceptionPipeline().stages == ["detection", "tracking"]

    def test_default_detection(self):
        assert isinstance(PerceptionPipeline().detection, ModelConfig)

    def test_default_segmentation(self):
        assert isinstance(PerceptionPipeline().segmentation, ModelConfig)

    def test_default_depth_estimation(self):
        assert PerceptionPipeline().depth_estimation is True

    def test_default_tracking(self):
        assert PerceptionPipeline().tracking is True

    def test_field_count(self):
        assert len(fields(PerceptionPipeline)) == 5

    def test_custom_stages(self):
        pipeline = PerceptionPipeline(stages=["detection"])
        assert pipeline.stages == ["detection"]

    def test_no_tracking(self):
        pipeline = PerceptionPipeline(tracking=False)
        assert pipeline.tracking is False

    def test_no_depth(self):
        pipeline = PerceptionPipeline(depth_estimation=False)
        assert pipeline.depth_estimation is False


# ═══════════════════════════════════════════════════════════════════════════
# 16. get_pipeline_profile
# ═══════════════════════════════════════════════════════════════════════════

class TestGetPipelineProfile:
    def test_low_power_returns_pipeline(self):
        assert isinstance(get_pipeline_profile("low_power"), PerceptionPipeline)

    def test_balanced_returns_pipeline(self):
        assert isinstance(get_pipeline_profile("balanced"), PerceptionPipeline)

    def test_high_performance_returns_pipeline(self):
        assert isinstance(get_pipeline_profile("high_performance"), PerceptionPipeline)

    def test_low_power_detection_model(self):
        p = get_pipeline_profile("low_power")
        assert p.detection.model_name == "yolov5n"

    def test_low_power_input_size(self):
        p = get_pipeline_profile("low_power")
        assert p.detection.input_size == 416

    def test_low_power_no_tracking(self):
        p = get_pipeline_profile("low_power")
        assert p.tracking is False

    def test_low_power_no_depth(self):
        p = get_pipeline_profile("low_power")
        assert p.depth_estimation is False

    def test_low_power_stages(self):
        p = get_pipeline_profile("low_power")
        assert p.stages == ["detection"]

    def test_balanced_detection_model(self):
        p = get_pipeline_profile("balanced")
        assert p.detection.model_name == "yolov8s"

    def test_balanced_input_size(self):
        p = get_pipeline_profile("balanced")
        assert p.detection.input_size == 640

    def test_balanced_has_tracking(self):
        p = get_pipeline_profile("balanced")
        assert p.tracking is True

    def test_balanced_has_depth(self):
        p = get_pipeline_profile("balanced")
        assert p.depth_estimation is True

    def test_balanced_stages(self):
        p = get_pipeline_profile("balanced")
        assert p.stages == ["detection", "tracking"]

    def test_high_perf_model(self):
        p = get_pipeline_profile("high_performance")
        assert p.detection.model_name == "yolov8l"

    def test_high_perf_input_size(self):
        p = get_pipeline_profile("high_performance")
        assert p.detection.input_size == 1280

    def test_high_perf_max_detections(self):
        p = get_pipeline_profile("high_performance")
        assert p.detection.max_detections == 200

    def test_high_perf_stages(self):
        p = get_pipeline_profile("high_performance")
        assert "segmentation" in p.stages
        assert "depth_estimation" in p.stages

    def test_high_perf_4_stages(self):
        p = get_pipeline_profile("high_performance")
        assert len(p.stages) == 4

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError):
            get_pipeline_profile("unknown")

    def test_empty_profile_raises(self):
        with pytest.raises(ValueError):
            get_pipeline_profile("")

    def test_profile_values_increasing(self):
        lp = get_pipeline_profile("low_power")
        bal = get_pipeline_profile("balanced")
        hp = get_pipeline_profile("high_performance")
        assert lp.detection.input_size < bal.detection.input_size < hp.detection.input_size

    def test_profile_detections_increasing(self):
        lp = get_pipeline_profile("low_power")
        bal = get_pipeline_profile("balanced")
        hp = get_pipeline_profile("high_performance")
        assert lp.detection.max_detections < bal.detection.max_detections < hp.detection.max_detections


# ═══════════════════════════════════════════════════════════════════════════
# 17. Package __init__ imports
# ═══════════════════════════════════════════════════════════════════════════

class TestPackageInit:
    def test_import_cpu_config(self):
        from hardware.jetson_nano import CPUConfig as CC
        assert CC is CPUConfig

    def test_import_gpu_config(self):
        from hardware.jetson_nano import GPUConfig as GC
        assert GC is GPUConfig

    def test_import_jetson_config(self):
        from hardware.jetson_nano import JetsonConfig as JC
        assert JC is JetsonConfig

    def test_import_object_detector(self):
        from hardware.jetson_nano import ObjectDetector as OD
        assert OD is ObjectDetector

    def test_import_camera_config(self):
        from hardware.jetson_nano import CameraConfig as CamC
        assert CamC is CameraConfig

    def test_import_perception_pipeline(self):
        from hardware.jetson_nano import PerceptionPipeline as PP
        assert PP is PerceptionPipeline

    def test_import_get_pipeline_profile(self):
        from hardware.jetson_nano import get_pipeline_profile as GPP
        assert GPP is get_pipeline_profile

    def test_import_agx_config(self):
        from hardware.jetson_nano import JetsonAGXOrinConfig as AGXC
        assert AGXC is JetsonAGXOrinConfig

    def test_import_orin_nano_config(self):
        from hardware.jetson_nano import JetsonOrinNanoConfig as ONC
        assert ONC is JetsonOrinNanoConfig
