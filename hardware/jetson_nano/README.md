# NVIDIA Jetson Hardware Configurations for NEXUS Marine Robotics

Production-ready hardware configuration modules for the full NVIDIA Jetson family, optimized for autonomous marine robotics, underwater computer vision, and fleet coordination.

## Supported Jetson Boards

| Board | GPU | CUDA Cores | Tensor Cores | RAM | Memory Type | JetPack | Target Workload |
|-------|-----|-----------|-------------|-----|-------------|---------|----------------|
| **Jetson Nano 4GB** | Maxwell | 128 | 0 | 4 GB | LPDDR4 | 4.6 | Basic perception, sensor fusion |
| **Jetson TX2** | Pascal | 256 | 0 | 8 GB | LPDDR4 | 4.6 | Marine robotics real-time CV |
| **Jetson Xavier NX** | Volta | 384 | 48 | 8 GB | LPDDR4x | 5.0 | Deep learning inference |
| **Jetson Orin Nano 8GB** | Ampere | 1024 | 32 | 8 GB | LPDDR5 | 5.1 | Next-gen edge AI |
| **Jetson Orin NX 16GB** | Ampere | 1024 | 32 | 16 GB | LPDDR5 | 5.1 | NEXUS fleet commander |
| **Jetson AGX Orin 64GB** | Ampere | 2048 | 64 | 64 GB | LPDDR5 | 5.1 | Flagship multi-model AI |

## Architecture

Each board module provides:

- **CPU Config** — ARM core count, type, clock speed
- **GPU Config** — Architecture name, CUDA cores, TFLOPS, Tensor cores, TOPS
- **Memory Config** — RAM capacity, bandwidth, memory type
- **Storage Config** — Boot media, USB/SATA/PCIe connectivity
- **Power Config** — Max wattage, thermal throttle, fan curve, power modes
- **AI Config** — TensorRT precision, batch size, resolution limits, DLAS/Triton support
- **Pin Mapping** — GPIO/I2C/PWM assignments for NEXUS marine peripherals

## Board Registry

```python
from hardware.jetson_nano import list_supported_boards, get_board_info

# List all supported boards
boards = list_supported_boards()
# ['jetson-nano', 'jetson-tx2', 'jetson-xavier-nx', ...]

# Get board details
info = get_board_info("jetson-orin-nx")
print(info["description"])
# "Jetson Orin NX 16GB — Ampere 1024 CUDA, 16GB LPDDR5, fleet commander"

# Create a config with overrides
from hardware.jetson_nano import create_jetson_tx2_config
config = create_jetson_tx2_config(
    power={"max_watts": 10},
    ai={"target_fps": 15}
)
```

## AI Perception Pipeline

The shared `ai_pipeline` module provides configurable multi-stage perception:

```python
from hardware.jetson_nano import get_pipeline_profile

# Low-power: yolov5n, 416px, detection only
pipeline = get_pipeline_profile("low_power")

# High-performance: yolov8l, 1280px, detection+segmentation+depth+tracking
pipeline = get_pipeline_profile("high_performance")
```

## Power Modes

Jetson TX2 supports NVIDIA nvpmodel power modes:
- **MAXN** (15W) — Maximum performance, all cores active
- **MAXQ** (7W) — Maximum efficiency
- **MAXQ_CORE_ALL_OFF** (5W) — GPU only, CPU cores disabled
- **MAXP_CORE_ALL_OFF** (6W) — Low power with GPU active

## Quick Start

```bash
# Run all Jetson configuration tests
python -m pytest hardware/jetson_nano/ -v
```

## NEXUS Integration

Each config class is designed for integration with the NEXUS agent system:
- **Agent Role Assignment** — Fleet commander (Orin NX), perception node (TX2), sensor hub (Nano)
- **Trust Level Configuration** — Safety-critical settings per board capability
- **Data Pipeline Config** — Bandwidth-aware model deployment and inference scheduling
