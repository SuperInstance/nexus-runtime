# NEXUS i.MX RT Hardware Configuration Library

**NXP i.MX RT Crossover MCU Configuration for the NEXUS Distributed Intelligence Platform — Marine Robotics Edge Compute**

The NEXUS i.MX RT library provides hardware configuration and peripheral management for NXP i.MX RT crossover MCUs deployed in high-performance marine robotics applications. These Cortex-M7 based MCUs deliver DSP-class processing for real-time sensor fusion, motor control, and edge AI inference on autonomous underwater vehicles (AUVs), remotely operated vehicles (ROVs), and autonomous surface vessels (ASVs).

## Supported Boards

| Board | CPU | Clock | SRAM | Flash | Best For |
|-------|-----|-------|------|-------|----------|
| **i.MX RT1060** | Cortex-M7 | 600 MHz | 1 MB | External | Real-time motor control, sensor processing |
| **i.MX RT1064** | Cortex-M7 | 600 MHz | 1 MB | 4 MB internal | Volume production, simplified BOM |
| **i.MX RT1170** | Dual M7+M4 | M7@1GHz, M4@400MHz | 3.5 MB | External | High-performance autopilot, multi-sensor AI |
| **i.MX RT1050** | Cortex-M7 | 600 MHz | 512 KB | External | Cost-optimized mid-range controller |

## Features

- **Cortex-M7 @ 600 MHz – 1 GHz** — DSP instructions, FPU with double-precision, 1328-2475 DMIPS
- **512 KB – 3.5 MB SRAM** — Tightly coupled memory (ITCM/DTCM) for deterministic real-time performance
- **FlexIO** — Programmable shifters and timers for custom serial protocols (underwater acoustic, DVL interfaces)
- **CAN-FD** — Controller Area Network with Flexible Data-rate for NMEA 2000 marine networks
- **Ethernet** — Up to 3x 10/100 Mbps with TSN support for real-time surface comms
- **8-10x UART** — Multi-sensor serial connectivity (GPS, echosounder, CTD, telemetry)
- **USB 2.0 OTG** — High-speed data download and firmware update

## Quick Start

```python
from hardware.imx_rt.config_imxrt1060 import IMXRT1060Config
from hardware.imx_rt.flexio_config import FlexIOManager, FlexIOProtocol, ShifterConfig, TimerConfig, ShifterMode, TimerMode

# Configure RT1060 for marine controller
mcu = IMXRT1060Config()
mcu.configure_marine_controller()

# Allocate peripherals
mcu.allocate_peripheral("gps_uart", "LPUART", 0)
mcu.allocate_peripheral("ctd_i2c", "LPI2C", 0)
mcu.allocate_peripheral("thruster_pwm", "PWM", 0)

# Set up FlexIO for custom sonar protocol
flexio = FlexIOManager()
protocol = FlexIOProtocol(
    name="sonar_rx",
    description="Custom sonar receive protocol",
    shifters=[
        ShifterConfig(index=0, mode=ShifterMode.RECEIVE, pin_select=10, timer_select=0),
    ],
    timers=[
        TimerConfig(index=0, mode=TimerMode.BAUD, pin_select=11, timer_compare=49),
    ],
    baud_rate=1_000_000,
)
errors = flexio.register_protocol(protocol, FlexIOInstance.FLEXIO1)
```

## Peripheral Reference

| Peripheral | RT1050 | RT1060 | RT1064 | RT1170 |
|-----------|--------|--------|--------|--------|
| ADC (12-bit) | 4 ch | 6 ch | 6 ch | 10 ch |
| UART | 8 | 8 | 8 | 10 |
| SPI | 4 | 4 | 4 | 4 |
| I2C | 4 | 4 | 4 | 5 |
| CAN-FD | 1 | 2 | 2 | 2 |
| Ethernet | 1 | 2 | 2 | 3 (TSN) |
| USB 2.0 | 2 | 2 | 2 | 2 |
| FlexIO | 2 | 2 | 2 | 2 |
| PWM modules | 2 (8 SM) | 2 (8 SM) | 2 (8 SM) | 2 (8 SM) |

## Architecture

```
hardware/imx_rt/
  __init__.py              Board registry and exports
  config_imxrt1060.py      RT1060 (Cortex-M7 @ 600MHz, 1MB SRAM)
  config_imxrt1064.py      RT1064 (extends RT1060 with 4MB internal flash)
  config_imxrt1170.py      RT1170 (Dual M7@1GHz + M4@400MHz, 3.5MB SRAM)
  config_imxrt1050.py      RT1050 (Cortex-M7 @ 600MHz, 512KB SRAM, cost-optimized)
  flexio_config.py         FlexIO programmable protocol engine
  tests/
    test_imxrt1060.py
    test_imxrt1170.py
    test_imxrt1050.py
```

## License

Proprietary — NEXUS Marine Robotics Platform. All rights reserved.
