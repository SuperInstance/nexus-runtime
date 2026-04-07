# NEXUS STM32 Marine Robotics Configuration Library

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

## NEXUS STM32 Marine Robotics Motor Controller CAN Bus Configuration

The **NEXUS distributed intelligence platform** provides production-grade hardware configuration libraries for **STM32 microcontrollers** deployed in **marine robotics** applications. This module covers the **STM32F4 (Cortex-M4)** and **STM32H7 (Cortex-M7)** families used across autonomous underwater vehicles (AUVs), remotely operated vehicles (ROVs), and unmanned surface vehicles (USVs).

### Supported Hardware

| MCU | Core | Clock | Flash | RAM | FPU | Primary Role |
|-----|------|-------|-------|-----|-----|-------------|
| **STM32F407VG** | Cortex-M4 | 168 MHz | 1 MB | 192 KB | Single-precision | Motor controller, sensor hub |
| **STM32H743VI** | Cortex-M7 | 480 MHz | 1 MB | 1 MB | Double-precision | High-rate sensor fusion, navigation computer |

### Key Features

- **CAN Bus Interface** — Full NMEA 2000 marine networking support with PGN definitions, CAN node configuration, and message scheduling for marine sensor networks.
- **Brushless DC Motor Controller** — ESC configuration, PID tuning parameters, thruster mapping, and failsafe logic for underwater thruster arrays.
- **DMA-Optimized Peripherals** — Pre-configured DMA streams for ADC sampling, UART telemetry, SPI sensor buses, and CAN Tx/Rx.
- **Clock Tree Configuration** — PLL setup with HSE, validated for marine-grade crystal tolerances.
- **Power Domain Management** — Voltage regulator, brown-out detection, and power mode profiles for battery-powered operation.

### Architecture

```
nexus-runtime/
└── hardware/
    └── stm32/
        ├── config_stm32f4.py    # STM32F407 register-level & peripheral config
        ├── config_stm32h7.py    # STM32H743 register-level & peripheral config
        ├── can_bus.py           # CAN / NMEA 2000 marine networking
        ├── motor_control.py     # BLDC motor controller & thruster config
        └── tests/
            ├── test_stm32f4.py
            ├── test_stm32h7.py
            ├── test_can_bus.py
            └── test_motor_control.py
```

### Quick Start

```python
from hardware.stm32.config_stm32f4 import STM32F407Config
from hardware.stm32.can_bus import CANConfig, CANNodeConfig
from hardware.stm32.motor_control import ESCConfig, PIDParams, ThrusterConfig

# STM32F407 motor controller node
mcu = STM32F407Config()
print(mcu)  # Cortex-M4 @ 168 MHz, 1MB flash, 192KB RAM

# CAN bus for NMEA 2000 sensor network
can = CANConfig(baud_rate=250000, node_id=42)
can_node = CANNodeConfig(node_id=42, name="starboard_motor", can_config=can)

# Thruster ESC with PID tuning
esc = ESCConfig(pwm_freq=16000, dead_time_ns=500)
pid = PIDParams(kp=0.85, ki=0.12, kd=0.04, output_limit=1.0)
thruster = ThrusterConfig(esc=esc, pid=pid, max_thrust_N=120.0, axis="Y")
```

### CAN Bus / NMEA 2000 Integration

This library provides first-class support for **NMEA 2000** marine networking over CAN bus:

- **PGN (Parameter Group Number) definitions** for vessel heading, speed, water depth, engine RPM, and battery status
- **Baud rates**: 250 kbps (NMEA 2000 standard) and 500 kbps / 1 Mbps (high-speed)
- **Arbitration and priority** management for real-time control messages
- **Transport protocol** for multi-frame messages (>8 bytes)

### Motor Controller Support

Configured for **brushless DC (BLDC) thrusters** used in marine propulsion:

- **ESC interface**: PWM frequency, dead-time insertion, commutation mode (FOC / trapezoidal)
- **PID closed-loop control**: Proportional-integral-derivative tuning with anti-windup and derivative filtering
- **Thruster arrays**: Configurable multi-thruster layouts for 6-DOF vehicle control
- **Failsafe**: Hardware watchdog timeout, loss-of-signal fallback, thermal derating

### Testing

```bash
cd /tmp/nexus-runtime && python -m pytest hardware/stm32/ -v --tb=short
```

All configuration objects are validated with 45+ unit tests covering serialisation, boundary checks, and cross-module integration.

### Deployment Targets

| Platform | MCU | Function |
|----------|-----|----------|
| NEXUS Thruster Node v2 | STM32F407VG | 4-channel BLDC motor controller |
| NEXUS Sensor Hub v3 | STM32F407VG | IMU, pressure, CTD, DVL interface |
| NEXUS CAN Bridge | STM32F407VG | NMEA 2000 / CAN gateway |
| NEXUS Nav Computer | STM32H743VI | Sensor fusion, INS, DVL processing |

### License

MIT License. See LICENSE file for details.

### Keywords

NEXUS, STM32, marine robotics, motor controller, CAN bus, NMEA 2000, brushless DC, thruster, AUV, ROV, USV, autonomous underwater vehicle, Cortex-M4, Cortex-M7, STM32F4, STM32H7, FOC, PID, sensor hub, distributed intelligence
