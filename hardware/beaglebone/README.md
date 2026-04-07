# BeagleBone Hardware Configurations for NEXUS Marine Robotics

Production-ready hardware configuration modules for BeagleBone platforms, featuring PRU (Programmable Real-time Unit) support for deterministic motor control and sensor interfacing in marine robotics applications.

## Supported BeagleBone Boards

| Board | CPU | RAM | Special Hardware | Target Workload |
|-------|-----|-----|-----------------|-----------------|
| **BeagleBone Black** | AM3358 Cortex-A8 @ 1.0 GHz | 512 MB DDR3 | 2x PRU-ICSS | Real-time motor control, sensor polling |
| **BeagleBone AI-64** | 2x Cortex-A15 @ 1.5 GHz | 2 GB DDR3 | 2x C66x DSP, 4x EVE, 4x PRU | DSP-accelerated signal processing, vision |

## Architecture

### BeagleBone Black (`config_black.py`)

- **CPU Config** — Single-core ARM Cortex-A8, 1.0 GHz
- **Memory Config** — 512 MB DDR3 @ 6.4 GB/s
- **Storage Config** — microSD / 4 GB eMMC
- **Power Config** — 5V DC, TPS65217C PMIC
- **GPIO Config** — Pin-header mappings for GPS, IMU, sonar, thruster PWM, LEDs
- **PRU Count** — 2 programmable real-time units

### BeagleBone AI-64 (`config_ai64.py`)

- **CPU Config** — Dual-core ARM Cortex-A15, 1.5 GHz
- **DSP Config** — Dual C66x floating-point DSPs @ 1.0 GHz, 4x EVE vision accelerators
- **Memory Config** — 2 GB DDR3 @ 12.8 GB/s
- **Storage Config** — microSD / 8 GB eMMC, USB 3.0
- **Power Config** — 12V DC / 5V USB-C, TPS659162 PMIC
- **GPIO Config** — Extended pin mappings for dual cameras, multi-channel PWM
- **PRU Count** — 4 programmable real-time units

## PRU Configuration (`pru_config.py`)

The PRU (Programmable Real-time Unit) subsystem provides deterministic real-time control for marine actuators:

### Features

- **Dual PRU Core Management** — Independent core configuration (motor control, sensor polling)
- **Shared Memory** — 12 KB on-chip + 256 KB DDR shared between PRU and ARM
- **Motor Channel Config** — Per-channel PWM frequency, pulse range, dead zone, encoder feedback
- **Safety Features** — Watchdog timer, emergency stop pin, safety clamp
- **200 Hz Control Loop** — Default real-time control frequency

### PRU Modes

| Mode | Description |
|------|-------------|
| `motor_control` | PWM generation for thrusters, rudders, ballast pumps |
| `sensor_polling` | Deterministic ADC/DAC/I2C sensor sampling |
| `pwm_generation` | Custom waveform generation |
| `custom_firmware` | User-supplied PRU firmware |
| `idle` | Core disabled |

### Usage

```python
from hardware.beaglebone import create_pru_controller_config

# Create default PRU config (2 cores, 4 motor channels)
config = create_pru_controller_config()

# Customize for high-speed control
config = create_pru_controller_config(
    control_loop_hz=1000,
    watchdog_timeout_ms=50,
    emergency_stop_pin="P8.15"
)
```

## Cape Manager (`cape_manager.py`)

Manages BeagleBone expansion boards (capes) for NEXUS marine deployments:

### Built-in Cape Profiles

| Cape | Description | Priority |
|------|-------------|----------|
| `nexus-motor-controller` | 4-channel ESC controller with encoder feedback | 10 |
| `nexus-power-monitor` | Battery voltage/current monitoring with LVC | 9 |
| `nexus-sensor-array` | IMU, pressure, temperature, dissolved O2 array | 8 |
| `beaglebone-canbus` | Dual CAN bus for marine sensor networking | 7 |
| `beaglebone-4ch-relay` | 4-channel relay for high-power switching | 5 |

### Usage

```python
from hardware.beaglebone import create_cape_manager, CapeSlot

manager = create_cape_manager()

# Load capes into expansion slots
manager.load_cape("nexus-motor-controller", CapeSlot.SLOT_0)
manager.load_cape("nexus-sensor-array", CapeSlot.SLOT_1)

# Inspect loaded capes
for cape in manager.detect_capes():
    print(f"Slot {cape.slot}: {cape.name} v{cape.version}")

# Check available slots
print(manager.available_slots())
```

## Board Registry

```python
from hardware.beaglebone import list_supported_boards, get_board_info

boards = list_supported_boards()
# ['beaglebone-ai64', 'beaglebone-black']

info = get_board_info("beaglebone-black")
print(info["description"])
# "BeagleBone Black — AM3358 Cortex-A8, 512MB DDR3, PRU real-time"
```

## Quick Start

```bash
# Run all BeagleBone configuration tests
python -m pytest hardware/beaglebone/ -v
```

## NEXUS Integration

BeagleBone boards serve specialized roles in the NEXUS fleet:

- **Motor Controller Node** (Black) — Deterministic thruster PWM via PRU, watchdog supervision
- **Signal Processing Node** (AI-64) — DSP-accelerated sonar processing, EVE-based vision preprocessing
- **Cape Extensibility** — Hot-pluggable sensor and actuator capes via Cape Manager
