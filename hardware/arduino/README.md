# NEXUS Arduino Hardware Configuration for Marine Robotics

[![NEXUS Platform](https://img.shields.io/badge/NEXUS-Marine%20Robotics-blue)](#) [![Arduino](https://img.shields.io/badge/Arduino-Supported-00979D)](#) [![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

## NEXUS: Distributed Intelligence for Marine Vessels

**NEXUS** is a distributed intelligence platform purpose-built for **marine robotics** — including **autonomous underwater vehicles (AUV)**, **remotely operated vehicles (ROV)**, **unmanned surface vehicles (USV)**, and **autonomous vessel** systems. The platform connects onboard sensors, actuators, and edge compute nodes through a reliable serial protocol, enabling real-time sensor integration and coordinated control across multiple microcontrollers.

Arduino boards serve as the primary I/O bridge in the NEXUS architecture, interfacing directly with marine sensors — GPS receivers, inertial measurement units (IMU), sonar modules, temperature probes, and pressure transducers — and relaying telemetry to higher-level compute modules over a structured wire protocol.

---

## Supported Arduino Boards

| Board | CPU | Architecture | Clock | Flash | RAM | UARTs | Digital Pins | Analog Pins | Voltage | Config Module |
|---|---|---|---|---|---|---|---|---|---|---|
| **Uno R3** | ATmega328P | AVR 8-bit | 16 MHz | 32 KB | 2 KB | 1 | 14 | 6 | 5.0 V | `config_uno.py` |
| **Mega 2560** | ATmega2560 | AVR 8-bit | 16 MHz | 256 KB | 8 KB | 4 | 54 | 16 | 5.0 V | `config_mega.py` |
| **Nano** | ATmega328P | AVR 8-bit | 16 MHz | 32 KB | 2 KB | 1 | 22 | 8 | 5.0 V | `config_nano.py` |
| **Due** | AT91SAM3X8E | ARM Cortex-M3 | 84 MHz | 512 KB | 96 KB | 4 | 54 | 12 | 3.3 V | `config_due.py` |
| **MKR WiFi 1010** | SAMD21G18A | ARM Cortex-M0+ | 48 MHz | 256 KB | 32 KB | 1 | 22 | 7 | 3.3 V | `config_mkr_wifi.py` |
| **Nano 33 IoT** | SAMD21G18A | ARM Cortex-M0+ | 48 MHz | 256 KB | 32 KB | 1 | 20 | 8 | 3.3 V | `config_nano33_iot.py` |

### Board Selection Guide

- **Uno R3** — Best for simple single-sensor experiments and prototyping
- **Mega 2560** — Ideal for multi-sensor arrays requiring 4 UARTs and 54 GPIO
- **Nano** — Compact breadboard-friendly form factor with extended analog inputs (A0–A7)
- **Due** — High-performance ARM Cortex-M3 for edge computing, sensor fusion, PID control
- **MKR WiFi 1010** — WiFi/BLE connectivity for wireless sensor gateway applications
- **Nano 33 IoT** — Ultra-compact WiFi/BLE mesh node for distributed sensor fleets

---

## Serial Communication with NEXUS

The NEXUS wire protocol runs over UART at **115200 baud, 8-N-1** by default. Every frame begins with a configurable preamble byte (`0xAA 0x55`) and includes a heartbeat at 500 ms intervals to maintain link health.

```python
from hardware.arduino import get_board_config

cfg = get_board_config("uno")
print(f"Board: {cfg.board_config.board_name}")
print(f"Baud : {cfg.serial_config.baud_rate}")
print(f"GPS  : TX=D{cfg.pin_mapping.GPS_TX} RX=D{cfg.pin_mapping.GPS_RX}")
```

### Frame Structure

```
[PREAMBLE 2B] [LENGTH 1B] [MSG_ID 1B] [PAYLOAD 0-252B] [CRC16 2B]
```

- **Preamble**: `0xAA 0x55` (configurable)
- **Max frame size**: 256 bytes (Uno/Nano), 512 bytes (Due/MKR)
- **Heartbeat interval**: 250–500 ms (configurable per board)

---

## Pin Mappings for Marine Sensors

### Arduino Uno — Single-Sensor Layout

| Function | Pin | Interface | Notes |
|---|---|---|---|
| GPS TX | D0 (RX) | UART | Shared with USB serial |
| GPS RX | D1 (TX) | UART | Use SoftwareSerial if USB needed |
| IMU SDA | A4 | I2C | Default I2C data line |
| IMU SCL | A5 | I2C | Default I2C clock line |
| Sonar Trig | D7 | GPIO | HC-SR04 trigger |
| Sonar Echo | D8 | GPIO | HC-SR04 echo |
| Temperature | A0 | Analog | NTC thermistor / DS18B20 |
| Pressure | A1 | Analog | MPX series / MS5837 over I2C |
| Servo | D9 | PWM | Rudder / ballast control |
| LED Status | D13 | GPIO | Onboard LED |
| Thruster PWM | D10 | PWM | Brushed ESC or thruster driver |

### Arduino Mega 2560 — Multi-Sensor Layout

| Function | Pin | Interface | Notes |
|---|---|---|---|
| GPS TX | D19 (RX1) | Serial1 | Dedicated UART |
| GPS RX | D18 (TX1) | Serial1 | Dedicated UART |
| IMU #1 SDA | D20 | I2C (Wire) | Primary IMU |
| IMU #1 SCL | D21 | I2C (Wire) | Primary IMU |
| IMU #2 SDA | D20 | I2C (Wire) | Secondary (shared bus) |
| Sonar #1 Trig | D22 | GPIO | Forward sonar |
| Sonar #1 Echo | D23 | GPIO | Forward sonar |
| Sonar #2 Trig | D24 | GPIO | Downward sonar |
| Sonar #2 Echo | D25 | GPIO | Downward sonar |
| Sonar #3 Trig | D26 | GPIO | Port sonar |
| Sonar #3 Echo | D27 | GPIO | Port sonar |
| Temperature #1 | A0 | Analog / 1-Wire | Forward hull |
| Temperature #2 | A1 | Analog / 1-Wire | Motor bay |
| Pressure | A2 | Analog / I2C | Depth sensor |
| Servo #1 | D44 | PWM | Rudder |
| Servo #2 | D45 | PWM | Ballast valve |
| Thruster Port | D46 | PWM | Port thruster ESC |
| Thruster Starboard | D47 | PWM | Starboard thruster ESC |
| Thruster Vertical | D48 | PWM | Vertical thruster ESC |
| LED Status | D13 | GPIO | Onboard LED |
| ESP / Companion | D16 (TX2) / D17 (RX2) | Serial2 | NEXUS backbone link |
| Aux Serial | D14 (TX3) / D15 (RX3) | Serial3 | Payload comms |

### Arduino Due — High-Performance Edge Layout

| Function | Pin | Interface | Notes |
|---|---|---|---|
| GPS | D19/D18 | Serial1 | Dedicated UART |
| IMU #1 | D20/D21 | I2C (Wire) | Primary IMU |
| IMU #2 | SDA1/SCL1 | I2C (Wire1) | Extended I2C header |
| Sonar Array | D22–D29 | GPIO | Up to 4 sonar modules |
| Temperature ×4 | A0–A3 | Analog / 1-Wire | Multi-zone monitoring |
| Pressure | A4 | Analog / I2C | Depth sensor |
| Water Quality | A5 | Analog | Turbidity / dissolved O2 |
| DAC0 / DAC1 | DAC0/DAC1 | Analog Out | Direct analog control |
| Thrusters ×4 | D34–D37 | PWM | 4-axis thruster control |
| Companion | D16/D17 | Serial2 | NEXUS backbone |
| Aux Serial | D14/D15 | Serial3 | Payload comms |
| Relays | D30–D31 | GPIO | Power management |

### Arduino Nano — Compact Sensor Node Layout

| Function | Pin | Interface | Notes |
|---|---|---|---|
| GPS | D0/D1 | UART | Shared with USB (SoftwareSerial) |
| IMU | A4/A5 | I2C | Default I2C pins |
| Sonar | D7/D8 | GPIO | HC-SR04 |
| Temperature | A0 | Analog | NTC / DS18B20 |
| Pressure | A1 | Analog / I2C | Depth sensor |
| Extra ADC | A2–A3 | Analog | Extended analog inputs |
| Servo | D9 | PWM | Rudder control |
| Thruster | D10 | PWM | ESC / thruster driver |
| LED | D13 | GPIO | Onboard LED |

### Arduino MKR WiFi 1010 — Wireless Sensor Gateway

| Function | Pin | Interface | Notes |
|---|---|---|---|
| GPS | D13/D14 | Serial1 | Dedicated UART |
| IMU | D11/D12 | I2C | Standard I2C |
| Sonar | D6/D7 | GPIO | HC-SR04 |
| Temperature | A0 | Analog / 1-Wire | NTC / DS18B20 |
| Pressure | A1 | I2C | MS5837 |
| Servos | D4/D5 | PWM | Dual servo control |
| SPI | D5/D8-D10 | SPI | External peripherals |
| WiFi | NINA-W102 | WiFi/BLE | u-blox module |

### Arduino Nano 33 IoT — Wireless Mesh Sensor Node

| Function | Pin | Interface | Notes |
|---|---|---|---|
| GPS | D0/D1 | Serial1 | Hardware UART |
| IMU | A4/A5 | I2C | External + built-in LSM6DS3 |
| Sonar | D7/D8 | GPIO | HC-SR04 |
| Temperature | A0 | Analog / 1-Wire | NTC / DS18B20 |
| Pressure | A1 | I2C | MS5837 |
| Servo | D9 | PWM | Single servo |
| Thruster | D3 | PWM | ESC control |
| LED | D13 + RGB | GPIO | Status + RGB indicator |
| WiFi/BLE | NINA-W102 | WiFi/BLE | Mesh networking |

---

## Getting Started

### 1. Install the NEXUS Python runtime

```bash
pip install nexus-runtime
```

### 2. Import and configure your board

```python
from hardware.arduino import get_board_config, list_supported_boards

# List all supported boards
boards = list_supported_boards()
print(boards)  # ['due', 'leonardo', 'mkr_wifi', 'mega', 'nano', 'nano33_iot', 'uno']

# Get Uno configuration
uno = get_board_config("uno")
print(uno.serial_config)       # SerialConfig(baud_rate=115200, ...)
print(uno.pin_mapping.GPS_TX)  # 0

# Get Due configuration (high-performance edge node)
due = get_board_config("due")
print(due.board_config.clock_hz)       # 84_000_000
print(due.nexus_edge.edge_processing_enabled)  # True

# Get MKR WiFi 1010 configuration (wireless gateway)
mkr = get_board_config("mkr_wifi")
print(mkr.wifi_config.wifi_enabled)        # True
print(mkr.nexus_bridge.bridge_protocol)    # MQTT

# Get Nano 33 IoT configuration (mesh sensor node)
nano_iot = get_board_config("nano33_iot")
print(nano_iot.wifi_config.low_power_mode)  # True
print(nano_iot.nexus_mesh.mesh_enabled)    # True
```

### 3. Configure sensor drivers

```python
from hardware.arduino.sensor_drivers import (
    GPSSensorConfig, IMUSensorConfig, SonarConfig
)

gps = GPSSensorConfig(baud_rate=9600, update_rate_hz=5)
imu = IMUSensorConfig(accel_range_g=16, gyro_range_dps=2000, protocol="I2C")
sonar = SonarConfig(max_range_cm=400, trigger_pulse_us=10)
```

### 4. Flash the NEXUS Arduino firmware

Upload the `nexus_arduino_bridge.ino` sketch to your board. The firmware
implements the NEXUS wire protocol, reads sensors per the pin mapping, and
relays data over the configured UART.

```bash
arduino-cli compile --fqbn arduino:avr:uno sketches/nexus_arduino_bridge
arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:avr:uno sketches/nexus_arduino_bridge
```

### 5. Verify connectivity

```python
import serial
ser = serial.Serial("/dev/ttyACM0", 115200, timeout=1)
# Expect heartbeat frames every 500 ms
frame = ser.read(256)
assert frame[:2] == b"\xaa\x55", "NEXUS preamble not found"
print("NEXUS link established.")
```

---

## Architecture Overview

```
┌───────────────────────────────────────────────┐
│                 NEXUS Central Hub              │
│          (Raspberry Pi / Jetson / PC)          │
└──────────┬───────────────────┬───────────────┘
           │ 115200 UART        │ WiFi/MQTT
   ┌───────┴──────┐   ┌────────┴─────────┐
   │ Arduino Mega │   │ MKR WiFi 1010   │
   │  (Sensors)   │   │ (WiFi Gateway)  │
   │ GPS, IMU,    │   │ WiFi/BLE Mesh   │
   │ Sonars, Temp │   │ OTA Updates     │
   └──────────────┘   └─────────────────┘
           │ 115200 UART        │ WiFi/BLE Mesh
   ┌───────┴──────┐   ┌────────┴─────────┐
   │ Arduino Due  │   │ Nano 33 IoT     │
   │  (Edge CPU)  │   │ (Sensor Node)   │
   │ Sensor Fusion│   │ GPS, Temp,      │
   │ PID Control  │   │ Sonar, IMU      │
   └──────────────┘   └─────────────────┘
           │ 115200 UART
   ┌───────┴──────┐
   │ Arduino Nano │
   │  (Compact I/O│
   │  + Extra ADC)│
   └──────────────┘
```

---

## Keywords

arduino, marine robotics, NEXUS, AUV, ROV, autonomous vessel, sensor integration, serial protocol, underwater robotics, USV, microcontroller, edge computing, telemetry, ocean engineering, ESP8266, ESP32, WiFi mesh, BLE.

---

## License

MIT — NEXUS Marine Intelligence Project
