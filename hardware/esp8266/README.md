# NEXUS ESP8266 Hardware Configuration for Marine IoT Sensor Nodes

[![NEXUS Platform](https://img.shields.io/badge/NEXUS-Marine%20Robotics-blue)](#) [![ESP8266](https://img.shields.io/badge/ESP8266-WiFi-333333)](#) [![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

## NEXUS: Distributed Intelligence for Marine Vessels

**NEXUS** is a distributed intelligence platform purpose-built for **marine robotics**. The ESP8266 module provides hardware configuration and WiFi connection management for ESP8266-based boards deployed in NEXUS wireless sensor networks. The ESP8266's ultra-low cost, 802.11 b/g/n WiFi, and compact form factor make it ideal for battery-powered marine sensor telemetry nodes.

---

## Supported ESP8266 Boards

| Board | Variant | CPU | Clock | Flash | SRAM | GPIO | ADC | WiFi | Config Module |
|---|---|---|---|---|---|---|---|---|---|
| **ESP-12E/NodeMCU** | Generic | Xtensa L106 | 80/160 MHz | 4 MB | 80 KB | 17 | 1 (10-bit) | 802.11 b/g/n | `config_esp8266.py` |
| **Wemos D1 Mini** | Compact | Xtensa L106 | 80/160 MHz | 4 MB | 80 KB | 11 | 1 (10-bit) | 802.11 b/g/n | `config_d1_mini.py` |

---

## Features

- **Xtensa L106 @ 80/160 MHz** — sufficient for sensor polling and telemetry forwarding
- **4 MB Flash** — ample storage for firmware and configuration data
- **80 KB SRAM** — supports JSON-based telemetry payloads and MQTT buffers
- **802.11 b/g/n WiFi** — station mode (STA), access point mode (AP), and AP+STA
- **WiFi Connection Manager** — automatic reconnection, AP fallback, signal quality monitoring
- **NEXUS Wire Protocol** — serial framing with CRC-16 validation
- **MQTT Integration** — native MQTT pub/sub for NEXUS telemetry topics
- **Comprehensive Test Suite** — 80+ unit tests covering all modules

---

## Getting Started

### 1. Import and configure your board

```python
from hardware.esp8266 import create_esp8266_config, create_d1_mini_config

# NodeMCU configuration
nodemcu = create_esp8266_config(
    board_name="ESP8266-TEMP-NODE",
    comms={"wifi_ssid": "MARINE-WIFI", "wifi_password": "secret"},
    pin_map={"temp_pin": 17, "relay": 16},
)
print(f"Board: {nodemcu.board_name}")
print(f"GPIO:  {nodemcu.gpio_count}")

# D1 Mini configuration
d1 = create_d1_mini_config(
    board_name="D1-DEPTH-SENSOR",
    pin_map={"pressure_sda": 4, "pressure_scl": 5},
)
print(f"Board: {d1.board_name}")
```

### 2. WiFi connection management

```python
from hardware.esp8266 import WiFiManager, WiFiManagerConfig, WiFiCredentials

manager = WiFiManager(
    config=WiFiManagerConfig(
        auto_reconnect=True,
        max_reconnect_attempts=10,
        ap_fallback=True,
        hostname="nexus-depth-sensor",
    ),
    credentials=WiFiCredentials(ssid="MARINE-WIFI", password="secret"),
)

# Check connection status
print(f"Connected: {manager.is_connected}")
print(f"Should reconnect: {manager.should_reconnect}")
print(f"Signal quality: {manager.signal_quality_percent()}%")

# Get full connection info
info = manager.get_connection_info()
print(f"State: {info['state']}")
```

### 3. NEXUS telemetry over WiFi

```python
from hardware.esp8266 import ESP8266BoardConfig, create_esp8266_config

cfg = create_esp8266_config(
    comms={
        "wifi_ssid": "MARINE-WIFI",
        "wifi_password": "secret",
        "mqtt_broker": "192.168.1.100",
        "mqtt_port": 1883,
    }
)

# Telemetry topic
print(f"MQTT Broker: {cfg.comms.mqtt_broker}:{cfg.comms.mqtt_port}")
print(f"Frame preamble: {cfg.protocol.frame_preamble}")
print(f"Max frame size: {cfg.protocol.max_frame_size}")
```

---

## Pin Mappings

### ESP-12E / NodeMCU

| Function | GPIO | NodeMCU D-Pin | Interface | Notes |
|---|---|---|---|---|
| GPS TX | GPIO1 | TX | UART | Shared with USB serial |
| GPS RX | GPIO3 | RX | UART | Shared with USB serial |
| IMU SDA | GPIO4 | D2 | I2C | SDA |
| IMU SCL | GPIO5 | D1 | I2C | SCL |
| Sonar Trig | GPIO14 | D5 | GPIO | HC-SR04 trigger |
| Sonar Echo | GPIO12 | D6 | GPIO | HC-SR04 echo |
| Servo 1 | GPIO0 | D3 | PWM | Boot strapping pin |
| Servo 2 | GPIO15 | D8 | PWM | Boot strapping pin |
| LED | GPIO2 | D4 | GPIO | Active low |
| Temperature | ADC0 | A0 | Analog | 10-bit, 0-1V |
| Relay | GPIO16 | D0 | GPIO | Wakeup from deep sleep |
| NEXUS TX | GPIO15 | D8 | SoftwareSerial | To companion |
| NEXUS RX | GPIO13 | D7 | SoftwareSerial | From companion |

### Wemos D1 Mini

| Function | GPIO | D-Pin | Interface | Notes |
|---|---|---|---|---|
| GPS TX | GPIO1 | — | UART | TXD0 |
| GPS RX | GPIO3 | — | UART | RXD0 |
| IMU SDA | GPIO4 | D2 | I2C | SDA |
| IMU SCL | GPIO5 | D1 | I2C | SCL |
| Sonar Trig | GPIO14 | D5 | GPIO | HC-SR04 |
| Sonar Echo | GPIO12 | D6 | GPIO | HC-SR04 |
| Servo | GPIO0 | D3 | PWM | Boot strapping |
| LED | GPIO2 | D4 | GPIO | Active low |
| Temperature | ADC0 | A0 | Analog | 10-bit |
| Relay | GPIO16 | D0 | GPIO | Wakeup |
| SPI SCK | GPIO14 | D5 | SPI | |
| SPI MISO | GPIO12 | D6 | SPI | |
| SPI MOSI | GPIO13 | D7 | SPI | |
| SPI SS | GPIO15 | D8 | SPI | Boot strapping |

---

## WiFi Connection Manager

The `WiFiManager` class provides a state machine for managing ESP8266 WiFi connections in the NEXUS fleet:

### Connection States

| State | Description |
|---|---|
| `DISCONNECTED` | Initial state or lost connection |
| `SCANNING` | Scanning for available networks |
| `CONNECTING` | Attempting to connect |
| `CONNECTED` | Successfully connected to AP |
| `RECONNECTING` | Attempting reconnection |
| `CONNECTION_FAILED` | Connection attempt failed |
| `AP_FALLBACK` | Fell back to AP mode after max retries |
| `AP_MODE` | Operating as access point |

### Features

- Automatic reconnection with configurable retry count and interval
- AP mode fallback when station connection fails
- Signal quality monitoring (RSSI to percentage)
- Static IP / DHCP configuration
- mDNS hostname advertisement
- Connection info dictionary for telemetry reporting

---

## Architecture

```
┌───────────────────────────────────────────────┐
│              NEXUS MQTT Broker                │
│          (Raspberry Pi / Jetson)              │
└──────────┬──────────────┬────────────────────┘
           │ WiFi/MQTT    │ WiFi/MQTT
   ┌───────┴──────┐ ┌──────┴──────┐
   │ NodeMCU      │ │ D1 Mini     │
   │ (Temp Node)  │ │ (Depth Node)│
   │ A0: Temp     │ │ A0: Pressure│
   │ D7: Sonar    │ │ D1/D2: IMU  │
   └──────────────┘ └─────────────┘
           │ SoftwareSerial
   ┌───────┴──────┐
   │ Arduino Nano │
   │ (I/O Bridge) │
   └──────────────┘
```

---

## Keywords

ESP8266, NodeMCU, Wemos D1 Mini, marine robotics, NEXUS, IoT sensor node, WiFi, MQTT, telemetry, underwater robotics, wireless sensor network, microcontroller, ocean engineering.

---

## License

MIT — NEXUS Marine Intelligence Project
