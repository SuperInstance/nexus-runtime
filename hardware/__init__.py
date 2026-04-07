"""
NEXUS Marine Robotics — Hardware Configuration Package
=======================================================
Pre-configured deployment profiles for embedded and edge platforms.

Supported Platform Families (50+ board configs across 11 families)
------------------------------------------------------------------
- **Arduino**:    Uno R3, Mega 2560, Nano, Due, MKR WiFi 1010, Nano 33 IoT
- **ESP32**:      Classic, S3, C3, C6, H2
- **ESP8266**:    NodeMCU, Wemos D1 Mini
- **Jetson**:     Nano, TX2, Xavier NX, Orin Nano, Orin NX, AGX Orin
- **Raspberry Pi**: Zero W, 3B+, 4B, 400, 5, CM4, Pico 2
- **STM32**:      F4, F7, G0, L4, WL, H7, MP1
- **Nordic nRF**: 52810, 52832, 52840, 5340
- **Teensy**:     3.6, 4.0, 4.1
- **i.MX RT**:    1050, 1060, 1064, 1170
- **RP2040**:     Pico, Pico W
- **BeagleBone**: Black, AI-64
"""

__version__ = "0.2.1"

# Canonical board registry — one source of truth
_BOARD_CATALOG: dict[str, list[str]] = {
    "arduino": [
        "uno", "mega", "nano", "due",
        "mkr-wifi-1010", "nano-33-iot",
    ],
    "esp32": [
        "esp32", "esp32-s3", "esp32-c3", "esp32-c6", "esp32-h2",
    ],
    "esp8266": ["esp8266", "d1-mini"],
    "jetson_nano": [
        "jetson-nano", "jetson-tx2", "jetson-xavier-nx",
        "jetson-orin-nano", "jetson-orin-nx", "jetson-agx-orin",
    ],
    "raspberry_pi": [
        "pi-zero-w", "pi-3b-plus", "pi-4b", "pi-400", "pi-5",
        "compute-module-4", "pico-2",
    ],
    "stm32": [
        "stm32f4", "stm32f7", "stm32g0", "stm32l4",
        "stm32wl", "stm32h7", "stm32mp1",
    ],
    "nrf52": ["nrf52810", "nrf52832", "nrf52840", "nrf5340"],
    "teensy": ["teensy-3.6", "teensy-4.0", "teensy-4.1"],
    "imx_rt": ["imxrt1050", "imxrt1060", "imxrt1064", "imxrt1170"],
    "rp2040": ["rp2040", "pico-w"],
    "beaglebone": ["beaglebone-black", "beaglebone-ai64"],
}

_PLATFORM_DISPLAY: dict[str, str] = {
    "arduino": "Arduino (ATmega / SAMD)",
    "esp32": "ESP32 (Xtensa / RISC-V)",
    "esp8266": "ESP8266 (Xtensa L106)",
    "jetson_nano": "NVIDIA Jetson",
    "raspberry_pi": "Raspberry Pi (BCM / RP2040 / RP2350)",
    "stm32": "STM32 (Cortex-M)",
    "nrf52": "Nordic nRF52 (BLE)",
    "teensy": "Teensy (NXP i.MX RT / Kinetis)",
    "imx_rt": "NXP i.MX RT (Cortex-M7)",
    "rp2040": "Raspberry Pi Pico (RP2040)",
    "beaglebone": "BeagleBone (AM335x / AM572x)",
}


def list_platforms() -> list[str]:
    """Return all registered platform family names."""
    return sorted(_BOARD_CATALOG.keys())


def list_boards(platform: str) -> list[str]:
    """Return all registered boards for a given platform family."""
    return _BOARD_CATALOG.get(platform, [])


def list_all_boards() -> dict[str, list[str]]:
    """Return the full platform → boards registry."""
    return dict(sorted(_BOARD_CATALOG.items()))


def get_platform_display(platform: str) -> str:
    """Return a human-readable description for a platform."""
    return _PLATFORM_DISPLAY.get(platform, platform)


def total_board_count() -> int:
    """Return the total number of supported board configurations."""
    return sum(len(v) for v in _BOARD_CATALOG.values())
