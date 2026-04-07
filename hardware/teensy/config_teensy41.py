"""Teensy 4.1 configuration for NEXUS marine robotics.

MCU: NXP i.MX RT1062 — Cortex-M7 @ 600 MHz
Memory: 1 MB SRAM, 8 MB PSRAM, 4 MB flash (external QSPI)
Extras: Ethernet (RMII), onboard SD card slot
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from hardware.teensy.config_teensy40 import PinMapping

CPU_FREQ_HZ = 600_000_000
SRAM_BYTES = 1_048_576
PSRAM_BYTES = 8_388_608  # 8 MB
FLASH_MBYTES = 4
NUM_DMA_CHANNELS = 32
NUM_FLEXIO_SHIFTERS = 8

# Ethernet pins (RMII)
ETH_MDIO_PIN = 17
ETH_MDC_PIN = 52
ETH_RXD0_PIN = 25
ETH_RXD1_PIN = 26
ETH_CRS_DV_PIN = 27
ETH_TXEN_PIN = 28
ETH_TXD0_PIN = 35
ETH_TXD1_PIN = 36
ETH_REF_CLK_PIN = 33

# PSRAM
PSRAM_CS_PIN = 6

# SD Card (onboard)
SD_CS_PIN = 10
SD_SCK_PIN = 13
SD_MOSI_PIN = 11
SD_MISO_PIN = 12


@dataclass
class Teensy41Config:
    """Full configuration for Teensy 4.1 as a NEXUS network node."""

    # Clock
    cpu_freq_hz: int = CPU_FREQ_HZ
    bus_freq_hz: int = 150_000_000

    # Memory
    sram_bytes: int = SRAM_BYTES
    psram_bytes: int = PSRAM_BYTES
    flash_mbytes: int = FLASH_MBYTES

    # DMA
    num_dma_channels: int = NUM_DMA_CHANNELS
    num_flexio_shifters: int = NUM_FLEXIO_SHIFTERS

    # NEXUS role
    nexus_role: str = "network_relay"
    trust_level: int = 3
    agent_id_prefix: str = "NX-T41-"

    # Serial
    baud_rate: int = 115200
    serial_interface: str = "/dev/ttyACM0"

    # Ethernet
    ethernet_enabled: bool = True
    ethernet_mdio_pin: int = ETH_MDIO_PIN
    ethernet_mdc_pin: int = ETH_MDC_PIN

    # SD Card
    sd_card_enabled: bool = True
    sd_cs_pin: int = SD_CS_PIN

    # Pin mapping (inherits from 4.0 plus extras)
    digital_pins: int = 55  # 40 + 15 additional
    analog_inputs: int = 14
    pins: PinMapping = field(default_factory=PinMapping)

    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.cpu_freq_hz <= 0 or self.sram_bytes <= 0:
            return False
        if self.trust_level < 0 or self.trust_level > 5:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "cpu_freq_hz": self.cpu_freq_hz,
            "bus_freq_hz": self.bus_freq_hz,
            "sram_bytes": self.sram_bytes,
            "psram_bytes": self.psram_bytes,
            "flash_mbytes": self.flash_mbytes,
            "nexus_role": self.nexus_role,
            "trust_level": self.trust_level,
            "ethernet_enabled": self.ethernet_enabled,
            "sd_card_enabled": self.sd_card_enabled,
            "digital_pins": self.digital_pins,
            "analog_inputs": self.analog_inputs,
        }

    def get_deploy_manifest(self) -> dict[str, Any]:
        """Return NEXUS deployment manifest."""
        return {
            "board": "teensy_41",
            "mcu": "IMXRT1062DVJ6A",
            "architecture": "ARM Cortex-M7",
            "cpu_freq_mhz": self.cpu_freq_hz // 1_000_000,
            "memory": {
                "sram_kb": self.sram_bytes // 1024,
                "psram_mb": self.psram_bytes // (1024 * 1024),
                "flash_mb": self.flash_mbytes,
            },
            "interfaces": {
                "serial": {"baud": self.baud_rate},
                "spi": True, "i2c": True, "can_fd": True,
                "ethernet": {
                    "enabled": self.ethernet_enabled,
                    "phy": "RMII",
                    "mdio_pin": self.ethernet_mdio_pin,
                },
                "sd_card": {
                    "enabled": self.sd_card_enabled,
                    "cs_pin": self.sd_cs_pin,
                },
                "flexio": {"shifters": self.num_flexio_shifters},
            },
            "nexus": {
                "role": self.nexus_role,
                "trust_level": self.trust_level,
                "agent_prefix": self.agent_id_prefix,
            },
        }

    def get_nexus_config(self) -> dict[str, Any]:
        """Return NEXUS-specific runtime configuration."""
        return {
            "role": self.nexus_role,
            "trust_level": self.trust_level,
            "agent_id_prefix": self.agent_id_prefix,
            "wire_protocol": "COBS_CRC16",
            "vm_enabled": True,
            "max_bytecode_size": self.sram_bytes // 4,
            "network": {
                "ethernet_enabled": self.ethernet_enabled,
                "mdio_pin": self.ethernet_mdio_pin,
                "mdc_pin": self.ethernet_mdc_pin,
            },
            "storage": {
                "sd_card_enabled": self.sd_card_enabled,
                "cs_pin": self.sd_cs_pin,
                "psram_buffer_mb": self.psram_bytes // (1024 * 1024),
            },
        }

    def __repr__(self) -> str:
        return (f"Teensy41Config(cpu={self.cpu_freq_hz//1_000_000}MHz, "
                f"sram={self.sram_bytes//1024}KB, psram={self.psram_bytes//(1024*1024)}MB, "
                f"eth={self.ethernet_enabled}, role={self.nexus_role})")


def create_teensy41(**overrides: Any) -> Teensy41Config:
    """Factory: create Teensy41Config with optional overrides."""
    base = Teensy41Config()
    if overrides:
        base = replace(base, **overrides)
    return base
