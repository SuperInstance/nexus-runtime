"""Teensy 3.6 configuration for NEXUS marine robotics.

MCU: NXP MK66FX1M0VMD18 — Cortex-M4F @ 180 MHz
Memory: 256 KB SRAM, 1 MB flash (built-in)
Legacy support for existing NEXUS deployments.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


CPU_FREQ_HZ = 180_000_000
SRAM_BYTES = 262_144  # 256 KB
FLASH_BYTES = 1_048_576  # 1 MB
NUM_DMA_CHANNELS = 16


@dataclass
class Teensy36Config:
    """Full configuration for Teensy 3.6 as a NEXUS legacy edge node."""

    cpu_freq_hz: int = CPU_FREQ_HZ
    bus_freq_hz: int = 60_000_000
    sram_bytes: int = SRAM_BYTES
    flash_bytes: int = FLASH_BYTES
    num_dma_channels: int = NUM_DMA_CHANNELS

    # NEXUS role
    nexus_role: str = "edge_sensor"
    trust_level: int = 2
    agent_id_prefix: str = "NX-T36-"

    # Serial
    baud_rate: int = 115200
    serial_interface: str = "/dev/ttyACM0"

    # Pin counts
    digital_pins: int = 64
    analog_inputs: int = 25
    pwm_pins: int = 22

    # USB
    usb_high_speed: bool = True

    # DAC
    dac_channels: int = 2

    def validate(self) -> bool:
        if self.cpu_freq_hz <= 0 or self.sram_bytes <= 0 or self.flash_bytes <= 0:
            return False
        return 0 <= self.trust_level <= 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_freq_hz": self.cpu_freq_hz,
            "bus_freq_hz": self.bus_freq_hz,
            "sram_bytes": self.sram_bytes,
            "flash_bytes": self.flash_bytes,
            "nexus_role": self.nexus_role,
            "trust_level": self.trust_level,
            "digital_pins": self.digital_pins,
            "analog_inputs": self.analog_inputs,
            "usb_high_speed": self.usb_high_speed,
            "dac_channels": self.dac_channels,
        }

    def get_deploy_manifest(self) -> dict[str, Any]:
        return {
            "board": "teensy_36",
            "mcu": "MK66FX1M0VMD18",
            "architecture": "ARM Cortex-M4F",
            "cpu_freq_mhz": self.cpu_freq_hz // 1_000_000,
            "memory": {
                "sram_kb": self.sram_bytes // 1024,
                "flash_kb": self.flash_bytes // 1024,
            },
            "interfaces": {
                "serial": {"baud": self.baud_rate},
                "spi": True, "i2c": True, "can": True,
                "usb_high_speed": self.usb_high_speed,
                "dac_channels": self.dac_channels,
            },
            "nexus": {
                "role": self.nexus_role,
                "trust_level": self.trust_level,
                "agent_prefix": self.agent_id_prefix,
            },
        }

    def get_nexus_config(self) -> dict[str, Any]:
        return {
            "role": self.nexus_role,
            "trust_level": self.trust_level,
            "agent_id_prefix": self.agent_id_prefix,
            "wire_protocol": "COBS_CRC16",
            "vm_enabled": True,
            "max_bytecode_size": self.sram_bytes // 4,
            "capabilities": {
                "usb_high_speed": self.usb_high_speed,
                "dac": self.dac_channels > 0,
                "fpu": True,
            },
        }

    def __repr__(self) -> str:
        return (f"Teensy36Config(cpu={self.cpu_freq_hz//1_000_000}MHz, "
                f"sram={self.sram_bytes//1024}KB, flash={self.flash_bytes//1024}KB, "
                f"role={self.nexus_role})")


def create_teensy36(**overrides: Any) -> Teensy36Config:
    """Factory: create Teensy36Config with optional overrides."""
    base = Teensy36Config()
    if overrides:
        base = replace(base, **overrides)
    return base
