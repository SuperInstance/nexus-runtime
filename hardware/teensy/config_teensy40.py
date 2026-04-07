"""Teensy 4.0 configuration for NEXUS marine robotics.

MCU: NXP i.MX RT1062 — Cortex-M7 @ 600 MHz
Memory: 1 MB SRAM (ITCM/DTCM/OCRAM), 4 MB flash (external QSPI)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class PinMapping:
    """Teensy 4.0 pin mapping."""
    digital_pins: int = 40
    analog_inputs: int = 14
    pwm_pins: int = 28
    spi_mosi: int = 11
    spi_miso: int = 12
    spi_sck: int = 13
    i2c_sda: int = 18
    i2c_scl: int = 19
    uart0_tx: int = 1
    uart0_rx: int = 0
    serial2_tx: int = 7
    serial2_rx: int = 8
    can0_tx: int = 3
    can0_rx: int = 4
    flexio_pins: list[int] = field(default_factory=lambda: [6, 9, 10, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33])
    analog_range: tuple[int, int] = (0, 3.3)


CPU_FREQ_HZ = 600_000_000
SRAM_BYTES = 1_048_576  # 1 MB
FLASH_MBYTES = 4
NUM_DMA_CHANNELS = 32
NUM_FLEXIO_SHIFTERS = 8


@dataclass
class Teensy40Config:
    """Full configuration for Teensy 4.0 as a NEXUS edge node."""

    # Clock
    cpu_freq_hz: int = CPU_FREQ_HZ
    bus_freq_hz: int = 150_000_000

    # Memory
    sram_bytes: int = SRAM_BYTES
    flash_mbytes: int = FLASH_MBYTES

    # DMA
    num_dma_channels: int = NUM_DMA_CHANNELS
    num_flexio_shifters: int = NUM_FLEXIO_SHIFTERS

    # NEXUS role
    nexus_role: str = "edge_sensor"
    trust_level: int = 2
    agent_id_prefix: str = "NX-T40-"

    # Serial
    baud_rate: int = 115200
    serial_interface: str = "/dev/ttyACM0"

    # Pin mapping
    pins: PinMapping = field(default_factory=PinMapping)

    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.cpu_freq_hz <= 0:
            return False
        if self.sram_bytes <= 0:
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
            "flash_mbytes": self.flash_mbytes,
            "num_dma_channels": self.num_dma_channels,
            "num_flexio_shifters": self.num_flexio_shifters,
            "nexus_role": self.nexus_role,
            "trust_level": self.trust_level,
            "baud_rate": self.baud_rate,
            "pins": {
                "digital": self.pins.digital_pins,
                "analog": self.pins.analog_inputs,
                "pwm": self.pins.pwm_pins,
            },
        }

    def get_deploy_manifest(self) -> dict[str, Any]:
        """Return NEXUS deployment manifest."""
        return {
            "board": "teensy_40",
            "mcu": "IMXRT1062DVL6A",
            "architecture": "ARM Cortex-M7",
            "cpu_freq_mhz": self.cpu_freq_hz // 1_000_000,
            "memory": {
                "sram_kb": self.sram_bytes // 1024,
                "flash_mb": self.flash_mbytes,
            },
            "interfaces": {
                "serial": {"baud": self.baud_rate, "interface": self.serial_interface},
                "spi": True,
                "i2c": True,
                "can_fd": True,
                "flexio": {"shifters": self.num_flexio_shifters},
                "dma": {"channels": self.num_dma_channels},
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
            "data_pipeline": {
                "sensor_buffer_size": 256,
                "actuator_queue_depth": 16,
            },
        }

    def __repr__(self) -> str:
        return (f"Teensy40Config(cpu={self.cpu_freq_hz//1_000_000}MHz, "
                f"sram={self.sram_bytes//1024}KB, flash={self.flash_mbytes}MB, "
                f"role={self.nexus_role})")


def create_teensy40(**overrides: Any) -> Teensy40Config:
    """Factory: create Teensy40Config with optional overrides."""
    base = Teensy40Config()
    if overrides:
        base = replace(base, **overrides)
    return base
