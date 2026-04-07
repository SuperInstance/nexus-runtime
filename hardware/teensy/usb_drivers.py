"""USB driver configurations for Teensy boards in NEXUS.

Supports CDC serial, Mass Storage, and composite device configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class USBEndpoint:
    """A USB endpoint configuration."""
    number: int
    direction: str  # "IN" or "OUT"
    transfer_type: str  # "BULK", "INTERRUPT", "ISOCHRONOUS"
    max_packet_size: int = 64


@dataclass(frozen=True)
class USBInterface:
    """A USB interface with its endpoints."""
    interface_number: int
    class_name: str  # "CDC", "MSC", "HID"
    endpoints: tuple[USBEndpoint, ...] = ()


@dataclass
class USBSerialConfig:
    """USB CDC serial configuration."""
    vid: int = 0x16C0  # Teensy VID
    pid: int = 0x0483  # CDC serial PID
    baud_rates: tuple[int, ...] = (9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600)
    endpoint_size: int = 64
    rx_buffer_size: int = 256
    tx_buffer_size: int = 256

    def to_dict(self) -> dict[str, Any]:
        return {"vid": f"0x{self.vid:04X}", "pid": f"0x{self.pid:04X}",
                "baud_rates": list(self.baud_rates), "endpoint_size": self.endpoint_size}


@dataclass
class USBCompositeConfig:
    """USB composite device (Serial + MSD) configuration."""

    vid: int = 0x16C0
    pid: int = 0x04D0
    serial_config: USBSerialConfig = field(default_factory=USBSerialConfig)

    # MSD (Mass Storage Device) settings
    msd_lun_count: int = 1
    msd_block_size: int = 512
    msd_sector_count: int = 0  # 0 = auto-detect from media

    # Endpoint pool (shared across all interfaces)
    total_endpoints: int = 6  # 3 IN + 3 OUT

    def validate(self) -> bool:
        if self.total_endpoints < 4:
            return False
        if self.msd_lun_count < 1:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "vid": f"0x{self.vid:04X}",
            "pid": f"0x{self.pid:04X}",
            "serial": self.serial_config.to_dict(),
            "msd": {"lun_count": self.msd_lun_count,
                    "block_size": self.msd_block_size,
                    "sector_count": self.msd_sector_count},
            "total_endpoints": self.total_endpoints,
        }

    def summary(self) -> str:
        return (f"USBComposite(VID=0x{self.vid:04X}, PID=0x{self.pid:04X}, "
                f"endpoints={self.total_endpoints}, "
                f"baud={self.serial_config.baud_rates[-1]})")

    def clone(self, **overrides: Any) -> USBCompositeConfig:
        """Create a copy with overrides."""
        return replace(self, **overrides)
