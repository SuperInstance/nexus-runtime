"""
BeagleBone Cape (Expansion Board) Detection and Management for NEXUS Marine Robotics.

Provides cape detection, slot management, and configuration for BeagleBone
expansion boards used in NEXUS marine deployments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class CapeSlot(str, Enum):
    """BeagleBone cape expansion slots."""
    SLOT_0 = "cape0"
    SLOT_1 = "cape1"
    SLOT_2 = "cape2"
    SLOT_3 = "cape3"


@dataclass(frozen=True)
class CapeInfo:
    """Information about a detected BeagleBone cape."""
    name: str
    version: str
    manufacturer: str
    part_number: str = ""
    slot: CapeSlot = CapeSlot.SLOT_0
    description: str = ""
    priority: int = 0


@dataclass
class CapeManager:
    """
    Manages BeagleBone cape detection, loading, and configuration.

    Provides a registry of known capes and methods for detecting and
    managing capes attached to the BeagleBone.
    """
    _capes: Dict[CapeSlot, CapeInfo] = field(default_factory=dict)
    _slots: List[CapeSlot] = field(
        default_factory=lambda: list(CapeSlot)
    )
    _known_capes: Dict[str, CapeInfo] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize the known capes registry with common marine capes."""
        self._known_capes = {
            "nexus-motor-controller": CapeInfo(
                name="NEXUS Motor Controller",
                version="2.1",
                manufacturer="NEXUS Labs",
                part_number="NXS-MC-001",
                description="4-channel motor/esc controller with encoder feedback",
                priority=10,
            ),
            "nexus-sensor-array": CapeInfo(
                name="NEXUS Sensor Array",
                version="1.5",
                manufacturer="NEXUS Labs",
                part_number="NXS-SA-001",
                description="Multi-sensor array: IMU, pressure, temperature, dissolved O2",
                priority=8,
            ),
            "nexus-power-monitor": CapeInfo(
                name="NEXUS Power Monitor",
                version="1.0",
                manufacturer="NEXUS Labs",
                part_number="NXS-PM-001",
                description="Battery voltage/current monitoring with low-voltage cutoff",
                priority=9,
            ),
            "beaglebone-4ch-relay": CapeInfo(
                name="4-Channel Relay Cape",
                version="1.2",
                manufacturer="Seeed Studio",
                part_number="BBB-RELAY-4CH",
                description="4-channel relay for high-power device switching",
                priority=5,
            ),
            "beaglebone-canbus": CapeInfo(
                name="CAN Bus Cape",
                version="1.1",
                manufacturer="BeagleBoard.org",
                part_number="BBB-CAN-001",
                description="Dual CAN bus interface for marine sensor networking",
                priority=7,
            ),
        }

    def detect_capes(self) -> List[CapeInfo]:
        """
        Detect capes attached to the BeagleBone.

        Returns:
            List of detected CapeInfo objects.
        """
        return list(self._capes.values())

    def load_cape(self, name: str, slot: CapeSlot = CapeSlot.SLOT_0) -> bool:
        """
        Load a cape into the specified slot.

        Args:
            name: The known cape name to load.
            slot: The expansion slot to assign.

        Returns:
            True if the cape was loaded successfully, False otherwise.
        """
        if name not in self._known_capes:
            return False
        cape = self._known_capes[name]
        cape_with_slot = CapeInfo(
            name=cape.name,
            version=cape.version,
            manufacturer=cape.manufacturer,
            part_number=cape.part_number,
            slot=slot,
            description=cape.description,
            priority=cape.priority,
        )
        self._capes[slot] = cape_with_slot
        return True

    def unload_cape(self, slot: CapeSlot) -> bool:
        """
        Unload a cape from the specified slot.

        Args:
            slot: The expansion slot to unload.

        Returns:
            True if a cape was removed, False if the slot was empty.
        """
        if slot in self._capes:
            del self._capes[slot]
            return True
        return False

    def get_cape_at_slot(self, slot: CapeSlot) -> Optional[CapeInfo]:
        """Return the cape at the given slot, or None if empty."""
        return self._capes.get(slot)

    def list_known_capes(self) -> List[str]:
        """Return a list of all known cape names."""
        return sorted(self._known_capes.keys())

    def get_known_cape(self, name: str) -> Optional[CapeInfo]:
        """Return info about a known cape by name, or None."""
        return self._known_capes.get(name)

    def loaded_count(self) -> int:
        """Return the number of currently loaded capes."""
        return len(self._capes)

    def available_slots(self) -> List[CapeSlot]:
        """Return list of slots that are currently empty."""
        return [s for s in self._slots if s not in self._capes]


def create_cape_manager() -> CapeManager:
    """Factory function for CapeManager."""
    return CapeManager()
