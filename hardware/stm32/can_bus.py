"""
NEXUS CAN Bus Configuration Module

CAN bus configuration for NMEA 2000 marine networking. Provides CAN bus setup,
node addressing, PGN (Parameter Group Number) definitions for marine sensors,
and message scheduling for the NEXUS distributed intelligence platform.

Supports:
    - NMEA 2000 standard (250 kbps, 29-bit extended identifiers)
    - High-speed CAN (500 kbps, 1 Mbps)
    - Single-frame (<8 bytes) and transport protocol (>8 bytes) messages
    - Priority-based arbitration for real-time control
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CANBaudRate(IntEnum):
    """Standard CAN bus baud rates."""
    NMEA_2000 = 250_000
    HIGH_SPEED = 500_000
    VERY_HIGH_SPEED = 1_000_000


class CANMode(Enum):
    NORMAL = "normal"
    LOOPBACK = "loopback"
    SILENT = "silent"
    SILENT_LOOPBACK = "silent_loopback"


class CANFilterMode(Enum):
    LIST = "list"           # Identifier match list
    MASK = "mask"           # Identifier + mask
    RANGE = "range"         # Identifier range


class CANFrameType(Enum):
    DATA = "data"
    REMOTE = "remote"


class PGNPriority(IntEnum):
    """NMEA 2000 priority levels (0 = highest)."""
    EMERGENCY = 0
    AUTOPILOT = 1
    NAVIGATION = 2
    CONTROL = 3
    HIGH_PRIORITY = 4
    STANDARD = 5
    LOW_PRIORITY = 6
    DEFAULT = 6
    INFO_LOW = 7


class TransmitRate(Enum):
    """PGN transmit rate categories."""
    CONTINUOUS = "continuous"
    ON_CHANGE = "on_change"
    ON_REQUEST = "on_request"
    PERIODIC_100MS = "periodic_100ms"
    PERIODIC_1S = "periodic_1s"
    PERIODIC_10S = "periodic_10s"
    PERIODIC_60S = "periodic_60s"


# ---------------------------------------------------------------------------
# PGN Definitions
# ---------------------------------------------------------------------------

@dataclass
class PGNDefinition:
    """
    NMEA 2000 Parameter Group Number definition.

    Each PGN defines a message type on the NMEA 2000 network with a unique
    identifier, data length, and field layout.
    """
    pgn: int
    name: str
    description: str = ""
    data_length: int = 8      # bytes
    priority: PGNPriority = PGNPriority.DEFAULT
    transmit_rate: TransmitRate = TransmitRate.PERIODIC_1S
    fields: List[dict] = field(default_factory=list)
    source: str = "NEXUS"

    @property
    def is_single_frame(self) -> bool:
        return self.data_length <= 8

    @property
    def is_fast_packet(self) -> bool:
        return self.data_length > 8

    @property
    def required_frames(self) -> int:
        """Number of CAN frames needed (fast packet: 1 header + ceil((len-6)/7))."""
        if self.is_single_frame:
            return 1
        return 1 + ((self.data_length - 6) + 6) // 7

    def can_id(self, source_addr: int, priority: Optional[PGNPriority] = None) -> int:
        """
        Build the 29-bit CAN extended identifier.

        Format: P[3] | EDP[1] | DP[1] | PGN[17] | SA[8]
        """
        p = (priority if priority is not None else self.priority).value
        edp = 0  # Extended Data Page (always 0 for NMEA 2000)
        dp = (self.pgn >> 16) & 0x01   # Data Page
        pgn_field = self.pgn & 0x1FFFF
        sa = source_addr & 0xFF
        return (p << 26) | (edp << 25) | (dp << 24) | (pgn_field << 8) | sa


# ---------------------------------------------------------------------------
# Pre-defined PGNs for marine robotics
# ---------------------------------------------------------------------------

# -- Vessel Dynamics --
PGN_HEADING = PGNDefinition(
    pgn=127250, name="Vessel Heading",
    description="True and magnetic heading, heading rate, sensor data",
    data_length=8, priority=PGNPriority.NAVIGATION,
    transmit_rate=TransmitRate.PERIODIC_100MS,
    fields=[{"name": "sid", "offset": 0, "len": 1, "unit": "count"},
            {"name": "heading", "offset": 1, "len": 2, "unit": "deg", "scale": 0.0001},
            {"name": "deviation", "offset": 3, "len": 2, "unit": "rad", "scale": 0.0001},
            {"name": "variation", "offset": 5, "len": 2, "unit": "rad", "scale": 0.0001}],
)

PGN_RATE_OF_TURN = PGNDefinition(
    pgn=127251, name="Rate of Turn",
    description="Rate of turn of vessel in radians per second",
    data_length=4, priority=PGNPriority.NAVIGATION,
    transmit_rate=TransmitRate.PERIODIC_100MS,
)

PGN_SPEED_WATER = PGNDefinition(
    pgn=128259, name="Speed, Water Referenced",
    description="Speed through water from speed log sensor",
    data_length=3, priority=PGNPriority.NAVIGATION,
    transmit_rate=TransmitRate.PERIODIC_1S,
)

PGN_SPEED_GROUND = PGNDefinition(
    pgn=129026, name="COG & SOG, Rapid Update",
    description="Course over ground and speed over ground",
    data_length=8, priority=PGNPriority.HIGH_PRIORITY,
    transmit_rate=TransmitRate.PERIODIC_100MS,
)

# -- Environment --
PGN_WATER_DEPTH = PGNDefinition(
    pgn=128267, name="Water Depth",
    description="Depth below transducer with offset",
    data_length=8, priority=PGNPriority.STANDARD,
    transmit_rate=TransmitRate.PERIODIC_1S,
)

PGN_ENVIRONMENTAL = PGNDefinition(
    pgn=130310, name="Environmental Parameters",
    description="Water temperature, humidity, pressure",
    data_length=28, priority=PGNPriority.STANDARD,
    transmit_rate=TransmitRate.PERIODIC_10S,
)

PGN_WIND = PGNDefinition(
    pgn=130306, name="Wind Data",
    description="Wind speed, angle, and gust data",
    data_length=8, priority=PGNPriority.STANDARD,
    transmit_rate=TransmitRate.PERIODIC_1S,
)

# -- Power & Propulsion --
PGN_BATTERY = PGNDefinition(
    pgn=127506, name="DC Detailed Status",
    description="Battery voltage, current, temperature, state of charge",
    data_length=14, priority=PGNPriority.STANDARD,
    transmit_rate=TransmitRate.PERIODIC_1S,
    fields=[{"name": "sid", "offset": 0, "len": 1},
            {"name": "voltage", "offset": 1, "len": 2, "unit": "V", "scale": 0.01},
            {"name": "current", "offset": 3, "len": 2, "unit": "A", "scale": 0.1},
            {"name": "temperature", "offset": 5, "len": 2, "unit": "K", "scale": 0.01}],
)

PGN_ENGINE_RPM = PGNDefinition(
    pgn=127488, name="Engine Parameters, Rapid Update",
    description="Engine speed (RPM) and boost pressure",
    data_length=8, priority=PGNPriority.CONTROL,
    transmit_rate=TransmitRate.PERIODIC_100MS,
)

# -- NEXUS Custom PGNs (for thruster/motor control) --
PGN_THRUSTER_COMMAND = PGNDefinition(
    pgn=130820, name="NEXUS Thruster Command",
    description="Thrust force command for single thruster (NEXUS proprietary)",
    data_length=8, priority=PGNPriority.CONTROL,
    transmit_rate=TransmitRate.CONTINUOUS,
    source="NEXUS",
    fields=[{"name": "thruster_id", "offset": 0, "len": 1},
            {"name": "thrust_N", "offset": 1, "len": 2, "unit": "N", "scale": 0.1, "signed": True},
            {"name": "rpm", "offset": 3, "len": 2, "unit": "rpm"},
            {"name": "status", "offset": 5, "len": 1},
            {"name": "checksum", "offset": 6, "len": 2}],
)

PGN_THRUSTER_STATUS = PGNDefinition(
    pgn=130821, name="NEXUS Thruster Status",
    description="Thruster telemetry: actual RPM, current, temperature",
    data_length=16, priority=PGNPriority.STANDARD,
    transmit_rate=TransmitRate.PERIODIC_100MS,
    source="NEXUS",
)

PGN_SENSOR_DATA = PGNDefinition(
    pgn=130822, name="NEXUS Sensor Data",
    description="Generic sensor data payload for NEXUS sensor hubs",
    data_length=64, priority=PGNPriority.LOW_PRIORITY,
    transmit_rate=TransmitRate.PERIODIC_10S,
    source="NEXUS",
)

PGN_NAV_ATTITUDE = PGNDefinition(
    pgn=130823, name="NEXUS Navigation Attitude",
    description="Full 3D attitude (roll, pitch, yaw) from INS/DVL fusion",
    data_length=20, priority=PGNPriority.NAVIGATION,
    transmit_rate=TransmitRate.PERIODIC_100MS,
    source="NEXUS",
)

# PGN registry
PGN_REGISTRY: Dict[int, PGNDefinition] = {
    p.pgn: p for p in [
        PGN_HEADING, PGN_RATE_OF_TURN, PGN_SPEED_WATER, PGN_SPEED_GROUND,
        PGN_WATER_DEPTH, PGN_ENVIRONMENTAL, PGN_WIND,
        PGN_BATTERY, PGN_ENGINE_RPM,
        PGN_THRUSTER_COMMAND, PGN_THRUSTER_STATUS, PGN_SENSOR_DATA, PGN_NAV_ATTITUDE,
    ]
}


# ---------------------------------------------------------------------------
# Configuration classes
# ---------------------------------------------------------------------------

@dataclass
class CANFilterConfig:
    """CAN acceptance filter configuration."""
    filter_id: int = 0
    mode: CANFilterMode = CANFilterMode.LIST
    identifiers: List[int] = field(default_factory=list)
    mask: int = 0x7FF         # For mask mode
    fifo: int = 0             # FIFO assignment: 0 or 1
    active: bool = True


@dataclass
class CANConfig:
    """
    CAN bus hardware configuration.

    Configures baud rate, timing parameters, filter banks, and operating mode
    for a CAN peripheral on an STM32. Used by NEXUS motor controllers and
    sensor hubs for NMEA 2000 marine networking.
    """
    instance: str = "CAN1"
    baud_rate: int = CANBaudRate.NMEA_2000
    mode: CANMode = CANMode.NORMAL
    node_id: int = 42         # Source address (0..253)

    # Bit timing
    prescaler: int = 14
    time_seg1: int = 9        # Bit Segment 1 time quanta
    time_seg2: int = 2        # Bit Segment 2 time quanta
    sjw: int = 1              # Synchronisation Jump Width

    # Error handling
    auto_bus_off: bool = True
    auto_wake_up: bool = False
    auto_retransmit: bool = True

    # Filters
    filters: List[CANFilterConfig] = field(default_factory=list)

    # Message buffers
    tx_mailboxes: int = 3     # STM32F4 has 3 TX mailboxes
    rx_fifo_size: int = 3     # Messages per RX FIFO

    def __post_init__(self):
        if not self.filters:
            self.filters = [CANFilterConfig(filter_id=0, mode=CANFilterMode.MASK,
                                           mask=0x1FFFFFFF, fifo=0)]

    @property
    def nominal_baud(self) -> int:
        """Calculate actual baud from APB1 clock and timing params."""
        apb1 = 42_000_000  # STM32F4 APB1 default
        return apb1 // (self.prescaler * (1 + self.time_seg1 + self.time_seg2))

    @property
    def sample_point_pct(self) -> float:
        total = 1 + self.time_seg1 + self.time_seg2
        return ((1 + self.time_seg1) / total) * 100.0

    @property
    def bit_quanta_total(self) -> int:
        return 1 + self.time_seg1 + self.time_seg2

    def validate(self) -> List[str]:
        errors = []
        if self.node_id < 0 or self.node_id > 253:
            errors.append(f"Node ID {self.node_id} out of [0, 253]")
        if self.prescaler < 1 or self.prescaler > 1024:
            errors.append(f"Prescaler {self.prescaler} out of [1, 1024]")
        if self.time_seg1 < 1 or self.time_seg1 > 16:
            errors.append(f"Time Seg 1 {self.time_seg1} out of [1, 16]")
        if self.time_seg2 < 1 or self.time_seg2 > 8:
            errors.append(f"Time Seg 2 {self.time_seg2} out of [1, 8]")
        if self.sjw < 1 or self.sjw > 4:
            errors.append(f"SJW {self.sjw} out of [1, 4]")
        if self.sample_point_pct < 75.0 or self.sample_point_pct > 90.0:
            errors.append(f"Sample point {self.sample_point_pct:.1f}% outside recommended [75, 90]%")
        return errors

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    def clone(self) -> "CANConfig":
        return copy.deepcopy(self)


@dataclass
class CANNodeConfig:
    """
    Complete CAN node configuration for a NEXUS device.

    Combines CAN bus hardware config with device identity, supported PGNs,
    and message scheduling.
    """
    node_id: int = 42
    name: str = "unnamed_node"
    device_class: str = "sensor"
    manufacturer: str = "NEXUS"
    can_config: CANConfig = field(default_factory=CANConfig)
    supported_pgns: List[int] = field(default_factory=list)
    transmit_pgns: List[int] = field(default_factory=list)
    receive_pgns: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.can_config.node_id = self.node_id
        if not self.supported_pgns:
            self.supported_pgns = list(PGN_REGISTRY.keys())

    def get_pgn(self, pgn: int) -> Optional[PGNDefinition]:
        return PGN_REGISTRY.get(pgn)

    def build_can_id(self, pgn: int, priority: Optional[PGNPriority] = None) -> Optional[int]:
        pgn_def = PGN_REGISTRY.get(pgn)
        if pgn_def is None:
            return None
        return pgn_def.can_id(self.node_id, priority)

    def validate(self) -> List[str]:
        errors = self.can_config.validate()
        if self.node_id < 0 or self.node_id > 253:
            errors.append(f"Node ID {self.node_id} out of [0, 253]")
        for pgn in self.transmit_pgns:
            if pgn not in PGN_REGISTRY:
                errors.append(f"Transmit PGN {pgn} not in registry")
        for pgn in self.receive_pgns:
            if pgn not in PGN_REGISTRY:
                errors.append(f"Receive PGN {pgn} not in registry")
        return errors

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    def clone(self) -> "CANNodeConfig":
        return copy.deepcopy(self)
