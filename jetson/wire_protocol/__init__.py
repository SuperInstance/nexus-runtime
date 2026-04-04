"""NEXUS wire_protocol package.

Provides COBS encoding, CRC-16, frame parsing, and high-level
node client for communicating with ESP32 nodes over the NEXUS
wire protocol via serial (RS-422).
"""

from .cobs import cobs_encode, cobs_decode
from .crc16 import crc16_ccitt
from .frame import FrameParser, parse_frame_header, FRAME_MAX_PAYLOAD
from .node_client import (
    NodeClient, MessageType, MessageFlag, Criticality, Direction,
    MSG_TYPE_INFO, MSG_TYPE_NAMES,
)

__all__ = [
    "cobs_encode", "cobs_decode",
    "crc16_ccitt",
    "FrameParser", "parse_frame_header", "FRAME_MAX_PAYLOAD",
    "NodeClient", "MessageType", "MessageFlag", "Criticality", "Direction",
    "MSG_TYPE_INFO", "MSG_TYPE_NAMES",
]
