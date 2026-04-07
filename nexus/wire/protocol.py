"""NEXUS Wire Protocol — COBS + CRC-16 framing for reliable marine communications.

Frame format (all fields little-endian):
    [PREAMBLE:2B][LENGTH:2B][PAYLOAD:NB][CRC16:2B]

Encoding pipeline (TX):
    payload → COBS encode → prepend length + preamble → append CRC-16

Decoding pipeline (RX):
    raw → locate preamble → extract length → COBS decode → verify CRC-16

COBS (Consistent Overhead Byte Stuffing) ensures no zero bytes in the
encoded payload, allowing the preamble ``0xAA55`` to be unambiguous.
"""

from __future__ import annotations

import enum
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PREAMBLE: bytes = b"\xaa\x55"
PREAMBLE_INT: int = 0x55AA  # little-endian interpretation
FRAME_HEADER_SIZE: int = 4  # preamble(2) + length(2)
CRC_SIZE: int = 2
MAX_FRAME_SIZE: int = 4096
MAX_PAYLOAD_SIZE: int = MAX_FRAME_SIZE - FRAME_HEADER_SIZE - CRC_SIZE


# ---------------------------------------------------------------------------
# Message Types
# ---------------------------------------------------------------------------

class MessageType(enum.IntEnum):
    """28 wire protocol message types for NEXUS marine communications."""

    # -- system --
    HEARTBEAT = 0x01
    DISCOVERY = 0x02
    DISCOVERY_ACK = 0x03

    # -- sensor data --
    SENSOR_DATA = 0x10
    SENSOR_CALIBRATION = 0x11
    SENSOR_STREAM = 0x12

    # -- command & control --
    COMMAND = 0x20
    COMMAND_ACK = 0x21
    COMMAND_NAK = 0x22
    ABORT = 0x23

    # -- telemetry --
    TELEMETRY = 0x30
    TELEMETRY_BURST = 0x31
    DIAGNOSTICS = 0x32

    # -- trust --
    TRUST_UPDATE = 0x40
    TRUST_QUERY = 0x41
    TRUST_REPORT = 0x42

    # -- roles & swarm --
    ROLE_ASSIGN = 0x50
    ROLE_RELEASE = 0x51
    SWARM_STATE = 0x52
    SWARM_COMMAND = 0x53

    # -- AAB --
    A2A_REQUEST = 0x60
    A2A_RESPONSE = 0x61
    A2A_NEGOTIATE = 0x62
    A2A_DELEGATE = 0x63
    A2A_REPORT = 0x64

    # -- data --
    BULK_TRANSFER = 0x70
    BULK_ACK = 0x71

    # -- error --
    ERROR = 0xF0
    RESET = 0xFF


# ---------------------------------------------------------------------------
# CRC-16/CCITT
# ---------------------------------------------------------------------------

class CRC16:
    """CRC-16/CCITT-FALSE (poly=0x1021, init=0xFFFF) implementation."""

    POLY = 0x1021
    INIT = 0xFFFF

    @staticmethod
    def _make_table() -> List[int]:
        table: List[int] = []
        for i in range(256):
            crc = i << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ CRC16.POLY) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
            table.append(crc)
        return table

    _TABLE: Optional[List[int]] = None

    @classmethod
    def _get_table(cls) -> List[int]:
        if cls._TABLE is None:
            cls._TABLE = cls._make_table()
        return cls._TABLE

    @classmethod
    def compute(cls, data: bytes) -> int:
        """Compute CRC-16 over *data* and return 16-bit checksum."""
        crc = cls.INIT
        table = cls._get_table()
        for byte in data:
            crc = ((crc << 8) ^ table[((crc >> 8) ^ byte) & 0xFF]) & 0xFFFF
        return crc

    @classmethod
    def verify(cls, data: bytes, expected: int) -> bool:
        """Verify that *data* matches *expected* CRC-16."""
        return cls.compute(data) == (expected & 0xFFFF)


# ---------------------------------------------------------------------------
# COBS Encoding / Decoding
# ---------------------------------------------------------------------------

class COBSCodec:
    """Zero-free byte stuffing codec for NEXUS wire protocol.

    Eliminates all zero bytes from the payload so that the ``0xAA55``
    preamble can be unambiguously detected. Uses escape-based stuffing:

    * ``0x00`` → ``0xFF 0x00``
    * ``0xFF`` → ``0xFF 0x01``
    * All other bytes pass through unchanged

    Overhead: at most 1 extra byte per zero or 0xFF byte in input.
    Worst-case expansion: 2× for all-zeros input.
    """

    @staticmethod
    def encode(data: bytes) -> bytes:
        """Encode *data* so the output contains no zero bytes."""
        if not data:
            return b""
        result: List[int] = []
        for byte in data:
            if byte == 0x00:
                result.extend((0xFF, 0x01))  # escaped zero
            elif byte == 0xFF:
                result.extend((0xFF, 0x02))  # escaped 0xFF
            else:
                result.append(byte)
        return bytes(result)

    @staticmethod
    def decode(data: bytes) -> bytes:
        """Decode zero-free encoded *data* back to the original bytes."""
        if not data:
            return b""
        result: List[int] = []
        i = 0
        while i < len(data):
            byte = data[i]
            if byte == 0xFF and i + 1 < len(data):
                next_byte = data[i + 1]
                if next_byte == 0x01:
                    result.append(0x00)
                elif next_byte == 0x02:
                    result.append(0xFF)
                else:
                    # Not a valid escape pair, pass through
                    result.append(byte)
                    result.append(next_byte)
                i += 2
            else:
                result.append(byte)
                i += 1
        return bytes(result)


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A typed wire protocol message with payload."""

    msg_type: MessageType
    payload: bytes = b""
    source: int = 0
    destination: int = 0
    sequence: int = 0
    timestamp: float = 0.0

    def encode_header(self) -> bytes:
        """Encode the message header (type + addressing) as 12 bytes."""
        return struct.pack(
            "<BBIIH",
            self.msg_type.value,
            0,  # flags/reserved
            self.source & 0xFFFFFFFF,
            self.destination & 0xFFFFFFFF,
            self.sequence & 0xFFFF,
        )

    @classmethod
    def decode_header(cls, data: bytes) -> "Message":
        """Decode a message header from 12 bytes."""
        if len(data) < 12:
            raise ValueError(f"Header too short: {len(data)} bytes (need 12)")
        msg_type_val, flags, source, dest, seq = struct.unpack("<BBIIH", data[:12])
        return cls(
            msg_type=MessageType(msg_type_val),
            source=source,
            destination=dest,
            sequence=seq,
        )

    def __repr__(self) -> str:
        name = MessageType(self.msg_type).name if isinstance(self.msg_type, int) else self.msg_type.name
        return f"Message({name}, src={self.source}, dst={self.destination}, seq={self.sequence}, len={len(self.payload)})"


# ---------------------------------------------------------------------------
# Frame encode / decode helpers
# ---------------------------------------------------------------------------

def encode_frame(msg: Message) -> bytes:
    """Encode a :class:`Message` into a complete wire frame."""
    header = msg.encode_header()
    raw_payload = header + msg.payload
    encoded_payload = COBSCodec.encode(raw_payload)
    crc = CRC16.compute(raw_payload)
    length = len(encoded_payload)
    frame = PREAMBLE + struct.pack("<H", length) + encoded_payload + struct.pack("<H", crc)
    return frame


def decode_frame(data: bytes) -> Optional[Message]:
    """Decode a single wire frame from *data*. Returns the Message or None."""
    if len(data) < FRAME_HEADER_SIZE:
        return None

    if data[:2] != PREAMBLE:
        return None

    length = struct.unpack("<H", data[2:4])[0]
    total_len = FRAME_HEADER_SIZE + length + CRC_SIZE

    if len(data) < total_len:
        return None

    encoded_payload = data[FRAME_HEADER_SIZE: FRAME_HEADER_SIZE + length]
    crc_received = struct.unpack("<H", data[FRAME_HEADER_SIZE + length: FRAME_HEADER_SIZE + length + CRC_SIZE])[0]

    raw_payload = COBSCodec.decode(encoded_payload)

    if not CRC16.verify(raw_payload, crc_received):
        return None

    msg = Message.decode_header(raw_payload[:12])
    msg.payload = raw_payload[12:]
    return msg
