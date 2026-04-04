"""NEXUS Wire Protocol - Frame parser and builder.

Wire format: [0x00] [COBS(header+payload+CRC)] [0x00]
Max decoded frame: 1036 bytes
Max COBS frame: 1051 bytes
Max wire frame: 1053 bytes
"""

from __future__ import annotations

import struct
import time
from typing import Optional

from .cobs import cobs_encode, cobs_decode
from .crc16 import crc16_ccitt

FRAME_MAX_DECODED = 1036
FRAME_MAX_COBS = 1051
FRAME_MAX_WIRE = 1053
FRAME_HEADER_SIZE = 10
FRAME_CRC_SIZE = 2
FRAME_MAX_PAYLOAD = 1024

FRAME_ERR_NONE = 0
FRAME_ERR_TOO_LARGE = 0x5001
FRAME_ERR_CRC_MISMATCH = 0x5003
FRAME_ERR_BUFFER_OVERFLOW = 0x5004


class FrameParser:
    """Frame reception state machine.

    States: IDLE, RECEIVING.
    Feed bytes one at a time or in chunks. Complete frames
    are returned via feed() as a list of decoded frames.
    """

    def __init__(self) -> None:
        self._buffer: bytearray = bytearray()
        self._in_frame = False
        self._error_count = 0

    @property
    def error_count(self) -> int:
        return self._error_count

    def reset(self) -> None:
        """Reset parser to initial state."""
        self._buffer.clear()
        self._in_frame = False

    def feed(self, data: bytes) -> list[bytes]:
        """Feed bytes into the parser.

        Args:
            data: Received bytes.

        Returns:
            List of complete decoded frames (header + payload, CRC stripped).
        """
        frames: list[bytes] = []
        for byte in data:
            if byte == 0x00:
                if self._in_frame and len(self._buffer) > 0:
                    # Frame complete - decode
                    decoded = cobs_decode(bytes(self._buffer))
                    dec_len = len(decoded)
                    if dec_len == 0:
                        self._error_count += 1
                    elif dec_len < FRAME_HEADER_SIZE + FRAME_CRC_SIZE:
                        self._error_count += 1
                    else:
                        # Validate payload length field
                        payload_len = (decoded[8] << 8) | decoded[9]
                        expected_len = FRAME_HEADER_SIZE + payload_len + FRAME_CRC_SIZE
                        if expected_len != dec_len:
                            self._error_count += 1
                        else:
                            # Verify CRC
                            received_crc = (decoded[dec_len - 2] << 8) | decoded[dec_len - 1]
                            computed_crc = crc16_ccitt(decoded[:dec_len - FRAME_CRC_SIZE])
                            if received_crc != computed_crc:
                                self._error_count += 1
                            else:
                                # Valid frame: return header + payload (strip CRC)
                                frames.append(bytes(decoded[:dec_len - FRAME_CRC_SIZE]))
                    self._buffer.clear()
                self._in_frame = not self._in_frame
            elif self._in_frame:
                if len(self._buffer) >= FRAME_MAX_COBS:
                    # Frame too large, discard
                    self._buffer.clear()
                    self._in_frame = False
                    self._error_count += 1
                else:
                    self._buffer.append(byte)
        return frames

    @staticmethod
    def encode_frame(payload: bytes, msg_type: int = 0,
                     flags: int = 0, seq: int = 0,
                     timestamp_ms: Optional[int] = None) -> bytes:
        """Encode a complete wire frame.

        Args:
            payload: Message payload bytes (max 1024 bytes).
            msg_type: Message type byte.
            flags: Message flags byte.
            seq: Sequence number.
            timestamp_ms: Timestamp in milliseconds (auto-generated if None).

        Returns:
            Complete wire frame with COBS encoding and 0x00 delimiters.
        """
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000) & 0xFFFFFFFF

        # Build header (10 bytes, big-endian)
        header = struct.pack(
            ">BBHIH",
            msg_type,
            flags,
            seq,
            timestamp_ms & 0xFFFFFFFF,
            len(payload),
        )
        # Compute CRC over header + payload
        data = header + payload
        crc = crc16_ccitt(data)
        data_with_crc = data + struct.pack(">H", crc)
        # COBS encode
        encoded = cobs_encode(data_with_crc)
        # Add delimiters
        return b"\x00" + encoded + b"\x00"


def parse_frame_header(frame_data: bytes) -> tuple:
    """Parse a decoded frame (header + payload, CRC stripped).

    Args:
        frame_data: Decoded frame bytes (at least 10 bytes).

    Returns:
        Tuple of (msg_type, flags, seq, timestamp_ms, payload_length, payload_bytes).

    Raises:
        ValueError: If frame data is too short.
    """
    if len(frame_data) < FRAME_HEADER_SIZE:
        raise ValueError(f"Frame too short: {len(frame_data)} bytes (minimum {FRAME_HEADER_SIZE})")
    msg_type = frame_data[0]
    flags = frame_data[1]
    seq = (frame_data[2] << 8) | frame_data[3]
    timestamp_ms = (frame_data[4] << 24) | (frame_data[5] << 16) | (frame_data[6] << 8) | frame_data[7]
    payload_length = (frame_data[8] << 8) | frame_data[9]
    payload = frame_data[FRAME_HEADER_SIZE:]
    if len(payload) != payload_length:
        raise ValueError(f"Payload length mismatch: header says {payload_length}, got {len(payload)}")
    return (msg_type, flags, seq, timestamp_ms, payload_length, payload)
