"""NEXUS Wire Protocol - Frame parser.

Wire format: [0x00] [COBS(header+payload+CRC)] [0x00]
Max decoded frame: 1036 bytes
Max COBS frame: 1051 bytes
"""

from __future__ import annotations

from .cobs import cobs_decode
from .crc16 import crc16_ccitt

FRAME_MAX_DECODED = 1036
FRAME_HEADER_SIZE = 10
FRAME_CRC_SIZE = 2


class FrameParser:
    """Frame reception state machine (stub)."""

    def __init__(self) -> None:
        self._buffer: bytearray = bytearray()
        self._in_frame = False

    def feed(self, data: bytes) -> list[bytes]:
        """Feed bytes into the parser.

        Args:
            data: Received bytes.

        Returns:
            List of complete decoded frames.
        """
        frames: list[bytes] = []
        for byte in data:
            if byte == 0x00:
                if self._in_frame and len(self._buffer) > 0:
                    # Frame complete - decode
                    try:
                        decoded = cobs_decode(bytes(self._buffer))
                        if len(decoded) >= FRAME_HEADER_SIZE + FRAME_CRC_SIZE:
                            frames.append(bytes(decoded))
                    except Exception:
                        pass
                    self._buffer.clear()
                self._in_frame = not self._in_frame
            elif self._in_frame:
                self._buffer.append(byte)
        return frames

    def encode_frame(self, payload: bytes, msg_type: int = 0,
                     flags: int = 0, seq: int = 0) -> bytes:
        """Encode a complete wire frame.

        Args:
            payload: Message payload bytes.
            msg_type: Message type byte.
            flags: Message flags byte.
            seq: Sequence number.

        Returns:
            Complete wire frame with COBS encoding and delimiters.
        """
        import struct
        # Build header (10 bytes, big-endian)
        header = struct.pack(
            ">BBHIH",
            msg_type,
            flags,
            seq,
            0,  # timestamp_ms placeholder
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
