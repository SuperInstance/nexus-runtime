"""
NEXUS Frame Builder and Parser — stream-level framing for the wire protocol.

The FrameBuilder constructs raw frames from Messages.
The FrameParser reassembles complete frames from byte streams,
handling partial reads, buffer splits, and framing errors.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

from nexus.wire.protocol import (
    PREAMBLE, PREAMBLE_INT, FRAME_HEADER_SIZE, CRC_SIZE,
    MAX_FRAME_SIZE, MAX_PAYLOAD_SIZE,
    COBSCodec, CRC16, Message, MessageType,
    encode_frame,
)


class FrameParserState(enum.Enum):
    """States for the frame parser state machine."""

    SEEK_PREAMBLE = "SEEK_PREAMBLE"
    READ_HEADER = "READ_HEADER"
    READ_PAYLOAD = "READ_PAYLOAD"
    READ_CRC = "READ_CRC"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


@dataclass
class FrameBuilder:
    """Constructs NEXUS wire frames with automatic encoding.

    Usage::

        builder = FrameBuilder(source_id=0x01)
        frame = builder.build(Message(msg_type=MessageType.HEARTBEAT))
    """

    source_id: int = 0
    sequence_counter: int = 0

    def build(self, msg: Message) -> bytes:
        """Build a complete wire frame from a Message."""
        msg.source = self.source_id
        msg.sequence = self.sequence_counter
        self.sequence_counter = (self.sequence_counter + 1) & 0xFFFF
        return encode_frame(msg)

    def build_heartbeat(self) -> bytes:
        """Build a heartbeat frame."""
        msg = Message(msg_type=MessageType.HEARTBEAT, payload=b"\x01")
        return self.build(msg)

    def reset(self) -> None:
        """Reset sequence counter."""
        self.sequence_counter = 0


@dataclass
class ParsedFrame:
    """Result of successful frame parsing."""

    message: Message
    raw_frame: bytes
    consumed_bytes: int


@dataclass
class FrameParser:
    """Stream reassembly parser for NEXUS wire frames.

    Accumulates bytes from partial reads and extracts complete frames.

    Usage::

        parser = FrameParser()
        parser.feed(chunk1)
        parser.feed(chunk2)
        while parser.has_frame():
            frame = parser.next_frame()
            process(frame.message)
    """

    def __init__(self, max_frame_size: int = MAX_FRAME_SIZE) -> None:
        self._buffer = bytearray()
        self._state = FrameParserState.SEEK_PREAMBLE
        self._max_frame_size = max_frame_size
        self._frames: List[ParsedFrame] = []
        self._errors: List[str] = []

    @property
    def state(self) -> FrameParserState:
        return self._state

    @property
    def errors(self) -> List[str]:
        return list(self._errors)

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def feed(self, data: bytes) -> int:
        """Feed raw bytes into the parser. Returns bytes consumed from input."""
        self._buffer.extend(data)
        self._process_buffer()
        return len(data)

    def has_frame(self) -> bool:
        """Check if a complete frame is available."""
        return len(self._frames) > 0

    def next_frame(self) -> ParsedFrame:
        """Pop the next available frame. Raises IndexError if none."""
        return self._frames.pop(0)

    def next_messages(self) -> List[Message]:
        """Pop all available messages."""
        msgs: List[Message] = []
        while self._frames:
            msgs.append(self._frames.pop(0).message)
        return msgs

    def reset(self) -> None:
        """Clear the parser state."""
        self._buffer.clear()
        self._state = FrameParserState.SEEK_PREAMBLE
        self._frames.clear()
        self._errors.clear()

    def _process_buffer(self) -> None:
        """Internal: try to extract frames from the buffer."""
        while True:
            # 1. Find preamble
            preamble_idx = self._buffer.find(PREAMBLE)
            if preamble_idx == -1:
                # No preamble found; keep only last byte (might be start of preamble)
                if len(self._buffer) > 1:
                    self._buffer = self._buffer[-1:]
                return

            # Discard bytes before preamble
            if preamble_idx > 0:
                self._buffer = self._buffer[preamble_idx:]

            # 2. Check if we have enough for the header
            if len(self._buffer) < FRAME_HEADER_SIZE:
                return

            # 3. Read length
            length = int.from_bytes(self._buffer[2:4], "little")

            if length > self._max_frame_size:
                self._errors.append(f"Frame too large: {length} bytes")
                logger.warning("Frame too large: %d bytes (max %d)", length, self._max_frame_size)
                # Skip this preamble and try again
                self._buffer = self._buffer[2:]
                continue

            total = FRAME_HEADER_SIZE + length + CRC_SIZE
            if len(self._buffer) < total:
                return

            # 4. Extract the complete frame
            frame_bytes = bytes(self._buffer[:total])
            self._buffer = self._buffer[total:]

            # 5. Decode
            encoded_payload = frame_bytes[FRAME_HEADER_SIZE:FRAME_HEADER_SIZE + length]
            crc_received = int.from_bytes(
                frame_bytes[FRAME_HEADER_SIZE + length:FRAME_HEADER_SIZE + length + CRC_SIZE],
                "little",
            )

            raw_payload = COBSCodec.decode(encoded_payload)

            if not CRC16.verify(raw_payload, crc_received):
                self._errors.append(f"CRC mismatch in frame")
                logger.warning("CRC mismatch in frame")
                continue

            try:
                msg = Message.decode_header(raw_payload[:12])
                msg.payload = raw_payload[12:]
                self._frames.append(ParsedFrame(message=msg, raw_frame=frame_bytes, consumed_bytes=total))
            except Exception as e:
                self._errors.append(f"Frame decode error: {e}")
                logger.warning("Frame decode error: %s", e)

    def feed_and_parse(self, data: bytes) -> List[Message]:
        """Convenience: feed data and return all extracted messages."""
        self.feed(data)
        return self.next_messages()
