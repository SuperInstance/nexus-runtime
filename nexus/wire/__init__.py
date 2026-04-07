"""NEXUS Wire Protocol — COBS + CRC-16 encoded messaging for marine networks."""

from nexus.wire.protocol import (
    MessageType, Message, COBSCodec, CRC16,
    PREAMBLE, FRAME_HEADER_SIZE, MAX_FRAME_SIZE,
    encode_frame, decode_frame,
)
from nexus.wire.framing import FrameBuilder, FrameParser, FrameParserState

__all__ = [
    "MessageType", "Message", "COBSCodec", "CRC16",
    "PREAMBLE", "FRAME_HEADER_SIZE", "MAX_FRAME_SIZE",
    "encode_frame", "decode_frame",
    "FrameBuilder", "FrameParser", "FrameParserState",
]
