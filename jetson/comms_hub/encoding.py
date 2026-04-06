"""Binary encoding/decoding: varint, protobuf-like messages, fixed-width types."""

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union
import struct


class WireType(IntEnum):
    VARINT = 0
    FIXED64 = 1
    BYTES = 2
    FIXED32 = 5


@dataclass
class EncodedValue:
    """A single encoded field."""
    tag: int
    value: Any
    wire_type: WireType


class BinaryEncoder:
    """Protobuf-inspired binary encoder/decoder using only stdlib."""

    # ------------------------------------------------------------------
    # Varint
    # ------------------------------------------------------------------
    @staticmethod
    def encode_varint(value: int) -> bytes:
        """Encode an unsigned integer as a varint."""
        if value < 0:
            raise ValueError("Varint value must be non-negative")
        result = bytearray()
        while value > 0x7F:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        result.append(value & 0x7F)
        return bytes(result)

    @staticmethod
    def decode_varint(data: bytes, offset: int = 0) -> Tuple[int, int]:
        """Decode a varint starting at *offset*. Returns (value, bytes_consumed)."""
        value = 0
        shift = 0
        consumed = 0
        while True:
            if offset + consumed >= len(data):
                raise ValueError("Truncated varint")
            byte = data[offset + consumed]
            value |= (byte & 0x7F) << shift
            consumed += 1
            shift += 7
            if (byte & 0x80) == 0:
                break
        return value, consumed

    # ------------------------------------------------------------------
    # Zigzag encoding for signed integers
    # ------------------------------------------------------------------
    @staticmethod
    def encode_zigzag(value: int) -> int:
        """Zigzag encode a signed integer to unsigned."""
        if value >= 0:
            return value * 2
        return (-value) * 2 - 1

    @staticmethod
    def decode_zigzag(value: int) -> int:
        """Zigzag decode an unsigned integer to signed."""
        return (value >> 1) ^ -(value & 1)

    # ------------------------------------------------------------------
    # Fixed-width types
    # ------------------------------------------------------------------
    @staticmethod
    def encode_fixed32(value: Union[int, float]) -> bytes:
        """Encode as little-endian 32-bit."""
        return struct.pack("<I", int(value) & 0xFFFFFFFF)

    @staticmethod
    def encode_fixed64(value: Union[int, float]) -> bytes:
        return struct.pack("<Q", int(value) & 0xFFFFFFFFFFFFFFFF)

    # ------------------------------------------------------------------
    # String
    # ------------------------------------------------------------------
    @staticmethod
    def encode_string(s: str) -> bytes:
        """Encode a UTF-8 string with length prefix (varint)."""
        encoded = s.encode("utf-8")
        length_prefix = BinaryEncoder.encode_varint(len(encoded))
        return length_prefix + encoded

    @staticmethod
    def decode_string(data: bytes, offset: int = 0) -> Tuple[str, int]:
        """Decode a varint-length-prefixed string. Returns (string, total_bytes_consumed)."""
        length, consumed = BinaryEncoder.decode_varint(data, offset)
        s = data[offset + consumed:offset + consumed + length].decode("utf-8")
        return s, consumed + length

    # ------------------------------------------------------------------
    # Float
    # ------------------------------------------------------------------
    @staticmethod
    def encode_float(value: float) -> bytes:
        """Encode a 32-bit IEEE 754 float."""
        return struct.pack("<f", value)

    @staticmethod
    def decode_float(data: bytes, offset: int = 0) -> float:
        """Decode a 32-bit IEEE 754 float at *offset*."""
        return struct.unpack("<f", data[offset:offset + 4])[0]

    # ------------------------------------------------------------------
    # Tag encoding (field_number << 3 | wire_type) as varint
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_tag(field_num: int, wire_type: WireType) -> bytes:
        """Encode tag as varint: (field_number << 3) | wire_type."""
        tag_value = (field_num << 3) | wire_type
        return BinaryEncoder.encode_varint(tag_value)

    @staticmethod
    def _decode_tag(data: bytes, offset: int) -> Tuple[int, WireType, int]:
        """Decode tag varint. Returns (field_number, wire_type, bytes_consumed)."""
        tag_value, consumed = BinaryEncoder.decode_varint(data, offset)
        field_number = tag_value >> 3
        wire_type = WireType(tag_value & 0x07)
        return field_number, wire_type, consumed

    # ------------------------------------------------------------------
    # Protobuf-like message encoding
    # ------------------------------------------------------------------
    @staticmethod
    def encode_message(fields: Dict[int, Any]) -> bytes:
        """Encode a dict of ``{field_number: value}`` into a protobuf-like binary.

        Type inference:
        - ``int`` → varint (wire type 0), signed via zigzag
        - ``float`` → fixed32 (wire type 5)
        - ``str`` → length-delimited (wire type 2)
        - ``bytes`` → length-delimited (wire type 2)
        """
        result = bytearray()
        for field_num, value in sorted(fields.items()):
            if isinstance(value, bool):
                tag = BinaryEncoder._encode_tag(field_num, WireType.VARINT)
                result.extend(tag)
                result.extend(BinaryEncoder.encode_varint(1 if value else 0))
            elif isinstance(value, int):
                tag = BinaryEncoder._encode_tag(field_num, WireType.VARINT)
                result.extend(tag)
                # Use zigzag for signed ints
                if value < 0:
                    zigzag = BinaryEncoder.encode_zigzag(value)
                    result.extend(BinaryEncoder.encode_varint(zigzag))
                else:
                    result.extend(BinaryEncoder.encode_varint(value))
            elif isinstance(value, float):
                tag = BinaryEncoder._encode_tag(field_num, WireType.FIXED32)
                result.extend(tag)
                result.extend(BinaryEncoder.encode_float(value))
            elif isinstance(value, str):
                tag = BinaryEncoder._encode_tag(field_num, WireType.BYTES)
                result.extend(tag)
                result.extend(BinaryEncoder.encode_string(value))
            elif isinstance(value, bytes):
                tag = BinaryEncoder._encode_tag(field_num, WireType.BYTES)
                result.extend(tag)
                result.extend(BinaryEncoder.encode_varint(len(value)))
                result.extend(value)
            elif value is None:
                continue
            else:
                tag = BinaryEncoder._encode_tag(field_num, WireType.BYTES)
                result.extend(tag)
                s = str(value).encode("utf-8")
                result.extend(BinaryEncoder.encode_varint(len(s)))
                result.extend(s)
        return bytes(result)

    @staticmethod
    def decode_message(data: bytes) -> Dict[int, EncodedValue]:
        """Decode protobuf-like binary into ``{field_number: EncodedValue}``."""
        result: Dict[int, EncodedValue] = {}
        offset = 0
        while offset < len(data):
            field_num, wire_type, consumed_tag = BinaryEncoder._decode_tag(data, offset)
            offset += consumed_tag

            if wire_type == WireType.VARINT:
                value, consumed = BinaryEncoder.decode_varint(data, offset)
                offset += consumed
                result[field_num] = EncodedValue(tag=field_num, value=value, wire_type=wire_type)
            elif wire_type == WireType.FIXED32:
                value = BinaryEncoder.decode_float(data, offset)
                offset += 4
                result[field_num] = EncodedValue(tag=field_num, value=value, wire_type=wire_type)
            elif wire_type == WireType.FIXED64:
                value = struct.unpack("<Q", data[offset:offset + 8])[0]
                offset += 8
                result[field_num] = EncodedValue(tag=field_num, value=value, wire_type=wire_type)
            elif wire_type == WireType.BYTES:
                length, consumed = BinaryEncoder.decode_varint(data, offset)
                offset += consumed
                raw = data[offset:offset + length]
                offset += length
                # Try to decode as UTF-8 string
                try:
                    s = raw.decode("utf-8")
                    result[field_num] = EncodedValue(tag=field_num, value=s, wire_type=wire_type)
                except UnicodeDecodeError:
                    result[field_num] = EncodedValue(tag=field_num, value=raw, wire_type=wire_type)
            else:
                raise ValueError(f"Unknown wire type: {wire_type}")
        return result
