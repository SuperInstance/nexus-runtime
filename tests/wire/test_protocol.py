"""Tests for the NEXUS Wire Protocol (25+ tests)."""

import pytest
import struct

from nexus.wire.protocol import (
    MessageType, Message, COBSCodec, CRC16,
    PREAMBLE, FRAME_HEADER_SIZE, CRC_SIZE,
    MAX_PAYLOAD_SIZE, encode_frame, decode_frame,
)


# ---------------------------------------------------------------------------
# CRC-16 tests
# ---------------------------------------------------------------------------

class TestCRC16:
    def test_known_value(self):
        # CRC-16/CCITT-FALSE of "123456789" is 0x29B1
        result = CRC16.compute(b"123456789")
        assert result == 0x29B1

    def test_empty_data(self):
        result = CRC16.compute(b"")
        assert isinstance(result, int)

    def test_single_byte(self):
        result = CRC16.compute(b"\x00")
        assert isinstance(result, int)

    def test_verify_correct(self):
        data = b"hello"
        crc = CRC16.compute(data)
        assert CRC16.verify(data, crc) is True

    def test_verify_incorrect(self):
        data = b"hello"
        assert CRC16.verify(data, 0x0000) is False

    def test_deterministic(self):
        data = b"test data for determinism"
        a = CRC16.compute(data)
        b = CRC16.compute(data)
        assert a == b


# ---------------------------------------------------------------------------
# COBS tests
# ---------------------------------------------------------------------------

class TestCOBS:
    def test_encode_empty(self):
        result = COBSCodec.encode(b"")
        assert isinstance(result, bytes)

    def test_decode_empty(self):
        result = COBSCodec.decode(b"")
        assert result == b""

    def test_no_zero_bytes(self):
        data = b"\x01\x02\x03"
        encoded = COBSCodec.encode(data)
        assert b"\x00" not in encoded

    def test_roundtrip_no_zeros(self):
        data = b"\x01\x02\x03\x04\x05"
        assert COBSCodec.decode(COBSCodec.encode(data)) == data

    def test_roundtrip_with_zeros(self):
        data = b"\x00\x01\x00\xFF\x00"
        assert COBSCodec.decode(COBSCodec.encode(data)) == data

    def test_encoded_no_zero_bytes(self):
        data = bytes(range(256))
        encoded = COBSCodec.encode(data)
        assert b"\x00" not in encoded

    def test_large_roundtrip(self):
        data = bytes(range(256)) * 4
        assert COBSCodec.decode(COBSCodec.encode(data)) == data

    def test_all_zeros(self):
        data = b"\x00" * 10
        encoded = COBSCodec.encode(data)
        decoded = COBSCodec.decode(encoded)
        assert decoded == data

    def test_single_byte(self):
        for b in range(256):
            data = bytes([b])
            assert COBSCodec.decode(COBSCodec.encode(data)) == data


# ---------------------------------------------------------------------------
# Message tests
# ---------------------------------------------------------------------------

class TestMessage:
    def test_create_message(self):
        msg = Message(msg_type=MessageType.HEARTBEAT)
        assert msg.msg_type == MessageType.HEARTBEAT

    def test_encode_header(self):
        msg = Message(
            msg_type=MessageType.SENSOR_DATA,
            source=1,
            destination=2,
            sequence=42,
        )
        header = msg.encode_header()
        assert len(header) == 12

    def test_decode_header(self):
        msg = Message(
            msg_type=MessageType.COMMAND,
            source=10,
            destination=20,
            sequence=99,
        )
        header = msg.encode_header()
        decoded = Message.decode_header(header)
        assert decoded.msg_type == MessageType.COMMAND
        assert decoded.source == 10
        assert decoded.destination == 20
        assert decoded.sequence == 99

    def test_header_too_short(self):
        with pytest.raises(ValueError):
            Message.decode_header(b"\x00" * 5)

    def test_message_repr(self):
        msg = Message(msg_type=MessageType.HEARTBEAT, source=1)
        assert "HEARTBEAT" in repr(msg)


# ---------------------------------------------------------------------------
# Frame encode/decode
# ---------------------------------------------------------------------------

class TestFrame:
    def test_encode_frame_starts_with_preamble(self):
        msg = Message(msg_type=MessageType.HEARTBEAT)
        frame = encode_frame(msg)
        assert frame[:2] == PREAMBLE

    def test_encode_decode_roundtrip(self):
        msg = Message(
            msg_type=MessageType.SENSOR_DATA,
            source=1,
            destination=2,
            sequence=10,
            payload=b"\x01\x02\x03",
        )
        frame = encode_frame(msg)
        decoded = decode_frame(frame)
        assert decoded is not None
        assert decoded.msg_type == MessageType.SENSOR_DATA
        assert decoded.source == 1
        assert decoded.destination == 2
        assert decoded.sequence == 10
        assert decoded.payload == b"\x01\x02\x03"

    def test_decode_returns_none_for_short_data(self):
        assert decode_frame(b"\xaa\x55") is None

    def test_decode_returns_none_for_bad_preamble(self):
        assert decode_frame(b"\x00\x00" + b"\x00" * 10) is None

    def test_decode_returns_none_for_crc_mismatch(self):
        msg = Message(msg_type=MessageType.HEARTBEAT)
        frame = bytearray(encode_frame(msg))
        # Corrupt the payload
        if len(frame) > FRAME_HEADER_SIZE:
            frame[FRAME_HEADER_SIZE] ^= 0xFF
        assert decode_frame(bytes(frame)) is None

    def test_all_message_types_roundtrip(self):
        for mt in MessageType:
            msg = Message(msg_type=mt, payload=b"test")
            frame = encode_frame(msg)
            decoded = decode_frame(frame)
            assert decoded is not None
            assert decoded.msg_type == mt

    def test_empty_payload_roundtrip(self):
        msg = Message(msg_type=MessageType.HEARTBEAT, payload=b"")
        frame = encode_frame(msg)
        decoded = decode_frame(frame)
        assert decoded is not None
        assert decoded.payload == b""

    def test_large_payload_roundtrip(self):
        payload = b"\xAB" * 500
        msg = Message(msg_type=MessageType.BULK_TRANSFER, payload=payload)
        frame = encode_frame(msg)
        decoded = decode_frame(frame)
        assert decoded is not None
        assert decoded.payload == payload

    def test_message_type_count(self):
        """Verify 28 message types are defined."""
        assert len(MessageType) >= 28


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_preamble(self):
        assert PREAMBLE == b"\xaa\x55"

    def test_frame_header_size(self):
        assert FRAME_HEADER_SIZE == 4

    def test_crc_size(self):
        assert CRC_SIZE == 2

    def test_max_payload_size(self):
        assert MAX_PAYLOAD_SIZE > 0
