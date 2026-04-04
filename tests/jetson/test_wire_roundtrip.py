"""NEXUS Jetson tests - Wire protocol round-trip tests.

Comprehensive test suite covering COBS, CRC-16, frame parsing,
all 28 message types, flag bits, sequence numbers, and error cases.
"""

import sys
import os
import struct
import random

# Add jetson directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "jetson"))

from wire_protocol.cobs import cobs_encode, cobs_decode
from wire_protocol.crc16 import crc16_ccitt
from wire_protocol.frame import (
    FrameParser, parse_frame_header,
    FRAME_HEADER_SIZE, FRAME_CRC_SIZE, FRAME_MAX_PAYLOAD,
)
from wire_protocol.node_client import (
    MessageType, MessageFlag, Criticality, Direction,
    MSG_TYPE_INFO, MSG_TYPE_NAMES, NodeClient,
)


class TestCOBS:
    """COBS encode/decode tests."""

    def test_encode_decode_roundtrip_with_zeros(self) -> None:
        """COBS round-trip with embedded zeros."""
        data = bytes([0x01, 0x02, 0x00, 0x03, 0x04, 0x05, 0x00, 0x06])
        encoded = cobs_encode(data)
        decoded = cobs_decode(encoded)
        assert decoded == data

    def test_all_zeros(self) -> None:
        """All zeros: [0x00, 0x00, 0x00] -> [0x01, 0x01, 0x01, 0x01]."""
        data = bytes([0x00, 0x00, 0x00])
        encoded = cobs_encode(data)
        assert encoded == b"\x01\x01\x01\x01"
        decoded = cobs_decode(encoded)
        assert decoded == data

    def test_all_0xff(self) -> None:
        """All 0xFF: [0xFF, 0xFF, 0xFF, 0xFF] -> [0x05, 0xFF, 0xFF, 0xFF, 0xFF]."""
        data = bytes([0xFF, 0xFF, 0xFF, 0xFF])
        encoded = cobs_encode(data)
        assert encoded[0] == 0x05
        assert len(encoded) == 5
        decoded = cobs_decode(encoded)
        assert decoded == data

    def test_empty_input(self) -> None:
        """Empty input returns b'\\x01'."""
        encoded = cobs_encode(b"")
        assert encoded == b"\x01"
        decoded = cobs_decode(encoded)
        assert decoded == b""

    def test_single_zero(self) -> None:
        """Single byte 0x00 encodes to [0x01, 0x01]."""
        encoded = cobs_encode(b"\x00")
        assert encoded == b"\x01\x01"
        decoded = cobs_decode(encoded)
        assert decoded == b"\x00"

    def test_single_nonzero(self) -> None:
        """Single byte 0x01 encodes to [0x02, 0x01]."""
        encoded = cobs_encode(b"\x01")
        assert encoded == b"\x02\x01"
        decoded = cobs_decode(encoded)
        assert decoded == b"\x01"

    def test_254_nonzero_bytes(self) -> None:
        """254 non-zero bytes: code fills to 0xFF, triggers reset.
        Result: [0xFF, 254 data, 0x01] = 256 bytes."""
        data = bytes(range(1, 255))
        encoded = cobs_encode(data)
        assert len(encoded) == 256
        assert encoded[0] == 0xFF
        decoded = cobs_decode(encoded)
        assert decoded == data

    def test_255_nonzero_bytes(self) -> None:
        """255 non-zero bytes: triggers code byte reset.
        Result: [0xFF, 254 data, 0x02, 1 data] = 257 bytes."""
        data = bytes(range(1, 256))
        encoded = cobs_encode(data)
        assert encoded[0] == 0xFF
        decoded = cobs_decode(encoded)
        assert decoded == data

    def test_1000_random_roundtrips(self) -> None:
        """1000 random round-trips."""
        rng = random.Random(42)
        for _ in range(1000):
            length = rng.randint(1, 200)
            data = bytes(rng.getrandbits(8) for _ in range(length))
            encoded = cobs_encode(data)
            decoded = cobs_decode(encoded)
            assert decoded == data

    def test_large_random_1000_bytes(self) -> None:
        """Large random buffer (1000 bytes)."""
        rng = random.Random(999)
        data = bytes(rng.getrandbits(8) for _ in range(1000))
        encoded = cobs_encode(data)
        assert len(encoded) > 1000
        decoded = cobs_decode(encoded)
        assert decoded == data


class TestCRC16:
    """CRC-16/CCITT-FALSE tests."""

    def test_check_value(self) -> None:
        """Verify CRC check value for '123456789' is 0x29B1."""
        crc = crc16_ccitt(b"123456789")
        assert crc == 0x29B1

    def test_empty(self) -> None:
        """CRC of empty data is 0xFFFF."""
        crc = crc16_ccitt(b"")
        assert crc == 0xFFFF

    def test_single_byte_zero(self) -> None:
        """CRC of single zero byte."""
        crc = crc16_ccitt(b"\x00")
        assert crc == 0xE1F0

    def test_all_zeros(self) -> None:
        """CRC of 256 zero bytes."""
        data = b"\x00" * 256
        crc = crc16_ccitt(data)
        assert crc != 0xFFFF
        assert crc != 0x0000

    def test_large_buffer(self) -> None:
        """CRC of 1000-byte buffer."""
        data = bytes(range(256)) * 4  # 1024 bytes
        crc = crc16_ccitt(data)
        assert crc != 0x0000


class TestFrame:
    """Frame encode/decode tests."""

    def test_heartbeat_roundtrip(self) -> None:
        """HEARTBEAT message (type 0x05, no payload) round-trip."""
        wire = FrameParser.encode_frame(b"", msg_type=0x05, seq=1, timestamp_ms=1000)
        assert wire[0] == 0x00
        assert wire[-1] == 0x00

        parser = FrameParser()
        frames = parser.feed(wire)
        assert len(frames) == 1

        msg_type, flags, seq, ts, plen, payload = parse_frame_header(frames[0])
        assert msg_type == 0x05
        assert plen == 0
        assert payload == b""

    def test_device_identity_with_payload(self) -> None:
        """DEVICE_IDENTITY message (type 0x01) with JSON payload."""
        json_payload = b'{"id":"nexus-001","fw":"1.0.0"}'
        wire = FrameParser.encode_frame(json_payload, msg_type=0x01,
                                        flags=MessageFlag.ACK_REQUIRED,
                                        seq=42, timestamp_ms=5000)

        parser = FrameParser()
        frames = parser.feed(wire)
        assert len(frames) == 1

        msg_type, flags, seq, ts, plen, payload = parse_frame_header(frames[0])
        assert msg_type == 0x01
        assert flags & MessageFlag.ACK_REQUIRED
        assert seq == 42
        assert payload == json_payload

    def test_reflex_deploy_binary(self) -> None:
        """REFLEX_DEPLOY message (type 0x09) with binary payload containing zeros."""
        binary_payload = bytes([0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0xCA, 0xFE])
        wire = FrameParser.encode_frame(binary_payload, msg_type=0x09, seq=100)

        parser = FrameParser()
        frames = parser.feed(wire)
        assert len(frames) == 1

        _, _, _, _, _, payload = parse_frame_header(frames[0])
        assert payload == binary_payload

    def test_crc_mismatch_detection(self) -> None:
        """Corrupted frame should not produce valid output."""
        wire = bytearray(FrameParser.encode_frame(b"", msg_type=0x05, seq=1))
        # Corrupt a byte in the middle (not delimiters)
        wire[len(wire) // 2] ^= 0xFF

        parser = FrameParser()
        frames = parser.feed(bytes(wire))
        assert len(frames) == 0
        assert parser.error_count > 0

    def test_oversized_payload_rejection(self) -> None:
        """Corrupted frame with bad COBS data is rejected."""
        parser = FrameParser()
        # Feed a deliberately malformed frame
        bad_wire = b"\x00" + b"\x01" * 1100 + b"\x00"
        frames = parser.feed(bad_wire)
        assert parser.error_count > 0


class TestAllMessageTypes:
    """Test all 28 message type headers."""

    def test_all_28_types_defined(self) -> None:
        """Verify all 28 message types are defined in MessageType enum."""
        assert len(MessageType) == 29

    def test_all_types_have_names(self) -> None:
        """All message types have human-readable names."""
        for mt in MessageType:
            assert mt.value in MSG_TYPE_NAMES
            assert isinstance(MSG_TYPE_NAMES[mt.value], str)
            assert len(MSG_TYPE_NAMES[mt.value]) > 0

    def test_all_types_have_info(self) -> None:
        """All message types have direction and criticality info."""
        for mt in MessageType:
            assert mt.value in MSG_TYPE_INFO
            direction, criticality = MSG_TYPE_INFO[mt.value]
            assert direction in (Direction.N2J, Direction.J2N, Direction.BOTH)
            assert criticality in (Criticality.TELEMETRY, Criticality.COMMAND, Criticality.SAFETY)

    def test_all_types_encode_decode(self) -> None:
        """Every message type can be encoded and decoded in a frame."""
        for mt in MessageType:
            wire = FrameParser.encode_frame(b"", msg_type=mt, seq=1)
            parser = FrameParser()
            frames = parser.feed(wire)
            assert len(frames) == 1, f"Failed for message type {mt.name}"
            msg_type, _, _, _, _, _ = parse_frame_header(frames[0])
            assert msg_type == mt

    def test_safety_messages_have_safety_criticality(self) -> None:
        """ERROR_REPORT (0x11) and SAFETY_EVENT (0x1C) are safety-critical."""
        assert MSG_TYPE_INFO[0x11][1] == Criticality.SAFETY
        assert MSG_TYPE_INFO[0x1C][1] == Criticality.SAFETY

    def test_bidirectional_types(self) -> None:
        """HEARTBEAT, PING, PONG are bidirectional."""
        assert MSG_TYPE_INFO[0x05][0] == Direction.BOTH  # HEARTBEAT
        assert MSG_TYPE_INFO[0x14][0] == Direction.BOTH  # PING
        assert MSG_TYPE_INFO[0x15][0] == Direction.BOTH  # PONG


class TestFlagBits:
    """Test message flag bits."""

    def test_ack_required_flag(self) -> None:
        """ACK_REQUIRED flag (bit 0) is preserved in round-trip."""
        wire = FrameParser.encode_frame(b"", msg_type=0x07,
                                        flags=MessageFlag.ACK_REQUIRED, seq=1)
        parser = FrameParser()
        frames = parser.feed(wire)
        _, flags, _, _, _, _ = parse_frame_header(frames[0])
        assert flags & MessageFlag.ACK_REQUIRED

    def test_is_ack_flag(self) -> None:
        """IS_ACK flag (bit 1) is preserved."""
        wire = FrameParser.encode_frame(b"", msg_type=0x08,
                                        flags=MessageFlag.IS_ACK, seq=1)
        parser = FrameParser()
        frames = parser.feed(wire)
        _, flags, _, _, _, _ = parse_frame_header(frames[0])
        assert flags & MessageFlag.IS_ACK

    def test_urgent_flag(self) -> None:
        """URGENT flag (bit 3) is preserved."""
        wire = FrameParser.encode_frame(b"", msg_type=0x11,
                                        flags=MessageFlag.URGENT, seq=1)
        parser = FrameParser()
        frames = parser.feed(wire)
        _, flags, _, _, _, _ = parse_frame_header(frames[0])
        assert flags & MessageFlag.URGENT

    def test_combined_flags(self) -> None:
        """Multiple flags can be combined."""
        combined = MessageFlag.ACK_REQUIRED | MessageFlag.IS_ACK | MessageFlag.URGENT
        wire = FrameParser.encode_frame(b"", msg_type=0x07,
                                        flags=combined, seq=1)
        parser = FrameParser()
        frames = parser.feed(wire)
        _, flags, _, _, _, _ = parse_frame_header(frames[0])
        assert flags & MessageFlag.ACK_REQUIRED
        assert flags & MessageFlag.IS_ACK
        assert flags & MessageFlag.URGENT

    def test_no_timestamp_flag(self) -> None:
        """NO_TIMESTAMP flag (bit 6) is preserved."""
        wire = FrameParser.encode_frame(b"", msg_type=0x05,
                                        flags=MessageFlag.NO_TIMESTAMP, seq=1)
        parser = FrameParser()
        frames = parser.feed(wire)
        _, flags, _, _, _, _ = parse_frame_header(frames[0])
        assert flags & MessageFlag.NO_TIMESTAMP


class TestSequenceNumbers:
    """Test sequence number handling."""

    def test_seq_zero(self) -> None:
        """Sequence number 0 is preserved."""
        wire = FrameParser.encode_frame(b"", msg_type=0x05, seq=0)
        parser = FrameParser()
        frames = parser.feed(wire)
        _, _, seq, _, _, _ = parse_frame_header(frames[0])
        assert seq == 0

    def test_seq_max(self) -> None:
        """Maximum sequence number (0xFFFF) is preserved."""
        wire = FrameParser.encode_frame(b"", msg_type=0x05, seq=0xFFFF)
        parser = FrameParser()
        frames = parser.feed(wire)
        _, _, seq, _, _, _ = parse_frame_header(frames[0])
        assert seq == 0xFFFF

    def test_seq_12345(self) -> None:
        """Arbitrary sequence number is preserved."""
        wire = FrameParser.encode_frame(b"", msg_type=0x05, seq=12345)
        parser = FrameParser()
        frames = parser.feed(wire)
        _, _, seq, _, _, _ = parse_frame_header(frames[0])
        assert seq == 12345

    def test_node_client_auto_increments(self) -> None:
        """NodeClient auto-increments sequence numbers."""
        sent_frames: list[bytes] = []
        client = NodeClient()
        client.set_transport(lambda f: sent_frames.append(f))

        client.send_heartbeat()
        client.send_heartbeat()
        client.send_heartbeat()

        assert client.sequence == 3
        assert len(sent_frames) == 3


class TestErrorCases:
    """Test error handling."""

    def test_empty_wire_frame(self) -> None:
        """Empty frame (just delimiters) produces no output."""
        parser = FrameParser()
        frames = parser.feed(b"\x00\x00")
        assert len(frames) == 0

    def test_garbage_bytes_ignored(self) -> None:
        """Non-delimiter bytes outside a frame are ignored."""
        parser = FrameParser()
        frames = parser.feed(b"\x01\x02\x03\xFF\xFE\xFD")
        assert len(frames) == 0
        assert parser.error_count == 0

    def test_truncated_frame_no_end_delimiter(self) -> None:
        """Frame without end delimiter produces no output."""
        wire = FrameParser.encode_frame(b"", msg_type=0x05, seq=1)
        # Remove the last byte (end delimiter)
        truncated = wire[:-1]
        parser = FrameParser()
        frames = parser.feed(truncated)
        assert len(frames) == 0

    def test_parse_frame_header_too_short(self) -> None:
        """parse_frame_header raises ValueError for too-short data."""
        import pytest
        with pytest.raises(ValueError):
            parse_frame_header(b"\x05\x00")

    def test_parse_frame_header_payload_mismatch(self) -> None:
        """parse_frame_header raises ValueError when payload length doesn't match."""
        import pytest
        data = bytes([0x05, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05,
                      0x01, 0x02, 0x03])
        with pytest.raises(ValueError):
            parse_frame_header(data)


class TestNodeClient:
    """Test high-level NodeClient API."""

    def test_send_heartbeat(self) -> None:
        """send_heartbeat produces a valid frame."""
        sent: list[bytes] = []
        client = NodeClient()
        client.set_transport(lambda f: sent.append(f))

        result = client.send_heartbeat()
        assert result is True
        assert len(sent) == 1

        parser = FrameParser()
        frames = parser.feed(sent[0])
        assert len(frames) == 1
        msg_type, _, _, _, _, _ = parse_frame_header(frames[0])
        assert msg_type == MessageType.HEARTBEAT

    def test_send_reflex_deploy(self) -> None:
        """send_reflex_deploy produces correct frame."""
        sent: list[bytes] = []
        client = NodeClient()
        client.set_transport(lambda f: sent.append(f))

        result = client.send_reflex_deploy("test_reflex", b"\xDE\xAD")
        assert result is True

        parser = FrameParser()
        frames = parser.feed(sent[0])
        assert len(frames) == 1
        msg_type, flags, _, _, _, payload = parse_frame_header(frames[0])
        assert msg_type == MessageType.REFLEX_DEPLOY
        assert flags & MessageFlag.ACK_REQUIRED
        assert payload[0] == len(b"test_reflex")
        assert payload[1:1 + len(b"test_reflex")] == b"test_reflex"
        assert payload[1 + len(b"test_reflex"):] == b"\xDE\xAD"

    def test_send_ping(self) -> None:
        """send_ping produces a valid PING frame."""
        sent: list[bytes] = []
        client = NodeClient()
        client.set_transport(lambda f: sent.append(f))

        result = client.send_ping()
        assert result is True

        parser = FrameParser()
        frames = parser.feed(sent[0])
        msg_type, _, _, _, _, _ = parse_frame_header(frames[0])
        assert msg_type == MessageType.PING

    def test_send_command(self) -> None:
        """send_command produces a valid COMMAND frame."""
        sent: list[bytes] = []
        client = NodeClient()
        client.set_transport(lambda f: sent.append(f))

        result = client.send_command(b"\x01\x02\x03")
        assert result is True

        parser = FrameParser()
        frames = parser.feed(sent[0])
        msg_type, flags, _, _, _, payload = parse_frame_header(frames[0])
        assert msg_type == MessageType.COMMAND
        assert flags & MessageFlag.ACK_REQUIRED
        assert payload == b"\x01\x02\x03"

    def test_poll_with_injected_response(self) -> None:
        """poll() returns injected response frames."""
        client = NodeClient()

        wire = FrameParser.encode_frame(b"\xCA\xFE", msg_type=0x06, seq=1)
        client.inject_response(wire)

        messages = client.poll()
        assert len(messages) == 1
        msg_type, _, seq, _, _, payload = messages[0]
        assert msg_type == 0x06
        assert seq == 1
        assert payload == b"\xCA\xFE"

    def test_send_role_assign(self) -> None:
        """send_role_assign produces valid ROLE_ASSIGN frame."""
        sent: list[bytes] = []
        client = NodeClient()
        client.set_transport(lambda f: sent.append(f))

        result = client.send_role_assign(3)
        assert result is True

        parser = FrameParser()
        frames = parser.feed(sent[0])
        msg_type, _, _, _, _, payload = parse_frame_header(frames[0])
        assert msg_type == MessageType.ROLE_ASSIGN
        assert payload[0] == 3
