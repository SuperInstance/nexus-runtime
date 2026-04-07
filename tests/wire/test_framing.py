"""Tests for the NEXUS Frame Builder and Parser (15+ tests)."""

import pytest

from nexus.wire.protocol import (
    Message, MessageType, PREAMBLE, encode_frame,
)
from nexus.wire.framing import FrameBuilder, FrameParser, FrameParserState, ParsedFrame


@pytest.fixture
def builder():
    return FrameBuilder(source_id=0x01)


@pytest.fixture
def parser():
    return FrameParser()


class TestFrameBuilder:
    def test_build_heartbeat(self, builder):
        frame = builder.build_heartbeat()
        assert frame[:2] == PREAMBLE

    def test_build_increments_sequence(self, builder):
        f1 = builder.build(Message(msg_type=MessageType.HEARTBEAT))
        f2 = builder.build(Message(msg_type=MessageType.HEARTBEAT))
        # Extract sequence numbers
        from nexus.wire.protocol import Message as Msg
        from nexus.wire.framing import FrameParser
        p = FrameParser()
        msgs = p.feed_and_parse(f1 + f2)
        assert msgs[0].sequence == 0
        assert msgs[1].sequence == 1

    def test_build_with_source_id(self, builder):
        frame = builder.build(Message(msg_type=MessageType.HEARTBEAT))
        p = FrameParser()
        msgs = p.feed_and_parse(frame)
        assert msgs[0].source == 0x01

    def test_reset_sequence(self, builder):
        builder.build(Message(msg_type=MessageType.HEARTBEAT))
        builder.build(Message(msg_type=MessageType.HEARTBEAT))
        builder.reset()
        frame = builder.build(Message(msg_type=MessageType.HEARTBEAT))
        p = FrameParser()
        msgs = p.feed_and_parse(frame)
        assert msgs[0].sequence == 0


class TestFrameParser:
    def test_parse_single_frame(self, parser):
        msg = Message(msg_type=MessageType.HEARTBEAT, source=1, destination=2)
        frame = encode_frame(msg)
        msgs = parser.feed_and_parse(frame)
        assert len(msgs) == 1
        assert msgs[0].msg_type == MessageType.HEARTBEAT
        assert msgs[0].source == 1
        assert msgs[0].destination == 2

    def test_empty_feed(self, parser):
        msgs = parser.feed_and_parse(b"")
        assert len(msgs) == 0

    def test_partial_feed(self, parser):
        msg = Message(msg_type=MessageType.HEARTBEAT)
        frame = encode_frame(msg)
        # Feed first half
        msgs1 = parser.feed_and_parse(frame[: len(frame) // 2])
        assert len(msgs1) == 0
        # Feed second half
        msgs2 = parser.feed_and_parse(frame[len(frame) // 2 :])
        assert len(msgs2) == 1

    def test_multiple_frames(self, parser):
        frames = b""
        for i in range(5):
            msg = Message(msg_type=MessageType.HEARTBEAT, sequence=i)
            frames += encode_frame(msg)
        msgs = parser.feed_and_parse(frames)
        assert len(msgs) == 5

    def test_garbage_before_frame(self, parser):
        msg = Message(msg_type=MessageType.HEARTBEAT)
        frame = encode_frame(msg)
        msgs = parser.feed_and_parse(b"\xFF\xFE\xFD" + frame)
        assert len(msgs) == 1

    def test_has_frame(self, parser):
        assert parser.has_frame() is False
        msg = Message(msg_type=MessageType.HEARTBEAT)
        frame = encode_frame(msg)
        parser.feed(frame)
        assert parser.has_frame() is True

    def test_next_frame(self, parser):
        msg = Message(msg_type=MessageType.HEARTBEAT)
        frame = encode_frame(msg)
        parser.feed(frame)
        parsed = parser.next_frame()
        assert isinstance(parsed, ParsedFrame)
        assert parsed.message.msg_type == MessageType.HEARTBEAT

    def test_next_frame_empty_raises(self, parser):
        with pytest.raises(IndexError):
            parser.next_frame()

    def test_next_messages(self, parser):
        for i in range(3):
            parser.feed(encode_frame(Message(msg_type=MessageType.HEARTBEAT, sequence=i)))
        msgs = parser.next_messages()
        assert len(msgs) == 3

    def test_reset(self, parser):
        msg = Message(msg_type=MessageType.HEARTBEAT)
        frame = encode_frame(msg)
        parser.feed(frame[:4])  # partial
        parser.reset()
        assert parser.buffer_size == 0

    def test_buffer_size(self, parser):
        parser.feed(b"\xAA")
        assert parser.buffer_size == 1

    def test_different_message_types(self, parser):
        types = [MessageType.HEARTBEAT, MessageType.SENSOR_DATA, MessageType.COMMAND]
        frames = b""
        for mt in types:
            frames += encode_frame(Message(msg_type=mt))
        msgs = parser.feed_and_parse(frames)
        assert [m.msg_type for m in msgs] == types

    def test_junk_between_frames(self, parser):
        f1 = encode_frame(Message(msg_type=MessageType.HEARTBEAT))
        f2 = encode_frame(Message(msg_type=MessageType.COMMAND))
        msgs = parser.feed_and_parse(f1 + b"\x00\xFF" + f2)
        assert len(msgs) >= 2


class TestStreamReassembly:
    def test_byte_by_byte(self):
        parser = FrameParser()
        msg = Message(msg_type=MessageType.SENSOR_DATA, payload=b"\x01\x02\x03")
        frame = encode_frame(msg)
        for byte in frame:
            parser.feed(bytes([byte]))
        assert parser.has_frame() is True
        parsed = parser.next_frame()
        assert parsed.message.msg_type == MessageType.SENSOR_DATA

    def test_chunked_frames(self):
        parser = FrameParser()
        f1 = encode_frame(Message(msg_type=MessageType.HEARTBEAT, sequence=0))
        f2 = encode_frame(Message(msg_type=MessageType.HEARTBEAT, sequence=1))
        combined = f1 + f2
        # Feed in 3-byte chunks
        for i in range(0, len(combined), 3):
            parser.feed(combined[i:i + 3])
        msgs = parser.next_messages()
        assert len(msgs) == 2
