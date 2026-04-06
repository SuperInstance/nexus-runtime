"""Tests for jetson.comms_hub.protocol."""

import time
import pytest
from jetson.comms_hub.protocol import ProtocolType, MessageFrame, ProtocolHandler


@pytest.fixture
def handler():
    return ProtocolHandler()


class TestProtocolType:
    def test_enum_values(self):
        assert ProtocolType.SERIAL.value == "serial"
        assert ProtocolType.UDP.value == "udp"
        assert ProtocolType.TCP.value == "tcp"
        assert ProtocolType.IRIDIUM_SBD.value == "iridium_sbd"
        assert ProtocolType.LORA.value == "lora"
        assert ProtocolType.RADIO.value == "radio"

    def test_enum_count(self):
        assert len(ProtocolType) == 6

    def test_enum_members_are_unique(self):
        values = [p.value for p in ProtocolType]
        assert len(values) == len(set(values))


class TestMessageFrame:
    def test_default_construction(self):
        f = MessageFrame(header={}, payload=b"hello")
        assert f.payload == b"hello"
        assert f.checksum == 0
        assert f.protocol_type == ProtocolType.UDP
        assert f.fragment_index == 0
        assert f.fragment_total == 1
        assert f.timestamp > 0

    def test_explicit_fields(self):
        f = MessageFrame(
            header={"x": 1},
            payload=b"data",
            checksum=42,
            timestamp=1000.0,
            protocol_type=ProtocolType.LORA,
            fragment_index=2,
            fragment_total=5,
            source="node-a",
            destination="node-b",
        )
        assert f.source == "node-a"
        assert f.destination == "node-b"
        assert f.fragment_index == 2
        assert f.fragment_total == 5
        assert f.checksum == 42

    def test_timestamp_auto_set(self):
        before = time.time()
        f = MessageFrame(header={}, payload=b"x")
        after = time.time()
        assert before <= f.timestamp <= after


class TestChecksum:
    def test_empty_data(self):
        crc = ProtocolHandler.compute_checksum(b"")
        assert isinstance(crc, int)

    def test_known_value(self):
        # Deterministic check
        crc1 = ProtocolHandler.compute_checksum(b"123456789")
        crc2 = ProtocolHandler.compute_checksum(b"123456789")
        assert crc1 == crc2
        assert crc1 != 0

    def test_different_inputs(self):
        crc_a = ProtocolHandler.compute_checksum(b"AAAA")
        crc_b = ProtocolHandler.compute_checksum(b"BBBB")
        assert crc_a != crc_b

    def test_single_byte(self):
        crc = ProtocolHandler.compute_checksum(b"\x00")
        assert isinstance(crc, int)

    def test_long_data(self):
        data = b"\xAB" * 10000
        crc = ProtocolHandler.compute_checksum(data)
        assert 0 <= crc <= 0xFFFF


class TestEncodeDecode:
    def test_roundtrip_udp(self, handler):
        payload = b"hello world"
        encoded = handler.encode(payload, ProtocolType.UDP)
        frame = handler.decode(encoded)
        assert frame.payload == payload
        assert frame.protocol_type == ProtocolType.UDP

    def test_roundtrip_serial(self, handler):
        payload = b"\x01\x02\x03"
        encoded = handler.encode(payload, ProtocolType.SERIAL)
        frame = handler.decode(encoded)
        assert frame.payload == payload

    def test_roundtrip_tcp(self, handler):
        payload = b"TCP payload test"
        encoded = handler.encode(payload, ProtocolType.TCP)
        frame = handler.decode(encoded)
        assert frame.payload == payload

    def test_roundtrip_iridium(self, handler):
        payload = b"SBD message"
        encoded = handler.encode(payload, ProtocolType.IRIDIUM_SBD)
        frame = handler.decode(encoded)
        assert frame.payload == payload

    def test_roundtrip_lora(self, handler):
        payload = b"LoRa data"
        encoded = handler.encode(payload, ProtocolType.LORA)
        frame = handler.decode(encoded)
        assert frame.payload == payload

    def test_roundtrip_radio(self, handler):
        payload = b"Radio voice"
        encoded = handler.encode(payload, ProtocolType.RADIO)
        frame = handler.decode(encoded)
        assert frame.payload == payload

    def test_decode_bad_magic(self, handler):
        with pytest.raises(ValueError, match="Bad magic"):
            handler.decode(b"\x00\x00" + b"\x00" * 11)

    def test_decode_too_short(self, handler):
        with pytest.raises(ValueError, match="too short"):
            handler.decode(b"\x00")

    def test_decode_with_protocol_check(self, handler):
        payload = b"data"
        encoded = handler.encode(payload, ProtocolType.LORA)
        frame = handler.decode(encoded, protocol=ProtocolType.LORA)
        assert frame.protocol_type == ProtocolType.LORA

    def test_decode_protocol_mismatch(self, handler):
        payload = b"data"
        encoded = handler.encode(payload, ProtocolType.LORA)
        with pytest.raises(ValueError, match="Protocol mismatch"):
            handler.decode(encoded, protocol=ProtocolType.UDP)

    def test_encode_empty_payload(self, handler):
        encoded = handler.encode(b"", ProtocolType.UDP)
        frame = handler.decode(encoded)
        assert frame.payload == b""

    def test_encode_large_payload(self, handler):
        payload = b"X" * 5000
        encoded = handler.encode(payload, ProtocolType.UDP)
        frame = handler.decode(encoded)
        assert frame.payload == payload

    def test_checksum_stored(self, handler):
        payload = b"checksum test"
        encoded = handler.encode(payload, ProtocolType.UDP)
        frame = handler.decode(encoded)
        assert frame.checksum != 0


class TestFragmentation:
    def test_no_fragmentation_needed(self, handler):
        frame = MessageFrame(header={}, payload=b"small", protocol_type=ProtocolType.UDP)
        frags = handler.fragment_message(frame, max_size=256)
        assert len(frags) == 1

    def test_fragments_split(self, handler):
        payload = b"A" * 100
        frame = MessageFrame(header={}, payload=payload, protocol_type=ProtocolType.UDP, checksum=12345)
        frags = handler.fragment_message(frame, max_size=40)
        assert len(frags) > 1
        # Reassemble
        reassembled = handler.reassemble_fragments(frags)
        assert reassembled.payload == payload

    def test_reassemble_single_fragment(self, handler):
        payload = b"hello"
        frame = MessageFrame(header={}, payload=payload, protocol_type=ProtocolType.UDP, checksum=42)
        frags = handler.fragment_message(frame, max_size=256)
        reassembled = handler.reassemble_fragments(frags)
        assert reassembled.payload == payload

    def test_reassemble_preserves_protocol(self, handler):
        for proto in ProtocolType:
            frame = MessageFrame(header={}, payload=b"test", protocol_type=proto, checksum=99)
            frags = handler.fragment_message(frame, max_size=256)
            reassembled = handler.reassemble_fragments(frags)
            assert reassembled.protocol_type == proto

    def test_fragment_empty_payload(self, handler):
        frame = MessageFrame(header={}, payload=b"", protocol_type=ProtocolType.UDP)
        frags = handler.fragment_message(frame, max_size=50)
        assert len(frags) >= 1
        reassembled = handler.reassemble_fragments(frags)
        assert reassembled.payload == b""

    def test_max_size_too_small(self, handler):
        frame = MessageFrame(header={}, payload=b"test", protocol_type=ProtocolType.UDP)
        with pytest.raises(ValueError, match="too small"):
            handler.fragment_message(frame, max_size=5)

    def test_reassemble_empty_list(self, handler):
        with pytest.raises(ValueError, match="No fragments"):
            handler.reassemble_fragments([])

    def test_reassemble_missing_fragments(self, handler):
        payload = b"A" * 200
        frame = MessageFrame(header={}, payload=payload, protocol_type=ProtocolType.UDP, checksum=1)
        frags = handler.fragment_message(frame, max_size=50)
        with pytest.raises(ValueError, match="Missing fragments"):
            handler.reassemble_fragments(frags[:1])

    def test_default_mtu_per_protocol(self, handler):
        for proto in ProtocolType:
            assert proto in handler.MTU
            assert handler.MTU[proto] > 0

    def test_large_data_roundtrip(self, handler):
        payload = bytes(range(256)) * 10
        frame = MessageFrame(header={}, payload=payload, protocol_type=ProtocolType.TCP, checksum=555)
        frags = handler.fragment_message(frame, max_size=100)
        reassembled = handler.reassemble_fragments(frags)
        assert reassembled.payload == payload


class TestBandwidthEstimation:
    def test_ideal_conditions(self, handler):
        bw = handler.estimate_bandwidth(ProtocolType.UDP)
        assert bw == handler.BASE_BANDWIDTH[ProtocolType.UDP]

    def test_weak_signal(self, handler):
        bw_full = handler.estimate_bandwidth(ProtocolType.LORA)
        bw_weak = handler.estimate_bandwidth(ProtocolType.LORA, {"signal_strength": 0.5})
        assert bw_weak < bw_full

    def test_interference(self, handler):
        bw_full = handler.estimate_bandwidth(ProtocolType.RADIO)
        bw_int = handler.estimate_bandwidth(ProtocolType.RADIO, {"interference": 0.8})
        assert bw_int < bw_full

    def test_weather(self, handler):
        bw_clear = handler.estimate_bandwidth(ProtocolType.LORA)
        bw_storm = handler.estimate_bandwidth(ProtocolType.LORA, {"weather_factor": 0.7})
        assert bw_storm < bw_clear

    def test_distance_decay_lora(self, handler):
        bw_near = handler.estimate_bandwidth(ProtocolType.LORA, {"distance_km": 1})
        bw_far = handler.estimate_bandwidth(ProtocolType.LORA, {"distance_km": 100})
        assert bw_far < bw_near

    def test_distance_no_effect_udp(self, handler):
        bw_near = handler.estimate_bandwidth(ProtocolType.UDP, {"distance_km": 1})
        bw_far = handler.estimate_bandwidth(ProtocolType.UDP, {"distance_km": 100})
        assert bw_near == bw_far

    def test_iridium_low_bandwidth(self, handler):
        bw = handler.estimate_bandwidth(ProtocolType.IRIDIUM_SBD)
        assert bw < handler.estimate_bandwidth(ProtocolType.SERIAL)

    def test_no_conditions(self, handler):
        bw = handler.estimate_bandwidth(ProtocolType.TCP, None)
        assert bw > 0

    def test_empty_conditions(self, handler):
        bw = handler.estimate_bandwidth(ProtocolType.LORA, {})
        assert bw == handler.BASE_BANDWIDTH[ProtocolType.LORA]

    def test_all_protocols_have_bandwidth(self, handler):
        for proto in ProtocolType:
            bw = handler.estimate_bandwidth(proto)
            assert bw > 0
