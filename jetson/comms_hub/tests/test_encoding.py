"""Tests for jetson.comms_hub.encoding."""

import struct
import math
import pytest
from jetson.comms_hub.encoding import EncodedValue, BinaryEncoder, WireType


@pytest.fixture
def encoder():
    return BinaryEncoder()


class TestEncodedValue:
    def test_construction(self):
        v = EncodedValue(tag=1, value=42, wire_type=WireType.VARINT)
        assert v.tag == 1
        assert v.value == 42
        assert v.wire_type == WireType.VARINT

    def test_wire_types(self):
        assert WireType.VARINT == 0
        assert WireType.FIXED64 == 1
        assert WireType.BYTES == 2
        assert WireType.FIXED32 == 5


class TestVarint:
    def test_zero(self, encoder):
        result = encoder.encode_varint(0)
        assert result == b"\x00"

    def test_small_value(self, encoder):
        result = encoder.encode_varint(127)
        assert result == b"\x7F"

    def test_two_byte(self, encoder):
        result = encoder.encode_varint(128)
        assert len(result) == 2

    def test_decode_zero(self, encoder):
        encoded = encoder.encode_varint(0)
        value, consumed = encoder.decode_varint(encoded)
        assert value == 0
        assert consumed == 1

    def test_roundtrip_small(self, encoder):
        for v in range(1000):
            encoded = encoder.encode_varint(v)
            decoded, consumed = encoder.decode_varint(encoded)
            assert decoded == v

    def test_roundtrip_large(self, encoder):
        big = 2**32 - 1
        encoded = encoder.encode_varint(big)
        decoded, consumed = encoder.decode_varint(encoded)
        assert decoded == big

    def test_roundtrip_very_large(self, encoder):
        big = 2**56
        encoded = encoder.encode_varint(big)
        decoded, _ = encoder.decode_varint(encoded)
        assert decoded == big

    def test_decode_at_offset(self, encoder):
        prefix = b"\xFF\xFF"
        encoded = prefix + encoder.encode_varint(42)
        value, consumed = encoder.decode_varint(encoded, offset=2)
        assert value == 42
        assert consumed > 0

    def test_negative_raises(self, encoder):
        with pytest.raises(ValueError):
            encoder.encode_varint(-1)

    def test_decode_truncated_raises(self, encoder):
        with pytest.raises(ValueError, match="Truncated"):
            encoder.decode_varint(b"\x80\x80")

    def test_one_byte_per_value(self, encoder):
        for v in range(128):
            assert len(encoder.encode_varint(v)) == 1


class TestZigzag:
    def test_positive(self, encoder):
        assert encoder.encode_zigzag(0) == 0
        assert encoder.encode_zigzag(1) == 2
        assert encoder.encode_zigzag(2) == 4

    def test_negative(self, encoder):
        assert encoder.encode_zigzag(-1) == 1
        assert encoder.encode_zigzag(-2) == 3

    def test_roundtrip(self, encoder):
        for v in [-100, -1, 0, 1, 100, 99999]:
            assert encoder.decode_zigzag(encoder.encode_zigzag(v)) == v


class TestFixedWidth:
    def test_fixed32_int(self, encoder):
        result = encoder.encode_fixed32(42)
        assert len(result) == 4
        assert struct.unpack("<I", result)[0] == 42

    def test_fixed64_int(self, encoder):
        result = encoder.encode_fixed64(123456789012345)
        assert len(result) == 8
        assert struct.unpack("<Q", result)[0] == 123456789012345

    def test_fixed32_float_cast(self, encoder):
        result = encoder.encode_fixed32(3.7)
        assert len(result) == 4

    def test_fixed32_zero(self, encoder):
        result = encoder.encode_fixed32(0)
        assert result == b"\x00\x00\x00\x00"

    def test_fixed64_zero(self, encoder):
        result = encoder.encode_fixed64(0)
        assert result == b"\x00" * 8


class TestStringEncoding:
    def test_empty_string(self, encoder):
        result = encoder.encode_string("")
        assert result == b"\x00"

    def test_ascii_string(self, encoder):
        result = encoder.encode_string("hello")
        assert len(result) == 1 + 5

    def test_decode_empty(self, encoder):
        encoded = encoder.encode_string("")
        s, consumed = encoder.decode_string(encoded)
        assert s == ""
        assert consumed == 1

    def test_roundtrip_ascii(self, encoder):
        for s in ["", "a", "hello", "12345", "The quick brown fox"]:
            encoded = encoder.encode_string(s)
            decoded, consumed = encoder.decode_string(encoded)
            assert decoded == s

    def test_roundtrip_unicode(self, encoder):
        s = "héllo wörld 日本語"
        encoded = encoder.encode_string(s)
        decoded, _ = encoder.decode_string(encoded)
        assert decoded == s

    def test_decode_at_offset(self, encoder):
        prefix = b"\xAA\xBB"
        encoded = prefix + encoder.encode_string("hi")
        s, consumed = encoder.decode_string(encoded, offset=2)
        assert s == "hi"
        assert consumed == 3

    def test_long_string(self, encoder):
        s = "X" * 10000
        encoded = encoder.encode_string(s)
        decoded, _ = encoder.decode_string(encoded)
        assert decoded == s


class TestFloatEncoding:
    def test_encode_float(self, encoder):
        result = encoder.encode_float(3.14)
        assert len(result) == 4

    def test_decode_float(self, encoder):
        result = encoder.decode_float(encoder.encode_float(3.14))
        assert abs(result - 3.14) < 1e-6

    def test_float_zero(self, encoder):
        assert encoder.decode_float(encoder.encode_float(0.0)) == 0.0

    def test_float_negative(self, encoder):
        val = -42.5
        assert abs(encoder.decode_float(encoder.encode_float(val)) - val) < 1e-6

    def test_float_inf(self, encoder):
        result = encoder.decode_float(encoder.encode_float(math.inf))
        assert result == math.inf

    def test_float_at_offset(self, encoder):
        prefix = b"\x00\x00"
        encoded = prefix + encoder.encode_float(1.5)
        result = encoder.decode_float(encoded, offset=2)
        assert abs(result - 1.5) < 1e-6


class TestMessageEncoding:
    def test_encode_int_field(self, encoder):
        result = encoder.encode_message({1: 42})
        assert len(result) > 0

    def test_decode_int_field(self, encoder):
        encoded = encoder.encode_message({1: 42})
        decoded = encoder.decode_message(encoded)
        assert 1 in decoded
        assert decoded[1].value == 42
        assert decoded[1].wire_type == WireType.VARINT

    def test_encode_float_field(self, encoder):
        encoded = encoder.encode_message({2: 3.14})
        decoded = encoder.decode_message(encoded)
        assert decoded[2].wire_type == WireType.FIXED32

    def test_encode_string_field(self, encoder):
        encoded = encoder.encode_message({3: "hello"})
        decoded = encoder.decode_message(encoded)
        assert decoded[3].value == "hello"
        assert decoded[3].wire_type == WireType.BYTES

    def test_encode_bytes_field(self, encoder):
        encoded = encoder.encode_message({4: b"\x01\x02\x03"})
        decoded = encoder.decode_message(encoded)
        assert 4 in decoded
        # b"\x01\x02\x03" is valid UTF-8, so it gets decoded as string
        assert decoded[4].value in (b"\x01\x02\x03", "\x01\x02\x03")
        assert decoded[4].wire_type == WireType.BYTES

    def test_encode_non_utf8_bytes(self, encoder):
        encoded = encoder.encode_message({5: b"\xFF\xFE"})
        decoded = encoder.decode_message(encoded)
        assert decoded[5].value == b"\xFF\xFE"
        assert decoded[5].wire_type == WireType.BYTES

    def test_multiple_fields(self, encoder):
        encoded = encoder.encode_message({
            1: 42,
            2: 3.14,
            3: "hello",
        })
        decoded = encoder.decode_message(encoded)
        assert decoded[1].value == 42
        assert abs(decoded[2].value - 3.14) < 1e-6
        assert decoded[3].value == "hello"

    def test_empty_message(self, encoder):
        encoded = encoder.encode_message({})
        assert encoded == b""

    def test_none_value_skipped(self, encoder):
        encoded = encoder.encode_message({1: None, 2: 42})
        decoded = encoder.decode_message(encoded)
        assert 1 not in decoded
        assert decoded[2].value == 42

    def test_large_field_number(self, encoder):
        encoded = encoder.encode_message({100: 1})
        decoded = encoder.decode_message(encoded)
        assert decoded[100].value == 1

    def test_zero_value(self, encoder):
        encoded = encoder.encode_message({1: 0})
        decoded = encoder.decode_message(encoded)
        assert decoded[1].value == 0

    def test_bool_field(self, encoder):
        encoded = encoder.encode_message({1: True})
        decoded = encoder.decode_message(encoded)
        assert decoded[1].value == 1

    def test_negative_int_zigzag(self, encoder):
        encoded = encoder.encode_message({1: -1})
        decoded = encoder.decode_message(encoded)
        # Decoded varint is zigzag-encoded
        zigzag_val = decoded[1].value
        assert encoder.decode_zigzag(zigzag_val) == -1
        assert decoded[1].wire_type == WireType.VARINT

    def test_fallback_to_str(self, encoder):
        class CustomObj:
            def __str__(self):
                return "custom"
        encoded = encoder.encode_message({1: CustomObj()})
        decoded = encoder.decode_message(encoded)
        assert decoded[1].value == "custom"

    def test_decode_empty(self, encoder):
        decoded = encoder.decode_message(b"")
        assert decoded == {}

    def test_large_positive_int(self, encoder):
        val = 2**30
        encoded = encoder.encode_message({1: val})
        decoded = encoder.decode_message(encoded)
        assert decoded[1].value == val


class TestTagEncoding:
    def test_small_field_number(self, encoder):
        field_num, wt, consumed = encoder._decode_tag(
            encoder._encode_tag(1, WireType.VARINT), 0
        )
        assert field_num == 1
        assert wt == WireType.VARINT
        assert consumed == 1

    def test_large_field_number(self, encoder):
        field_num, wt, consumed = encoder._decode_tag(
            encoder._encode_tag(100, WireType.BYTES), 0
        )
        assert field_num == 100
        assert wt == WireType.BYTES

    def test_all_wire_types_in_tag(self, encoder):
        for wt in WireType:
            field_num, decoded_wt, _ = encoder._decode_tag(
                encoder._encode_tag(5, wt), 0
            )
            assert field_num == 5
            assert decoded_wt == wt
