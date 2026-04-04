"""NEXUS Jetson tests - Wire protocol round-trip tests."""

import sys
import os

# Add jetson directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "jetson"))

from wire_protocol.cobs import cobs_encode, cobs_decode
from wire_protocol.crc16 import crc16_ccitt


class TestCOBS:
    """COBS encode/decode tests."""

    def test_encode_decode_roundtrip(self) -> None:
        """Verify COBS round-trip preserves data."""
        data = bytes([0x01, 0x02, 0x00, 0x03, 0x04, 0x05, 0x00, 0x06])
        encoded = cobs_encode(data)
        decoded = cobs_decode(encoded)
        assert decoded == data

    def test_empty_input(self) -> None:
        """Verify empty input handling."""
        encoded = cobs_encode(b"")
        assert encoded == b"\x01"

    def test_no_zeros(self) -> None:
        """Verify data without zeros."""
        data = bytes([0x01, 0x02, 0x03, 0x04])
        encoded = cobs_encode(data)
        decoded = cobs_decode(encoded)
        assert decoded == data


class TestCRC16:
    """CRC-16/CCITT-FALSE tests."""

    def test_check_value(self) -> None:
        """Verify CRC check value for '123456789'."""
        crc = crc16_ccitt(b"123456789")
        assert crc == 0x29B1

    def test_empty(self) -> None:
        """Verify CRC of empty data."""
        crc = crc16_ccitt(b"")
        assert crc == 0xFFFF
