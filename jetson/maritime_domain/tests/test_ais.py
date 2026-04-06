"""Tests for AIS processing module."""

import math
import time

import pytest

from jetson.maritime_domain.ais import (
    AISDecoder,
    AISMessage,
    AISMessageType,
)


class TestAISMessage:
    """Tests for AISMessage dataclass."""

    def test_default_values(self):
        msg = AISMessage(mmsi=123456789)
        assert msg.mmsi == 123456789
        assert msg.vessel_name == ""
        assert msg.vessel_type == 0
        assert msg.position is None
        assert msg.heading == 0.0
        assert msg.speed == 0.0
        assert msg.course == 0.0
        assert msg.destination == ""
        assert msg.timestamp == 0.0
        assert msg.msg_type == 0
        assert msg.imo == ""
        assert msg.callsign == ""
        assert msg.draft == 0.0
        assert msg.length == 0.0
        assert msg.beam == 0.0
        assert msg.nav_status == 0

    def test_full_construction(self):
        msg = AISMessage(
            mmsi=211111111,
            vessel_name="TEST VESSEL",
            vessel_type=70,
            position=(53.5, -3.0),
            heading=180.0,
            speed=12.5,
            course=195.0,
            destination="LIVERPOOL",
            timestamp=1700000000.0,
            msg_type=1,
        )
        assert msg.mmsi == 211111111
        assert msg.vessel_name == "TEST VESSEL"
        assert msg.position == (53.5, -3.0)

    def test_position_tuple(self):
        msg = AISMessage(mmsi=1, position=(50.0, -5.0))
        assert msg.position[0] == 50.0
        assert msg.position[1] == -5.0


class TestAISDecoderInit:
    """Tests for AISDecoder initialization."""

    def test_init(self):
        decoder = AISDecoder()
        assert isinstance(decoder, AISDecoder)

    def test_empty_vessel_tracks(self):
        decoder = AISDecoder()
        assert decoder.get_vessel_track(999) == []


class TestValidateChecksum:
    """Tests for checksum validation."""

    def test_valid_checksum(self):
        # Compute correct checksum
        body = "AIVDM,1,1,,B,13u@FT0P00PlM1LN4lV<7?v<04lH,0"
        computed = 0
        for ch in body:
            computed ^= ord(ch)
        sentence = f"!{body}*{computed:02X}"
        assert AISDecoder.validate_checksum(sentence) is True

    def test_valid_checksum_simple(self):
        # Simple test: body checksum
        body = "AIVDM"
        computed = 0
        for ch in body:
            computed ^= ord(ch)
        checksum = f"{computed:02X}"
        sentence = f"!{body}*{checksum}"
        assert AISDecoder.validate_checksum(sentence) is True

    def test_invalid_checksum(self):
        sentence = "!AIVDM,1,1,,B,13u@FT0P00PlM1LN4lV<7?v<04lH,0*FF"
        assert AISDecoder.validate_checksum(sentence) is False

    def test_no_checksum_delimiter(self):
        sentence = "!AIVDM,1,1,,B,13u@FT0P00PlM1LN4lV<7?v<04lH,0"
        assert AISDecoder.validate_checksum(sentence) is False

    def test_empty_sentence(self):
        assert AISDecoder.validate_checksum("") is False

    def test_malformed_checksum(self):
        sentence = "!AIVDM*XYZ"
        assert AISDecoder.validate_checksum(sentence) is False

    def test_whitespace_handling(self):
        body = "AIVDM"
        computed = 0
        for ch in body:
            computed ^= ord(ch)
        checksum = f"{computed:02X}"
        sentence = f"  !{body}*{checksum}  "
        assert AISDecoder.validate_checksum(sentence) is True


class TestDecodeMessage:
    """Tests for AIS message decoding."""

    def test_invalid_sentence_returns_none(self):
        decoder = AISDecoder()
        assert decoder.decode_message("invalid") is None

    def test_bad_checksum_returns_none(self):
        decoder = AISDecoder()
        assert decoder.decode_message("!AIVDM,1,1,,B,test,0*FF") is None

    def test_non_vdm_tag_returns_none(self):
        decoder = AISDecoder()
        # Valid checksum but wrong tag
        body = "GPGLL"
        computed = 0
        for ch in body:
            computed ^= ord(ch)
        sentence = f"!{body},1234.5,N,5678.9,E*{computed:02X}"
        assert decoder.decode_message(sentence) is None

    def test_insufficient_parts(self):
        decoder = AISDecoder()
        body = "AIVDM"
        computed = 0
        for ch in body:
            computed ^= ord(ch)
        sentence = f"!{body}*{computed:02X}"
        assert decoder.decode_message(sentence) is None

    def test_decode_returns_ais_message(self):
        decoder = AISDecoder()
        # Create a valid VDM sentence with a valid checksum
        tag = "AIVDM,1,1,,B,15M6FB0P00JlN4lV<7?v<04lH,0"
        computed = 0
        for ch in tag:
            computed ^= ord(ch)
        sentence = f"!{tag}*{computed:02X}"
        result = decoder.decode_message(sentence)
        assert result is not None
        assert isinstance(result, AISMessage)

    def test_decode_extracts_msg_type(self):
        decoder = AISDecoder()
        # Type 1 position report (6-bit '000001' starts with '1' in armored)
        # Armored: 1 => 6-bit value 1 = '1'
        tag = "AIVDM,1,1,,B,15M6FB0P00JlN4lV<7?v<04lH,0"
        computed = 0
        for ch in tag:
            computed ^= ord(ch)
        sentence = f"!{tag}*{computed:02X}"
        result = decoder.decode_message(sentence)
        assert result is not None
        assert result.msg_type == 1

    def test_decode_extracts_mmsi(self):
        decoder = AISDecoder()
        tag = "AIVDM,1,1,,B,15M6FB0P00JlN4lV<7?v<04lH,0"
        computed = 0
        for ch in tag:
            computed ^= ord(ch)
        sentence = f"!{tag}*{computed:02X}"
        result = decoder.decode_message(sentence)
        assert result is not None
        assert isinstance(result.mmsi, int)
        assert result.mmsi > 0

    def test_decode_timestamp_set(self):
        decoder = AISDecoder()
        tag = "AIVDM,1,1,,B,15M6FB0P00JlN4lV<7?v<04lH,0"
        computed = 0
        for ch in tag:
            computed ^= ord(ch)
        sentence = f"!{tag}*{computed:02X}"
        before = time.time()
        result = decoder.decode_message(sentence)
        after = time.time()
        assert result is not None
        assert before <= result.timestamp <= after

    def test_decode_type1_tracks_vessel(self):
        decoder = AISDecoder()
        tag = "AIVDM,1,1,,B,15M6FB0P00JlN4lV<7?v<04lH,0"
        computed = 0
        for ch in tag:
            computed ^= ord(ch)
        sentence = f"!{tag}*{computed:02X}"
        result = decoder.decode_message(sentence)
        assert result is not None
        track = decoder.get_vessel_track(result.mmsi)
        assert len(track) >= 1


class TestParsePosition:
    """Tests for position parsing."""

    def test_parse_short_payload(self):
        decoder = AISDecoder()
        lat, lon = decoder.parse_position("0" * 50)
        assert (lat, lon) == (0.0, 0.0)

    def test_parse_empty_payload(self):
        decoder = AISDecoder()
        lat, lon = decoder.parse_position("")
        assert (lat, lon) == (0.0, 0.0)

    def test_parse_valid_position(self):
        decoder = AISDecoder()
        # Create payload with specific lat/lon values
        # Lat = 5400000 / 600000 = 9.0 degrees
        # Lon = 10800000 / 600000 = 18.0 degrees
        # Need to encode as 6-bit binary string
        payload = "0" * 168  # enough bits
        # Set lat at bits 89-115 (27 bits): 5400000
        lat_val = 5400000
        lat_bits = f"{lat_val:027b}"
        # Set lon at bits 61-88 (28 bits): 10800000
        lon_val = 10800000
        lon_bits = f"{lon_val:028b}"
        payload = payload[:61] + lon_bits + lat_bits + payload[116:]
        lat, lon = decoder.parse_position(payload)
        assert abs(lat - 9.0) < 0.001
        assert abs(lon - 18.0) < 0.001


class TestClassifyVesselType:
    """Tests for vessel type classification."""

    def test_unknown_negative(self):
        assert AISDecoder.classify_vessel_type(-1) == "Unknown"

    def test_not_available(self):
        assert AISDecoder.classify_vessel_type(0) == "Not Available"

    def test_reserved(self):
        assert AISDecoder.classify_vessel_type(1) == "Reserved"
        assert AISDecoder.classify_vessel_type(6) == "Reserved"

    def test_cargo(self):
        assert AISDecoder.classify_vessel_type(7) == "Cargo"
        assert AISDecoder.classify_vessel_type(9) == "Cargo"

    def test_tanker(self):
        assert AISDecoder.classify_vessel_type(8) == "Tanker"

    def test_fishing(self):
        assert AISDecoder.classify_vessel_type(10) == "Fishing"

    def test_tug(self):
        assert AISDecoder.classify_vessel_type(11) == "Tug"

    def test_pilot(self):
        assert AISDecoder.classify_vessel_type(12) == "Pilot"
        assert AISDecoder.classify_vessel_type(50) == "Pilot"
        assert AISDecoder.classify_vessel_type(60) == "Pilot"

    def test_sar(self):
        assert AISDecoder.classify_vessel_type(13) == "SAR"

    def test_pleasure(self):
        assert AISDecoder.classify_vessel_type(14) == "Pleasure"
        assert AISDecoder.classify_vessel_type(20) == "Pleasure"
        assert AISDecoder.classify_vessel_type(33) == "Pleasure"
        assert AISDecoder.classify_vessel_type(61) == "Pleasure"

    def test_high_speed(self):
        assert AISDecoder.classify_vessel_type(15) == "High Speed"
        assert AISDecoder.classify_vessel_type(64) == "High Speed"

    def test_sailing(self):
        assert AISDecoder.classify_vessel_type(18) == "Sailing"
        assert AISDecoder.classify_vessel_type(32) == "Sailing"
        assert AISDecoder.classify_vessel_type(62) == "Sailing"

    def test_dredger(self):
        assert AISDecoder.classify_vessel_type(30) == "Dredger"
        assert AISDecoder.classify_vessel_type(31) == "Dredger"

    def test_law_enforcement(self):
        assert AISDecoder.classify_vessel_type(34) == "Law Enforcement"

    def test_large_tanker(self):
        assert AISDecoder.classify_vessel_type(80) == "Tanker"
        assert AISDecoder.classify_vessel_type(89) == "Tanker"

    def test_range_90_99_unknown(self):
        assert AISDecoder.classify_vessel_type(90) == "Unknown"
        assert AISDecoder.classify_vessel_type(99) == "Unknown"

    def test_very_large_code(self):
        assert AISDecoder.classify_vessel_type(999) == "Unknown"


class TestComputeCPA:
    """Tests for Closest Point of Approach computation."""

    def test_head_on_convergence(self):
        # Own: at origin, moving north at 10 kts
        # Target: 1nm north, moving south at 10 kts
        cpa_dist, cpa_time = AISDecoder.compute_cpa(
            (0.0, 0.0), (10.0, 0.0),
            (1.0 / 60.0, 0.0), (-10.0, 0.0),
        )
        assert cpa_dist < 0.05  # near miss
        assert cpa_time > 0

    def test_parallel_courses(self):
        # Same direction, offset 2nm east
        cpa_dist, cpa_time = AISDecoder.compute_cpa(
            (0.0, 0.0), (10.0, 0.0),
            (0.0, 2.0 / 60.0), (10.0, 0.0),
        )
        assert abs(cpa_dist - 2.0) < 0.1
        assert cpa_time == float('inf')  # never closes

    def test_diverging_vessels(self):
        cpa_dist, cpa_time = AISDecoder.compute_cpa(
            (0.0, 0.0), (10.0, 0.0),
            (1.0 / 60.0, 0.0), (10.0, 5.0),
        )
        assert cpa_time == 0.0  # already diverging

    def test_same_position(self):
        cpa_dist, cpa_time = AISDecoder.compute_cpa(
            (50.0, -5.0), (0.0, 0.0),
            (50.0, -5.0), (0.0, 0.0),
        )
        assert cpa_dist == 0.0

    def test_crossing_course(self):
        # Own going north, target going east, offset 1nm north
        cpa_dist, cpa_time = AISDecoder.compute_cpa(
            (0.0, 0.0), (10.0, 0.0),
            (0.0, 1.0 / 60.0), (0.0, 10.0),
        )
        assert cpa_dist >= 0.0
        assert isinstance(cpa_time, float)

    def test_cpa_returns_tuple(self):
        result = AISDecoder.compute_cpa(
            (0.0, 0.0), (5.0, 3.0),
            (0.5, 0.5), (3.0, 5.0),
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_cpa_high_latitude(self):
        # Test at high latitude where lon compression matters
        cpa_dist, cpa_time = AISDecoder.compute_cpa(
            (70.0, 0.0), (5.0, 0.0),
            (70.5, 1.0), (-5.0, 0.0),
        )
        assert isinstance(cpa_dist, float)
        assert cpa_dist >= 0


class TestTrackVessel:
    """Tests for vessel tracking."""

    def test_track_single_message(self):
        decoder = AISDecoder()
        msg = AISMessage(mmsi=123, timestamp=100.0)
        track = decoder.track_vessel(123, [msg])
        assert len(track) == 1
        assert track[0].mmsi == 123

    def test_track_multiple_messages(self):
        decoder = AISDecoder()
        msgs = [
            AISMessage(mmsi=123, timestamp=100.0, speed=10.0),
            AISMessage(mmsi=123, timestamp=110.0, speed=11.0),
            AISMessage(mmsi=123, timestamp=120.0, speed=12.0),
        ]
        track = decoder.track_vessel(123, msgs)
        assert len(track) == 3
        assert track[0].speed == 10.0
        assert track[2].speed == 12.0

    def test_track_sorted_by_timestamp(self):
        decoder = AISDecoder()
        msgs = [
            AISMessage(mmsi=456, timestamp=300.0),
            AISMessage(mmsi=456, timestamp=100.0),
            AISMessage(mmsi=456, timestamp=200.0),
        ]
        track = decoder.track_vessel(456, msgs)
        assert track[0].timestamp == 100.0
        assert track[1].timestamp == 200.0
        assert track[2].timestamp == 300.0

    def test_track_deduplication(self):
        decoder = AISDecoder()
        msgs = [
            AISMessage(mmsi=789, timestamp=100.0, speed=10.0),
            AISMessage(mmsi=789, timestamp=100.0, speed=15.0),  # same timestamp
        ]
        track = decoder.track_vessel(789, msgs)
        # Should keep only one entry per timestamp
        timestamps = [m.timestamp for m in track]
        assert len(timestamps) == len(set(timestamps))

    def test_track_independent_mmsi(self):
        decoder = AISDecoder()
        msg1 = AISMessage(mmsi=111, timestamp=100.0)
        msg2 = AISMessage(mmsi=222, timestamp=100.0)
        decoder.track_vessel(111, [msg1])
        decoder.track_vessel(222, [msg2])
        assert len(decoder.get_vessel_track(111)) == 1
        assert len(decoder.get_vessel_track(222)) == 1

    def test_track_appends_to_existing(self):
        decoder = AISDecoder()
        decoder.track_vessel(100, [AISMessage(mmsi=100, timestamp=100.0)])
        decoder.track_vessel(100, [AISMessage(mmsi=100, timestamp=200.0)])
        track = decoder.get_vessel_track(100)
        assert len(track) == 2

    def test_get_nonexistent_track(self):
        decoder = AISDecoder()
        assert decoder.get_vessel_track(999999) == []

    def test_empty_message_list(self):
        decoder = AISDecoder()
        track = decoder.track_vessel(500, [])
        assert track == []


class TestAISMessageType:
    """Tests for AISMessageType enum."""

    def test_type_1(self):
        assert AISMessageType.POSITION_REPORT_CLASS_A == 1

    def test_type_5(self):
        assert AISMessageType.STATIC_AND_VOYAGE == 5

    def test_type_18(self):
        assert AISMessageType.STANDARD_CLASS_B == 18

    def test_type_19(self):
        assert AISMessageType.EXTENDED_CLASS_B == 19
