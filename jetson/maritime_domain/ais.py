"""
AIS (Automatic Identification System) Processing Module.

Decodes AIS NMEA sentences, validates checksums, parses positions,
classifies vessel types, computes CPA (Closest Point of Approach),
and tracks vessel history.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple


class AISMessageType(IntEnum):
    """AIS message type identifiers."""
    POSITION_REPORT_CLASS_A = 1
    POSITION_REPORT_CLASS_A_ASSIGNED = 2
    POSITION_REPORT_CLASS_A_RESPONSE = 3
    BASE_STATION_REPORT = 4
    STATIC_AND_VOYAGE = 5
    BINARY_ADDRESSED = 6
    BINARY_BROADCAST = 7
    BINARY_ACKNOWLEDGE = 8
    SAR_AIRCRAFT_POSITION = 9
    UTC_DATE_INQUIRY = 10
    UTC_DATE_RESPONSE = 11
    ADDRESSED_SAFETY = 12
    SAFETY_BROADCAST = 14
    INTERROGATION = 15
    ASSIGNMENT_COMMAND = 16
    DGNSS_BROADCAST = 17
    STANDARD_CLASS_B = 18
    EXTENDED_CLASS_B = 19
    DATA_LINK = 23
    GROUP_ASSIGNMENT = 25


@dataclass
class AISMessage:
    """Represents a decoded AIS message."""
    mmsi: int
    vessel_name: str = ""
    vessel_type: int = 0
    position: Optional[Tuple[float, float]] = None  # (lat, lon)
    heading: float = 0.0          # degrees true, 511 = not available
    speed: float = 0.0            # knots, 102.3 = not available
    course: float = 0.0           # degrees true, 360 = not available
    destination: str = ""
    timestamp: float = 0.0        # unix epoch
    msg_type: int = 0
    imo: str = ""
    callsign: str = ""
    draft: float = 0.0            # meters
    length: float = 0.0           # meters
    beam: float = 0.0             # meters
    nav_status: int = 0           # navigation status
    raw_payload: str = ""


class AISDecoder:
    """Decodes AIS NMEA VDM/VDO sentences and manages vessel tracking."""

    def __init__(self) -> None:
        self._fragment_buffer: dict = {}
        self._fragment_count: dict = {}
        self._vessel_tracks: dict[int, List[AISMessage]] = {}

    @staticmethod
    def validate_checksum(sentence: str) -> bool:
        """
        Validate the checksum of an NMEA sentence.
        Format: !<tag>*<hex_checksum>
        """
        sentence = sentence.strip()
        if '*' not in sentence:
            return False
        try:
            body, checksum_str = sentence.rsplit('*', 1)
            body = body.lstrip('!')
            computed = 0
            for ch in body:
                computed ^= ord(ch)
            expected = int(checksum_str, 16)
            return computed == expected
        except (ValueError, IndexError):
            return False

    @staticmethod
    def _sixbit_to_ascii(sixbit: int) -> str:
        """Convert a 6-bit AIS value to the corresponding ASCII character."""
        if sixbit < 32:
            return chr(sixbit + 64)
        if sixbit < 64:
            return chr(sixbit)
        return '?'

    @staticmethod
    def _decode_sixbit_string(payload: str, start_bit: int, num_chars: int) -> str:
        """Decode a string from a 6-bit packed AIS payload."""
        result = []
        for i in range(num_chars):
            byte_offset = (start_bit + i * 6) // 6
            if byte_offset < len(payload):
                val = int(payload[byte_offset], 2) if len(payload[byte_offset]) == 6 else 0
                char = AISDecoder._sixbit_to_ascii(val)
                if char == '@':
                    break
                result.append(char)
        return ''.join(result).strip()

    @staticmethod
    def _get_bits(payload: str, start: int, length: int) -> int:
        """Extract bits from a binary string payload."""
        bits = payload[start:start + length]
        if not bits:
            return 0
        return int(bits, 2) if len(bits) == length else 0

    @staticmethod
    def _nmea_to_sixbit(armored: str) -> str:
        """Convert NMEA-armored 6-bit string to binary string."""
        result = []
        for ch in armored:
            val = ord(ch) - 48
            if val >= 40:
                val -= 8
            result.append(f"{val:06b}")
        return ''.join(result)

    def decode_message(self, raw_nmea: str) -> Optional[AISMessage]:
        """
        Decode a raw AIS NMEA sentence into an AISMessage.

        Supports VDM (VHF Data-link Message) sentences.
        Handles both single-part and multi-part messages.
        """
        raw_nmea = raw_nmea.strip()
        if not raw_nmea.startswith('!') or not self.validate_checksum(raw_nmea):
            return None

        parts = raw_nmea.split(',')
        if len(parts) < 7:
            return None

        tag = parts[0]
        if tag not in ('!AIVDM', '!AIVDO'):
            return None

        fragment_count = int(parts[1]) if parts[1] else 1
        fragment_num = int(parts[2]) if parts[2] else 1
        seq_id = int(parts[3]) if parts[3] else 0
        armored_payload = parts[5]
        # Last field may contain *checksum, extract just the fill bits
        last_field = parts[6].split("*")[0] if parts[6] else "0"
        fill_bits = int(last_field) if last_field else 0

        # Handle multi-part messages
        if fragment_count > 1:
            key = (seq_id, fragment_count)
            if key not in self._fragment_buffer:
                self._fragment_buffer[key] = {}
                self._fragment_count[key] = fragment_count
            self._fragment_buffer[key][fragment_num] = (armored_payload, fill_bits)

            if len(self._fragment_buffer[key]) == fragment_count:
                full_armored = ''
                final_fill = fill_bits
                for i in range(1, fragment_count + 1):
                    p, f = self._fragment_buffer[key][i]
                    full_armored += p
                    if i == fragment_count:
                        final_fill = f
                payload = self._nmea_to_sixbit(full_armored)[:-final_fill] if final_fill else self._nmea_to_sixbit(full_armored)
                del self._fragment_buffer[key]
                del self._fragment_count[key]
            else:
                return None
        else:
            payload = self._nmea_to_sixbit(armored_payload)
            if fill_bits:
                payload = payload[:-fill_bits]

        msg_type = self._get_bits(payload, 0, 6)
        mmsi = self._get_bits(payload, 8, 30)
        msg = AISMessage(
            mmsi=mmsi,
            msg_type=msg_type,
            timestamp=time.time(),
            raw_payload=payload,
        )

        if msg_type in (1, 2, 3):
            self._decode_position_report(payload, msg)
        elif msg_type == 5:
            self._decode_static_voyage(payload, msg)
        elif msg_type == 18:
            self._decode_class_b(payload, msg)
        elif msg_type == 19:
            self._decode_extended_class_b(payload, msg)

        # Track vessel
        self._record_track(mmsi, msg)
        return msg

    def _decode_position_report(self, payload: str, msg: AISMessage) -> None:
        """Decode Type 1/2/3 position report."""
        msg.speed = self._get_bits(payload, 50, 10) * 0.1
        accuracy = self._get_bits(payload, 60, 1)
        lon_raw = self._get_bits(payload, 61, 28)
        lat_raw = self._get_bits(payload, 89, 27)
        msg.course = self._get_bits(payload, 116, 12) * 0.1
        msg.heading = self._get_bits(payload, 128, 9)
        msg.nav_status = self._get_bits(payload, 38, 4)

        if lon_raw != 0x6791AC0:
            lon = lon_raw / 600000.0
            if accuracy:
                lon = round(lon, 4)
            msg.position = (0.0, lon)  # placeholder lat
        if lat_raw != 0x3412140:
            lat = lat_raw / 600000.0
            if accuracy:
                lat = round(lat, 4)
            if msg.position:
                msg.position = (lat, msg.position[1])
            else:
                msg.position = (lat, 0.0)

    def _decode_static_voyage(self, payload: str, msg: AISMessage) -> None:
        """Decode Type 5 static and voyage related data."""
        msg.imo = self._decode_sixbit_string(payload, 70, 7)
        msg.callsign = self._decode_sixbit_string(payload, 112, 7)
        msg.vessel_name = self._decode_sixbit_string(payload, 160, 20)
        msg.vessel_type = self._get_bits(payload, 300, 8)
        msg.length = self._get_bits(payload, 308, 9)
        msg.beam = self._get_bits(payload, 323, 6)
        msg.destination = self._decode_sixbit_string(payload, 384, 20)
        msg.draft = self._get_bits(payload, 376, 8) * 0.1

    def _decode_class_b(self, payload: str, msg: AISMessage) -> None:
        """Decode Type 18 standard Class B CS position report."""
        msg.speed = self._get_bits(payload, 46, 10) * 0.1
        accuracy = self._get_bits(payload, 56, 1)
        lon_raw = self._get_bits(payload, 57, 28)
        lat_raw = self._get_bits(payload, 85, 27)
        msg.course = self._get_bits(payload, 112, 12) * 0.1
        msg.heading = self._get_bits(payload, 124, 9)

        lat = lat_raw / 600000.0
        lon = lon_raw / 600000.0
        if lat_raw != 0x3412140 or lon_raw != 0x6791AC0:
            msg.position = (lat, lon)

    def _decode_extended_class_b(self, payload: str, msg: AISMessage) -> None:
        """Decode Type 19 extended Class B CS position report."""
        msg.speed = self._get_bits(payload, 46, 10) * 0.1
        accuracy = self._get_bits(payload, 56, 1)
        lon_raw = self._get_bits(payload, 57, 28)
        lat_raw = self._get_bits(payload, 85, 27)
        msg.course = self._get_bits(payload, 112, 12) * 0.1
        msg.heading = self._get_bits(payload, 124, 9)
        msg.vessel_name = self._decode_sixbit_string(payload, 143, 20)
        msg.vessel_type = self._get_bits(payload, 263, 8)

        lat = lat_raw / 600000.0
        lon = lon_raw / 600000.0
        if lat_raw != 0x3412140 or lon_raw != 0x6791AC0:
            msg.position = (lat, lon)

    def parse_position(self, payload: str) -> Tuple[float, float]:
        """
        Parse latitude/longitude from a 6-bit binary payload string.
        Returns (latitude, longitude) in decimal degrees.
        """
        if not payload or len(payload) < 116:
            return (0.0, 0.0)
        lon_raw = self._get_bits(payload, 61, 28)
        lat_raw = self._get_bits(payload, 89, 27)
        lat = lat_raw / 600000.0
        lon = lon_raw / 600000.0
        return (lat, lon)

    @staticmethod
    def classify_vessel_type(type_code: int) -> str:
        """
        Classify a vessel by its AIS type code into a human-readable category.
        Returns one of: 'Cargo', 'Tanker', 'Passenger', 'Fishing', 'Tug',
        'Pilot', 'SAR', 'Pleasure', 'Military', 'Sailing', 'High Speed',
        'Dredger', 'Law Enforcement', 'Unknown', 'Not Available'.
        """
        classification = {
            (0,): 'Not Available',
            (1, 2, 3, 4, 5, 6): 'Reserved',
            (7,): 'Cargo',
            (8,): 'Tanker',
            (9,): 'Cargo',
            (10,): 'Fishing',
            (11,): 'Tug',
            (12,): 'Pilot',
            (13,): 'SAR',
            (14,): 'Pleasure',
            (15,): 'High Speed',
            (16, 17): 'High Speed',
            (18,): 'Sailing',
            (19,): 'Reserved',
            (20,): 'Pleasure',
            (21, 22, 23, 24, 25, 26, 27, 28, 29): 'Cargo',
            (30,): 'Dredger',
            (31,): 'Dredger',
            (32,): 'Sailing',
            (33,): 'Pleasure',
            (34,): 'Law Enforcement',
            (35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49): 'SAR',
            (50,): 'Pilot',
            (51, 52, 53, 54, 55, 56, 57, 58, 59): 'SAR',
            (60,): 'Pilot',
            (61,): 'Pleasure',
            (62,): 'Sailing',
            (63,): 'Reserved',
            (64,): 'High Speed',
            (65,): 'High Speed',
            (66,): 'High Speed',
            (67,): 'Cargo',
            (68,): 'Cargo',
            (69,): 'Cargo',
            (70,): 'Cargo',
            (71,): 'Cargo',
            (72,): 'Cargo',
            (73,): 'Cargo',
            (74,): 'Cargo',
            (75,): 'Cargo',
            (76,): 'Cargo',
            (77,): 'Cargo',
            (78,): 'Cargo',
            (79,): 'Cargo',
            (80,): 'Tanker',
            (81,): 'Tanker',
            (82,): 'Tanker',
            (83,): 'Tanker',
            (84,): 'Tanker',
            (85,): 'Tanker',
            (86,): 'Tanker',
            (87,): 'Tanker',
            (88,): 'Tanker',
            (89,): 'Tanker',
        }
        for codes, category in classification.items():
            if type_code in codes:
                return category

        # Broader ranges
        if 90 <= type_code <= 99:
            return 'Unknown'
        return 'Unknown'

    @staticmethod
    def compute_cpa(
        own_position: Tuple[float, float],
        own_velocity: Tuple[float, float],
        ais_position: Tuple[float, float],
        ais_velocity: Tuple[float, float],
    ) -> Tuple[float, float]:
        """
        Compute Closest Point of Approach (CPA) between own vessel and an AIS target.
        Returns (cpa_distance_nm, cpa_time_minutes).
        Positions in (lat, lon), velocities in (north_knots, east_knots).
        """
        # Relative position in nautical miles (approximate)
        dlat = (ais_position[0] - own_position[0]) * 60.0  # nm
        dlon = (ais_position[1] - own_position[1]) * 60.0 * math.cos(math.radians(own_position[0]))
        rel_north = dlat
        rel_east = dlon

        # Relative velocity
        rel_vn = ais_velocity[0] - own_velocity[0]
        rel_ve = ais_velocity[1] - own_velocity[1]

        speed_sq = rel_vn ** 2 + rel_ve ** 2

        if speed_sq < 1e-10:
            # Vessels on parallel or near-zero relative speed
            cpa_dist = math.sqrt(rel_north ** 2 + rel_east ** 2)
            return (round(cpa_dist, 4), float('inf'))

        # Time to CPA
        t_cpa = -(rel_north * rel_vn + rel_east * rel_ve) / speed_sq

        if t_cpa < 0:
            # Vessels diverging
            cpa_dist = math.sqrt(rel_north ** 2 + rel_east ** 2)
            return (round(cpa_dist, 4), 0.0)

        # CPA position
        cpa_north = rel_north + rel_vn * t_cpa
        cpa_east = rel_east + rel_ve * t_cpa
        cpa_dist = math.sqrt(cpa_north ** 2 + cpa_east ** 2)

        cpa_time_min = t_cpa * 60.0  # convert hours to minutes
        return (round(cpa_dist, 4), round(cpa_time_min, 2))

    def track_vessel(self, mmsi: int, messages: List[AISMessage]) -> List[AISMessage]:
        """
        Return the sorted track history for a given vessel (MMSI).
        Merges new messages with existing track, sorted by timestamp.
        """
        if mmsi not in self._vessel_tracks:
            self._vessel_tracks[mmsi] = []
        self._vessel_tracks[mmsi].extend(messages)
        self._vessel_tracks[mmsi].sort(key=lambda m: m.timestamp)
        # Deduplicate by timestamp (keep latest)
        seen = set()
        deduped = []
        for msg in reversed(self._vessel_tracks[mmsi]):
            if msg.timestamp not in seen:
                seen.add(msg.timestamp)
                deduped.append(msg)
        deduped.reverse()
        self._vessel_tracks[mmsi] = deduped
        return list(deduped)

    def get_vessel_track(self, mmsi: int) -> List[AISMessage]:
        """Return the track history for a vessel without adding new messages."""
        return list(self._vessel_tracks.get(mmsi, []))

    def _record_track(self, mmsi: int, msg: AISMessage) -> None:
        """Record a decoded message into the vessel track store."""
        if mmsi not in self._vessel_tracks:
            self._vessel_tracks[mmsi] = []
        self._vessel_tracks[mmsi].append(msg)
