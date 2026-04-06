"""Multi-protocol support: serial, UDP, TCP-like, Iridium SBD, LoRa, Radio."""

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional, List
import struct
import time


class ProtocolType(Enum):
    SERIAL = "serial"
    UDP = "udp"
    TCP = "tcp"
    IRIDIUM_SBD = "iridium_sbd"
    LORA = "lora"
    RADIO = "radio"


@dataclass
class MessageFrame:
    """A framed message ready for transmission."""
    header: dict
    payload: bytes
    checksum: int = 0
    timestamp: float = 0.0
    protocol_type: ProtocolType = ProtocolType.UDP
    fragment_index: int = 0
    fragment_total: int = 1
    source: str = ""
    destination: str = ""

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class ProtocolHandler:
    """Encode/decode messages across multiple protocols with CRC-16, fragmentation."""

    BASE_BANDWIDTH = {
        ProtocolType.SERIAL: 115_200,
        ProtocolType.UDP: 1_000_000_000,
        ProtocolType.TCP: 1_000_000_000,
        ProtocolType.IRIDIUM_SBD: 2_400,
        ProtocolType.LORA: 50_000,
        ProtocolType.RADIO: 9_600,
    }

    MTU = {
        ProtocolType.SERIAL: 256,
        ProtocolType.UDP: 65507,
        ProtocolType.TCP: 65535,
        ProtocolType.IRIDIUM_SBD: 340,
        ProtocolType.LORA: 255,
        ProtocolType.RADIO: 128,
    }

    # Wire: [2B magic][2B crc][1B proto_id][4B ts][4B plen][payload] = 13 header
    WIRE_HEADER_SIZE = 13

    # Fragment: [2B frag_crc][2B orig_crc][2B total][1B index][2B plen][1B proto_id][payload]
    FRAG_HEADER_SIZE = 10

    def __init__(self):
        self._msg_counter = 0

    def _next_id(self) -> int:
        self._msg_counter += 1
        return self._msg_counter

    @staticmethod
    def compute_checksum(data: bytes) -> int:
        """CRC-16/CCITT-FALSE over *data*."""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc <<= 1
                crc &= 0xFFFF
        return crc

    def encode(self, payload: bytes, protocol: ProtocolType, **meta) -> bytes:
        """Encode *payload* into a protocol-specific wire format.

        Wire format::
            [2B magic] [2B checksum] [1B proto_id] [4B timestamp] [4B payload_len] [payload]
        """
        proto_id = list(ProtocolType).index(protocol)
        ts = int(time.time())
        body = struct.pack("!BII", proto_id, ts, len(payload)) + payload
        crc = self.compute_checksum(body)
        magic = 0x4E58
        frame = struct.pack("!HH", magic, crc) + body
        return frame

    def decode(self, data: bytes, protocol: Optional[ProtocolType] = None) -> MessageFrame:
        """Decode wire *data* into a :class:`MessageFrame`."""
        if len(data) < self.WIRE_HEADER_SIZE:
            raise ValueError("Frame too short to decode")
        magic, crc_received = struct.unpack_from("!HH", data, 0)
        if magic != 0x4E58:
            raise ValueError(f"Bad magic: 0x{magic:04X}")
        proto_id, ts, plen = struct.unpack_from("!BII", data, 4)
        # body = proto_id(1) + ts(4) + plen(4) + payload(plen) = 9 + plen
        body = data[4:4 + 9 + plen]
        crc_computed = self.compute_checksum(body)
        if crc_received != crc_computed:
            raise ValueError(f"Checksum mismatch: recv=0x{crc_received:04X} calc=0x{crc_computed:04X}")
        payload = data[self.WIRE_HEADER_SIZE:self.WIRE_HEADER_SIZE + plen]
        proto = list(ProtocolType)[proto_id]
        if protocol is not None and protocol != proto:
            raise ValueError(f"Protocol mismatch: expected {protocol}, got {proto}")
        return MessageFrame(
            header={"proto_id": proto_id, "ts": ts, "msg_id": self._next_id()},
            payload=payload,
            checksum=crc_received,
            timestamp=float(ts),
            protocol_type=proto,
        )

    def fragment_message(self, frame: MessageFrame, max_size: Optional[int] = None) -> List[bytes]:
        """Split *frame* into fragments no larger than *max_size* bytes.

        Fragment format: [2B frag_crc][2B orig_crc][2B total][1B idx][2B plen][1B proto_id][payload]
        """
        if max_size is None:
            max_size = self.MTU.get(frame.protocol_type, 256)
        overhead = self.FRAG_HEADER_SIZE  # 10
        payload = frame.payload
        data_per_frag = max_size - overhead
        if data_per_frag <= 0:
            raise ValueError(f"max_size {max_size} too small for overhead {overhead}")
        n = max(1, (len(payload) + data_per_frag - 1) // data_per_frag)
        fragments: List[bytes] = []
        proto_id = list(ProtocolType).index(frame.protocol_type)
        for i in range(n):
            chunk = payload[i * data_per_frag:(i + 1) * data_per_frag]
            inner = struct.pack("!HHBHB", frame.checksum, n, i, len(chunk), proto_id) + chunk
            frag_crc = self.compute_checksum(inner)
            fragments.append(struct.pack("!H", frag_crc) + inner)
        return fragments

    def reassemble_fragments(self, fragments: List[bytes]) -> MessageFrame:
        """Reassemble a list of fragment bytes into a single :class:`MessageFrame`."""
        if not fragments:
            raise ValueError("No fragments provided")
        parts: dict = {}
        original_checksum = None
        proto = None
        total_expected = None
        for frag in fragments:
            if len(frag) < self.FRAG_HEADER_SIZE:
                raise ValueError("Fragment too short")
            frag_crc = struct.unpack_from("!H", frag, 0)[0]
            body = frag[2:]
            if self.compute_checksum(body) != frag_crc:
                raise ValueError("Fragment CRC mismatch")
            # Parse inner: orig_crc(H), total(H), idx(B), plen(H), proto_id(B) = 8 bytes header
            orig_crc, total, idx, plen, pid = struct.unpack_from("!HHBHB", body, 0)
            original_checksum = orig_crc
            total_expected = total
            proto = list(ProtocolType)[pid]
            payload_chunk = body[8:8 + plen]
            parts[idx] = payload_chunk
        if total_expected is not None and len(parts) != total_expected:
            raise ValueError(f"Missing fragments: got {len(parts)}/{total_expected}")
        full_payload = b"".join(parts[i] for i in sorted(parts))
        return MessageFrame(
            header={"reassembled": True, "fragments": total_expected or 0},
            payload=full_payload,
            checksum=original_checksum,
            timestamp=time.time(),
            protocol_type=proto,
        )

    def estimate_bandwidth(self, protocol: ProtocolType, conditions: Optional[dict] = None) -> int:
        """Return estimated bits/sec for *protocol* given *conditions*.

        *conditions* keys: ``signal_strength`` (0-1), ``interference`` (0-1),
        ``weather_factor`` (0-1, 1=bad), ``distance_km`` (float).
        """
        base = self.BASE_BANDWIDTH.get(protocol, 9_600)
        factor = 1.0
        if conditions:
            signal = conditions.get("signal_strength", 1.0)
            interference = conditions.get("interference", 0.0)
            weather = conditions.get("weather_factor", 0.0)
            distance = conditions.get("distance_km", 0.0)
            factor *= max(0.1, signal)
            factor *= max(0.1, 1.0 - interference)
            factor *= max(0.1, 1.0 - weather)
            if distance > 0 and protocol in (ProtocolType.LORA, ProtocolType.RADIO):
                factor *= max(0.05, 1.0 / (1.0 + distance / 10.0))
        return max(1, int(base * factor))
