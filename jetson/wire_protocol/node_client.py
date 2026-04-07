"""NEXUS Wire Protocol - High-level node client.

Provides a high-level API for communicating with ESP32 nodes
over the NEXUS wire protocol via serial (RS-422).
"""

from __future__ import annotations

import logging
import struct
import time
from enum import IntEnum
from typing import Optional, Callable

logger = logging.getLogger(__name__)

from .frame import FrameParser, parse_frame_header, FRAME_MAX_PAYLOAD
from .cobs import cobs_encode, cobs_decode
from .crc16 import crc16_ccitt


class MessageType(IntEnum):
    """NEXUS wire protocol message types."""
    DEVICE_IDENTITY = 0x01
    ROLE_ASSIGN = 0x02
    ROLE_ACK = 0x03
    AUTO_DETECT_RESULT = 0x04
    HEARTBEAT = 0x05
    SENSOR_TELEMETRY = 0x06
    COMMAND = 0x07
    COMMAND_ACK = 0x08
    REFLEX_DEPLOY = 0x09
    REFLEX_STATUS = 0x0A
    FIRMWARE_QUERY = 0x0B
    OBS_RECORD_START = 0x0C
    OBS_RECORD_STOP = 0x0D
    OBS_DATA_CHUNK = 0x0E
    OBS_DATA_ACK = 0x0F
    SELFTEST_RESULT = 0x10
    ERROR_REPORT = 0x11
    BAUD_NEGOTIATE = 0x12
    BAUD_NEGOTIATE_ACK = 0x13
    PING = 0x14
    PONG = 0x15
    SAFETY_EVENT = 0x1C
    DEBUG_LOG = 0x20
    DEBUG_CMD = 0x21
    PARAM_GET = 0x22
    PARAM_SET = 0x23
    PARAM_RESPONSE = 0x24
    FIRMWARE_CHUNK = 0x42
    FIRMWARE_VERIFY = 0x43


class MessageFlag(IntEnum):
    """Message header flag bits."""
    ACK_REQUIRED = (1 << 0)
    IS_ACK = (1 << 1)
    IS_ERROR = (1 << 2)
    URGENT = (1 << 3)
    COMPRESSED = (1 << 4)
    ENCRYPTED = (1 << 5)
    NO_TIMESTAMP = (1 << 6)


class Criticality(IntEnum):
    """Message criticality levels."""
    TELEMETRY = 0
    COMMAND = 1
    SAFETY = 2


class Direction(IntEnum):
    """Message direction."""
    N2J = 0  # Node to Jetson
    J2N = 1  # Jetson to Node
    BOTH = 2  # Bidirectional


# Message type metadata: (direction, criticality)
MSG_TYPE_INFO: dict[int, tuple[int, int]] = {
    0x01: (Direction.N2J, Criticality.TELEMETRY),
    0x02: (Direction.J2N, Criticality.COMMAND),
    0x03: (Direction.N2J, Criticality.COMMAND),
    0x04: (Direction.N2J, Criticality.TELEMETRY),
    0x05: (Direction.BOTH, Criticality.TELEMETRY),
    0x06: (Direction.N2J, Criticality.TELEMETRY),
    0x07: (Direction.J2N, Criticality.COMMAND),
    0x08: (Direction.N2J, Criticality.COMMAND),
    0x09: (Direction.J2N, Criticality.COMMAND),
    0x0A: (Direction.N2J, Criticality.TELEMETRY),
    0x0B: (Direction.J2N, Criticality.COMMAND),
    0x0C: (Direction.J2N, Criticality.COMMAND),
    0x0D: (Direction.J2N, Criticality.COMMAND),
    0x0E: (Direction.N2J, Criticality.TELEMETRY),
    0x0F: (Direction.N2J, Criticality.TELEMETRY),
    0x10: (Direction.N2J, Criticality.TELEMETRY),
    0x11: (Direction.N2J, Criticality.SAFETY),
    0x12: (Direction.J2N, Criticality.COMMAND),
    0x13: (Direction.N2J, Criticality.COMMAND),
    0x14: (Direction.BOTH, Criticality.TELEMETRY),
    0x15: (Direction.BOTH, Criticality.TELEMETRY),
    0x1C: (Direction.N2J, Criticality.SAFETY),
    0x20: (Direction.N2J, Criticality.TELEMETRY),
    0x21: (Direction.J2N, Criticality.COMMAND),
    0x22: (Direction.J2N, Criticality.COMMAND),
    0x23: (Direction.J2N, Criticality.COMMAND),
    0x24: (Direction.N2J, Criticality.TELEMETRY),
    0x42: (Direction.J2N, Criticality.COMMAND),
    0x43: (Direction.J2N, Criticality.COMMAND),
}

MSG_TYPE_NAMES: dict[int, str] = {
    mt.value: mt.name for mt in MessageType
}


class NodeClient:
    """High-level client for communicating with NEXUS ESP32 nodes.

    Supports both real serial connections (with pyserial) and
    in-memory transport for testing.
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baud_rate: int = 921600) -> None:
        self.port = port
        self.baud_rate = baud_rate
        self._sequence = 0
        self._connected = False
        self._serial = None
        self._parser = FrameParser()
        self._transport: Optional[Callable[[bytes], None]] = None
        self._response_frames: list[bytes] = []

    def set_transport(self, transport: Callable[[bytes], None]) -> None:
        """Set an alternative transport function (for testing).

        Args:
            transport: Function that accepts bytes to send.
        """
        self._transport = transport

    def inject_response(self, frame_data: bytes) -> None:
        """Inject a raw response frame (for testing).

        Args:
            frame_data: Complete wire frame bytes.
        """
        frames = self._parser.feed(frame_data)
        self._response_frames.extend(frames)

    def connect(self) -> bool:
        """Open serial connection to the node.

        Returns:
            True if connection opened, False on error.
        """
        try:
            import serial
            self._serial = serial.Serial(self.port, self.baud_rate, timeout=1.0)
            self._connected = True
            return True
        except (ImportError, Exception):
            return False

    def disconnect(self) -> None:
        """Close serial connection."""
        if self._serial:
            try:
                self._serial.close()
            except Exception as e:
                logger.warning("Error closing serial connection: %s", e)
            self._serial = None
        self._connected = False

    def _next_seq(self) -> int:
        self._sequence = (self._sequence + 1) & 0xFFFF
        return self._sequence

    def _send_raw(self, wire_frame: bytes) -> bool:
        """Send raw wire frame bytes."""
        if self._transport:
            self._transport(wire_frame)
            return True
        if self._serial and self._connected:
            try:
                self._serial.write(wire_frame)
                return True
            except Exception:
                return False
        return False

    def _send(self, msg_type: int, payload: bytes = b"",
              flags: int = 0, ack_required: bool = False) -> bool:
        """Send a message with auto-incremented sequence number.

        Args:
            msg_type: Message type byte.
            payload: Payload data.
            flags: Additional flags.
            ack_required: Set ACK_REQUIRED flag.

        Returns:
            True if sent successfully.
        """
        if len(payload) > FRAME_MAX_PAYLOAD:
            return False
        if ack_required:
            flags |= MessageFlag.ACK_REQUIRED
        seq = self._next_seq()
        wire_frame = FrameParser.encode_frame(payload, msg_type, flags, seq)
        return self._send_raw(wire_frame)

    def send_heartbeat(self) -> bool:
        """Send a heartbeat message (type 0x05, no payload)."""
        return self._send(MessageType.HEARTBEAT)

    def send_ping(self) -> bool:
        """Send a ping message (type 0x14)."""
        return self._send(MessageType.PING)

    def send_reflex_deploy(self, reflex_name: str, bytecode: bytes) -> bool:
        """Deploy bytecode to the node (type 0x09).

        Args:
            reflex_name: Name of the reflex (null-terminated string in payload).
            bytecode: Compiled bytecode binary.

        Returns:
            True if sent successfully.
        """
        # Payload: name_len(1) + name + bytecode
        name_bytes = reflex_name.encode("utf-8")[:255]
        payload = bytes([len(name_bytes)]) + name_bytes + bytecode
        if len(payload) > FRAME_MAX_PAYLOAD:
            return False
        return self._send(MessageType.REFLEX_DEPLOY, payload, ack_required=True)

    def send_command(self, command_data: bytes) -> bool:
        """Send a command message (type 0x07).

        Args:
            command_data: Raw command payload.

        Returns:
            True if sent successfully.
        """
        return self._send(MessageType.COMMAND, command_data, ack_required=True)

    def send_obs_record_start(self, config: bytes = b"") -> bool:
        """Start observation recording (type 0x0C)."""
        return self._send(MessageType.OBS_RECORD_START, config, ack_required=True)

    def send_obs_record_stop(self) -> bool:
        """Stop observation recording (type 0x0D)."""
        return self._send(MessageType.OBS_RECORD_STOP, ack_required=True)

    def send_obs_data_ack(self, chunk_id: int) -> bool:
        """Acknowledge an observation data chunk (type 0x0F)."""
        payload = struct.pack(">I", chunk_id)
        return self._send(MessageType.OBS_DATA_ACK, payload)

    def send_role_assign(self, role_id: int, capabilities: bytes = b"") -> bool:
        """Assign a role to the node (type 0x02).

        Args:
            role_id: Role identifier.
            capabilities: Optional capabilities blob.

        Returns:
            True if sent successfully.
        """
        payload = struct.pack(">B", role_id) + capabilities
        return self._send(MessageType.ROLE_ASSIGN, payload, ack_required=True)

    def send_baud_negotiate(self, requested_baud: int) -> bool:
        """Negotiate baud rate (type 0x12)."""
        payload = struct.pack(">I", requested_baud)
        return self._send(MessageType.BAUD_NEGOTIATE, payload, ack_required=True)

    def send_param_set(self, param_id: int, value: bytes) -> bool:
        """Set a parameter (type 0x23)."""
        payload = struct.pack(">H", param_id) + value
        return self._send(MessageType.PARAM_SET, payload, ack_required=True)

    def send_param_get(self, param_id: int) -> bool:
        """Get a parameter (type 0x22)."""
        payload = struct.pack(">H", param_id)
        return self._send(MessageType.PARAM_GET, payload)

    def send_firmware_query(self) -> bool:
        """Query current firmware version (type 0x0B)."""
        return self._send(MessageType.FIRMWARE_QUERY, ack_required=True)

    def send_firmware_chunk(self, offset: int, data: bytes) -> bool:
        """Send a firmware update chunk (type 0x42)."""
        payload = struct.pack(">I", offset) + data
        if len(payload) > FRAME_MAX_PAYLOAD:
            return False
        return self._send(MessageType.FIRMWARE_CHUNK, payload, ack_required=True)

    def send_firmware_verify(self, expected_crc: int) -> bool:
        """Verify firmware integrity (type 0x43)."""
        payload = struct.pack(">H", expected_crc)
        return self._send(MessageType.FIRMWARE_VERIFY, payload, ack_required=True)

    def send_debug_cmd(self, cmd: bytes) -> bool:
        """Send a debug command (type 0x21)."""
        return self._send(MessageType.DEBUG_CMD, cmd)

    def poll(self) -> list[tuple]:
        """Poll for received messages.

        Returns:
            List of parsed message tuples:
            (msg_type, flags, seq, timestamp_ms, payload_length, payload)
        """
        if self._response_frames:
            results = []
            for frame in self._response_frames:
                results.append(parse_frame_header(frame))
            self._response_frames.clear()
            return results

        if self._serial and self._connected:
            try:
                data = self._serial.read(self._serial.in_waiting or 1)
                if data:
                    frames = self._parser.feed(data)
                    return [parse_frame_header(f) for f in frames]
            except Exception as e:
                logger.warning("Error reading from serial: %s", e)
        return []

    @property
    def is_connected(self) -> bool:
        """Check if connected to a node."""
        return self._connected

    @property
    def sequence(self) -> int:
        """Current sequence number."""
        return self._sequence
