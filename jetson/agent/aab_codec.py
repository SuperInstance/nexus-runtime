"""NEXUS Agent Runtime - Agent-Annotated Bytecode (AAB) codec.

AAB format: 8-byte core instruction + variable-length TLV metadata trailer.
Each AAB instruction = [Core: 8 bytes] [TLV Block: variable]
TLV format: [Tag:1][Length:2][Value:N] ... [Tag:0x00 = end]
ESP32 receives only the stripped 8-byte core. Zero execution overhead.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

# ===================================================================
# TLV Tag Constants (13 tags)
# ===================================================================

TLV_TYPE_DESC = 0x01
TLV_CAP_REQ = 0x02
TLV_SAFETY_CONSTRAINT = 0x03
TLV_TRUST_REQ = 0x04
TLV_NARRATIVE = 0x05
TLV_AGENT_SOURCE = 0x06
TLV_INTENT_TAG = 0x07
TLV_PRECONDITION = 0x08
TLV_POSTCONDITION = 0x09
TLV_PERFORMANCE_HINT = 0x0A
TLV_DOMAIN_TAG = 0x0B
TLV_VESSEL_CAP = 0x0C
TLV_TRUST_MULTIPLIER = 0x0D
TLV_END = 0x00

TLV_TAG_NAMES: dict[int, str] = {
    0x00: "END",
    0x01: "TYPE_DESC",
    0x02: "CAP_REQ",
    0x03: "SAFETY_CONSTRAINT",
    0x04: "TRUST_REQ",
    0x05: "NARRATIVE",
    0x06: "AGENT_SOURCE",
    0x07: "INTENT_TAG",
    0x08: "PRECONDITION",
    0x09: "POSTCONDITION",
    0x0A: "PERFORMANCE_HINT",
    0x0B: "DOMAIN_TAG",
    0x0C: "VESSEL_CAP",
    0x0D: "TRUST_MULTIPLIER",
}


# ===================================================================
# TLV Entry
# ===================================================================

@dataclass
class TLVEntry:
    """Type-Length-Value metadata entry."""

    tag: int
    value: bytes

    def encode(self) -> bytes:
        """Encode TLV entry to bytes: [Tag:1][Length:2][Value:N]."""
        return bytes([self.tag]) + len(self.value).to_bytes(2, "little") + self.value

    @classmethod
    def decode(cls, data: bytes, offset: int) -> tuple[TLVEntry, int]:
        """Decode a TLV entry from bytes.

        Returns:
            Tuple of (TLVEntry, next_offset).
        """
        tag = data[offset]
        if tag == TLV_END:
            return cls(tag=TLV_END, value=b""), offset + 1
        length = int.from_bytes(data[offset + 1:offset + 3], "little")
        value = data[offset + 3:offset + 3 + length]
        return cls(tag=tag, value=value), offset + 3 + length

    def __str__(self) -> str:
        name = TLV_TAG_NAMES.get(self.tag, f"UNKNOWN(0x{self.tag:02X})")
        try:
            val_str = self.value.decode("utf-8")
        except UnicodeDecodeError:
            val_str = self.value.hex()
        return f"TLV({name}, {val_str!r})"


# ===================================================================
# AAB Instruction
# ===================================================================

@dataclass
class AABInstruction:
    """A single AAB instruction: 8-byte core + variable TLV metadata."""

    core: bytes          # Exactly 8 bytes
    metadata: list[TLVEntry] = field(default_factory=list)


# ===================================================================
# AAB File Format Constants
# ===================================================================

AAB_MAGIC = b"NXAB"
AAB_VERSION = 0x0001
AAB_HEADER_SIZE = 8  # 4 magic + 2 version + 2 instruction count


# ===================================================================
# AAB Codec Functions
# ===================================================================

def aab_encode(instructions: list[AABInstruction]) -> bytes:
    """Encode a list of AAB instructions into bytes.

    File format:
        Header: NXAB (4) + version (2) + count (2) = 8 bytes
        Body: N instructions, each = 8-byte core + variable TLV block
    """
    # Build header
    header = AAB_MAGIC + AAB_VERSION.to_bytes(2, "little") + len(instructions).to_bytes(2, "little")

    # Build body
    body = b""
    for instr in instructions:
        if len(instr.core) != 8:
            raise ValueError(f"Core instruction must be exactly 8 bytes, got {len(instr.core)}")
        body += instr.core
        for tlv in instr.metadata:
            body += tlv.encode()
        body += bytes([TLV_END])  # End of TLV block for this instruction

    return header + body


def aab_decode(data: bytes) -> list[AABInstruction]:
    """Decode AAB bytes into a list of AAB instructions.

    Returns:
        List of AABInstruction objects.
    """
    if len(data) < AAB_HEADER_SIZE:
        raise ValueError(f"Data too short for AAB header: {len(data)} bytes")

    # Parse header
    magic = data[0:4]
    if magic != AAB_MAGIC:
        raise ValueError(f"Invalid AAB magic: {magic!r}, expected {AAB_MAGIC!r}")

    version = int.from_bytes(data[4:6], "little")
    if version != AAB_VERSION:
        raise ValueError(f"Unsupported AAB version: {version}, expected {AAB_VERSION}")

    count = int.from_bytes(data[6:8], "little")

    instructions: list[AABInstruction] = []
    offset = AAB_HEADER_SIZE

    for _ in range(count):
        if offset + 8 > len(data):
            raise ValueError(f"Unexpected end of data at offset {offset}, expected 8 bytes for core")

        core = data[offset:offset + 8]
        offset += 8

        metadata: list[TLVEntry] = []
        while offset < len(data):
            tlv, offset = TLVEntry.decode(data, offset)
            if tlv.tag == TLV_END:
                break
            metadata.append(tlv)

        instructions.append(AABInstruction(core=core, metadata=metadata))

    return instructions


def aab_strip(data: bytes) -> bytes:
    """Strip all TLV metadata, returning only raw 8-byte core instructions.

    Args:
        data: AAB-encoded bytes (with header).

    Returns:
        Raw 8-byte core instructions concatenated.
    """
    instructions = aab_decode(data)
    return b"".join(instr.core for instr in instructions)


# ===================================================================
# Convenience helpers
# ===================================================================

def make_tlv(tag: int, value: str | bytes | float | int) -> TLVEntry:
    """Create a TLVEntry with automatic encoding."""
    if isinstance(value, str):
        return TLVEntry(tag=tag, value=value.encode("utf-8"))
    elif isinstance(value, float):
        return TLVEntry(tag=tag, value=struct.pack("<f", value))
    elif isinstance(value, int):
        return TLVEntry(tag=tag, value=value.to_bytes(4, "little", signed=True))
    elif isinstance(value, bytes):
        return TLVEntry(tag=tag, value=value)
    else:
        raise TypeError(f"Unsupported value type: {type(value)}")


def struct_pack_f32(value: float) -> bytes:
    """Pack a float32 value to bytes (little-endian)."""
    return struct.pack("<f", value)


def struct_unpack_f32(data: bytes) -> float:
    """Unpack a float32 value from bytes (little-endian)."""
    return struct.unpack("<f", data)[0]


def get_tlv_string(metadata: list[TLVEntry], tag: int) -> str | None:
    """Get a string value from metadata by tag."""
    for entry in metadata:
        if entry.tag == tag:
            try:
                return entry.value.decode("utf-8")
            except UnicodeDecodeError:
                return None
    return None


def get_tlv_float(metadata: list[TLVEntry], tag: int) -> float | None:
    """Get a float value from metadata by tag."""
    for entry in metadata:
        if entry.tag == tag and len(entry.value) == 4:
            return struct_unpack_f32(entry.value)
    return None


# Keep legacy AABCodec class for backward compatibility
class AABCodec:
    """Agent-Annotated Bytecode encoder/decoder (wraps module functions)."""

    def encode(self, bytecode: bytes, metadata: list[TLVEntry]) -> bytes:
        """Encode bytecode with AAB metadata (legacy single-instruction API)."""
        instructions = [AABInstruction(core=bytecode[:8], metadata=metadata)]
        return aab_encode(instructions)

    def decode(self, aab_data: bytes) -> tuple[bytes, list[TLVEntry]]:
        """Decode AAB data into core bytecode and metadata (legacy API)."""
        instructions = aab_decode(aab_data)
        if len(instructions) == 0:
            return b"", []
        instr = instructions[0]
        return instr.core, instr.metadata

    def strip(self, aab_data: bytes) -> bytes:
        """Strip all TLV metadata, returning only core bytecode."""
        return aab_strip(aab_data)
