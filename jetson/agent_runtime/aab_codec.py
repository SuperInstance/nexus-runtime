"""NEXUS Agent Runtime - Agent-Annotated Bytecode (AAB) codec.

AAB format: 8-byte core instruction + variable-length TLV metadata trailer.
TLV tags: INTENT, CAPABILITY, SAFETY, TRUST, NARRATIVE.
ESP32 receives only the stripped 8-byte core. Zero execution overhead.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TLVEntry:
    """Type-Length-Value metadata entry."""

    tag: int
    value: bytes

    def encode(self) -> bytes:
        """Encode TLV entry to bytes."""
        return bytes([self.tag]) + len(self.value).to_bytes(2, "little") + self.value

    @classmethod
    def decode(cls, data: bytes, offset: int) -> tuple[TLVEntry, int]:
        """Decode a TLV entry from bytes.

        Returns:
            Tuple of (TLVEntry, next_offset).
        """
        tag = data[offset]
        length = int.from_bytes(data[offset + 1:offset + 3], "little")
        value = data[offset + 3:offset + 3 + length]
        return cls(tag=tag, value=value), offset + 3 + length


# TLV tag constants
TLV_INTENT = 0x01
TLV_CAPABILITY = 0x02
TLV_SAFETY = 0x03
TLV_TRUST = 0x04
TLV_NARRATIVE = 0x05


class AABCodec:
    """Agent-Annotated Bytecode encoder/decoder (stub)."""

    def encode(self, bytecode: bytes, metadata: list[TLVEntry]) -> bytes:
        """Encode bytecode with AAB metadata.

        Args:
            bytecode: Raw 8-byte-aligned bytecode.
            metadata: List of TLV metadata entries.

        Returns:
            AAB-encoded bytes.
        """
        # TODO: Implement AAB encoding
        return bytecode

    def decode(self, aab_data: bytes) -> tuple[bytes, list[TLVEntry]]:
        """Decode AAB data into core bytecode and metadata.

        Args:
            aab_data: AAB-encoded bytes.

        Returns:
            Tuple of (core_bytecode, metadata_entries).
        """
        # TODO: Implement AAB decoding
        return aab_data, []

    def strip(self, aab_data: bytes) -> bytes:
        """Strip all TLV metadata, returning only core bytecode.

        Args:
            aab_data: AAB-encoded bytes.

        Returns:
            Core 8-byte bytecode only.
        """
        core, _ = self.decode(aab_data)
        return core
