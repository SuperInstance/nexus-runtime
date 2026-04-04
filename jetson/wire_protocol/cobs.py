"""NEXUS Wire Protocol - COBS (Consistent Overhead Byte Stuffing).

COBS encode/decode for zero-byte delimited framing.
Worst-case overhead: 1 byte per 254 bytes (0.4%).
"""

from __future__ import annotations


def cobs_encode(data: bytes) -> bytes:
    """COBS-encode a byte sequence.

    Args:
        data: Input bytes (may contain zeros).

    Returns:
        COBS-encoded bytes (no zeros except potential trailing delimiter).
        Returns b"\\x01" for empty input.
    """
    if not data:
        return b"\x01"

    result: list[int] = []
    code_idx = 0
    code = 1

    result.append(0)  # placeholder for first code byte
    code_idx = 0

    for byte in data:
        if byte == 0:
            result[code_idx] = code
            code = 1
            code_idx = len(result)
            result.append(0)  # placeholder
        else:
            result.append(byte)
            code += 1
            if code == 0xFF:
                result[code_idx] = code
                code = 1
                code_idx = len(result)
                result.append(0)  # placeholder

    result[code_idx] = code
    return bytes(result)


def cobs_decode(data: bytes) -> bytes:
    """COBS-decode a byte sequence.

    Args:
        data: COBS-encoded bytes.

    Returns:
        Decoded original bytes.
        Returns b"" for empty input.
    """
    if not data:
        return b""

    result: list[int] = []
    idx = 0

    while idx < len(data):
        code = data[idx]
        idx += 1
        for _ in range(1, code):
            if idx >= len(data):
                break
            result.append(data[idx])
            idx += 1
        if code < 0xFF and idx < len(data):
            result.append(0)

    return bytes(result)
