"""NEXUS Wire Protocol - CRC-16/CCITT-FALSE.

Polynomial: 0x1021
Initial:     0xFFFF
Final XOR:   0x0000
Check value: 0x29B1 for "123456789"
"""

from __future__ import annotations


def crc16_ccitt(data: bytes, initial: int = 0xFFFF) -> int:
    """Compute CRC-16/CCITT-FALSE.

    Args:
        data: Input bytes.
        initial: Initial CRC value (default 0xFFFF).

    Returns:
        16-bit CRC value.
    """
    crc = initial
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc
