"""Data compression: RLE, delta, dictionary, and auto-select."""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import time
import hashlib


@dataclass
class CompressionResult:
    """Result of a compression operation."""
    original_size: int
    compressed_size: int
    ratio: float
    algorithm: str
    time_ms: float


class Compressor:
    """Collection of pure-Python compression algorithms."""

    # ------------------------------------------------------------------
    # Run-Length Encoding
    # ------------------------------------------------------------------
    @staticmethod
    def compress_rle(data: bytes) -> bytes:
        """Compress *data* using run-length encoding.

        Format: for each run of identical bytes — [1B value][2B count].
        Non-repeating bytes are stored as [value][count=1].
        """
        if not data:
            return b""
        result = bytearray()
        i = 0
        while i < len(data):
            val = data[i]
            count = 1
            while i + count < len(data) and data[i + count] == val and count < 65535:
                count += 1
            result.append(val)
            result.extend(count.to_bytes(2, "big"))
            i += count
        return bytes(result)

    @staticmethod
    def decompress_rle(data: bytes) -> bytes:
        """Decompress RLE data back to original bytes."""
        if not data:
            return b""
        result = bytearray()
        i = 0
        while i < len(data):
            val = data[i]
            count = int.from_bytes(data[i + 1:i + 3], "big")
            result.extend([val] * count)
            i += 3
        return bytes(result)

    # ------------------------------------------------------------------
    # Delta Encoding
    # ------------------------------------------------------------------
    @staticmethod
    def compress_delta(data: bytes, reference: bytes) -> bytes:
        """Produce a delta patch from *reference* to *data*.

        Format: [4B total_len][list of (4B offset, 4B length, <patch_bytes>)].
        """
        if not data and not reference:
            return b""
        total_len = len(data)
        # Simple byte-level delta: store offset, length of changed region, and new bytes
        patches: List[Tuple[int, int, bytes]] = []
        i = 0
        while i < len(data):
            ref_byte = reference[i] if i < len(reference) else 0
            if data[i] != ref_byte:
                start = i
                while i < len(data) and (i >= len(reference) or data[i] != reference[i]):
                    i += 1
                patches.append((start, i - start, data[start:i]))
            else:
                i += 1
        result = bytearray()
        result.extend(total_len.to_bytes(4, "big"))
        result.extend(len(patches).to_bytes(4, "big"))
        for offset, length, patch in patches:
            result.extend(offset.to_bytes(4, "big"))
            result.extend(length.to_bytes(4, "big"))
            result.extend(patch)
        return bytes(result)

    @staticmethod
    def decompress_delta(data: bytes, reference: bytes) -> bytes:
        """Apply a delta patch to *reference* to reconstruct original data."""
        if not data:
            return reference if reference else b""
        total_len = int.from_bytes(data[0:4], "big")
        num_patches = int.from_bytes(data[4:8], "big")
        output = bytearray(reference[:total_len])
        # Pad or truncate to total_len
        if len(output) < total_len:
            output.extend(b"\x00" * (total_len - len(output)))
        elif len(output) > total_len:
            output = output[:total_len]
        offset = 8
        for _ in range(num_patches):
            po = int.from_bytes(data[offset:offset + 4], "big")
            pl = int.from_bytes(data[offset + 4:offset + 8], "big")
            patch = data[offset + 8:offset + 8 + pl]
            output[po:po + pl] = patch
            offset += 8 + pl
        return bytes(output)

    # ------------------------------------------------------------------
    # Dictionary Compression
    # ------------------------------------------------------------------
    @staticmethod
    def compress_dictionary(data: bytes, dictionary: dict) -> bytes:
        """Compress *data* using a token dictionary.

        *dictionary* maps ``bytes -> int`` (token id). Tokens replace matching
        byte-sequences. Non-matching bytes pass through with a 0xFF escape.

        Format: [1B: token or 0xFF literal][2B length if literal | 2B token_id if token]
        """
        result = bytearray()
        i = 0
        # Sort dictionary keys by length descending for greedy longest-match
        sorted_entries = sorted(dictionary.items(), key=lambda x: len(x[0]), reverse=True)
        while i < len(data):
            matched = False
            for pattern, token in sorted_entries:
                if not isinstance(pattern, bytes):
                    pattern = bytes(pattern, "utf-8") if isinstance(pattern, str) else bytes(pattern)
                if data[i:i + len(pattern)] == pattern:
                    # Token marker 0x01 + 2B token id
                    result.append(0x01)
                    result.extend(int(token).to_bytes(2, "big"))
                    i += len(pattern)
                    matched = True
                    break
            if not matched:
                # Literal: 0xFF + 1B byte
                result.append(0xFF)
                result.append(data[i])
                i += 1
        return bytes(result)

    @staticmethod
    def decompress_dictionary(data: bytes, dictionary: dict) -> bytes:
        """Decompress dictionary-compressed data.

        *dictionary* maps ``int (token_id) -> bytes``.
        """
        reverse = {}
        for pattern, token in dictionary.items():
            if isinstance(token, int):
                reverse[token] = pattern
            elif isinstance(token, str):
                reverse[int(token)] = pattern
        result = bytearray()
        i = 0
        while i < len(data):
            marker = data[i]
            if marker == 0x01:
                token_id = int.from_bytes(data[i + 1:i + 3], "big")
                replacement = reverse.get(token_id, b"")
                if isinstance(replacement, str):
                    replacement = bytes(replacement, "utf-8")
                result.extend(replacement)
                i += 3
            elif marker == 0xFF:
                result.append(data[i + 1])
                i += 2
            else:
                # Fallback: treat as literal
                result.append(marker)
                i += 1
        return bytes(result)

    # ------------------------------------------------------------------
    # Auto-compress: pick best method
    # ------------------------------------------------------------------
    def auto_compress(self, data: bytes, threshold: float = 0.9) -> CompressionResult:
        """Try RLE and delta, pick the best result.

        *threshold* — only accept if compressed_size / original_size < threshold.
        Returns uncompressed data if nothing helps.
        """
        if not data:
            return CompressionResult(0, 0, 1.0, "none", 0.0)

        original_size = len(data)
        best_result: Optional[CompressionResult] = None

        # Try RLE
        t0 = time.monotonic()
        rle_compressed = self.compress_rle(data)
        t_rle = (time.monotonic() - t0) * 1000
        rle_ratio = len(rle_compressed) / original_size if original_size > 0 else 1.0
        rle_result = CompressionResult(original_size, len(rle_compressed), rle_ratio, "rle", t_rle)
        if rle_ratio < threshold:
            best_result = rle_result

        # Try delta (with empty reference)
        t0 = time.monotonic()
        delta_compressed = self.compress_delta(data, b"\x00" * original_size)
        t_delta = (time.monotonic() - t0) * 1000
        delta_ratio = len(delta_compressed) / original_size if original_size > 0 else 1.0
        delta_result = CompressionResult(original_size, len(delta_compressed), delta_ratio, "delta", t_delta)
        if delta_ratio < threshold:
            if best_result is None or delta_ratio < best_result.ratio:
                best_result = delta_result

        if best_result is None:
            return CompressionResult(original_size, original_size, 1.0, "none", 0.0)
        return best_result

    # ------------------------------------------------------------------
    # Dictionary builder
    # ------------------------------------------------------------------
    @staticmethod
    def build_dictionary(samples: List[bytes]) -> dict:
        """Build a byte-pattern dictionary from sample data.

        Uses frequency analysis of byte bigrams and trigrams.
        Returns ``{bytes_pattern: token_id}``.
        """
        freq: Dict[bytes, int] = {}
        for sample in samples:
            for n in (2, 3, 4):
                for i in range(len(sample) - n + 1):
                    ngram = sample[i:i + n]
                    freq[ngram] = freq.get(ngram, 0) + 1
        # Keep top patterns (by frequency * length for best compression)
        scored = sorted(freq.items(), key=lambda x: len(x[0]) * x[1], reverse=True)
        # Keep at most 65535 entries (fits in 2B token)
        top = scored[:65535]
        return {pattern: idx for idx, (pattern, _) in enumerate(top)}
