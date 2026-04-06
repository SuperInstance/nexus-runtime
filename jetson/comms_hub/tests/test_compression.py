"""Tests for jetson.comms_hub.compression."""

import pytest
from jetson.comms_hub.compression import CompressionResult, Compressor


@pytest.fixture
def compressor():
    return Compressor()


class TestCompressionResult:
    def test_construction(self):
        r = CompressionResult(original_size=100, compressed_size=50, ratio=0.5, algorithm="rle", time_ms=1.0)
        assert r.original_size == 100
        assert r.compressed_size == 50
        assert r.ratio == 0.5
        assert r.algorithm == "rle"
        assert r.time_ms == 1.0


class TestRLE:
    def test_compress_empty(self, compressor):
        result = compressor.compress_rle(b"")
        assert result == b""

    def test_decompress_empty(self, compressor):
        assert compressor.decompress_rle(b"") == b""

    def test_roundtrip_simple(self, compressor):
        original = b"AAAA"
        compressed = compressor.compress_rle(original)
        decompressed = compressor.decompress_rle(compressed)
        assert decompressed == original

    def test_roundtrip_mixed(self, compressor):
        original = b"AABBCCCCDD"
        compressed = compressor.compress_rle(original)
        decompressed = compressor.decompress_rle(compressed)
        assert decompressed == original

    def test_roundtrip_single_bytes(self, compressor):
        original = b"ABCD"
        compressed = compressor.compress_rle(original)
        decompressed = compressor.decompress_rle(compressed)
        assert decompressed == original

    def test_all_same(self, compressor):
        original = b"\xFF" * 1000
        compressed = compressor.compress_rle(original)
        decompressed = compressor.decompress_rle(compressed)
        assert decompressed == original

    def test_all_same_compresses_well(self, compressor):
        original = b"\x00" * 10000
        compressed = compressor.compress_rle(original)
        assert len(compressed) < len(original)

    def test_no_compression_for_unique(self, compressor):
        original = bytes(range(256))
        compressed = compressor.compress_rle(original)
        # Each byte is unique: 3 bytes per byte = 768, original 256, so RLE is bigger
        assert len(compressed) > len(original)

    def test_roundtrip_binary(self, compressor):
        original = b"\x00\x00\x01\x01\x01\x02\x03\x03\x03\x03"
        compressed = compressor.compress_rle(original)
        decompressed = compressor.decompress_rle(compressed)
        assert decompressed == original

    def test_max_run_length(self, compressor):
        original = b"\xAB" * 65535
        compressed = compressor.compress_rle(original)
        decompressed = compressor.decompress_rle(compressed)
        assert decompressed == original


class TestDelta:
    def test_compress_empty(self, compressor):
        result = compressor.compress_delta(b"", b"")
        assert result == b""

    def test_no_changes(self, compressor):
        data = b"12345"
        compressed = compressor.compress_delta(data, data)
        # Header: 4B total_len + 4B num_patches(0) = 8 bytes
        assert len(compressed) == 8

    def test_roundtrip_with_changes(self, compressor):
        reference = b"HELLO WORLD"
        data = b"HALLO WORLD"
        compressed = compressor.compress_delta(data, reference)
        decompressed = compressor.decompress_delta(compressed, reference)
        assert decompressed == data

    def test_completely_different(self, compressor):
        reference = b"AAAA"
        data = b"BBBB"
        compressed = compressor.compress_delta(data, reference)
        decompressed = compressor.decompress_delta(compressed, reference)
        assert decompressed == data

    def test_data_longer_than_ref(self, compressor):
        reference = b"ABC"
        data = b"ABCDEF"
        compressed = compressor.compress_delta(data, reference)
        decompressed = compressor.decompress_delta(compressed, reference)
        assert decompressed == data

    def test_data_shorter_than_ref(self, compressor):
        reference = b"ABCDEF"
        data = b"ABC"
        compressed = compressor.compress_delta(data, reference)
        decompressed = compressor.decompress_delta(compressed, reference)
        assert decompressed == data

    def test_empty_reference(self, compressor):
        data = b"NEW"
        compressed = compressor.compress_delta(data, b"")
        decompressed = compressor.decompress_delta(compressed, b"")
        assert decompressed == data

    def test_empty_data_with_ref(self, compressor):
        compressed = compressor.compress_delta(b"", b"REF")
        decompressed = compressor.decompress_delta(compressed, b"REF")
        assert decompressed == b""

    def test_large_data(self, compressor):
        reference = b"\x00" * 1000
        data = b"\xFF" * 1000
        compressed = compressor.compress_delta(data, reference)
        decompressed = compressor.decompress_delta(compressed, reference)
        assert decompressed == data

    def test_sparsely_changed(self, compressor):
        reference = b"A" * 100
        data = b"A" * 50 + b"B" * 1 + b"A" * 49
        compressed = compressor.compress_delta(data, reference)
        decompressed = compressor.decompress_delta(compressed, reference)
        assert decompressed == data

    def test_single_byte_diff(self, compressor):
        ref = b"ABCDEFGHIJ"
        data = b"ABCXEFGHIJ"
        compressed = compressor.compress_delta(data, ref)
        decompressed = compressor.decompress_delta(compressed, ref)
        assert decompressed == data


class TestDictionary:
    def test_compress_with_dict(self, compressor):
        data = b"HELLO HELLO WORLD"
        dictionary = {b"HELLO": 0, b"WORLD": 1}
        compressed = compressor.compress_dictionary(data, dictionary)
        assert len(compressed) > 0

    def test_roundtrip_dict(self, compressor):
        data = b"HELLO WORLD"
        dictionary = {b"HELLO": 0, b"WORLD": 1}
        compressed = compressor.compress_dictionary(data, dictionary)
        decompressed = compressor.decompress_dictionary(compressed, dictionary)
        assert decompressed == data

    def test_unknown_bytes_pass_through(self, compressor):
        data = b"XYZ"
        dictionary = {b"HELLO": 0}
        compressed = compressor.compress_dictionary(data, dictionary)
        decompressed = compressor.decompress_dictionary(compressed, dictionary)
        assert decompressed == data

    def test_empty_data(self, compressor):
        compressed = compressor.compress_dictionary(b"", {b"X": 0})
        assert compressed == b""

    def test_empty_dictionary(self, compressor):
        data = b"ABC"
        compressed = compressor.compress_dictionary(data, {})
        decompressed = compressor.decompress_dictionary(compressed, {})
        assert decompressed == data

    def test_overlapping_patterns(self, compressor):
        data = b"ABABAB"
        dictionary = {b"AB": 0, b"ABAB": 1}
        compressed = compressor.compress_dictionary(data, dictionary)
        decompressed = compressor.decompress_dictionary(compressed, dictionary)
        assert decompressed == data

    def test_string_keys(self, compressor):
        data = b"hello hello"
        dictionary = {"hello": 0}
        compressed = compressor.compress_dictionary(data, dictionary)
        decompressed = compressor.decompress_dictionary(compressed, dictionary)
        assert decompressed == data

    def test_longest_match(self, compressor):
        data = b"ABCDE"
        # "ABC" should be preferred over "AB" when both exist
        dictionary = {b"AB": 0, b"ABC": 1, b"DE": 2}
        compressed = compressor.compress_dictionary(data, dictionary)
        decompressed = compressor.decompress_dictionary(compressed, dictionary)
        assert decompressed == data


class TestAutoCompress:
    def test_empty_data(self, compressor):
        result = compressor.auto_compress(b"")
        assert result.original_size == 0
        assert result.compressed_size == 0
        assert result.algorithm == "none"

    def test_rle_wins(self, compressor):
        data = b"\x00" * 10000
        result = compressor.auto_compress(data)
        assert result.compressed_size < result.original_size
        assert result.algorithm == "rle"

    def test_no_improvement(self, compressor):
        # Unique bytes — neither RLE nor delta (against zero-ref) will help much
        data = bytes(range(256))
        result = compressor.auto_compress(data)
        # With zero reference, delta might not help either
        assert result.algorithm in ("none", "rle", "delta")

    def test_threshold_param(self, compressor):
        data = b"\xFF" * 100
        result = compressor.auto_compress(data, threshold=0.99)
        assert result.compressed_size <= data.__len__() * 0.99 + 1

    def test_result_has_timing(self, compressor):
        data = b"X" * 1000
        result = compressor.auto_compress(data)
        assert result.time_ms >= 0

    def test_ratio_field(self, compressor):
        data = b"\xAA" * 500
        result = compressor.auto_compress(data)
        assert 0 < result.ratio <= 1.0


class TestBuildDictionary:
    def test_empty_samples(self):
        d = Compressor.build_dictionary([])
        assert d == {}

    def test_single_sample(self):
        samples = [b"ABABAB"]
        d = Compressor.build_dictionary(samples)
        assert len(d) > 0

    def test_multiple_samples(self):
        samples = [b"HELLO WORLD", b"WORLD PEACE"]
        d = Compressor.build_dictionary(samples)
        assert len(d) > 0
        # Should contain common patterns
        all_patterns = b"".join(p for p in d)
        assert b"WORLD" in all_patterns or b"ORLD" in all_patterns

    def test_no_duplicate_tokens(self):
        samples = [b"ABCABC"]
        d = Compressor.build_dictionary(samples)
        tokens = list(d.values())
        assert len(tokens) == len(set(tokens))

    def test_max_entries_limited(self):
        # Create many unique ngrams
        samples = [bytes([i % 256, (i + 1) % 256, (i + 2) % 256]) for i in range(70000)]
        d = Compressor.build_dictionary(samples)
        assert len(d) <= 65535

    def test_token_ids_are_integers(self):
        d = Compressor.build_dictionary([b"ABC"])
        for token in d.values():
            assert isinstance(token, int)

    def test_patterns_are_bytes(self):
        d = Compressor.build_dictionary([b"ABCD"])
        for pattern in d:
            assert isinstance(pattern, bytes)
