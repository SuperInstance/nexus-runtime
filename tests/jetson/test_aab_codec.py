"""NEXUS Jetson tests - AAB codec tests.

10+ tests covering AAB encode/decode/strip and TLV tags.
"""

import struct

import pytest

from agent_runtime.aab_codec import (
    AABInstruction,
    AABCodec,
    AAB_MAGIC,
    AAB_VERSION,
    AAB_HEADER_SIZE,
    TLVEntry,
    TLV_END,
    TLV_TYPE_DESC,
    TLV_CAP_REQ,
    TLV_SAFETY_CONSTRAINT,
    TLV_TRUST_REQ,
    TLV_NARRATIVE,
    TLV_AGENT_SOURCE,
    TLV_INTENT_TAG,
    TLV_PRECONDITION,
    TLV_POSTCONDITION,
    TLV_PERFORMANCE_HINT,
    TLV_DOMAIN_TAG,
    TLV_VESSEL_CAP,
    TLV_TRUST_MULTIPLIER,
    TLV_TAG_NAMES,
    aab_decode,
    aab_encode,
    aab_strip,
    get_tlv_float,
    get_tlv_string,
    make_tlv,
    struct_pack_f32,
    struct_unpack_f32,
)
from agent_runtime.a2a_opcodes import (
    A2A_OPCODES,
    AgentContext,
    ActionResult,
    interpret_a2a_opcode,
    get_a2a_opcodes,
    get_a2a_opcode_count,
    is_a2a_opcode,
    a2a_opcode_name,
)


def _make_core(opcode: int = 0x09, flags: int = 0, op1: int = 0, op2: int = 0) -> bytes:
    return struct.pack("<BBHI", opcode, flags, op1, op2)


# ===================================================================
# Test 1: AAB Encode/Decode Round-Trip
# ===================================================================

class TestAABRoundTrip:
    def test_single_instruction_round_trip(self) -> None:
        instr = AABInstruction(
            core=_make_core(0x09, 0, 0, 0),
            metadata=[TLVEntry(tag=TLV_NARRATIVE, value=b"Subtract heading error")],
        )
        encoded = aab_encode([instr])
        decoded = aab_decode(encoded)
        assert len(decoded) == 1
        assert decoded[0].core == instr.core
        assert len(decoded[0].metadata) == 1
        assert decoded[0].metadata[0].tag == TLV_NARRATIVE
        assert decoded[0].metadata[0].value == b"Subtract heading error"

    def test_multiple_instructions_round_trip(self) -> None:
        instructions = [
            AABInstruction(
                core=_make_core(0x1A, 0x01, 0, 0),
                metadata=[TLVEntry(tag=TLV_CAP_REQ, value=b"sensor:compass")],
            ),
            AABInstruction(
                core=_make_core(0x09, 0, 0, 0),
                metadata=[TLVEntry(tag=TLV_TYPE_DESC, value=b"f32->f32:degrees")],
            ),
            AABInstruction(
                core=_make_core(0x1B, 0x01, 0, 0),
                metadata=[
                    TLVEntry(tag=TLV_SAFETY_CONSTRAINT, value=b"max_turn_rate:30"),
                    TLVEntry(tag=TLV_TRUST_REQ, value=struct.pack("<f", 0.5)),
                ],
            ),
        ]
        encoded = aab_encode(instructions)
        decoded = aab_decode(encoded)
        assert len(decoded) == 3
        for i in range(3):
            assert decoded[i].core == instructions[i].core
            assert len(decoded[i].metadata) == len(instructions[i].metadata)


# ===================================================================
# Test 2: AAB Strip
# ===================================================================

class TestAABStrip:
    def test_strip_removes_all_tlv(self) -> None:
        instructions = [
            AABInstruction(
                core=_make_core(0x1A, 0x01, 0, 0),
                metadata=[
                    TLVEntry(tag=TLV_NARRATIVE, value=b"Read compass"),
                    TLVEntry(tag=TLV_CAP_REQ, value=b"sensor:imu:compass"),
                ],
            ),
            AABInstruction(
                core=_make_core(0x10, 0x04, 0, 0),
                metadata=[TLVEntry(tag=TLV_SAFETY_CONSTRAINT, value=b"range:-30,30")],
            ),
        ]
        encoded = aab_encode(instructions)
        stripped = aab_strip(encoded)
        assert len(stripped) == 16
        assert stripped == _make_core(0x1A, 0x01, 0, 0) + _make_core(0x10, 0x04, 0, 0)

    def test_strip_empty_metadata(self) -> None:
        instr = AABInstruction(core=_make_core(0x00))
        encoded = aab_encode([instr])
        stripped = aab_strip(encoded)
        assert stripped == _make_core(0x00)


# ===================================================================
# Test 3: Multiple TLV Entries
# ===================================================================

class TestMultipleTLV:
    def test_five_tags_per_instruction(self) -> None:
        metadata = [
            TLVEntry(tag=TLV_TYPE_DESC, value=b"f32->f32"),
            TLVEntry(tag=TLV_CAP_REQ, value=b"sensor:imu"),
            TLVEntry(tag=TLV_SAFETY_CONSTRAINT, value=b"max:100"),
            TLVEntry(tag=TLV_NARRATIVE, value=b"Read and convert"),
            TLVEntry(tag=TLV_AGENT_SOURCE, value=b"autopilot-v2"),
        ]
        instr = AABInstruction(core=_make_core(), metadata=metadata)
        encoded = aab_encode([instr])
        decoded = aab_decode(encoded)
        assert len(decoded[0].metadata) == 5


# ===================================================================
# Test 4: All 13 Tag Types
# ===================================================================

class TestAllTLVTags:
    def test_all_13_tags_defined(self) -> None:
        expected_tags = {
            TLV_TYPE_DESC, TLV_CAP_REQ, TLV_SAFETY_CONSTRAINT,
            TLV_TRUST_REQ, TLV_NARRATIVE, TLV_AGENT_SOURCE,
            TLV_INTENT_TAG, TLV_PRECONDITION, TLV_POSTCONDITION,
            TLV_PERFORMANCE_HINT, TLV_DOMAIN_TAG, TLV_VESSEL_CAP,
            TLV_TRUST_MULTIPLIER,
        }
        for tag in expected_tags:
            assert tag in TLV_TAG_NAMES

    def test_all_13_tags_encode_decode(self) -> None:
        tags_and_values = [
            (TLV_TYPE_DESC, b"f32->f32:degrees"),
            (TLV_CAP_REQ, b"sensor:imu:compass"),
            (TLV_SAFETY_CONSTRAINT, b"max_turn_rate:30deg/s"),
            (TLV_TRUST_REQ, struct.pack("<f", 0.75)),
            (TLV_NARRATIVE, b"Compute heading error"),
            (TLV_AGENT_SOURCE, b"autopilot-agent-v3"),
            (TLV_INTENT_TAG, b"heading_hold"),
            (TLV_PRECONDITION, b"compass_calibrated"),
            (TLV_POSTCONDITION, b"rudder_angle_valid"),
            (TLV_PERFORMANCE_HINT, b"hot_path:true"),
            (TLV_DOMAIN_TAG, b"marine:autonomy"),
            (TLV_VESSEL_CAP, b"has_rudder:true"),
            (TLV_TRUST_MULTIPLIER, struct.pack("<f", 0.5)),
        ]
        for tag, value in tags_and_values:
            entry = TLVEntry(tag=tag, value=value)
            encoded = entry.encode()
            decoded_entry, _ = TLVEntry.decode(encoded, 0)
            assert decoded_entry.tag == tag
            assert decoded_entry.value == value


# ===================================================================
# Test 5: File Format Magic/Version
# ===================================================================

class TestFileFormat:
    def test_magic_bytes(self) -> None:
        instr = AABInstruction(core=_make_core())
        encoded = aab_encode([instr])
        assert encoded[0:4] == AAB_MAGIC

    def test_version(self) -> None:
        instr = AABInstruction(core=_make_core())
        encoded = aab_encode([instr])
        version = int.from_bytes(encoded[4:6], "little")
        assert version == AAB_VERSION

    def test_instruction_count(self) -> None:
        instructions = [AABInstruction(core=_make_core(i)) for i in range(7)]
        encoded = aab_encode(instructions)
        count = int.from_bytes(encoded[6:8], "little")
        assert count == 7

    def test_invalid_magic_rejected(self) -> None:
        with pytest.raises(ValueError, match="magic"):
            aab_decode(b"BAD!" + b"\x01\x00" + b"\x01\x00" + _make_core() + b"\x00")

    def test_invalid_version_rejected(self) -> None:
        data = AAB_MAGIC + b"\x99\x99" + b"\x01\x00" + _make_core() + b"\x00"
        with pytest.raises(ValueError, match="version"):
            aab_decode(data)


# ===================================================================
# Test 6: Empty Metadata
# ===================================================================

class TestEmptyMetadata:
    def test_instruction_with_no_metadata(self) -> None:
        instr = AABInstruction(core=_make_core(0x08))
        encoded = aab_encode([instr])
        decoded = aab_decode(encoded)
        assert len(decoded) == 1
        assert decoded[0].core == instr.core
        assert decoded[0].metadata == []


# ===================================================================
# Test 7: Large Instruction Count
# ===================================================================

class TestLargeCount:
    def test_100_instructions(self) -> None:
        instructions = [
            AABInstruction(core=_make_core(i % 32), metadata=[])
            for i in range(100)
        ]
        encoded = aab_encode(instructions)
        decoded = aab_decode(encoded)
        assert len(decoded) == 100
        for i in range(100):
            assert decoded[i].core == instructions[i].core


# ===================================================================
# Test 8: Binary Encoding Correctness
# ===================================================================

class TestBinaryEncoding:
    def test_tlv_encoding_format(self) -> None:
        entry = TLVEntry(tag=0x05, value=b"hello")
        encoded = entry.encode()
        assert len(encoded) == 1 + 2 + 5
        assert encoded[0] == 0x05
        assert int.from_bytes(encoded[1:3], "little") == 5
        assert encoded[3:8] == b"hello"

    def test_empty_value_tlv(self) -> None:
        entry = TLVEntry(tag=0x01, value=b"")
        encoded = entry.encode()
        assert len(encoded) == 3
        assert encoded[0] == 0x01
        assert int.from_bytes(encoded[1:3], "little") == 0


# ===================================================================
# Test 9: Agent Source Tag
# ===================================================================

class TestAgentSourceTag:
    def test_agent_source_retrieval(self) -> None:
        metadata = [
            TLVEntry(tag=TLV_AGENT_SOURCE, value=b"planning-agent"),
            TLVEntry(tag=TLV_NARRATIVE, value=b"Plan route"),
        ]
        agent = get_tlv_string(metadata, TLV_AGENT_SOURCE)
        assert agent == "planning-agent"
        narrative = get_tlv_string(metadata, TLV_NARRATIVE)
        assert narrative == "Plan route"
        missing = get_tlv_string(metadata, TLV_TRUST_REQ)
        assert missing is None


# ===================================================================
# Test 10: Trust Multiplier Tag
# ===================================================================

class TestTrustMultiplierTag:
    def test_trust_multiplier_float(self) -> None:
        metadata = [
            TLVEntry(tag=TLV_TRUST_MULTIPLIER, value=struct.pack("<f", 0.5)),
        ]
        multiplier = get_tlv_float(metadata, TLV_TRUST_MULTIPLIER)
        assert multiplier is not None
        assert abs(multiplier - 0.5) < 0.001

    def test_trust_multiplier_wrong_tag(self) -> None:
        metadata = [TLVEntry(tag=TLV_NARRATIVE, value=b"hello")]
        assert get_tlv_float(metadata, TLV_TRUST_MULTIPLIER) is None


# ===================================================================
# Test: make_tlv Helper
# ===================================================================

class TestMakeTLV:
    def test_string_tlv(self) -> None:
        tlv = make_tlv(TLV_NARRATIVE, "test description")
        assert tlv.tag == TLV_NARRATIVE
        assert tlv.value == b"test description"

    def test_float_tlv(self) -> None:
        tlv = make_tlv(TLV_TRUST_REQ, 0.75)
        assert tlv.tag == TLV_TRUST_REQ
        assert len(tlv.value) == 4
        val = struct.unpack("<f", tlv.value)[0]
        assert abs(val - 0.75) < 0.001


# ===================================================================
# Test: Legacy AABCodec class
# ===================================================================

class TestLegacyCodec:
    def test_encode_decode(self) -> None:
        codec = AABCodec()
        core = _make_core(0x09)
        metadata = [TLVEntry(tag=TLV_NARRATIVE, value=b"test")]
        encoded = codec.encode(core, metadata)
        decoded_core, decoded_meta = codec.decode(encoded)
        assert decoded_core == core
        assert len(decoded_meta) == 1

    def test_strip(self) -> None:
        codec = AABCodec()
        core = _make_core(0x09)
        metadata = [TLVEntry(tag=TLV_NARRATIVE, value=b"test")]
        encoded = codec.encode(core, metadata)
        stripped = codec.strip(encoded)
        assert stripped == core


# ===================================================================
# Test: A2A Opcodes (29)
# ===================================================================

class TestA2AOpcodes:
    def test_all_29_opcodes_defined(self) -> None:
        assert len(A2A_OPCODES) == 29

    def test_opcode_count(self) -> None:
        assert get_a2a_opcode_count() == 29

    def test_intent_opcodes(self) -> None:
        for op in [0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29]:
            assert is_a2a_opcode(op)
            assert a2a_opcode_name(op) is not None

    def test_communication_opcodes(self) -> None:
        for op in [0x30, 0x31, 0x32, 0x33, 0x34, 0x35]:
            assert is_a2a_opcode(op)

    def test_capability_opcodes(self) -> None:
        for op in [0x40, 0x41, 0x42, 0x43, 0x44, 0x45]:
            assert is_a2a_opcode(op)

    def test_safety_opcodes(self) -> None:
        for op in [0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56]:
            assert is_a2a_opcode(op)

    def test_unknown_opcode(self) -> None:
        assert not is_a2a_opcode(0x99)

    def test_interpret_declare_intent(self) -> None:
        ctx = AgentContext()
        result = interpret_a2a_opcode(0x20, (1, 0), [], ctx)
        assert result.success
        assert result.opcode == 0x20

    def test_interpret_trust_check_pass(self) -> None:
        ctx = AgentContext(trust_level=0.9)
        result = interpret_a2a_opcode(0x50, (128, 0), [], ctx)
        assert result.success
        assert not result.blocked

    def test_interpret_trust_check_fail(self) -> None:
        ctx = AgentContext(trust_level=0.1)
        result = interpret_a2a_opcode(0x50, (200, 0), [], ctx)
        assert not result.success
        assert result.blocked

    def test_interpret_unknown_opcode(self) -> None:
        ctx = AgentContext()
        result = interpret_a2a_opcode(0x99, (0, 0), [], ctx)
        assert not result.success
