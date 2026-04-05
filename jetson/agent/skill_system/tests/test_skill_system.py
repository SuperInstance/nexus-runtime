"""NEXUS Skill Loading System — Comprehensive tests.

Covers:
  - Cartridge creation from bytecode and JSON
  - Cartridge serialization/deserialization
  - Skill loading from directory
  - Trust level gating
  - Registry search and filtering
  - Built-in skills are valid cartridges
  - Built-in skills have valid bytecode (passes safety validator)
  - Skill parameter constraints are enforced
  - Edge cases: empty cartridge, corrupted JSON, missing bytecode
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
from pathlib import Path

import pytest

from agent.skill_system.cartridge import SkillCartridge, SkillParameter
from agent.skill_system.cartridge_builder import CartridgeBuilder, ValidationResult
from agent.skill_system.skill_loader import SkillLoader
from agent.skill_system.skill_registry import SkillRegistry
from agent.skill_system.builtin_skills import (
    get_builtin_skills,
    get_builtin_skill,
    list_builtin_skills,
)


# ===================================================================
# Fixtures
# ===================================================================

def _make_simple_bytecode() -> bytes:
    """Create a minimal valid bytecode program (3 instructions).

    READ_PIN 10, PUSH_F32 100.0, SUB_F
    Stack: [depth] -> [depth, 100.0] -> [depth-100.0]
    """
    bytecode = b""
    # READ_PIN 10
    bytecode += struct.pack("<BBHI", 0x1A, 0x01, 10, 0)
    # PUSH_F32 100.0
    val = struct.unpack("<I", struct.pack("<f", 100.0))[0]
    bytecode += struct.pack("<BBHI", 0x03, 0x02, 0, val)
    # SUB_F
    bytecode += struct.pack("<BBHI", 0x09, 0x00, 0, 0)
    return bytecode


def _make_write_bytecode() -> bytes:
    """Create bytecode with WRITE_PIN (trust L2 required).

    READ_PIN 4, PUSH_F32 90.0, SUB_F, CLAMP_F -45 45, WRITE_PIN 4, NOP
    """
    from reflex.bytecode_emitter import BytecodeEmitter
    em = BytecodeEmitter()
    em.emit_read_pin(4)
    em.emit_push_f32(90.0)
    em.emit_sub_f()
    em.emit_clamp_f(-45.0, 45.0)
    em.emit_write_pin(4)
    em.emit_nop()
    return em.get_bytecode()


def _make_sample_metadata() -> dict:
    """Create sample metadata for cartridge construction."""
    return {
        "name": "test_skill",
        "version": "1.2.3",
        "description": "A test skill cartridge",
        "domain": "marine",
        "trust_required": 2,
        "inputs": [
            {
                "name": "depth",
                "type": "sensor",
                "pin": 10,
                "range_min": 0.0,
                "range_max": 500.0,
                "unit": "meters",
                "description": "Depth sensor",
            },
        ],
        "outputs": [
            {
                "name": "thruster",
                "type": "actuator",
                "pin": 5,
                "range_min": -100.0,
                "range_max": 100.0,
                "unit": "percent",
                "description": "Thruster output",
            },
        ],
        "parameters": {"gain": 1.0, "tolerance": 5.0},
        "constraints": {"max_depth": {"min": 0.0, "max": 500.0}},
        "provenance": {"author": "test", "review_status": "tested"},
        "metadata": {"category": "test"},
    }


@pytest.fixture
def builder() -> CartridgeBuilder:
    """Provide a CartridgeBuilder instance."""
    return CartridgeBuilder()


@pytest.fixture
def sample_bytecode() -> bytes:
    """Provide a simple valid bytecode program."""
    return _make_simple_bytecode()


@pytest.fixture
def sample_metadata() -> dict:
    """Provide sample cartridge metadata."""
    return _make_sample_metadata()


@pytest.fixture
def sample_cartridge(sample_bytecode: bytes, sample_metadata: dict) -> SkillCartridge:
    """Provide a pre-built sample cartridge."""
    return CartridgeBuilder().from_bytecode(sample_bytecode, sample_metadata)


@pytest.fixture
def temp_cartridge_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample cartridge JSON files."""
    cart_dir = tmp_path / "cartridges"
    cart_dir.mkdir()

    # Create a few sample cartridge files
    for i, (name, trust) in enumerate([
        ("skill_alpha", 1),
        ("skill_beta", 2),
        ("skill_gamma", 0),
        ("skill_delta", 3),
    ]):
        bc = _make_simple_bytecode()
        cartridge = SkillCartridge(
            name=name,
            version="1.0.0",
            description=f"Test skill {name}",
            domain="marine",
            trust_required=trust,
            bytecode=bc,
        )
        data = cartridge.to_dict()
        with open(cart_dir / f"{name}.json", "w") as f:
            json.dump(data, f)

    return cart_dir


# ===================================================================
# SkillParameter Tests
# ===================================================================

class TestSkillParameter:
    """Tests for SkillParameter dataclass."""

    def test_create_parameter(self):
        param = SkillParameter(
            name="depth", type="sensor", pin=10,
            range_min=0.0, range_max=500.0,
            unit="meters", description="Depth sensor",
        )
        assert param.name == "depth"
        assert param.type == "sensor"
        assert param.pin == 10
        assert param.range_min == 0.0
        assert param.range_max == 500.0
        assert param.unit == "meters"

    def test_parameter_without_pin(self):
        param = SkillParameter(
            name="gain", type="variable",
            description="PID gain",
        )
        assert param.pin is None
        assert param.range_min is None
        assert param.range_max is None
        assert param.unit is None

    def test_serialization_roundtrip(self):
        param = SkillParameter(
            name="heading", type="sensor", pin=4,
            range_min=0.0, range_max=360.0,
            unit="degrees", description="Compass heading",
        )
        d = param.to_dict()
        restored = SkillParameter.from_dict(d)
        assert restored.name == param.name
        assert restored.type == param.type
        assert restored.pin == param.pin
        assert restored.range_min == param.range_min
        assert restored.range_max == param.range_max
        assert restored.unit == param.unit

    def test_validate_value_in_range(self):
        param = SkillParameter(
            name="depth", type="sensor",
            range_min=0.0, range_max=500.0,
        )
        ok, msg = param.validate_value(250.0)
        assert ok is True
        assert msg == ""

    def test_validate_value_below_min(self):
        param = SkillParameter(
            name="depth", type="sensor",
            range_min=0.0, range_max=500.0,
        )
        ok, msg = param.validate_value(-10.0)
        assert ok is False
        assert "below minimum" in msg

    def test_validate_value_above_max(self):
        param = SkillParameter(
            name="depth", type="sensor",
            range_min=0.0, range_max=500.0,
        )
        ok, msg = param.validate_value(600.0)
        assert ok is False
        assert "above maximum" in msg

    def test_validate_value_no_range(self):
        param = SkillParameter(name="gain", type="variable")
        ok, msg = param.validate_value(99999.0)
        assert ok is True

    def test_from_dict_minimal(self):
        param = SkillParameter.from_dict({"name": "x", "type": "variable"})
        assert param.name == "x"
        assert param.type == "variable"
        assert param.pin is None


# ===================================================================
# SkillCartridge Tests
# ===================================================================

class TestSkillCartridge:
    """Tests for SkillCartridge dataclass."""

    def test_create_cartridge(self, sample_bytecode):
        cart = SkillCartridge(
            name="test", bytecode=sample_bytecode,
            trust_required=2,
        )
        assert cart.name == "test"
        assert cart.trust_required == 2
        assert cart.version == "1.0.0"
        assert cart.domain == "marine"
        assert cart.is_bytecode_valid is True
        assert cart.instruction_count == 3
        assert cart.bytecode_size == 24

    def test_empty_bytecode_invalid(self):
        cart = SkillCartridge(name="empty", bytecode=b"")
        assert cart.is_bytecode_valid is False
        assert cart.instruction_count == 0

    def test_misaligned_bytecode_invalid(self):
        cart = SkillCartridge(name="bad", bytecode=b"\x00\x00\x00")
        assert cart.is_bytecode_valid is False
        assert cart.instruction_count == 0

    def test_serialization_roundtrip(self, sample_bytecode, sample_metadata):
        cart = CartridgeBuilder().from_bytecode(sample_bytecode, sample_metadata)
        d = cart.to_dict()
        restored = SkillCartridge.from_dict(d)

        assert restored.name == cart.name
        assert restored.version == cart.version
        assert restored.description == cart.description
        assert restored.domain == cart.domain
        assert restored.trust_required == cart.trust_required
        assert restored.bytecode == cart.bytecode
        assert len(restored.inputs) == len(cart.inputs)
        assert len(restored.outputs) == len(cart.outputs)
        assert restored.parameters == cart.parameters

    def test_summary(self, sample_cartridge):
        summary = sample_cartridge.summary()
        assert "test_skill" in summary
        assert "marine" in summary
        assert "24 bytes" in summary

    def test_default_values(self):
        cart = SkillCartridge(name="defaults")
        assert cart.version == "1.0.0"
        assert cart.domain == "marine"
        assert cart.trust_required == 0
        assert cart.bytecode == b""
        assert cart.inputs == []
        assert cart.outputs == []
        assert cart.parameters == {}


# ===================================================================
# CartridgeBuilder Tests
# ===================================================================

class TestCartridgeBuilder:
    """Tests for CartridgeBuilder."""

    def test_from_bytecode_basic(self, builder, sample_bytecode, sample_metadata):
        cart = builder.from_bytecode(sample_bytecode, sample_metadata)
        assert cart.name == "test_skill"
        assert cart.bytecode == sample_bytecode
        assert cart.trust_required == 2
        assert len(cart.inputs) == 1
        assert len(cart.outputs) == 1

    def test_from_bytecode_empty_raises(self, builder, sample_metadata):
        with pytest.raises(ValueError, match="empty"):
            builder.from_bytecode(b"", sample_metadata)

    def test_from_bytecode_misaligned_raises(self, builder, sample_metadata):
        with pytest.raises(ValueError, match="8-byte aligned"):
            builder.from_bytecode(b"\x00\x00\x00", sample_metadata)

    def test_from_bytecode_missing_name_raises(self, builder, sample_bytecode):
        with pytest.raises(ValueError, match="name"):
            builder.from_bytecode(sample_bytecode, {"name": ""})

    def test_from_intent_json(self, builder):
        """Test building from a JSON reflex definition."""
        intent = json.dumps({
            "name": "heading_hold",
            "intent": "Maintain heading 270 degrees",
            "body": [
                {"op": "READ_PIN", "arg": 0},
                {"op": "PUSH_F32", "value": 270.0},
                {"op": "SUB_F"},
                {"op": "CLAMP_F", "lo": -30.0, "hi": 30.0},
                {"op": "WRITE_PIN", "arg": 0},
                {"op": "NOP", "flags": "0x80", "operand1": 1},
            ],
        })
        cart = builder.from_intent(intent, {
            "name": "heading_hold_from_intent",
            "trust_required": 5,  # Needs L5 for SYSCALL (HALT)
        })
        assert cart.name == "heading_hold_from_intent"
        assert cart.is_bytecode_valid is True

    def test_from_intent_text_fallback(self, builder):
        """Test building from plain text (fallback synthesizer)."""
        cart = builder.from_intent("monitor depth sensor", {
            "name": "text_intent_skill",
            "trust_required": 0,
        })
        assert cart.name == "text_intent_skill"
        assert cart.is_bytecode_valid is True
        assert cart.instruction_count == 5

    def test_to_json_and_from_json(self, builder, sample_bytecode, sample_metadata, tmp_path):
        """Test round-trip JSON serialization."""
        cart = builder.from_bytecode(sample_bytecode, sample_metadata)
        json_path = str(tmp_path / "test_skill.json")

        builder.to_json(cart, json_path)

        assert os.path.exists(json_path)
        loaded = builder.from_json(json_path)
        assert loaded.name == cart.name
        assert loaded.bytecode == cart.bytecode
        assert loaded.version == cart.version
        assert loaded.trust_required == cart.trust_required

    def test_from_json_file_not_found(self, builder):
        with pytest.raises(FileNotFoundError):
            builder.from_json("/nonexistent/path/skill.json")

    def test_from_json_corrupted(self, builder, tmp_path):
        bad_path = str(tmp_path / "bad.json")
        with open(bad_path, "w") as f:
            f.write("{invalid json{{{")
        with pytest.raises(ValueError, match="Invalid JSON"):
            builder.from_json(bad_path)

    def test_from_json_missing_name(self, builder, tmp_path):
        bad_path = str(tmp_path / "noname.json")
        with open(bad_path, "w") as f:
            json.dump({"version": "1.0.0"}, f)
        with pytest.raises(ValueError, match="name"):
            builder.from_json(bad_path)

    def test_from_json_not_dict(self, builder, tmp_path):
        bad_path = str(tmp_path / "array.json")
        with open(bad_path, "w") as f:
            json.dump([1, 2, 3], f)
        with pytest.raises(ValueError, match="JSON object"):
            builder.from_json(bad_path)

    def test_to_json_no_name_raises(self, builder, tmp_path):
        cart = SkillCartridge(name="")
        with pytest.raises(ValueError, match="name"):
            builder.to_json(cart, str(tmp_path / "noname.json"))

    def test_validate_valid_cartridge(self, builder, sample_cartridge):
        result = builder.validate(sample_cartridge)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_empty_name(self, builder, sample_bytecode):
        cart = SkillCartridge(name="", bytecode=sample_bytecode)
        result = builder.validate(cart)
        assert result.valid is False
        assert any("name" in e for e in result.errors)

    def test_validate_empty_bytecode(self, builder):
        cart = SkillCartridge(name="empty_bc")
        result = builder.validate(cart)
        assert result.valid is False
        assert any("empty" in e.lower() for e in result.errors)

    def test_validate_bad_trust_level(self, builder, sample_bytecode):
        cart = SkillCartridge(name="bad_trust", bytecode=sample_bytecode, trust_required=10)
        result = builder.validate(cart)
        assert result.valid is False
        assert any("trust" in e.lower() for e in result.errors)

    def test_validate_invalid_param_type(self, builder, sample_bytecode):
        cart = SkillCartridge(
            name="bad_param", bytecode=sample_bytecode,
            inputs=[SkillParameter(name="x", type="invalid_type")],
        )
        result = builder.validate(cart)
        assert result.valid is False
        assert any("invalid type" in e.lower() for e in result.errors)

    def test_validate_inverted_range(self, builder, sample_bytecode):
        cart = SkillCartridge(
            name="bad_range", bytecode=sample_bytecode,
            inputs=[SkillParameter(
                name="x", type="sensor",
                range_min=100.0, range_max=0.0,
            )],
        )
        result = builder.validate(cart)
        assert result.valid is False
        assert any("range_min" in e for e in result.errors)

    def test_validate_version_warning(self, builder, sample_bytecode):
        cart = SkillCartridge(name="v", bytecode=sample_bytecode, version="not-semver")
        result = builder.validate(cart)
        assert result.valid is True  # Warning only, not error
        assert any("version" in w.lower() for w in result.warnings)


# ===================================================================
# SkillLoader Tests
# ===================================================================

class TestSkillLoader:
    """Tests for SkillLoader."""

    def test_list_available(self, temp_cartridge_dir):
        loader = SkillLoader(str(temp_cartridge_dir))
        available = loader.list_available()
        assert "skill_alpha" in available
        assert "skill_beta" in available
        assert "skill_gamma" in available
        assert "skill_delta" in available
        assert len(available) == 4

    def test_load_skill(self, temp_cartridge_dir):
        loader = SkillLoader(str(temp_cartridge_dir))
        cart = loader.load("skill_alpha")
        assert cart.name == "skill_alpha"
        assert cart.trust_required == 1
        assert cart.is_bytecode_valid is True

    def test_load_caches(self, temp_cartridge_dir):
        loader = SkillLoader(str(temp_cartridge_dir))
        cart1 = loader.load("skill_alpha")
        cart2 = loader.load("skill_alpha")
        assert cart1 is cart2  # Same object, cached

    def test_load_nonexistent_raises(self, temp_cartridge_dir):
        loader = SkillLoader(str(temp_cartridge_dir))
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent_skill")

    def test_unload(self, temp_cartridge_dir):
        loader = SkillLoader(str(temp_cartridge_dir))
        loader.load("skill_alpha")
        assert loader.is_loaded("skill_alpha")
        loader.unload("skill_alpha")
        assert not loader.is_loaded("skill_alpha")

    def test_unload_nonexistent_raises(self, temp_cartridge_dir):
        loader = SkillLoader(str(temp_cartridge_dir))
        with pytest.raises(KeyError):
            loader.unload("nonexistent")

    def test_get_skill(self, temp_cartridge_dir):
        loader = SkillLoader(str(temp_cartridge_dir))
        loader.load("skill_alpha")
        cart = loader.get_skill("skill_alpha")
        assert cart is not None
        assert cart.name == "skill_alpha"

    def test_get_skill_not_loaded(self, temp_cartridge_dir):
        loader = SkillLoader(str(temp_cartridge_dir))
        assert loader.get_skill("skill_alpha") is None

    def test_check_trust_allowed(self, temp_cartridge_dir):
        loader = SkillLoader(str(temp_cartridge_dir))
        loader.load("skill_alpha")  # trust_required=1
        assert loader.check_trust("skill_alpha", 1) is True
        assert loader.check_trust("skill_alpha", 2) is True
        assert loader.check_trust("skill_alpha", 5) is True

    def test_check_trust_denied(self, temp_cartridge_dir):
        loader = SkillLoader(str(temp_cartridge_dir))
        loader.load("skill_delta")  # trust_required=3
        assert loader.check_trust("skill_delta", 0) is False
        assert loader.check_trust("skill_delta", 2) is False
        assert loader.check_trust("skill_delta", 3) is True

    def test_load_all(self, temp_cartridge_dir):
        loader = SkillLoader(str(temp_cartridge_dir))
        loaded = loader.load_all()
        assert len(loaded) == 4
        assert "skill_alpha" in loaded
        assert "skill_delta" in loaded

    def test_load_cartridge_direct(self):
        loader = SkillLoader("/nonexistent/dir")
        cart = SkillCartridge(name="direct_skill", bytecode=_make_simple_bytecode())
        loaded = loader.load_cartridge_direct(cart)
        assert loaded is cart
        assert loader.is_loaded("direct_skill")
        assert loader.get_skill("direct_skill") is cart

    def test_list_loaded(self, temp_cartridge_dir):
        loader = SkillLoader(str(temp_cartridge_dir))
        loader.load("skill_alpha")
        loader.load("skill_beta")
        loaded = loader.list_loaded()
        assert loaded == ["skill_alpha", "skill_beta"]

    def test_get_by_domain(self, temp_cartridge_dir):
        loader = SkillLoader(str(temp_cartridge_dir))
        loader.load_all()
        marine = loader.get_by_domain("marine")
        assert len(marine) == 4

    def test_get_by_trust_level(self, temp_cartridge_dir):
        loader = SkillLoader(str(temp_cartridge_dir))
        loader.load_all()
        low_trust = loader.get_by_trust_level(1)
        # trust_required: alpha=1, beta=2, gamma=0, delta=3
        # max_trust=1 includes: alpha(1), gamma(0)
        names = {c.name for c in low_trust}
        assert "skill_gamma" in names
        assert "skill_alpha" in names
        assert "skill_delta" not in names

    def test_empty_directory(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        loader = SkillLoader(str(empty_dir))
        assert loader.list_available() == []
        assert loader.load_all() == {}

    def test_reload(self, temp_cartridge_dir):
        loader = SkillLoader(str(temp_cartridge_dir))
        cart1 = loader.load("skill_alpha")
        cart2 = loader.reload("skill_alpha")
        assert cart2.name == "skill_alpha"
        # Should be a new object after reload
        assert cart2 is not cart1


# ===================================================================
# SkillRegistry Tests
# ===================================================================

class TestSkillRegistry:
    """Tests for SkillRegistry."""

    def test_register_and_get(self):
        registry = SkillRegistry()
        cart = SkillCartridge(name="test_reg", bytecode=_make_simple_bytecode())
        skill_id = registry.register(cart)

        assert skill_id is not None
        assert isinstance(skill_id, str)
        assert len(skill_id) > 0

        record = registry.get(skill_id)
        assert record is not None
        assert record.cartridge.name == "test_reg"
        assert record.status == "available"
        assert record.source == "unknown"

    def test_register_with_source(self):
        registry = SkillRegistry()
        cart = SkillCartridge(name="builtin_skill", bytecode=_make_simple_bytecode())
        skill_id = registry.register(cart, source="builtin")

        record = registry.get(skill_id)
        assert record.source == "builtin"

    def test_register_no_name_raises(self):
        registry = SkillRegistry()
        cart = SkillCartridge(name="", bytecode=_make_simple_bytecode())
        with pytest.raises(ValueError, match="name"):
            registry.register(cart)

    def test_register_replaces_same_name(self):
        registry = SkillRegistry()
        cart1 = SkillCartridge(
            name="skill", version="1.0.0",
            bytecode=_make_simple_bytecode(),
        )
        cart2 = SkillCartridge(
            name="skill", version="2.0.0",
            bytecode=_make_simple_bytecode(),
        )
        id1 = registry.register(cart1)
        id2 = registry.register(cart2)

        # The second registration should replace the first
        assert registry.count() == 1
        record = registry.get(id2)
        assert record.cartridge.version == "2.0.0"

    def test_deregister(self):
        registry = SkillRegistry()
        cart = SkillCartridge(name="to_remove", bytecode=_make_simple_bytecode())
        skill_id = registry.register(cart)
        assert registry.count() == 1

        registry.deregister(skill_id)
        assert registry.count() == 0
        assert registry.get(skill_id) is None

    def test_deregister_nonexistent_raises(self):
        registry = SkillRegistry()
        with pytest.raises(KeyError):
            registry.deregister("nonexistent-id")

    def test_get_by_name(self):
        registry = SkillRegistry()
        cart = SkillCartridge(name="findme", bytecode=_make_simple_bytecode())
        registry.register(cart)

        record = registry.get_by_name("findme")
        assert record is not None
        assert record.cartridge.name == "findme"

    def test_get_by_name_not_found(self):
        registry = SkillRegistry()
        assert registry.get_by_name("nonexistent") is None

    def test_find_by_domain(self):
        registry = SkillRegistry()
        registry.register(SkillCartridge(name="m1", domain="marine", bytecode=_make_simple_bytecode()))
        registry.register(SkillCartridge(name="m2", domain="marine", bytecode=_make_simple_bytecode()))
        registry.register(SkillCartridge(name="a1", domain="aerial", bytecode=_make_simple_bytecode()))

        marine = registry.find_by_domain("marine")
        assert len(marine) == 2
        assert all(c.domain == "marine" for c in marine)

    def test_find_by_trust_level(self):
        registry = SkillRegistry()
        registry.register(SkillCartridge(name="t0", trust_required=0, bytecode=_make_simple_bytecode()))
        registry.register(SkillCartridge(name="t1", trust_required=1, bytecode=_make_simple_bytecode()))
        registry.register(SkillCartridge(name="t3", trust_required=3, bytecode=_make_simple_bytecode()))

        low = registry.find_by_trust_level(1)
        names = {c.name for c in low}
        assert "t0" in names
        assert "t1" in names
        assert "t3" not in names

    def test_get_compatible(self):
        registry = SkillRegistry()
        registry.register(SkillCartridge(name="c0", trust_required=0, domain="marine", bytecode=_make_simple_bytecode()))
        registry.register(SkillCartridge(name="c1", trust_required=1, domain="marine", bytecode=_make_simple_bytecode()))
        registry.register(SkillCartridge(name="c2", trust_required=2, domain="marine", bytecode=_make_simple_bytecode()))
        registry.register(SkillCartridge(name="a1", trust_required=1, domain="aerial", bytecode=_make_simple_bytecode()))

        compatible = registry.get_compatible(trust_level=1, domain="marine")
        names = [c.name for c in compatible]
        assert names == ["c0", "c1"]  # Sorted by trust_required

    def test_get_compatible_no_domain_filter(self):
        registry = SkillRegistry()
        registry.register(SkillCartridge(name="m1", trust_required=0, domain="marine", bytecode=_make_simple_bytecode()))
        registry.register(SkillCartridge(name="a1", trust_required=1, domain="aerial", bytecode=_make_simple_bytecode()))

        compatible = registry.get_compatible(trust_level=1)
        assert len(compatible) == 2

    def test_list_all(self):
        registry = SkillRegistry()
        for i in range(5):
            registry.register(SkillCartridge(
                name=f"skill_{i}", bytecode=_make_simple_bytecode(),
            ))
        all_skills = registry.list_all()
        assert len(all_skills) == 5

    def test_list_names(self):
        registry = SkillRegistry()
        registry.register(SkillCartridge(name="charlie", bytecode=_make_simple_bytecode()))
        registry.register(SkillCartridge(name="alpha", bytecode=_make_simple_bytecode()))
        registry.register(SkillCartridge(name="bravo", bytecode=_make_simple_bytecode()))

        names = registry.list_names()
        assert names == ["alpha", "bravo", "charlie"]

    def test_has_skill(self):
        registry = SkillRegistry()
        registry.register(SkillCartridge(name="exists", bytecode=_make_simple_bytecode()))
        assert registry.has_skill("exists") is True
        assert registry.has_skill("nope") is False

    def test_update_status(self):
        registry = SkillRegistry()
        cart = SkillCartridge(name="s", bytecode=_make_simple_bytecode())
        skill_id = registry.register(cart)

        registry.update_status(skill_id, "deployed")
        assert registry.get(skill_id).status == "deployed"

    def test_update_status_invalid(self):
        registry = SkillRegistry()
        cart = SkillCartridge(name="s", bytecode=_make_simple_bytecode())
        skill_id = registry.register(cart)
        with pytest.raises(ValueError, match="Invalid status"):
            registry.update_status(skill_id, "invalid_status")

    def test_find_by_source(self):
        registry = SkillRegistry()
        registry.register(SkillCartridge(name="b1", bytecode=_make_simple_bytecode()), source="builtin")
        registry.register(SkillCartridge(name="n1", bytecode=_make_simple_bytecode()), source="network")
        registry.register(SkillCartridge(name="b2", bytecode=_make_simple_bytecode()), source="builtin")

        builtin = registry.find_by_source("builtin")
        assert len(builtin) == 2
        assert all(c.name.startswith("b") for c in builtin)

    def test_clear(self):
        registry = SkillRegistry()
        for i in range(10):
            registry.register(SkillCartridge(
                name=f"skill_{i}", bytecode=_make_simple_bytecode(),
            ))
        assert registry.count() == 10
        registry.clear()
        assert registry.count() == 0


# ===================================================================
# Built-in Skills Tests
# ===================================================================

class TestBuiltinSkills:
    """Tests for the built-in marine skill cartridges."""

    def test_all_five_skills_exist(self):
        skills = get_builtin_skills()
        expected = {
            "surface_navigation",
            "depth_monitoring",
            "station_keeping",
            "emergency_surface",
            "sensor_survey",
        }
        assert set(skills.keys()) == expected

    def test_list_builtin_skills(self):
        names = list_builtin_skills()
        assert len(names) == 5
        assert names == sorted(names)  # Should be sorted

    def test_get_builtin_skill(self):
        skill = get_builtin_skill("surface_navigation")
        assert skill is not None
        assert skill.name == "surface_navigation"

    def test_get_builtin_skill_nonexistent(self):
        assert get_builtin_skill("nonexistent") is None

    def test_each_skill_has_valid_bytecode(self):
        """Every built-in skill must have non-empty, 8-byte aligned bytecode."""
        skills = get_builtin_skills()
        for name, skill in skills.items():
            assert skill.is_bytecode_valid, f"{name}: bytecode is invalid"
            assert skill.instruction_count > 0, f"{name}: no instructions"

    def test_each_skill_has_required_metadata(self):
        """Every built-in skill must have complete metadata."""
        skills = get_builtin_skills()
        for name, skill in skills.items():
            assert skill.name == name
            assert skill.version, f"{name}: missing version"
            assert skill.description, f"{name}: missing description"
            assert skill.domain == "marine", f"{name}: domain should be marine"
            assert isinstance(skill.trust_required, int), f"{name}: trust_required not int"
            assert 0 <= skill.trust_required <= 5, f"{name}: trust_required out of range"

    def test_each_skill_has_inputs(self):
        """Every built-in skill should define at least one input."""
        skills = get_builtin_skills()
        for name, skill in skills.items():
            assert len(skill.inputs) > 0, f"{name}: no inputs defined"

    def test_each_skill_has_parameters(self):
        """Every built-in skill should have configurable parameters."""
        skills = get_builtin_skills()
        for name, skill in skills.items():
            assert len(skill.parameters) > 0, f"{name}: no parameters"

    def test_each_skill_has_provenance(self):
        """Every built-in skill should have provenance metadata."""
        skills = get_builtin_skills()
        for name, skill in skills.items():
            assert "author" in skill.provenance, f"{name}: missing author"
            assert "review_status" in skill.provenance, f"{name}: missing review_status"

    def test_trust_levels_are_correct(self):
        """Verify trust levels match the specification."""
        skills = get_builtin_skills()
        assert skills["emergency_surface"].trust_required == 2  # Bug C6: now L2 (needs WRITE_PIN)
        assert skills["depth_monitoring"].trust_required == 1
        assert skills["sensor_survey"].trust_required == 1
        assert skills["surface_navigation"].trust_required == 2
        assert skills["station_keeping"].trust_required == 3

    def test_bytecode_passes_safety_validator(self):
        """Each built-in skill's bytecode must pass the safety pipeline
        at its declared trust level."""
        from core.safety_validator.pipeline import BytecodeSafetyPipeline

        skills = get_builtin_skills()
        for name, skill in skills.items():
            pipeline = BytecodeSafetyPipeline(trust_level=skill.trust_required)
            report = pipeline.validate(skill.bytecode)
            # Stages 1-5 must pass (stage 6 is informational/warnings)
            critical_stages = [s for s in report.stages if s.stage_name != "adversarial_probing"]
            for stage in critical_stages:
                assert stage.passed, (
                    f"{name}: safety pipeline stage '{stage.stage_name}' failed "
                    f"at trust L{skill.trust_required}: {stage.errors}"
                )

    def test_emergency_surface_is_readonly(self):
        """Emergency surface (L2) should use L2 opcodes including WRITE_PIN."""
        from core.safety_validator.rules import L2_OPCODES

        skill = get_builtin_skill("emergency_surface")
        bc = skill.bytecode
        for i in range(len(bc) // 8):
            opcode = bc[i * 8]
            assert opcode in L2_OPCODES, (
                f"emergency_surface instruction {i}: opcode 0x{opcode:02X} "
                f"not in L2 opcodes"
            )

    def test_surface_navigation_uses_write_pin(self):
        """Surface navigation (L2) should include WRITE_PIN."""
        skill = get_builtin_skill("surface_navigation")
        bc = skill.bytecode
        has_write = False
        for i in range(len(bc) // 8):
            if bc[i * 8] == 0x1B:  # WRITE_PIN
                has_write = True
                break
        assert has_write, "surface_navigation should use WRITE_PIN"

    def test_station_keeping_has_multiple_instructions(self):
        """Station keeping should be a multi-step program."""
        skill = get_builtin_skill("station_keeping")
        assert skill.instruction_count >= 10

    def test_builtin_skills_are_immutable_cache(self):
        """Calling get_builtin_skills() twice returns the same objects."""
        s1 = get_builtin_skills()
        s2 = get_builtin_skills()
        assert s1 is s2  # Same cached dict


# ===================================================================
# Integration Tests
# ===================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_build_validate_load_flow(self, tmp_path):
        """Full flow: build cartridge, validate, save, load."""
        builder = CartridgeBuilder()
        bc = _make_write_bytecode()
        cart = builder.from_bytecode(bc, {
            "name": "integration_skill",
            "trust_required": 2,
            "description": "Integration test skill",
        })

        # Validate
        result = builder.validate(cart)
        assert result.valid is True

        # Save to JSON
        cart_dir = tmp_path / "cartridges"
        cart_dir.mkdir()
        builder.to_json(cart, str(cart_dir / "integration_skill.json"))

        # Load from directory
        loader = SkillLoader(str(cart_dir))
        loaded = loader.load("integration_skill")
        assert loaded.name == "integration_skill"
        assert loaded.bytecode == cart.bytecode

    def test_registry_with_loader(self, temp_cartridge_dir):
        """Load skills via loader, register in registry, query."""
        loader = SkillLoader(str(temp_cartridge_dir))
        loaded = loader.load_all()

        registry = SkillRegistry()
        for cart in loaded.values():
            registry.register(cart, source="file")

        assert registry.count() == 4

        # Find compatible at trust L1
        compatible = registry.get_compatible(trust_level=1, domain="marine")
        names = [c.name for c in compatible]
        assert "skill_gamma" in names  # trust_required=0
        assert "skill_alpha" in names  # trust_required=1
        assert "skill_delta" not in names  # trust_required=3

    def test_builtin_skills_in_registry(self):
        """All built-in skills can be registered and queried."""
        registry = SkillRegistry()
        builtin = get_builtin_skills()

        for cart in builtin.values():
            registry.register(cart, source="builtin")

        assert registry.count() == 5
        assert registry.has_skill("emergency_surface")
        assert registry.has_skill("station_keeping")

        marine = registry.find_by_domain("marine")
        assert len(marine) == 5

        # Emergency surface should be available at L2 (requires WRITE_PIN)
        l0_skills = registry.get_compatible(trust_level=2)
        assert any(c.name == "emergency_surface" for c in l0_skills)

    def test_trust_gating_with_loader_and_registry(self, temp_cartridge_dir):
        """Combined trust gating across loader and registry."""
        loader = SkillLoader(str(temp_cartridge_dir))
        loader.load_all()

        registry = SkillRegistry()
        for cart in loader.loaded_skills.values():
            registry.register(cart)

        # At trust L1, only gamma (0) and alpha (1) should pass
        for cart in registry.get_compatible(trust_level=1):
            assert loader.check_trust(cart.name, 1)

    def test_builtin_skills_validation_via_builder(self):
        """Validate all built-in skills via CartridgeBuilder.validate()."""
        builder = CartridgeBuilder()
        skills = get_builtin_skills()

        for name, skill in skills.items():
            result = builder.validate(skill)
            # May have warnings (e.g., version format) but no errors
            assert result.valid, (
                f"{name}: validation failed: {result.errors}"
            )


# ===================================================================
# Edge Case Tests
# ===================================================================

class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_cartridge_with_minimal_data(self):
        cart = SkillCartridge(name="minimal")
        assert cart.name == "minimal"
        assert cart.bytecode == b""
        assert cart.inputs == []

    def test_cartridge_from_dict_empty_bytecode(self):
        data = {"name": "empty_bc", "bytecode": "", "version": "1.0.0"}
        cart = SkillCartridge.from_dict(data)
        assert cart.bytecode == b""
        assert cart.is_bytecode_valid is False

    def test_cartridge_from_dict_missing_fields(self):
        data = {"name": "sparse"}
        cart = SkillCartridge.from_dict(data)
        assert cart.name == "sparse"
        assert cart.version == "1.0.0"
        assert cart.domain == "marine"
        assert cart.trust_required == 0

    def test_parameter_constraint_with_inverted_bounds(self, builder, sample_bytecode):
        cart = SkillCartridge(
            name="bad_constraint",
            bytecode=sample_bytecode,
            constraints={"limit": {"min": 100.0, "max": 50.0}},
        )
        result = builder.validate(cart)
        assert result.valid is False
        assert any("min" in e and "max" in e for e in result.errors)

    def test_loader_with_corrupted_json_file(self, tmp_path):
        """Loader should skip corrupted files gracefully."""
        cart_dir = tmp_path / "cartridges"
        cart_dir.mkdir()

        # Write a corrupted JSON file
        with open(cart_dir / "corrupted.json", "w") as f:
            f.write("NOT VALID JSON {{{{")

        # Write a valid file
        valid_cart = SkillCartridge(name="valid", bytecode=_make_simple_bytecode())
        with open(cart_dir / "valid.json", "w") as f:
            json.dump(valid_cart.to_dict(), f)

        loader = SkillLoader(str(cart_dir))
        loaded = loader.load_all()
        # The corrupted file should be skipped
        assert "valid" in loaded
        # corrupted.json might appear in list_available but not load_all

    def test_registry_deregister_twice_raises(self):
        registry = SkillRegistry()
        cart = SkillCartridge(name="temp", bytecode=_make_simple_bytecode())
        sid = registry.register(cart)
        registry.deregister(sid)
        with pytest.raises(KeyError):
            registry.deregister(sid)

    def test_trust_level_boundary(self, builder):
        """Test trust level exactly at boundary."""
        bc = _make_simple_bytecode()
        # L0 bytecode (READ_PIN + PUSH + SUB) should be valid at all levels
        for level in range(6):
            cart = SkillCartridge(name=f"t{level}", bytecode=bc, trust_required=level)
            result = builder.validate(cart)
            assert result.valid is True
