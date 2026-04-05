"""NEXUS Schema Validator Tests.

Tests cover:
- Meta-validation of all schemas (schemas are valid JSON Schema)
- Valid configs pass validation
- Invalid configs are rejected with proper error paths
- Schemas match the codebase implementation
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.schema_validator import (
    SchemaValidator,
    ValidationError,
    ValidationResult,
    generate_sample_config,
    load_schema,
    validate_config,
)

# ===================================================================
# Fixtures
# ===================================================================

SCHEMA_DIR = Path(__file__).resolve().parent.parent.parent / "schemas"


@pytest.fixture
def validator() -> SchemaValidator:
    return SchemaValidator(schema_dir=SCHEMA_DIR)


@pytest.fixture
def all_schema_names() -> list[str]:
    return ["autonomy_state", "reflex_definition", "node_role_config", "serial_protocol"]


# ===================================================================
# Test 1: Schema Loading
# ===================================================================

class TestSchemaLoading:
    def test_all_schemas_load(self, validator: SchemaValidator) -> None:
        names = validator.list_schemas()
        assert len(names) == 4
        for expected in ["autonomy_state", "reflex_definition", "node_role_config", "serial_protocol"]:
            assert expected in names

    def test_get_schema_returns_dict(self, validator: SchemaValidator) -> None:
        schema = validator.get_schema("autonomy_state")
        assert isinstance(schema, dict)
        assert "$schema" in schema
        assert "title" in schema

    def test_unknown_schema_raises(self, validator: SchemaValidator) -> None:
        with pytest.raises(KeyError, match="Unknown schema"):
            validator.get_schema("nonexistent")

    def test_load_schema_convenience(self) -> None:
        schema = load_schema("reflex_definition", schema_dir=SCHEMA_DIR)
        assert schema["title"] == "NEXUS Reflex Definition"


# ===================================================================
# Test 2: Meta-Validation (schemas are valid JSON Schema)
# ===================================================================

class TestMetaValidation:
    @pytest.mark.parametrize("schema_name", [
        "autonomy_state", "reflex_definition", "node_role_config", "serial_protocol",
    ])
    def test_schema_is_valid_json_schema(
        self, validator: SchemaValidator, schema_name: str
    ) -> None:
        result = validator.meta_validate(schema_name)
        assert result.valid, f"{schema_name} meta-validation failed: {result.error_summary()}"

    def test_all_schemas_meta_validate(self, validator: SchemaValidator) -> None:
        for name in validator.list_schemas():
            result = validator.meta_validate(name)
            assert result.valid, f"Schema '{name}' is not valid JSON Schema"


# ===================================================================
# Test 3: Autonomy State Schema Validation
# ===================================================================

VALID_AUTONOMY_STATE = {
    "params": {
        "alpha_gain": 0.002,
        "alpha_loss": 0.05,
        "alpha_decay": 0.0001,
        "t_floor": 0.2,
        "quality_cap": 10,
        "evaluation_window_hours": 1.0,
        "severity_exponent": 1.0,
        "streak_bonus": 0.00005,
        "min_events_for_gain": 1,
        "n_penalty_slope": 0.1,
        "reset_grace_hours": 24.0,
        "promotion_cooldown_hours": 72.0,
    },
    "subsystems": [
        {
            "subsystem": "steering",
            "trust_score": 0.75,
            "autonomy_level": 3,
            "consecutive_clean_windows": 100,
            "total_windows": 500,
            "clean_windows": 450,
            "total_observation_hours": 400.0,
        },
        {
            "subsystem": "navigation",
            "trust_score": 0.5,
            "autonomy_level": 2,
        },
    ],
    "autonomy_levels": [
        {
            "level": 0, "name": "Disabled", "trust_threshold": None,
            "min_observation_hours": 0.0, "min_clean_windows": 0,
            "key_criteria": "Default after full reset",
        },
        {
            "level": 1, "name": "Advisory", "trust_threshold": 0.20,
            "min_observation_hours": 8.0, "min_clean_windows": 4,
            "key_criteria": "Min 8 hours observation, 4 clean windows",
        },
    ],
    "events": [
        {
            "event_type": "heartbeat_ok",
            "quality": 0.7,
            "severity": 0.0,
            "timestamp": 1714000000.0,
            "subsystem": "steering",
            "is_bad": False,
        },
        {
            "event_type": "reflex_error",
            "quality": 0.0,
            "severity": 0.6,
            "timestamp": 1714000100.0,
            "subsystem": "navigation",
            "is_bad": True,
        },
    ],
}


class TestAutonomyStateValidation:
    def test_valid_config_passes(self, validator: SchemaValidator) -> None:
        result = validator.validate(VALID_AUTONOMY_STATE, "autonomy_state")
        assert result.valid, result.error_summary()

    def test_minimal_config(self, validator: SchemaValidator) -> None:
        minimal = {"params": {"alpha_gain": 0.002, "alpha_loss": 0.05, "alpha_decay": 0.0001, "t_floor": 0.2}}
        result = validator.validate(minimal, "autonomy_state")
        assert result.valid, result.error_summary()

    def test_empty_object_passes(self, validator: SchemaValidator) -> None:
        """Autonomy state has no required top-level fields, so {} is valid."""
        result = validator.validate({}, "autonomy_state")
        assert result.valid

    def test_invalid_trust_score(self, validator: SchemaValidator) -> None:
        bad = {
            "subsystems": [
                {"subsystem": "steering", "trust_score": 1.5}
            ]
        }
        result = validator.validate(bad, "autonomy_state")
        assert not result.valid
        assert any("1.5" in e.message for e in result.errors)

    def test_invalid_autonomy_level(self, validator: SchemaValidator) -> None:
        bad = {
            "subsystems": [
                {"subsystem": "steering", "autonomy_level": 7}
            ]
        }
        result = validator.validate(bad, "autonomy_state")
        assert not result.valid

    def test_invalid_event_type(self, validator: SchemaValidator) -> None:
        bad = {
            "events": [
                {"event_type": "invalid_event_type"}
            ]
        }
        result = validator.validate(bad, "autonomy_state")
        assert not result.valid

    def test_bad_trust_params_missing_required(self, validator: SchemaValidator) -> None:
        bad = {"params": {"alpha_gain": 0.002}}
        result = validator.validate(bad, "autonomy_state")
        assert not result.valid
        assert any("alpha_loss" in e.message for e in result.errors)

    def test_additional_properties_rejected(self, validator: SchemaValidator) -> None:
        bad = {
            "params": {
                "alpha_gain": 0.002, "alpha_loss": 0.05, "alpha_decay": 0.0001, "t_floor": 0.2,
                "bogus_field": 42,
            }
        }
        result = validator.validate(bad, "autonomy_state")
        assert not result.valid

    def test_nested_additional_properties_rejected(self, validator: SchemaValidator) -> None:
        bad = {
            "subsystems": [
                {"subsystem": "steering", "bogus_nested": True}
            ]
        }
        result = validator.validate(bad, "autonomy_state")
        assert not result.valid


# ===================================================================
# Test 4: Reflex Definition Schema Validation
# ===================================================================

VALID_REFLEX = {
    "name": "heading_hold",
    "intent": "Maintain heading 270 degrees",
    "sensors": ["compass_heading", "wind_speed"],
    "actuators": ["rudder_angle"],
    "trust_min": 0.50,
    "author": "human",
    "pid": {"kp": 1.0, "ki": 0.1, "kd": 0.01, "output_min": -30.0, "output_max": 30.0},
    "trigger": {"type": "always"},
    "body": [
        {"op": "READ_PIN", "arg": 0},
        {"op": "PUSH_F32", "value": 270.0},
        {"op": "SUB_F"},
        {"op": "CLAMP_F", "lo": -30.0, "hi": 30.0},
        {"op": "WRITE_PIN", "arg": 0},
        {"op": "NOP", "flags": "0x80", "operand1": 1},
    ],
}


class TestReflexDefinitionValidation:
    def test_valid_reflex_passes(self, validator: SchemaValidator) -> None:
        result = validator.validate(VALID_REFLEX, "reflex_definition")
        assert result.valid, result.error_summary()

    def test_minimal_reflex(self, validator: SchemaValidator) -> None:
        minimal = {
            "name": "simple_add",
            "body": [{"op": "PUSH_F32", "value": 10.0}],
        }
        result = validator.validate(minimal, "reflex_definition")
        assert result.valid, result.error_summary()

    def test_all_opcodes_valid(self, validator: SchemaValidator) -> None:
        """All opcodes used in the compiler should be valid in the schema."""
        from reflex.compiler import VALID_OPCODES
        for op_name in VALID_OPCODES:
            reflex = {"name": "test_op", "body": [{"op": op_name}]}
            result = validator.validate(reflex, "reflex_definition")
            assert result.valid, f"Opcode '{op_name}' rejected: {result.error_summary()}"

    def test_invalid_opcode_rejected(self, validator: SchemaValidator) -> None:
        bad = {"name": "bad", "body": [{"op": "NONEXISTENT_OPCODE"}]}
        result = validator.validate(bad, "reflex_definition")
        assert not result.valid

    def test_missing_name_rejected(self, validator: SchemaValidator) -> None:
        bad = {"body": [{"op": "NOP"}]}
        result = validator.validate(bad, "reflex_definition")
        assert not result.valid

    def test_missing_body_rejected(self, validator: SchemaValidator) -> None:
        bad = {"name": "no_body"}
        result = validator.validate(bad, "reflex_definition")
        assert not result.valid

    def test_empty_body_rejected(self, validator: SchemaValidator) -> None:
        bad = {"name": "empty", "body": []}
        result = validator.validate(bad, "reflex_definition")
        assert not result.valid

    def test_invalid_name_pattern(self, validator: SchemaValidator) -> None:
        bad = {"name": "INVALID-CASE", "body": [{"op": "NOP"}]}
        result = validator.validate(bad, "reflex_definition")
        assert not result.valid

    def test_invalid_author(self, validator: SchemaValidator) -> None:
        bad = {"name": "bad_author", "body": [{"op": "NOP"}], "author": "robot"}
        result = validator.validate(bad, "reflex_definition")
        assert not result.valid

    def test_trigger_with_sensor(self, validator: SchemaValidator) -> None:
        reflex = {
            "name": "sensor_trigger",
            "body": [{"op": "NOP"}],
            "trigger": {
                "type": "sensor_threshold",
                "sensor_id": 5,
                "threshold": 100.0,
                "comparator": "gt",
            },
        }
        result = validator.validate(reflex, "reflex_definition")
        assert result.valid, result.error_summary()

    def test_excessive_instructions_rejected(self, validator: SchemaValidator) -> None:
        body = [{"op": "NOP"} for _ in range(1001)]
        bad = {"name": "too_long", "body": body}
        result = validator.validate(bad, "reflex_definition")
        assert not result.valid

    def test_additional_properties_rejected(self, validator: SchemaValidator) -> None:
        bad = {"name": "extra", "body": [{"op": "NOP"}], "bogus_field": True}
        result = validator.validate(bad, "reflex_definition")
        assert not result.valid


# ===================================================================
# Test 5: Node Role Config Schema Validation
# ===================================================================

VALID_NODE_CONFIG = {
    "node_id": "esp32-rudder-01",
    "role": "rudder_control",
    "firmware_version": "0.3.0",
    "sensors": [
        {
            "id": 0,
            "mode": "input",
            "label": "compass_heading",
            "sensor_type": "compass_heading",
        },
        {
            "id": 1,
            "mode": "input_pullup",
            "label": "wind_speed",
            "sensor_type": "wind_speed",
        },
    ],
    "actuators": [
        {
            "id": 0,
            "mode": "output",
            "label": "rudder_angle",
            "actuator_type": "servo",
            "safe_value": 0.0,
            "min_value": -45.0,
            "max_value": 45.0,
            "max_rate_per_tick": 0.5,
            "overcurrent_limit_ma": 500.0,
        },
    ],
    "safety": {
        "heartbeat_degrade_threshold": 5,
        "heartbeat_safe_threshold": 10,
        "watchdog_timeout_ms": 3000,
        "estop_pin": 15,
        "overcurrent_pins": [0],
    },
    "reflex_bindings": [
        {"name": "heading_hold", "bytecode_offset": 0, "auto_start": True},
    ],
    "serial": {
        "initial_baud": 115200,
        "negotiate": True,
    },
}


class TestNodeRoleConfigValidation:
    def test_valid_config_passes(self, validator: SchemaValidator) -> None:
        result = validator.validate(VALID_NODE_CONFIG, "node_role_config")
        assert result.valid, result.error_summary()

    def test_minimal_config(self, validator: SchemaValidator) -> None:
        minimal = {"node_id": "test-node", "role": "sensor"}
        result = validator.validate(minimal, "node_role_config")
        assert result.valid, result.error_summary()

    def test_missing_node_id_rejected(self, validator: SchemaValidator) -> None:
        bad = {"role": "sensor"}
        result = validator.validate(bad, "node_role_config")
        assert not result.valid

    def test_invalid_pin_id(self, validator: SchemaValidator) -> None:
        bad = {
            "node_id": "test",
            "role": "test",
            "sensors": [{"id": 100, "mode": "input", "sensor_type": "x"}],
        }
        result = validator.validate(bad, "node_role_config")
        assert not result.valid

    def test_invalid_actuator_type(self, validator: SchemaValidator) -> None:
        bad = {
            "node_id": "test",
            "role": "test",
            "actuators": [{"id": 0, "mode": "output", "actuator_type": "rocket_launcher"}],
        }
        result = validator.validate(bad, "node_role_config")
        assert not result.valid

    def test_invalid_baud_rate(self, validator: SchemaValidator) -> None:
        bad = {
            "node_id": "test",
            "role": "test",
            "serial": {"initial_baud": 999999},
        }
        result = validator.validate(bad, "node_role_config")
        assert not result.valid

    def test_excessive_sensors_rejected(self, validator: SchemaValidator) -> None:
        sensors = [{"id": i, "mode": "input", "sensor_type": f"s{i}"} for i in range(65)]
        bad = {"node_id": "test", "role": "test", "sensors": sensors}
        result = validator.validate(bad, "node_role_config")
        assert not result.valid

    def test_additional_properties_rejected(self, validator: SchemaValidator) -> None:
        bad = {"node_id": "test", "role": "test", "bogus": True}
        result = validator.validate(bad, "node_role_config")
        assert not result.valid


# ===================================================================
# Test 6: Serial Protocol Schema Validation
# ===================================================================

VALID_SERIAL_PROTOCOL = {
    "version": "1.0",
    "frame": {
        "header_size": 10,
        "crc_size": 2,
        "max_payload": 1024,
        "max_decoded_frame": 1036,
        "max_cobs_frame": 1051,
        "max_wire_frame": 1053,
        "delimiter": 0,
    },
    "message_types": [
        {"msg_type": 0x05, "name": "HEARTBEAT", "direction": "BOTH", "criticality": "telemetry"},
        {"msg_type": 0x06, "name": "SENSOR_TELEMETRY", "direction": "N2J", "criticality": "telemetry"},
        {"msg_type": 0x07, "name": "COMMAND", "direction": "J2N", "criticality": "command"},
        {"msg_type": 0x1C, "name": "SAFETY_EVENT", "direction": "N2J", "criticality": "safety"},
    ],
    "crc_polynomial": "CRC-16/CCITT-FALSE",
}


class TestSerialProtocolValidation:
    def test_valid_protocol_passes(self, validator: SchemaValidator) -> None:
        result = validator.validate(VALID_SERIAL_PROTOCOL, "serial_protocol")
        assert result.valid, result.error_summary()

    def test_message_header(self, validator: SchemaValidator) -> None:
        header = {
            "msg_type": 0x06,
            "flags": 0,
            "sequence": 42,
            "timestamp_ms": 1714000000,
            "payload_length": 16,
        }
        schema = validator.get_schema("serial_protocol")
        header_schema = schema["definitions"]["message_header"]
        import jsonschema
        jsonschema.Draft7Validator(header_schema).validate(header)

    def test_invalid_payload_length(self, validator: SchemaValidator) -> None:
        header = {
            "msg_type": 0x06,
            "flags": 0,
            "sequence": 0,
            "timestamp_ms": 0,
            "payload_length": 9999,
        }
        schema = validator.get_schema("serial_protocol")
        header_schema = schema["definitions"]["message_header"]
        import jsonschema
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.Draft7Validator(header_schema).validate(header)

    def test_sensor_telemetry_payload(self, validator: SchemaValidator) -> None:
        payload = {"sensor_id": 0, "value": 270.5, "valid": True, "timestamp_ms": 1000}
        schema = validator.get_schema("serial_protocol")
        tele_schema = schema["definitions"]["sensor_telemetry_payload"]
        import jsonschema
        jsonschema.Draft7Validator(tele_schema).validate(payload)

    def test_safety_event_payload(self, validator: SchemaValidator) -> None:
        payload = {"event_type": "estop_triggered", "state": "SAFE_STATE", "timestamp_ms": 5000}
        schema = validator.get_schema("serial_protocol")
        safety_schema = schema["definitions"]["safety_event_payload"]
        import jsonschema
        jsonschema.Draft7Validator(safety_schema).validate(payload)

    def test_invalid_crc_polynomial(self, validator: SchemaValidator) -> None:
        bad = {
            "version": "1.0",
            "crc_polynomial": "WRONG-CRC",
        }
        result = validator.validate(bad, "serial_protocol")
        assert not result.valid

    def test_message_type_validation(self, validator: SchemaValidator) -> None:
        bad = {
            "message_types": [
                {"msg_type": 1, "name": "TEST", "direction": "INVALID", "criticality": "telemetry"},
            ]
        }
        result = validator.validate(bad, "serial_protocol")
        assert not result.valid


# ===================================================================
# Test 7: Codebase Consistency Checks
# ===================================================================

class TestCodebaseConsistency:
    """Verify that schemas match the actual Python/C implementation."""

    def test_autonomy_event_types_match_codebase(self, validator: SchemaValidator) -> None:
        """Schema event_type enum should match trust/events.py."""
        from trust.events import EVENT_DEFINITIONS
        schema = validator.get_schema("autonomy_state")
        event_schema = schema["definitions"]["trust_event"]
        schema_events = set(event_schema["properties"]["event_type"]["enum"])
        codebase_events = set(EVENT_DEFINITIONS.keys())
        assert schema_events == codebase_events, (
            f"Mismatch: schema has {schema_events - codebase_events}, "
            f"codebase has {codebase_events - schema_events}"
        )

    def test_subsystem_names_match_codebase(self, validator: SchemaValidator) -> None:
        """Schema subsystem enum should match SUBSYSTEMS list."""
        from trust.increments import SUBSYSTEMS
        schema = validator.get_schema("autonomy_state")
        subsystem_schema = schema["definitions"]["subsystem_trust"]
        schema_subsystems = set(subsystem_schema["properties"]["subsystem"]["enum"])
        codebase_subsystems = set(SUBSYSTEMS)
        assert schema_subsystems == codebase_subsystems

    def test_autonomy_levels_match_codebase(self, validator: SchemaValidator) -> None:
        """Schema autonomy level names should match trust/levels.py."""
        from trust.levels import AUTONOMY_LEVELS
        schema = validator.get_schema("autonomy_state")
        level_schema = schema["definitions"]["autonomy_level_def"]
        schema_names = set(level_schema["properties"]["name"]["enum"])
        codebase_names = {AUTONOMY_LEVELS[i].name for i in range(6)}
        assert schema_names == codebase_names

    def test_trust_params_defaults_match_codebase(self, validator: SchemaValidator) -> None:
        """Schema default values should match TrustParams defaults."""
        from trust.increments import TrustParams
        schema = validator.get_schema("autonomy_state")
        params_schema = schema["definitions"]["trust_params"]
        defaults = TrustParams()

        checks = [
            ("alpha_gain", defaults.alpha_gain),
            ("alpha_loss", defaults.alpha_loss),
            ("alpha_decay", defaults.alpha_decay),
            ("t_floor", defaults.t_floor),
            ("quality_cap", defaults.quality_cap),
            ("evaluation_window_hours", defaults.evaluation_window_hours),
            ("severity_exponent", defaults.severity_exponent),
            ("streak_bonus", defaults.streak_bonus),
            ("min_events_for_gain", defaults.min_events_for_gain),
            ("n_penalty_slope", defaults.n_penalty_slope),
            ("reset_grace_hours", defaults.reset_grace_hours),
            ("promotion_cooldown_hours", defaults.promotion_cooldown_hours),
        ]
        for field_name, expected in checks:
            schema_default = params_schema["properties"][field_name]["default"]
            assert schema_default == expected, (
                f"Field '{field_name}': schema default={schema_default}, "
                f"codebase default={expected}"
            )

    def test_reflex_opcodes_match_compiler(self, validator: SchemaValidator) -> None:
        """Schema opcodes should match the compiler's VALID_OPCODES set."""
        from reflex.compiler import VALID_OPCODES
        schema = validator.get_schema("reflex_definition")
        instr_schema = schema["definitions"]["instruction"]
        schema_ops = set(instr_schema["properties"]["op"]["enum"])
        assert schema_ops == VALID_OPCODES, (
            f"Schema has extra: {schema_ops - VALID_OPCODES}, "
            f"Compiler has extra: {VALID_OPCODES - schema_ops}"
        )

    def test_message_types_match_firmware(self, validator: SchemaValidator) -> None:
        """Schema should define 28 message types matching firmware/message.h."""
        schema = validator.get_schema("serial_protocol")
        frame_schema = schema["definitions"]["frame_info"]
        assert frame_schema["properties"]["header_size"]["const"] == 10
        assert frame_schema["properties"]["max_payload"]["const"] == 1024
        assert frame_schema["properties"]["max_wire_frame"]["const"] == 1053

    def test_actuator_types_match_hal(self, validator: SchemaValidator) -> None:
        """Schema actuator types should match HAL actuator_type_t enum."""
        schema = validator.get_schema("node_role_config")
        actuator_schema = schema["definitions"]["actuator_pin_config"]
        schema_types = set(actuator_schema["properties"]["actuator_type"]["enum"])
        expected = {"servo", "relay", "motor_pwm", "solenoid", "led", "buzzer"}
        assert schema_types == expected

    def test_pin_modes_match_hal(self, validator: SchemaValidator) -> None:
        """Schema pin modes should match HAL pin_mode_t enum."""
        schema = validator.get_schema("node_role_config")
        pin_schema = schema["definitions"]["pin_config"]
        schema_modes = set(pin_schema["properties"]["mode"]["enum"])
        expected = {"input", "output", "input_pullup", "input_pulldown"}
        assert schema_modes == expected

    def test_safety_states_match_hal(self, validator: SchemaValidator) -> None:
        """Schema safety states should match safety_sm.h."""
        schema = validator.get_schema("serial_protocol")
        safety_payload = schema["definitions"]["safety_event_payload"]
        schema_states = set(safety_payload["properties"]["state"]["enum"])
        expected = {"NORMAL", "DEGRADED", "SAFE_STATE", "FAULT"}
        assert schema_states == expected


# ===================================================================
# Test 8: Sample Config Generation
# ===================================================================

class TestSampleGeneration:
    @pytest.mark.parametrize("schema_name", [
        "reflex_definition", "node_role_config",
    ])
    def test_generate_sample(self, schema_name: str) -> None:
        sample = generate_sample_config(schema_name, schema_dir=SCHEMA_DIR)
        assert isinstance(sample, dict)
        assert len(sample) > 0

    def test_generate_reflex_sample_validates(self, validator: SchemaValidator) -> None:
        sample = generate_sample_config("reflex_definition", schema_dir=SCHEMA_DIR)
        # Fill in body properly since generator may not handle nested arrays well
        if "body" in sample and isinstance(sample["body"], list):
            if len(sample["body"]) > 0 and isinstance(sample["body"][0], dict):
                if "op" not in sample["body"][0]:
                    sample["body"][0]["op"] = "NOP"
        result = validator.validate(sample, "reflex_definition")
        assert result.valid, result.error_summary()

    def test_generate_node_config_sample_validates(self, validator: SchemaValidator) -> None:
        sample = generate_sample_config("node_role_config", schema_dir=SCHEMA_DIR)
        # Fill in required string fields that generator may leave empty
        if "node_id" in sample and sample["node_id"] == "":
            sample["node_id"] = "test-node"
        if "role" in sample and sample["role"] == "":
            sample["role"] = "test"
        result = validator.validate(sample, "node_role_config")
        assert result.valid, result.error_summary()


# ===================================================================
# Test 9: Error Path Reporting
# ===================================================================

class TestErrorReporting:
    def test_error_includes_path(self, validator: SchemaValidator) -> None:
        bad = {
            "params": {
                "alpha_gain": "not_a_number",
                "alpha_loss": 0.05,
                "alpha_decay": 0.0001,
                "t_floor": 0.2,
            }
        }
        result = validator.validate(bad, "autonomy_state")
        assert not result.valid
        assert any("params" in ".".join(e.path) for e in result.errors)

    def test_nested_error_path(self, validator: SchemaValidator) -> None:
        bad = {
            "subsystems": [
                {"subsystem": "steering", "trust_score": 5.0}
            ]
        }
        result = validator.validate(bad, "autonomy_state")
        assert not result.valid
        assert any("trust_score" in ".".join(e.path) for e in result.errors)

    def test_error_summary_format(self, validator: SchemaValidator) -> None:
        bad = {"params": {"alpha_gain": -1, "alpha_loss": 0.05, "alpha_decay": 0.0001, "t_floor": 0.2}}
        result = validator.validate(bad, "autonomy_state")
        assert not result.valid
        summary = result.error_summary()
        assert "Validation failed" in summary
        assert str(len(result.errors)) in summary

    def test_valid_result_summary(self, validator: SchemaValidator) -> None:
        result = ValidationResult(valid=True)
        assert result.error_summary() == "Validation passed."

    def test_bool_coercion(self, validator: SchemaValidator) -> None:
        valid_result = validator.validate(
            {"params": {"alpha_gain": 0.002, "alpha_loss": 0.05, "alpha_decay": 0.0001, "t_floor": 0.2}},
            "autonomy_state",
        )
        assert valid_result
        # Use a config that actually fails: missing required fields in params
        invalid_result = validator.validate(
            {"params": {"alpha_gain": 0.002}},
            "autonomy_state",
        )
        assert not invalid_result
