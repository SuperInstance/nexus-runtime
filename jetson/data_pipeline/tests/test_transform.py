"""Tests for transform.py — Transform and TransformPipeline."""

import pytest

from jetson.data_pipeline.transform import Transform, TransformPipeline


# ── Transform dataclass ────────────────────────────────────────

class TestTransform:

    def test_create_minimal(self):
        t = Transform(name="double", transform_fn=lambda x: x * 2)
        assert t.name == "double"
        assert t.input_schema is None
        assert t.output_schema is None

    def test_create_with_schemas(self):
        t = Transform(
            name="add1", transform_fn=lambda x: x + 1,
            input_schema={"type": int},
            output_schema={"type": int},
        )
        assert t.input_schema == {"type": int}
        assert t.output_schema == {"type": int}

    def test_call_transform_fn(self):
        t = Transform(name="neg", transform_fn=lambda x: -x)
        assert t.transform_fn(5) == -5


# ── TransformPipeline management ───────────────────────────────

class TestTransformPipelineManagement:

    def test_add_transform(self):
        tp = TransformPipeline()
        tp.add_transform(Transform("a", lambda x: x))
        assert tp.transform_count == 1

    def test_add_multiple(self):
        tp = TransformPipeline()
        tp.add_transform(Transform("a", lambda x: x))
        tp.add_transform(Transform("b", lambda x: x))
        assert tp.transform_count == 2

    def test_remove_transform(self):
        tp = TransformPipeline()
        tp.add_transform(Transform("a", lambda x: x))
        assert tp.remove_transform("a") is True
        assert tp.transform_count == 0

    def test_remove_nonexistent(self):
        tp = TransformPipeline()
        assert tp.remove_transform("nope") is False

    def test_remove_preserves_others(self):
        tp = TransformPipeline()
        tp.add_transform(Transform("a", lambda x: x + 1))
        tp.add_transform(Transform("b", lambda x: x * 2))
        tp.remove_transform("a")
        assert tp.list_transforms() == ["b"]

    def test_list_transforms(self):
        tp = TransformPipeline()
        tp.add_transform(Transform("x", lambda x: x))
        tp.add_transform(Transform("y", lambda x: x))
        assert tp.list_transforms() == ["x", "y"]


# ── apply ──────────────────────────────────────────────────────

class TestTransformPipelineApply:

    def test_apply_single(self):
        tp = TransformPipeline()
        tp.add_transform(Transform("double", lambda x: x * 2))
        assert tp.apply(5) == 10

    def test_apply_chain(self):
        tp = TransformPipeline()
        tp.add_transform(Transform("double", lambda x: x * 2))
        tp.add_transform(Transform("add10", lambda x: x + 10))
        assert tp.apply(3) == 16  # (3*2)+10

    def test_apply_string_transform(self):
        tp = TransformPipeline()
        tp.add_transform(Transform("upper", lambda s: s.upper()))
        assert tp.apply("hello") == "HELLO"

    def test_apply_empty_pipeline(self):
        tp = TransformPipeline()
        assert tp.apply(42) == 42


class TestTransformPipelineBatch:

    def test_apply_batch(self):
        tp = TransformPipeline()
        tp.add_transform(Transform("sq", lambda x: x ** 2))
        result = tp.apply_batch([1, 2, 3, 4])
        assert result == [1, 4, 9, 16]

    def test_apply_batch_empty(self):
        tp = TransformPipeline()
        assert tp.apply_batch([]) == []


# ── compose ────────────────────────────────────────────────────

class TestCompose:

    def test_compose_two(self):
        t1 = Transform("a", lambda x: x + 1)
        t2 = Transform("b", lambda x: x * 2)
        fn = TransformPipeline.compose([t1, t2])
        assert fn(3) == 8  # (3+1)*2

    def test_compose_empty(self):
        fn = TransformPipeline.compose([])
        assert fn(42) == 42

    def test_compose_three(self):
        t1 = Transform("inc", lambda x: x + 1)
        t2 = Transform("double", lambda x: x * 2)
        t3 = Transform("neg", lambda x: -x)
        fn = TransformPipeline.compose([t1, t2, t3])
        assert fn(5) == -12  # (5+1)*2 = 12, negated


# ── schema validation ──────────────────────────────────────────

class TestSchemaValidation:

    def test_type_match_int(self):
        tp = TransformPipeline()
        assert tp.validate_schema(42, {"type": int}) is True

    def test_type_mismatch(self):
        tp = TransformPipeline()
        assert tp.validate_schema("42", {"type": int}) is False

    def test_type_match_str(self):
        tp = TransformPipeline()
        assert tp.validate_schema("hi", {"type": str}) is True

    def test_type_match_float(self):
        tp = TransformPipeline()
        assert tp.validate_schema(3.14, {"type": float}) is True

    def test_required_keys_present(self):
        tp = TransformPipeline()
        assert tp.validate_schema({"a": 1, "b": 2},
                                  {"required_keys": ["a", "b"]}) is True

    def test_required_keys_missing(self):
        tp = TransformPipeline()
        assert tp.validate_schema({"a": 1},
                                  {"required_keys": ["a", "b"]}) is False

    def test_min_value_pass(self):
        tp = TransformPipeline()
        assert tp.validate_schema(5, {"min_value": 0}) is True

    def test_min_value_fail(self):
        tp = TransformPipeline()
        assert tp.validate_schema(-1, {"min_value": 0}) is False

    def test_max_value_pass(self):
        tp = TransformPipeline()
        assert tp.validate_schema(5, {"max_value": 10}) is True

    def test_max_value_fail(self):
        tp = TransformPipeline()
        assert tp.validate_schema(15, {"max_value": 10}) is False

    def test_range_both(self):
        tp = TransformPipeline()
        assert tp.validate_schema(5, {"min_value": 0, "max_value": 10}) is True
        assert tp.validate_schema(-1, {"min_value": 0, "max_value": 10}) is False
        assert tp.validate_schema(11, {"min_value": 0, "max_value": 10}) is False

    def test_allow_none_true(self):
        tp = TransformPipeline()
        assert tp.validate_schema(None, {"allow_none": True}) is True

    def test_allow_none_false(self):
        tp = TransformPipeline()
        assert tp.validate_schema(None, {"allow_none": False}) is False

    def test_no_constraints_always_valid(self):
        tp = TransformPipeline()
        assert tp.validate_schema("anything", {}) is True


# ── schema registry ────────────────────────────────────────────

class TestSchemaRegistry:

    def test_register_and_get(self):
        tp = TransformPipeline()
        schema = {"type": int, "min_value": 0}
        tp.register_schema("positive_int", schema)
        assert tp.get_schema("positive_int") == schema

    def test_get_nonexistent(self):
        tp = TransformPipeline()
        assert tp.get_schema("nope") is None

    def test_register_overwrites(self):
        tp = TransformPipeline()
        tp.register_schema("x", {"v": 1})
        tp.register_schema("x", {"v": 2})
        assert tp.get_schema("x")["v"] == 2
