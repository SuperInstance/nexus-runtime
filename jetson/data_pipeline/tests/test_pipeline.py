"""Tests for pipeline.py — PipelineStage, PipelineResult, DataPipeline."""

import time

import pytest

from jetson.data_pipeline.pipeline import DataPipeline, PipelineResult, PipelineStage


# ── PipelineStage dataclass ────────────────────────────────────

class TestPipelineStage:

    def test_create_minimal(self):
        s = PipelineStage(name="s1", process_fn=lambda x: x)
        assert s.name == "s1"
        assert s.parallel is False
        assert s.batch_size == 1

    def test_create_full(self):
        s = PipelineStage(name="s1", process_fn=lambda x: x, parallel=True, batch_size=10)
        assert s.parallel is True
        assert s.batch_size == 10


# ── PipelineResult dataclass ───────────────────────────────────

class TestPipelineResult:

    def test_create(self):
        r = PipelineResult(output=42, processing_time=0.1, records_processed=1)
        assert r.output == 42
        assert r.errors == []

    def test_with_errors(self):
        r = PipelineResult(output=None, processing_time=0.01,
                           records_processed=0, errors=["stage1: boom"])
        assert len(r.errors) == 1


# ── DataPipeline stage management ──────────────────────────────

class TestDataPipelineManagement:

    def test_add_stage(self):
        p = DataPipeline()
        p.add_stage(PipelineStage("s1", lambda x: x))
        assert p.stage_count == 1

    def test_add_multiple(self):
        p = DataPipeline()
        p.add_stage(PipelineStage("s1", lambda x: x))
        p.add_stage(PipelineStage("s2", lambda x: x))
        assert p.stage_count == 2

    def test_remove_stage(self):
        p = DataPipeline()
        p.add_stage(PipelineStage("s1", lambda x: x))
        assert p.remove_stage("s1") is True
        assert p.stage_count == 0

    def test_remove_nonexistent(self):
        p = DataPipeline()
        assert p.remove_stage("nope") is False

    def test_list_stages(self):
        p = DataPipeline()
        p.add_stage(PipelineStage("a", lambda x: x))
        p.add_stage(PipelineStage("b", lambda x: x))
        assert p.list_stages() == ["a", "b"]

    def test_set_parallel(self):
        p = DataPipeline()
        p.add_stage(PipelineStage("a", lambda x: x, parallel=True))
        p.add_stage(PipelineStage("b", lambda x: x))
        p.set_parallel(["a"])
        stats = p.get_pipeline_stats()
        assert "a" in stats["parallel_stages"]

    def test_set_parallel_removes_old(self):
        p = DataPipeline()
        p.set_parallel(["old"])
        p.set_parallel(["new"])
        assert p.get_pipeline_stats()["parallel_stages"] == ["new"]


# ── DataPipeline processing ────────────────────────────────────

class TestDataPipelineProcess:

    def test_process_single_stage(self):
        p = DataPipeline()
        p.add_stage(PipelineStage("double", lambda x: x * 2))
        result = p.process(5)
        assert result.output == 10
        assert result.records_processed == 1
        assert result.errors == []

    def test_process_chain(self):
        p = DataPipeline()
        p.add_stage(PipelineStage("add1", lambda x: x + 1))
        p.add_stage(PipelineStage("double", lambda x: x * 2))
        result = p.process(3)
        assert result.output == 8  # (3+1)*2

    def test_process_time_recorded(self):
        p = DataPipeline()
        p.add_stage(PipelineStage("noop", lambda x: x))
        result = p.process(1)
        assert result.processing_time >= 0

    def test_process_error_captured(self):
        def boom(_):
            raise ValueError("kaboom")

        p = DataPipeline()
        p.add_stage(PipelineStage("bad", boom))
        result = p.process(1)
        assert len(result.errors) == 1
        assert "kaboom" in result.errors[0]

    def test_process_empty_pipeline(self):
        p = DataPipeline()
        result = p.process(42)
        assert result.output == 42

    def test_process_dict_data(self):
        p = DataPipeline()
        p.add_stage(PipelineStage("add_key", lambda d: {**d, "processed": True}))
        result = p.process({"value": 10})
        assert result.output["processed"] is True
        assert result.output["value"] == 10


# ── DataPipeline stream processing ─────────────────────────────

class TestDataPipelineStream:

    def test_process_stream(self):
        p = DataPipeline()
        p.add_stage(PipelineStage("double", lambda x: x * 2))
        results = list(p.process_stream([1, 2, 3]))
        assert len(results) == 3
        assert results[0].output == 2
        assert results[1].output == 4
        assert results[2].output == 6

    def test_process_stream_empty(self):
        p = DataPipeline()
        p.add_stage(PipelineStage("noop", lambda x: x))
        results = list(p.process_stream([]))
        assert len(results) == 0


# ── DataPipeline batch processing ──────────────────────────────

class TestDataPipelineBatch:

    def test_process_batch(self):
        p = DataPipeline()
        p.add_stage(PipelineStage("inc", lambda x: x + 1))
        results = p.process_batch([10, 20, 30])
        assert len(results) == 3
        assert results[0].output == 11

    def test_process_batch_empty(self):
        p = DataPipeline()
        assert p.process_batch([]) == []

    def test_process_parallel(self):
        p = DataPipeline()
        p.add_stage(PipelineStage("sq", lambda x: x ** 2))
        results = p.process_parallel([1, 2, 3, 4], max_workers=2)
        outputs = sorted(r.output for r in results)
        assert outputs == [1, 4, 9, 16]

    def test_process_parallel_preserves_order(self):
        p = DataPipeline()
        p.add_stage(PipelineStage("id", lambda x: x))
        results = p.process_parallel([10, 20, 30], max_workers=2)
        assert results[0].output == 10
        assert results[1].output == 20
        assert results[2].output == 30


# ── config-based construction ──────────────────────────────────

class TestCreateFromConfig:

    def test_basic_config(self):
        config = {
            "name": "test_pipe",
            "stages": [
                {"name": "s1", "fn": lambda x: x + 1},
                {"name": "s2", "fn": lambda x: x * 2},
            ],
        }
        p = DataPipeline.create_pipeline_from_config(config)
        assert p.name == "test_pipe"
        assert p.stage_count == 2
        result = p.process(3)
        assert result.output == 8

    def test_config_with_parallel(self):
        config = {
            "name": "par",
            "stages": [
                {"name": "s1", "fn": lambda x: x, "parallel": True},
            ],
        }
        p = DataPipeline.create_pipeline_from_config(config)
        stats = p.get_pipeline_stats()
        assert "s1" in stats["parallel_stages"]

    def test_config_default_name(self):
        p = DataPipeline.create_pipeline_from_config({"stages": []})
        assert p.name == "config_pipeline"

    def test_config_with_batch_size(self):
        config = {
            "stages": [
                {"name": "s1", "fn": lambda x: x, "batch_size": 50},
            ],
        }
        p = DataPipeline.create_pipeline_from_config(config)
        assert p.list_stages() == ["s1"]


# ── introspection ──────────────────────────────────────────────

class TestPipelineStats:

    def test_stats_structure(self):
        p = DataPipeline(name="stats_test")
        p.add_stage(PipelineStage("s1", lambda x: x))
        stats = p.get_pipeline_stats()
        assert stats["name"] == "stats_test"
        assert stats["stage_count"] == 1
        assert stats["stages"] == ["s1"]

    def test_stats_records_calls(self):
        p = DataPipeline()
        p.add_stage(PipelineStage("s1", lambda x: x))
        p.process(1)
        p.process(2)
        stats = p.get_pipeline_stats()
        assert stats["stats"]["s1"]["calls"] == 2

    def test_stats_records_errors(self):
        def boom(_):
            raise RuntimeError("err")

        p = DataPipeline()
        p.add_stage(PipelineStage("bad", boom))
        p.process(1)
        stats = p.get_pipeline_stats()
        assert stats["stats"]["bad"]["errors"] == 1


# ── reset ──────────────────────────────────────────────────────

class TestPipelineReset:

    def test_reset_clears_stages(self):
        p = DataPipeline()
        p.add_stage(PipelineStage("s1", lambda x: x))
        p.reset()
        assert p.stage_count == 0

    def test_reset_clears_stats(self):
        p = DataPipeline()
        p.add_stage(PipelineStage("s1", lambda x: x))
        p.process(1)
        p.reset()
        assert p.get_pipeline_stats()["stats"] == {}

    def test_reset_clears_parallel(self):
        p = DataPipeline()
        p.set_parallel(["a"])
        p.reset()
        assert p.get_pipeline_stats()["parallel_stages"] == []
