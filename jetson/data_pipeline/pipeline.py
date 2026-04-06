"""Pipeline orchestration — PipelineStage, PipelineResult, and DataPipeline."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Set


@dataclass
class PipelineStage:
    """A single processing stage in the pipeline."""
    name: str
    process_fn: Callable[[Any], Any]
    parallel: bool = False
    batch_size: int = 1


@dataclass
class PipelineResult:
    """Result of running a pipeline."""
    output: Any
    processing_time: float
    records_processed: int = 0
    errors: List[str] = field(default_factory=list)


class DataPipeline:
    """Orchestrates a sequence of processing stages."""

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._stages: List[PipelineStage] = []
        self._parallel_stages: Set[str] = set()
        self._stats: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    # ── stage management ───────────────────────────────────────

    def add_stage(self, stage: PipelineStage) -> None:
        """Append a stage to the pipeline."""
        self._stages.append(stage)

    def remove_stage(self, name: str) -> bool:
        """Remove a stage by name.  Returns True if found."""
        for i, s in enumerate(self._stages):
            if s.name == name:
                self._stages.pop(i)
                self._parallel_stages.discard(name)
                return True
        return False

    def set_parallel(self, stages: List[str]) -> None:
        """Mark named stages for parallel execution."""
        self._parallel_stages = set(stages)

    # ── processing ─────────────────────────────────────────────

    def process(self, data: Any) -> PipelineResult:
        """Run *data* through all stages sequentially (or parallel where marked)."""
        start = time.time()
        current = data
        records = 1
        errors: List[str] = []

        for stage in self._stages:
            try:
                current = stage.process_fn(current)
                self._record_stat(stage.name, "success")
            except Exception as exc:
                errors.append(f"{stage.name}: {exc}")
                self._record_stat(stage.name, "error", error=str(exc))

        elapsed = time.time() - start
        self._record_stat("__pipeline__", "run", time=elapsed)
        return PipelineResult(
            output=current, processing_time=elapsed,
            records_processed=records, errors=errors,
        )

    def process_stream(self, stream) -> Generator[PipelineResult, None, None]:
        """Process items from a stream, yielding a PipelineResult per item."""
        for item in stream:
            yield self.process(item)

    def process_batch(self, data_list: List[Any]) -> List[PipelineResult]:
        """Process a list of items, returning one result per item."""
        return [self.process(item) for item in data_list]

    # ── parallel batch processing ──────────────────────────────

    def process_parallel(self, data_list: List[Any], max_workers: int = 4) -> List[PipelineResult]:
        """Process a batch of items in parallel using a thread pool."""
        results: List[PipelineResult] = [None] * len(data_list)

        def _run(idx: int, item: Any) -> Tuple[int, PipelineResult]:
            return idx, self.process(item)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_run, i, item) for i, item in enumerate(data_list)]
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results

    # ── config-based construction ──────────────────────────────

    @staticmethod
    def create_pipeline_from_config(config: Dict[str, Any]) -> "DataPipeline":
        """Build a DataPipeline from a configuration dict.

        Expected format::

            {
                "name": "my_pipeline",
                "stages": [
                    {"name": "s1", "fn": <callable>, "parallel": True},
                    ...
                ]
            }
        """
        name = config.get("name", "config_pipeline")
        pipeline = DataPipeline(name=name)
        parallel_names: List[str] = []
        for sc in config.get("stages", []):
            stage = PipelineStage(
                name=sc["name"],
                process_fn=sc["fn"],
                parallel=sc.get("parallel", False),
                batch_size=sc.get("batch_size", 1),
            )
            pipeline.add_stage(stage)
            if stage.parallel:
                parallel_names.append(stage.name)
        if parallel_names:
            pipeline.set_parallel(parallel_names)
        return pipeline

    # ── introspection ──────────────────────────────────────────

    def get_pipeline_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "stage_count": len(self._stages),
            "stages": [s.name for s in self._stages],
            "parallel_stages": list(self._parallel_stages),
            "stats": dict(self._stats),
        }

    def reset(self) -> None:
        """Clear all stages and stats."""
        self._stages.clear()
        self._parallel_stages.clear()
        self._stats.clear()

    @property
    def stage_count(self) -> int:
        return len(self._stages)

    def list_stages(self) -> List[str]:
        return [s.name for s in self._stages]

    # ── internal ───────────────────────────────────────────────

    def _record_stat(self, stage_name: str, event: str, **kwargs) -> None:
        with self._lock:
            if stage_name not in self._stats:
                self._stats[stage_name] = {"calls": 0, "errors": 0, "last_event": None}
            entry = self._stats[stage_name]
            entry["calls"] += 1
            if event == "error":
                entry["errors"] += 1
            entry["last_event"] = {"event": event, **kwargs}
