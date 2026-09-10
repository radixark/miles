"""Exact per-LoRA step-duration summaries for a single pressure-test trial."""

import json
import math
import statistics
from pathlib import Path


def summarize_seconds(values):
    ordered = sorted(values)
    assert ordered and all(math.isfinite(value) and value >= 0 for value in ordered)

    def percentile(fraction):
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "steps": len(ordered),
        "mean_seconds": statistics.mean(ordered),
        "p50_seconds": percentile(0.50),
        "p90_seconds": percentile(0.90),
        "p95_seconds": percentile(0.95),
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
        "std_seconds": statistics.pstdev(ordered),
    }


class StepTimings:
    def __init__(self, directory: Path, *, adapter: str, model_id: str, phase: str, clients: int):
        self.directory = directory
        self.adapter = adapter
        self.metadata = {"adapter": adapter, "model_id": model_id, "phase": phase, "clients": clients}
        self.seconds = []
        # A restarted trial must use a fresh directory, not mix model identities.
        (directory / f"{adapter}-step-times.jsonl").touch(exist_ok=False)

    def add(self, step, stages):
        assert step == len(self.seconds) + 1
        self.seconds.append(stages["step_seconds"])
        record = {**self.metadata, "step": step, **stages}
        with (self.directory / f"{self.adapter}-step-times.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        summary = {**self.metadata, **summarize_seconds(self.seconds)}
        path = self.directory / f"{self.adapter}-timing-summary.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
        return summary
