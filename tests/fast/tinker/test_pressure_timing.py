"""Known-duration statistics and per-model trial isolation."""

import json
import math

import pytest

from examples.multi_lora.pressure_timing import StepTimings, summarize_seconds


def test_duration_distribution():
    result = summarize_seconds([40.0, 10.0, 30.0, 20.0])
    assert result == pytest.approx(
        dict(
            steps=4,
            mean_seconds=25.0,
            p50_seconds=25.0,
            p90_seconds=37.0,
            p95_seconds=38.5,
            min_seconds=10.0,
            max_seconds=40.0,
            std_seconds=math.sqrt(125.0),
        )
    )
    first = summarize_seconds([5.0])
    assert first["p95_seconds"] == 5.0
    assert first["std_seconds"] == 0.0


def test_loras_do_not_share_samples_or_overwrite_previous_trials(tmp_path):
    a = StepTimings(tmp_path, adapter="lora_000", model_id="a", phase="continuous", clients=2)
    b = StepTimings(tmp_path, adapter="lora_001", model_id="b", phase="continuous", clients=2)
    a.add(1, {"step_seconds": 10.0, "publication_seconds": 1.0})
    b.add(1, {"step_seconds": 40.0, "publication_seconds": 2.0})
    a.add(2, {"step_seconds": 20.0, "publication_seconds": 3.0})
    summary = json.loads((tmp_path / "lora_000-timing-summary.json").read_text())
    assert summary["mean_seconds"] == 15.0
    assert summary["steps"] == 2
    assert summary["model_id"] == "a"
    assert summary["phase"] == "continuous"
    rows = [json.loads(line) for line in (tmp_path / "lora_001-step-times.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["step_seconds"] == 40.0
    with pytest.raises(FileExistsError):
        StepTimings(tmp_path, adapter="lora_000", model_id="new-a", phase="search", clients=3)
