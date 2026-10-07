"""Forecast conversion preserves evidence and keeps resolutions out of inputs."""

import json
from pathlib import Path

import pytest

from examples.clef.data import read_examples
from examples.clef.prepare_forecast_validation import convert_forecast


def test_forecast_conversion_preserves_input_and_aligns_target(tmp_path: Path) -> None:
    row = {"_evaluation": {"catalog_id": 48, "run_id": "48:forecast:1"},
           "state": {"forecast_as_of": "2024-01-01", "event": {"question": "Will it rain?"}},
           "questions": {"answer": {"type": "choice", "instructions": "Estimate probability.", "criteria": {"yes": "Rain", "no": "No rain"}}},
           "expected": {"answer": "no"}, "metadata": {"resolution_source": "frozen-file"}}
    converted = convert_forecast(row)
    assert converted["record"]["state"] is row["state"]
    assert converted["record"]["questions"] is row["questions"]
    assert "expected" not in converted["record"] and "targets" not in converted["record"]
    path = tmp_path / "forecast.jsonl"
    path.write_text(json.dumps(converted) + "\n")
    example = read_examples(path)[0]
    assert example.targets == {"answer": {"yes": 0.0, "no": 1.0}}
    row["expected"]["answer"] = "unresolved"
    with pytest.raises(ValueError, match="unresolved"):
        convert_forecast(row)
