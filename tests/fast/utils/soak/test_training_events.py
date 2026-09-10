import json
from pathlib import Path

import pytest
from tests.utils.soak.training_events import _read_events


def test_live_eval_read_preserves_the_evaluated_rollout_and_start_identity(tmp_path: Path) -> None:
    """Evaluation evidence must survive capture outside the trainer-controller event files."""
    event = {
        "type": "metric",
        "timestamp": "2026-09-11T00:00:02Z",
        "evaluation_started_at": "2026-09-11T00:00:01Z",
        "source": {"component": "rollout_executor"},
        "rollout_id": 12,
        "metrics": {"eval/gsm8k": 0.6},
    }
    (tmp_path / "rollout_executor.jsonl").write_text(json.dumps(event) + "\n")
    (observed,) = _read_events(tmp_path)
    assert observed.rollout_id == 12
    assert observed.evaluation_started_at < observed.timestamp
    assert observed.metrics == {"eval/gsm8k": 0.6}


def test_live_read_defers_an_unfinished_record_but_keeps_completed_progress(tmp_path: Path) -> None:
    """A concurrent append becomes visible only after its terminating newline is written."""
    path = tmp_path / "trainer_controller_actor.jsonl"
    first = {
        "type": "train_group_step_end",
        "timestamp": "2026-09-11T00:00:00Z",
        "source": {"component": "trainer_controller", "trainer_id": "actor", "model_id": None},
        "rollout_id": 3,
        "cell_outcomes": {"0": ["normal"]},
    }
    second = {**first, "rollout_id": 4}
    serialized = json.dumps(second)
    path.write_text(json.dumps(first) + "\n" + serialized[:20])
    assert [event.rollout_id for event in _read_events(tmp_path)] == [3]

    with path.open("a") as stream:
        stream.write(serialized[20:] + "\n")
    assert [event.rollout_id for event in _read_events(tmp_path)] == [3, 4]


def test_complete_malformed_training_record_fails_instead_of_hiding_missing_evidence(tmp_path: Path) -> None:
    """A completed corrupt record cannot be silently discarded from recovery evidence."""
    (tmp_path / "trainer_controller_actor.jsonl").write_text('{"type":\n')
    with pytest.raises(json.JSONDecodeError):
        _read_events(tmp_path)


def test_missing_training_directory_is_a_read_failure(tmp_path: Path) -> None:
    """An unavailable event directory is distinct from a successful empty observation."""
    with pytest.raises(FileNotFoundError):
        _read_events(tmp_path / "missing")
