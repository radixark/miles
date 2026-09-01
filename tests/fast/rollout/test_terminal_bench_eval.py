from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import cast

import pytest

from examples.experimental.eval.parallel_sft.parallel_command_eval import _read_metrics
from examples.experimental.eval.parallel_sft.terminal_bench_eval import (
    DEFAULT_TASK_LIST,
    Args,
    build_payload,
    load_task_ids,
    summarize,
)


def test_bundled_terminal_bench_task_list_is_the_verified_89_task_set() -> None:
    assert len(load_task_ids(DEFAULT_TASK_LIST)) == 89
    assert hashlib.sha256(DEFAULT_TASK_LIST.read_bytes()).hexdigest() == ("3778b86071c74c8a342222cfff41089916adb5c7338d1afa69f42c9cae21fe3e")


def test_load_task_ids_rejects_duplicates(tmp_path) -> None:
    task_list = tmp_path / "tasks.txt"
    task_list.write_text("task-a\ntask-a\n")

    with pytest.raises(ValueError, match="duplicate"):
        load_task_ids(task_list)


def test_build_payload_preserves_qwen_and_terminus_sampling_settings() -> None:
    args = cast(
        Args,
        SimpleNamespace(
            agent_name="terminus-2",
            base_url="http://checkpoint-router:30000/v1",
            model="openai/qwen-checkpoint",
            max_seq_len=262144,
            max_tokens=131072,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
        ),
    )

    payload = build_payload(args, "build-pmars", "dummy")

    assert payload["model"] == "openai/qwen-checkpoint"
    assert payload["max_seq_len"] == 262144
    assert payload["sampling_params"] == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 131072,
        "api_key": "dummy",
    }


def test_summary_accuracy_uses_all_scheduled_trials_and_feeds_command_consumer(tmp_path) -> None:
    output_path = tmp_path / "tb21.jsonl"
    rows = [
        {"instance_id": "task-a", "trial_idx": 0, "status_code": 200, "reward": 1.0},
        {"instance_id": "task-a", "trial_idx": 1, "status_code": 200, "reward": 0.0},
        {
            "instance_id": "task-b",
            "trial_idx": 0,
            "status_code": 0,
            "error": "timeout",
        },
    ]
    output_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    summary = summarize(output_path, ["task-a", "task-b"], n_trials=2)

    assert summary["metrics"] == {
        "tasks_total": 2,
        "trials_per_task": 2,
        "trials_total": 4,
        "completed": 3,
        "graded": 2,
        "passes": 1,
        "failures": 1,
        "errors": 2,
        "missing": 1,
        "accuracy": 0.25,
        "request_success_rate": 0.5,
        "timeouts": 1,
        "sequence_length_exceeded": 0,
        "time_limit_exceeded": 0,
        "problems_any_correct": 1,
        "problem_any_correct_rate": 0.5,
        "unexpected_records": 0,
    }
    assert summary["rewards"] == [1.0, 0.0, None, None]
    assert summary["per_task"]["task-a"]["accuracy"] == 0.5
    assert summary["per_task"]["task-b"]["accuracy"] == 0.0

    summary_path = tmp_path / "tb21_summary.json"
    summary_path.write_text(json.dumps(summary))
    _payload, metrics, rewards = _read_metrics(summary_path)
    assert metrics["accuracy"] == 0.25
    assert rewards == [1.0, 0.0, None, None]
