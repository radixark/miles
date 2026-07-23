import argparse
import json

import pytest
import torch

from miles.ray.rollout.debug_data import save_debug_rollout_data, trajectory_rows
from miles.utils.types import Sample


def make_sample(**kwargs):
    defaults = dict(index=7, group_index=2, prompt="rendered prompt", status=Sample.Status.COMPLETED)
    return Sample(**{**defaults, **kwargs})


MESSAGES = [
    {"role": "user", "content": "q"},
    {"role": "assistant", "content": "a"},
    {"role": "tool", "name": "get_year", "content": "2026"},
    {"role": "assistant", "content": "done"},
]


def test_rows_carry_the_raw_conversation():
    sample = make_sample(reward=1.0, metadata={"messages": MESSAGES})
    (row,) = trajectory_rows([sample])
    assert row["sample_index"] == 7 and row["group_index"] == 2
    assert row["reward"] == 1.0 and row["prompt"] == "rendered prompt"
    assert row["messages"] == MESSAGES


def test_rows_skip_samples_without_a_conversation():
    with_messages = make_sample(index=1, metadata={"messages": MESSAGES})
    without = make_sample(index=2)
    rows = trajectory_rows([without, with_messages])
    assert [row["sample_index"] for row in rows] == [1]


def test_non_numeric_reward_becomes_null():
    sample = make_sample(reward={"score": 1}, metadata={"messages": MESSAGES})
    (row,) = trajectory_rows([sample])
    assert row["reward"] is None


@pytest.fixture
def dump_args(tmp_path):
    return argparse.Namespace(
        save_debug_rollout_data=str(tmp_path / "rollout_data" / "{rollout_id}.pt"),
        save_debug_trajectory_data=str(tmp_path / "trajectory" / "{rollout_id}.jsonl"),
    )


def test_save_writes_sidecar_and_strips_messages_from_pt(dump_args, tmp_path):
    samples = [make_sample(index=i, metadata={"messages": MESSAGES}) for i in range(2)]
    save_debug_rollout_data(dump_args, samples, 5, evaluation=False)
    lines = (tmp_path / "trajectory" / "5.jsonl").read_text().splitlines()
    assert [json.loads(line)["sample_index"] for line in lines] == [0, 1]
    saved = torch.load(tmp_path / "rollout_data" / "5.pt", weights_only=False)
    assert all("messages" not in s["metadata"] for s in saved["samples"])


def test_save_eval_flattens_datasets(dump_args, tmp_path):
    data = {"math": {"samples": [make_sample(metadata={"messages": MESSAGES})]}}
    save_debug_rollout_data(dump_args, data, 1, evaluation=True)
    (line,) = (tmp_path / "trajectory" / "eval_1.jsonl").read_text().splitlines()
    assert json.loads(line)["messages"] == MESSAGES


def test_no_conversations_means_no_file(dump_args, tmp_path):
    save_debug_rollout_data(dump_args, [make_sample()], 9, evaluation=False)
    assert (tmp_path / "rollout_data" / "9.pt").exists()
    assert not (tmp_path / "trajectory").exists()


def test_messages_stay_in_pt_when_sidecar_disabled(dump_args, tmp_path):
    dump_args.save_debug_trajectory_data = None
    save_debug_rollout_data(dump_args, [make_sample(metadata={"messages": MESSAGES})], 8, evaluation=False)
    saved = torch.load(tmp_path / "rollout_data" / "8.pt", weights_only=False)
    assert saved["samples"][0]["metadata"]["messages"] == MESSAGES
