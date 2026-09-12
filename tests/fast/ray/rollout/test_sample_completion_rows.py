from __future__ import annotations

from unittest.mock import Mock

from tests.fast.ray.rollout.conftest import make_args, make_sample

from miles.ray.rollout import metrics as metrics_module
from miles.ray.rollout.metrics import SAMPLE_COMPLETION_COLUMNS, log_rollout_data, sample_completion_rows
from miles.utils.types import Sample


def _sample(prompt, response: str, reward=1.0, status=Sample.Status.COMPLETED, group_index=0):
    s = make_sample(response_length=len(response), status=status)
    s.prompt = prompt
    s.response = response
    s.reward = reward
    s.group_index = group_index
    return s


def test_rows_take_one_sample_per_prompt_first():
    # log_rollout_data receives the flat sample list of the step; prompts are
    # told apart by group_index.
    samples = [
        _sample("p0", "a", group_index=0),
        _sample("p0", "b", group_index=0),
        _sample("p1", "c", group_index=1),
        _sample("p1", "d", group_index=1),
    ]

    columns, rows = sample_completion_rows(samples, limit=3)

    assert columns == SAMPLE_COMPLETION_COLUMNS
    assert [(row[0], row[1]) for row in rows] == [("p0", "a"), ("p1", "c"), ("p0", "b")]


def test_rows_are_strings_and_ints_only():
    chat = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    truncated = _sample(chat, "resp", reward={"score": 0.5}, status=Sample.Status.TRUNCATED)
    no_reward = _sample("q", "xy", reward=None, group_index=1)

    _, rows = sample_completion_rows([truncated, no_reward], limit=10)

    assert rows[0][0] == "system: be brief\nuser: hi"
    assert rows[0][2] == str({"score": 0.5})
    assert rows[0][3] == "TRUNCATED"
    assert rows[0][4] == 4
    # None and numbers become strings too, so a wandb.Table column never mixes types.
    assert rows[1][2] == "None"
    assert all(isinstance(cell, (str, int)) for row in rows for cell in row)


def test_empty_step_gives_no_rows():
    assert sample_completion_rows([], limit=4) == (SAMPLE_COMPLETION_COLUMNS, [])


def test_log_rollout_data_logs_a_table_only_when_enabled(monkeypatch):
    monkeypatch.setattr(metrics_module.tracking, "log", Mock())
    log_table = Mock()
    monkeypatch.setattr(metrics_module.tracking, "log_table", log_table)
    samples = [_sample("p0", "a"), _sample("p0", "b")]

    log_rollout_data(0, make_args(log_sample_completions=0), samples, {}, rollout_time=1.0)
    assert log_table.call_count == 0

    log_rollout_data(0, make_args(log_sample_completions=1), samples, {}, rollout_time=1.0)
    assert log_table.call_count == 1
    (name, columns, rows), _ = log_table.call_args
    assert name == "rollout/completions"
    assert columns == SAMPLE_COMPLETION_COLUMNS
    assert len(rows) == 1
