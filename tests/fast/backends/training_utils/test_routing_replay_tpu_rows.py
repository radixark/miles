import logging

import pytest
import torch

from miles.backends.training_utils.replay import routing_replay
from miles.backends.training_utils.replay.routing_replay import check_tpu_rollout_routed_experts


@pytest.fixture(autouse=True)
def _rank_zero(monkeypatch):
    monkeypatch.setattr(routing_replay.dist, "get_rank", lambda: 0)


def _sample(prefill_rows: int, decode_rows: int, *, prefill_value: int = -1) -> torch.Tensor:
    """`[num_tokens - 1, num_layers=2, topk=2]` as a TPU engine with sglang-torchtpu#1580 reports it."""
    prefill = torch.full((prefill_rows, 2, 2), prefill_value, dtype=torch.int32)
    decode = torch.tensor([[[1, 3], [0, 2]]], dtype=torch.int32).expand(decode_rows, 2, 2)
    return torch.cat([prefill, decode])


def test_uncaptured_prefill_rows_pass_and_are_counted(caplog):
    with caplog.at_level(logging.INFO):
        check_tpu_rollout_routed_experts([_sample(5, 3), _sample(2, 4)])
    assert "7/14 rows carry no engine ids" in caplog.text


def test_an_engine_capturing_in_compiled_prefill_is_refused():
    """Prefill capture corrupts compiled TPU prefill programs; the run must not train on its rollouts."""
    with pytest.raises(RuntimeError, match="sglang-torchtpu#1580"):
        check_tpu_rollout_routed_experts([_sample(5, 3, prefill_value=4)])


def test_a_one_token_response_decodes_no_rows_and_passes():
    """Its rows are all prefill rows, so all -1, correctly."""
    check_tpu_rollout_routed_experts([_sample(6, 0), _sample(2, 4), _sample(0, 0)])


def test_a_batch_without_any_routed_experts_is_refused():
    with pytest.raises(RuntimeError, match="no routed experts for any row"):
        check_tpu_rollout_routed_experts([_sample(6, 0), _sample(3, 0)])
