from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from miles.backends.fsdp_utils import actor as actor_module
from miles.backends.megatron_utils.ft.types import TrainStepOutcome, TrainStepOutput
from miles.backends.training_utils.torch_native import actor as base_module


@contextmanager
def _noop_timer(_name: str) -> Iterator[None]:
    yield


def test_fsdp_train_debug_rollout_only_returns_a_normal_output(monkeypatch):
    """A debug-rollout-only FSDP step trains nothing yet answers the driver with a NORMAL output."""
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor.args = Namespace(offload_train=False, debug_rollout_only=True)
    actor._heartbeat = Mock()
    actor._train_core = Mock()
    actor.wake_up = Mock()
    monkeypatch.setattr(
        base_module, "get_rollout_data", lambda _args, _ref, **_kwargs: ({"tokens": []}, nullcontext())
    )
    monkeypatch.setattr(base_module, "timer", _noop_timer)
    monkeypatch.setattr(base_module, "inverse_timer", _noop_timer)

    result = actor.train(3, object())

    assert result == TrainStepOutput(outcome=TrainStepOutcome.NORMAL)
    actor._train_core.assert_not_called()


@pytest.mark.parametrize(
    ("store_prefix", "use_sampling_support_replay", "expected"),
    [
        ("", True, True),
        ("ref_", True, False),
        ("", False, False),
    ],
)
def test_only_the_actor_pass_scores_replay_rollout_sampling_support(
    monkeypatch, store_prefix, use_sampling_support_replay, expected
):
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor.args = Namespace(
        use_sampling_support_replay=use_sampling_support_replay,
        data_pad_size_multiplier=1,
        qkv_format="thd",
    )
    actor.prof = SimpleNamespace(iterate_train_log_probs=lambda values: values)
    runner = SimpleNamespace(
        forward_only_step=lambda batches, compute: [compute(torch.zeros(1, 1, 4), batch) for batch in batches]
    )
    sampling_masks = [object()]
    captured = {}
    requested_keys = []

    def get_batch(_iterator, keys, *_args, **_kwargs):
        requested_keys.append(list(keys))
        return {
            "unconcat_tokens": [torch.tensor([0])],
            "total_lengths": [1],
            "response_lengths": [1],
            "max_seq_lens": None,
        }

    def get_log_probs_and_entropy(**kwargs):
        captured.update(kwargs)
        return {"log_probs": [torch.zeros(1)]}

    monkeypatch.setattr(base_module, "timer", _noop_timer)
    monkeypatch.setattr(base_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(base_module, "get_batch", get_batch)
    monkeypatch.setattr(base_module, "get_rollout_sampling_masks", lambda _batch: sampling_masks)
    monkeypatch.setattr(base_module, "get_log_probs_and_entropy", get_log_probs_and_entropy)
    monkeypatch.setattr(base_module, "aggregate_forward_results", lambda *_args, **_kwargs: {})

    actor._log_probs(runner, Mock(), [1], store_prefix=store_prefix)

    assert (captured["rollout_sampling_mask"] is sampling_masks) is expected
    assert ("rollout_sampling_mask_ids" in requested_keys[0]) is expected
