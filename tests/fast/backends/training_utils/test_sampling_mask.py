from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.loss_hub import logit_processors
from miles.backends.training_utils.loss_hub.math_utils import _calculate_log_probs_and_entropy_true_on_policy
from miles.backends.training_utils.sampling_mask import (
    build_local_sampling_mask,
    get_rollout_sampling_mask,
)


def test_build_local_sampling_mask_selects_original_response_rows_and_tp_shard():
    logits = torch.zeros(2, 4)
    mask = build_local_sampling_mask(
        logits,
        sampling_mask_ids=[1, 3, 4, 2, 5, 6, 7],
        sampling_mask_offsets=[0, 2, 4, 7],
        response_indices=[2, 0],
        response_length=3,
        tp_rank=1,
    )

    torch.testing.assert_close(
        mask,
        torch.tensor(
            [
                [False, True, True, True],
                [False, False, False, False],
            ]
        ),
    )


def test_build_local_sampling_mask_requires_one_offset_per_response_token():
    with pytest.raises(ValueError, match=r"offsets length 3 != response length \+ 1 \(4\)"):
        build_local_sampling_mask(
            torch.zeros(1, 4),
            sampling_mask_ids=[0, 1],
            sampling_mask_offsets=[0, 1, 2],
            response_indices=[0],
            response_length=3,
            tp_rank=0,
        )


def test_true_on_policy_masks_logprob_but_keeps_full_vocab_entropy():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]], requires_grad=True)
    tokens = torch.tensor([0])
    sampling_mask = torch.tensor([[True, False, True, False]])

    log_probs, entropy = _calculate_log_probs_and_entropy_true_on_policy(
        logits,
        tokens,
        None,
        with_entropy=True,
        sampling_mask=sampling_mask,
    )

    expected_masked_logprob = torch.log_softmax(logits.masked_fill(~sampling_mask, float("-inf")), dim=-1)[0, 0]
    full_log_probs = torch.log_softmax(logits, dim=-1)
    expected_entropy = -(full_log_probs.exp() * full_log_probs).sum(dim=-1)
    torch.testing.assert_close(log_probs, expected_masked_logprob.unsqueeze(0))
    torch.testing.assert_close(entropy, expected_entropy)


def test_get_log_probs_and_entropy_applies_per_response_sampling_support(monkeypatch):
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0, group=None),
        cp=SimpleNamespace(rank=0, size=1),
    )
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: parallel_state)
    args = SimpleNamespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        true_on_policy_mode=True,
        bf16=False,
        fp16=False,
        log_probs_chunk_size=-1,
        vocab_size=4,
        allgather_cp=False,
    )
    logits = torch.tensor(
        [
            [
                [2.0, 1.0, 0.0, -1.0],
                [-1.0, 0.0, 1.0, 2.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )

    result = logit_processors.get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=[torch.tensor([2, 0, 3])],
        total_lengths=[3],
        response_lengths=[2],
        rollout_sampling_mask_ids=[[0, 2, 1, 3]],
        rollout_sampling_mask_offsets=[[0, 2, 4]],
    )

    expected = torch.stack(
        [
            torch.log_softmax(logits[0, 0, [0, 2]], dim=-1)[0],
            torch.log_softmax(logits[0, 1, [1, 3]], dim=-1)[1],
        ]
    )
    torch.testing.assert_close(result["log_probs"][0], expected)


def test_get_rollout_sampling_mask_fails_when_required_support_is_missing():
    with pytest.raises(ValueError, match="top-p actor scoring requires"):
        get_rollout_sampling_mask({})
