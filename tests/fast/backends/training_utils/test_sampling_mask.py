from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils import cp_utils
from miles.backends.training_utils.loss_hub import logit_processors
from miles.backends.training_utils.loss_hub.math_utils import _calculate_log_probs_and_entropy_true_on_policy
from miles.backends.training_utils.sampling_mask import build_local_sampling_mask
from miles.utils.sampling_mask import RolloutSamplingMask


def test_build_local_sampling_mask_selects_original_response_rows_and_tp_shard():
    logits = torch.zeros(2, 4)
    mask = build_local_sampling_mask(
        logits,
        sampling_mask=RolloutSamplingMask.from_mask_list([[1, 3], [4, 2], [5, 6, 7]]),
        response_indices=[2, 0],
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


def test_build_local_sampling_mask_rejects_out_of_range_response_index():
    with pytest.raises(ValueError, match=r"response indices must be in \[0, 1\)"):
        build_local_sampling_mask(
            torch.zeros(1, 4),
            sampling_mask=RolloutSamplingMask.from_mask_list([[0]]),
            response_indices=[1],
            tp_rank=0,
        )


def test_build_local_sampling_mask_rejects_row_misalignment():
    with pytest.raises(ValueError, match="sampling-mask rows must align with logits: indices=1, logits=2"):
        build_local_sampling_mask(
            torch.zeros(2, 4),
            sampling_mask=RolloutSamplingMask.from_mask_list([[0], [1]]),
            response_indices=[0],
            tp_rank=0,
        )


def test_build_local_sampling_mask_skips_selection_for_empty_local_rows(monkeypatch):
    def unexpected_selection(*args, **kwargs):
        raise AssertionError("empty local rows must not select sampling-mask ids")

    monkeypatch.setattr(RolloutSamplingMask, "_select_masks", unexpected_selection)

    mask = build_local_sampling_mask(
        torch.zeros(0, 4),
        sampling_mask=RolloutSamplingMask.from_mask_list([[0]]),
        response_indices=range(0),
        tp_rank=0,
    )

    assert mask.shape == (0, 4)
    assert mask.dtype == torch.bool


@pytest.mark.parametrize(
    "response_indices",
    [torch.tensor(0), torch.empty(0, dtype=torch.float32), torch.empty((0, 1), dtype=torch.long)],
)
def test_build_local_sampling_mask_validates_malformed_tensor_indices(response_indices):
    with pytest.raises(ValueError, match="must be one-dimensional integers"):
        build_local_sampling_mask(
            torch.zeros(0, 4),
            sampling_mask=RolloutSamplingMask.from_mask_list([[0]]),
            response_indices=response_indices,
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
        rollout_sampling_mask=[RolloutSamplingMask.from_mask_list([[0, 2], [1, 3]])],
    )

    expected = torch.stack(
        [
            torch.log_softmax(logits[0, 0, [0, 2]], dim=-1)[0],
            torch.log_softmax(logits[0, 1, [1, 3]], dim=-1)[1],
        ]
    )
    torch.testing.assert_close(result["log_probs"][0], expected)


def test_get_log_probs_and_entropy_rejects_mask_shorter_than_response(monkeypatch):
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0, group=None),
        cp=SimpleNamespace(rank=0, size=1),
    )
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: parallel_state)
    args = SimpleNamespace(qkv_format="thd", rollout_temperature=1.0, true_on_policy_mode=False, allgather_cp=False)

    with pytest.raises(ValueError, match="sampling-mask length 1 != response length 2"):
        logit_processors.get_log_probs_and_entropy(
            torch.zeros(1, 3, 4),
            args=args,
            unconcat_tokens=[torch.tensor([2, 0, 3])],
            total_lengths=[3],
            response_lengths=[2],
            rollout_sampling_mask=[RolloutSamplingMask.from_mask_list([[0]])],
        )


@pytest.mark.parametrize(("cp_rank", "expected_indices"), [(0, [0, 1]), (1, [2, 3])])
def test_allgather_cp_response_rows_keep_global_response_indices(monkeypatch, cp_rank, expected_indices):
    parallel_state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=2))
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: parallel_state)
    args = SimpleNamespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        true_on_policy_mode=False,
        allgather_cp=True,
    )

    response_chunks = list(
        logit_processors._iter_response_chunks(
            torch.zeros(1, 3, 4),
            args=args,
            unconcat_tokens=[torch.arange(6)],
            total_lengths=[6],
            response_lengths=[4],
            include_response_indices=True,
        )
    )
    logits_chunk, tokens_chunk, response_indices = response_chunks[0]

    assert list(response_indices) == expected_indices
    assert tokens_chunk.tolist() == [2 + index for index in expected_indices]
    assert logits_chunk.size(0) == len(expected_indices)


@pytest.mark.parametrize(("cp_rank", "expected_indices"), [(0, [4]), (1, [0, 1, 2, 3])])
def test_zigzag_cp_response_rows_keep_global_response_indices(monkeypatch, cp_rank, expected_indices):
    parallel_state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=2))
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: parallel_state)
    args = SimpleNamespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        true_on_policy_mode=False,
        allgather_cp=False,
    )

    response_chunks = list(
        logit_processors._iter_response_chunks(
            torch.zeros(1, 4, 4),
            args=args,
            unconcat_tokens=[torch.arange(8)],
            total_lengths=[8],
            response_lengths=[5],
            include_response_indices=True,
        )
    )
    logits_chunk, tokens_chunk, response_indices = response_chunks[0]

    assert list(response_indices) == expected_indices
    assert tokens_chunk.tolist() == [3 + index for index in expected_indices]
    assert logits_chunk.size(0) == len(expected_indices)


def test_zigzag_cp_log_probs_processes_discontiguous_parts_separately(monkeypatch):
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0, group=None),
        cp=SimpleNamespace(rank=1, size=2),
    )
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: parallel_state)
    args = SimpleNamespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        true_on_policy_mode=False,
        allgather_cp=False,
        log_probs_chunk_size=-1,
    )
    processed_shapes = []

    def fake_calculate(logits, tokens, *_args, **_kwargs):
        processed_shapes.append(tuple(logits.shape))
        values = tokens.to(torch.float32)
        return values, values + 10

    monkeypatch.setattr(logit_processors, "calculate_log_probs_and_entropy", fake_calculate)

    result = logit_processors.get_log_probs_and_entropy(
        torch.arange(32, dtype=torch.float32).reshape(1, 4, 8),
        args=args,
        unconcat_tokens=[torch.arange(8)],
        total_lengths=[8],
        response_lengths=[5],
        with_entropy=True,
    )

    assert processed_shapes == [(2, 8), (2, 8)]
    torch.testing.assert_close(result["log_probs"][0], torch.tensor([3.0, 4.0, 5.0, 6.0]))
    torch.testing.assert_close(result["entropy"][0], torch.tensor([13.0, 14.0, 15.0, 16.0]))
