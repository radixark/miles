from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils import cp_utils, loss as loss_utils
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


def _make_parallel_state(
    *,
    cp_size: int,
    cp_rank: int = 0,
    cp_comm_type: str | None = None,
    intra_dp_size: int = 2,
    intra_dp_cp_size: int = 8,
) -> ParallelState:
    return ParallelState(
        intra_dp=GroupInfo(rank=0, size=intra_dp_size, group=None),
        intra_dp_cp=GroupInfo(rank=0, size=intra_dp_cp_size, group=None),
        cp=GroupInfo(rank=cp_rank, size=cp_size, group=None),
        tp=GroupInfo(rank=0, size=1, group=None),
        cp_comm_type=cp_comm_type,
    )


def test_parallel_state_distinguishes_standard_and_ulysses_cp():
    standard = _make_parallel_state(cp_size=2, cp_comm_type="p2p")
    ulysses = _make_parallel_state(cp_size=2, cp_comm_type="a2a")
    no_cp = _make_parallel_state(cp_size=1, cp_comm_type="a2a")

    assert standard.cp_mode == "standard"
    assert not standard.is_ulysses_cp
    assert standard.effective_cp_size == 2

    assert ulysses.cp_mode == "ulysses"
    assert ulysses.is_ulysses_cp
    assert ulysses.effective_cp_size == 1

    assert no_cp.cp_mode == "none"
    assert not no_cp.is_ulysses_cp
    assert no_cp.effective_cp_size == 1


def test_slice_helpers_keep_full_sequence_for_ulysses_cp():
    ulysses = _make_parallel_state(cp_size=2, cp_comm_type="a2a")
    standard = _make_parallel_state(cp_size=2, cp_comm_type="p2p")
    tokens = torch.tensor([10, 11, 12, 13, 14], dtype=torch.long)
    log_probs = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)

    set_parallel_state(ulysses)
    torch.testing.assert_close(cp_utils.slice_with_cp(tokens, 0), tokens)
    torch.testing.assert_close(cp_utils.slice_log_prob_with_cp(log_probs, 5, 3), log_probs)
    torch.testing.assert_close(cp_utils.all_gather_with_cp(log_probs, 5, 3), log_probs)

    set_parallel_state(standard)
    sliced_tokens = cp_utils.slice_with_cp(tokens, 0)
    sliced_log_probs = cp_utils.slice_log_prob_with_cp(log_probs, 5, 3)

    assert not torch.equal(sliced_tokens, tokens)
    assert sliced_log_probs.numel() < log_probs.numel()


def test_sum_of_sample_mean_uses_full_response_masks_for_ulysses_cp():
    ulysses = _make_parallel_state(cp_size=2, cp_comm_type="a2a")
    set_parallel_state(ulysses)

    reducer = cp_utils.get_sum_of_sample_mean(
        total_lengths=[6],
        response_lengths=[4],
        loss_masks=[torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float32)],
    )

    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    expected = torch.tensor((1.0 + 3.0 + 4.0) / 3.0, dtype=torch.float32)
    torch.testing.assert_close(reducer(x), expected)


def test_get_responses_treats_ulysses_cp_as_full_sequence():
    ulysses = _make_parallel_state(cp_size=2, cp_comm_type="a2a")
    set_parallel_state(ulysses)

    args = Namespace(qkv_format="thd", rollout_temperature=1.0)
    logits = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
    tokens = [torch.tensor([20, 21, 22, 23, 24], dtype=torch.long)]

    responses = list(
        loss_utils.get_responses(
            logits,
            args=args,
            unconcat_tokens=tokens,
            total_lengths=[5],
            response_lengths=[2],
        )
    )

    assert len(responses) == 1
    logits_chunk, tokens_chunk = responses[0]
    torch.testing.assert_close(logits_chunk, torch.tensor([[2.0], [3.0]], dtype=torch.float32))
    torch.testing.assert_close(tokens_chunk, torch.tensor([23, 24], dtype=torch.long))


def test_loss_function_uses_effective_cp_size_for_ulysses_scaling(monkeypatch):
    args = Namespace(
        loss_type="policy_loss",
        calculate_per_token_loss=False,
        qkv_format="thd",
        recompute_loss_function=False,
        global_batch_size=4,
        use_dynamic_global_batch_size=False,
        allgather_cp=False,
    )
    batch = {
        "loss_masks": [torch.tensor([1.0, 1.0], dtype=torch.float32)],
        "response_lengths": [2],
        "total_lengths": [3],
    }
    logits = torch.zeros((1, 3, 8), dtype=torch.float32)

    monkeypatch.setattr(
        loss_utils,
        "policy_loss_function",
        lambda args, batch, logits, sum_of_sample_mean: (
            torch.tensor(2.0),
            {"loss": torch.tensor(2.0)},
        ),
    )

    ulysses = _make_parallel_state(cp_size=4, cp_comm_type="a2a", intra_dp_size=2, intra_dp_cp_size=8)
    set_parallel_state(ulysses)
    ulysses_loss, _, _ = loss_utils.loss_function(
        args,
        batch,
        num_microbatches=2,
        logits=logits,
        apply_megatron_loss_scaling=True,
    )

    standard = _make_parallel_state(cp_size=4, cp_comm_type="p2p", intra_dp_size=2, intra_dp_cp_size=8)
    set_parallel_state(standard)
    standard_loss, _, _ = loss_utils.loss_function(
        args,
        batch,
        num_microbatches=2,
        logits=logits,
        apply_megatron_loss_scaling=True,
    )

    torch.testing.assert_close(ulysses_loss, torch.tensor(2.0))
    torch.testing.assert_close(standard_loss, torch.tensor(8.0))
    assert ulysses_loss < standard_loss
