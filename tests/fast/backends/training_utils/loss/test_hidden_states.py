"""Unit tests for `get_hidden_states`, the disaggregated-OPD-teacher forward-pass
callback that extracts response-aligned final hidden states instead of logits.

Also covers `get_responses`'s `scale_by_temperature` opt-out: hidden states have
`hidden_size > 1` in their last dimension, the same shape signature `get_responses`
otherwise uses to detect "these are vocab logits, apply --rollout-temperature".
"""

from argparse import Namespace

import torch

from miles.backends.training_utils.loss_hub.logit_processors import get_hidden_states, get_responses
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


def _make_trivial_parallel_state() -> None:
    def _trivial_group() -> GroupInfo:
        return GroupInfo(rank=0, size=1, group=None)

    set_parallel_state(
        ParallelState(
            intra_dp=_trivial_group(),
            intra_dp_cp=_trivial_group(),
            cp=_trivial_group(),
            tp=_trivial_group(),
            pp=_trivial_group(),
            ep=_trivial_group(),
            etp=_trivial_group(),
            indep_dp=_trivial_group(),
            is_pp_last_stage=True,
        )
    )


def _make_args(**overrides) -> Namespace:
    defaults = dict(
        qkv_format="thd",
        rollout_temperature=1.0,
        allgather_cp=False,
        true_on_policy_mode=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _hidden_states_by_position(total_len: int, hidden_size: int) -> torch.Tensor:
    """`[1, total_len, hidden_size]` where every value at position t equals t."""
    return torch.arange(total_len, dtype=torch.float32).view(1, total_len, 1).expand(1, total_len, hidden_size).clone()


def test_get_hidden_states_extracts_response_aligned_window():
    _make_trivial_parallel_state()
    hidden_size = 3
    total_lengths = [5, 4]
    response_lengths = [2, 1]
    unconcat_tokens = [torch.arange(length) for length in total_lengths]
    hidden_states = _hidden_states_by_position(sum(total_lengths), hidden_size)

    result = get_hidden_states(
        hidden_states,
        args=_make_args(),
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
    )

    assert list(result.keys()) == ["teacher_hidden_states"]
    sample_0, sample_1 = result["teacher_hidden_states"]
    assert torch.equal(sample_0, torch.tensor([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]))
    assert torch.equal(sample_1, torch.tensor([[7.0, 7.0, 7.0]]))


def test_get_hidden_states_does_not_scale_by_temperature():
    _make_trivial_parallel_state()
    hidden_size = 3
    total_lengths = [5]
    response_lengths = [2]
    unconcat_tokens = [torch.arange(5)]
    hidden_states = _hidden_states_by_position(5, hidden_size)

    result = get_hidden_states(
        hidden_states,
        args=_make_args(rollout_temperature=2.0),
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
    )

    assert torch.equal(result["teacher_hidden_states"][0], torch.tensor([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]))


def test_get_responses_temperature_scaling_is_opt_in():
    _make_trivial_parallel_state()
    hidden_size = 3
    total_lengths = [5]
    response_lengths = [2]
    unconcat_tokens = [torch.arange(5)]
    hidden_states = _hidden_states_by_position(5, hidden_size).squeeze(0).unsqueeze(0)

    (scaled_chunk, _), = get_responses(
        hidden_states,
        args=_make_args(rollout_temperature=2.0),
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
    )
    (unscaled_chunk, _), = get_responses(
        hidden_states,
        args=_make_args(rollout_temperature=2.0),
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        scale_by_temperature=False,
    )

    assert torch.equal(scaled_chunk, unscaled_chunk / 2.0)
