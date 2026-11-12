"""The Dr.GRPO example reducer reads miles' parallel state, so it runs on every training backend."""

import sys

import pytest
import torch

import miles.backends.training_utils.parallel as parallel
from miles.utils.ft_utils.process_group_utils import GroupInfo
from miles.utils.function_registry import load_function

REDUCER_PATH = "examples.experimental.DrGRPO.custom_reducer.get_pg_loss_reducer"


def _parallel_state(*, cp_size: int) -> parallel.ParallelState:
    """What the FSDP and torchtitan backends set: no Megatron mpu is initialized."""
    trivial = GroupInfo(rank=0, size=1, group=None)
    return parallel.ParallelState(
        intra_dp=trivial,
        intra_dp_cp=trivial,
        cp=GroupInfo(rank=0, size=cp_size, group=None),
        tp=trivial,
        pp=trivial,
        ep=trivial,
        etp=trivial,
        indep_dp=trivial,
    )


@pytest.fixture
def reducer_factory(monkeypatch):
    # A Megatron install without initialized model parallelism behaves like the FSDP/torchtitan backends.
    monkeypatch.delitem(sys.modules, "examples.experimental.DrGRPO.custom_reducer", raising=False)
    return load_function(REDUCER_PATH)


@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
def test_reducer_runs_without_megatron_parallel_state(monkeypatch, reducer_factory, calculate_per_token_loss):
    monkeypatch.setattr(parallel, "_parallel_state", _parallel_state(cp_size=1))
    from examples.experimental.DrGRPO import custom_reducer

    response_lengths = [3, 2]
    loss_masks = [torch.tensor([1.0, 1.0, 0.0]), torch.tensor([1.0, 1.0])]
    reduce = reducer_factory([10, 8], response_lengths, loss_masks, calculate_per_token_loss)

    x = torch.tensor([1.0, 2.0, 100.0, 3.0, 4.0])
    masked_sum = 1.0 + 2.0 + 3.0 + 4.0  # the masked-out 100 never counts
    expected = masked_sum if calculate_per_token_loss else masked_sum / custom_reducer.DIVISOR
    assert reduce(x).item() == pytest.approx(expected)


def test_reducer_refuses_context_parallelism(monkeypatch, reducer_factory):
    monkeypatch.setattr(parallel, "_parallel_state", _parallel_state(cp_size=2))
    with pytest.raises(AssertionError, match="cp_size == 1"):
        reducer_factory([4], [2], [torch.ones(2)], False)
