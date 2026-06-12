"""CPU unit tests for `--pg-loss-divisor` (Dr.GRPO constant pg-loss normalization).

Pins the contract of the `divisor` mode of `get_sum_of_sample_mean`
(arXiv:2503.20783, Eq. 2): pg_loss is divided by a shared constant instead of
each sample's active-token count, the default `divisor=None` behavior is
unchanged, the per-token-loss path never applies the divisor (Megatron
normalizes by token count itself), per-CP-rank contributions sum back to the
single-rank value, and misconfiguration fails loud at startup.
"""

from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils import cp_utils
from miles.backends.training_utils.parallel import GroupInfo, ParallelState

from .loss.loss_test_utils import make_args, make_batch, make_inputs, make_parallel_state


def _parallel_state(*, cp_size: int, cp_rank: int = 0) -> ParallelState:
    return ParallelState(
        intra_dp=GroupInfo(rank=0, size=1, group=None),
        intra_dp_cp=GroupInfo(rank=0, size=cp_size, group=None),
        cp=GroupInfo(rank=cp_rank, size=cp_size, group=None),
        tp=GroupInfo(rank=0, size=1, group=None),
        pp=GroupInfo(rank=0, size=1, group=None),
        ep=GroupInfo(rank=0, size=1, group=None),
        etp=GroupInfo(rank=0, size=1, group=None),
    )


@pytest.fixture(autouse=True)
def _stub_cp_size_one(monkeypatch):
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: _parallel_state(cp_size=1))


def test_divisor_none_keeps_per_sample_token_mean():
    response_lengths = [3, 2]
    total_lengths = [5, 3]
    loss_masks = [torch.tensor([1.0, 0.0, 1.0]), torch.ones(2)]
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    reducer = cp_utils.get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks)
    # (1 + 3) / 2 + (4 + 5) / 2 = 6.5
    assert reducer(x).item() == pytest.approx(6.5)


def test_constant_divisor_replaces_active_token_count():
    response_lengths = [3]
    total_lengths = [5]
    loss_masks = [torch.ones(3)]
    x = torch.tensor([1.0, 2.0, 3.0])

    reducer = cp_utils.get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, divisor=4.0)
    # sum(x * mask) / divisor = 6 / 4, not the active-token mean 6 / 3.
    assert reducer(x).item() == pytest.approx(1.5)


def test_all_samples_share_one_constant_divisor():
    response_lengths = [2, 3]
    total_lengths = [3, 4]
    loss_masks = [torch.ones(2), torch.tensor([1.0, 0.0, 1.0])]
    x = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

    reducer = cp_utils.get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, divisor=10.0)
    # Masked-out tokens drop from the numerator; the denominator stays 10.
    assert reducer(x).item() == pytest.approx(0.4)


def test_per_token_loss_ignores_divisor():
    response_lengths = [3]
    total_lengths = [4]
    loss_masks = [torch.ones(3)]
    x = torch.tensor([1.0, 2.0, 3.0])

    reducer = cp_utils.get_sum_of_sample_mean(
        total_lengths, response_lengths, loss_masks, True, "thd", None, divisor=4.0
    )
    assert reducer(x).item() == pytest.approx(6.0)


def test_cp_rank_contributions_sum_to_single_rank_value(monkeypatch):
    """The divisor is a shared constant, so Megatron's gradient sum-allreduce
    across CP ranks reproduces the cp_size == 1 value with no extra
    denominator communication."""
    total_lengths = [5]
    response_lengths = [4]
    loss_masks = [torch.tensor([1.0, 1.0, 1.0, 0.0])]
    x_full = torch.tensor([2.0, 3.0, 5.0, 7.0])
    divisor = 8.0

    ref = cp_utils.get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, divisor=divisor)(x_full)

    cp_size = 2
    cp_total = torch.zeros(())
    for cp_rank in range(cp_size):
        state = _parallel_state(cp_size=cp_size, cp_rank=cp_rank)
        monkeypatch.setattr(cp_utils, "get_parallel_state", lambda state=state: state)
        prompt_length = total_lengths[0] - response_lengths[0]
        _, _, _, tokens_offset = cp_utils.get_logits_and_tokens_offset_with_cp(
            total_lengths[0], response_lengths[0], "thd", None
        )
        x_local = torch.cat(
            [x_full[start - prompt_length : end - prompt_length] for start, end in tokens_offset], dim=0
        )
        reducer = cp_utils.get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, divisor=divisor)
        cp_total = cp_total + reducer(x_local)

    assert cp_total.item() == pytest.approx(ref.item())


def test_pg_loss_divisor_reaches_policy_loss():
    """End-to-end wiring: `args.pg_loss_divisor` swaps the pg_loss reducer in
    policy_loss_function while every other metric keeps the default reducer.

    With one sample of response length L and an all-ones mask, the default
    pg_loss is (token sum) / L and the divisor-mode pg_loss is (token sum) / D,
    so the two runs differ by exactly L / D.
    """
    from miles.backends.training_utils.loss_hub.losses import policy_loss_function

    make_parallel_state()
    response_len, divisor = 8, 4.0
    args_default = make_args()
    inputs = make_inputs(
        seed=42, batch_size=1, prompt_lens=[4], response_lens=[response_len], vocab_size=32, args=args_default
    )

    def run(args):
        batch = make_batch(inputs, "policy_loss")
        som = cp_utils.get_sum_of_sample_mean(
            batch["total_lengths"],
            batch["response_lengths"],
            batch["loss_masks"],
            args.calculate_per_token_loss,
            args.qkv_format,
            batch.get("max_seq_lens", None),
        )
        _, metrics = policy_loss_function(args, batch, inputs["policy_logits"].clone(), som)
        return metrics

    metrics_default = run(args_default)
    metrics_divisor = run(make_args(pg_loss_divisor=divisor))

    expected = metrics_default["pg_loss"].item() * response_len / divisor
    assert metrics_divisor["pg_loss"].item() == pytest.approx(expected)
    assert metrics_divisor["ppo_kl"].item() == pytest.approx(metrics_default["ppo_kl"].item())
    assert metrics_divisor["pg_clipfrac"].item() == pytest.approx(metrics_default["pg_clipfrac"].item())


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_non_positive_divisor_rejected_at_startup(bad):
    from miles.utils.arguments import miles_validate_args

    args = Namespace(pg_loss_divisor=bad, custom_pg_loss_reducer_function_path=None)
    with pytest.raises(ValueError, match="--pg-loss-divisor"):
        miles_validate_args(args)


def test_divisor_conflicts_with_custom_reducer_at_startup():
    from miles.utils.arguments import miles_validate_args

    args = Namespace(pg_loss_divisor=4.0, custom_pg_loss_reducer_function_path="my.module.get_reducer")
    with pytest.raises(ValueError, match="mutually exclusive"):
        miles_validate_args(args)
